"""Mamba2 SSD mixer (mistralai/Mamba-Codestral-7B-v0.1, etc.).

Implementation mirrors vLLM's ``MambaMixer2``
(``vllm/model_executor/layers/mamba/mamba_mixer2.py``) so that:

  - the kernel calls (causal_conv1d_fn, causal_conv1d_update,
    mamba_chunk_scan_combined_varlen, selective_state_update,
    rms_norm_gated) are bit-for-bit identical to vLLM
  - the parameter layout / weight names match HF Mamba2 checkpoints
  - tensor parallelism reuses vLLM's
    ``mamba_v2_sharded_weight_loader`` and
    ``extra_groups_for_head_shards`` policies

State (conv_state, ssm_state) and per-batch metadata (which slot to
read/write, which sequences carry an initial state, prefill vs decode
split, chunk indices, ...) are read from fastkernels's global ``Context``
(``infra/context.py``) — analogous to vLLM's ``ForwardContext``.

Weight names from HF Mamba2 checkpoint
--------------------------------------
    mixer.in_proj.weight        [intermediate + 2*g*N + nheads, hidden]
    mixer.conv1d.weight         [conv_dim, 1, conv_kernel]
    mixer.conv1d.bias           [conv_dim]
    mixer.A_log                 [num_heads]
    mixer.D                     [num_heads]
    mixer.dt_bias               [num_heads]
    mixer.norm.weight           [intermediate_size]
    mixer.out_proj.weight       [hidden, intermediate]

where ``conv_dim = intermediate_size + 2 * n_groups * state_size`` and
``intermediate_size = num_heads * head_dim``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# vLLM-aligned kernels (bit-identical numerics with vLLM's MambaMixer2).
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)

from ....infra.context import get_context
from ....infra.tp import _tp_rank, _tp_size
from .parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


# ---------------------------------------------------------------------------
# Sharding helpers (verbatim port of vLLM utilities)
# ---------------------------------------------------------------------------
def extra_groups_for_head_shards(n_groups: int, tp_size: int) -> int:
    """Replicate enough extra groups so every head shard gets its own copy.

    Mirrors vLLM ``MambaStateShapeCalculator.extra_groups_for_head_shards``.
    """
    if n_groups % tp_size == 0:
        return 0
    return tp_size - n_groups


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, bool]],
    tp_size: int,
    tp_rank: int,
):
    """Sharded loader for Mamba2 in_proj / conv1d weight & bias.

    Verbatim port of vLLM's helper of the same name. ``shard_spec`` is a
    list of ``(full_dim, extra, duplicate_groups)`` entries describing the
    successive concatenated chunks of the loaded tensor.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        boundary, loaded_boundary = 0, 0
        for full_dim, extra, duplicate_groups in shard_spec:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip
            take = min(shard_size, full_dim - extra - loaded_skip)
            param.data[boundary : boundary + take, ...] = loaded_weight[
                loaded_start_idx : loaded_start_idx + take
            ]
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


def _replace_loader(param: nn.Parameter, loader) -> None:
    """Override the ``weight_loader`` attribute installed by parallel_linear."""
    if hasattr(param, "weight_loader"):
        # Setattr on tensor subclasses requires plain attribute assignment;
        # Parameter inherits __setattr__ from Tensor which permits this.
        try:
            delattr(param, "weight_loader")
        except AttributeError:
            pass
    param.weight_loader = loader  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Gated RMSNorm (per-group, optional TP-aware reduction)
# ---------------------------------------------------------------------------
class Mixer2RMSNormGated(nn.Module):
    """Per-group gated RMSNorm matching vLLM's ``Mixer2RMSNormGated``.

    Applies ``x * silu(gate)`` then per-group RMS normalization. When
    ``n_groups == 1`` and TP > 1, reduction crosses TP ranks via all-reduce.
    Otherwise each rank reduces locally (assumes ``n_groups % tp_size == 0``).
    """

    def __init__(self, full_hidden_size: int, full_n_groups: int,
                 use_rms_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.tp_size = _tp_size()
        self.tp_rank = _tp_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size
        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if use_rms_norm:
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            # Sharded along dim 0 (per-rank slice of the global weight).
            self.weight.weight_loader = self._weight_loader
        else:
            self.register_parameter("weight", None)

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, self.tp_rank * shard, shard))

    def forward_native(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x * F.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                import torch.distributed as dist
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                dist.all_reduce(local_sums)
                count = self.tp_size * x.shape[-1]
                variance = local_sums / count
            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            *prefix, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_g = x.view(*prefix, group_count, self.group_size)
            variance = x_g.pow(2).mean(-1, keepdim=True)
            x_g = x_g * torch.rsqrt(variance + self.variance_epsilon)
            x = x_g.view(*prefix, hidden_dim)
        return self.weight * x.to(input_dtype)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # vLLM uses the fused triton ``rms_norm_gated`` kernel only when
        # n_groups == 1 (single global reduction); otherwise the per-group
        # path is required.
        input_dtype = x.dtype
        if not self.use_rms_norm:
            return x * F.silu(gate.to(torch.float32)).to(input_dtype)
        if self.n_groups != 1:
            return self.forward_native(x, gate)
        return rms_norm_gated(
            x,
            self.weight.data,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            norm_before_gate=False,
        )


# ---------------------------------------------------------------------------
# Mamba2 mixer
# ---------------------------------------------------------------------------
class Mamba2Mixer(nn.Module):
    """Mamba2 SSD mixer.

    Constructor signature mirrors vLLM's ``MambaMixer2.__init__`` keyword
    arguments so external callers can swap implementations.
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        num_heads: int = 128,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        activation: str = "silu",
        use_rms_norm: bool = True,
        chunk_size: int = 256,
        layer_idx: int = 0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.tp_size = _tp_size()
        tp_rank = _tp_rank()

        assert num_heads % self.tp_size == 0, (
            "Tensor parallel world size must divide num heads."
        )
        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If TP world size does not divide n_groups, n_groups must equal 1."
        )

        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # n_groups: extend if TP doesn't divide so each head shard gets
        # its own (replicated) groups (n_groups == 1 case).
        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            self.n_groups = n_groups + extra_groups_for_head_shards(
                n_groups, self.tp_size,
            )
        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = intermediate_size + 2 * self.groups_ssm_state_size

        # Use ColumnParallelLinear (single fused tensor) for both branches
        # so the loaded HF weight can be installed in one shot via the
        # mamba_v2_sharded_weight_loader (matches vLLM's
        # weight_loader contract: ``loader(param, loaded_weight)``).
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
            quant_config=None,
        )
        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size + self.conv_dim + self.num_heads,
            bias=use_bias,
            quant_config=quant_config,
        )

        # When n_groups is divisible by tp, each B/C group simply shards
        # across tp_size; otherwise (n_groups == 1, tp > 1) each rank
        # gets its own replicated copy of B and C.
        n_groups_orig = n_groups
        group_shard = (
            self.groups_ssm_state_size,
            (self.n_groups - n_groups_orig) * self.ssm_state_size,
            n_groups_orig == 1,
        )
        inter_shard = (intermediate_size, 0, False)
        head_shard = (self.num_heads, 0, False)

        if use_conv_bias:
            _replace_loader(
                self.conv1d.bias,
                mamba_v2_sharded_weight_loader(
                    [inter_shard, group_shard, group_shard],
                    self.tp_size, tp_rank,
                ),
            )
        _replace_loader(
            self.conv1d.weight,
            mamba_v2_sharded_weight_loader(
                [inter_shard, group_shard, group_shard],
                self.tp_size, tp_rank,
            ),
        )
        _replace_loader(
            self.in_proj.weight,
            mamba_v2_sharded_weight_loader(
                [inter_shard, inter_shard, group_shard, group_shard, head_shard],
                self.tp_size, tp_rank,
            ),
        )

        # conv1d.weight stored in ``MergedColumnParallelLinear`` as 2D —
        # unsqueeze to the canonical (conv_dim, 1, conv_kernel) layout used
        # by ``causal_conv1d_fn`` after loading.
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2),
        )
        self.register_buffer("conv_weights", conv_weights, persistent=False)

        # A is stored as A_log in HF; we transform on load to the
        # negative-exp form expected by the kernels.
        self.A = nn.Parameter(
            torch.empty(num_heads // self.tp_size, dtype=torch.float32),
        )
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        # Sharded along dim 0 with optional A_log -> -exp() transform.
        def _shard0_loader(param, loaded_weight):
            shard = param.data.size(0)
            param.data.copy_(
                loaded_weight.narrow(0, tp_rank * shard, shard).to(param.dtype),
            )

        def _A_loader(param, loaded_weight):
            shard = param.data.size(0)
            slice_ = loaded_weight.narrow(0, tp_rank * shard, shard).float()
            param.data.copy_(-torch.exp(slice_))

        self.A.weight_loader = _A_loader
        self.D.weight_loader = _shard0_loader
        self.dt_bias.weight_loader = _shard0_loader

        self.out_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=use_bias, quant_config=quant_config,
        )

        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, use_rms_norm=use_rms_norm,
            eps=rms_norm_eps,
        )
        self._use_custom_op = False
        self._layer_name = ""

        # Pre-computed per-rank sizes used for splitting in forward.
        self.tped_intermediate_size = intermediate_size // self.tp_size
        self.tped_conv_size = self.conv_dim // self.tp_size
        self.tped_dt_size = num_heads // self.tp_size
        self.tped_groups_state = self.groups_ssm_state_size // self.tp_size
        self.register_buffer(
            "_ssm_out_buf",
            torch.empty(0, self.tped_intermediate_size),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _split_BC(self, hsBC: torch.Tensor):
        return torch.split(
            hsBC,
            [self.tped_intermediate_size,
             self.tped_groups_state,
             self.tped_groups_state],
            dim=-1,
        )

    def set_shared_ssm_out_buffer(self, buffer: torch.Tensor) -> None:
        self._ssm_out_buf = buffer

    def _get_ssm_out_buffer(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = self._ssm_out_buf
        if (
            buf.device != device
            or buf.dtype != dtype
            or buf.shape[-1] != self.tped_intermediate_size
            or buf.shape[0] < num_tokens
        ):
            buf = torch.empty(
                max(1, num_tokens),
                self.tped_intermediate_size,
                dtype=dtype,
                device=device,
            )
            self._ssm_out_buf = buf
        return buf[:num_tokens]

    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Run the conv + SSM core and write the result into ``output``."""
        hsBC, dt = torch.split(
            projected_states[..., self.tped_intermediate_size:],
            [self.tped_conv_size, self.tped_dt_size],
            dim=-1,
        )

        ctx = get_context()
        mamba_state = getattr(ctx, "mamba_state", None)
        mamba_meta = getattr(ctx, "mamba_metadata", None)

        if mamba_state is None or mamba_meta is None:
            # Profile / warmup path: no cache available, skip SSM state IO.
            hsBC = hsBC.contiguous()
            x, _B, _C = self._split_BC(hsBC)
            output.copy_(x)
            return

        # MambaStateManager allocates as ``[N, kernel-1, conv_dim]``;
        # transpose to ``[N, conv_dim, kernel-1]`` so the conv kernels'
        # ``stride_istate_dim == 1`` requirement is satisfied.
        conv_state = mamba_state.conv_states[self.layer_idx].transpose(-1, -2)
        ssm_state = mamba_state.ssm_states[self.layer_idx]

        num_prefill_tokens = mamba_meta.num_prefill_tokens
        num_decode_tokens = mamba_meta.num_decode_tokens
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual = num_prefill_tokens + num_decode_tokens

        # Prefill tokens come first, decode tokens last (fastkernels convention).
        hsBC_p, hsBC_d = torch.split(
            hsBC[:num_actual], [num_prefill_tokens, num_decode_tokens], dim=0,
        )
        dt_p, dt_d = torch.split(
            dt[:num_actual], [num_prefill_tokens, num_decode_tokens], dim=0,
        )

        ssm_out = output[:num_actual]
        ssm_out_p, ssm_out_d = torch.split(
            ssm_out, [num_prefill_tokens, num_decode_tokens], dim=0,
        )

        if has_prefill:
            x_in = hsBC_p.transpose(0, 1)  # [conv_dim_per_rank, T_p]
            hsBC_p = causal_conv1d_fn(
                x_in,
                self.conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=mamba_meta.has_initial_states_p,
                cache_indices=mamba_meta.state_indices_p,
                query_start_loc=mamba_meta.query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]

            x_p, B_p, C_p = self._split_BC(hsBC_p)

            initial_states = None
            if mamba_meta.has_initial_states_p is not None and mamba_meta.prep_initial_states:
                initial_states = torch.where(
                    mamba_meta.has_initial_states_p[:, None, None, None],
                    ssm_state[mamba_meta.state_indices_p],
                    0,
                )

            n_groups_per_rank = self.n_groups // self.tp_size
            varlen_states = mamba_chunk_scan_combined_varlen(
                x_p.view(num_prefill_tokens,
                         self.num_heads // self.tp_size, self.head_dim),
                dt_p,
                self.A,
                B_p.view(num_prefill_tokens, n_groups_per_rank, -1),
                C_p.view(num_prefill_tokens, n_groups_per_rank, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=mamba_meta.seq_idx_p,
                cu_seqlens=mamba_meta.query_start_loc_p,
                cu_chunk_seqlens=mamba_meta.cu_chunk_seqlen_p,
                last_chunk_indices=mamba_meta.last_chunk_indices_p,
                initial_states=initial_states,
                return_intermediate_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=ssm_out_p.view(num_prefill_tokens, -1, self.head_dim),
                state_dtype=ssm_state.dtype,
            )
            ssm_state[mamba_meta.state_indices_p] = varlen_states

        if has_decode:
            hsBC_d = causal_conv1d_update(
                hsBC_d,
                conv_state,
                self.conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_meta.state_indices_d,
            )
            x_d, B_d, C_d = self._split_BC(hsBC_d)

            n_groups_per_rank = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(torch.float32)
            )
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None].expand(-1, self.head_dim)
            D_d = self.D[:, None].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups_per_rank, B_d.shape[1] // n_groups_per_rank)
            C_d = C_d.view(-1, n_groups_per_rank, C_d.shape[1] // n_groups_per_rank)
            x_d = x_d.view(-1, self.num_heads // self.tp_size, self.head_dim)

            selective_state_update(
                ssm_state,
                x_d,
                dt_d,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=mamba_meta.state_indices_d,
                out=ssm_out_d.view(num_decode_tokens, -1, self.head_dim),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected_states = self.in_proj(hidden_states)
        ssm_output = self._get_ssm_out_buffer(
            projected_states.shape[0],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if self._use_custom_op:
            torch.ops.fastkernels.mamba2_conv_ssm_forward(
                projected_states,
                ssm_output,
                self._layer_name,
            )
        else:
            self.conv_ssm_forward(projected_states=projected_states, output=ssm_output)
        gate = projected_states[..., : self.tped_intermediate_size]
        hidden_states = self.norm(ssm_output, gate)
        return self.out_proj(hidden_states)
