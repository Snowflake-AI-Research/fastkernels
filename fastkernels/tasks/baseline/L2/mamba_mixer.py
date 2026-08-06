"""Mamba v1 mixer (selective state-space model).

Implementation mirrors vLLM's ``MambaMixer``
(``vllm/model_executor/layers/mamba/mamba_mixer.py``) so that:

  - kernel calls (causal_conv1d_fn / causal_conv1d_update /
    selective_scan_fn / selective_state_update) are bit-identical to vLLM
  - parameter layout / weight names match HF Mamba checkpoints
    (state-spaces/mamba-* family)
  - tensor parallelism uses ColumnParallelLinear (in_proj, conv1d,
    dt_proj) and RowParallelLinear (x_proj, out_proj), matching vLLM

State (conv_state, ssm_state) and per-batch metadata are read from
fastkernels's global ``Context`` (``infra/context.py``), analogous to
vLLM's ``ForwardContext``.

Weight names from HF Mamba checkpoint
-------------------------------------
    mixer.in_proj.weight        [2*intermediate, hidden]   (gate + x)
    mixer.conv1d.weight         [intermediate, 1, conv_kernel]
    mixer.conv1d.bias           [intermediate]
    mixer.x_proj.weight         [time_step_rank + 2*state_size, intermediate]
    mixer.dt_proj.weight        [intermediate, time_step_rank]
    mixer.dt_proj.bias          [intermediate]
    mixer.A_log                 [intermediate, state_size]
    mixer.D                     [intermediate]
    mixer.out_proj.weight       [hidden, intermediate]
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)

from ....infra.context import get_context
from ....infra.tp import _tp_rank, _tp_size
from .parallel_linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

# Mamba v1 keeps its activations channel-major, ``[dim, num_tokens]``, because
# that is the layout the conv1d / selective-scan kernels want.  ``x_proj`` and
# ``out_proj`` consume those tensors as ``t.transpose(-2, -1)``, which hands
# cuBLAS a column-major operand whose *contiguous* axis is the token axis.  The
# length of that axis sets the kernel's vector width, so an unaligned token
# count drops cuBLAS off its nvjet path onto an ``align1`` CUTLASS fallback.
# Measured on a B200 at ~16K tokens, bf16:
#
#     token count   out_proj   x_proj
#     16136 (8|n)    0.28 ms   0.05 ms
#     16130 (2|n)    1.41 ms   0.16 ms
#     16131 (odd)    3.43 ms   0.29 ms
#
# 3.4 ms x 64 layers is ~200 ms per step, and the token count is whatever the
# scheduler happened to pack (prompt tails + one token per running decode), so
# it is odd about half the time.  For ``out_proj`` -- by far the more expensive
# of the two -- rounding the operand's token axis *up* and slicing the GEMM's
# output back down costs at most 7 extra token-rows.  ``x_proj``'s operand is
# the conv output, whose width is the prefill token count; the scheduler keeps
# that aligned where it can (``ModelRunner._mamba_prefill_align``).
# vLLM's Mamba2 path never hits this because it keeps activations token-major.
_LDA_ALIGN = 8  # 8 bf16 elements == 16 bytes

# Kill switch for the split in_proj in ``MambaMixer._project_input``.
_SPLIT_IN_PROJ = os.environ.get("FASTKERNELS_MAMBA_SPLIT_IN_PROJ", "1") == "1"


def _needs_token_pad(t: torch.Tensor) -> bool:
    """Whether ``t.transpose(-2, -1)`` would hit the unaligned GEMM path.

    ``stride(-2) == 1`` means the transpose is already row-major, so its
    contiguous axis is ``dim`` (always a multiple of 8) rather than the token
    count -- that is the decode half, and it is fine as-is.
    """
    if t.stride(-2) == 1:
        return False
    return t.size(-1) % _LDA_ALIGN != 0


def _padded_token_cat(parts: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Join ``[dim, num_tokens]`` halves into one aligned GEMM operand.

    Returns ``(buffer, num_real_tokens)``.  ``buffer`` may be wider than
    ``num_real_tokens``; callers slice the GEMM's output to that length.  A
    mixed batch pays a concatenation either way, so the padding is free there;
    a single unaligned half costs one copy, which is still ~20x cheaper than
    the GEMM fallback it avoids.
    """
    total = sum(p.size(-1) for p in parts)
    if len(parts) == 1 and not _needs_token_pad(parts[0]):
        return parts[0], total
    padded = (total + _LDA_ALIGN - 1) // _LDA_ALIGN * _LDA_ALIGN
    buf = torch.empty(
        (*parts[0].shape[:-1], padded),
        dtype=parts[0].dtype, device=parts[0].device,
    )
    offset = 0
    for p in parts:
        width = p.size(-1)
        buf[..., offset:offset + width].copy_(p)
        offset += width
    if offset < padded:
        buf[..., offset:].zero_()  # keep the discarded tail rows finite
    return buf, total


class MambaMixer(nn.Module):
    """Mamba v1 selective-scan mixer block."""

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        activation: str = "silu",
        layer_idx: int = 0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.tp_size = _tp_size()
        self.tp_rank = _tp_rank()

        assert intermediate_size % self.tp_size == 0, (
            "Mamba v1 requires intermediate_size divisible by tp_size."
        )

        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.activation = activation
        self.layer_idx = layer_idx

        # conv1d as a column-parallel linear over the intermediate dim
        # (output_size == intermediate_size sharded across TP).
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
            quant_config=None,
        )
        # Promote to depthwise-conv weight layout (D, 1, K) after load.
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # in_proj packs [x, gate], each of size intermediate_size.
        self.in_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=use_bias,
            quant_config=quant_config,
        )

        # x_proj: produces [dt, B, C] from x.  RowParallel because input
        # dim is intermediate (which is sharded), output is replicated.
        self.x_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=time_step_rank + 2 * ssm_state_size,
            bias=False,
            quant_config=None,
        )

        # dt_proj: time_step_rank -> intermediate (column-parallel).
        # Bias is added by the selective-scan kernel, so we keep it
        # separately and pass it through.
        self.dt_proj = ColumnParallelLinear(
            input_size=time_step_rank,
            output_size=intermediate_size,
            bias=True,
            quant_config=None,
        )

        tp_inter = intermediate_size // self.tp_size
        self.A = nn.Parameter(
            torch.empty(tp_inter, ssm_state_size, dtype=torch.float32),
        )
        self.D = nn.Parameter(torch.ones(tp_inter))

        # A_log is sharded along dim 0 with the -exp() transform applied
        # at load time so the kernel sees A directly.
        def _shard0_loader(param, loaded_weight):
            shard = param.data.size(0)
            param.data.copy_(
                loaded_weight.narrow(0, self.tp_rank * shard, shard).to(param.dtype),
            )

        def _A_loader(param, loaded_weight):
            shard = param.data.size(0)
            slice_ = loaded_weight.narrow(0, self.tp_rank * shard, shard).float()
            param.data.copy_(-torch.exp(slice_))

        self.A.weight_loader = _A_loader
        self.D.weight_loader = _shard0_loader

        self.out_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=use_bias, quant_config=quant_config,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _project_input(self, hidden_states: torch.Tensor, *, split: bool):
        """in_proj, returning ``(hidden_states_BC, gate)`` as ``[dim, N]``.

        The two halves feed kernels that disagree about layout: the varlen
        conv1d wants ``hidden_states_BC`` channel-contiguous, while
        ``selective_scan_fn`` requires ``z`` (the gate) with
        ``stride(-1) == 1`` and clones it otherwise.  One merged projection can
        only satisfy one of them -- vLLM keeps the merged GEMM and pays a
        transposing clone of the gate inside the scan wrapper, 0.37 ms per layer
        at 16K tokens (an uncoalesced copy running at ~0.7 TB/s), so ~24 ms per
        step across 64 layers.  Running the two halves of the same weight as two
        GEMMs costs ~0.02 ms and lands each half in the layout its kernel wants.

        Only worth doing when the step has prefill tokens: the decode kernels
        take ``z`` as-is, so decode-only steps (the CUDA-graph path) keep the
        merged projection.  It also needs an aligned token count -- the split
        gate GEMM stores a ``[dim, num_tokens]`` result, and an unaligned row
        length costs it more (0.98 ms) than the clone it saves (0.37 ms).
        Finally it falls back to the merged projection when the weight is not a
        plain matrix (fp8) or carries a bias, since the split would otherwise
        have to reimplement the quantized ``linear_op``.
        """
        in_proj = self.in_proj
        if (
            not split
            or not _SPLIT_IN_PROJ
            or hidden_states.dim() != 2
            or hidden_states.size(0) % _LDA_ALIGN != 0
            or getattr(in_proj, "use_fp8", False)
            or in_proj.bias is not None
        ):
            projected_states = in_proj(hidden_states).transpose(-2, -1)
            return projected_states.chunk(2, dim=-2)

        import torch.nn.functional as F
        weight = in_proj.weight
        half = weight.size(0) // 2
        hidden_states_BC = F.linear(
            hidden_states, weight[:half],
        ).transpose(-2, -1)
        gate = torch.mm(weight[half:], hidden_states.transpose(-2, -1))
        return hidden_states_BC, gate

    def _ssm_transform(self, x: torch.Tensor, *, dt_contiguous: bool = False):
        """Compute (dt, B, C) from x via x_proj + dt_proj.

        x: [N, intermediate_per_rank]
        Returns:
          dt: [intermediate_per_rank, N]
          B:  [N, ssm_state_size]
          C:  [N, ssm_state_size]

        ``dt_contiguous`` returns ``dt`` already contiguous in that shape.  See
        the call site in the prefill branch for why.
        """
        ssm_params = self.x_proj(x)  # [N, dt_rank + 2*N_state]
        dt, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        # dt_proj (skip bias add - the kernel handles it).
        # ColumnParallelLinear adds bias inside; we want it raw, so we
        # call F.linear without the bias and pass the bias separately.
        import torch.nn.functional as F
        if dt_contiguous:
            dt = torch.mm(self.dt_proj.weight, dt.transpose(-2, -1))
        else:
            dt = F.linear(dt, self.dt_proj.weight, None).transpose(-2, -1)
        return dt, B, C

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Mamba v1 mixer forward.

        Reads cache state and per-batch metadata from the global Context.
        Mirrors vLLM ``MambaMixer.forward_impl`` and keeps the mixed
        batch in decode-first token order.

        ``hidden_states`` shape: [num_tokens, hidden_size]
        """
        ctx = get_context()
        mamba_state = getattr(ctx, "mamba_state", None)
        mamba_meta = getattr(ctx, "mamba_metadata", None)

        hidden_states_BC, gate = self._project_input(
            hidden_states,
            split=mamba_meta is not None and mamba_meta.num_prefill_tokens > 0,
        )

        if mamba_state is None or mamba_meta is None:
            # Profile / warmup path (no cache available).
            return self.out_proj(hidden_states_BC.transpose(-2, -1))

        # MambaStateManager allocates as ``[N, kernel-1, dim]`` so we
        # transpose to the kernel's expected ``[N, dim, kernel-1]`` view
        # which keeps ``stride(dim) == 1``.
        conv_state = mamba_state.conv_states[self.layer_idx].transpose(-1, -2)
        ssm_state = mamba_state.ssm_states[self.layer_idx]

        num_prefill_tokens = mamba_meta.num_prefill_tokens
        num_decode_tokens = mamba_meta.num_decode_tokens
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual = num_prefill_tokens + num_decode_tokens

        if has_prefill and has_decode:
            hidden_states_BC_d, hidden_states_BC_p = torch.split(
                hidden_states_BC[:, :num_actual],
                [num_decode_tokens, num_prefill_tokens],
                dim=-1,
            )
            gate_d, gate_p = torch.split(
                gate[:, :num_actual],
                [num_decode_tokens, num_prefill_tokens],
                dim=-1,
            )
        elif has_prefill:
            hidden_states_BC_p = hidden_states_BC[:, :num_prefill_tokens]
            gate_p = gate[:, :num_prefill_tokens]
            hidden_states_BC_d = None
            gate_d = None
        else:
            hidden_states_BC_d = hidden_states_BC[:, :num_decode_tokens]
            gate_d = gate[:, :num_decode_tokens]
            hidden_states_BC_p = None
            gate_p = None

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2),
        )
        ssm_outputs = []
        time_proj_bias = self.dt_proj.bias.float() if self.dt_proj.bias is not None else None

        if has_decode:
            conv_out_d = causal_conv1d_update(
                hidden_states_BC_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_meta.state_indices_d,
                null_block_id=-1,
            ).transpose(0, 1)

            dt_d, B_d, C_d = self._ssm_transform(conv_out_d.transpose(-2, -1))
            out_d = torch.empty_like(hidden_states_BC_d.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1),
                dt_d.transpose(0, 1),
                self.A,
                B_d,
                C_d,
                self.D,
                dt_bias=time_proj_bias,
                z=gate_d.transpose(0, 1),
                dt_softplus=True,
                state_batch_indices=mamba_meta.state_indices_d,
                null_block_id=-1,
                out=out_d,
            )
            ssm_outputs.append(out_d.transpose(0, 1))

        if has_prefill:
            conv_out_p = causal_conv1d_fn(
                hidden_states_BC_p,
                conv_weights,
                self.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=mamba_meta.query_start_loc_p,
                cache_indices=mamba_meta.state_indices_p,
                has_initial_state=mamba_meta.has_initial_states_p,
                activation=self.activation,
                metadata=mamba_meta,
            )

            # ``selective_scan_fn`` requires ``delta`` with ``stride(-1) == 1``
            # and clones it otherwise.  The natural ``F.linear(...).transpose``
            # produces exactly the layout that fails that test, so the wrapper
            # materialises a [dim, num_tokens] transposing copy -- an
            # uncoalesced read that runs at ~0.7 TB/s, 0.37 ms per layer at 16K
            # tokens, i.e. ~24 ms per step across 64 layers.  Computing
            # ``W @ dt^T`` costs the same as the linear + view and lands the
            # values in the layout the kernel wants, so it receives exactly the
            # tensor it would have built for itself.
            dt_p, B_p, C_p = self._ssm_transform(
                conv_out_p.transpose(-2, -1), dt_contiguous=True,
            )
            scan_out_p = selective_scan_fn(
                conv_out_p,
                ssm_state,
                dt_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                gate_p,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_meta.state_indices_p,
                has_initial_state=mamba_meta.has_initial_states_p,
                query_start_loc=mamba_meta.query_start_loc_p,
            )
            ssm_outputs.append(scan_out_p)

        scan_outputs, num_scan_tokens = _padded_token_cat(ssm_outputs)
        return self.out_proj(
            scan_outputs.transpose(-2, -1),
        )[:num_scan_tokens]
