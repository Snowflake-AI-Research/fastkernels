"""Qwen3-Next Gated Delta Net (GDN) linear attention (L2).

Implements the GDN block used in Qwen3-Next linear attention layers:
  x -> in_proj_qkvz -> [q,k,v,z]
  x -> in_proj_ba   -> [b,a]
  mixed_qkv = cat(q,k,v) -> causal_conv1d (SiLU) -> split q,k,v
  g = -exp(A_log) * softplus(a + dt_bias)   (forget gate, per v-head)
  beta = sigmoid(b)                          (learning rate, per v-head)
  q,k expanded to num_v_heads if num_v_heads > num_k_heads
  o = chunk_gated_delta_rule(q, k, v, g, beta)  [prefill]
   or fused_recurrent_gated_delta_rule(...)     [decode]
  o = RMSNormGated(o, z)                     (norm_before_gate=True, swish)
  o = out_proj(o)

Uses vLLM/FlashInfer's GDN and causal-conv kernels directly, plus the
local ``RMSNormGated`` wrapper and canonical TP linears in
``parallel_linear``.

Key dimensions (80B-A3B defaults):
  num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128
  key_dim=2048, value_dim=4096, conv_dim=8192
  conv_kernel_size=4

Weight names match HuggingFace checkpoint:
  linear_attn.in_proj_qkvz.weight   [2*key_dim + 2*value_dim, hidden_size]
  linear_attn.in_proj_ba.weight     [2*num_v_heads, hidden_size]
  linear_attn.conv1d.weight         [conv_dim, 1, kernel_size]
  linear_attn.A_log                 [num_v_heads]
  linear_attn.dt_bias               [num_v_heads]
  linear_attn.norm.weight           [head_v_dim]
  linear_attn.out_proj.weight       [hidden_size, value_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as _vllm_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _vllm_fused_recurrent_gdn,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as _vllm_causal_conv1d_fn,
    causal_conv1d_update as _vllm_causal_conv1d_update,
)
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator

from ....infra.context import get_context
from ....infra.tp import _tp_size, _tp_rank
from ..L1.rms_norm_gated import RMSNormGated
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


def _flashinfer_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
):
    """FlashInfer GDN prefill path matching vLLM's Qwen3-Next implementation."""
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
    )

    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    output, final_state = _fi_chunk_gated_delta_rule(
        q=q.squeeze(0).contiguous(),
        k=k.squeeze(0).contiguous(),
        v=v.squeeze(0).contiguous(),
        g=torch.exp(g.squeeze(0).contiguous().to(torch.float32)),
        beta=beta.squeeze(0).contiguous().to(torch.float32),
        initial_state=initial_state.to(torch.float32),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return output.unsqueeze(0), final_state


@triton.jit
def _fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold,
        (1 / beta) * tl.log(1 + tl.exp(beta * x)),
        x,
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off,
        blk_beta_output.to(beta_output.dtype.element_ty),
        mask=mask,
    )


def _fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused vLLM-equivalent GDN gating: g and beta in one Triton launch."""
    batch, num_heads = a.shape
    seq_len = 1
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    _fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        1.0,
        20.0,
        8,
        num_warps=1,
    )
    return g, beta_output


class _Conv1dWeight(nn.Module):
    """Parameter container for the GDN causal-conv1d kernel.

    The checkpoint stores conv1d weight as ``[channels, 1, kernel]``
    (``nn.Conv1d`` layout) where channels are organized as
    ``[Q(key_dim), K(key_dim), V(value_dim)]``. For TP, each segment
    must be sharded independently because the runtime input layout per
    rank is ``[Q_local, K_local, V_local]``. ``ColumnParallelLinear``
    can't express that piecewise sharding, so we hold the parameter
    here with a custom loader.
    """

    def __init__(self, channels: int, kernel_size: int,
                 segment_sizes: list[int] | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self._segment_sizes = segment_sizes
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        tp, rank = _tp_size(), _tp_rank()

        if self._segment_sizes is None or tp == 1:
            shard = param.data.size(0)
            param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))
            return

        offset_src = 0
        offset_dst = 0
        for seg_size in self._segment_sizes:
            local_seg = seg_size // tp
            src = loaded_weight.narrow(0, offset_src + rank * local_seg, local_seg)
            param.data[offset_dst:offset_dst + local_seg].copy_(src)
            offset_src += seg_size
            offset_dst += local_seg


class Qwen3NextGDNAttention(nn.Module):
    """Gated Delta Net linear attention for Qwen3-Next."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        layer_idx: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.local_k_heads = num_k_heads // tp
        self.local_v_heads = num_v_heads // tp
        self.v_per_k = num_v_heads // num_k_heads
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size

        # in_proj_qkvz: projects to [Q, K, V, Z] organized by K head groups
        # Layout per K head group: [Q_k_dim, K_k_dim, V_v_per_k*v_dim, Z_v_per_k*v_dim]
        qkvz_dim = 2 * self.key_dim + 2 * self.value_dim
        self.in_proj_qkvz = ColumnParallelLinear(hidden_size, qkvz_dim)

        # in_proj_ba: projects to [b, a] each of size num_v_heads.
        # Match vLLM's MergedColumnParallelLinear(output_sizes=[num_v_heads]*2)
        # which splits at midpoint then TP-shards each half independently.
        self.in_proj_ba = ColumnParallelLinear(hidden_size, 2 * num_v_heads)
        self.in_proj_ba.weight.weight_loader = self._ba_weight_loader

        # Causal conv1d on concatenated [Q, K, V]
        conv_dim = 2 * self.key_dim + self.value_dim
        local_conv_dim = conv_dim // tp
        self.conv1d = _Conv1dWeight(
            local_conv_dim, conv_kernel_size,
            segment_sizes=[self.key_dim, self.key_dim, self.value_dim],
        )

        # Decay parameters (sharded across TP)
        self.A_log = nn.Parameter(torch.empty(self.local_v_heads))
        self.A_log.weight_loader = self._sharded_weight_loader
        self.dt_bias = nn.Parameter(torch.empty(self.local_v_heads))
        self.dt_bias.weight_loader = self._sharded_weight_loader

        # Output norm: RMSNorm(x) * silu(z) with norm_before_gate=True
        # vLLM's rmsnorm_fn (wrapped here as L1.RMSNormGated) gives bitwise
        # alignment with vLLM's runtime.
        self.norm = RMSNormGated(
            head_v_dim, eps=rms_norm_eps,
            norm_before_gate=True, activation="swish",
        )

        # Output projection
        self.out_proj = RowParallelLinear(self.value_dim, hidden_size)
        self._triton_allocator_ready = False
        self._use_flashinfer_prefill = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
        )

    @staticmethod
    def _sharded_weight_loader(param, loaded_weight):
        rank = _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def _ba_weight_loader(self, param, loaded_weight):
        """Load in_proj_ba weight matching vLLM's MergedColumnParallelLinear.

        vLLM uses ``output_sizes=[num_v_heads, num_v_heads]`` which splits the
        ``[2*num_v_heads, hidden_size]`` weight at the midpoint, then TP-shards
        each half independently. This creates non-contiguous V-head assignments
        per rank but matches vLLM's exact behavior for token-level alignment.
        """
        tp, rank = _tp_size(), _tp_rank()
        if tp == 1:
            param.data.copy_(loaded_weight)
            return
        half = loaded_weight.size(0) // 2  # num_v_heads
        shard_size = half // tp
        part0 = loaded_weight.narrow(0, rank * shard_size, shard_size)
        part1 = loaded_weight.narrow(0, half + rank * shard_size, shard_size)
        param.data.copy_(torch.cat([part0, part1], dim=0))

    def _unpack_qkvz(self, proj_out):
        """Unpack in_proj_qkvz output into q, k, v, z."""
        N = proj_out.shape[0]
        per_group = (
            self.head_k_dim + self.head_k_dim
            + self.v_per_k * self.head_v_dim
            + self.v_per_k * self.head_v_dim
        )
        x = proj_out.view(N, self.local_k_heads, per_group)

        q = x[:, :, :self.head_k_dim]
        off = self.head_k_dim
        k = x[:, :, off:off + self.head_k_dim]
        off += self.head_k_dim
        v_size = self.v_per_k * self.head_v_dim
        v = x[:, :, off:off + v_size]
        off += v_size
        z = x[:, :, off:off + v_size]

        v = v.reshape(N, self.local_v_heads, self.head_v_dim)
        z = z.reshape(N, self.local_v_heads, self.head_v_dim)
        return q, k, v, z

    def _unpack_ba(self, proj_out):
        """Unpack in_proj_ba output into b, a (organized by K head groups)."""
        N = proj_out.shape[0]
        x = proj_out.view(N, self.local_k_heads, 2 * self.v_per_k)
        b = x[:, :, :self.v_per_k].reshape(N, self.local_v_heads)
        a = x[:, :, self.v_per_k:].reshape(N, self.local_v_heads)
        return b, a

    def _ensure_triton_allocator(self, device: torch.device) -> None:
        if not self._triton_allocator_ready:
            set_triton_allocator(device)
            self._triton_allocator_ready = True

    def forward(self, hidden_states: torch.Tensor, state_manager=None) -> torch.Tensor:
        md = get_context().kda_metadata
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextGDNAttention requires engine-managed recurrent state "
                "and metadata",
            )
        self._ensure_triton_allocator(hidden_states.device)

        x_flat = hidden_states.reshape(-1, self.hidden_size)
        N = x_flat.shape[0]

        # 1. Input projections
        qkvz_out = self.in_proj_qkvz(x_flat)
        ba_out = self.in_proj_ba(x_flat)

        q, k, v, z = self._unpack_qkvz(qkvz_out)
        b, a = self._unpack_ba(ba_out)

        # 2. Causal conv1d on concatenated [q, k, v]
        q_flat = q.reshape(N, self.local_k_heads * self.head_k_dim)
        k_flat = k.reshape(N, self.local_k_heads * self.head_k_dim)
        v_flat = v.reshape(N, self.local_v_heads * self.head_v_dim)
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)

        conv_state = state_manager.gdn_conv[self.layer_idx]
        conv_weights = self.conv1d.weight
        if md.num_prefills > 0:
            mixed_qkv = _vllm_causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                None,
                conv_state,
                md.non_spec_query_start_loc.to(torch.int32),
                cache_indices=md.non_spec_state_indices_tensor.to(torch.int32),
                has_initial_state=md.has_initial_state,
                activation="silu",
                metadata=md,
                validate_data=True,
            ).transpose(0, 1)
        else:
            mixed_qkv = _vllm_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                None,
                activation="silu",
                conv_state_indices=md.non_spec_state_indices_tensor[
                    : md.num_decodes
                ],
                validate_data=True,
            )

        local_k_dim = self.local_k_heads * self.head_k_dim
        local_v_dim = self.local_v_heads * self.head_v_dim
        q_conv, k_conv, v_conv = mixed_qkv.split(
            [local_k_dim, local_k_dim, local_v_dim], dim=-1
        )

        q_4d = q_conv.view(1, N, self.local_k_heads, self.head_k_dim)
        k_4d = k_conv.view(1, N, self.local_k_heads, self.head_k_dim)
        v_4d = v_conv.view(1, N, self.local_v_heads, self.head_v_dim)

        # 3. Gating: g = -exp(A_log) * softplus(a + dt_bias), beta=sigmoid(b).
        # Keep this fused to match vLLM's Qwen3-Next hot path.
        g, beta = _fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        recurrent_full = state_manager.recurrent[self.layer_idx]
        if md.num_prefills > 0:
            state_idx = md.non_spec_state_indices_tensor.long()
            init_state = recurrent_full.index_select(0, state_idx).contiguous()
            if md.has_initial_state is not None:
                zero_mask = (~md.has_initial_state).nonzero(as_tuple=True)[0]
                if zero_mask.numel() > 0:
                    init_state[zero_mask] = 0
            if self._use_flashinfer_prefill:
                o, final_state = _flashinfer_gdn_prefill(
                    q=q_4d.contiguous(),
                    k=k_4d.contiguous(),
                    v=v_4d.contiguous(),
                    g=g.contiguous(),
                    beta=beta.contiguous(),
                    initial_state=init_state,
                    output_final_state=True,
                    cu_seqlens=md.non_spec_query_start_loc.to(torch.long),
                )
            else:
                o, final_state = _vllm_chunk_gated_delta_rule(
                    q=q_4d.contiguous(),
                    k=k_4d.contiguous(),
                    v=v_4d.contiguous(),
                    g=g.contiguous(),
                    beta=beta.contiguous(),
                    initial_state=init_state,
                    output_final_state=True,
                    cu_seqlens=md.non_spec_query_start_loc.to(torch.long),
                    use_qk_l2norm_in_kernel=True,
                )
            recurrent_full.index_copy_(
                0,
                state_idx,
                final_state.to(recurrent_full.dtype),
            )
        else:
            o, _ = _vllm_fused_recurrent_gdn(
                q=q_4d.contiguous(),
                k=k_4d.contiguous(),
                v=v_4d.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                initial_state=recurrent_full,
                inplace_final_state=True,
                cu_seqlens=md.non_spec_query_start_loc[
                    : md.num_decodes + 1
                ].to(torch.long),
                ssm_state_indices=md.non_spec_state_indices_tensor[
                    : md.num_decodes
                ],
                use_qk_l2norm_in_kernel=True,
            )

        # 5. Output gating: RMSNorm(o) * silu(z) (L1 op)
        o_flat = o.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        o = self.norm(o_flat, z_flat).view(N, self.local_v_heads, self.head_v_dim)

        # 6. Output projection
        o = o.reshape(N, self.local_v_heads * self.head_v_dim)
        return self.out_proj(o)
