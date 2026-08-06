"""Qwen3-Next Gated Delta Net (GDN) linear attention (L2).

Implements the GDN block used in Qwen3-Next linear attention layers:
  x -> in_proj_qkvz -> [q,k,v,z]
  x -> in_proj_ba   -> [b,a]
  mixed_qkv = [q|k|v] -> causal_conv1d (SiLU) -> split q,k,v
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

The projection deinterleave (``_unpack_qkvz_ba``) and the post-conv split
(``_split_conv_qkv``) are single Triton launches rather than the chains of
strided ``reshape``/``contiguous``/``cat`` the shapes invite. vLLM gets that
collapse from Inductor because its model is ``torch.compile``d; this module runs
eager, where each of those tensor ops is a separate kernel and, at batch 1, the
launches are the whole cost -- ten per GDN layer across 36 layers was 0.9 ms of
a 5.5 ms decode step. Prefill uses vLLM's own ``fused_post_conv_prep`` for the
same reason.

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
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as _vllm_chunk_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update as _vllm_fused_sigmoid_gating_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep as _vllm_fused_post_conv_prep,
)
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
    """FlashInfer GDN prefill path matching vLLM's Qwen3-Next implementation.

    ``q``/``k`` arrive already L2-normalised and contiguous from
    ``fused_post_conv_prep``, so this is only the fp32 promotion and the
    ``exp(g)`` that vLLM's ``fi_chunk_gated_delta_rule`` does with
    ``use_qk_l2norm_in_kernel=False``.
    """
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
    )

    output, final_state = _fi_chunk_gated_delta_rule(
        q=q.squeeze(0),
        k=k.squeeze(0),
        v=v.squeeze(0),
        g=torch.exp(g.squeeze(0).to(torch.float32)),
        beta=beta.squeeze(0).to(torch.float32),
        initial_state=initial_state.to(torch.float32),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return output.unsqueeze(0), final_state


@triton.jit
def _unpack_qkvz_ba_kernel(
    qkvz_ptr,
    ba_ptr,
    mixed_qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    n_tokens,
    stride_qkvz,
    stride_ba,
    stride_mixed,
    stride_z,
    stride_b,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    VP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BVZ: tl.constexpr,
    BVP: tl.constexpr,
):
    """Deinterleave ``in_proj_qkvz``/``in_proj_ba`` into the GDN kernels' layout.

    ``in_proj_qkvz`` emits one group per K head -- ``[q(K) k(K) v(VP*V)
    z(VP*V)]`` -- while the conv wants ``[q_all | k_all | v_all]`` packed and
    the output gate wants ``z`` as ``[T, H*VP, V]``. Done with tensor ops that
    is six strided copies plus a cat; one program per (token block, K head)
    does the whole permutation in a single pass over the projection output.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < n_tokens

    if pid_h < H:
        h = pid_h
        group = h * (2 * K + 2 * VP * V)

        dk = tl.arange(0, BK)
        k_mask = t_mask[:, None] & (dk < K)[None, :]

        q_src = qkvz_ptr + t[:, None] * stride_qkvz + (group + dk)[None, :]
        q_dst = mixed_qkv_ptr + t[:, None] * stride_mixed + (h * K + dk)[None, :]
        tl.store(q_dst, tl.load(q_src, mask=k_mask), mask=k_mask)

        k_src = qkvz_ptr + t[:, None] * stride_qkvz + (group + K + dk)[None, :]
        k_dst = (
            mixed_qkv_ptr + t[:, None] * stride_mixed
            + (H * K + h * K + dk)[None, :]
        )
        tl.store(k_dst, tl.load(k_src, mask=k_mask), mask=k_mask)

        dv = tl.arange(0, BVZ)
        v_mask = t_mask[:, None] & (dv < VP * V)[None, :]

        v_src = qkvz_ptr + t[:, None] * stride_qkvz + (group + 2 * K + dv)[None, :]
        v_dst = (
            mixed_qkv_ptr + t[:, None] * stride_mixed
            + (2 * H * K + h * VP * V + dv)[None, :]
        )
        tl.store(v_dst, tl.load(v_src, mask=v_mask), mask=v_mask)

        z_src = (
            qkvz_ptr + t[:, None] * stride_qkvz
            + (group + 2 * K + VP * V + dv)[None, :]
        )
        z_dst = z_ptr + t[:, None] * stride_z + (h * VP * V + dv)[None, :]
        tl.store(z_dst, tl.load(z_src, mask=v_mask), mask=v_mask)
    else:
        # One extra program column splits ``ba`` -- also grouped per K head, as
        # ``[b(VP) a(VP)]`` -- into the flat ``[T, H*VP]`` the gating wants.
        dp = tl.arange(0, BVP)
        p_mask = dp < VP
        for h in tl.range(0, H):
            src = h * 2 * VP
            dst = h * VP
            m = t_mask[:, None] & p_mask[None, :]
            b_src = ba_ptr + t[:, None] * stride_ba + (src + dp)[None, :]
            b_dst = b_ptr + t[:, None] * stride_b + (dst + dp)[None, :]
            tl.store(b_dst, tl.load(b_src, mask=m), mask=m)
            a_src = ba_ptr + t[:, None] * stride_ba + (src + VP + dp)[None, :]
            a_dst = a_ptr + t[:, None] * stride_b + (dst + dp)[None, :]
            tl.store(a_dst, tl.load(a_src, mask=m), mask=m)


def _unpack_qkvz_ba(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    v_per_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mixed_qkv, z, b, a)``, all contiguous, in one Triton launch."""
    n = qkvz.shape[0]
    hv = num_k_heads * v_per_k
    conv_dim = 2 * num_k_heads * head_k_dim + hv * head_v_dim
    mixed_qkv = torch.empty(n, conv_dim, dtype=qkvz.dtype, device=qkvz.device)
    z = torch.empty(n, hv, head_v_dim, dtype=qkvz.dtype, device=qkvz.device)
    b = torch.empty(n, hv, dtype=ba.dtype, device=ba.device)
    a = torch.empty(n, hv, dtype=ba.dtype, device=ba.device)
    if n == 0:
        return mixed_qkv, z, b, a

    block_t = 16 if n >= 16 else 1
    _unpack_qkvz_ba_kernel[(triton.cdiv(n, block_t), num_k_heads + 1)](
        qkvz,
        ba,
        mixed_qkv,
        z,
        b,
        a,
        n,
        qkvz.stride(0),
        ba.stride(0),
        mixed_qkv.stride(0),
        z.stride(0),
        b.stride(0),
        H=num_k_heads,
        K=head_k_dim,
        V=head_v_dim,
        VP=v_per_k,
        BLOCK_T=block_t,
        BK=triton.next_power_of_2(head_k_dim),
        BVZ=triton.next_power_of_2(v_per_k * head_v_dim),
        BVP=triton.next_power_of_2(v_per_k),
    )
    return mixed_qkv, z, b, a


@triton.jit
def _split_conv_qkv_kernel(
    mixed_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    n_tokens,
    stride_mixed,
    stride_q,
    stride_v,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Split post-conv ``[T, 2*H*K + HV*V]`` into contiguous q, k, v.

    ``program_id(1)`` in ``[0, H)`` copies one Q head and its K head; beyond
    that it copies one V head, so all three tensors are written in one launch.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < n_tokens

    if pid_h < H:
        h = pid_h
        d = tl.arange(0, BK)
        m = t_mask[:, None] & (d < K)[None, :]
        base = mixed_ptr + t[:, None] * stride_mixed
        tl.store(
            q_ptr + t[:, None] * stride_q + (h * K + d)[None, :],
            tl.load(base + (h * K + d)[None, :], mask=m),
            mask=m,
        )
        tl.store(
            k_ptr + t[:, None] * stride_q + (h * K + d)[None, :],
            tl.load(base + (H * K + h * K + d)[None, :], mask=m),
            mask=m,
        )
    else:
        hv = pid_h - H
        d = tl.arange(0, BV)
        m = t_mask[:, None] & (d < V)[None, :]
        tl.store(
            v_ptr + t[:, None] * stride_v + (hv * V + d)[None, :],
            tl.load(
                mixed_ptr + t[:, None] * stride_mixed
                + (2 * H * K + hv * V + d)[None, :],
                mask=m,
            ),
            mask=m,
        )


def _split_conv_qkv(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous ``(q, k, v)`` as ``[1, T, heads, dim]`` in one launch.

    The recurrent decode kernel calls ``.contiguous()`` on q/k/v itself, so
    slicing the conv output and letting it copy costs three separate kernels
    per layer; this does the same movement in one.
    """
    n = mixed_qkv.shape[0]
    dev, dt = mixed_qkv.device, mixed_qkv.dtype
    q = torch.empty(1, n, num_k_heads, head_k_dim, dtype=dt, device=dev)
    k = torch.empty(1, n, num_k_heads, head_k_dim, dtype=dt, device=dev)
    v = torch.empty(1, n, num_v_heads, head_v_dim, dtype=dt, device=dev)
    if n == 0:
        return q, k, v
    block_t = 16 if n >= 16 else 1
    _split_conv_qkv_kernel[(triton.cdiv(n, block_t), num_k_heads + num_v_heads)](
        mixed_qkv,
        q,
        k,
        v,
        n,
        mixed_qkv.stride(0),
        num_k_heads * head_k_dim,
        num_v_heads * head_v_dim,
        H=num_k_heads,
        K=head_k_dim,
        V=head_v_dim,
        BLOCK_T=block_t,
        BK=triton.next_power_of_2(head_k_dim),
        BV=triton.next_power_of_2(head_v_dim),
    )
    return q, k, v


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
        reduce_output: bool = True,
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

        # Decay parameters (sharded across TP). ``A_log`` is FP32 in vLLM
        # (``qwen_gdn_linear_attn``: ``dtype=torch.float32``) while ``dt_bias``
        # is model dtype (``torch.ones(...)``); ``-exp(A_log)`` is the decay
        # rate, so rounding it costs precision the recurrence then compounds.
        self.A_log = nn.Parameter(
            torch.empty(self.local_v_heads, dtype=torch.float32),
        )
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

        # Output projection. ``reduce_output=False`` defers the all-reduce to
        # the decoder layer's next norm, which fuses the two.
        self.out_proj = RowParallelLinear(
            self.value_dim, hidden_size, reduce_results=reduce_output,
        )
        # Filled by process_weights_after_loading: both input projections
        # aliased into one buffer so they run as a single GEMM.
        self._in_proj_w: torch.Tensor | None = None
        self._qkvz_dim = qkvz_dim // tp

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

    def process_weights_after_loading(self) -> None:
        """Alias both input projections into one buffer, for a single GEMM.

        ``in_proj_ba`` is 32 columns wide per rank, which cuBLAS serves with a
        split-K GEMM plus its reduction for 4.5 us of pure launch latency, 36
        times per decode step -- 0.16 ms of a 4.03 ms step. Stacking its rows
        under ``in_proj_qkvz``'s computes the same dot products inside the wide
        GEMM that has to run anyway, and ``_unpack_qkvz_ba`` takes independent
        pointers and row strides, so both halves are read straight out of the
        joint output with no copy.

        Both parameters are rebound as contiguous *views* into the joint buffer,
        so nothing is duplicated at steady state -- keeping a second copy would
        have cost 25 MiB per layer, 0.9 GiB of KV cache. This has to run after
        loading rather than in ``__init__``: the loader moves the whole model
        with ``model.to(device, dtype)``, which reassigns every parameter's
        storage and would leave an earlier view dangling.
        """
        if self._in_proj_w is not None:
            return
        qkvz = self.in_proj_qkvz.weight
        ba = self.in_proj_ba.weight
        n_qkvz = qkvz.shape[0]
        merged = torch.empty(
            n_qkvz + ba.shape[0], qkvz.shape[1],
            dtype=qkvz.dtype, device=qkvz.device,
        )
        merged[:n_qkvz].copy_(qkvz.data)
        merged[n_qkvz:].copy_(ba.data)
        self._qkvz_dim = n_qkvz
        self.in_proj_qkvz.weight.data = merged[:n_qkvz]
        self.in_proj_ba.weight.data = merged[n_qkvz:]
        self._in_proj_w = merged

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

        # 1. Input projections as one GEMM, then one pass to deinterleave the
        # per-K-head groups into the packed [q|k|v] the conv wants plus
        # contiguous z/b/a.
        if self._in_proj_w is not None:
            proj = torch.nn.functional.linear(x_flat, self._in_proj_w)
            qkvz_out = proj[:, :self._qkvz_dim]
            ba_out = proj[:, self._qkvz_dim:]
        else:
            # process_weights_after_loading never ran (unit tests, or a loader
            # that does not call it); two GEMMs, same result.
            qkvz_out = self.in_proj_qkvz(x_flat)
            ba_out = self.in_proj_ba(x_flat)
        mixed_qkv, z, b, a = _unpack_qkvz_ba(
            qkvz_out, ba_out, self.local_k_heads, self.head_k_dim,
            self.head_v_dim, self.v_per_k,
        )

        # 2. Causal conv1d on the packed [q, k, v]
        conv_state = state_manager.gdn_conv[self.layer_idx]
        conv_weights = self.conv1d.weight
        cu_seqlens = md.query_start_loc_int32
        if cu_seqlens is None:
            cu_seqlens = md.non_spec_query_start_loc.to(torch.int32)
        if md.num_prefills > 0:
            mixed_qkv = _vllm_causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                None,
                conv_state,
                cu_seqlens,
                cache_indices=md.non_spec_state_indices_tensor,
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
                null_block_id=-1,
                validate_data=True,
            )

        # 3. Post-conv prep + recurrence. Prefill materializes g/beta because
        # the chunk kernel takes them as inputs, and vLLM's
        # ``fused_post_conv_prep`` produces them together with the split and
        # L2-normalised q/k in a single launch. Decode does not: vLLM's
        # ``fused_sigmoid_gating_delta_rule_update`` takes A_log/a/b/dt_bias and
        # keeps g and beta in fp32 registers inside the recurrent kernel, so
        # rounding beta to bf16 here would compound over a 512-token decode.
        recurrent_full = state_manager.recurrent[self.layer_idx]
        if md.num_prefills > 0:
            q_c, k_c, v_c, g, beta = _vllm_fused_post_conv_prep(
                conv_output=mixed_qkv,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.local_k_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            state_idx = md.state_indices_long
            if state_idx is None:
                state_idx = md.non_spec_state_indices_tensor.long()
            init_state = recurrent_full.index_select(0, state_idx)
            # Which slots carry state is decided host-side when the step's chunk
            # plan is built, so the mask never has to come back from the device
            # -- reading it with ``nonzero()`` here cost a full stream sync per
            # GDN layer, 36 of them per prefill step. The device mask is still
            # the fallback, so metadata that does not carry the host-side
            # summary is slower, never wrong.
            if md.has_initial_state is None or md.all_have_initial_state:
                pass
            elif md.any_have_initial_state:
                keep = md.has_initial_state.view(
                    -1, *([1] * (init_state.dim() - 1)),
                )
                init_state.masked_fill_(~keep, 0)
            else:
                init_state.zero_()
            if self._use_flashinfer_prefill:
                o, final_state = _flashinfer_gdn_prefill(
                    q=q_c.unsqueeze(0),
                    k=k_c.unsqueeze(0),
                    v=v_c.unsqueeze(0),
                    g=g.unsqueeze(0),
                    beta=beta.unsqueeze(0),
                    initial_state=init_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )
            else:
                o, final_state = _vllm_chunk_gated_delta_rule(
                    q=q_c.unsqueeze(0),
                    k=k_c.unsqueeze(0),
                    v=v_c.unsqueeze(0),
                    g=g.unsqueeze(0),
                    beta=beta.unsqueeze(0),
                    initial_state=init_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=False,
                )
            recurrent_full.index_copy_(
                0,
                state_idx,
                final_state.to(recurrent_full.dtype),
            )
        else:
            q_c, k_c, v_c = _split_conv_qkv(
                mixed_qkv, self.local_k_heads, self.head_k_dim,
                self.local_v_heads, self.head_v_dim,
            )
            o, _ = _vllm_fused_sigmoid_gating_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=q_c,
                k=k_c,
                v=v_c,
                initial_state=recurrent_full,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens[: md.num_decodes + 1],
                ssm_state_indices=md.non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )

        # 4. Output gating: RMSNorm(o) * silu(z) (L1 op)
        o_flat = o.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        o = self.norm(o_flat, z_flat).view(N, self.local_v_heads, self.head_v_dim)

        # 5. Output projection
        o = o.reshape(N, self.local_v_heads * self.head_v_dim)
        return self.out_proj(o)
