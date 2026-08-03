"""DSA (DeepSeek Sparse Attention) indexer.

Produces top-k token indices for sparse attention. Components:
- wq_b: replicated linear (q_lora_rank -> head_dim * n_head) with FP8
- wk: replicated linear (hidden_size -> head_dim) with FP8
- k_norm: LayerNorm(head_dim, eps=1e-6)
- weights_proj: replicated linear (hidden_size -> n_head) — NO FP8
- Own K cache: [num_blocks, block_size, 132] uint8
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ....infra.context import get_context
from ..L1.layer_norm import LayerNorm
from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.indexer_k_cache import IndexerKCacheStore, IndexerKCacheGather
from ..L1.fp8_mqa_logits import Fp8MQALogits, Fp8PagedMQALogitsMetadata
from ..L1.top_k_per_row import TopKPerRow
from .parallel_linear import ReplicatedLinear


def _kv_spans_from_batches(
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    device: torch.device,
    N: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token KV span boundaries for causal prefill indexer logits.

    Args:
        cu_seqlens_q: [B+1] cumulative query token counts.
        seq_lens_k:   [B] full KV sequence length per batch.
        N: total number of query tokens. Pass explicitly to avoid a D2H
           sync on ``cu_seqlens_q[-1].item()``; callers know this count
           from the shape of the Q tensor.

    Returns:
        (cu_seqlen_ks, cu_seqlen_ke): both [N] int32, per-query-token
        start (inclusive) and end (exclusive) into concatenated KV.
    """
    q = cu_seqlens_q.long()
    L = seq_lens_k.long()
    B = L.numel()
    counts = q[1:] - q[:-1]
    if N is None:
        # Fallback: sync on cu_seqlens_q[-1]. Avoid this on the hot path.
        N = int(q[-1].item())
    if N == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty

    kv_starts = torch.cumsum(L, dim=0) - L
    batch_id = torch.repeat_interleave(torch.arange(B, device=device), counts)
    start_tensor = kv_starts[batch_id]

    L_expand = torch.repeat_interleave(L, counts)
    m_expand = torch.repeat_interleave(counts, counts)
    pos_within = (
        torch.arange(N, dtype=torch.long, device=device)
        - torch.repeat_interleave(q[:-1], counts)
        + 1
    )
    local_pos = L_expand - m_expand + pos_within
    end_location = start_tensor + local_pos

    return start_tensor.int(), end_location.int()


# Bound on the per-chunk indexer logits tensor (M*N*4 bytes). Mirrors vLLM's
# VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (default 512; vllm/envs.py).
_INDEXER_MAX_LOGITS_MB = int(
    os.environ.get("FASTKERNELS_SPARSE_INDEXER_MAX_LOGITS_MB", "512"))


def _split_prefill_chunks(
    seq_lens: list[int],
    query_lens: list[int],
    workspace_size: int,
    max_logits_elems: int,
) -> list[tuple[int, int, int, int, bool]]:
    """Split prefill requests into chunks for the DSA indexer, respecting the
    N-constraint (Σ seq_lens_k ≤ workspace_size, bounds the reused K-gather
    workspace) and the logits-constraint (M·N ≤ max_logits_elems, bounds the
    per-chunk logits tensor). When a single request alone exceeds the logits
    budget, sub-chunks on the query (M) dimension. Verbatim port of vLLM's
    ``split_indexer_prefill_chunks`` (vllm/v1/attention/backends/mla/indexer.py).

    Returns ``(r0, r1, q_off, sub_m, skip_gather)`` tuples: request span
    ``[r0, r1)``, query sub-range ``[q_off, q_off+sub_m)`` within the chunk's
    query block, and ``skip_gather`` (True for M-subchunks after the first —
    the single request's KV is already resident in the workspace).
    """
    chunks: list[tuple[int, int, int, int, bool]] = []
    n = len(seq_lens)
    end = 0
    while end < n:
        start = end
        chunk_m = chunk_n = 0
        while end < n:
            q, s = query_lens[end], seq_lens[end]
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break
        # A single request can exceed the logits budget -> M-subchunking.
        if end == start:
            chunk_m, chunk_n = query_lens[end], seq_lens[end]
            end += 1
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        first = True
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((start, end, q_off, sub_m, not first))
            first = False
    return chunks


class SparseAttnIndexer(nn.Module):
    """DSA sparse attention indexer.

    Produces topk_indices [M, topk_tokens] consumed by sparse FlashMLA.

    Args:
        hidden_size: model hidden dimension
        q_lora_rank: query latent dimension
        n_head: number of indexer heads (64)
        head_dim: indexer head dimension (128)
        rope_dim: RoPE dimension (64)
        topk_tokens: number of tokens to select per query
        quant_config: FP8 quantization config
    """

    def __init__(self, hidden_size: int, q_lora_rank: int,
                 n_head: int, head_dim: int, rope_dim: int,
                 topk_tokens: int, quant_config: dict | None = None,
                 topk_indices_buffer: torch.Tensor | None = None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.topk_tokens = topk_tokens
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = head_dim ** -0.5

        self.wq_b = ReplicatedLinear(
            q_lora_rank, head_dim * n_head,
            bias=False, quant_config=quant_config,
        )
        self.wk = ReplicatedLinear(
            hidden_size, head_dim,
            bias=False, quant_config=quant_config,
        )
        self.k_norm = LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size, n_head, bias=False,  # NO FP8
        )

        self.k_cache_store = IndexerKCacheStore()
        self.k_cache_gather = IndexerKCacheGather()
        self.fp8_mqa_logits = Fp8MQALogits()
        self.paged_mqa_metadata = Fp8PagedMQALogitsMetadata()
        self.topk_per_row = TopKPerRow()
        self.fp8_quant = PerTokenGroupQuantFp8()

        # Indexer K cache: [num_blocks, block_size, 132] uint8
        self.indexer_k_cache = torch.tensor([])

        self._quant_block_size = 128

        # Opt-in run-to-run determinism for the DSA sparse path (see forward_impl).
        self._sort_topk = os.environ.get("FASTKERNELS_DSA_SORT_TOPK", "0") != "0"

        # Fused wk + weights_proj weight, built after load by
        # ``DeepSeekMLAAttention.compute_absorbed_weights`` (which also
        # dequantizes the FP8 ``wk`` to BF16). vLLM computes wk AND weights_proj
        # in ONE bf16 GEMM (``wk_weights_proj``, deepseek_v2.py:636-643/706-708);
        # running weights_proj as a SEPARATE GEMM picks a different-N cuBLAS
        # path. Fusing matches vLLM's exact invocation.
        # (Note: this removes the fused-vs-separate divergence but the residual
        # cross-process cuBLAS bf16 accumulation difference is irreducible — see
        # the long-context findings memo.) ``None`` until finalized (bf16
        # checkpoints / warmup fall back to the separate wk/weights_proj modules).
        # Plain attribute (not a registered buffer): derived at load, kept
        # on-device by construction, and read inside the opaque
        # ``fastkernels::sparse_attn_indexer`` custom op which runs eagerly.
        self._wk_wp_fused: torch.Tensor | None = None

        # Shared buffer to avoid per-step allocation (matches vllm)
        self.topk_indices_buffer = topk_indices_buffer

        # Reused prefill K-gather workspace (one buffer shared by all indexer
        # layers, attached by the engine at KV-cache alloc). When present,
        # prefill chunks over it (bounded memory, no per-step alloc); when
        # absent (warmup / non-engine callers) forward falls back to single-shot.
        self._gather_ws_k_fp8: torch.Tensor | None = None
        self._gather_ws_k_scale: torch.Tensor | None = None
        self._gather_ws_tokens: int = 0
        self._max_logits_elems: int = _INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4

        # Custom-op dispatch scaffolding (matches MLAAttention / Attention).
        self._use_custom_op = False
        self._layer_name = ""
        # Reference to the enclosing ``YarnRotaryEmbedding`` for indexer RoPE.
        # Wired up by the parent ``DeepSeekMLAAttention``; stored via
        # ``object.__setattr__`` to avoid double-registration as a submodule.
        self._rope_emb: nn.Module | None = None

    def forward(self, hidden_states: torch.Tensor, q_latent: torch.Tensor,
                positions: torch.Tensor,
                rope_emb: nn.Module | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [M, hidden_size]
            q_latent: [M, q_lora_rank] - compressed query from fused_qkv_a_proj
            positions: [M] position ids
            rope_emb: YarnRotaryEmbedding for indexer (optional if already
                     wired via ``self._rope_emb``).

        Returns:
            topk_indices: [M, topk_tokens] int32
        """
        if rope_emb is not None and self._rope_emb is None:
            object.__setattr__(self, "_rope_emb", rope_emb)

        if self._use_custom_op:
            return torch.ops.fastkernels.sparse_attn_indexer(
                hidden_states, q_latent, positions, self._layer_name,
            )
        return self.forward_impl(hidden_states, q_latent, positions)

    def forward_impl(self, hidden_states: torch.Tensor, q_latent: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
        rope_emb = self._rope_emb
        assert rope_emb is not None, "SparseAttnIndexer._rope_emb is not wired"
        ctx = get_context()
        M = hidden_states.shape[0]

        # Q path: wq_b -> reshape -> split pe/nope -> RoPE -> concat
        q = self.wq_b(q_latent)  # [M, head_dim * n_head]
        q = q.view(M, self.n_head, self.head_dim)
        q_pe = q[..., :self.rope_dim]  # [M, n_head, rope_dim]
        q_nope = q[..., self.rope_dim:]  # [M, n_head, head_dim - rope_dim]

        # K path: (fused wk|weights_proj) -> split -> k_norm -> split pe/nope.
        # vLLM runs wk and weights_proj as ONE fused bf16 GEMM; we replicate it
        # (single ``F.linear`` over the concatenated weight) so both ``k`` and
        # the weights_proj output are bit-identical to vLLM. Only ``k`` goes
        # through ``k_norm`` (matches vLLM: weights split off before the norm).
        if self._wk_wp_fused is not None:
            kw = torch.nn.functional.linear(hidden_states, self._wk_wp_fused)
            k = kw[:, :self.head_dim]  # [M, head_dim]
            wp_out = kw[:, self.head_dim:]  # [M, n_head]
        else:
            k = self.wk(hidden_states)  # [M, head_dim]
            wp_out = self.weights_proj(hidden_states)  # [M, n_head]
        k = self.k_norm(k)
        k_pe = k[:, :self.rope_dim]  # [M, rope_dim]
        k_nope = k[:, self.rope_dim:]  # [M, head_dim - rope_dim]

        # RoPE on pe components
        q_pe, k_pe_out = rope_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(M, self.n_head, self.rope_dim)
        k_pe_out = k_pe_out.reshape(M, 1, self.rope_dim)

        # Concat pe + nope
        q = torch.cat([q_pe, q_nope], dim=-1)  # [M, n_head, head_dim]
        k = torch.cat([k_pe_out.squeeze(1), k_nope], dim=-1)  # [M, head_dim]

        # FP8 quantize Q via the public L1 op
        q_flat = q.reshape(-1, self.head_dim).contiguous()
        q_fp8 = torch.empty_like(q_flat, dtype=torch.float8_e4m3fn)
        q_scale = torch.empty(
            q_flat.shape[0], self.head_dim // self._quant_block_size,
            dtype=torch.float32, device=q_flat.device,
        )
        self.fp8_quant(q_flat, q_fp8, q_scale)
        q_fp8 = q_fp8.view(M, self.n_head, self.head_dim)
        q_scale = q_scale.view(M, self.n_head, -1)

        # Store K to indexer cache
        if ctx.slot_mapping is not None and self.indexer_k_cache.numel():
            self.k_cache_store(k, self.indexer_k_cache, ctx.slot_mapping)

        # ``wp_out`` is the weights_proj output from the fused GEMM above
        # (bit-identical to vLLM). Fold in the per-token Q scale + softmax
        # scale exactly as vLLM (deepseek_v2.py:738-741).
        weights = (
            wp_out.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head ** -0.5
        )
        weights = weights.squeeze(-1)

        # Use pre-allocated buffer if available, otherwise allocate
        if self.topk_indices_buffer is not None and M <= self.topk_indices_buffer.shape[0]:
            buf = self.topk_indices_buffer
            if buf.device != hidden_states.device:
                buf = buf.to(hidden_states.device)
                self.topk_indices_buffer = buf
            topk_indices = buf[:M, :self.topk_tokens]
            topk_indices.fill_(-1)
        else:
            topk_indices = torch.full((M, self.topk_tokens), -1, dtype=torch.int32,
                                      device=hidden_states.device)

        if ctx.is_prefill or (ctx.is_mixed and ctx.num_prefill_tokens > 0):
            # Prefill path: gather K from cache, compute logits, top-k
            if ctx.is_mixed:
                np_ = ctx.num_prefill_tokens
                cu_q = ctx.prefill_cu_seqlens_q
                cu_k = ctx.prefill_cu_seqlens_k
                bt = ctx.prefill_block_tables
                q_fp8_pf = q_fp8[:np_]
                weights_pf = weights[:np_]
            else:
                np_ = M
                cu_q = ctx.cu_seqlens_q
                cu_k = ctx.cu_seqlens_k
                bt = ctx.block_tables
                q_fp8_pf = q_fp8
                weights_pf = weights

            if cu_q is None or cu_k is None or bt is None:
                return topk_indices

            # Gather + logits + top-k, chunked over the reused workspace when one
            # is wired (bounded memory for long context); writes results in place
            # into ``topk_indices[:np_]``. Bit-identical (per-query top-k set) to
            # single-shot — the kernel reads exactly each query's [ks,ke).
            self._prefill_topk(
                ctx, q_fp8_pf, weights_pf, cu_q, cu_k, bt, np_,
                topk_indices, hidden_states.device,
            )

            if ctx.is_mixed and ctx.num_decode_tokens > 0:
                nd = ctx.num_decode_tokens
                q_fp8_dc = q_fp8[np_:]
                weights_dc = weights[np_:]
                topk_indices[np_:] = self._decode_topk(
                    q_fp8_dc, weights_dc, ctx, hidden_states.device,
                )
        else:
            # Decode-only batch. Write results INTO the (possibly shared) buffer
            # view — do NOT rebind to a fresh tensor. Under DSA index sharing the
            # skip layers reuse this shared ``topk_indices_buffer``; rebinding
            # would leave it at -1 and feed skip layers invalid indices on every
            # decode step. Matches vLLM, which writes the decode top-k straight
            # into ``topk_indices_buffer`` (sparse_attn_indexer.py:335).
            topk_indices[:] = self._decode_topk(
                q_fp8, weights, ctx, hidden_states.device)

        # Optional determinism: the radix top-k kernels (cooperative_topk /
        # persistent_topk / top_k_per_row_prefill) select the correct top-k SET
        # but emit it in a NONDETERMINISTIC ORDER run-to-run when there is real
        # sparsity (n_valid > topk). The downstream ``flash_mla_sparse_fwd`` is
        # order-sensitive (~1e-4 per permutation), so that order race makes
        # greedy decode nondeterministic at seq>index_topk — for vLLM too, since
        # these are the same vendored kernels. Sorting the valid indices into a
        # canonical ascending order (padding -1 last) makes fk run-to-run
        # deterministic AND batch-invariant. OFF by default (matches vLLM's
        # native, nondeterministic kernel order); enable with
        # FASTKERNELS_DSA_SORT_TOPK=1.
        if self._sort_topk:
            _INT_MAX = 2147483647
            tmp = torch.where(topk_indices >= 0, topk_indices,
                              torch.full_like(topk_indices, _INT_MAX))
            tmp, _ = torch.sort(tmp, dim=-1)
            topk_indices[:] = torch.where(
                tmp == _INT_MAX, torch.full_like(tmp, -1), tmp)

        return topk_indices

    def _build_prefill_chunk_meta(self, cu_q, cu_k, np_, device):
        """Compute the prefill chunk plan + per-chunk causal spans ONCE per
        forward (cached on the Context, shared by all indexer layers — matches
        vLLM's metadata builder). Returns a list of per-chunk tuples
        ``(r0, r1, skip_gather, cu_k_chunk, ks, ke, tok0, tok1, total_tokens)``.
        """
        cu_q_cpu = cu_q.to("cpu", torch.int64).tolist()
        cu_k_cpu = cu_k.to("cpu", torch.int64).tolist()
        seq_list = [cu_k_cpu[i + 1] - cu_k_cpu[i] for i in range(len(cu_k_cpu) - 1)]
        query_list = [cu_q_cpu[i + 1] - cu_q_cpu[i] for i in range(len(cu_q_cpu) - 1)]
        plan = _split_prefill_chunks(
            seq_list, query_list, self._gather_ws_tokens, self._max_logits_elems)
        seq_lens_k_t = cu_k[1:] - cu_k[:-1]
        meta = []
        for (r0, r1, q_off, sub_m, skip_gather) in plan:
            total_tokens = sum(seq_list[r0:r1])
            cu_k_chunk = None if skip_gather else (cu_k[r0:r1 + 1] - cu_k[r0])
            tok0 = cu_q_cpu[r0] + q_off
            tok1 = tok0 + sub_m
            full_q = cu_q_cpu[r1] - cu_q_cpu[r0]
            if q_off == 0 and sub_m == full_q:
                # Request-level chunk covering the FULL query range of [r0:r1):
                # chunk-local causal spans via the shared span builder on the
                # rebased cu_q (KV offsets start at 0 within the chunk's gather).
                cu_q_chunk = (cu_q[r0:r1 + 1] - cu_q[r0]).to(torch.int32)
                ks, ke = _kv_spans_from_batches(
                    cu_q_chunk, seq_lens_k_t[r0:r1], device, N=sub_m)
            else:
                # Single-request M-subchunk: attend from KV start (ks=0) up to
                # each query's causal cutoff = num_computed + q_off + j + 1.
                num_computed = seq_list[r0] - query_list[r0]
                ks = torch.zeros(sub_m, dtype=torch.int32, device=device)
                ke = (num_computed + q_off + 1
                      + torch.arange(sub_m, dtype=torch.int32, device=device))
            meta.append((r0, r1, skip_gather, cu_k_chunk, ks, ke,
                         tok0, tok1, total_tokens))
        return meta

    def _prefill_topk(self, ctx, q_fp8_pf, weights_pf, cu_q, cu_k, bt, np_,
                      topk_indices, device) -> None:
        """Prefill indexer top-k, written in place into ``topk_indices[:np_]``.

        Single-shot when no reused workspace is wired (warmup / non-engine
        callers); otherwise chunked over the workspace (vLLM
        ``split_indexer_prefill_chunks``), with the chunk plan computed once per
        forward and cached on ``ctx``. Chunking preserves each query's exact
        [ks,ke), so the per-query top-k SET is identical to single-shot (the
        radix kernel's within-row order may differ, which is irrelevant to the
        order-invariant sparse attention that consumes these indices).
        """
        if self._gather_ws_k_fp8 is None:
            # Single-shot fallback (writing in place).
            k_fp8, k_scale = self.k_cache_gather(self.indexer_k_cache, bt, cu_k)
            ks, ke = _kv_spans_from_batches(
                cu_q, cu_k[1:] - cu_k[:-1], device, N=np_)
            logits = self.fp8_mqa_logits.forward_prefill(
                q_fp8_pf.view(-1, self.n_head, self.head_dim),
                (k_fp8, k_scale.view(torch.float32).flatten()),
                weights_pf, ks, ke,
            )
            self.topk_per_row.forward_prefill(
                logits, ks, ke, self.topk_tokens, out=topk_indices[:np_])
            return

        # Chunked path: build the plan once per forward (cached on ctx).
        meta = getattr(ctx, "indexer_prefill_meta", None)
        if meta is None:
            meta = self._build_prefill_chunk_meta(cu_q, cu_k, np_, device)
            ctx.indexer_prefill_meta = meta

        ws_k_fp8 = self._gather_ws_k_fp8
        ws_k_scale = self._gather_ws_k_scale
        for (r0, r1, skip_gather, cu_k_chunk, ks, ke,
             tok0, tok1, total_tokens) in meta:
            if not skip_gather:
                # Gather chunk KV into the reused workspace (precomputed
                # total_tokens -> no D2H sync; out_* reuse the workspace).
                k_fp8, k_scale = self.k_cache_gather(
                    self.indexer_k_cache, bt[r0:r1], cu_k_chunk,
                    total_tokens=total_tokens,
                    out_k_fp8=ws_k_fp8, out_k_scale=ws_k_scale,
                )
            else:
                k_fp8 = ws_k_fp8[:total_tokens]
                k_scale = ws_k_scale[:total_tokens]
            logits = self.fp8_mqa_logits.forward_prefill(
                q_fp8_pf[tok0:tok1].view(-1, self.n_head, self.head_dim),
                (k_fp8, k_scale.view(torch.float32).flatten()),
                weights_pf[tok0:tok1], ks, ke,
            )
            self.topk_per_row.forward_prefill(
                logits, ks, ke, self.topk_tokens,
                out=topk_indices[tok0:tok1],
            )

    def _decode_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        ctx,
        device: torch.device,
    ) -> torch.Tensor:
        """Paged FP8 MQA logits + top-k for decode."""
        M = q_fp8.shape[0]
        out = torch.full((M, self.topk_tokens), -1, dtype=torch.int32, device=device)

        if not self.indexer_k_cache.numel():
            return out
        if ctx.decode_context_lens is None or ctx.decode_block_tables is None:
            return out

        block_size = int(self.indexer_k_cache.shape[1])
        max_ctx = int(ctx.decode_max_context_len or ctx.max_context_len or 1)

        B = ctx.decode_context_lens.shape[0]
        next_n = M // B if B > 0 else 1

        # The 2D context lengths and the DeepGEMM schedule are BATCH metadata:
        # vLLM builds them once per step (``decode_metadata.seq_lens`` /
        # ``decode_metadata.schedule_metadata`` in its metadata builder), while
        # every compute layer here would otherwise rebuild them -- ~20 layers per
        # step for GLM-5.2's index_topk_freq=4, each paying a device-side
        # ``get_paged_mqa_logits_metadata`` launch. Memoize on the per-step
        # context.
        _memo = getattr(ctx, "_fk_indexer_decode_meta", None)
        if _memo is not None and _memo[0] == next_n:
            cl_2d, schedule = _memo[1], _memo[2]
        else:
            if device.type == "cuda":
                num_sms = torch.cuda.get_device_properties(
                    device).multi_processor_count
            else:
                num_sms = 1
            # deep_gemm's paged MQA-logits API requires 2D context_lens
            # (B, next_n): (B, 1) for normal decode, per-position causal lengths
            # for spec decode. vLLM feeds decode_metadata.seq_lens the same way
            # (2D) — see vllm sparse_attn_indexer.py:299-301 "deep_gemm ...
            # requires 2D context_lens". (deep_gemm asserts
            # context_lens.dim()==2.) The radix top-k kernel below still takes
            # the 1D [B] per-sequence lengths.
            cl = ctx.decode_context_lens.to(torch.int32)
            if next_n == 1:
                cl_2d = cl.view(B, 1)
            else:
                j = torch.arange(next_n, device=device, dtype=torch.int32)
                cl_2d = (
                    cl.view(B, 1) - next_n + 1 + j.view(1, next_n)
                ).clamp_min_(0)
            cl_2d = cl_2d.contiguous()
            schedule = self.paged_mqa_metadata(cl_2d, block_size, num_sms)
            ctx._fk_indexer_decode_meta = (next_n, cl_2d, schedule)

        q_fp8_4d = q_fp8.view(B, next_n, self.n_head, self.head_dim)
        kv_cache_4d = self.indexer_k_cache.unsqueeze(-2)
        logits = self.fp8_mqa_logits.forward_decode(
            q_fp8_4d,
            kv_cache_4d,
            weights[:B * next_n],
            cl_2d,
            ctx.decode_block_tables,
            schedule,
            max_context_len=max_ctx,
        )
        return self.topk_per_row.forward_decode(
            logits, ctx.decode_context_lens, next_n=next_n, topk=self.topk_tokens,
            max_seq_len=max_ctx,
        )
