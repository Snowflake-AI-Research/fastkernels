"""DSA (DeepSeek Sparse Attention) indexer.

Produces top-k token indices for sparse attention. Components:
- wq_b: replicated linear (q_lora_rank -> head_dim * n_head) with FP8
- wk: replicated linear (hidden_size -> head_dim) with FP8
- k_norm: LayerNorm(head_dim, eps=1e-6)
- weights_proj: replicated linear (hidden_size -> n_head) — NO FP8
- Own K cache: [num_blocks, block_size, 132] uint8
"""

from __future__ import annotations

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

        # Shared buffer to avoid per-step allocation (matches vllm)
        self.topk_indices_buffer = topk_indices_buffer

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
            return torch.ops.kb_nano.sparse_attn_indexer(
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

        # K path: wk -> k_norm -> split pe/nope
        k = self.wk(hidden_states)  # [M, head_dim]
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

        weights = self.weights_proj(hidden_states)  # [M, n_head]
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head ** -0.5
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

            k_fp8, k_scale_bytes = self.k_cache_gather(
                self.indexer_k_cache, bt, cu_k)

            num_seqs = cu_k.shape[0] - 1
            seq_lens_k = cu_k[1:] - cu_k[:-1]
            cu_seqlen_ks, cu_seqlen_ke = _kv_spans_from_batches(
                cu_q, seq_lens_k, hidden_states.device, N=np_)

            logits = self.fp8_mqa_logits.forward_prefill(
                q_fp8_pf.view(-1, self.n_head, self.head_dim),
                (k_fp8, k_scale_bytes.view(torch.float32).flatten()),
                weights_pf,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )

            topk_indices[:np_] = self.topk_per_row.forward_prefill(
                logits, cu_seqlen_ks, cu_seqlen_ke,
                self.topk_tokens,
            )

            if ctx.is_mixed and ctx.num_decode_tokens > 0:
                nd = ctx.num_decode_tokens
                q_fp8_dc = q_fp8[np_:]
                weights_dc = weights[np_:]
                topk_indices[np_:] = self._decode_topk(
                    q_fp8_dc, weights_dc, ctx, hidden_states.device,
                )
        else:
            # Decode path (decode-only batch)
            topk_indices = self._decode_topk(q_fp8, weights, ctx, hidden_states.device)

        return topk_indices

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

        if device.type == "cuda":
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        else:
            num_sms = 1

        B = ctx.decode_context_lens.shape[0]
        next_n = M // B if B > 0 else 1

        schedule = self.paged_mqa_metadata(
            ctx.decode_context_lens,
            block_size,
            num_sms,
        )
        q_fp8_4d = q_fp8.view(B, next_n, self.n_head, self.head_dim)
        kv_cache_4d = self.indexer_k_cache.unsqueeze(-2)
        logits = self.fp8_mqa_logits.forward_decode(
            q_fp8_4d,
            kv_cache_4d,
            weights[:B * next_n],
            ctx.decode_context_lens,
            ctx.decode_block_tables,
            schedule,
            max_context_len=max_ctx,
        )
        return self.topk_per_row.forward_decode(
            logits, ctx.decode_context_lens, next_n=next_n, topk=self.topk_tokens,
        )
