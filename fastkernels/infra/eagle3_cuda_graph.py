"""CUDA graph capture / replay for the EAGLE-3 hot path.

Mirrors sglang's three captured forwards
(`/home/yak/vllm_repo/sglang/python/sglang/srt/speculative/`):

  * ``TargetVerifyGraphRunner``  -- captures the 32-layer 8B target verify
    forward (``model + greedy argmax``) on ``B*N`` tokens.
  * ``DraftChainGraphRunner``    -- captures the entire ``S-1``-step draft
    decode loop on ``B*K`` tokens (one graph per B; **Phase B**).
  * ``DraftExtendGraphRunner``   -- captures ``draft.forward_draft +
    compute_logits + topk`` for the post-verify digest (**Phase C**).

For each runner we capture **one graph per batch size** in
``[1..cuda_graph_max_bs]``. Target verify additionally captures several
prefix-length buckets because FA3 bakes ``max_seqlen_k`` into the graph. At
replay we ``bisect_left`` the real ``B`` to the smallest captured ``B' >= B``
and pad the trailing rows with sentinels
(``slot_mapping = scratch_slot``, ``cache_seqlens = 1``,
``block_table = scratch_block``).  Outputs are sliced back to ``[:B*N]``.

Persistent buffers
==================
All input tensors that the captured kernels read from MUST live in
allocations with stable data pointers (CUDA graphs bake kernel-arg
addresses at capture time).  Each runner owns one set of input buffers
sized at ``B_max`` and copies real data into the prefix on every replay.

Output handling
===============
Target verify reads graph-owned output tensors directly after replay to avoid
large per-step aux-hidden copies. Draft graph outputs are still copied into
runner-owned buffers because those outputs are small and shared across bucket
graphs.

Scratch slot
============
The engine reserves the first block of both KV caches as a "scratch" page
that is never used by real sequences; ghost rows write to / read from
slot 0 of that block.  ``cache_seqlens = 1`` keeps FA3 happy (it always
reads at least one valid token's worth of K/V).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from .context import set_forward_context

if TYPE_CHECKING:
    from .eagle3_engine import LlamaEagle3Engine


# ---------------------------------------------------------------------------
# Persistent buffers
# ---------------------------------------------------------------------------

@dataclass
class _VerifyBuffers:
    """Persistent input buffers for ``TargetVerifyGraphRunner``.

    All tensors are sized at ``B_max * N`` tokens / ``B_max`` sequences so a
    single set of allocations services every captured bucket.  Captured
    graphs use sliced views ``buf[:B*N]`` / ``buf[:B]`` as kernel inputs.
    """

    # Inputs
    input_ids: torch.Tensor                 # [B_max*N] int64
    positions: torch.Tensor                 # [B_max*N] int64
    slot_mapping: torch.Tensor              # [B_max*N] int32
    block_table_prefix: torch.Tensor        # [B_max, max_blocks] int32
    cu_seqlens_q_prefix: torch.Tensor       # [B_max+1] int32, fixed arange
    cache_seqlens_prefix: torch.Tensor      # [B_max] int32
    page_table_expand: torch.Tensor         # [B_max*N, N] int32
    cache_seqlens_expand: torch.Tensor      # [B_max*N] int32
    cu_seqlens_q_expand: torch.Tensor       # [B_max*N+1] int32, fixed arange
    context_lens: torch.Tensor              # [B_max] int32 (legacy, unused)

    # Outputs
    hidden_states: Optional[torch.Tensor] = None  # graph path does not export h


# ---------------------------------------------------------------------------
# Target verify runner
# ---------------------------------------------------------------------------

class TargetVerifyGraphRunner:
    """Captures one CUDA graph per ``B`` for the target verify forward.

    The captured region is::

        h, aux = target.model(input_ids, positions)
        target_predict = target.compute_logits(h).argmax(dim=-1)

    Everything before the model call (slot/block-table construction,
    cascade-metadata builds) runs Python-side per replay and writes results
    into persistent input buffers via ``copy_()``. Outputs are graph-owned
    tensors saved from capture and read directly after replay.
    """

    def __init__(
        self,
        engine: "LlamaEagle3Engine",
        cuda_graph_max_bs: int,
        scratch_block_t: int,
        graph_pool=None,
    ):
        self.engine = engine
        self.B_max = cuda_graph_max_bs
        self.N = engine.num_draft_tokens
        self.scratch_block_t = scratch_block_t
        self.scratch_slot_t = scratch_block_t * engine.block_size
        # Capture every integer batch size up to the max -- mirrors sglang's
        # spec-decoding default schedule.
        self.capture_bs: List[int] = list(range(1, self.B_max + 1))
        prefix_buckets = [160, 256, 512, 1024, 2048, engine.max_model_len]
        self.capture_prefix_lens: List[int] = sorted({
            min(engine.max_model_len, b) for b in prefix_buckets
            if b > 0
        })
        self.graph_pool = graph_pool

        device = engine.device
        block_size = engine.block_size
        max_blocks = (engine.max_model_len + block_size - 1) // block_size
        N = self.N
        B_max = self.B_max

        # --- input buffers ------------------------------------------------
        bufs = _VerifyBuffers(
            input_ids=torch.zeros(B_max * N, dtype=torch.int64, device=device),
            positions=torch.zeros(B_max * N, dtype=torch.int64, device=device),
            slot_mapping=torch.full(
                (B_max * N,), self.scratch_slot_t,
                dtype=torch.int32, device=device,
            ),
            block_table_prefix=torch.full(
                (B_max, max_blocks), scratch_block_t,
                dtype=torch.int32, device=device,
            ),
            cu_seqlens_q_prefix=torch.arange(
                0, (B_max + 1) * N, N, dtype=torch.int32, device=device,
            ),
            cache_seqlens_prefix=torch.ones(
                B_max, dtype=torch.int32, device=device,
            ),
            page_table_expand=torch.full(
                (B_max * N, N), self.scratch_slot_t,
                dtype=torch.int32, device=device,
            ),
            cache_seqlens_expand=torch.ones(
                B_max * N, dtype=torch.int32, device=device,
            ),
            cu_seqlens_q_expand=torch.arange(
                B_max * N + 1, dtype=torch.int32, device=device,
            ),
            context_lens=torch.ones(B_max, dtype=torch.int32, device=device),
        )
        self.bufs = bufs
        self.max_blocks = max_blocks

        # Captured graphs ((B, max_prefix_len) -> CUDAGraph). Outputs are read
        # directly from graph-owned tensors captured in self.outputs.
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.outputs: dict[
            tuple[int, int], tuple[List[torch.Tensor], torch.Tensor]
        ] = {}

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture_all(self) -> None:
        """Capture one graph per ``B`` in ``self.capture_bs``.

        Captures must run in DESCENDING order of ``B`` so that the largest
        allocation pattern is recorded first; the shared mempool then
        reuses those slabs for smaller buckets.  Sglang follows the same
        order for the same reason.
        """
        for prefix_len in sorted(self.capture_prefix_lens, reverse=True):
            for B in sorted(self.capture_bs, reverse=True):
                self._capture_one(B, prefix_len)
        torch.cuda.synchronize()

    def _capture_one(self, B: int, max_seqlen_k_prefix: int) -> None:
        bufs = self.bufs
        N = self.N
        BN = B * N

        # Sliced views into persistent buffers (kernel-arg pointers).
        in_ids = bufs.input_ids[:BN]
        positions = bufs.positions[:BN]
        slot_mapping = bufs.slot_mapping[:BN]
        bt_prefix = bufs.block_table_prefix[:B]
        cu_q_prefix = bufs.cu_seqlens_q_prefix[:B + 1]
        cache_seq_prefix = bufs.cache_seqlens_prefix[:B]
        pt_expand = bufs.page_table_expand[:BN]
        cache_seq_expand = bufs.cache_seqlens_expand[:BN]
        cu_q_expand = bufs.cu_seqlens_q_expand[:BN + 1]
        ctx_lens = bufs.context_lens[:B]

        # max_seqlen_k_prefix is a kernel launch parameter baked into the graph.
        # Capture several buckets so replay can use a tight launch shape for
        # short prompts without sacrificing correctness on longer contexts.

        def _run_forward():
            with set_forward_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=ctx_lens,
                block_tables=bt_prefix,
                max_context_len=max_seqlen_k_prefix,
                is_tree_verify=True,
                tree_num_verify_tokens=N,
                tree_block_table_prefix=bt_prefix,
                tree_cache_seqlens_prefix=cache_seq_prefix,
                tree_cu_seqlens_q_prefix=cu_q_prefix,
                tree_max_seqlen_q_prefix=N,
                tree_max_seqlen_k_prefix=max_seqlen_k_prefix,
                tree_page_table_expand=pt_expand,
                tree_cache_seqlens_expand=cache_seq_expand,
                tree_cu_seqlens_q_expand=cu_q_expand,
                is_cuda_graph_replay=True,
            ):
                out = self.engine.target.model(in_ids, positions)
                if isinstance(out, tuple):
                    h_local, aux_local = out
                else:
                    h_local, aux_local = out, []
                # Greedy verify only needs argmax. Avoid compute_logits(),
                # which casts the full [B*N, vocab] projection to fp32.
                # EAGLE-3 is TP=1 in this engine, so no vocab gather is needed.
                logits_local = self.engine.target.lm_head.project(h_local)
                target_predict_local = logits_local.argmax(dim=-1)
            return h_local, aux_local, target_predict_local

        # Warmup: run forward once eagerly to JIT kernels and prime caches.
        # PyTorch's CUDA graph capture requires a separate warmup pass so
        # the caching allocator state is stable.
        h_w, aux_w, pred_w = _run_forward()
        del h_w, aux_w, pred_w
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            h_local, aux_local, target_predict_local = _run_forward()

        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        key = (B, max_seqlen_k_prefix)
        self.graphs[key] = graph
        self.outputs[key] = (aux_local, target_predict_local)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def replay(
        self,
        input_ids_real: torch.Tensor,         # [B*N] int64
        positions_real: torch.Tensor,         # [B*N] int64
        slot_mapping_real: torch.Tensor,      # [B*N] int32
        block_table_real: torch.Tensor,       # [B, k_real] int32
        cache_seqlens_prefix_real: torch.Tensor,  # [B] int32
        page_table_expand_real: torch.Tensor,     # [B*N, N] int32
        cache_seqlens_expand_real: torch.Tensor,  # [B*N] int32
        max_seqlen_k_prefix_real: int,
        raw_bs: int,
    ):
        """Pad the real inputs to the smallest captured bucket and replay.

        Returns sliced views of the persistent output buffers
        ``(hidden_states, aux_list, target_predict)``. ``hidden_states`` is
        always ``None`` in the graph path because the caller only needs EAGLE
        aux hidden states for the accepted path.
        """
        N = self.N
        B_pad = self.capture_bs[bisect.bisect_left(self.capture_bs, raw_bs)]
        prefix_idx = bisect.bisect_left(
            self.capture_prefix_lens, max_seqlen_k_prefix_real,
        )
        prefix_cap = self.capture_prefix_lens[prefix_idx]
        BN_real = raw_bs * N
        BN_pad = B_pad * N

        bufs = self.bufs
        scratch_slot = self.scratch_slot_t
        scratch_block = self.scratch_block_t

        # --- padding: reset ghost regions to scratch defaults -------------
        if BN_pad > BN_real:
            bufs.input_ids[BN_real:BN_pad].zero_()
            bufs.positions[BN_real:BN_pad].zero_()
            bufs.slot_mapping[BN_real:BN_pad].fill_(scratch_slot)
            bufs.cache_seqlens_expand[BN_real:BN_pad].fill_(1)
            bufs.page_table_expand[BN_real:BN_pad].fill_(scratch_slot)
        if B_pad > raw_bs:
            bufs.cache_seqlens_prefix[raw_bs:B_pad].fill_(1)
            bufs.block_table_prefix[raw_bs:B_pad].fill_(scratch_block)

        # --- copy real data into persistent prefixes ---------------------
        if input_ids_real.data_ptr() != bufs.input_ids[:BN_real].data_ptr():
            bufs.input_ids[:BN_real].copy_(input_ids_real, non_blocking=True)
        if positions_real.data_ptr() != bufs.positions[:BN_real].data_ptr():
            bufs.positions[:BN_real].copy_(positions_real, non_blocking=True)
        if slot_mapping_real.data_ptr() != bufs.slot_mapping[:BN_real].data_ptr():
            bufs.slot_mapping[:BN_real].copy_(slot_mapping_real, non_blocking=True)

        k_real = block_table_real.shape[1]
        if k_real >= self.max_blocks:
            bufs.block_table_prefix[:raw_bs].copy_(
                block_table_real[:, :self.max_blocks], non_blocking=True,
            )
        else:
            bufs.block_table_prefix[:raw_bs, :k_real].copy_(
                block_table_real, non_blocking=True,
            )
            bufs.block_table_prefix[:raw_bs, k_real:].fill_(scratch_block)

        if (
            cache_seqlens_prefix_real.data_ptr()
            != bufs.cache_seqlens_prefix[:raw_bs].data_ptr()
        ):
            bufs.cache_seqlens_prefix[:raw_bs].copy_(
                cache_seqlens_prefix_real, non_blocking=True,
            )
        if (
            cache_seqlens_expand_real.data_ptr()
            != bufs.cache_seqlens_expand[:BN_real].data_ptr()
        ):
            bufs.cache_seqlens_expand[:BN_real].copy_(
                cache_seqlens_expand_real, non_blocking=True,
            )
        if (
            page_table_expand_real.data_ptr()
            != bufs.page_table_expand[:BN_real].data_ptr()
        ):
            bufs.page_table_expand[:BN_real].copy_(
                page_table_expand_real, non_blocking=True,
            )

        # --- replay ------------------------------------------------------
        key = (B_pad, prefix_cap)
        self.graphs[key].replay()
        aux_list, target_predict = self.outputs[key]

        return (
            None,
            [a[:BN_real] for a in aux_list],
            target_predict[:BN_real],
        )


# ---------------------------------------------------------------------------
# Draft chain runner (Phase B)
# ---------------------------------------------------------------------------

@dataclass
class _ChainBuffers:
    """Persistent input + output buffers for ``DraftChainGraphRunner``.

    The captured graph runs the full ``S-1`` step decode loop.  Per-step
    metadata (slot_mapping, cache_seqlens, positions) is computed *inside*
    the captured graph from a few high-level Python-side inputs
    (``base_pos``, ``r_per_seq``, ``n_full_prefix_t``, ``bt_branch``).
    """

    # --- inputs -------------------------------------------------------
    cur_top_i_t: torch.Tensor       # [B_max*K] int64  -- step-0 input ids
    cur_hidden: torch.Tensor        # [B_max*K, H_d]   -- step-0 hidden
    cur_top_p: torch.Tensor         # [B_max, K]       -- step-0 scores
    base_pos: torch.Tensor          # [B_max] int64    -- d_committed_len/seq
    r_per_seq: torch.Tensor         # [B_max] int64    -- d_committed_len % bsz
    n_full_prefix_t: torch.Tensor   # [B_max] int64    -- d_committed_len // bsz
    bt_branch: torch.Tensor         # [B_max*K, max_blocks_branch] int32

    # Constant index buffers used by the captured kernels.
    arange_bk: torch.Tensor         # [B_max*K] int64  -- arange(B_max*K)
    arange_offset_K: torch.Tensor   # [B_max*K] int64  -- arange(0, B*K, K)
                                    #                     repeated K times
    parent_const_step0: torch.Tensor  # [K+1] int64    -- arange(-1, K)

    # --- per-step outputs (one slot per loop iteration f in 0..S-2) ---
    score_lists: List[torch.Tensor] = field(default_factory=list)
    token_lists: List[torch.Tensor] = field(default_factory=list)
    parents_lists: List[torch.Tensor] = field(default_factory=list)
    # Final running scores (after the loop)
    scores_out: Optional[torch.Tensor] = None


class DraftChainGraphRunner:
    """Captures the entire S-1 step draft decode loop as one graph per B.

    Mirrors sglang's ``EAGLEDraftCudaGraphRunner``: per replay we copy in
    just the high-level Python-side state (initial topk, hidden, base_pos
    / r / n_full_prefix per seq, and the branch block table); the graph
    runs all ``S-1`` forward+topk+select_top_k iterations and writes the
    per-step scores / tokens / parents tensors that
    ``organize_draft_results`` consumes back to Python.
    """

    def __init__(
        self,
        engine: "LlamaEagle3Engine",
        cuda_graph_max_bs: int,
        scratch_block_d: int,
        max_blocks_branch: int,
        graph_pool=None,
    ):
        self.engine = engine
        self.B_max = cuda_graph_max_bs
        self.K = engine.topk
        self.S = engine.spec_steps
        self.scratch_block_d = scratch_block_d
        self.scratch_slot_d = scratch_block_d * engine.block_size
        self.max_blocks_branch = max_blocks_branch
        self.capture_bs: List[int] = list(range(1, self.B_max + 1))
        self.graph_pool = graph_pool
        self.bsz = engine.block_size

        device = engine.device
        dtype = engine.dtype
        H_d = engine.draft_config.hidden_size
        B_max = self.B_max
        K = self.K
        S = self.S

        bufs = _ChainBuffers(
            cur_top_i_t=torch.zeros(B_max * K, dtype=torch.int64, device=device),
            cur_hidden=torch.zeros(B_max * K, H_d, dtype=dtype, device=device),
            cur_top_p=torch.zeros(B_max, K, dtype=torch.float32, device=device),
            base_pos=torch.zeros(B_max, dtype=torch.int64, device=device),
            r_per_seq=torch.zeros(B_max, dtype=torch.int64, device=device),
            n_full_prefix_t=torch.zeros(B_max, dtype=torch.int64, device=device),
            bt_branch=torch.full(
                (B_max * K, max_blocks_branch), scratch_block_d,
                dtype=torch.int32, device=device,
            ),
            arange_bk=torch.arange(B_max * K, dtype=torch.int64, device=device),
            arange_offset_K=torch.arange(
                0, B_max * K, K, dtype=torch.int64, device=device,
            ).repeat_interleave(K),
            parent_const_step0=torch.arange(
                -1, K, dtype=torch.int64, device=device,
            ),
        )

        # Per-step output buffers (one entry per loop iteration f).
        bufs.score_lists = [
            torch.zeros(B_max, K, K, dtype=torch.float32, device=device)
            for _ in range(S - 1)
        ]
        bufs.token_lists = [
            torch.zeros(B_max, K * K, dtype=torch.int64, device=device)
            for _ in range(S - 1)
        ]
        bufs.parents_lists = [
            torch.zeros(B_max, K, dtype=torch.int64, device=device)
            for _ in range(S - 1)
        ]
        bufs.scores_out = torch.zeros(
            B_max, K, dtype=torch.float32, device=device,
        )
        self.bufs = bufs

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture_all(self) -> None:
        for B in sorted(self.capture_bs, reverse=True):
            self._capture_one(B)
        torch.cuda.synchronize()

    def _capture_one(self, B: int) -> None:
        bufs = self.bufs
        K = self.K
        S = self.S
        bsz = self.bsz
        BK = B * K

        # Sliced views into persistent inputs.
        cur_top_i_t = bufs.cur_top_i_t[:BK]
        cur_hidden = bufs.cur_hidden[:BK]
        scores_in = bufs.cur_top_p[:B]
        base_pos = bufs.base_pos[:B]
        r_per_seq = bufs.r_per_seq[:B]
        n_full_prefix_t = bufs.n_full_prefix_t[:B]
        bt_branch = bufs.bt_branch[:BK]
        arange_bk = bufs.arange_bk[:BK]
        arange_offset_K = bufs.arange_offset_K[:BK]

        max_ctx_const = self.engine.max_model_len

        def _run_chain():
            input_ids_bk = cur_top_i_t
            hidden_bk = cur_hidden
            scores = scores_in
            base_pos_bk = base_pos.repeat_interleave(K)
            for f in range(S - 1):
                positions_step = base_pos_bk + f                       # [BK]
                offset_full = r_per_seq + f                            # [B]
                block_idx_within_branch = (offset_full // bsz)         # [B]
                offset_in_block = (offset_full % bsz)                  # [B]
                block_table_col = (
                    n_full_prefix_t + block_idx_within_branch
                ).repeat_interleave(K)                                 # [BK]
                block_ids_at_f = bt_branch[
                    arange_bk, block_table_col,
                ].to(torch.int64)                                      # [BK]
                slot_mapping_step = (
                    block_ids_at_f * bsz
                    + offset_in_block.repeat_interleave(K)
                ).to(torch.int32)                                      # [BK]
                cache_seqlens = (base_pos + (f + 1)).to(torch.int32)   # [B]
                cache_seqlens_bk = cache_seqlens.repeat_interleave(K)  # [BK]

                with set_forward_context(
                    is_prefill=False,
                    slot_mapping=slot_mapping_step,
                    context_lens=cache_seqlens_bk,
                    block_tables=bt_branch,
                    max_context_len=max_ctx_const,
                    is_cuda_graph_replay=True,
                ):
                    h2l, h2a = self.engine.draft.forward_draft(
                        input_ids_bk, positions_step.to(torch.long),
                        hidden_bk,
                    )
                    logits = self.engine.draft.compute_logits(h2l)

                log_probs = torch.log_softmax(logits, dim=-1)          # [BK, V_d]
                topk_p, topk_index = torch.topk(log_probs, K, dim=-1)  # [BK, K]
                topk_index_t = self.engine.draft.remap_draft_ids(topk_index)

                expand_scores = (
                    scores.unsqueeze(2) + topk_p.reshape(B, K, K)
                )                                                      # [B, K, K]
                topk_cs_p, topk_cs_index = torch.topk(
                    expand_scores.flatten(start_dim=1), K, dim=-1,
                )                                                      # [B, K]
                scores = topk_cs_p

                topk_index_t_flat = topk_index_t.reshape(B, K * K)
                new_input_ids = torch.gather(
                    topk_index_t_flat, dim=1, index=topk_cs_index,
                ).reshape(BK)
                sel_idx = topk_cs_index.flatten() // K + arange_offset_K
                new_hidden = h2a[sel_idx]

                # Write per-step outputs into persistent buffers.
                bufs.score_lists[f][:B].copy_(expand_scores)
                bufs.token_lists[f][:B].copy_(topk_index_t_flat)
                bufs.parents_lists[f][:B].copy_(
                    topk_cs_index + (K * K * f + K),
                )

                input_ids_bk = new_input_ids
                hidden_bk = new_hidden

            bufs.scores_out[:B].copy_(scores)

        # Warmup
        _run_chain()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            _run_chain()

        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[B] = graph

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def replay(
        self,
        cur_top_i_t_real: torch.Tensor,    # [B*K] int64
        cur_hidden_real: torch.Tensor,     # [B*K, H_d]
        cur_top_p_real: torch.Tensor,      # [B, K]
        base_pos_real: torch.Tensor,       # [B] int64
        r_per_seq_real: torch.Tensor,      # [B] int64
        n_full_prefix_real: torch.Tensor,  # [B] int64
        bt_branch_real: torch.Tensor,      # [B*K, w] int32
        raw_bs: int,
    ):
        """Pad real inputs to the smallest captured bucket and replay.

        Returns ``(score_list_steps, token_list_steps, parents_list_steps,
        cur_top_p_view)`` -- per-loop-step persistent views (sliced to
        ``raw_bs``).  The engine wraps these together with the step-0
        entries (derived from the input buffers) into the full
        ``score_list / token_list / parents_list`` lists for
        ``organize_draft_results``.
        """
        K = self.K
        B_pad = self.capture_bs[bisect.bisect_left(self.capture_bs, raw_bs)]
        BK_real = raw_bs * K
        BK_pad = B_pad * K

        bufs = self.bufs
        scratch_block = self.scratch_block_d
        scratch_slot = self.scratch_slot_d

        # --- ghost-row reset --------------------------------------------
        if BK_pad > BK_real:
            bufs.cur_top_i_t[BK_real:BK_pad].zero_()
            bufs.cur_hidden[BK_real:BK_pad].zero_()
            # bt_branch: ensure ghost rows point at scratch slot to keep FA
            # decode safe (kernel will read 1 valid token from scratch).
            bufs.bt_branch[BK_real:BK_pad].fill_(scratch_block)
        if B_pad > raw_bs:
            bufs.cur_top_p[raw_bs:B_pad].zero_()
            # base_pos / r / n_full_prefix for ghost seqs: pick a small
            # consistent value so the computed slot_mapping/cache_seqlens
            # land in the scratch block.  base_pos = scratch_slot,
            # r_per_seq = 0, n_full_prefix_t = scratch_block makes the
            # block_table_col index a valid (always-scratch) row.
            bufs.base_pos[raw_bs:B_pad].fill_(0)
            bufs.r_per_seq[raw_bs:B_pad].fill_(0)
            bufs.n_full_prefix_t[raw_bs:B_pad].fill_(0)

        # --- copy real data ---------------------------------------------
        bufs.cur_top_i_t[:BK_real].copy_(cur_top_i_t_real, non_blocking=True)
        bufs.cur_hidden[:BK_real].copy_(cur_hidden_real, non_blocking=True)
        bufs.cur_top_p[:raw_bs].copy_(cur_top_p_real, non_blocking=True)
        bufs.base_pos[:raw_bs].copy_(base_pos_real, non_blocking=True)
        bufs.r_per_seq[:raw_bs].copy_(r_per_seq_real, non_blocking=True)
        bufs.n_full_prefix_t[:raw_bs].copy_(
            n_full_prefix_real, non_blocking=True,
        )

        w_real = bt_branch_real.shape[1]
        if w_real >= self.max_blocks_branch:
            bufs.bt_branch[:BK_real].copy_(
                bt_branch_real[:, :self.max_blocks_branch],
                non_blocking=True,
            )
        else:
            bufs.bt_branch[:BK_real, :w_real].copy_(
                bt_branch_real, non_blocking=True,
            )
            bufs.bt_branch[:BK_real, w_real:].fill_(scratch_block)

        # --- replay ------------------------------------------------------
        self.graphs[B_pad].replay()

        # Sliced views (callers should clone if they need to retain them
        # across subsequent replays of the same runner).
        score_list_steps = [s[:raw_bs] for s in bufs.score_lists]
        token_list_steps = [t[:raw_bs] for t in bufs.token_lists]
        parents_list_steps = [p[:raw_bs] for p in bufs.parents_lists]
        cur_top_p_view = bufs.cur_top_p[:raw_bs]

        return (
            score_list_steps,
            token_list_steps,
            parents_list_steps,
            cur_top_p_view,
        )


# ---------------------------------------------------------------------------
# Draft extend runner (Phase C)
# ---------------------------------------------------------------------------

@dataclass
class _ExtendBuffers:
    """Persistent input + output buffers for ``DraftExtendGraphRunner``.

    Captured input shape is fixed at ``B_max * (S+1)`` tokens.  Per-seq
    accepted length ``L_i in [1, S+1]`` is encoded as:
      * ``cu_seqlens_q`` = ``arange(0, B_max*(S+1)+1, S+1)`` (constant).
      * ``cu_seqlens_k`` = cumulative ``base_pos_i + L_i`` (variable).
      * ``slot_mapping[ghost tokens]`` = scratch slot.
      * ``accept_length`` (variable) tells the captured graph where to
        gather the LAST valid token's logits per seq.
    """

    # Inputs
    input_ids: torch.Tensor            # [B_max*(S+1)] int64
    positions: torch.Tensor            # [B_max*(S+1)] int64
    hidden: torch.Tensor               # [B_max*(S+1), 3*H_t]
    slot_mapping: torch.Tensor         # [B_max*(S+1)] int32
    block_table: torch.Tensor          # [B_max, max_blocks_d] int32
    cu_seqlens_q: torch.Tensor         # [B_max+1] int32, fixed arange
    cu_seqlens_k: torch.Tensor         # [B_max+1] int32
    accept_length: torch.Tensor        # [B_max] int32
    seq_arange_offsets: torch.Tensor   # [B_max] int64, fixed arange

    # Outputs
    top_p: torch.Tensor                # [B_max, K] float
    top_i: torch.Tensor                # [B_max, K] int64 (draft vocab)
    top_i_t: torch.Tensor              # [B_max, K] int64 (target vocab)
    last_aux_h: torch.Tensor           # [B_max, H_d]


class DraftExtendGraphRunner:
    """Captures the post-verify ``draft_extend + topk`` forward.

    Mirrors sglang's ``EAGLEDraftExtendCudaGraphRunner``: input is padded
    to ``B*(S+1)`` tokens; ghost tokens write to a scratch slot and are
    discarded.  ``cu_seqlens_k`` carries the per-seq actual key length
    (``base_pos + accept_length``) so the prefill kernel only attends to
    valid K/V.
    """

    def __init__(
        self,
        engine: "LlamaEagle3Engine",
        cuda_graph_max_bs: int,
        scratch_block_d: int,
        graph_pool=None,
    ):
        self.engine = engine
        self.B_max = cuda_graph_max_bs
        self.K = engine.topk
        self.S = engine.spec_steps
        self.scratch_block_d = scratch_block_d
        self.scratch_slot_d = scratch_block_d * engine.block_size
        self.capture_bs: List[int] = list(range(1, self.B_max + 1))
        self.graph_pool = graph_pool

        device = engine.device
        dtype = engine.dtype
        block_size = engine.block_size
        H_t = engine.target_config.hidden_size
        H_d = engine.draft_config.hidden_size
        max_blocks = (engine.max_model_len + block_size - 1) // block_size
        B_max = self.B_max
        K = self.K
        S = self.S
        SP1 = S + 1

        bufs = _ExtendBuffers(
            input_ids=torch.zeros(
                B_max * SP1, dtype=torch.int64, device=device,
            ),
            positions=torch.zeros(
                B_max * SP1, dtype=torch.int64, device=device,
            ),
            hidden=torch.zeros(
                B_max * SP1, 3 * H_t, dtype=dtype, device=device,
            ),
            slot_mapping=torch.full(
                (B_max * SP1,), self.scratch_slot_d,
                dtype=torch.int32, device=device,
            ),
            block_table=torch.full(
                (B_max, max_blocks), scratch_block_d,
                dtype=torch.int32, device=device,
            ),
            cu_seqlens_q=torch.arange(
                0, (B_max + 1) * SP1, SP1, dtype=torch.int32, device=device,
            ),
            cu_seqlens_k=torch.arange(
                B_max + 1, dtype=torch.int32, device=device,
            ),  # default values get overwritten on every replay
            accept_length=torch.ones(
                B_max, dtype=torch.int32, device=device,
            ),
            seq_arange_offsets=torch.arange(
                0, B_max * SP1, SP1, dtype=torch.int64, device=device,
            ),
            top_p=torch.zeros(B_max, K, dtype=torch.float32, device=device),
            top_i=torch.zeros(B_max, K, dtype=torch.int64, device=device),
            top_i_t=torch.zeros(B_max, K, dtype=torch.int64, device=device),
            last_aux_h=torch.zeros(B_max, H_d, dtype=dtype, device=device),
        )
        self.bufs = bufs
        self.max_blocks = max_blocks
        self.SP1 = SP1

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture_all(self) -> None:
        for B in sorted(self.capture_bs, reverse=True):
            self._capture_one(B)
        torch.cuda.synchronize()

    def _capture_one(self, B: int) -> None:
        bufs = self.bufs
        K = self.K
        SP1 = self.SP1
        BS = B * SP1

        # Sliced views.
        input_ids = bufs.input_ids[:BS]
        positions = bufs.positions[:BS]
        hidden = bufs.hidden[:BS]
        slot_mapping = bufs.slot_mapping[:BS]
        block_table = bufs.block_table[:B]
        cu_q = bufs.cu_seqlens_q[:B + 1]
        cu_k = bufs.cu_seqlens_k[:B + 1]
        accept_length = bufs.accept_length[:B]
        seq_arange_offsets = bufs.seq_arange_offsets[:B]

        max_sk = self.engine.max_model_len

        def _run_extend():
            with set_forward_context(
                is_prefill=True,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=SP1,
                max_seqlen_k=max_sk,
                slot_mapping=slot_mapping,
                block_tables=block_table,
                is_cuda_graph_replay=True,
            ):
                h2l, h2a = self.engine.draft.forward_draft(
                    input_ids, positions, hidden,
                )

            # Gather logits/aux at the last VALID token per seq.
            last_idx = seq_arange_offsets + accept_length.to(torch.int64) - 1
            h2l_last = h2l[last_idx]                 # [B, H_d]
            last_aux_h = h2a[last_idx]               # [B, H_d]

            # Manual single-token-per-seq logits projection (bypasses the
            # cu_seqlens_q-driven gather inside ParallelLMHead.project).
            lm_head = self.engine.draft.lm_head
            logits = lm_head.linear_op(
                h2l_last, lm_head.embedding_op.emb.weight,
            ).float()                                # [B, draft_vocab]

            log_probs = torch.log_softmax(logits, dim=-1)
            top_p, top_i = torch.topk(log_probs, K, dim=-1)
            top_i_t = self.engine.draft.remap_draft_ids(top_i)

            bufs.top_p[:B].copy_(top_p)
            bufs.top_i[:B].copy_(top_i)
            bufs.top_i_t[:B].copy_(top_i_t)
            bufs.last_aux_h[:B].copy_(last_aux_h)

        # Warmup
        _run_extend()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            _run_extend()

        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[B] = graph

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def replay(
        self,
        per_seq_input_ids: List[torch.Tensor],   # B tensors, [L_i] each
        per_seq_positions: List[torch.Tensor],   # B tensors, [L_i] each
        per_seq_hidden: List[torch.Tensor],      # B tensors, [L_i, 3*H_t]
        per_seq_slot_mapping: List[torch.Tensor],  # B tensors, [L_i] each
        block_table_real: torch.Tensor,          # [B, k_real] int32
        cu_seqlens_k_padded: torch.Tensor,       # [B_pad+1] int32 (precomputed
                                                 # by caller incl. ghost +1's)
        accept_length_padded: torch.Tensor,      # [B_pad] int32 (incl. ghost 1's)
        raw_bs: int,
    ):
        """Pad real inputs to bucket B' and replay.

        Caller is expected to have built ``cu_seqlens_k_padded`` and
        ``accept_length_padded`` already padded out to ``B_pad`` so we
        avoid any host-device syncs in this hot path.

        Returns sliced views of ``(top_p, top_i, top_i_t, last_aux_h)`` of
        shape ``[raw_bs, ...]``.  Caller should clone if they need to
        retain across subsequent replays.
        """
        SP1 = self.SP1
        B_pad = self.capture_bs[bisect.bisect_left(self.capture_bs, raw_bs)]
        BS_pad = B_pad * SP1

        bufs = self.bufs
        scratch_slot = self.scratch_slot_d
        scratch_block = self.scratch_block_d

        # --- reset token slots to ghost defaults ------------------------
        # Unused padded queries are computed but discarded. Their ids,
        # positions, and hidden states may safely retain previous values as
        # long as their KV writes go to the scratch slot.
        bufs.slot_mapping[:BS_pad].fill_(scratch_slot)

        # --- scatter real per-seq data into persistent slots ------------
        # Tiny Python loop (raw_bs <= cuda_graph_max_bs <= 8); each
        # ``copy_`` is a small device-to-device async kernel launch, no
        # host sync.
        for i in range(raw_bs):
            L = per_seq_input_ids[i].shape[0]
            dst_start = i * SP1
            bufs.input_ids[dst_start:dst_start + L].copy_(
                per_seq_input_ids[i], non_blocking=True,
            )
            bufs.positions[dst_start:dst_start + L].copy_(
                per_seq_positions[i], non_blocking=True,
            )
            bufs.hidden[dst_start:dst_start + L].copy_(
                per_seq_hidden[i], non_blocking=True,
            )
            bufs.slot_mapping[dst_start:dst_start + L].copy_(
                per_seq_slot_mapping[i], non_blocking=True,
            )

        # --- cu_seqlens_k + accept_length (already padded by caller) ----
        bufs.cu_seqlens_k[:B_pad + 1].copy_(
            cu_seqlens_k_padded, non_blocking=True,
        )
        bufs.accept_length[:B_pad].copy_(
            accept_length_padded, non_blocking=True,
        )

        # --- block table ------------------------------------------------
        k_real = block_table_real.shape[1]
        if k_real >= self.max_blocks:
            bufs.block_table[:raw_bs].copy_(
                block_table_real[:, :self.max_blocks], non_blocking=True,
            )
        else:
            bufs.block_table[:raw_bs, :k_real].copy_(
                block_table_real, non_blocking=True,
            )
            bufs.block_table[:raw_bs, k_real:].fill_(scratch_block)
        if B_pad > raw_bs:
            bufs.block_table[raw_bs:B_pad].fill_(scratch_block)

        # --- replay ------------------------------------------------------
        self.graphs[B_pad].replay()

        return (
            bufs.top_p[:raw_bs],
            bufs.top_i[:raw_bs],
            bufs.top_i_t[:raw_bs],
            bufs.last_aux_h[:raw_bs],
        )
