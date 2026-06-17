"""EAGLE-3 speculative-decoding engine for kb_nano (eager, tree draft).

Implements EAGLE-3 with K-branch tree drafting (matches sglang's
``select_top_k_tokens`` + ``build_tree_kernel_efficient`` + ``verify_tree_greedy``
pipeline). The default config mirrors sglang's reference defaults:
``spec_steps=3``, ``topk=4``, ``num_draft_tokens=16``.

For each speculative step:

  1. ``_draft_chain`` runs S-1 forward passes on the draft model, producing K
     candidates per seq per step. Each step expands K * K candidates and selects
     the top K by cumulative log-probability. All branches share the prefix
     KV (set up by the most recent ``_draft_extend``); each branch additionally
     writes to a per-branch tail block in the draft paged KV cache.
  2. ``build_tree_kernel_efficient`` packs the K * (S-1) + K candidates into a
     flat tree of ``num_draft_tokens`` verify positions with the right
     parent / sibling / position table.
  3. ``_target_verify`` runs the target on the flat tree and ``verify_tree_greedy``
     selects the longest accepted path.
  4. ``_remap_target_kv_after_verify`` re-orders target KV slots so logical
     position ``t_committed_len + k`` holds the K/V for the k-th accepted
     token (vs. the k-th tree node in flat write order). This is a no-op
     for chain drafting but required for correctness with tree drafting.
  5. ``_draft_extend`` digests the accepted tokens (with target aux hidden
     states) into the draft KV cache, capturing the next round's K topk.

Scope:
  - single GPU, TP=1
  - eager forwards (no CUDA graphs) on both target and draft
  - greedy verify only (T=0)
  - one batch in flight at a time
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from .context import get_attn_backend_config, set_context, set_forward_context
from .weight_loader import load_eagle3_draft_model, load_model
from ..tasks.baseline.L1.eagle_tree_ops import (
    build_tree_kernel_efficient_with_metadata,
    organize_draft_results,
    verify_tree_greedy,
)


@dataclass
class Eagle3SamplingParams:
    max_tokens: int = 512
    ignore_eos: bool = False


@dataclass
class Eagle3Output:
    prompt_token_ids: List[int]
    token_ids: List[int]
    generated_text: str = ""


class _PagedKVCache:
    """Lightweight paged KV cache + free list for either target or draft."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        layout: str = "NHD",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.layout = layout

        if layout == "HND":
            shape = (2, num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        else:
            shape = (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.kv = torch.empty(*shape, dtype=dtype, device=device)
        self._free = list(range(num_blocks))

    def reset(self):
        self._free = list(range(self.num_blocks))

    def num_free_blocks(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> List[int]:
        assert n <= len(self._free), f"OOM: need {n} blocks, have {len(self._free)}"
        out = self._free[:n]
        self._free = self._free[n:]
        return out

    def free(self, blocks: List[int]):
        self._free.extend(blocks)

    def attach(self, attn_layers):
        assert len(attn_layers) == self.num_layers, (
            f"layer count mismatch: cache has {self.num_layers} layers but "
            f"model has {len(attn_layers)} attn layers"
        )
        for i, mod in enumerate(attn_layers):
            mod.k_cache = self.kv[0, i]
            mod.v_cache = self.kv[1, i]


class _Eagle3Sequence:
    _next_id = 0

    def __init__(self, prompt_ids: List[int], max_tokens: int, ignore_eos: bool):
        _Eagle3Sequence._next_id += 1
        self.seq_id = _Eagle3Sequence._next_id
        self.prompt_ids = list(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.generated_ids: List[int] = []
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

        self.t_blocks: List[int] = []
        self.d_blocks: List[int] = []
        # Branch-specific tail blocks for the current draft chain (allocated
        # at the start of each ``_draft_chain`` and freed before the next
        # draft extend). Length K per seq.
        self._chain_tail_blocks: List[int] = []

        # The last accepted token id.
        self.last_token: Optional[int] = None
        # Number of tokens whose KV is committed in the target paged cache.
        self.t_committed_len: int = 0
        # Number of positions populated in the draft KV cache.
        # In sglang convention this stays equal to ``t_committed_len`` after
        # each draft-extend (once the draft has digested the latest accepted
        # tokens). Between the chain forwards and the post-verify extend it
        # may transiently exceed t_committed_len; we always free those extra
        # slots before the next extend.
        self.d_committed_len: int = 0

        # Output of the latest draft-extend: the top-K logprobs / token ids
        # (in draft vocab) for the FIRST speculative position, and the draft
        # hidden state at the last extend position. These feed chain step 0.
        self.draft_topk_p: Optional[torch.Tensor] = None       # [K] (float)
        self.draft_topk_i: Optional[torch.Tensor] = None       # [K] (long, draft vocab)
        self.draft_topk_i_t: Optional[torch.Tensor] = None     # [K] (long, target vocab)
        self.draft_hidden: Optional[torch.Tensor] = None       # [H_d]

        self.finished = False


def _gather_attn_layers(model) -> list:
    out = []
    for m in model.modules():
        if hasattr(m, "k_cache") and hasattr(m, "v_cache") and hasattr(m, "scale"):
            out.append(m)
    return out


class LlamaEagle3Engine:
    """Single-process EAGLE-3 engine with chain drafting (eager, TP=1)."""

    def __init__(
        self,
        model_name: str,
        draft_repo: str,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_num_seqs: int = 32,
        spec_steps: int = 3,
        spec_topk: int = 4,
        num_draft_tokens: Optional[int] = None,
        enforce_eager: bool = False,
        cuda_graph_max_bs: int = 8,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda")
        self.dtype = dtype
        self.spec_steps = spec_steps
        self.topk = spec_topk
        # sglang default: 1 (root) + topk + (spec_steps - 1) * topk * topk
        # candidates produced; the build_tree kernel keeps the top
        # num_draft_tokens - 1 by score and prepends the root.
        if num_draft_tokens is None:
            num_draft_tokens = 1 + spec_topk + (spec_steps - 1) * spec_topk * spec_topk
        self.num_draft_tokens = num_draft_tokens
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.cuda_graph_max_bs = cuda_graph_max_bs

        # EAGLE-3 is currently TP=1. Avoid initializing NCCL for world_size=1:
        # some CUDA/NCCL combinations fail eager_connect_single_device even
        # though the model itself runs correctly on the selected GPU.
        torch.cuda.set_device(0)

        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        attn_cfg = get_attn_backend_config()
        self.block_size = attn_cfg.block_size
        self.kv_layout = attn_cfg.kv_layout

        print(f"[EAGLE-3] Loading target model: {model_name}")
        self.target, self.target_config = load_model(
            model_name, device=self.device, dtype=dtype,
        )
        self.target.set_eagle3_layers_to_capture(None)

        print(f"[EAGLE-3] Loading draft model: {draft_repo}")
        self.draft, self.draft_config = load_eagle3_draft_model(
            draft_repo, self.target_config, device=self.device, dtype=dtype,
        )
        self.draft.set_embed_tokens(self.target.model.embed_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.target_attn_layers = _gather_attn_layers(self.target)
        self.draft_attn_layers = _gather_attn_layers(self.draft)

        self._allocate_kv_caches(gpu_memory_utilization)

        # Reserve block 0 of each KV cache as the "scratch" page that ghost
        # rows of CUDA-graph padding write to / read from. This block must
        # never be allocated to a real sequence, so we pop it out of the
        # free list at engine init and never return it.
        self._scratch_block_t = self.target_kv.alloc(1)[0]
        self._scratch_block_d = self.draft_kv.alloc(1)[0]

        # CUDA graph runners (built lazily on first generate() call so we
        # don't pay capture cost when the engine is only used for prefill
        # debugging / tests). Captures depend on max_running_requests which
        # is bounded by max_num_seqs.
        self._target_verify_runner = None
        self._draft_chain_runner = None
        self._draft_extend_runner = None
        self._graph_pool = None
        # Width of the captured ``bt_branch`` block table = enough to cover
        # the full draft prefix (at max_model_len) plus the branch tail
        # (at most 2 blocks for typical S-1=2 step chains).
        self._chain_max_blocks_branch = (
            (self.max_model_len + self.block_size - 1) // self.block_size + 2
        )

        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def _allocate_kv_caches(self, gpu_mem_util: float):
        free, total = torch.cuda.mem_get_info()
        used = total - free

        elem_size = torch.finfo(self.dtype).bits // 8
        n_t_layers = len(self.target_attn_layers)
        n_d_layers = len(self.draft_attn_layers)
        n_kv_t = self.target_config.num_key_value_heads
        n_kv_d = self.draft_config.num_key_value_heads
        head_dim_t = self.target_config.head_dim
        head_dim_d = self.draft_config.head_dim

        block_bytes_t = (
            2 * n_t_layers * self.block_size * n_kv_t * head_dim_t * elem_size
        )
        block_bytes_d = (
            2 * n_d_layers * self.block_size * n_kv_d * head_dim_d * elem_size
        )

        available = max(0, int(total * gpu_mem_util) - used)
        # Reserve some headroom for activations / draft tree buffers (~512 MiB).
        available = max(0, available - (512 << 20))

        target_budget = int(available * 0.97)
        draft_budget = available - target_budget

        n_blocks_t = max(64, target_budget // block_bytes_t)
        n_blocks_d = max(64, draft_budget // block_bytes_d)

        max_blocks_per_seq = (
            self.max_model_len + self.block_size - 1
        ) // self.block_size
        n_blocks_t = min(n_blocks_t, self.max_num_seqs * max_blocks_per_seq + 64)
        n_blocks_d = min(n_blocks_d, self.max_num_seqs * max_blocks_per_seq + 64)

        print(f"[EAGLE-3] Target KV cache: {n_blocks_t} blocks "
              f"x {self.block_size} = {n_blocks_t * self.block_size} slots")
        print(f"[EAGLE-3] Draft  KV cache: {n_blocks_d} blocks "
              f"x {self.block_size} = {n_blocks_d * self.block_size} slots")

        self.target_kv = _PagedKVCache(
            n_t_layers, n_kv_t, head_dim_t, n_blocks_t, self.block_size,
            layout=self.kv_layout, dtype=self.dtype, device=self.device,
        )
        self.draft_kv = _PagedKVCache(
            n_d_layers, n_kv_d, head_dim_d, n_blocks_d, self.block_size,
            layout=self.kv_layout, dtype=self.dtype, device=self.device,
        )
        self.target_kv.attach(self.target_attn_layers)
        self.draft_kv.attach(self.draft_attn_layers)

    # ------------------------------------------------------------------
    # CUDA graph runner construction (lazy)
    # ------------------------------------------------------------------
    def _maybe_build_graph_runners(self):
        """Build and capture CUDA graph runners on first use.

        Skips entirely if ``enforce_eager`` is set. Capture is one-shot;
        subsequent generate() calls reuse the captured graphs.
        """
        if self.enforce_eager:
            return
        if self._target_verify_runner is not None:
            return

        from ..tasks.baseline.L1.tree_attn_prefill import _FA3_AVAILABLE

        if not _FA3_AVAILABLE:
            print(
                "[EAGLE-3] Skipping CUDA graph capture: FA3 paged tree "
                "attention is unavailable in this environment."
            )
            return

        from .eagle3_cuda_graph import (
            TargetVerifyGraphRunner,
            DraftChainGraphRunner,
            DraftExtendGraphRunner,
        )

        # Cap at max_num_seqs -- there's no point capturing larger buckets
        # than the engine will ever batch.
        max_bs = min(self.cuda_graph_max_bs, self.max_num_seqs)
        if max_bs < 1:
            return

        print(
            f"[EAGLE-3] Capturing target-verify CUDA graphs for "
            f"B in 1..{max_bs} (N={self.num_draft_tokens})..."
        )
        runner = TargetVerifyGraphRunner(
            engine=self,
            cuda_graph_max_bs=max_bs,
            scratch_block_t=self._scratch_block_t,
            graph_pool=self._graph_pool,
        )
        runner.capture_all()
        if self._graph_pool is None:
            self._graph_pool = runner.graph_pool
        self._target_verify_runner = runner
        print(
            f"[EAGLE-3] Captured {len(runner.graphs)} target-verify graphs."
        )

        if self.spec_steps > 1:
            print(
                f"[EAGLE-3] Capturing draft-chain CUDA graphs for "
                f"B in 1..{max_bs} (K={self.topk}, S-1={self.spec_steps-1})..."
            )
            chain_runner = DraftChainGraphRunner(
                engine=self,
                cuda_graph_max_bs=max_bs,
                scratch_block_d=self._scratch_block_d,
                max_blocks_branch=self._chain_max_blocks_branch,
                graph_pool=self._graph_pool,
            )
            chain_runner.capture_all()
            if self._graph_pool is None:
                self._graph_pool = chain_runner.graph_pool
            self._draft_chain_runner = chain_runner
            print(
                f"[EAGLE-3] Captured {len(chain_runner.graphs)} "
                f"draft-chain graphs."
            )

        print(
            f"[EAGLE-3] Capturing draft-extend CUDA graphs for "
            f"B in 1..{max_bs} (S+1={self.spec_steps + 1})..."
        )
        extend_runner = DraftExtendGraphRunner(
            engine=self,
            cuda_graph_max_bs=max_bs,
            scratch_block_d=self._scratch_block_d,
            graph_pool=self._graph_pool,
        )
        extend_runner.capture_all()
        if self._graph_pool is None:
            self._graph_pool = extend_runner.graph_pool
        self._draft_extend_runner = extend_runner
        print(
            f"[EAGLE-3] Captured {len(extend_runner.graphs)} "
            f"draft-extend graphs."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_target_capacity(self, seq: _Eagle3Sequence, target_len: int):
        needed = (target_len + self.block_size - 1) // self.block_size
        cur = len(seq.t_blocks)
        if needed > cur:
            seq.t_blocks.extend(self.target_kv.alloc(needed - cur))

    def _ensure_draft_capacity(self, seq: _Eagle3Sequence, target_len: int):
        needed = (target_len + self.block_size - 1) // self.block_size
        cur = len(seq.d_blocks)
        if needed > cur:
            seq.d_blocks.extend(self.draft_kv.alloc(needed - cur))

    def _free_draft_tail(self, seq: _Eagle3Sequence, keep_len: int):
        """Free draft KV blocks beyond ``keep_len`` slots."""
        keep_blocks = (keep_len + self.block_size - 1) // self.block_size
        if keep_blocks < len(seq.d_blocks):
            released = seq.d_blocks[keep_blocks:]
            self.draft_kv.free(released)
            seq.d_blocks = seq.d_blocks[:keep_blocks]

    def _slot_mapping_for_range(
        self, blocks: List[int], start: int, end: int,
    ) -> List[int]:
        out = []
        for p in range(start, end):
            blk = blocks[p // self.block_size]
            out.append(blk * self.block_size + (p % self.block_size))
        return out

    def _slot_for(self, blocks: List[int], pos: int) -> int:
        return blocks[pos // self.block_size] * self.block_size + (pos % self.block_size)

    @torch.inference_mode()
    def _remap_target_kv_after_verify(
        self,
        seqs: List[_Eagle3Sequence],
        accept_index_cpu: List[List[int]],
        accept_num_cpu: List[int],
    ):
        """After tree verify, K/V for tree node ``i`` was written at the slot
        for logical position ``t_committed_len + i`` (linear write order).
        For the next iteration we need K/V for the k-th accepted token at the
        slot for logical position ``t_committed_len + k``. With chain drafting
        accepted tree indices are already ``[0, 1, 2, ...]`` so this is a no-op,
        but with tree drafting the accepted chain typically picks non-contiguous
        tree nodes and we MUST remap, otherwise subsequent attention reads
        garbage K/V from rejected branches.

        Mirrors sglang's ``move_kv_cache(tgt_cache_loc, src_cache_loc)``.
        """
        N = self.num_draft_tokens
        src_slots: list[int] = []
        dst_slots: list[int] = []
        for i, seq in enumerate(seqs):
            n_accept = int(accept_num_cpu[i])
            base = seq.t_committed_len
            for k in range(n_accept + 1):
                flat = int(accept_index_cpu[i][k])
                tree_idx = flat - i * N
                if tree_idx == k:
                    continue
                src_slots.append(self._slot_for(seq.t_blocks, base + tree_idx))
                dst_slots.append(self._slot_for(seq.t_blocks, base + k))

        if not src_slots:
            return

        src_t = torch.tensor(src_slots, device=self.device, dtype=torch.long)
        dst_t = torch.tensor(dst_slots, device=self.device, dtype=torch.long)
        kv = self.target_kv.kv
        BS = self.block_size
        if self.kv_layout == "NHD":
            # kv: [2, L, NB, BS, H, D] -> view as [2, L, NB*BS, H, D].
            flat = kv.view(
                2, self.target_kv.num_layers,
                self.target_kv.num_blocks * BS,
                self.target_kv.num_kv_heads, self.target_kv.head_dim,
            )
            src_vals = flat[:, :, src_t].clone()
            flat[:, :, dst_t] = src_vals
        else:
            # HND: kv [2, L, NB, H, BS, D]. NB and BS are non-adjacent so
            # we cannot trivially view as [..., NB*BS, ...]. Instead index
            # block / offset directly per slot.
            src_blk = (src_t // BS).tolist()
            src_off = (src_t % BS).tolist()
            dst_blk = (dst_t // BS).tolist()
            dst_off = (dst_t % BS).tolist()
            for n in range(len(src_blk)):
                kv[:, :, dst_blk[n], :, dst_off[n], :] = \
                    kv[:, :, src_blk[n], :, src_off[n], :].clone()

    # ------------------------------------------------------------------
    # Target prefill on the prompt
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _target_prefill(self, seqs: List[_Eagle3Sequence]):
        """Run target prefill on the prompts.

        Returns ``aux_per_seq``: list of per-seq aux-cat tensors of shape
        ``[P_i, 3*H_t]`` (concat of the three captured target hidden states).
        These feed the initial draft-extend.
        """
        input_ids: list[int] = []
        positions: list[int] = []
        cu_q = [0]
        cu_k = [0]
        slot_mapping: list[int] = []
        max_blocks = 0

        for seq in seqs:
            sl = len(seq.prompt_ids)
            self._ensure_target_capacity(seq, sl)
            input_ids.extend(seq.prompt_ids)
            positions.extend(range(sl))
            cu_q.append(cu_q[-1] + sl)
            cu_k.append(cu_k[-1] + sl)
            slot_mapping.extend(self._slot_mapping_for_range(seq.t_blocks, 0, sl))
            max_blocks = max(max_blocks, len(seq.t_blocks))
            seq.t_committed_len = sl

        n = len(seqs)
        bt = np.full((n, max_blocks), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            bt[i, :len(seq.t_blocks)] = seq.t_blocks

        max_sq = max(len(s.prompt_ids) for s in seqs)
        max_sk = max_sq

        set_context(
            True,
            torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            block_tables=torch.from_numpy(bt).to(self.device),
        )
        ids_t = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        pos_t = torch.tensor(positions, dtype=torch.int64, device=self.device)

        out = self.target.model(ids_t, pos_t)
        if isinstance(out, tuple):
            hidden_states, aux_list = out
        else:
            hidden_states, aux_list = out, []

        # compute_logits internally slices to last-token-per-seq using context.
        logits = self.target.compute_logits(hidden_states)
        next_ids = logits.argmax(dim=-1).tolist()

        # Collect full aux per seq (for the initial draft extend).
        aux_per_seq: list[torch.Tensor] = []
        for i in range(n):
            s_idx, e_idx = cu_q[i], cu_q[i + 1]
            if len(aux_list) > 0:
                aux_per_seq.append(
                    torch.cat([a[s_idx:e_idx] for a in aux_list], dim=-1)
                )
            else:
                aux_per_seq.append(torch.zeros(
                    (e_idx - s_idx, self.target_config.hidden_size * 3),
                    device=self.device, dtype=self.dtype,
                ))

        for i, seq in enumerate(seqs):
            seq.last_token = int(next_ids[i])
            seq.token_ids.append(seq.last_token)
            seq.generated_ids.append(seq.last_token)

        return aux_per_seq

    # ------------------------------------------------------------------
    # Draft EXTEND: digest a stretch of tokens (with target aux) into draft KV
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _draft_extend(
        self,
        seqs: List[_Eagle3Sequence],
        ext_input_ids: list[torch.Tensor],   # per-seq, shape [L_i] target-vocab ids
        ext_hiddens: list[torch.Tensor],     # per-seq, shape [L_i, 3*H_t]
        ext_lens: list[int],                 # per-seq L_i
        post_verify: bool = False,
    ):
        """Run the draft model in extend mode and capture, per-seq, the
        first-step (topk_p, topk_i, topk_i_t, hidden) for the next chain.

        Each seq's draft KV is grown from ``d_committed_len`` to
        ``d_committed_len + L_i``. Updates ``d_committed_len`` in-place.

        ``post_verify=True`` indicates the call is the post-verify digest
        with bounded ``L_i in [1, S+1]``; this allows the captured
        ``DraftExtendGraphRunner`` fast-path.
        """
        bs = len(seqs)
        K = self.topk

        # ------------------------------------------------------------------
        # Fast path: captured CUDA graph for the post-verify draft extend.
        # ------------------------------------------------------------------
        runner = (
            self._draft_extend_runner
            if (post_verify and bs > 0) else None
        )
        if (
            runner is not None
            and bs <= runner.B_max
            and all(L <= runner.SP1 for L in ext_lens)
        ):
            self._draft_extend_via_graph(
                seqs, ext_input_ids, ext_hiddens, ext_lens, runner,
            )
            return

        positions: list[int] = []
        cu_q = [0]
        cu_k = [0]
        slot_mapping: list[int] = []
        max_blocks = 0
        last_idx_per_seq: list[int] = []

        for i, seq in enumerate(seqs):
            L = ext_lens[i]
            prefix = seq.d_committed_len
            new_total = prefix + L
            self._ensure_draft_capacity(seq, new_total)
            positions.extend(range(prefix, new_total))
            cu_q.append(cu_q[-1] + L)
            cu_k.append(cu_k[-1] + new_total)
            slot_mapping.extend(
                self._slot_mapping_for_range(seq.d_blocks, prefix, new_total)
            )
            max_blocks = max(max_blocks, len(seq.d_blocks))
            last_idx_per_seq.append(cu_q[-1] - 1)

        bt = np.full((bs, max_blocks), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            bt[i, :len(seq.d_blocks)] = seq.d_blocks

        max_sq = max(ext_lens)
        max_sk = max(seqs[i].d_committed_len + ext_lens[i] for i in range(bs))

        set_context(
            True,
            torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            block_tables=torch.from_numpy(bt).to(self.device),
        )

        ids_t = torch.cat(ext_input_ids).to(torch.int64).to(self.device)
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
        hidden_t = torch.cat(ext_hiddens, dim=0).to(self.device)

        if os.environ.get("KB_NANO_EAGLE3_DEBUG") == "1":
            print(f"  [_draft_extend] ids_t shape={ids_t.shape} dtype={ids_t.dtype} "
                  f"min={int(ids_t.min())} max={int(ids_t.max())}", flush=True)
            print(f"  [_draft_extend] pos_t shape={pos_t.shape} max={int(pos_t.max())}", flush=True)
            print(f"  [_draft_extend] hidden_t shape={hidden_t.shape} dtype={hidden_t.dtype}", flush=True)
            print(f"  [_draft_extend] last_idx_per_seq={last_idx_per_seq}", flush=True)

        hidden_to_logits, hidden_to_aux = self.draft.forward_draft(
            ids_t, pos_t, hidden_t,
        )

        if os.environ.get("KB_NANO_EAGLE3_DEBUG") == "1":
            print(f"  [_draft_extend] hidden_to_logits shape={hidden_to_logits.shape}", flush=True)

        # ``compute_logits`` (ParallelLMHead.project) slices to the last
        # token per seq using cu_seqlens_q when ctx.is_prefill=True, so we
        # pass the FULL hidden_to_logits tensor.
        logits = self.draft.compute_logits(hidden_to_logits)       # [bs, draft_vocab]

        last_idx_t = torch.tensor(
            last_idx_per_seq, device=self.device, dtype=torch.long,
        )
        last_aux_h = hidden_to_aux[last_idx_t]                     # [bs, H_d]

        log_probs = torch.log_softmax(logits, dim=-1)
        top_p, top_i = torch.topk(log_probs, K, dim=-1)            # [bs, K]
        top_i_target = self.draft.remap_draft_ids(top_i)           # [bs, K]

        for i, seq in enumerate(seqs):
            seq.d_committed_len += ext_lens[i]
            seq.draft_topk_p = top_p[i].clone()
            seq.draft_topk_i = top_i[i].clone()
            seq.draft_topk_i_t = top_i_target[i].clone()
            seq.draft_hidden = last_aux_h[i].clone()

    # ------------------------------------------------------------------
    # CUDA-graph fast path for the post-verify draft extend.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _draft_extend_via_graph(
        self,
        seqs: List[_Eagle3Sequence],
        ext_input_ids: list[torch.Tensor],
        ext_hiddens: list[torch.Tensor],
        ext_lens: list[int],
        runner,
    ):
        """Fast-path that delegates to ``DraftExtendGraphRunner.replay``.

        Builds the per-seq inputs / cu_seqlens / accept_length tensors
        Python-side, allocates draft KV, and invokes ``runner.replay``.
        Updates each ``seq``'s ``d_committed_len`` / ``draft_topk_*`` /
        ``draft_hidden`` from the persistent output views.
        """
        import bisect
        bs = len(seqs)
        K = self.topk
        B_pad = runner.capture_bs[bisect.bisect_left(runner.capture_bs, bs)]
        device = self.device

        per_seq_input_ids: list[torch.Tensor] = []
        per_seq_positions: list[torch.Tensor] = []
        per_seq_hidden: list[torch.Tensor] = []
        per_seq_slot_mapping: list[torch.Tensor] = []

        cu_k_padded = [0] * (B_pad + 1)
        accept_padded = [1] * B_pad
        max_blocks = 0

        for i, seq in enumerate(seqs):
            L = ext_lens[i]
            prefix = seq.d_committed_len
            new_total = prefix + L
            self._ensure_draft_capacity(seq, new_total)

            per_seq_input_ids.append(ext_input_ids[i].to(device).to(torch.int64))
            per_seq_positions.append(
                torch.arange(prefix, new_total, dtype=torch.long, device=device)
            )
            per_seq_hidden.append(ext_hiddens[i].to(device))

            sm = self._slot_mapping_for_range(seq.d_blocks, prefix, new_total)
            per_seq_slot_mapping.append(
                torch.tensor(sm, dtype=torch.int32, device=device)
            )

            cu_k_padded[i + 1] = cu_k_padded[i] + new_total
            accept_padded[i] = L
            max_blocks = max(max_blocks, len(seq.d_blocks))

        # Ghost seqs (i in [bs, B_pad)): cu_k delta = 1, accept_length = 1.
        for i in range(bs, B_pad):
            cu_k_padded[i + 1] = cu_k_padded[i] + 1

        cu_k_padded_t = torch.tensor(
            cu_k_padded, dtype=torch.int32, device=device,
        )
        accept_padded_t = torch.tensor(
            accept_padded, dtype=torch.int32, device=device,
        )

        bt = np.full((bs, max_blocks), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            bt[i, :len(seq.d_blocks)] = seq.d_blocks
        bt_t = torch.from_numpy(bt).to(device)

        top_p_view, top_i_view, top_i_t_view, last_aux_h_view = runner.replay(
            per_seq_input_ids=per_seq_input_ids,
            per_seq_positions=per_seq_positions,
            per_seq_hidden=per_seq_hidden,
            per_seq_slot_mapping=per_seq_slot_mapping,
            block_table_real=bt_t,
            cu_seqlens_k_padded=cu_k_padded_t,
            accept_length_padded=accept_padded_t,
            raw_bs=bs,
        )

        for i, seq in enumerate(seqs):
            seq.d_committed_len += ext_lens[i]
            # These are runner-owned persistent views. They are consumed by
            # the next draft-chain before the next draft-extend replay can
            # overwrite them, so keeping views avoids four small device clones
            # on every speculative step.
            seq.draft_topk_p = top_p_view[i]
            seq.draft_topk_i = top_i_view[i]
            seq.draft_topk_i_t = top_i_t_view[i]
            seq.draft_hidden = last_aux_h_view[i]

    # ------------------------------------------------------------------
    # Draft chain (topk=K, S steps). Tree drafting:
    #   - Step 0: record the K candidates from the latest draft-extend.
    #   - Forward step f (f=0..S-2): run B*K decode queries, each branch in
    #     its own per-branch tail block; produce K new candidates per branch
    #     and reduce K*K -> K by cumulative log-prob (sglang select_top_k).
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _draft_chain(self, seqs: List[_Eagle3Sequence]):
        """Produce sglang-format tree inputs from S draft steps with K branches."""
        S = self.spec_steps
        K = self.topk
        bs = len(seqs)

        verified_id = torch.tensor(
            [s.last_token for s in seqs], device=self.device, dtype=torch.long,
        )

        # Pull initial state from most recent draft extend.
        cur_top_p = torch.stack([s.draft_topk_p for s in seqs])           # [bs, K]
        cur_top_i_t = torch.stack([s.draft_topk_i_t for s in seqs])       # [bs, K]
        cur_hidden = torch.stack([s.draft_hidden for s in seqs])          # [bs, H_d]

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []

        # Step 0: just record (no forward).
        score_list.append(cur_top_p.unsqueeze(1))                          # [bs, 1, K]
        token_list.append(cur_top_i_t.view(bs, K))                         # [bs, K]
        parents_list.append(
            torch.arange(-1, K, dtype=torch.long, device=self.device)
            .unsqueeze(0).repeat(bs, 1)                                    # [bs, K + 1]
        )

        if S <= 1:
            parent_list, top_scores_index, draft_tokens = organize_draft_results(
                score_list, token_list, parents_list, self.num_draft_tokens,
            )
            return parent_list, top_scores_index, draft_tokens, verified_id

        # Per-(seq, branch) initial state for the forward loop.
        input_ids_bk = cur_top_i_t.reshape(bs * K).to(torch.int64)
        hidden_bk = cur_hidden.repeat_interleave(K, dim=0)                # [bs*K, H_d]
        scores = cur_top_p                                                # [bs, K]

        # ------------------------------------------------------------------
        # Build per-branch block tables. Each branch needs an ISOLATED tail
        # past d_committed_len (so K branches don't collide on shared slots).
        # When d_committed_len isn't block-aligned, the last prefix block has
        # `r = d_committed_len % block_size` real slots and `block_size - r`
        # unused-but-shared slots. The branch tail starts at position
        # d_committed_len, which lies in the partial last prefix block. We
        # therefore COPY the partial last prefix block per branch and append
        # extra blocks if the tail spills past one block_size boundary.
        bsz = self.block_size
        max_tail_pos = (S - 2)  # last forward writes at position d + (S-2)
        # Per-seq: number of branch blocks = blocks needed to cover positions
        # [aligned_down(d), aligned_down(d) + r + max_tail_pos].
        per_seq_branch_blocks: list[int] = []
        for seq in seqs:
            r = seq.d_committed_len % bsz
            span = r + max_tail_pos + 1  # +1 for inclusive count
            n = max(1, (span + bsz - 1) // bsz)
            per_seq_branch_blocks.append(n)
        max_bb = max(per_seq_branch_blocks) if per_seq_branch_blocks else 1

        prefix_aligned_blocks = [
            (s.d_committed_len + bsz - 1) // bsz for s in seqs
        ]

        # block_table per (seq, branch) length:
        #   prefix_blocks_truncated ++ branch_blocks
        # where prefix_blocks_truncated drops the last prefix block IF the
        # branch needs to own the partial slot (i.e. r > 0 OR we need to
        # extend); the dropped slot's content is copied into branch_blocks[0].
        max_prefix_kept = max(
            (s.d_committed_len // bsz) for s in seqs
        )  # full blocks before the partial one
        max_blocks_branch = max_prefix_kept + max_bb

        bt_branch_np = np.full((bs * K, max_blocks_branch), -1, dtype=np.int32)
        branch_blocks_first_np = np.empty((bs, K), dtype=np.int64)

        # Collect (src, dst) block ids for the batched partial-prefix-block
        # copy so we can issue one kernel instead of B*K small kernels.
        copy_src: list[int] = []
        copy_dst: list[int] = []

        for j, seq in enumerate(seqs):
            r = seq.d_committed_len % bsz
            n_full_prefix = seq.d_committed_len // bsz
            n_branch = per_seq_branch_blocks[j]
            full_prefix = seq.d_blocks[:n_full_prefix]

            # Allocate K * n_branch fresh blocks for this seq.
            new_blocks = self.draft_kv.alloc(K * n_branch)
            seq._chain_tail_blocks = new_blocks

            need_partial_copy = r > 0 and len(seq.d_blocks) > n_full_prefix
            partial_src = int(seq.d_blocks[n_full_prefix]) if need_partial_copy else -1

            for k in range(K):
                bb = new_blocks[k * n_branch:(k + 1) * n_branch]
                if need_partial_copy:
                    copy_src.append(partial_src)
                    copy_dst.append(bb[0])
                base = j * K + k
                bt_branch_np[base, :n_full_prefix] = full_prefix
                bt_branch_np[base, n_full_prefix:n_full_prefix + n_branch] = bb
                branch_blocks_first_np[j, k] = bb[0]

        if copy_src:
            src_t = torch.tensor(copy_src, device=self.device, dtype=torch.long)
            dst_t = torch.tensor(copy_dst, device=self.device, dtype=torch.long)
            self.draft_kv.kv[:, :, dst_t] = self.draft_kv.kv[:, :, src_t]

        bt_branch = torch.from_numpy(bt_branch_np).to(self.device)
        branch_blocks_first = torch.from_numpy(branch_blocks_first_np).to(
            self.device,
        )                                                                  # [B, K]

        base_pos_np = np.array(
            [s.d_committed_len for s in seqs], dtype=np.int64,
        )
        base_pos_t = torch.from_numpy(base_pos_np).to(self.device)         # [B]
        r_per_seq_t = torch.from_numpy(base_pos_np % bsz).to(self.device)  # [B]
        n_full_prefix_t_input = torch.from_numpy(base_pos_np // bsz).to(
            self.device,
        )                                                                  # [B]

        chain_runner = self._draft_chain_runner
        if (
            chain_runner is not None
            and bs <= chain_runner.B_max
            and bt_branch.shape[1] <= chain_runner.max_blocks_branch
        ):
            score_steps, token_steps, parents_steps, _ = chain_runner.replay(
                cur_top_i_t_real=input_ids_bk,
                cur_hidden_real=hidden_bk,
                cur_top_p_real=cur_top_p,
                base_pos_real=base_pos_t,
                r_per_seq_real=r_per_seq_t,
                n_full_prefix_real=n_full_prefix_t_input,
                bt_branch_real=bt_branch,
                raw_bs=bs,
            )
            for f in range(S - 1):
                score_list.append(score_steps[f])
                token_list.append(token_steps[f])
                parents_list.append(parents_steps[f])

            parent_list, top_scores_index, draft_tokens = (
                organize_draft_results(
                    score_list, token_list, parents_list,
                    self.num_draft_tokens,
                )
            )
            return parent_list, top_scores_index, draft_tokens, verified_id

        # ------------------------------------------------------------------
        # Eager fallback: per-step forward+topk+select_top_k loop.
        # ------------------------------------------------------------------
        base_pos_bk = base_pos_t.repeat_interleave(K)                      # [B*K]
        d_committed_t = base_pos_t                                         # [B]
        r_per_seq = r_per_seq_t                                            # [B]
        max_base_pos_cpu = int(base_pos_np.max())

        for f in range(S - 1):
            # Position for this forward step: base_pos + f for every branch.
            positions = base_pos_bk + f                                    # [B*K]

            # Slot mapping: branch's first block * block_size + (r + f), for
            # the case where the tail still fits in the first branch block.
            # If (r + f) >= block_size we'd need to look at branch_blocks[1+...];
            # we restrict S so that r + (S-2) < 2*block_size at most and
            # allocate enough branch blocks above; here pick the right block.
            offset_full = r_per_seq + f                                    # [B]
            block_idx_within_branch = (offset_full // bsz).to(torch.long)  # [B]
            offset_in_block = (offset_full % bsz).to(torch.long)           # [B]
            # Resolve the block id for each (seq, branch) at this f.
            # The branch's blocks live at bt_branch[base, n_full_prefix..]
            # so the i-th branch block id can be computed from
            # branch_blocks_first when block_idx == 0 (the common case) or
            # by gathering from bt_branch otherwise.
            # For simplicity, always gather from bt_branch.
            n_full_prefix_t = (d_committed_t // bsz).to(torch.long)        # [B]
            block_table_col = (
                n_full_prefix_t + block_idx_within_branch
            ).repeat_interleave(K)                                         # [B*K]
            block_ids_at_f = bt_branch[
                torch.arange(bs * K, device=self.device), block_table_col,
            ].to(torch.int64)                                              # [B*K]
            slot_mapping = (
                block_ids_at_f * bsz
                + offset_in_block.repeat_interleave(K).to(torch.int64)
            ).to(torch.int32)                                              # [B*K]

            # cache_seqlens for decode-mode attention = base_pos + f + 1.
            cache_seqlens = (d_committed_t + (f + 1)).to(torch.int32)      # [B]
            cache_seqlens_bk = cache_seqlens.repeat_interleave(K)          # [B*K]
            max_ctx = max_base_pos_cpu + (f + 1)

            with set_forward_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=cache_seqlens_bk,
                block_tables=bt_branch,
                max_context_len=max_ctx,
            ):
                hidden_to_logits, hidden_to_aux = self.draft.forward_draft(
                    input_ids_bk, positions.to(torch.long), hidden_bk,
                )
                # Decode-mode: ParallelLMHead.project returns one logit per
                # query, so this gives [B*K, draft_vocab].
                logits = self.draft.compute_logits(hidden_to_logits)

            log_probs = torch.log_softmax(logits, dim=-1)                  # [B*K, V_d]
            topk_p, topk_index = torch.topk(log_probs, K, dim=-1)          # [B*K, K]
            topk_index_t = self.draft.remap_draft_ids(topk_index)          # [B*K, K]

            # select_top_k: combine cumulative log-prob scores (B, K) with new
            # K*K candidate log-probs and pick top K per seq. Sglang multiplies
            # probabilities; we add log-probs which is mathematically equivalent
            # for ordering and avoids underflow on long chains.
            expand_scores = (
                scores.unsqueeze(2) + topk_p.reshape(bs, K, K)
            )                                                              # [B, K, K]
            topk_cs_p, topk_cs_index = torch.topk(
                expand_scores.flatten(start_dim=1), K, dim=-1,
            )                                                              # [B, K]
            scores = topk_cs_p

            topk_index_t_flat = topk_index_t.reshape(bs, K * K)            # [B, K*K]
            new_input_ids = torch.gather(
                topk_index_t_flat, dim=1, index=topk_cs_index,
            ).reshape(bs * K)                                              # [B*K]

            # Gather selected hidden states for the next step (per branch).
            sel_idx = (
                topk_cs_index.flatten() // K
                + torch.arange(0, bs * K, K, device=self.device)
                .repeat_interleave(K)
            )                                                              # [B*K]
            new_hidden = hidden_to_aux[sel_idx]

            # Tree-info bookkeeping for organize_draft_results.
            score_list.append(expand_scores)                               # [B, K, K]
            token_list.append(topk_index_t_flat)                           # [B, K*K]
            parents_list.append(topk_cs_index + (K * K * f + K))            # [B, K]

            input_ids_bk = new_input_ids
            hidden_bk = new_hidden

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.num_draft_tokens,
        )
        return parent_list, top_scores_index, draft_tokens, verified_id

    # ------------------------------------------------------------------
    # Verify forward on the target
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _prepare_target_verify_inputs(self, seqs: List[_Eagle3Sequence]) -> dict:
        """Allocate target KV slots and build static verify metadata.

        This runs before tree construction so the fused tree-build kernel can
        directly produce FA3 expand metadata from the target slot mapping.
        """
        bs = len(seqs)
        N = self.num_draft_tokens

        slot_mapping: list[int] = []
        block_tables_rows: list[list[int]] = []
        prefix_lens_cpu: list[int] = []
        for seq in seqs:
            new_total = seq.t_committed_len + N
            self._ensure_target_capacity(seq, new_total)
            slot_mapping.extend(
                self._slot_mapping_for_range(
                    seq.t_blocks, seq.t_committed_len, new_total,
                )
            )
            block_tables_rows.append(seq.t_blocks)
            prefix_lens_cpu.append(seq.t_committed_len)

        max_blocks = max(len(b) for b in block_tables_rows)
        bt = np.full((bs, max_blocks), -1, dtype=np.int32)
        for i, b in enumerate(block_tables_rows):
            bt[i, :len(b)] = b

        device = self.device
        return {
            "slot_t": torch.tensor(slot_mapping, dtype=torch.int32, device=device),
            "bt_t": torch.from_numpy(bt).to(device),
            "prefix_lens_cpu": prefix_lens_cpu,
            "cache_seqlens_prefix": torch.tensor(
                prefix_lens_cpu, dtype=torch.int32, device=device,
            ),
            "max_seqlen_k_prefix": max(prefix_lens_cpu) if prefix_lens_cpu else 0,
        }

    @torch.inference_mode()
    def _target_verify(
        self,
        seqs: List[_Eagle3Sequence],
        full_draft_tokens: torch.Tensor,
        positions_tensor: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,
        target_inputs: dict,
        page_table_expand: torch.Tensor,
        cache_seqlens_expand: torch.Tensor,
    ):
        bs = len(seqs)
        N = self.num_draft_tokens

        slot_t = target_inputs["slot_t"]
        bt_t = target_inputs["bt_t"]
        cascade = {
            "cache_seqlens_prefix": target_inputs["cache_seqlens_prefix"],
            "max_seqlen_k_prefix": target_inputs["max_seqlen_k_prefix"],
            "page_table_expand": page_table_expand,
            "cache_seqlens_expand": cache_seqlens_expand,
        }

        runner = self._target_verify_runner
        if (
            runner is not None
            and bs <= runner.B_max
        ):
            # Fast path: replay a captured CUDA graph. Outputs are views into
            # runner-owned persistent buffers. They are consumed before the
            # next target-verify replay, so avoid a large logits/aux clone
            # on every speculative step.
            ids_t = full_draft_tokens.to(torch.int64)
            pos_t = positions_tensor.to(torch.long)
            h_view, aux_views, target_predict = runner.replay(
                input_ids_real=ids_t,
                positions_real=pos_t,
                slot_mapping_real=slot_t,
                block_table_real=bt_t,
                cache_seqlens_prefix_real=cascade["cache_seqlens_prefix"],
                page_table_expand_real=cascade["page_table_expand"],
                cache_seqlens_expand_real=cascade["cache_seqlens_expand"],
                max_seqlen_k_prefix_real=cascade["max_seqlen_k_prefix"],
                raw_bs=bs,
            )
            hidden_states = h_view
            aux_list = aux_views
        else:
            prefix_lens_cpu = target_inputs["prefix_lens_cpu"]
            ctx_lens_cpu = [p + N for p in prefix_lens_cpu]
            ctx_t = torch.tensor(ctx_lens_cpu, dtype=torch.int32, device=self.device)
            cu_seqlens_q_prefix = torch.arange(
                0, (bs + 1) * N, N, device=self.device, dtype=torch.int32,
            )
            cu_seqlens_q_expand = torch.arange(
                bs * N + 1, device=self.device, dtype=torch.int32,
            )
            with set_forward_context(
                is_prefill=False,
                slot_mapping=slot_t,
                context_lens=ctx_t,
                block_tables=bt_t,
                max_context_len=max(ctx_lens_cpu),
                is_tree_verify=True,
                tree_num_verify_tokens=N,
                tree_block_table_prefix=bt_t,
                tree_cache_seqlens_prefix=cascade["cache_seqlens_prefix"],
                tree_cu_seqlens_q_prefix=cu_seqlens_q_prefix,
                tree_max_seqlen_q_prefix=N,
                tree_max_seqlen_k_prefix=cascade["max_seqlen_k_prefix"],
                tree_page_table_expand=cascade["page_table_expand"],
                tree_cache_seqlens_expand=cascade["cache_seqlens_expand"],
                tree_cu_seqlens_q_expand=cu_seqlens_q_expand,
            ):
                ids_t = full_draft_tokens.to(torch.int64)
                out = self.target.model(ids_t, positions_tensor.to(torch.long))
                if isinstance(out, tuple):
                    hidden_states, aux_list = out
                else:
                    hidden_states, aux_list = out, []

                # compute_logits while still in is_prefill=False context so it
                # returns one logit per *flat tree token* (shape [B*N, vocab]).
                logits = self.target.compute_logits(hidden_states)
            target_predict = logits.argmax(dim=-1)  # [B*N]

        candidates = full_draft_tokens.view(bs, N).to(torch.long)
        target_predict_2d = target_predict.view(bs, N).to(torch.long)
        predicts = torch.empty(
            bs * N, device=self.device, dtype=torch.int32,
        )
        accept_index = torch.empty(
            bs, self.spec_steps + 1,
            device=self.device, dtype=torch.int32,
        )
        accept_token_num = torch.empty(
            bs, device=self.device, dtype=torch.int32,
        )

        verify_tree_greedy(
            predicts, accept_index, accept_token_num,
            candidates,
            retrive_index.view(bs, N).to(torch.long),
            retrive_next_token.view(bs, N).to(torch.long),
            retrive_next_sibling.view(bs, N).to(torch.long),
            target_predict_2d,
        )
        accepted_aux_padded = None
        verified_ids_padded = None

        return (
            predicts,
            accept_index,
            accept_token_num,
            hidden_states,
            aux_list,
            accepted_aux_padded,
            verified_ids_padded,
        )

    # ------------------------------------------------------------------
    # Single EAGLE-3 step: draft chain -> tree build -> verify -> accept
    # ------------------------------------------------------------------
    def _eagle3_step(self, seqs: List[_Eagle3Sequence], eos_id: int):
        import os as _os
        _dbg = _os.environ.get("KB_NANO_EAGLE3_DEBUG") == "1"
        if _dbg:
            print(f"[dbg] step start, n_active={len(seqs)} "
                  f"committed_lens={[s.t_committed_len for s in seqs]} "
                  f"d_committed_lens={[s.d_committed_len for s in seqs]}", flush=True)
        # 1. Draft chain (uses each seq's draft_topk_* + draft_hidden cached
        #    by the most recent draft-extend).
        parent_list, top_scores_index, draft_tokens, verified_id = self._draft_chain(seqs)
        if _dbg:
            print(f"[dbg] draft chain done", flush=True)

        # 2. Build tree and the target-verify metadata it depends on.
        verify_runner = self._target_verify_runner
        target_inputs = self._prepare_target_verify_inputs(seqs)
        BN = len(seqs) * self.num_draft_tokens
        if verify_runner is not None and len(seqs) <= verify_runner.B_max:
            position_buf = verify_runner.bufs.positions[:BN]
            page_table_expand_buf = verify_runner.bufs.page_table_expand[:BN]
            cache_seqlens_expand_buf = verify_runner.bufs.cache_seqlens_expand[:BN]
        else:
            position_buf = None
            page_table_expand_buf = None
            cache_seqlens_expand_buf = None
        (
            positions, retrive_index, retrive_next_token,
            retrive_next_sibling, full_draft_tokens,
            page_table_expand, cache_seqlens_expand,
        ) = build_tree_kernel_efficient_with_metadata(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=target_inputs["cache_seqlens_prefix"],
            slot_mapping_draft=target_inputs["slot_t"],
            topk=self.topk,
            spec_steps=self.spec_steps,
            num_verify_tokens=self.num_draft_tokens,
            position_buf=position_buf,
            page_table_expand_buf=page_table_expand_buf,
            cache_seqlens_expand_buf=cache_seqlens_expand_buf,
        )

        if _dbg:
            print(f"[dbg] tree built, full_draft_tokens={full_draft_tokens.shape}", flush=True)

        # 3. Verify.
        (
            predicts, accept_index, accept_token_num, hidden_states, aux_list,
            accepted_aux_padded, verified_ids_padded,
        ) = self._target_verify(
            seqs, full_draft_tokens, positions,
            retrive_index, retrive_next_token, retrive_next_sibling,
            target_inputs, page_table_expand, cache_seqlens_expand,
        )
        if _dbg:
            print(f"[dbg] verify done, accept_token_num={accept_token_num.tolist()}, "
                  f"accept_index[0]={accept_index[0].tolist()}, "
                  f"committed_pre={[s.t_committed_len for s in seqs]}", flush=True)

        # 4. Accept tokens; roll back rejected target KV.
        N = self.num_draft_tokens
        accept_num_cpu = accept_token_num.tolist()
        accept_index_cpu = accept_index.tolist()
        predicts_cpu = predicts.tolist() if predicts is not None else None
        verified_ids_cpu = (
            verified_ids_padded.tolist()
            if verified_ids_padded is not None else None
        )

        # Remap target K/V so that logical position ``t_committed_len + k``
        # holds the K/V for the k-th accepted token (NOT for the k-th tree
        # node in flat write order). Must run before we touch t_committed_len
        # or free any blocks. With chain drafting this is a no-op; with tree
        # drafting it is required for correctness.
        self._remap_target_kv_after_verify(seqs, accept_index_cpu, accept_num_cpu)

        accepted_ids_per_seq: list[list[int]] = []
        flat_accepted_indices: list[int] = []
        aux_offsets = [0]
        for i in range(len(seqs)):
            n_accept = int(accept_num_cpu[i])
            indices = accept_index_cpu[i][: n_accept + 1]
            if verified_ids_cpu is not None:
                accepted_ids_per_seq.append(
                    [int(x) for x in verified_ids_cpu[i][: n_accept + 1]]
                )
            else:
                accepted_ids_per_seq.append(
                    [int(predicts_cpu[idx]) for idx in indices]
                )
            flat_accepted_indices.extend(indices)
            aux_offsets.append(len(flat_accepted_indices))

        if accepted_aux_padded is not None:
            accepted_aux_flat = None
        elif len(aux_list) > 0:
            # Only the accepted path feeds the draft extend. Avoid materializing
            # [B*N, 3H] every step when the accepted set is at most B*(S+1).
            flat_idx_t = torch.tensor(
                flat_accepted_indices, device=self.device, dtype=torch.long,
            )
            accepted_aux_flat = torch.cat(
                [a[flat_idx_t] for a in aux_list], dim=-1,
            )
        else:
            accepted_aux_flat = None

        # Per-seq accepted tokens + aux for the post-verify draft extend.
        accepted_lists: list[list[int]] = []
        accepted_aux_per_seq: list[torch.Tensor] = []

        for i, seq in enumerate(seqs):
            n_accept = int(accept_num_cpu[i])  # # accepted SPECULATIVE
            accepted_ids = accepted_ids_per_seq[i]
            seq.token_ids.extend(accepted_ids)
            seq.generated_ids.extend(accepted_ids)
            seq.last_token = accepted_ids[-1]
            accepted_lists.append(accepted_ids)

            tentative_committed = seq.t_committed_len + N
            new_committed = seq.t_committed_len + (n_accept + 1)
            old_blocks = (tentative_committed + self.block_size - 1) // self.block_size
            new_blocks = (new_committed + self.block_size - 1) // self.block_size
            if new_blocks < old_blocks:
                released = seq.t_blocks[new_blocks:]
                self.target_kv.free(released)
            seq.t_blocks = seq.t_blocks[:new_blocks]
            seq.t_committed_len = new_committed

            if accepted_aux_padded is not None:
                acc_aux = accepted_aux_padded[i, : n_accept + 1]
                accepted_aux_per_seq.append(acc_aux)
            elif accepted_aux_flat is not None:
                acc_aux = accepted_aux_flat[
                    aux_offsets[i]:aux_offsets[i + 1]
                ]                                          # [n_accept+1, 3*H_t]
                accepted_aux_per_seq.append(acc_aux)
            else:
                accepted_aux_per_seq.append(torch.zeros(
                    (len(accepted_ids), self.target_config.hidden_size * 3),
                    device=self.device, dtype=self.dtype,
                ))

            # Free the per-branch tail blocks allocated by _draft_chain.
            # The draft KV will be re-extended with the actually-accepted
            # tokens (and target aux hidden states) below.
            tail = getattr(seq, "_chain_tail_blocks", None)
            if tail is not None and len(tail) > 0:
                self.draft_kv.free(list(tail))
                seq._chain_tail_blocks = []
            # Also drop any stray prefix-tail blocks (none should exist after
            # the prior extend, but keep this for safety).
            self._free_draft_tail(seq, seq.d_committed_len)

            if not seq.ignore_eos and (eos_id in accepted_ids):
                pos = accepted_ids.index(eos_id)
                drop = len(accepted_ids) - (pos + 1)
                if drop > 0:
                    del seq.token_ids[-drop:]
                    del seq.generated_ids[-drop:]
                seq.finished = True
            elif len(seq.generated_ids) >= seq.max_tokens:
                if len(seq.generated_ids) > seq.max_tokens:
                    excess = len(seq.generated_ids) - seq.max_tokens
                    del seq.token_ids[-excess:]
                    del seq.generated_ids[-excess:]
                seq.finished = True

        # 5. Post-verify draft extend: digest the accepted tokens into draft KV
        #    and capture (topk_p, topk_i, hidden) for the next chain step 0.
        live_seqs: list[_Eagle3Sequence] = []
        ext_input_ids: list[torch.Tensor] = []
        ext_hiddens: list[torch.Tensor] = []
        ext_lens: list[int] = []
        for i, seq in enumerate(seqs):
            if seq.finished:
                continue
            acc_ids = accepted_lists[i]
            acc_aux = accepted_aux_per_seq[i]
            ids = torch.tensor(acc_ids, dtype=torch.int64, device=self.device)
            ext_input_ids.append(ids)
            ext_hiddens.append(acc_aux)
            ext_lens.append(len(acc_ids))
            live_seqs.append(seq)

        if live_seqs:
            self._draft_extend(
                live_seqs, ext_input_ids, ext_hiddens, ext_lens,
                post_verify=True,
            )
            if _dbg:
                print(f"[dbg] post-verify draft extend done; "
                      f"d_committed_lens={[s.d_committed_len for s in live_seqs]}",
                      flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        self.target_kv.reset()
        self.draft_kv.reset()

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[List[int]] | List[str],
        sampling_params: Eagle3SamplingParams | List[Eagle3SamplingParams],
        use_tqdm: bool = False,
        decode_text: bool = True,
    ) -> List[Eagle3Output]:
        if isinstance(sampling_params, Eagle3SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)

        # One-shot CUDA graph capture (if enabled).
        self._maybe_build_graph_runners()

        prompt_token_ids: List[List[int]] = []
        for p in prompts:
            if isinstance(p, str):
                prompt_token_ids.append(
                    self.tokenizer.encode(p, add_special_tokens=False)
                )
            else:
                prompt_token_ids.append(list(p))

        eos_id = self.tokenizer.eos_token_id

        all_outputs: List[Optional[Eagle3Output]] = [None] * len(prompts)
        idx_pool = list(range(len(prompts)))
        active: list[tuple[int, _Eagle3Sequence]] = []
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=len(prompts), desc="EAGLE-3 generate")
        else:
            pbar = None

        def _finalize(seq: _Eagle3Sequence, src_i: int):
            if seq.t_blocks:
                self.target_kv.free(seq.t_blocks)
                seq.t_blocks.clear()
            if seq.d_blocks:
                self.draft_kv.free(seq.d_blocks)
                seq.d_blocks.clear()
            all_outputs[src_i] = Eagle3Output(
                prompt_token_ids=seq.prompt_ids,
                token_ids=list(seq.generated_ids),
                generated_text=(
                    self.tokenizer.decode(seq.generated_ids)
                    if decode_text else ""
                ),
            )
            if pbar is not None:
                pbar.update(1)

        def _admit_requests():
            if not idx_pool or len(active) >= self.max_num_seqs:
                return

            n_new = min(self.max_num_seqs - len(active), len(idx_pool))
            batch_idx = idx_pool[:n_new]
            del idx_pool[:n_new]

            new_pairs: list[tuple[int, _Eagle3Sequence]] = []
            new_seqs: list[_Eagle3Sequence] = []
            for i in batch_idx:
                seq = _Eagle3Sequence(
                    prompt_token_ids[i],
                    sampling_params[i].max_tokens,
                    sampling_params[i].ignore_eos,
                )
                new_pairs.append((i, seq))
                new_seqs.append(seq)

            aux_per_seq = self._target_prefill(new_seqs)

            # Initial draft EXTEND on the prompt (sglang's
            # ``forward_draft_extend`` for prefill). Input is the prompt
            # shifted left by 1, with the bonus token replacing the final
            # slot:  [tok_1, tok_2, ..., tok_{P-1}, bonus_P].
            ext_input_ids: list[torch.Tensor] = []
            ext_hiddens: list[torch.Tensor] = []
            ext_lens: list[int] = []
            live_pairs: list[tuple[int, _Eagle3Sequence]] = []
            live_for_extend: list[_Eagle3Sequence] = []
            for i, (src_i, seq) in enumerate(new_pairs):
                P = len(seq.prompt_ids)
                if (
                    (not seq.ignore_eos and seq.last_token == eos_id)
                    or len(seq.generated_ids) >= seq.max_tokens
                ):
                    seq.finished = True
                    _finalize(seq, src_i)
                    continue

                ids = list(seq.prompt_ids[1:]) + [seq.last_token]
                ext_input_ids.append(
                    torch.tensor(ids, dtype=torch.int64, device=self.device)
                )
                ext_hiddens.append(aux_per_seq[i])
                ext_lens.append(P)
                live_pairs.append((src_i, seq))
                live_for_extend.append(seq)

            if live_for_extend:
                self._draft_extend(
                    live_for_extend, ext_input_ids, ext_hiddens, ext_lens,
                )
                active.extend(live_pairs)

        _admit_requests()
        while active:
            self._eagle3_step([seq for _, seq in active], eos_id)

            kept: list[tuple[int, _Eagle3Sequence]] = []
            for src_i, seq in active:
                if seq.finished:
                    _finalize(seq, src_i)
                else:
                    kept.append((src_i, seq))
            active = kept
            _admit_requests()

        if pbar is not None:
            pbar.close()

        assert all(o is not None for o in all_outputs)
        return all_outputs
