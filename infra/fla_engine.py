"""FLA inference engine for recurrent linear-attention models.

Distinct from ``LlamaEngine`` because recurrent models (GLA / RetNet /
RWKV7) carry per-sequence state matrices instead of a paged KV cache.
The semantic gap is too wide to share scheduling code:

  - No paged KV cache: state is a per-layer ``[B, H, K, V]`` tensor.
  - No flat token layout: prefill and decode are batched ``[B, T, D]``
    forwards; padding and per-seq attention masks instead of varlen.
  - No CUDA-graph capture against fixed slot tables: the state buffers
    move whenever the active batch shape changes.

Surface area mirrors ``LlamaEngine`` for the bits ``bench_*.py`` and
user code touch:

  - ``FLAEngine(model_name, dtype, seed, max_num_seqs, ...).generate(prompts, sampling_params)``
  - ``SamplingParams`` and ``GenerationOutput`` are re-exported from
    ``infra.engine`` so callers can use either engine interchangeably.

Scheduling: continuous batching with chunked prefill. Each loop
iteration runs at most ONE B=1 prefill chunk (fixed
``chunked_prefill_size``, default 1024) AND one B=N batched decode step
over all sequences whose prefill is complete. New sequences enter the
decode batch as soon as their final prefill chunk produces a sample,
without waiting for the rest of the prefill backlog.

The chunk size is forced to a multiple of 64 (FLA's chunk-vs-recurrent
threshold) and the *last* chunk of each prompt is absorbed if it would
leave a sub-64-token tail; this keeps every chunk on the chunk kernel
and avoids tiny FP differences vs single-shot prefill that can flip
argmax for low-confidence tokens.

State is held in a persistent slot-allocated cache. Each in-flight
sequence owns a slot in ``[max_num_seqs, *state_shape]`` buffers (one
buffer per L2 layer, lazily allocated on first commit). Decode forward
operates directly on the previous step's output cache when membership
is unchanged ("live-cache reuse"); only when the active set changes
(seq finishes / admission) do we flush the live cache to slots and
re-gather the new active subset via a single ``index_select`` per
layer. This drives per-step engine overhead from O(layers x batch)
tiny CUDA copies (the previous design) down to amortized O(1).

Tensor parallel is not implemented (TP=1 only) - all FLA models we
target are < 10B params and fit on a single H200.
"""

from __future__ import annotations

import os
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import numpy as np
import torch

from .engine import GenerationOutput, SamplingParams  # re-exported

__all__ = ["FLAEngine", "SamplingParams", "GenerationOutput"]


# Suppress one expected FLA warning: ``seq_len < num_heads`` triggers a
# layout heuristic check during T=1 RWKV7 decode. Our shapes are correct.
warnings.filterwarnings(
    "ignore",
    message=r".*seq_len.*<.*num_heads.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Per-sequence bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class _ActiveSeq:
    """Bookkeeping for a single in-flight sequence."""
    seq_id: int
    prompt_ids: list[int]
    sampling: SamplingParams
    max_tokens: int = 512
    ignore_eos: bool = False

    # prefill progress (number of prompt tokens already consumed)
    prefill_pos: int = 0
    # tokens we've sampled (does not include the prompt)
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False

    # Slot index into the engine's persistent state cache (-1 = unallocated).
    slot_id: int = -1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_REGISTRY = {
    "fla-hub/gla-2.7B-100B": ("kb_nano.tasks.baseline.L4.gla", "GLAConfig", "GLAForCausalLM"),
    "fla-hub/retnet-2.7B-100B": ("kb_nano.tasks.baseline.L4.retnet", "RetNetConfig", "RetNetForCausalLM"),
    "fla-hub/rwkv7-2.9B-g1": ("kb_nano.tasks.baseline.L4.rwkv7", "RWKV7Config", "RWKV7ForCausalLM"),
    "fla-hub/rwkv7-2.9B-world": ("kb_nano.tasks.baseline.L4.rwkv7", "RWKV7Config", "RWKV7ForCausalLM"),
}


def _load_model(model_name: str, dtype: torch.dtype, device: torch.device):
    """Load an FLA model via snapshot_download + safetensors weight copy."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import importlib

    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unsupported FLA model: {model_name!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )

    module_name, config_cls, model_cls = _REGISTRY[model_name]
    mod = importlib.import_module(module_name)
    Config = getattr(mod, config_cls)
    Model = getattr(mod, model_cls)

    model_path = snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )
    config = Config.from_pretrained(model_path)
    model = Model(config)

    # FLA checkpoints store the token embedding at ``model.embeddings.weight``
    # but the L1 ``Embedding`` op nests ``nn.Embedding`` as ``self.emb``, so
    # the actual parameter path is ``model.embeddings.emb.weight``. Remap.
    def _remap(name: str) -> str:
        if name == "model.embeddings.weight":
            return "model.embeddings.emb.weight"
        return name

    sf_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    loaded = 0
    for sf in sf_files:
        with safe_open(sf, "pt", "cpu") as f:
            for name in f.keys():
                mapped = _remap(name)
                try:
                    param = model.get_parameter(mapped)
                except AttributeError:
                    continue
                src = f.get_tensor(name)
                if src.shape != param.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch loading {name}: "
                        f"checkpoint {tuple(src.shape)} vs model {tuple(param.data.shape)}"
                    )
                param.data.copy_(src)
                loaded += 1
    print(f"  [FLAEngine] Loaded {loaded} weights from {model_path}")

    model = model.to(device=device, dtype=dtype).eval()
    return model, model_path


# ---------------------------------------------------------------------------
# Persistent slot-allocated state cache
# ---------------------------------------------------------------------------
class _SlotCache:
    """Persistent per-layer recurrent state, addressed by slot index.

    Each L2 module's state lives in a single ``[max_num_seqs, *shape]``
    tensor, lazily allocated on the first commit (when we learn the
    state shape from a forward pass). All gather/scatter happens via
    one ``index_select`` / ``index_copy_`` per layer per call -- a
    constant number of CUDA launches regardless of the active batch
    size, replacing the previous O(layers x batch) copy loop.
    """

    def __init__(self, max_num_seqs: int, device: torch.device):
        self.max_num_seqs = max_num_seqs
        self.device = device
        # layer_id / module_id -> [max_num_seqs, *state_shape]
        self.states: dict[int, torch.Tensor] = {}
        self.conv_states: dict[int, torch.Tensor] = {}

    def _ensure(self, store: dict, key: int, sample: torch.Tensor) -> None:
        if key not in store:
            store[key] = torch.zeros(
                (self.max_num_seqs,) + sample.shape[1:],
                dtype=sample.dtype, device=sample.device,
            )

    def gather(self, slot_ids: torch.Tensor):
        """Build a RecurrentCache from the rows at ``slot_ids``.

        ``slot_ids`` is an int64 tensor of shape ``[B]``. The returned
        cache holds contiguous ``[B, *state_shape]`` tensors (one
        index_select per layer).
        """
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache
        c = RecurrentCache()
        for k, buf in self.states.items():
            c.states[k] = buf.index_select(0, slot_ids)
        for k, buf in self.conv_states.items():
            c.conv_states[k] = buf.index_select(0, slot_ids)
        return c

    def scatter(self, slot_ids: torch.Tensor, src) -> None:
        """Scatter src.states / src.conv_states into the slot rows."""
        for k, t in src.states.items():
            self._ensure(self.states, k, t)
            self.states[k].index_copy_(0, slot_ids, t)
        for k, t in src.conv_states.items():
            self._ensure(self.conv_states, k, t)
            self.conv_states[k].index_copy_(0, slot_ids, t)

    def zero_slot(self, slot_id: int) -> None:
        """Reset a single slot to all-zero (called on slot reuse)."""
        for buf in self.states.values():
            buf[slot_id].zero_()
        for buf in self.conv_states.values():
            buf[slot_id].zero_()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class FLAEngine:
    """Single-rank inference engine for FLA recurrent models.

    Public methods mirror ``LlamaEngine``:
      - ``generate(prompts, sampling_params, use_tqdm=False, collect_logits=False)``
    """

    # FLA's chunk-kernel boundary. Sub-64 tails fall back to fused-recurrent
    # which is mathematically equivalent but not bit-identical, so we never
    # leave a chunk smaller than this.
    _CHUNK_BOUNDARY = 64

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        max_num_seqs: int = 512,
        chunked_prefill_size: int = 1024,
        max_prefill_tokens: int = 196608,
        trust_remote_code: bool = True,
    ):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        # Round up to a multiple of 64 so every prefill chunk dispatches
        # to the chunk kernel (not fused-recurrent), matching single-shot
        # prefill bit-for-bit at the chunk boundary.
        cps = max(chunked_prefill_size, self._CHUNK_BOUNDARY)
        self.chunked_prefill_size = ((cps + self._CHUNK_BOUNDARY - 1)
                                     // self._CHUNK_BOUNDARY) * self._CHUNK_BOUNDARY
        # Cap on tokens per batched-prefill forward (B * chunk_size).
        # Larger models (RWKV7-2.9B) have big intermediate activations
        # and OOM at B=256, T=1024 = 262k tokens; ~192k stays safely
        # under H200 capacity across all three model families while
        # allowing one-shot prefill for the bench's 200-seq scenario.
        self.max_prefill_tokens = max(max_prefill_tokens, self.chunked_prefill_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self._set_seeds(seed)

        self.model, self.model_path = _load_model(model_name, dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Cache the ordered list of L2 attention + FFN module ids so state
        # gather/scatter walks them in deterministic, layer-natural order.
        from kb_nano.tasks.baseline.L2.gla_attention import GatedLinearAttention
        from kb_nano.tasks.baseline.L2.rwkv7_attention import RWKV7Attention
        from kb_nano.tasks.baseline.L2.rwkv7_ffn import RWKV7FeedForward

        attn_classes = (GatedLinearAttention, RWKV7Attention)
        # Order matters for stable cache walks
        self._attn_layer_ids: list[int] = [
            id(m) for m in self.model.modules() if isinstance(m, attn_classes)
        ]
        # Conv-state-bearing modules (RWKV7 attn + FFN). Order matches
        # discovery order so gather/scatter is deterministic.
        self._conv_module_ids: list[int] = [
            id(m) for m in self.model.modules()
            if isinstance(m, (RWKV7Attention, RWKV7FeedForward))
        ]
        self._is_rwkv7 = any(
            isinstance(m, RWKV7Attention) for m in self.model.modules()
        )
        self._supports_padded_initial_prefill = (
            not self._is_rwkv7
            and all(
                not getattr(m, "use_rotary", False)
                for m in self.model.modules()
                if isinstance(m, GatedLinearAttention)
            )
        )

        # Persistent slot cache + free-slot pool (per-generate state).
        self._slot_cache: _SlotCache | None = None
        self._free_slots: list[int] = []
        # Live cache reuse: if the previous forward's output cache covers
        # the SAME active sequences in the SAME order, we can feed it
        # directly to the next forward and skip the gather entirely.
        self._live_cache = None
        self._live_active: list[_ActiveSeq] | None = None

    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        if params.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())
        scaled = logits.float() / params.temperature
        if params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum - probs >= params.top_p
            sorted_logits[mask] = float("-inf")
            scaled = scaled.scatter(0, sorted_idx, sorted_logits)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _sample_batch(
        self, logits: torch.Tensor, seqs: list[_ActiveSeq],
    ) -> list[int]:
        # Greedy fast path (most common): one argmax launch + one sync,
        # no per-row Python work, no fp32 upcast.
        if all(s.sampling.temperature == 0.0 for s in seqs):
            return logits.argmax(dim=-1).tolist()
        return [self._sample(logits[i], s.sampling) for i, s in enumerate(seqs)]

    # ------------------------------------------------------------------
    # Slot-cache plumbing
    # ------------------------------------------------------------------
    def _reset_slot_cache(self) -> None:
        self._slot_cache = _SlotCache(self.max_num_seqs, self.device)
        self._free_slots = list(reversed(range(self.max_num_seqs)))  # pop() yields 0,1,2...
        self._live_cache = None
        self._live_active = None

    def _acquire_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError(
                f"FLAEngine: no free slots (max_num_seqs={self.max_num_seqs})"
            )
        slot = self._free_slots.pop()
        # Clear any stale state from the previous occupant. Cheap (one
        # zero_() per buffer, no per-layer Python loop overhead at the
        # call site).
        if self._slot_cache is not None:
            self._slot_cache.zero_slot(slot)
        return slot

    def _release_slot(self, seq: _ActiveSeq) -> None:
        if seq.slot_id < 0:
            return
        # If the live cache references this seq, flush it back to slots
        # first so the survivors don't lose their state when we drop the
        # live cache.
        if self._live_active is not None and seq in self._live_active:
            self._flush_live()
        self._free_slots.append(seq.slot_id)
        seq.slot_id = -1

    def _flush_live(self) -> None:
        """Scatter the live cache back into the slot store."""
        if self._live_cache is None or not self._live_active:
            self._live_cache = None
            self._live_active = None
            return
        slot_ids = torch.tensor(
            [s.slot_id for s in self._live_active],
            dtype=torch.int64, device=self.device,
        )
        self._slot_cache.scatter(slot_ids, self._live_cache)
        self._live_cache = None
        self._live_active = None

    @staticmethod
    def _same_active(a: list[_ActiveSeq], b: list[_ActiveSeq] | None) -> bool:
        if b is None or len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x is not y:
                return False
        return True

    def _build_input_cache(self, active: list[_ActiveSeq]):
        """Return a RecurrentCache covering ``active``, reusing the live
        cache if possible.
        """
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache
        if self._same_active(active, self._live_active):
            cache = self._live_cache
        else:
            self._flush_live()
            slot_ids = torch.tensor(
                [s.slot_id for s in active],
                dtype=torch.int64, device=self.device,
            )
            cache = self._slot_cache.gather(slot_ids)
        cache.seq_offsets = torch.tensor(
            [s.prefill_pos for s in active],
            dtype=torch.int64, device=self.device,
        )
        return cache

    # ------------------------------------------------------------------
    # Prefill / decode primitives
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prefill_chunk_batched(
        self, seqs: list[_ActiveSeq], chunk_sizes: int | list[int],
    ) -> list[torch.Tensor | None]:
        """Run one packed, variable-length prefill chunk over ``seqs``.

        Each seq consumes its own next chunk starting at ``prefill_pos``.
        GLA prefill is packed as ``[1, sum(chunk_sizes)]`` plus
        ``cu_seqlens`` so different prompt lengths share one kernel launch
        without padding or using unknown decode lengths for batching.

        Returns a list of per-row logits ([T, V]) for seqs whose prefill
        is complete after this chunk (so the caller can sample the first
        generated token), or ``None`` for seqs that still need more
        chunks. Skipping the lm_head copy for the latter halves the
        return-side memory bandwidth in chunked-prefill paths.
        """
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache

        # Prefill writes the slots in ``seqs``. If any of those slots is
        # currently held by the live decode cache, flush so the
        # decode-side state isn't trampled by our scatter below.
        if (self._live_active is not None
                and any(s in self._live_active for s in seqs)):
            self._flush_live()

        if isinstance(chunk_sizes, int):
            chunk_sizes = [chunk_sizes] * len(seqs)
        chunks = [
            s.prompt_ids[s.prefill_pos:s.prefill_pos + chunk_size]
            for s, chunk_size in zip(seqs, chunk_sizes, strict=True)
        ]
        slot_ids = torch.tensor(
            [s.slot_id for s in seqs], dtype=torch.int64, device=self.device,
        )

        if self._slot_cache.states or self._slot_cache.conv_states:
            cache = self._slot_cache.gather(slot_ids)
        else:
            # First forward ever: buffers not yet allocated. Pass an empty
            # cache; the L2 attn dispatch sees no initial_state and
            # starts from zero.
            cache = RecurrentCache()
        cache.seq_offsets = torch.tensor(
            [s.prefill_pos for s in seqs], dtype=torch.int64, device=self.device,
        )

        # Fast dense path for first chunks: left-pad with zero embeddings.
        # With a zero recurrent state, leading all-zero tokens leave the
        # state unchanged, so real tokens end at the last column and we can
        # use the dense chunk kernel instead of the slower varlen path.
        use_dense_leftpad = (
            self._supports_padded_initial_prefill
            and all(s.prefill_pos == 0 for s in seqs)
        )
        if use_dense_leftpad:
            max_len = max(len(chunk) for chunk in chunks)
            ids = torch.zeros((len(chunks), max_len), dtype=torch.long, device=self.device)
            pad_mask = torch.zeros((len(chunks), max_len, 1), dtype=self.dtype, device=self.device)
            for i, chunk in enumerate(chunks):
                start = max_len - len(chunk)
                ids[i, start:] = torch.tensor(chunk, dtype=torch.long, device=self.device)
                pad_mask[i, start:, 0] = 1
            embeds = self.model.model.embeddings(ids) * pad_mask
            out = self.model(
                inputs_embeds=embeds, past_key_values=cache, use_cache=True,
                num_logits_to_keep=1,
            )
        else:
            flat_ids = [tok for chunk in chunks for tok in chunk]
            ids = torch.tensor([flat_ids], dtype=torch.long, device=self.device)
            lengths = torch.tensor(
                [len(chunk) for chunk in chunks], dtype=torch.int64, device=self.device,
            )
            cu_seqlens = torch.empty(len(chunks) + 1, dtype=torch.int64, device=self.device)
            cu_seqlens[0] = 0
            cu_seqlens[1:] = lengths.cumsum(0)
            logits_indices = cu_seqlens[1:] - 1
            out = self.model(
                input_ids=ids, past_key_values=cache, use_cache=True,
                logits_indices=logits_indices, cu_seqlens=cu_seqlens,
            )
        self._slot_cache.scatter(slot_ids, out.past_key_values)

        # out.logits has shape [N, 1, V] -- index by request row, take
        # position 0, hand back to caller.
        results: list[torch.Tensor | None] = []
        for i, (s, chunk_size) in enumerate(zip(seqs, chunk_sizes, strict=True)):
            s.prefill_pos += chunk_size
            results.append(out.logits[i, 0] if s.prefill_pos >= len(s.prompt_ids) else None)
        return results

    @torch.no_grad()
    def _decode_step(self, active: list[_ActiveSeq]) -> list[int]:
        ids = torch.tensor(
            [[seq.generated_ids[-1]] for seq in active],
            dtype=torch.long, device=self.device,
        )
        cache = self._build_input_cache(active)
        # T=1 here so num_logits_to_keep=1 is a no-op shape-wise but lets
        # the L4 layer skip the [:, -1:, :] slice (and the resulting
        # contiguous() it sometimes triggers).
        out = self.model(
            input_ids=ids, past_key_values=cache, use_cache=True,
            num_logits_to_keep=1,
        )
        # Keep the output cache LIVE for the next decode step. Only when
        # membership changes (next call's active != self._live_active)
        # will we flush it to the slot store.
        self._live_cache = out.past_key_values
        self._live_active = list(active)
        for seq in active:
            seq.prefill_pos += 1  # decode step appends one token to the state
        return self._sample_batch(out.logits[:, 0, :], active)

    # ------------------------------------------------------------------
    # Chunk planner -- tail-absorption to keep every chunk on the chunk
    # kernel (T >= 64) when the prompt allows.
    # ------------------------------------------------------------------
    def _next_chunk_size(self, seq: _ActiveSeq) -> int:
        remaining = len(seq.prompt_ids) - seq.prefill_pos
        if remaining <= self.chunked_prefill_size + self._CHUNK_BOUNDARY:
            # Take everything that's left to avoid leaving a sub-64 tail.
            return remaining
        return self.chunked_prefill_size

    def _next_prefill_batch(
        self, prefilling: list[_ActiveSeq],
    ) -> tuple[list[_ActiveSeq], list[int]]:
        """Select the next packed varlen prefill batch by prompt state.

        The only batching inputs are current prompt progress, available
        sequence slots, and the prefill token budget. Requested output
        lengths are intentionally ignored because a real engine does not
        know them before generation.
        """
        if not prefilling:
            return [], []
        batch: list[_ActiveSeq] = []
        chunk_sizes: list[int] = []
        token_count = 0
        dense_leftpad = (
            self._supports_padded_initial_prefill
            and prefilling[0].prefill_pos == 0
        )
        dense_len = self._next_chunk_size(prefilling[0]) if dense_leftpad else 0
        for seq in prefilling:
            if len(batch) >= self.max_num_seqs:
                break
            chunk_size = self._next_chunk_size(seq)
            if dense_leftpad:
                if seq.prefill_pos != 0:
                    break
                if batch and (len(batch) + 1) * dense_len > self.max_prefill_tokens:
                    break
            else:
                if batch and token_count + chunk_size > self.max_prefill_tokens:
                    break
            batch.append(seq)
            chunk_sizes.append(chunk_size)
            token_count += chunk_size
        return batch, chunk_sizes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        collect_logits: bool = False,
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
        if collect_logits:
            raise NotImplementedError("collect_logits is not yet supported by FLAEngine.")

        sp_list = (sampling_params if isinstance(sampling_params, list)
                   else [sampling_params] * len(prompts))
        seed = sp_list[0].seed
        if seed is not None:
            self._set_seeds(seed)

        # Fresh slot cache per generate() call. Old caches would leak
        # into the next set of prompts otherwise.
        self._reset_slot_cache()

        eos = self.tokenizer.eos_token_id
        all_seqs: list[_ActiveSeq] = []
        for i, p in enumerate(prompts):
            ids = p if isinstance(p, list) else self.tokenizer.encode(p)
            sp = sp_list[i]
            all_seqs.append(_ActiveSeq(
                seq_id=i,
                prompt_ids=list(ids),
                sampling=sp,
                max_tokens=sp.max_tokens,
                ignore_eos=sp.ignore_eos,
            ))

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=len(all_seqs), desc="FLAEngine prompts")
        finished_count = 0

        # Schedule by prompt length, using only information known at
        # admission time. Recurrent prefill cannot use padded varlen
        # batches in this local model path, so grouping long prompts
        # first maximizes full-size chunk batches and reduces the small
        # tail-fragmentation that real WildChat prompts otherwise cause.
        waiting: deque[_ActiveSeq] = deque(
            sorted(all_seqs, key=lambda s: len(s.prompt_ids), reverse=True)
        )
        # In-progress prefill (not yet sampled their first token)
        prefilling: list[_ActiveSeq] = []
        # Decoding (have at least one generated token, still under max_tokens)
        running: list[_ActiveSeq] = []

        def _admit() -> None:
            while waiting and (len(prefilling) + len(running)) < self.max_num_seqs:
                seq = waiting.popleft()
                seq.slot_id = self._acquire_slot()
                prefilling.append(seq)

        _admit()

        # Main loop: each iteration runs at most one BATCHED prefill chunk
        # (over all currently-prefilling seqs that want the same chunk
        # size) and one B=N decode step over running seqs. New seqs that
        # finish prefill are added to ``running`` immediately so they
        # decode on the next iteration. Batching prefill across all
        # waiting prompts is the difference between B=1 prompt-by-prompt
        # prefill (compute-light, memory-bandwidth-bound) and a single
        # B=k prefill that saturates the GPU's tensor cores.
        while prefilling or running:
            # ---- prefill: gather a chunk-aligned batch and run ONE forward
            if prefilling:
                prefilling.sort(key=self._next_chunk_size, reverse=True)
                batch, chunk_sizes = self._next_prefill_batch(prefilling)
                results = self._prefill_chunk_batched(batch, chunk_sizes)
                # Sample first tokens for completed seqs in one batched
                # argmax / multinomial call.
                completed: list[_ActiveSeq] = []
                completed_logits: list[torch.Tensor] = []
                for seq, logits in zip(batch, results):
                    if logits is not None:
                        # logits is already [V] (last-token-only from
                        # num_logits_to_keep=1)
                        completed.append(seq)
                        completed_logits.append(logits)
                if completed:
                    last_logits = torch.stack(completed_logits, dim=0)
                    first_tokens = self._sample_batch(last_logits, completed)
                    for seq, tok in zip(completed, first_tokens):
                        seq.generated_ids.append(tok)
                        if not seq.ignore_eos and eos is not None and tok == eos:
                            seq.finished = True
                        elif len(seq.generated_ids) >= seq.max_tokens:
                            seq.finished = True
                # Now remove all completed seqs from the prefilling head
                # (they form a contiguous prefix because _next_prefill_batch
                # gave us the head, and EVERYTHING in batch advanced by
                # chunk_size). Seqs whose prefill is still mid-prompt stay
                # in prefilling for the next iter; complete ones move to
                # running (or finish).
                for seq in batch:
                    if seq.prefill_pos < len(seq.prompt_ids):
                        # Multi-chunk prompt; stays in prefilling
                        continue
                    prefilling.remove(seq)
                    if seq.finished:
                        finished_count += 1
                        self._release_slot(seq)
                        if pbar is not None:
                            pbar.update(1)
                    else:
                        running.append(seq)
                _admit()

            # ---- decode: one step over all running seqs
            if running:
                tokens = self._decode_step(running)
                still_running: list[_ActiveSeq] = []
                finished_this_step: list[_ActiveSeq] = []
                for seq, tok in zip(running, tokens):
                    seq.generated_ids.append(tok)
                    if not seq.ignore_eos and eos is not None and tok == eos:
                        seq.finished = True
                    elif len(seq.generated_ids) >= seq.max_tokens:
                        seq.finished = True
                    if seq.finished:
                        finished_this_step.append(seq)
                        finished_count += 1
                        if pbar is not None:
                            pbar.update(1)
                    else:
                        still_running.append(seq)
                # If the running set changed, flush the live cache once
                # before releasing the freed slots; this preserves the
                # survivors' decode state (they'd otherwise lose it when
                # the live cache is reset by _release_slot).
                if finished_this_step:
                    self._flush_live()
                    for seq in finished_this_step:
                        self._release_slot(seq)
                running = still_running
                _admit()

        if pbar is not None:
            pbar.close()

        outputs: list[GenerationOutput] = []
        for seq in all_seqs:
            text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
            prompt_text = prompts[seq.seq_id] if isinstance(prompts[seq.seq_id], str) else ""
            outputs.append(GenerationOutput(
                prompt=prompt_text,
                generated_text=text,
                token_ids=list(seq.generated_ids),
            ))
        return outputs
