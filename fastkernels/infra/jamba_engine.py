"""Jamba inference engine.

Single-rank serving engine for AI21Labs Jamba (triple-hybrid
Transformer + Mamba-1 + sparse MoE).  Mirrors the project's other LLM
engines (:class:`infra.engine.LlamaEngine`,
:class:`infra.fla_engine.FLAEngine`, ``LlamaEngine._generate_mamba``):
**continuous-batching scheduler with `Sequence` + `BlockManager` for
paged KV + a per-seq Mamba state-slot pool.**  Each iteration of the
main loop admits as many waiting prompts as fit (slot + block pool +
``max_num_seqs`` cap), runs one batched prefill kernel call over any
prefilling sequences, then runs one batched decode kernel call over
the running sequences.  Sequences finish via EOS or ``max_tokens``;
their slots and blocks are released back to the pool and the next
admission picks them up.

Layout (matches vLLM / `LlamaAttention`'s convention):

  * Pipeline is **flat-varlen** ``[N, hidden]`` end-to-end.
    ``input_ids`` is flat ``[N]`` int64; ``positions`` is flat ``[N]``;
    L3 / L4 layers thread ``hidden_states`` and ``residual`` through
    unchanged.
  * **Paged KV cache** for the 4 attention layers, allocated as one
    global ``[2, num_attn_layers, num_blocks, page_size, num_kv_heads,
    head_dim]`` (NHD on Hopper) or ``[2, num_attn_layers, num_blocks,
    num_kv_heads, page_size, head_dim]`` (HND on Blackwell).  Each
    :class:`L2.attention_impl.Attention` gets bound to its slice via
    ``module.k_cache = kv_cache[0, i]`` (same as
    ``LlamaEngine.allocate_kv_cache``).  Blocks are partitioned by a
    ``BlockManager`` -- per-seq ``block_table`` lists are populated at
    admit time and may grow during decode (``may_append``).
  * **Mamba state slots** for the 28 Mamba layers: per-layer slabs of
    shape ``[num_slots, intermediate, K-1]`` (conv) and ``[num_slots,
    intermediate, ssm_state_size]`` (ssm), with ``num_slots ==
    max_num_seqs``.  Each ``Sequence`` is assigned a ``state_slot``
    on admission and uses it across all Mamba layers; freed on
    finish.  Same pattern as ``MambaStateManager`` in
    ``infra.engine`` but without the multi-rank SHM dance (this
    engine is single-GPU).
  * Per-step state is published on the global ``Context`` via
    ``set_jamba_context`` -- standard fields ``slot_mapping`` /
    ``block_tables`` / ``context_lens`` / ``cu_seqlens_q`` /
    ``cu_seqlens_k`` / ``mamba_state`` / ``mamba_metadata``.

Scope of continuous batching: **phase-pure** (each step is either a
batched prefill or a batched decode, never mixed).  This matches
:class:`FLAEngine`'s scheduler.  vLLM's chunked prefill goes one step
further (mixed prefill + decode in a single forward pass via
per-token metadata splits inside the Attention / Mamba kernels);
that is a meaningful structural piece for prefill-heavy workloads
where waiting prompts can be admitted *during* an in-flight decode
step, but it requires per-token kernel dispatch logic in the L2
modules and is out of scope for this iteration.  Documented as a
follow-up.

Tensor parallel: NOT supported -- single-GPU.  Open Jamba models
(tiny-dev = 318M, v0.1 = 52B) fit on a B200.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .context import (
    auto_register_no_compile_layers,
    get_attn_backend_config,
    register_no_compile_layers,
    reset_context,
    set_jamba_context,
)


# ---------------------------------------------------------------------------
# Re-exported / locally defined dataclasses.
#
# We don't ``from .engine import Sequence, BlockManager, SamplingParams,
# GenerationOutput`` because ``infra.engine`` transitively imports the
# whole model zoo (vision encoders, MLA, encoder-decoder, ...) and
# pulls in heavy deps like the standalone ``flash_attn`` package that
# JambaEngine has no use for.  FLAEngine takes the same approach: it
# defines its own per-request bookkeeping struct (``_ActiveSeq``) for
# this same reason, so "consistency with the project pattern" means
# *pattern-level* (waiting/running deques + slot pools + admit/release
# loop), not *class-level* identity.  The dataclasses below match the
# ones in ``infra.engine`` field-for-field so callers can use either
# interchangeably and the scheduler logic is a 1-to-1 translation of
# what LlamaEngine / `_generate_mamba` / FLAEngine do.
# ---------------------------------------------------------------------------
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = None
    ignore_eos: bool = False


@dataclass
class GenerationOutput:
    prompt: str
    generated_text: str
    token_ids: list[int]
    logits_history: list | None = None


from enum import Enum


class SeqStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


# Block size for the paged KV cache.  Matches
# ``infra.context.AttnBackendConfig.block_size`` for the auto-detected
# backend (16 for TRTLLM-gen on Blackwell, 256 for FA3 elsewhere) --
# we read it from the config inside ``__init__`` so the engine state
# matches whatever ``L2.attention_impl.Attention`` will dispatch to.


class Sequence:
    """Per-request bookkeeping.  Field-compatible with
    ``infra.engine.Sequence`` for the slice the JambaEngine scheduler
    actually uses (``token_ids``, ``generated_ids``, ``block_table``,
    ``num_computed_tokens``, ``state_slot``, ``status``,
    ``max_tokens``, ``ignore_eos``).
    """

    _next_id = 0

    def __init__(
        self, prompt_ids: list[int],
        max_tokens: int = 512,
        ignore_eos: bool = False,
    ):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1
        self.prompt_ids = list(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.generated_ids: list[int] = []
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.block_table: list[int] = []
        self.status = SeqStatus.WAITING
        self.num_computed_tokens: int = 0
        # Mamba slot index (None for non-mamba models).
        self.state_slot: int | None = None

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.generated_ids.append(token_id)


class BlockManager:
    """Free-block pool with deque-based allocation.  Field-compatible
    with ``infra.engine.BlockManager`` for the methods the JambaEngine
    scheduler uses (``reset``, ``can_allocate_n``, ``allocate_n``,
    ``deallocate``).
    """

    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def reset(self) -> None:
        self.free_block_ids = deque(range(self._num_blocks))

    def can_allocate_n(self, n_blocks: int) -> bool:
        return len(self.free_block_ids) >= n_blocks

    def allocate_n(self, seq: Sequence, n_blocks: int) -> None:
        for _ in range(n_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def deallocate(self, seq: Sequence) -> None:
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()


__all__ = ["JambaEngine", "SamplingParams", "GenerationOutput"]


# ---------------------------------------------------------------------------
# Mamba metadata (per-batch, per-step).  Mirrors the project's existing
# ``Mamba1AttentionMetadata`` pattern; the L2 mamba mixer reads
# ``conv_states`` / ``ssm_states`` / ``cache_indices`` plus
# ``query_start_loc`` (prefill) / ``has_initial_state`` (prefill) and
# runs the flat-varlen vendor kernels.
# ---------------------------------------------------------------------------
@dataclass
class JambaMambaMetadata:
    """Per-batch Mamba metadata.  Supports homogeneous prefill, homogeneous
    decode, and mixed prefill+decode batches (the latter for chunked-prefill
    mid-decode steps -- mirrors ``infra.engine.run_mamba_mixed``'s metadata).

    Homogeneous paths use ``cache_indices`` (per-seq Mamba slot for the
    one phase) plus ``query_start_loc`` / ``has_initial_state`` for the
    prefill case.

    Mixed path (``set_jamba_context(is_mixed=True, ...)``) uses
    ``state_indices_p`` / ``state_indices_d`` plus ``query_start_loc`` /
    ``has_initial_state`` referring to the prefill subset only.
    """
    conv_states: list[torch.Tensor]
    ssm_states: list[torch.Tensor]
    cache_indices: torch.Tensor          # int32 [num_seqs] -- homogeneous slot index
    is_decode: bool = True
    query_start_loc: torch.Tensor | None = None    # int32 [num_prefills+1] (prefill / mixed-prefill subset)
    has_initial_state: torch.Tensor | None = None  # bool [num_prefills] (prefill / mixed-prefill subset)
    pad_mask_flat: torch.Tensor | None = None      # legacy; unused with flat layout
    # Mixed-batch fields (None unless ctx.is_mixed=True).
    state_indices_p: torch.Tensor | None = None    # int32 [num_prefills]
    state_indices_d: torch.Tensor | None = None    # int32 [num_decodes]


# ---------------------------------------------------------------------------
# Mamba state slot pool.  Each Sequence holds a ``state_slot`` index
# into this pool; the per-layer slabs stay live for the whole engine
# lifetime (allocated once at __init__) and slots are recycled as
# sequences finish.
# ---------------------------------------------------------------------------
class _MambaSlotPool:
    """Persistent per-layer Mamba ``(conv_state, ssm_state)`` slabs with
    a free-slot deque.  Single-rank counterpart of
    ``infra.engine.MambaStateManager``.

    Allocate one slot per active sequence; deallocate on
    ``Sequence.preempt`` or finish.  ``zero_slot`` resets a slot to
    all-zero between reuses so a new sequence's recurrence starts from
    a clean state.
    """

    def __init__(
        self,
        max_num_seqs: int,
        num_mamba_layers: int,
        intermediate_size: int,
        ssm_state_size: int,
        conv_kernel: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_num_seqs = max_num_seqs
        self.num_mamba_layers = num_mamba_layers
        K_minus_1 = max(conv_kernel - 1, 1)

        # Per-layer slabs.  conv_state stride convention follows the
        # vLLM kernel contract (``stride(intermediate) == 1``): allocate
        # as ``[num_slots, K-1, intermediate]`` and transpose the last
        # two dims to expose the kernel-required layout.
        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []
        for _ in range(num_mamba_layers):
            raw_conv = torch.zeros(
                max_num_seqs, K_minus_1, intermediate_size,
                dtype=dtype, device=device,
            )
            self.conv_states.append(raw_conv.transpose(-1, -2))
            self.ssm_states.append(torch.zeros(
                max_num_seqs, intermediate_size, ssm_state_size,
                dtype=dtype, device=device,
            ))

        self._free: deque[int] = deque(range(max_num_seqs))
        self._in_use: set[int] = set()

    def has_free(self) -> bool:
        return len(self._free) > 0

    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        slot = self._free.popleft()
        self._in_use.add(slot)
        self._zero_slot(slot)
        return slot

    def free(self, slot: int) -> None:
        if slot in self._in_use:
            self._in_use.remove(slot)
            self._zero_slot(slot)
            self._free.append(slot)

    def reset(self) -> None:
        """Return all slots to the free pool (called between generate() calls)."""
        for slot in list(self._in_use):
            self.free(slot)

    def _zero_slot(self, slot: int) -> None:
        for cs in self.conv_states:
            cs[slot].zero_()
        for ss in self.ssm_states:
            ss[slot].zero_()


# ---------------------------------------------------------------------------
# Decode-step CUDA graph entry.  All tensor identities are stable
# across replays; callers mutate values in-place between replays.
# ---------------------------------------------------------------------------
@dataclass
class _JambaDecodeGraph:
    bucket_size: int
    graph: torch.cuda.CUDAGraph
    step_input_ids: torch.Tensor    # [B]                      int64
    step_positions: torch.Tensor    # [B]                      int64
    slot_mapping: torch.Tensor      # [B]                      int64
    context_lens: torch.Tensor      # [B]                      int32
    block_tables: torch.Tensor      # [B, max_blocks_per_seq]  int32
    cache_indices: torch.Tensor     # [B]                      int32 (Mamba slot map)
    next_tokens: torch.Tensor       # [B]                      int64 (output)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class JambaEngine:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        max_num_seqs: int = 32,
        max_model_len: int = 4096,
        max_num_batched_tokens: int = 16384,
        trust_remote_code: bool = True,
    ):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from ..tasks.baseline.L4.jamba import JambaConfig, JambaForCausalLM

        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        # Per-step token budget for chunked prefill.  Mirrors vLLM's
        # ``max_num_batched_tokens`` (default 16384) and the chunked-
        # prefill scheduler in ``infra.engine`` /
        # ``infra.fla_engine``.  No prefill kernel call ever processes
        # more than this many flat tokens; long prompts are split
        # across multiple steps with ``has_initial_state`` continuing
        # the conv/SSM state, and short-prompt batches are admitted
        # multiple-at-a-time within the budget.
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = torch.device(device)
        self.dtype = dtype
        self._set_seeds(seed)

        # ------------------------------------------------------------------
        # Model + tokenizer
        # ------------------------------------------------------------------
        model_path = snapshot_download(
            model_name, allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        )
        self.model_path = model_path
        self.config = JambaConfig.from_pretrained(model_path)
        self.config.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"  [JambaEngine] Loading config from {model_path}")
        print(f"  [JambaEngine] Architecture: "
              f"L={self.config.num_hidden_layers}, "
              f"hidden={self.config.hidden_size}, "
              f"attn_layers={self.config.attn_layer_offset}/"
              f"{self.config.attn_layer_period}, "
              f"moe_layers={self.config.expert_layer_offset}/"
              f"{self.config.expert_layer_period}, "
              f"experts={self.config.num_experts}, "
              f"top_k={self.config.num_experts_per_tok}")

        with torch.device("cpu"):
            self.model = JambaForCausalLM(self.config)
        n_loaded = self.model.load_weights(model_path)
        print(f"  [JambaEngine] Loaded {n_loaded} weight tensors")

        # GPU + dtype (Mamba A must be fp32 for the SSM kernel).
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        for layer in self.model.model.layers:
            mamba = getattr(layer, "mamba", None)
            if mamba is not None:
                mamba.A.data = mamba.A.data.float()
        torch.cuda.synchronize()

        cfg = self.config
        self._mamba_intermediate = cfg.mamba_expand * cfg.hidden_size
        self._mamba_d_state = cfg.mamba_d_state
        self._mamba_conv_kernel = cfg.mamba_d_conv
        self._n_mamba_layers = len(self.model.model.mamba_layer_indices)
        self._n_attn_layers = len(self.model.model.attention_layer_indices)
        self._head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._n_kv_heads = cfg.num_key_value_heads

        # ------------------------------------------------------------------
        # Attention backend & paged-cache layout (auto-detected; matches
        # the L2 dispatcher in JambaAttention).
        # ------------------------------------------------------------------
        attn_cfg = get_attn_backend_config()
        self._page_size = attn_cfg.block_size
        self._kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm

        # Block budget: enough blocks for every seq in the bucket to
        # cover the full max_model_len prompt + decode horizon.
        # Conservative -- LlamaEngine sizes this dynamically from
        # available GPU memory; we round up to keep the engine simple
        # since the open Jamba models leave plenty of memory headroom.
        self._max_blocks_per_seq = (
            (max_model_len + self._page_size - 1) // self._page_size
        )
        total_blocks = self.max_num_seqs * self._max_blocks_per_seq
        self._num_blocks = total_blocks

        # Allocate the global paged KV cache and bind per-layer slices.
        if self._kv_layout == "HND":
            self._kv_cache = torch.zeros(
                2, self._n_attn_layers, total_blocks,
                self._n_kv_heads, self._page_size, self._head_dim,
                dtype=dtype, device=self.device,
            )
        else:
            self._kv_cache = torch.zeros(
                2, self._n_attn_layers, total_blocks,
                self._page_size, self._n_kv_heads, self._head_dim,
                dtype=dtype, device=self.device,
            )

        # Bind k_cache/v_cache slices to each L2 ``Attention`` instance.
        # Walk model.modules() in order; the inner Attention sits at
        # ``model.layers.{i}.self_attn.attn`` for attention decoder
        # layers and is identifiable by the ``k_cache`` / ``v_cache``
        # attributes the Attention class sets in __init__.
        attn_modules = []
        no_compile_layers: dict[str, "torch.nn.Module"] = {}
        for name, mod in self.model.named_modules():
            if (hasattr(mod, "k_cache") and hasattr(mod, "v_cache")
                    and type(mod).__name__ == "Attention"):
                attn_modules.append(mod)
                no_compile_layers[name] = mod
        assert len(attn_modules) == self._n_attn_layers, (
            f"Expected {self._n_attn_layers} Attention instances, found "
            f"{len(attn_modules)}; check L2 JambaAttention wiring."
        )
        for i, attn in enumerate(attn_modules):
            attn.k_cache = self._kv_cache[0, i]
            attn.v_cache = self._kv_cache[1, i]
            attn._layer_name = next(
                n for n, m in no_compile_layers.items() if m is attn
            )
        register_no_compile_layers(no_compile_layers)
        auto_register_no_compile_layers(self.model)

        # Share TRTLLM workspace across all Attention modules.  Mirrors
        # :meth:`infra.engine.LlamaEngine._share_trtllm_workspace`:
        # without this, each TRTLLMDecode and TRTLLMPrefill allocates
        # its own 512 MB workspace -- 4 attn layers x 2 ops x 512 MB =
        # 4 GB wasted on duplicate scratch buffers.  The shared tensor
        # stays alive because each ``attn.{decode_op,prefill_op}._workspace``
        # holds a ref after ``set_trtllm_workspace``.
        if self._use_trtllm:
            shared_workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device=self.device,
            )
            for attn in attn_modules:
                if hasattr(attn, "set_trtllm_workspace"):
                    attn.set_trtllm_workspace(shared_workspace)
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Block manager (paged KV) + Mamba state slot pool.
        # ------------------------------------------------------------------
        self.block_manager = BlockManager(num_blocks=total_blocks)
        self.mamba_pool = _MambaSlotPool(
            max_num_seqs=max_num_seqs,
            num_mamba_layers=self._n_mamba_layers,
            intermediate_size=self._mamba_intermediate,
            ssm_state_size=self._mamba_d_state,
            conv_kernel=self._mamba_conv_kernel,
            dtype=dtype,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # Async D2H pipeline for decode tokens.  Mirrors the Mamba engine's
        # ``run_mamba_decode_fast_async`` / ``_wait_async_mamba_tokens``
        # pattern (see ``infra.engine`` and the Mamba section of the
        # README): after a captured decode graph replays, the GPU's
        # ``next_tokens[:n]`` is copied to a pinned host buffer on a
        # SEPARATE stream, with an event recorded; the caller waits on
        # that event before reading the buffer to a Python list.
        #
        # The two wins from this pattern:
        #   1. The non-blocking copy can overlap with the next graph's
        #      kernel launches on the main stream.
        #   2. Avoids the explicit ``torch.cuda.synchronize()`` in the
        #      old hot loop (which forced a global wait).
        #
        # Effective on tiny-dev (~1 ms / step out of ~3 ms wall = ~30%
        # reduction in per-step latency); modest on v0.1 (per-step is
        # ~36 ms GPU-heavy so the relative win is small).  Either way,
        # consistency with the project's async-decode pattern.
        self._copy_stream = torch.cuda.Stream()
        self._copy_event = torch.cuda.Event()
        self._pinned_tokens = torch.empty(
            max_num_seqs, dtype=torch.long, pin_memory=True,
        )

        # ------------------------------------------------------------------
        # CUDA graph capture (multi-bucket).  Captures the pure-decode
        # forward at several batch sizes ([1, 2, 4, ..., max_num_seqs])
        # so the scheduler can dispatch each step to the smallest
        # bucket >= the live running batch size.  Mirrors
        # :meth:`LlamaEngine.capture_cudagraph`'s shared-mempool
        # largest-first strategy.
        # ------------------------------------------------------------------
        self._use_cuda_graphs = (
            os.environ.get("FASTKERNELS_JAMBA_CUDA_GRAPHS", "1") not in ("0", "false", "False")
        )
        self._use_compile = (
            os.environ.get("FASTKERNELS_JAMBA_COMPILE", "0") not in ("0", "false", "False")
        )
        # Decode bucket schedule.  vLLM uses [1, 2, 4, 8, 16, 24, 32,
        # 40, ...]; we cap at max_num_seqs.  Override via
        # ``FASTKERNELS_JAMBA_BUCKETS=1,2,4,8,16,32`` if you need a denser
        # or sparser schedule for a specific workload.
        env_buckets = os.environ.get("FASTKERNELS_JAMBA_BUCKETS")
        if env_buckets:
            buckets = sorted({int(x) for x in env_buckets.split(",") if x.strip()})
        else:
            base = [1, 2, 4]
            base += list(range(8, max_num_seqs + 1, 8))
            buckets = sorted(set(b for b in base if b <= max_num_seqs))
        if max_num_seqs not in buckets:
            buckets.append(max_num_seqs)
            buckets = sorted(set(buckets))
        self._decode_buckets = buckets
        # Pre-create a shared CUDA-graph mempool so all bucket captures
        # share one address space (avoids ``cudaErrorIllegalAddress``
        # when smaller-bucket replays alias larger-bucket pool blocks).
        self._cuda_graph_mempool_id = (
            torch.cuda.graph_pool_handle() if self._use_cuda_graphs else None
        )
        self._compiled_decode_step = None
        self._decode_graphs: dict[int, _JambaDecodeGraph] = {}
        self._decode_static_buffers: dict[int, dict] = {}
        if self._use_cuda_graphs:
            print(
                f"  [JambaEngine] Capturing decode graphs at "
                f"B={self._decode_buckets} (paged KV: {total_blocks} blocks "
                f"x {self._page_size} = {total_blocks * self._page_size} "
                f"token slots, {self._n_attn_layers} attn layers, "
                f"{self._kv_layout} layout)"
            )
            # Capture LARGEST bucket first so subsequent smaller-bucket
            # captures see a memory layout consistent with what their
            # tensors will be in at runtime (LlamaEngine pattern).
            for bucket in reversed(self._decode_buckets):
                self._capture_decode_graph(bucket)

    # ------------------------------------------------------------------
    # Random seeds
    # ------------------------------------------------------------------
    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Decode-graph static buffers + capture.
    # ------------------------------------------------------------------
    def _alloc_decode_buffers(self, B: int) -> dict:
        """Allocate the static-identity tensors a B-bucket decode graph
        reads from.  Tensors are reused across replays; the host loop
        mutates their values in-place between replays.
        """
        if B in self._decode_static_buffers:
            return self._decode_static_buffers[B]
        device = self.device
        bps = self._max_blocks_per_seq
        bufs = {
            "step_input_ids": torch.zeros(B, dtype=torch.long, device=device),
            "step_positions": torch.zeros(B, dtype=torch.long, device=device),
            "slot_mapping": torch.zeros(B, dtype=torch.long, device=device),
            "context_lens": torch.zeros(B, dtype=torch.int32, device=device),
            "block_tables": torch.zeros(B, bps, dtype=torch.int32, device=device),
            "cache_indices": torch.zeros(B, dtype=torch.int32, device=device),
            "next_tokens": torch.zeros(B, dtype=torch.long, device=device),
        }
        self._decode_static_buffers[B] = bufs
        return bufs

    def _capture_decode_graph(self, B: int) -> _JambaDecodeGraph:
        if B in self._decode_graphs:
            return self._decode_graphs[B]
        bufs = self._alloc_decode_buffers(B)
        max_context_len = self._max_blocks_per_seq * self._page_size

        mamba_meta = JambaMambaMetadata(
            conv_states=self.mamba_pool.conv_states,
            ssm_states=self.mamba_pool.ssm_states,
            cache_indices=bufs["cache_indices"],
            is_decode=True,
        )

        # Optional: torch.compile the inner JambaModel + lm_head + argmax.
        # OFF by default because Inductor's fused elementwise paths
        # (RMSNorm + residual + SwiGLU) drift from vLLM's hand-written
        # CUDA kernels by ~1e-3 per layer in bf16 and tank match-tokens
        # vs vLLM (see README).  The eager path uses the same vLLM
        # ``_C.fused_add_rms_norm`` / ``_C.silu_and_mul`` kernels vLLM
        # itself uses, so bf16 numerics are bit-identical.
        if self._use_compile and self._compiled_decode_step is None:
            inner = self.model.model
            lm_head = self.model.lm_head

            def _forward_for_compile(input_ids, positions):
                hidden = inner(input_ids, positions)
                logits = lm_head(hidden)
                return logits.argmax(dim=-1)

            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                self.config.num_hidden_layers * 2 + 32,
            )
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit,
                self.config.num_hidden_layers * 4 + 64,
            )
            if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
            self._compiled_decode_step = torch.compile(
                _forward_for_compile,
                mode="default",
                dynamic=False,
                fullgraph=False,
            )

        def _decode_step():
            set_jamba_context(
                is_prefill=False,
                slot_mapping=bufs["slot_mapping"],
                context_lens=bufs["context_lens"],
                block_tables=bufs["block_tables"],
                max_context_len=max_context_len,
                mamba_metadata=mamba_meta,
            )
            try:
                if self._compiled_decode_step is not None:
                    tok = self._compiled_decode_step(
                        bufs["step_input_ids"], bufs["step_positions"],
                    )
                else:
                    hidden = self.model(
                        bufs["step_input_ids"], bufs["step_positions"],
                    )
                    logits = self.model.compute_logits(hidden)
                    tok = logits.argmax(dim=-1)
                bufs["next_tokens"].copy_(tok)
            finally:
                reset_context()

        # Warmup outside the graph stream so allocator state settles.
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _decode_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._cuda_graph_mempool_id):
            _decode_step()

        entry = _JambaDecodeGraph(
            bucket_size=B,
            graph=graph,
            step_input_ids=bufs["step_input_ids"],
            step_positions=bufs["step_positions"],
            slot_mapping=bufs["slot_mapping"],
            context_lens=bufs["context_lens"],
            block_tables=bufs["block_tables"],
            cache_indices=bufs["cache_indices"],
            next_tokens=bufs["next_tokens"],
        )
        self._decode_graphs[B] = entry
        return entry

    def _pick_bucket(self, n_active: int) -> int:
        for b in self._decode_buckets:
            if b >= n_active:
                return b
        return self.max_num_seqs

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_one(logits_row: torch.Tensor, sp: SamplingParams) -> int:
        if sp.temperature == 0.0:
            return int(logits_row.argmax().item())
        scaled = logits_row.float() / sp.temperature
        if sp.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum - probs >= sp.top_p
            sorted_logits[mask] = float("-inf")
            scaled = scaled.scatter(0, sorted_idx, sorted_logits)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------------
    # generate() -- continuous-batching scheduler.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
        if isinstance(sampling_params, SamplingParams):
            sp_list = [sampling_params] * len(prompt_token_ids)
        else:
            sp_list = list(sampling_params)
        assert len(sp_list) == len(prompt_token_ids)

        seed = sp_list[0].seed
        if seed is not None:
            self._set_seeds(seed)

        eos = self.tokenizer.eos_token_id
        # Reset state pools for a fresh generate() call.
        self.block_manager.reset()
        self.mamba_pool.reset()

        # Build sequences in input order.  Sequence carries
        # block_table, num_computed_tokens, generated_ids, state_slot,
        # status -- all the bookkeeping fields the scheduler needs.
        all_seqs: list[Sequence] = []
        seq_idx_for_id: dict[int, int] = {}
        seq_sp: dict[int, SamplingParams] = {}
        for i, ids in enumerate(prompt_token_ids):
            sp = sp_list[i]
            seq = Sequence(
                list(ids),
                max_tokens=sp.max_tokens,
                ignore_eos=sp.ignore_eos,
            )
            all_seqs.append(seq)
            seq_idx_for_id[id(seq)] = i
            seq_sp[id(seq)] = sp

        outputs: list[list[int]] = [[] for _ in range(len(all_seqs))]

        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=len(all_seqs), desc="fastkernels Jamba")
        else:
            pbar = None

        # Schedule: longer prompts first.  Same heuristic as FLAEngine:
        # batches its full-size prefill chunks and minimises tail
        # fragmentation on real WildChat-shape data.
        waiting: deque[Sequence] = deque(
            sorted(all_seqs, key=lambda s: len(s.prompt_ids), reverse=True),
        )
        # Three deques: waiting (not yet admitted), prefilling (admitted,
        # mid-prompt), running (prompt fully prefilled, decoding).  Same
        # pattern as ``LlamaEngine.generate`` / ``FLAEngine.generate`` /
        # ``LlamaEngine._generate_mamba``.
        prefilling: list[Sequence] = []
        running: list[Sequence] = []

        def _admit() -> list[Sequence]:
            """Take from ``waiting`` while there's room in slot pool,
            block pool, and ``max_num_seqs`` cap.  Allocates lifetime
            blocks at admit time -- decode-phase block table doesn't
            grow.  Mirrors ``FLAEngine._admit``'s eager-allocate
            pattern.
            """
            admitted: list[Sequence] = []
            while (waiting and
                   len(prefilling) + len(running) + len(admitted)
                   < self.max_num_seqs):
                if not self.mamba_pool.has_free():
                    break
                seq = waiting[0]
                lifetime_tokens = len(seq.prompt_ids) + seq.max_tokens
                lifetime_blocks = (
                    (lifetime_tokens + self._page_size - 1) // self._page_size
                )
                if lifetime_blocks > self._num_blocks:
                    raise RuntimeError(
                        f"Sequence requires {lifetime_blocks} blocks "
                        f"({lifetime_tokens} tokens) but the global pool only "
                        f"holds {self._num_blocks}.  Increase "
                        f"max_model_len or reduce max_num_seqs to fit."
                    )
                if not self.block_manager.can_allocate_n(lifetime_blocks):
                    break
                waiting.popleft()
                seq.state_slot = self.mamba_pool.allocate()
                self.block_manager.allocate_n(seq, lifetime_blocks)
                seq.status = SeqStatus.RUNNING  # placeholder; truly RUNNING
                                                 # only after prefill completes
                admitted.append(seq)
            return admitted

        def _next_prefill_chunks(
            prefilling_local: list[Sequence],
        ) -> list[tuple[Sequence, int]]:
            """Pick this step's chunk for each prefilling seq, capped
            at ``max_num_batched_tokens`` total.  Mirrors FLAEngine's
            ``_next_prefill_batch``; analogous to LlamaEngine's
            ``prepare_mixed_batch`` chunk-size selection.
            """
            chunks: list[tuple[Sequence, int]] = []
            tokens_used = 0
            for seq in prefilling_local:
                remaining = len(seq.prompt_ids) - seq.num_computed_tokens
                if remaining <= 0:
                    continue
                budget_left = self.max_num_batched_tokens - tokens_used
                if budget_left <= 0:
                    break
                chunk = min(remaining, budget_left)
                chunks.append((seq, chunk))
                tokens_used += chunk
            return chunks

        def _finalize_decode_token(seq, tok):
            """Apply a sampled decode token to ``seq``; return True if
            the seq is done (eos / max_tokens)."""
            seq.append_token(tok)
            seq.num_computed_tokens = len(seq)
            return (
                len(seq.generated_ids) >= seq.max_tokens
                or (not seq.ignore_eos and eos is not None and tok == eos)
            )

        while waiting or prefilling or running:
            # ---- admit fresh seqs from waiting into prefilling ----
            prefilling.extend(_admit())

            # ---- chunked prefill step ----
            if prefilling:
                chunks = _next_prefill_chunks(prefilling)
                chunk_logits, completed_mask = self._run_prefill_chunks(chunks)
                # Process each (seq, chunk_size) result.  Completed
                # seqs (whose prefill finished this step) get their
                # last-chunk logit sampled; others advance and stay
                # in prefilling.
                completed_seqs: list[Sequence] = []
                completed_logits: list[torch.Tensor] = []
                for i, ((seq, cs), is_done) in enumerate(zip(chunks, completed_mask)):
                    seq.num_computed_tokens += cs
                    if is_done:
                        completed_seqs.append(seq)
                        completed_logits.append(chunk_logits[i])

                # Sample first decode token for completed prefills.
                if completed_seqs:
                    stacked = torch.stack(completed_logits, dim=0)
                    if all(seq_sp[id(s)].temperature == 0.0 for s in completed_seqs):
                        first_toks = stacked.argmax(dim=-1).tolist()
                    else:
                        first_toks = [
                            self._sample_one(stacked[i], seq_sp[id(s)])
                            for i, s in enumerate(completed_seqs)
                        ]
                    finished_now: list[Sequence] = []
                    for seq, tok in zip(completed_seqs, first_toks):
                        if _finalize_decode_token(seq, tok):
                            finished_now.append(seq)
                        else:
                            running.append(seq)
                    for seq in finished_now:
                        self._finish_seq(seq, outputs, eos, seq_idx_for_id)
                        if pbar is not None:
                            pbar.update(1)
                # Drop completed seqs from prefilling.
                prefilling = [
                    s for s in prefilling
                    if s.num_computed_tokens < len(s.prompt_ids)
                ]

            # ---- decode step over running ----
            if running:
                tokens = self._run_decode_step(running)
                still_running: list[Sequence] = []
                finished_now = []
                for seq, tok in zip(running, tokens):
                    if _finalize_decode_token(seq, tok):
                        finished_now.append(seq)
                    else:
                        still_running.append(seq)
                for seq in finished_now:
                    self._finish_seq(seq, outputs, eos, seq_idx_for_id)
                    if pbar is not None:
                        pbar.update(1)
                running = still_running

        if pbar is not None:
            pbar.close()

        # Materialise outputs in input order.
        results: list[GenerationOutput] = []
        for tokens in outputs:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(GenerationOutput(
                prompt="", generated_text=text, token_ids=list(tokens),
            ))
        return results

    # ------------------------------------------------------------------
    # _finish_seq: release slot + blocks, capture final tokens.
    # ------------------------------------------------------------------
    def _finish_seq(
        self,
        seq: Sequence,
        outputs: list[list[int]],
        eos: int | None,
        seq_idx_for_id: dict[int, int],
    ) -> None:
        idx = seq_idx_for_id[id(seq)]
        # Trim trailing eos (callers don't include it in token_ids
        # by convention -- mirrors what the prior engine did).
        toks = list(seq.generated_ids)
        if (not seq.ignore_eos and eos is not None
                and toks and toks[-1] == eos
                and len(toks) > 1):
            # Keep eos in output -- bench's match-tokens metric
            # consumes the raw token sequence including the
            # final eos, so don't trim it here.
            pass
        outputs[idx] = toks
        if seq.state_slot is not None:
            self.mamba_pool.free(seq.state_slot)
            seq.state_slot = None
        self.block_manager.deallocate(seq)
        seq.status = SeqStatus.FINISHED

    # ------------------------------------------------------------------
    # _run_prefill_chunks: batched chunked-prefill step over a list of
    # (seq, chunk_size) pairs.  Each seq contributes ``chunk_size`` new
    # prompt tokens this step (positions ``[num_computed,
    # num_computed + chunk)``).  Mirrors ``LlamaEngine.prepare_mixed_batch``
    # 's chunked-prefill metadata builder.
    #
    # Mamba state continues across chunks via ``has_initial_state[i] =
    # (num_computed_i > 0)`` -- the vendored vLLM Mamba kernels read
    # the prior conv/SSM state at ``cache_indices[i]`` and update it
    # in place.  Paged-attn likewise sees the full K/V history (prior
    # chunks already wrote slots ``[0, num_computed)``; this chunk
    # writes ``[num_computed, num_computed + chunk)``).
    #
    # Returns the per-seq logits ONLY for seqs whose prefill COMPLETES
    # this step (i.e., ``num_computed_i + chunk_i == len(prompt_ids_i)``);
    # mid-prefill seqs return no logit (they keep prefilling next step).
    # The caller advances ``seq.num_computed_tokens`` and decides which
    # seqs promote to ``running`` based on this completion mask.
    # ------------------------------------------------------------------
    def _run_prefill_chunks(
        self,
        chunks: list[tuple[Sequence, int]],
    ) -> tuple[torch.Tensor, list[bool]]:
        device = self.device
        page_size = self._page_size
        B = len(chunks)

        # Per-seq chunk metadata.
        plens = np.array([cs for _, cs in chunks], dtype=np.int32)  # this step
        starts = np.array(
            [s.num_computed_tokens for s, _ in chunks], dtype=np.int32,
        )
        kvlens = starts + plens  # post-write K/V length per seq
        completed_mask = [
            (s.num_computed_tokens + cs) >= len(s.prompt_ids)
            for s, cs in chunks
        ]

        # Flat input_ids = concatenation of this step's chunks.
        flat_ids: list[int] = []
        for s, cs in chunks:
            start = s.num_computed_tokens
            flat_ids.extend(s.prompt_ids[start:start + cs])
        cu_q_np = np.zeros(B + 1, dtype=np.int32)
        np.cumsum(plens, out=cu_q_np[1:])
        cu_q = torch.from_numpy(cu_q_np).to(device)

        # cu_seqlens_k carries the FULL post-write K/V length per seq
        # (prior-chunk K/V already in cache + this chunk's K/V about to
        # be written).  This is what ``Attention._forward_pure``'s
        # paged-prefill kernel needs.  ``max_seqlen_q`` is per-step
        # chunk_size max; ``max_seqlen_k`` is total K-len max.
        cu_k_np = np.zeros(B + 1, dtype=np.int32)
        np.cumsum(kvlens, out=cu_k_np[1:])
        cu_k = torch.from_numpy(cu_k_np).to(device)

        # Per-token absolute positions (carry the seq's start offset).
        pos_np = np.concatenate([
            np.arange(s.num_computed_tokens,
                      s.num_computed_tokens + cs, dtype=np.int64)
            for s, cs in chunks
        ])
        positions = torch.from_numpy(pos_np).to(device)
        input_ids = torch.tensor(flat_ids, dtype=torch.long, device=device)

        # Per-seq block_tables.
        bps = self._max_blocks_per_seq
        bt_np = np.full((B, bps), -1, dtype=np.int32)
        for i, (s, _) in enumerate(chunks):
            bt_np[i, :len(s.block_table)] = s.block_table
        block_tables = torch.from_numpy(bt_np).to(device)

        # Per-token slot_mapping.  position-in-seq vector (absolute).
        seq_idx_np = np.repeat(np.arange(B, dtype=np.int64), plens)
        block_idxs = pos_np // page_size
        slot_in_blocks = pos_np % page_size
        slot_np = (
            bt_np[seq_idx_np, block_idxs].astype(np.int64) * page_size
            + slot_in_blocks
        )
        slot_mapping = torch.from_numpy(slot_np).to(device)

        # Mamba metadata.  ``has_initial_state[i] = (num_computed > 0)``
        # tells the kernel to continue from the prior chunk's state
        # rather than initialise to zero -- this is the load-bearing
        # bit for chunked Mamba prefill.
        cache_indices = torch.tensor(
            [s.state_slot for s, _ in chunks], dtype=torch.int32, device=device,
        )
        has_initial_state = torch.tensor(
            [s.num_computed_tokens > 0 for s, _ in chunks],
            dtype=torch.bool, device=device,
        )

        mamba_meta = JambaMambaMetadata(
            conv_states=self.mamba_pool.conv_states,
            ssm_states=self.mamba_pool.ssm_states,
            cache_indices=cache_indices,
            is_decode=False,
            query_start_loc=cu_q,
            has_initial_state=has_initial_state,
        )

        max_seqlen_q = int(plens.max())
        max_seqlen_k = int(kvlens.max())
        set_jamba_context(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            mamba_metadata=mamba_meta,
        )
        try:
            hidden = self.model(input_ids, positions)
            # Last-token hidden of THIS chunk per seq (cu_q[i+1]-1).
            last_idx = cu_q[1:].long() - 1
            last_hidden = hidden.index_select(0, last_idx)
            chunk_last_logits = self.model.compute_logits(last_hidden)
        finally:
            reset_context()
        return chunk_last_logits, completed_mask

    # ------------------------------------------------------------------
    # _run_decode_step: one decode step over ``running``.  Picks the
    # smallest CUDA-graph bucket >= len(running), pads block_tables /
    # context_lens / slot_mapping / cache_indices into the static
    # buffer slots [:n] (and -1 / 0 for the trailing pad slots), then
    # replays the captured graph.  Reads back next_tokens[:n].
    # ------------------------------------------------------------------
    def _run_decode_step(self, running: list[Sequence]) -> list[int]:
        n = len(running)
        if self._use_cuda_graphs:
            B = self._pick_bucket(n)
            graph = self._decode_graphs.get(B)
            if graph is None:
                graph = self._capture_decode_graph(B)
        else:
            graph = None
            B = n

        device = self.device
        page_size = self._page_size

        # Per-seq decode arrays for the active rows.  Last-token id
        # (input for this step), absolute position, paged slot for the
        # new K/V, post-write context_len, padded block_table row, and
        # Mamba state slot.
        ids_np = np.empty(n, dtype=np.int64)
        pos_np = np.empty(n, dtype=np.int64)
        slot_np = np.empty(n, dtype=np.int64)
        ctx_np = np.empty(n, dtype=np.int32)
        cache_idx_np = np.empty(n, dtype=np.int32)
        bps = self._max_blocks_per_seq
        bt_np = np.full((n, bps), -1, dtype=np.int32)
        for i, s in enumerate(running):
            ids_np[i] = s.last_token
            new_pos = len(s) - 1  # absolute position of the new token
            pos_np[i] = new_pos
            block_idx = new_pos // page_size
            slot_in_block = new_pos % page_size
            slot_np[i] = (
                int(s.block_table[block_idx]) * page_size + slot_in_block
            )
            ctx_np[i] = new_pos + 1  # post-write context length
            bt_np[i, :len(s.block_table)] = s.block_table
            cache_idx_np[i] = s.state_slot

        if graph is not None:
            # Pad to bucket B with the project-standard sentinels so
            # the captured graph runs cleanly on partial buckets.
            # Mirrors LlamaEngine._run_graph_from_numpy
            # (slot_mapping[pad]=-1, context_lens[pad]=0) and
            # _generate_mamba's _MAMBA_PAD_SLOT_ID = -1 (the vendored
            # vLLM Mamba kernels skip rows whose state_indices is -1).
            #
            # An earlier version of this method copied row 0's
            # block_table and cache_indices into the pad rows -- that
            # was a CORRECTNESS bug for the Mamba layer: multiple pad
            # rows pointing at the same Mamba slot would race-write
            # the real seq's state.  Using -1 sentinels skips them.
            bufs_input_ids = graph.step_input_ids
            bufs_positions = graph.step_positions
            bufs_slot_mapping = graph.slot_mapping
            bufs_context_lens = graph.context_lens
            bufs_block_tables = graph.block_tables
            bufs_cache_indices = graph.cache_indices

            full_ids = np.zeros(B, dtype=np.int64)
            full_ids[:n] = ids_np
            full_pos = np.zeros(B, dtype=np.int64)
            full_pos[:n] = pos_np
            # Paged-attn pad rows: slot=-1 (skip store), ctx_len=0
            # (kernel attends to nothing), block_table all-(-1)
            # (sentinel; never read because ctx_len=0).
            full_slot = np.full(B, -1, dtype=np.int64)
            full_slot[:n] = slot_np
            full_ctx = np.zeros(B, dtype=np.int32)
            full_ctx[:n] = ctx_np
            full_bt = np.full((B, bps), -1, dtype=np.int32)
            full_bt[:n] = bt_np
            # Mamba pad rows: cache_indices=-1 -- the vendored
            # causal_conv1d_update / selective_state_update kernels
            # skip rows whose state index is -1 (per
            # ``_MAMBA_PAD_SLOT_ID = -1`` precedent in
            # ``infra.engine``).
            full_cache_idx = np.full(B, -1, dtype=np.int32)
            full_cache_idx[:n] = cache_idx_np

            bufs_input_ids.copy_(torch.from_numpy(full_ids))
            bufs_positions.copy_(torch.from_numpy(full_pos))
            bufs_slot_mapping.copy_(torch.from_numpy(full_slot))
            bufs_context_lens.copy_(torch.from_numpy(full_ctx))
            bufs_block_tables.copy_(torch.from_numpy(full_bt))
            bufs_cache_indices.copy_(torch.from_numpy(full_cache_idx))

            # Replay graph + async D2H of the sampled tokens.  Mirrors
            # the Mamba engine's ``run_mamba_decode_fast_async`` (see
            # ``infra.engine``): copy on a separate stream, record an
            # event, and synchronize before reading the pinned host
            # buffer.  The async copy overlaps with next-step kernel
            # launches on the main stream, eliminating the blocking
            # ``torch.cuda.synchronize()`` that previously dominated
            # per-step latency on small models.
            graph.graph.replay()
            main_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_stream(main_stream)
                self._pinned_tokens[:n].copy_(
                    graph.next_tokens[:n], non_blocking=True,
                )
                self._copy_event.record(self._copy_stream)
            self._copy_event.synchronize()
            tokens = self._pinned_tokens[:n].tolist()
        else:
            # Eager fallback (no graph capture).  Same kernel calls,
            # just with fresh tensors each step.
            input_ids = torch.from_numpy(ids_np).to(device)
            positions = torch.from_numpy(pos_np).to(device)
            slot_mapping = torch.from_numpy(slot_np).to(device)
            context_lens = torch.from_numpy(ctx_np).to(device)
            block_tables = torch.from_numpy(bt_np).to(device)
            cache_indices = torch.from_numpy(cache_idx_np).to(device)
            mamba_meta = JambaMambaMetadata(
                conv_states=self.mamba_pool.conv_states,
                ssm_states=self.mamba_pool.ssm_states,
                cache_indices=cache_indices,
                is_decode=True,
            )
            set_jamba_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                max_context_len=self._max_blocks_per_seq * self._page_size,
                mamba_metadata=mamba_meta,
            )
            try:
                hidden = self.model(input_ids, positions)
                logits = self.model.compute_logits(hidden)
                tokens = logits.argmax(dim=-1).tolist()
            finally:
                reset_context()

        return tokens
