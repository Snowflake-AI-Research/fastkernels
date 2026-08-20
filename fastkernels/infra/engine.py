"""
Batched inference engine with paged KV cache and tensor parallelism.

Architecture closely follows nano-vllm:
  - ModelRunner handles model init, KV cache, CUDA graphs on each GPU
  - For TP>1, rank 0 serializes method calls via shared memory
  - Non-rank-0 workers block in a loop inside ModelRunner.__init__
  - LlamaEngine (rank 0 only) drives scheduling and sampling

No vLLM imports.
"""

from __future__ import annotations

import atexit
import contextlib
import itertools
import json
import os
import pickle
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from .context import (
    AttnBackendConfig, CUDAGraphMode, auto_register_no_compile_layers,
    clear_no_compile_layers, disable_custom_ops, enable_custom_ops,
    get_attn_backend_config, get_context,
    KimiLinearMetadata, reset_context, set_context, set_forward_context,
    set_mamba_context, set_mixed_context,
)
from .mamba_state import (
    KimiLinearStateManager, Mamba2Metadata, MambaMetadata, MambaStateManager,
    build_chunk_metadata, compute_causal_conv1d_metadata,
)
from ..tasks.baseline.L1.allreduce import set_custom_ar
from .weight_loader import load_model

MAX_MODEL_LEN = 131072
NCCL_PORT = int(os.environ.get("FASTKERNELS_NCCL_PORT", "29501"))

# Bound each device-resident hybrid decode chunk while still amortizing the
# per-step host round trips.
_BULK_CHUNK_CAP = 512

# Mamba v1 keeps activations channel-major, so a step's token count lands as the
# contiguous axis of a cuBLAS operand. Keep it 16-byte aligned (8 bf16 elements)
# -- see ``tasks/baseline/L2/mamba_mixer._padded_token_cat``.
_MAMBA_GEMM_ALIGN = 8

# How many audio sequences to push through the Whisper encoder per forward. The
# encoder is not covered by the KV-cache memory budget, so this -- not
# max_num_seqs -- is what bounds its activation peak. See the call site in
# LlamaEngine.generate. Override for A/B.
_WHISPER_ENCODER_CHUNK = int(
    os.environ.get("FASTKERNELS_WHISPER_ENCODER_CHUNK", "64")
)


def _load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except AttributeError as exc:
        msg = str(exc)
        if "extra_special_tokens" not in msg and "keys" not in msg:
            raise
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_name, "tokenizer_config.json")
        with open(cfg_path) as f:
            tok_cfg = json.load(f)
        extra = tok_cfg.get("extra_special_tokens")
        if not isinstance(extra, list):
            raise
        extra_map = {
            f"extra_special_token_{i}": token
            for i, token in enumerate(extra)
        }
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            extra_special_tokens=extra_map,
        )


@contextlib.contextmanager
def _flashinfer_autotune():
    """Run the enclosed forward with FlashInfer's autotuner recording.

    FlashInfer ships several tactics per op and picks by heuristic unless it has
    been tuned; the tuned choice is cached and reused by later calls. vLLM does
    this once at startup (``kernel_warmup.flashinfer_autotune``, gated on
    ``KernelConfig.enable_flashinfer_autotune``, which defaults to True on
    Hopper/Blackwell) by running ``_dummy_run`` at
    ``max_num_batched_tokens`` inside the context -- tuning at *m* tokens covers
    every count up to *m*. Skipping it leaves ops like the trtllm-gen MXFP4 MoE
    on heuristic tactics, so the warmup pass here enters the same context.

    A no-op when FlashInfer is absent or its autotuner API is unavailable, and
    tolerant of tuning failures: a bad tactic search must not stop startup.

    One op is excluded at world_size 1: tuning
    ``flashinfer::trtllm_fp4_block_scale_moe`` segfaults inside
    ``BatchedGemmInterface::run`` for gpt-oss-120b's tp=1 shape, where the
    intermediate is unsharded and equals the hidden width after 256-alignment
    (both 3072). The crash is non-deterministic -- roughly two runs in three at
    ``max_model_len`` 24736, and it disappears entirely under cuda-gdb -- so it is
    a latent timing- or layout-sensitive fault rather than a bad shape. It is not
    memory (reproduces at gpu_memory_utilization 0.90, 0.75 and 0.50), it is not
    reachable from a standalone call to the same op with the same shapes (12/12
    tuning profiles pass), and vLLM does not hit it. Disabling tuning for this op
    avoids it; ``skip_ops`` is the same mechanism vLLM uses to exclude ops whose
    tuning misbehaves (``_flashinfer_autotune_skip_ops``). Scoped to world_size 1
    so tensor-parallel runs keep full tuning, since the fault has never appeared
    there.
    """
    if os.environ.get("FASTKERNELS_FLASHINFER_AUTOTUNE", "1") == "0":
        yield
        return
    try:
        from flashinfer import autotune as _fi_autotune
    except Exception:
        yield
        return

    kwargs: dict = {}
    try:
        import torch.distributed as _dist

        single_rank = not (_dist.is_available() and _dist.is_initialized()) or (
            _dist.get_world_size() == 1
        )
    except Exception:
        single_rank = True
    if single_rank:
        # Both spellings: the autotuner logs the namespaced name, and it is not
        # contractual which form skip_ops matches against.
        kwargs["skip_ops"] = {
            "trtllm_fp4_block_scale_moe",
            "flashinfer::trtllm_fp4_block_scale_moe",
        }
    try:
        with _fi_autotune(True, **kwargs):
            yield
    except TypeError:
        # Older FlashInfer without skip_ops: tune everything rather than nothing.
        with _fi_autotune(True):
            yield
    except Exception as exc:  # pragma: no cover - tuning is best-effort
        print(f"  FlashInfer autotune skipped: {exc}", flush=True)
        yield


def _detect_scheduling_defaults() -> tuple[int, int]:
    """Choose max_num_batched_tokens and max_num_seqs based on GPU memory.

    Mirrors vLLM's heuristic: high-memory GPUs (>=70 GiB, non-A100) get
    larger defaults; everything else gets conservative values.
    """
    if not torch.cuda.is_available():
        return 8192, 256
    _GiB = 1 << 30
    _, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0).lower()
    if total >= 70 * _GiB and "a100" not in name:
        return 16384, 1024
    return 8192, 256


def _is_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


# vLLM ``VllmConfig._set_cudagraph_sizes``: min(max_num_seqs * 2, 512).
# Hopper Mamba-Codestral only has ~849 cache blocks at 0.9 util / 128k, so
# vLLM's default max_num_seqs=1024 fails CUDA-graph capture. Matching this
# cap keeps Hopper under the block count without changing Blackwell.
_VLLM_DEFAULT_MAX_CUDAGRAPH_CAPTURE_SIZE = 512


def _vllm_default_cudagraph_capture_sizes(max_capture: int) -> list[int]:
    """Match vLLM ``VllmConfig._set_cudagraph_sizes`` (non-interactivity)."""
    sizes = [i for i in (1, 2, 4) if i <= max_capture]
    if max_capture >= 8:
        sizes += list(range(8, min(max_capture + 1, 256), 8))
    if max_capture >= 256:
        sizes += list(range(256, max_capture + 1, 16))
    return sorted(set(sizes))


_DEFAULT_MAX_NUM_BATCHED_TOKENS, _DEFAULT_MAX_NUM_SEQS = (
    _detect_scheduling_defaults()
)

_PROFILE = os.environ.get("FASTKERNELS_PROFILE", "0") == "1"


ATTN_BACKEND_CONFIG = get_attn_backend_config()
USE_TRTLLM = ATTN_BACKEND_CONFIG.use_trtllm
BLOCK_SIZE = ATTN_BACKEND_CONFIG.block_size

# Stand-in for a sampled token that is still on the GPU while the host runs
# ahead one decode step. Only ever read back by the deferred patch, never by
# the model: the graph gets its tokens device-to-device.
_RUNAHEAD_PLACEHOLDER = 0
USE_FLASHINFER = USE_TRTLLM  # back-compat alias


# ---------------------------------------------------------------------------
# Data classes
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
    token_ids: list[int] = field(default_factory=list)
    logits_history: list[torch.Tensor] | None = None


class SeqStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    FINISHED = auto()


# ---------------------------------------------------------------------------
# Sequence — must be picklable for shared memory transfer
# ---------------------------------------------------------------------------
class Sequence:
    _next_id = 0

    def __init__(self, prompt_ids: list[int], max_tokens: int = 512,
                 ignore_eos: bool = False):
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
        # Multimodal fields
        self.pixel_values = None  # preprocessed image pixels
        self.image_grid_thw = None  # list of [t, h, w] per image
        self.video_pixel_values = None
        self.video_grid_thw = None
        self.video_second_per_grid = None
        self.input_audio_features = None
        self.audio_feature_lengths = None
        self.mrope_position_delta: int = 0
        self.mrope_positions = None  # (3, seq_len) tensor computed at prefill
        # Positions of this sequence's placeholder (image/video/audio) tokens
        # within token_ids, as a sorted int64 array, plus the placeholder id.
        # Precomputed once during preprocessing so the per-step vision merge can
        # slice the chunk's rows with searchsorted instead of building masks on
        # the GPU and reading them back with .item().
        self.vis_positions = None
        self.vis_token_id: int | None = None
        # Encoder-decoder fields (Whisper)
        self.encoder_features: torch.Tensor | None = None  # [num_mel_bins, T] log-mel
        self.encoder_seq_len: int = 0  # num encoder tokens (after conv)
        self.cross_block_table: list[int] = []  # paged KV blocks for cross-attn
        self.encoder_computed: bool = False  # True after encoder has run
        # Mamba/SSM models: index into MambaStateManager's slot pool.
        # Allocated for the lifetime of the sequence; None for non-mamba models.
        self.state_slot: int | None = None

    def __len__(self):
        if self.token_ids is not None:
            return len(self.token_ids)
        return self._num_tokens

    @property
    def num_blocks(self):
        return (len(self) + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def last_block_num_tokens(self):
        r = len(self) % BLOCK_SIZE
        return r if r else BLOCK_SIZE

    @property
    def last_token(self):
        if self.token_ids is not None:
            return self.token_ids[-1]
        return self._last_token

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_ids)

    @property
    def num_remaining_prefill(self):
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    def blocks_needed_for(self, num_tokens):
        """Number of NEW blocks needed to store num_tokens more KV slots."""
        total_after = self.num_computed_tokens + num_tokens
        blocks_after = (total_after + BLOCK_SIZE - 1) // BLOCK_SIZE
        return max(0, blocks_after - len(self.block_table))

    def preempt(self):
        """Reset to re-prefillable state (vLLM-style recompute preemption)."""
        self.token_ids = list(self.prompt_ids)
        self.generated_ids.clear()
        self.block_table.clear()
        self.cross_block_table.clear()
        self.num_computed_tokens = 0
        self.encoder_computed = False
        # state_slot is freed by the engine before preempt(); leave the
        # field as-is so callers can detect it.
        self.status = SeqStatus.WAITING

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.generated_ids.append(token_id)

    def append_tokens(self, token_ids: list[int]):
        """Append multiple accepted speculative tokens at once."""
        self.token_ids.extend(token_ids)
        self.generated_ids.extend(token_ids)

    def __getstate__(self):
        """Minimal pickling for shared memory transfer to non-rank-0 workers."""
        return (len(self), len(self.prompt_ids), self.block_table,
                self.num_computed_tokens,
                self.state_slot,
                self.token_ids if not self.generated_ids else self.last_token)

    def __setstate__(self, state):
        (self._num_tokens, num_prompt, self.block_table,
         self.num_computed_tokens, self.state_slot) = state[:-1]
        if isinstance(state[-1], list):
            self.token_ids = state[-1]
        else:
            self.token_ids = None
            self._last_token = state[-1]
        self.prompt_ids = []
        self.generated_ids = []
        # Not transferred: the placeholder-position cache would roughly double
        # this payload (one int64 per vision token, so ~35KB for a 32-frame
        # clip) to save a np.flatnonzero over token_ids that costs microseconds.
        # _run_mm_lm recomputes it per rank instead.
        self.vis_positions = None
        self.vis_token_id = None


# ---------------------------------------------------------------------------
# Block Manager
# ---------------------------------------------------------------------------
class BlockManager:
    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def reset(self):
        """Return all blocks to the free pool."""
        self.free_block_ids = deque(range(self._num_blocks))
        if hasattr(self, '_num_cross_blocks'):
            self.cross_free_block_ids = deque(range(self._num_cross_blocks))

    def can_allocate(self, seq):
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq):
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def can_allocate_n(self, n_blocks):
        return len(self.free_block_ids) >= n_blocks

    def allocate_n(self, seq, n_blocks):
        for _ in range(n_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def can_append(self, seq):
        return len(self.free_block_ids) >= (len(seq) % BLOCK_SIZE == 1)

    def may_append(self, seq):
        if len(seq) % BLOCK_SIZE == 1:
            seq.block_table.append(self.free_block_ids.popleft())

    def deallocate(self, seq):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()

    def free_tail_blocks(self, seq, n_blocks: int) -> int:
        """Release the last ``n_blocks`` blocks of ``seq`` back to the pool.

        Returns the number of blocks actually released. Used by the EAGLE-3
        verify path when speculative tokens are rejected and their KV slots no
        longer need to be cached.
        """
        n_blocks = min(n_blocks, len(seq.block_table))
        if n_blocks <= 0:
            return 0
        released = seq.block_table[-n_blocks:]
        del seq.block_table[-n_blocks:]
        self.free_block_ids.extend(released)
        return n_blocks

    def deallocate_cross(self, seq):
        """Return cross-attention KV blocks to the cross-attn free pool."""
        if hasattr(self, 'cross_free_block_ids') and seq.cross_block_table:
            self.cross_free_block_ids.extend(seq.cross_block_table)
            seq.cross_block_table.clear()


# ---------------------------------------------------------------------------
# ModelRunner — runs on EACH TP rank
# ---------------------------------------------------------------------------
def _seq_encoder_tokens(seq, merge_size: int) -> int:
    """Return post-merge vision tokens contributed by one sequence."""
    total = 0
    for attr in ("image_grid_thw", "video_grid_thw"):
        grid = getattr(seq, attr, None)
        if grid is None:
            continue
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.long)
        total += int((grid.prod(-1) // (merge_size**2)).sum().item())
    return total


class ModelRunner:
    def __init__(self, model_name: str, rank: int, world_size: int,
                 dtype: torch.dtype | None, enforce_eager: bool,
                 event, shm_name: str,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = MAX_MODEL_LEN,
                 max_num_seqs: int | None = None,
                 max_num_batched_tokens: int | None = None,
                 max_layers: int | None = None,
                 kv_cache_dtype: str | None = None):
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.enforce_eager = enforce_eager
        self.event = event
        self.block_size = BLOCK_SIZE
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = ((max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 2) * BLOCK_SIZE
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else _DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        self.max_layers = max_layers
        # MLA paged-KV cache dtype. Also selects the MLA attention backend, so it
        # has to reach the model at construction time, not just the allocator.
        self.kv_cache_dtype = kv_cache_dtype
        # Batch size whose sampled tokens are already in graph_vars["input_ids"]
        # on device; -1 when there is no valid handoff.
        self._staged_ids_n = -1
        # DSA indexer paged-MQA-logits schedule. Only the MLA KV-cache path fills
        # this in, but ``_refresh_indexer_schedule`` is called unconditionally from
        # every decode site and from capture_cudagraph, and it returns early on
        # None. Default it here so non-MLA models (Llama, Mixtral, BitNet, ...)
        # reach that early return instead of an AttributeError.
        self._indexer_sched = None
        # FA3 paged-decode scheduler metadata.  Collected in
        # ``_init_fa3_decode_buffers``; ``_refresh_fa3_decode_schedule``
        # returns early when empty (TRTLLM / MLA / no FlashAttn layers).
        self._fa_decode_ops: list = []
        self._fa_decode_groups: list = []

        torch.cuda.set_device(rank)
        self._dist_initialized = False
        if world_size > 1:
            dist.init_process_group(
                "nccl", f"tcp://localhost:{NCCL_PORT}",
                world_size=world_size, rank=rank,
                device_id=torch.device(f"cuda:{rank}"),
            )
            self._dist_initialized = True

        self.custom_ar = None
        if world_size > 1:
            self.cpu_group = dist.new_group(backend="gloo")
            if not os.environ.get("FASTKERNELS_DISABLE_CUSTOM_AR", "0") == "1":
                from ..tasks.baseline.L1.allreduce import CustomAllreduce
                self.custom_ar = CustomAllreduce(
                    self.cpu_group, rank, max_size=8 * 1024 * 1024
                )
                set_custom_ar(self.custom_ar)

        if dtype is None:
            # DeepSeek V3.2 checkpoints use ``model_type: "deepseek_v32"`` which
            # is not yet a registered AutoConfig key in transformers, so load
            # with ``trust_remote_code=True`` and fall back to raw config.json
            # parsing if AutoConfig still refuses.
            from transformers import AutoConfig
            _cfg = None
            cfg_dtype = None
            model_type = ""
            try:
                _cfg = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=True,
                )
                cfg_dtype = getattr(_cfg, "torch_dtype", None)
                model_type = getattr(_cfg, "model_type", "")
            except (ValueError, KeyError, OSError):
                # BitNet b1.58 ships an ``auto_map`` pointing at
                # ``configuration_bitnet.py`` / ``modeling_bitnet.py`` files
                # that don't actually exist in the repo (the model_type is
                # registered natively in transformers).  Retry without
                # ``trust_remote_code`` so AutoConfig uses the registered
                # class instead of the dynamic loader.
                try:
                    _cfg = AutoConfig.from_pretrained(
                        model_name, trust_remote_code=False,
                    )
                    cfg_dtype = getattr(_cfg, "torch_dtype", None)
                    model_type = getattr(_cfg, "model_type", "")
                except Exception:
                    try:
                        from huggingface_hub import hf_hub_download
                        import json as _json
                        if os.path.isdir(model_name):
                            _cfg_path = os.path.join(model_name, "config.json")
                        else:
                            _cfg_path = hf_hub_download(model_name, "config.json")
                        with open(_cfg_path) as _f:
                            _cfg_dict = _json.load(_f)
                        _td = _cfg_dict.get("torch_dtype", None)
                        cfg_dtype = getattr(torch, _td) if isinstance(_td, str) else None
                        model_type = _cfg_dict.get("model_type", "")
                        if isinstance(cfg_dtype, torch.dtype):
                            dtype = cfg_dtype
                    except Exception:
                        pass
            if model_type in ("mamba", "mamba2") and cfg_dtype == torch.float32:
                # HF Mamba configs often advertise fp32, but the serving path
                # restores only the recurrent SSM parameters to fp32 after load
                # and keeps the main activations/weights in reduced precision.
                dtype = torch.bfloat16
            elif dtype is None and isinstance(cfg_dtype, torch.dtype):
                dtype = cfg_dtype
            elif dtype is None:
                dtype = torch.bfloat16
        self.dtype = dtype
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        import time as _time
        _t0 = _time.perf_counter()
        if rank == 0:
            print(f"  [1/6] Loading model weights...", flush=True)
        self.model, self.config = load_model(
            model_name, torch.device(f"cuda:{rank}"), dtype,
            max_layers=self.max_layers,
            max_num_batched_tokens=self.max_num_batched_tokens,
            kv_cache_dtype=self.kv_cache_dtype,
            max_model_len=self.max_model_len,
        )
        model_type = getattr(self.config, "model_type", "")
        self.is_kimi_linear = model_type == "kimi_linear"
        self.is_qwen3_next = model_type == "qwen3_next"
        self.is_mamba2 = model_type == "mamba2"
        self.is_mamba = (
            model_type in ("mamba", "mamba2")
            or self.is_kimi_linear
            or self.is_qwen3_next
        )
        # Hopper-only: vLLM's 1024-seq default exceeds Mamba cache blocks on
        # H200 (~849 at 0.9 util / 128k). Cap at vLLM's hopper CUDA-graph
        # size (512). Blackwell has more HBM and keeps 1024.
        if (
            model_type in ("mamba", "mamba2")
            and _is_hopper_gpu()
            and self.max_num_seqs > _VLLM_DEFAULT_MAX_CUDAGRAPH_CAPTURE_SIZE
        ):
            self.max_num_seqs = _VLLM_DEFAULT_MAX_CUDAGRAPH_CAPTURE_SIZE
        self.model_family = "mamba" if self.is_mamba else "attention"
        # Granularity a *continuing* prefill chunk must end on.
        #
        # Mamba2's SSD scan requires chunk_size-aligned continuations, else the
        # carried SSM state diverges from a single-shot prefill.  Mamba v1 has
        # no such constraint -- its conv/ssm state carries token by token -- but
        # its mixer keeps activations channel-major (``[dim, num_tokens]``), so
        # the token count lands as the contiguous axis of a cuBLAS operand and
        # must stay 16-byte aligned or cuBLAS drops to a scalar path (see
        # ``L2/mamba_mixer._padded_token_cat``).  Rounding v1 to 256 as well
        # would just throw away 1.5% of every step's token budget.
        if self.is_mamba2:
            self._mamba_prefill_align = (
                getattr(self.config, "chunk_size", 256) or 256
            )
        else:
            self._mamba_prefill_align = _MAMBA_GEMM_ALIGN
        self.is_gpt_oss = model_type == "gpt_oss" or "gpt-oss" in model_name.lower()
        self.is_bitnet = model_type == "bitnet"
        self.is_gemma4 = model_type == "gemma4"
        self.is_moe = hasattr(self.config, "num_local_experts") or getattr(self.config, "is_moe", False)
        self.is_qwen_vl = hasattr(self.config, "mrope_section")
        self.is_qwen3_vl = self.is_qwen_vl and hasattr(
            getattr(self.config, "vision", None), "deepstack_visual_indexes"
        )
        self.is_whisper = getattr(self.config, "is_encoder_decoder", False)
        self.is_deepseek_mla = hasattr(self.config, "kv_lora_rank")
        if self.is_whisper:
            self.max_model_len = min(
                self.max_model_len,
                getattr(self.config, "max_target_positions", self.max_model_len),
            )

        auto_register_no_compile_layers(self.model)

        self._compiled = False
        self.num_blocks = 0  # set by allocate_kv_cache (attention) only
        self.mamba_state_manager: MambaStateManager | None = None
        if rank == 0:
            print(f"  [1/6] Model loaded in {_time.perf_counter()-_t0:.1f}s", flush=True)
        if self.is_mamba:
            if rank == 0:
                print("  [2/6] Sharing activation buffers...", flush=True)
            self._share_activation_buffers()
            if self.is_kimi_linear:
                if rank == 0:
                    print("  [3/6] Allocating MLA KV cache...", flush=True)
                self._allocate_mla_kv_cache()
                if rank == 0:
                    print("  [4/6] Allocating Kimi-Linear state cache...", flush=True)
                self.allocate_mamba_state_cache()
                if not self.enforce_eager:
                    # Compile before capturing. Kimi (and Qwen3-Next) were the
                    # only non-eager models that captured graphs over EAGER
                    # modules -- ``_compile_model`` is called on the generic
                    # mamba2 branch and the attention branch, but never here.
                    # Two costs followed, both visible in a tp=2 bs=1 profile:
                    # the AR+RMSNorm post-grad fusion never ran (55 separate
                    # ``cross_device_reduce`` at 484 us plus 54 separate RMSNorm
                    # at 146 us, where Mixtral/gpt-oss show one fused
                    # ``oneshotAllreduceFusionKernel``), and no Inductor
                    # elementwise fusion happened at all (82 ``direct_copy``
                    # kernels per step, 180 us). vLLM compiles this model.
                    if rank == 0:
                        print("  [5/6] Compiling Kimi-Linear...", flush=True)
                    self._compile_model()
                    if rank == 0:
                        print("  [5/6] Preparing Kimi decode buffers...", flush=True)
                    self._init_kimi_decode_buffers()
                    if rank == 0:
                        print("  [6/6] Capturing Kimi CUDA graphs...", flush=True)
                    self.capture_kimi_cudagraph()
            elif self.is_qwen3_next:
                if rank == 0:
                    print("  [3/6] Allocating Qwen3-Next state/KV cache...", flush=True)
                # Before the cache sizing, not after: the fused-collective
                # workspace and its zero-residual buffer are hundreds of MiB, and
                # claiming them after the KV cache has already taken everything
                # gpu_memory_utilization allows would OOM.
                self._init_fi_allreduce_fusion_workspace()
                self.allocate_mamba_state_cache()
                if not self.enforce_eager:
                    if rank == 0:
                        print("  [4/6] Preparing Qwen3-Next decode buffers...", flush=True)
                    self._init_kimi_decode_buffers()
                    if rank == 0:
                        print("  [5/6] Capturing Qwen3-Next CUDA graphs...", flush=True)
                    self.capture_kimi_cudagraph()
                if rank == 0:
                    print("  [6/6] Warming Qwen3-Next prefill kernels...", flush=True)
                self._warmup_qwen3_next_prefill()
            else:
                if rank == 0:
                    print("  [3/6] Profiling Mamba state cache...", flush=True)
                self._profile_mamba_run()
                self.allocate_mamba_state_cache()
                if self.is_mamba2 and not self.enforce_eager:
                    if rank == 0:
                        print("  [4/6] Compiling Mamba2...", flush=True)
                    self._compile_model()
                if rank == 0:
                    print("  [5/6] Preparing Mamba decode buffers...", flush=True)
                self._init_mamba_decode_buffers()
                if not self.enforce_eager:
                    if rank == 0:
                        print("  [6/6] Capturing Mamba CUDA graphs...", flush=True)
                    self.capture_mamba_cudagraph()
            if rank == 0:
                print(f"  Engine ready in {_time.perf_counter()-_t0:.1f}s total", flush=True)
        else:
            self._share_trtllm_workspace()
            self._share_activation_buffers()
            if rank == 0:
                print(f"  [2/6] Warmup forward pass...", flush=True)
            _t1 = _time.perf_counter()
            self.warmup_model()
            if rank == 0:
                print(f"  [2/6] Warmup done in {_time.perf_counter()-_t1:.1f}s", flush=True)
                print(f"  [3/6] DeepGEMM warmup...", flush=True)
            _t2 = _time.perf_counter()
            self._warmup_deepgemm()
            if rank == 0:
                print(f"  [3/6] DeepGEMM done in {_time.perf_counter()-_t2:.1f}s", flush=True)
                print(f"  [4/6] Allocating KV cache...", flush=True)
            _t3 = _time.perf_counter()
            self.allocate_kv_cache()
            self._init_fa3_decode_buffers()
            self._init_cross_attn_decode_buffers()
            if rank == 0:
                print(f"  [4/6] KV cache done in {_time.perf_counter()-_t3:.1f}s", flush=True)
            if not self.enforce_eager and not self._supports_cudagraphs():
                self.enforce_eager = True
                if rank == 0:
                    print("  [5/6] Forcing eager: decode CUDA graphs are unsafe "
                          "for this model (see _supports_cudagraphs).",
                          flush=True)
            if not self.enforce_eager:
                if rank == 0:
                    print(f"  [5/6] Compiling + capturing CUDA graphs...", flush=True)
                _t4 = _time.perf_counter()
                self._compile_model()
                self.capture_cudagraph()
                if rank == 0:
                    print(f"  [5/6] CUDA graphs done in {_time.perf_counter()-_t4:.1f}s", flush=True)
            if rank == 0:
                print(f"  [6/6] Init greedy buffers...", flush=True)
            self._init_greedy_buffers()
            if rank == 0:
                # How much of the multimodal runtime reserve was actually
                # needed. The reserve is an empirical floor (30% of total) taken
                # out of the KV cache because the pre-compile profiling pass
                # cannot observe the compiled forward's peak; printing the real
                # post-capture peak and the leftover free memory says how much
                # KV cache that floor is costing.
                _free, _tot = torch.cuda.mem_get_info()
                self._post_capture_alloc = torch.cuda.memory_allocated()
                print(
                    f"  Post-capture memory: peak_alloc="
                    f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                    f"cur_alloc={torch.cuda.memory_allocated() / 2**30:.1f}G "
                    f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}G "
                    f"free={_free / 2**30:.1f}G of {_tot / 2**30:.0f}G",
                    flush=True,
                )
                # Baseline the peak counter here so the mm_runtime_peak figure
                # reported after generate() is headroom used at *runtime*, not
                # whatever weight loading transiently touched.
                torch.cuda.reset_peak_memory_stats()
            elif self.is_qwen_vl:
                _free_r, _tot_r = torch.cuda.mem_get_info()
                print(f"  [rank {rank}] Post-capture memory: "
                      f"cur_alloc={torch.cuda.memory_allocated() / 2**30:.1f}G "
                      f"free={_free_r / 2**30:.1f}G", flush=True)
                self._post_capture_alloc = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                print(f"  Engine ready in {_time.perf_counter()-_t0:.1f}s total", flush=True)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # TP shared memory setup
        if world_size > 1:
            self._init_shm_layout()
            if rank == 0:
                self.shm = SharedMemory(
                    name=shm_name, create=True, size=self._SHM_SIZE,
                )
                self.shm.buf[self._SHM_FLAG_OFFSET] = 0
                self.shm.buf[self._SHM_SEQ_OFFSET:self._SHM_SEQ_OFFSET+4] = (0).to_bytes(4, "little")
                self.shm.buf[self._SHM_ACK_OFFSET:self._SHM_SEQ_OFFSET] = bytes(
                    self._SHM_SEQ_OFFSET - self._SHM_ACK_OFFSET
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()  # Non-rank-0 blocks here forever

    def exit(self):
        if self.custom_ar is not None:
            self.custom_ar.close()
            self.custom_ar = None
            set_custom_ar(None)
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if self._dist_initialized:
            dist.destroy_process_group()

    # SHM layout for spin-wait signaling:
    # byte[-1] (_SHM_FLAG_OFFSET): 0=generic, 1=attn decode_greedy,
    #                              2=mamba decode_greedy,
    #                              3=hybrid recurrent decode_greedy
    # bytes[-5:-1] (_SHM_SEQ_OFFSET): 4-byte little-endian sequence counter
    # bytes[-37:-5] (_SHM_ACK_OFFSET): per-rank consumed-sequence counters
    #
    # The data region below _SHM_ACK_OFFSET must hold the largest decode
    # payload any dispatch path can produce.  ``_init_shm_layout`` sizes it
    # from the *final* scheduler/cache config, so the offsets are instance
    # attributes rather than constants -- a fixed 1 MiB region silently
    # truncated the block-table field once
    # ``max_num_seqs * max_num_blocks * 4`` exceeded it, and the resulting
    # short-slice assignment surfaced as an opaque
    # "memoryview assignment: lvalue and rvalue have different structures".
    _SHM_CONTROL_BYTES = 5 + 4 * 8

    def _init_shm_layout(self) -> None:
        """Size the TP shared-memory segment for the worst-case decode payload.

        Every rank derives the layout from the same already-finalized config
        (``max_num_seqs``, ``max_model_len``, block size, MRoPE/MLA dtypes),
        so rank 0's ``create=True`` size and the workers' offsets agree
        without any extra handshake.
        """
        max_bs = self.max_num_seqs
        max_num_blocks = (self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        # ``n`` and ``max_bt`` are 2-byte header fields in every decode
        # payload, so both counts must stay addressable there.
        if max_bs > 0xFFFF or max_num_blocks > 0xFFFF:
            raise RuntimeError(
                f"TP decode SHM header cannot address max_num_seqs={max_bs} / "
                f"max_num_blocks={max_num_blocks} (2-byte fields, limit 65535)"
            )

        # Attention decode: [n(2)][max_bt(2)][staged(1)][pad(3)][ids][pos][sm][cl][bt]
        pos_elems = 3 * max_bs if self.is_qwen_vl else max_bs
        sm_bytes = max_bs * (8 if self.is_deepseek_mla else 4)
        attn_bytes = (
            8                       # header, padded so the int64s stay aligned
            + max_bs * 8            # ids   (int64)
            + pos_elems * 8         # pos   (int64, 3x for MRoPE)
            + sm_bytes              # slot_mapping (int64 for MLA, else int32)
            + max_bs * 4            # context_lens (int32)
            + max_bs * max_num_blocks * 4   # block_tables (int32)
        )

        # Hybrid (Kimi-Linear / Qwen3-Next) decode:
        #   [n(2)][max_blocks(2)][ids(8)][pos(8)][si(4)][sl(4)][slot(4)][bt]
        hybrid_bytes = (
            4 + max_bs * (8 + 8 + 4 + 4 + 4) + max_bs * max_num_blocks * 4
        )

        # Mamba decode: [n(2)][_(2)][ids(8)][pos(8)][si(4)]
        mamba_bytes = 4 + max_bs * (8 + 8 + 4)

        # ``call()`` also pickles arbitrary method payloads through the same
        # region; keep a floor so small-model configs retain the historical
        # 1 MiB of headroom for those.
        data_bytes = max(attn_bytes, hybrid_bytes, mamba_bytes, 2**20)

        # Page-align the total so the mapping stays a whole number of pages.
        total = data_bytes + self._SHM_CONTROL_BYTES
        page = 4096
        total = ((total + page - 1) // page) * page

        self._SHM_SIZE = total
        self._SHM_FLAG_OFFSET = total - 1
        self._SHM_SEQ_OFFSET = total - 5
        self._SHM_ACK_OFFSET = total - 5 - 4 * 8

    def loop(self):
        """Worker loop: spin-wait on SHM sequence counter for decode, event for generic."""
        buf = self.shm.buf
        flag_off = self._SHM_FLAG_OFFSET
        seq_off = self._SHM_SEQ_OFFSET
        last_seq = int.from_bytes(buf[seq_off:seq_off+4], "little")
        while True:
            cur_seq = int.from_bytes(buf[seq_off:seq_off+4], "little")
            if cur_seq != last_seq:
                last_seq = cur_seq
                flag = buf[flag_off]
                if flag != 0:
                    if flag == 1:
                        self._loop_decode_greedy(cur_seq)
                    elif flag == 2:
                        self._loop_mamba_decode_greedy(cur_seq)
                    elif flag == 3:
                        self._loop_kimi_decode_greedy(cur_seq)
                    continue
                n = int.from_bytes(buf[0:4], "little")
                method_name, *args = pickle.loads(buf[4:n+4])
                self._ack_consumed(cur_seq)
                getattr(self, method_name)(*args)
                if method_name == "exit":
                    break
                continue
            # Yield briefly to avoid pure busy-wait burning power
            pass

    def _signal_workers(self):
        """Wake workers and wait until each has copied the command payload."""
        buf = self.shm.buf
        seq_off = self._SHM_SEQ_OFFSET
        cur = int.from_bytes(buf[seq_off:seq_off+4], "little")
        nxt = (cur + 1) & 0xFFFFFFFF
        buf[seq_off:seq_off+4] = nxt.to_bytes(4, "little")
        self._await_workers_consumed(nxt)

    def _await_workers_consumed(self, target_seq):
        buf = self.shm.buf
        target = target_seq.to_bytes(4, "little")
        for rank in range(1, self.world_size):
            offset = self._SHM_ACK_OFFSET + 4 * rank
            while bytes(buf[offset:offset+4]) != target:
                pass

    def _ack_consumed(self, seq):
        offset = self._SHM_ACK_OFFSET + 4 * self.rank
        self.shm.buf[offset:offset+4] = seq.to_bytes(4, "little")

    def call(self, method_name, *args):
        """Called by rank 0 to execute method on ALL ranks."""
        if self.world_size > 1 and self.rank == 0:
            data = pickle.dumps([method_name, *args])
            n = len(data)
            buf = self.shm.buf
            if n + 4 > self._SHM_ACK_OFFSET:
                raise RuntimeError(
                    f"SHM payload too large for {method_name!r}: {n} bytes "
                    f"(limit {self._SHM_ACK_OFFSET - 4})"
                )
            buf[0:4] = n.to_bytes(4, "little")
            buf[4:n+4] = data
            buf[self._SHM_FLAG_OFFSET] = 0  # generic path
            self._signal_workers()
        return getattr(self, method_name)(*args)

    def _share_trtllm_workspace(self):
        """Replace per-layer TRTLLM workspace buffers with a single shared one."""
        if not ATTN_BACKEND_CONFIG.use_trtllm:
            return
        self._attn_layers = []
        self._cross_attn_layers = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                if getattr(module, "is_cross_attn", False):
                    self._cross_attn_layers.append(module)
                else:
                    self._attn_layers.append(module)
        trtllm_workspace = torch.zeros(
            512 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{self.rank}"
        )
        for layer in self._attn_layers:
            # MLA layers (DeepSeek-V3.2 / GLM-5.2) expose k_cache/v_cache but use
            # the FlashMLA workspace, not the TRT-LLM per-layer workspace, so they
            # do not implement set_trtllm_workspace — skip them.
            if hasattr(layer, "set_trtllm_workspace"):
                layer.set_trtllm_workspace(trtllm_workspace)
        # Hybrid models (Qwen3-Next, Kimi-Linear) keep their KV cache in the
        # engine's state manager rather than on the layer, so those layers
        # carry no ``k_cache`` attribute and are not in ``_attn_layers``.
        # Hand them the shared workspace too -- otherwise every such layer
        # holds the 512 MiB buffer it allocated in its own __init__.
        seen = {id(layer) for layer in self._attn_layers}
        for module in self.model.modules():
            if id(module) in seen:
                continue
            if hasattr(module, "set_trtllm_workspace"):
                module.set_trtllm_workspace(trtllm_workspace)
        torch.cuda.empty_cache()

    def _share_activation_buffers(self):
        """No-op (was: share SiluAndMul output buffers across layers).

        The previous implementation made all ``SiluAndMul`` instances point
        to a single ``_ActivationBuffer`` so the output tensor would be
        reused across layers (saving allocations).  However the underlying
        CUDA kernel writes to ``out`` assuming a contiguous
        ``[num_tokens, d]`` layout (row stride == ``d``), and the shared
        buffer returned a non-contiguous slice ``buf[:rows, :cols]`` whose
        row stride was the buffer's *full* width.  When two SiluAndMul
        instances had different ``d`` (e.g. dense MLP at d=9216 and the
        DeepSeek-V3 shared expert at d=2048), the shared-expert output
        was silently corrupted — every row past row 0 wrote to the wrong
        storage offset.  ``SiluAndMul.forward_cuda`` now allocates a
        fresh contiguous output per call (matches vLLM exactly).
        """
        from ..tasks.baseline.L2.mamba2_mixer import Mamba2Mixer

        mamba2_modules = [
            m for m in self.model.modules() if isinstance(m, Mamba2Mixer)
        ]
        if len(mamba2_modules) > 1:
            shared_ssm_out = torch.empty(
                self.max_num_batched_tokens,
                mamba2_modules[0].tped_intermediate_size,
                dtype=self.dtype,
                device=f"cuda:{self.rank}",
            )
            for module in mamba2_modules:
                module.set_shared_ssm_out_buffer(shared_ssm_out)

    def _init_deepstack_buffers(self):
        """Persistent DeepStack input buffers, as vLLM registers at init.

        Qwen3-VL adds a per-level residual to the first N decoder layers, so a
        multimodal prefill step needs one (num_tokens, hidden) tensor per level.
        Allocating those fresh each step costs 402MB of alloc/free per step at a
        full 16384-token budget (3 x 16384 x 4096 x 2B) -- it inflates the
        per-step activation peak, which is charged against the KV cache, and it
        churns the allocator, which is where the "2.32 GiB reserved but
        unallocated" in the video OOM came from.

        vLLM sizes these once to max_num_batched_tokens
        (Qwen3VLForConditionalGeneration.__init__ -> deepstack_input_embeds) and
        reuses them, zeroing only the rows it used. Allocating here -- before
        allocate_kv_cache() runs -- means the KV budget accounts for them once,
        exactly as vLLM's does, instead of leaving them to fight the KV cache for
        headroom at runtime.
        """
        self._deepstack_bufs = None
        model = getattr(self, "model", None)
        visual = getattr(model, "visual", None)
        if visual is None or not hasattr(visual, "deepstack_merger_list"):
            return
        num_levels = len(visual.deepstack_visual_indexes)
        if num_levels <= 0:
            return
        rows = int(self.max_num_batched_tokens)
        hidden = self.config.hidden_size
        self._deepstack_bufs = [
            torch.zeros(rows, hidden, device=f"cuda:{self.rank}",
                        dtype=self.dtype)
            for _ in range(num_levels)
        ]
        if self.rank == 0:
            print(f"  DeepStack buffers: {num_levels} x {rows}x{hidden} "
                  f"({num_levels * rows * hidden * 2 / 2**30:.2f}G persistent)",
                  flush=True)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if self.is_qwen_vl:
            self._init_deepstack_buffers()
            self._warmup_vision_encoder()

        if self.is_whisper:
            self._warmup_whisper()
        else:
            warmup_len = min(self.max_model_len, self.max_num_batched_tokens)
            num_seqs = min(self.max_num_batched_tokens // warmup_len, self.max_num_seqs)
            seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
            with _flashinfer_autotune():
                self.run(seqs, True)

        torch.cuda.empty_cache()

    def _warmup_deepgemm(self):
        """Pre-JIT DeepGEMM FP8 kernels for all weight shapes at decode and
        prefill batch sizes, pre-allocate decode buffers per instance and
        shared prefill buffers per unique (K, N) shape.

        Respects VLLM_DEEP_GEMM_WARMUP env var: "skip" disables JIT warmup
        (buffers are still allocated), "relax"/"full" run warmup normally.
        """
        import os
        try:
            from ..tasks.baseline.L1.fp8_linear import (
                Fp8Linear, _Fp8PrefillBufs, _alloc_colmajor_scale,
            )
        except ImportError:
            return

        skip_jit = os.environ.get("VLLM_DEEP_GEMM_WARMUP", "").lower() == "skip"

        fp8_modules = []
        for module in self.model.modules():
            linear_op = getattr(module, 'linear_op', None)
            if isinstance(linear_op, Fp8Linear):
                fp8_modules.append((module, linear_op))

        if not fp8_modules:
            return

        import deep_gemm

        max_decode = self.max_num_seqs
        max_prefill = self.max_num_batched_tokens
        device = next(self.model.parameters()).device

        decode_bs = [1, 2, 4, 8] + list(range(16, max_decode + 1, 16))
        prefill_bs = [s for s in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
                      if s <= max_prefill and s > max_decode]

        prefill_bufs: dict[tuple[int, int], _Fp8PrefillBufs] = {}

        seen_shapes = set()
        for module, linear_op in fp8_modules:
            w = module.weight
            ws = module.weight_scale_inv
            N, K = w.shape

            linear_op._ensure_buffers(max_decode, K, N, device)

            key = (N, K)
            if key not in prefill_bufs:
                prefill_bufs[key] = _Fp8PrefillBufs(max_prefill, K, N, device)
            linear_op._pf = prefill_bufs[key]

            if skip_jit or key in seen_shapes:
                seen_shapes.add(key)
                continue
            seen_shapes.add(key)

            a_fp8 = linear_op._a_buf
            out = linear_op._o_buf
            # The activation scale factor must be column-major with its minor
            # (group) stride equal to the row count M -- DeepGEMM's SM100
            # ``fp8_gemm_nt`` asserts ``batched_sf.stride(2) == mn``. Slicing the
            # leading dim of a pre-allocated ``(max_tokens, num_groups)``
            # column-major buffer leaves that stride at ``max_tokens``, so we
            # allocate a fresh exact-M scale per size, mirroring ``Fp8Linear.
            # forward`` (which likewise never reuses ``_s_buf``).
            num_groups = (K + Fp8Linear.BLOCK_SIZE - 1) // Fp8Linear.BLOCK_SIZE
            for num_tokens in decode_bs:
                if num_tokens > max_decode:
                    break
                a_scale = _alloc_colmajor_scale(num_tokens, num_groups, device)
                deep_gemm.fp8_gemm_nt(
                    (a_fp8[:num_tokens], a_scale),
                    (w, ws),
                    out[:num_tokens],
                )

            pf = prefill_bufs[key]
            for num_tokens in prefill_bs:
                a_scale = _alloc_colmajor_scale(num_tokens, num_groups, device)
                deep_gemm.fp8_gemm_nt(
                    (pf.a[:num_tokens], a_scale),
                    (w, ws),
                    pf.o[:num_tokens],
                )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self.rank == 0:
            if skip_jit:
                print(f"  DeepGEMM warmup: skipped JIT (VLLM_DEEP_GEMM_WARMUP=skip), "
                      f"{len(seen_shapes)} shapes buffered")
            else:
                print(f"  DeepGEMM warmup: {len(seen_shapes)} unique FP8 weight shapes")

    def _warmup_whisper(self):
        """Warmup Whisper encoder + decoder with dummy audio input.

        Exercises the full forward path: encoder -> cross-attn KV write ->
        decoder prefill with cross-attn read.
        """
        config = self.config
        num_mel_bins = config.num_mel_bins
        max_source_positions = config.max_source_positions
        max_target_positions = config.max_target_positions

        T_mel = max_source_positions * 2
        dummy_features = torch.randn(
            1, num_mel_bins, T_mel,
            device=f"cuda:{self.rank}", dtype=torch.get_default_dtype(),
        )
        with torch.inference_mode():
            encoder_outputs = self.model.get_multimodal_embeddings(dummy_features)

        warmup_len = min(max_target_positions, self.max_num_batched_tokens)
        num_seqs = min(self.max_num_batched_tokens // warmup_len, self.max_num_seqs)
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)

        if self.rank == 0:
            print(f"  Whisper warmup: encoder T_mel={T_mel}, "
                  f"decoder warmup_len={warmup_len}")

    def _warmup_vision_encoder(self):
        """Run vision encoder with worst-case dummy inputs to capture peak
        activation memory, following vLLM's profile_run() approach."""
        import math
        from PIL import Image
        from transformers import AutoProcessor

        model = self.model
        vision_cfg = self.config.vision
        patch_size = vision_cfg.patch_size
        merge_size = vision_cfg.spatial_merge_size
        temporal_patch_size = getattr(vision_cfg, "temporal_patch_size", 2)
        unit = patch_size * merge_size

        processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True)

        # Use the REAL per-image pixel cap the processor enforces. Newer Qwen-VL
        # processors express it as size={longest_edge: max_pixels, ...}; older
        # ones as image_processor.max_pixels or config.vision.max_pixels. The
        # earlier hard-coded 1280*28*28 fallback undercounts it by >16x for
        # Qwen3-VL (real longest_edge=16777216), so the profiled image was tiny
        # and its self-attention memory nowhere near the real worst case -> the
        # KV cache left far too little headroom and the first big image OOM'd.
        ip = getattr(processor, "image_processor", processor)
        max_pixels = getattr(vision_cfg, "max_pixels", None)
        size = getattr(ip, "size", None)
        if size is not None:
            # transformers exposes this as a plain dict on some versions and a
            # SizeDict (attribute access, not a dict subclass) on others.
            if isinstance(size, dict):
                le = size.get("longest_edge") or size.get("max_pixels")
            else:
                le = (getattr(size, "longest_edge", None)
                      or getattr(size, "max_pixels", None))
            max_pixels = le or max_pixels
        max_pixels = max_pixels or getattr(ip, "max_pixels", None) or 1280 * 28 * 28

        # The scheduler admits at most max_num_batched_tokens merged vision
        # tokens of image per step (the has_mm encoder budget), so the worst
        # case is a single image that fills that budget -- its full attention
        # over all patches dominates activation memory. Cap the dummy there.
        max_merged_tokens = max(
            1, min(max_pixels // (unit * unit), self.max_num_batched_tokens))
        max_patches = max_merged_tokens

        def _closest_factor_pair(n):
            for d in range(math.isqrt(n), 0, -1):
                if n % d == 0:
                    return d, n // d
            return 1, n

        hf, wf = 1, max_patches
        for s in range(max_patches, 0, -1):
            hf, wf = _closest_factor_pair(s)
            if wf / hf <= 200:
                break
        img_h, img_w = unit * hf, unit * wf

        dummy_img = Image.new("RGB", (img_w, img_h), color=255)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": dummy_img},
            {"type": "text", "text": "x"},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[dummy_img], return_tensors="pt", padding=True)

        pv = inputs["pixel_values"].cuda()
        grid_thw = inputs["image_grid_thw"].cpu()

        # _run_vision_encoder batches ALL images scheduled in a step into a
        # single model.visual() call, bounded by the scheduler's encoder
        # budget (== max_num_batched_tokens; see the has_mm path in the
        # scheduler).  Profiling a single image underestimates peak activation
        # memory by that batching factor, so allocate_kv_cache would leave no
        # headroom and OOM on the first multi-image step.  Replicate the
        # worst-case batch here, mirroring vLLM's profile_run over
        # max_mm_items_per_batch = encoder_budget // max_tokens_per_item.
        tokens_per_item = max(
            1, int(grid_thw.prod(dim=-1).sum().item()) // (merge_size ** 2))
        max_items = max(1, self.max_num_batched_tokens // tokens_per_item)
        max_items = min(max_items, self.max_num_seqs)
        if max_items > 1:
            pv = pv.repeat(max_items, 1)
            grid_thw = grid_thw.repeat(max_items, 1)

        if self.rank == 0:
            print(f"  Vision warmup: {max_items} image(s) {img_w}x{img_h} "
                  f"(~{tokens_per_item} tok/img)")

        # Measure the transient activation this worst-case forward allocates so
        # allocate_kv_cache can reserve exactly that much runtime headroom.
        torch.cuda.reset_peak_memory_stats()
        _base = torch.cuda.memory_allocated()
        with torch.inference_mode():
            vis_out = model.visual(pv, grid_thw=grid_thw)
        mm_peak = torch.cuda.max_memory_allocated() - _base

        self._warmup_encoder_cache = {}
        if isinstance(vis_out, tuple):
            embeds, ds = vis_out
            self._warmup_encoder_cache["img"] = (embeds, ds)
        else:
            self._warmup_encoder_cache["img"] = vis_out

        num_frames = 32
        padded_nf = num_frames + num_frames % temporal_patch_size
        vid_h = min(img_h, 420)
        vid_w = min(img_w, 420)
        vid_h = (vid_h // unit) * unit or unit
        vid_w = (vid_w // unit) * unit or unit

        if self.rank == 0:
            print(f"  Vision warmup: video {num_frames}f {vid_w}x{vid_h}")

        dummy_vid = np.full(
            (num_frames, vid_h, vid_w, 3), 255, dtype=np.uint8)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": list(dummy_vid)},
            {"type": "text", "text": "x"},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        vid_frames_pil = [Image.fromarray(f) for f in dummy_vid]
        inputs = processor(
            text=[text], videos=[vid_frames_pil],
            return_tensors="pt", padding=True)

        vpv = inputs["pixel_values_videos"].cuda()
        vgrid = inputs["video_grid_thw"].cpu()
        torch.cuda.reset_peak_memory_stats()
        _base = torch.cuda.memory_allocated()
        with torch.inference_mode():
            vis_out = model.visual(vpv, grid_thw=vgrid)
        mm_peak = max(mm_peak, torch.cuda.max_memory_allocated() - _base)

        if isinstance(vis_out, tuple):
            embeds, ds = vis_out
            self._warmup_encoder_cache["vid"] = (embeds, ds)
        else:
            self._warmup_encoder_cache["vid"] = vis_out

        # Reserved by allocate_kv_cache as multimodal runtime headroom.
        self._mm_activation_peak_bytes = int(mm_peak)
        if self.rank == 0:
            print(f"  Vision activation peak: {mm_peak / 2**30:.1f}G", flush=True)

        del processor, dummy_img, dummy_vid, vid_frames_pil, pv, vpv
        del inputs, messages, text

    def _init_fa3_decode_buffers(self):
        """Collect FlashAttnDecode modules and preallocate graph-stable buffers.

        FA3's ``scheduler_metadata`` must live in a persistent allocation:
        CUDA graphs record the pointer, and ``_refresh_fa3_decode_schedule``
        rewrites the contents before every capture and replay (vLLM's
        ``FlashAttentionMetadataBuilder`` pattern).
        """
        from ..tasks.baseline.L1.flash_attn_decode import FlashAttnDecode
        from .fa_utils import group_fa_decode_ops

        ops = [m for m in self.model.modules() if isinstance(m, FlashAttnDecode)]
        self._fa_decode_ops = ops
        self._fa_decode_groups = group_fa_decode_ops(ops)
        if not ops:
            return
        device = torch.device(f"cuda:{self.rank}")
        for group in self._fa_decode_groups:
            lead = group[0]
            if not lead.page_size:
                lead.page_size = BLOCK_SIZE
            lead.preallocate(self.max_num_seqs, device)
            for op in group[1:]:
                op.page_size = lead.page_size
                op._cu_seqlens_q = lead._cu_seqlens_q
                op._sched_buf = lead._sched_buf
                op._sched_meta = lead._sched_meta

    def _refresh_fa3_decode_schedule(
        self,
        context_lens: torch.Tensor,
        max_seqlen_k: int | None = None,
    ) -> None:
        """Recompute FA3 tile-scheduler metadata for this decode step.

        Must run OUTSIDE CUDA graph capture/replay.  ``context_lens`` is
        the [B] decode lengths *including* the padded tail the captured
        graph covers, matching ``_refresh_indexer_schedule``.
        """
        from .fa_utils import refresh_fa3_decode_schedule

        if max_seqlen_k is None:
            max_seqlen_k = self.max_model_len
        refresh_fa3_decode_schedule(
            self._fa_decode_groups,
            context_lens,
            max_seqlen_k,
            qkv_dtype=getattr(self, "dtype", torch.bfloat16),
        )

    def allocate_kv_cache(self):
        if not hasattr(self, '_attn_layers') or not self._attn_layers:
            self._attn_layers = []
            self._cross_attn_layers = []
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    if getattr(module, "is_cross_attn", False):
                        self._cross_attn_layers.append(module)
                    else:
                        self._attn_layers.append(module)

        if self.is_deepseek_mla:
            self._allocate_mla_kv_cache()
            return
        if getattr(self.config, "model_type", "") == "gemma4":
            self._allocate_variable_kv_cache()
            return

        # Free the vision warmup's encoder outputs *before* measuring. Nothing
        # reads them, and they used to be dropped at the end of this method --
        # i.e. after the KV cache had already been sized around them. For
        # Qwen3-VL that is a 16384-token image's embeddings plus three DeepStack
        # levels plus a 32-frame clip, so the KV cache was sized to avoid
        # roughly a GiB of memory that was about to be released.
        if hasattr(self, '_warmup_encoder_cache'):
            del self._warmup_encoder_cache
            import gc
            gc.collect()

        # Return the warmup's freed activations to the driver before measuring.
        # mem_get_info() reports allocator-cached blocks as *used*, so without
        # this the budget below silently forfeits everything warmup_model() left
        # in the cache -- measured at several GB, which on the VLM scenarios is
        # the difference between holding all 1000 requests and preempting 47 of
        # them into a second decode wave.
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = self.config.num_key_value_heads // self.world_size
        head_dim = self.config.head_dim
        num_self_attn_layers = len(self._attn_layers)
        num_cross_attn_layers = len(self._cross_attn_layers)
        elem_size = torch.finfo(torch.get_default_dtype()).bits // 8

        available_bytes = int(total * self.gpu_memory_utilization - used - peak + current)

        if num_cross_attn_layers > 0:
            max_encoder_tokens = getattr(self.config, 'max_source_positions', 1500)
            cross_blocks_per_seq = (max_encoder_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
            cross_block_bytes = (
                2 * num_cross_attn_layers * BLOCK_SIZE * num_kv_heads * head_dim * elem_size
            )
            cross_bytes_per_seq = cross_blocks_per_seq * cross_block_bytes
            max_cross_seqs = min(
                self.max_num_seqs,
                max(1, int(available_bytes * 0.5) // cross_bytes_per_seq),
            )
            self.max_num_seqs = max_cross_seqs
            num_cross_blocks = cross_blocks_per_seq * max_cross_seqs
            cross_cache_bytes = num_cross_blocks * cross_block_bytes
            available_bytes -= cross_cache_bytes

            self.num_cross_blocks = num_cross_blocks
            self.cross_blocks_per_seq = cross_blocks_per_seq
            self.max_encoder_tokens = max_encoder_tokens

            # Cross-attention layers declare their own layout (they read the
            # cache with FlashAttention, i.e. NHD, even when the global
            # backend is trtllm/HND for self-attention).
            cross_layout = ATTN_BACKEND_CONFIG.kv_layout
            if self._cross_attn_layers:
                cross_layout = getattr(
                    self._cross_attn_layers[0], "kv_layout", cross_layout,
                )
            if cross_layout == "HND":
                self.cross_kv_cache = torch.empty(
                    2, num_cross_attn_layers, num_cross_blocks,
                    num_kv_heads, BLOCK_SIZE, head_dim,
                )
            else:
                self.cross_kv_cache = torch.empty(
                    2, num_cross_attn_layers, num_cross_blocks,
                    BLOCK_SIZE, num_kv_heads, head_dim,
                )
            for i, module in enumerate(self._cross_attn_layers):
                module.k_cache = self.cross_kv_cache[0, i]
                module.v_cache = self.cross_kv_cache[1, i]

            self._cross_free_block_ids_init = num_cross_blocks
            if self.rank == 0:
                print(f"  Cross-attn KV cache: {num_cross_blocks} blocks "
                      f"({cross_blocks_per_seq} per seq x {max_cross_seqs} seqs, "
                      f"capped from {_DEFAULT_MAX_NUM_SEQS})")

        if self.is_qwen_vl:
            # Reserve runtime headroom for the multimodal per-step forward.
            # The KV-sizing formula's own (peak - current) term collapses to ~0
            # for VLMs: warmup runs eagerly and BEFORE the KV cache exists, so it
            # never allocates the real per-step activation. So, like vLLM (which
            # reserves a profiled activation peak plus a CUDA-graph estimate), we
            # reserve an explicit runtime headroom, never smaller than the
            # measured worst-case encoder peak (_warmup_vision_encoder already
            # replicates a full max_num_batched_tokens encoder batch).
            #
            # The floor used to be 30% of total -- 53.5G on a B200 -- which was
            # 26x what is actually needed and cost most of the KV cache. vLLM's
            # own accounting for Qwen3-VL-8B on this GPU reports "16.97 GiB for
            # weight, 1.51 GiB for peak activation, 0.23 GiB for non-torch
            # memory, and 0.52 GiB for CUDAGraph memory", i.e. ~2.3G of headroom,
            # and it sizes 1,032,592 KV tokens where we sized 624,720. Measured
            # after the fact, the multimodal steps' real runtime headroom is
            # ~3.0G (reported as ``mm_runtime_peak`` by FASTKERNELS_STEP_PROFILE).
            #
            # ``available_bytes`` above already subtracts (peak - current), and
            # warmup_model() runs _warmup_vision_encoder() *before* this, so that
            # term already carries the worst-case encoder activation. Reserving
            # ``measured * margin`` on top double-counts it. Only the shortfall
            # beyond what is already held back is charged here.
            #
            # This matters because KV capacity is a cliff, not a gradient: with
            # 1000 VisionArena requests at ~1062 tokens each, holding them all
            # means one decode wave, and holding 905 of them means two -- 95
            # preemptions, 44k output tokens recomputed, and ~494 of 1034 decode
            # steps running at under 128 sequences. vLLM's own stats on this
            # scenario show "Running: 1000 reqs" with KV usage peaking at 95.2%,
            # i.e. it stays on the near side of that cliff, and the whole
            # remaining gap came from us sitting on the far side of it.
            measured = getattr(self, "_mm_activation_peak_bytes", 0)
            # This reserve is a formality, and it is important to be clear about
            # what actually provides safety. What protects the multimodal steps is
            # the part of the GPU that gpu_memory_utilization leaves unallocated:
            # measured post-capture free memory is 18.0G against a 2.8G runtime
            # peak, i.e. 6x headroom. vLLM relies on exactly the same thing --
            # its own accounting sums to 161.04 GiB against a 160.52 GiB budget,
            # slightly over, with only ~2G nominally attributed to activation.
            # Holding back more than a token amount here simply forfeits KV
            # blocks, and KV capacity is what bounds occupancy: dropping this
            # from 4G to 0.5G took the cache from 62.7k to 64.3k blocks (vLLM has
            # 64.5k) and Qwen3-VL video from 0.94x to 0.97x.
            #
            # The free-memory clamp below is what keeps this honest at any
            # tensor-parallel width -- it guarantees the reserve exists in real
            # free memory rather than only in this formula.
            margin = float(os.environ.get("FASTKERNELS_MM_RESERVE_MARGIN", "0.2"))
            floor_gib = float(os.environ.get("FASTKERNELS_MM_RUNTIME_RESERVE_GIB", "0.5"))
            already_held = max(0, peak - current)
            # Hold back the measured worst case *once*. ``available_bytes`` above
            # already subtracted (peak - current) -- and warmup_model() runs
            # _warmup_vision_encoder() before this, so that term is largely the
            # same encoder activation the reserve is for. Stacking both held back
            # 3.6G where vLLM holds back 2.03G ("1.51 GiB for peak activation ...
            # 0.52 GiB for CUDAGraph memory"), and that difference is most of why
            # its KV is 141.81 GiB against our 137.7 GiB out of the same budget.
            # Giving the generic term back and charging a single holdback keeps
            # the guarantee while recovering the KV blocks that decide whether
            # all 1000 requests fit in one decode wave.
            # The floor has to clear the *runtime* peak with room for allocator
            # fragmentation, not just the profiled encoder peak: mm_runtime_peak
            # measures 3.0G on the image scenario, and a 2.4G reserve OOM'd once
            # the text scenario had run first and left 2.3G reserved-but-
            # unallocated. 4G clears both with ~700 KV blocks still spare.
            available_bytes += already_held
            mm_reserve = max(int(measured * margin), int(floor_gib * (1 << 30)))
            available_bytes -= mm_reserve
            # Make the reserve a guarantee about *actual* free memory rather than
            # about the budget formula above. That formula
            # (total*util - used - (peak - current)) proved optimistic at tp>1:
            # with the reserve subtracted it still sized a KV cache that left
            # rank 1's vision encoder 74 MiB to work with, and _broadcast_visual
            # died trying to allocate 152 MiB. Clamping against mem_get_info
            # keeps the reserve real at any tensor-parallel width.
            free_now, _total_now = torch.cuda.mem_get_info()
            available_bytes = min(available_bytes,
                                  max(0, free_now - mm_reserve))
            self._mm_reserve_bytes = mm_reserve
            if self.rank == 0:
                print(f"  Multimodal runtime reserve: {mm_reserve / 2**30:.1f}G "
                      f"(measured encoder peak {measured / 2**30:.1f}G x{margin}, "
                      f"generic holdback {already_held / 2**30:.1f}G returned, "
                      f"floor {floor_gib:.0f}G)", flush=True)

        self_attn_block_bytes = (
            2 * num_self_attn_layers * BLOCK_SIZE * num_kv_heads * head_dim * elem_size
        )
        num_blocks = available_bytes // self_attn_block_bytes
        assert num_blocks > 0, f"Not enough GPU memory for KV cache on rank {self.rank}"
        if self.world_size > 1:
            # Agree on one size across ranks, as vLLM does (it takes the min of
            # each worker's available KV memory). Each rank derives its own
            # figure from its own mem_get_info and they disagree by ~1% in
            # practice; rank 0's block manager hands out ids from *its* count, so
            # a rank that sized fewer blocks than rank 0 would index past the end
            # of its cache.
            nb = torch.tensor([num_blocks], dtype=torch.int64, device="cuda")
            dist.all_reduce(nb, op=dist.ReduceOp.MIN)
            num_blocks = int(nb.item())
        self.num_blocks = num_blocks
        if self.rank == 0:
            print(f"  KV cache: {num_blocks} blocks x {BLOCK_SIZE} = {num_blocks * BLOCK_SIZE} token slots")
        elif self.is_qwen_vl:
            # Every rank sizes its own KV cache from its own mem_get_info, so a
            # rank that measured more free memory would quietly build a larger
            # cache and then run out during the (replicated) vision encoder.
            print(f"  [rank {self.rank}] KV cache: {num_blocks} blocks "
                  f"({num_blocks * BLOCK_SIZE} slots), "
                  f"available={available_bytes / 2**30:.1f}G", flush=True)

        if ATTN_BACKEND_CONFIG.kv_layout == "HND":
            self.kv_cache = torch.empty(
                2, num_self_attn_layers, num_blocks, num_kv_heads, BLOCK_SIZE, head_dim,
            )
        else:
            self.kv_cache = torch.empty(
                2, num_self_attn_layers, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim,
            )
        for i, module in enumerate(self._attn_layers):
            module.k_cache = self.kv_cache[0, i]
            module.v_cache = self.kv_cache[1, i]

        if self.rank == 0:
            cfg = ATTN_BACKEND_CONFIG
            print(f"  Attention backend: {cfg.backend} "
                  f"(block_size={cfg.block_size}, kv_layout={cfg.kv_layout})")


    def _allocate_variable_kv_cache(self):
        """Allocate per-layer KV caches for models with non-uniform KV shape.

        Every layer gets ``num_blocks`` blocks and every layer is indexed by the
        same ``seq.block_table``, so a sequence of length L holds ceil(L/16)
        blocks in *all* layers.

        This is a known divergence from vLLM for Gemma4, whose 25 of 30 layers
        are ``sliding_attention`` with a 1024-token window. vLLM puts those in a
        ``SlidingWindowSpec`` group whose ``SlidingWindowManager`` frees blocks
        that fall out of the window on every ``allocate_slots``, capping a
        request at ``cdiv(sliding_window - 1 + max_in_flight_tokens,
        block_size) + 1`` blocks in each sliding layer no matter how long the
        sequence gets. Holding the full length there instead costs capacity:
        498,272 token slots against vLLM's 1,862,998 on a B200 at
        gpu_memory_utilization=0.9.

        It costs *capacity only*, not correctness or compute:
          - the per-layer mask is still right (``sliding_window`` is passed
            through to ``window_size=(1023, 0)``), so the extra resident KV is
            simply never attended to;
          - vLLM's Triton unified-attention kernel bounds its tile loop by
            ``SLIDING_WINDOW`` (``loop_lo``/``loop_hi``), so the out-of-window
            blocks are not read either.

        Closing it needs per-layer-group block tables (one for the sliding
        group, one for the full group) threaded through the CUDA-graph capture,
        which today assumes a single ``block_tables`` tensor. Measured on the
        mixed scenario it would buy nothing (sequences average ~737 tokens, so
        nothing ever leaves the 1024 window) and on long-context the phase is
        89% prefill, so it only starts to pay at higher long-sequence
        concurrency than either workload reaches.
        """
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        elem_size = torch.finfo(torch.get_default_dtype()).bits // 8
        available_bytes = int(
            total * self.gpu_memory_utilization - used - peak + current
        )
        per_block_bytes = 0
        for layer in self._attn_layers:
            per_block_bytes += (
                2 * BLOCK_SIZE * layer.num_kv_heads * layer.head_size * elem_size
            )
        num_blocks = available_bytes // per_block_bytes
        assert num_blocks > 0, f"Not enough GPU memory for KV cache on rank {self.rank}"
        self.num_blocks = num_blocks
        self.kv_cache = []
        for layer in self._attn_layers:
            # Each layer allocates in the layout its own selected backend
            # indexes: trtllm-gen reads HND, FlashAttention and the Triton
            # unified kernel read NHD.  Gemma4 mixes both in one model.
            layer_layout = getattr(
                layer, "kv_layout", ATTN_BACKEND_CONFIG.kv_layout,
            )
            if layer_layout == "HND":
                cache = torch.empty(
                    2, num_blocks, layer.num_kv_heads, BLOCK_SIZE,
                    layer.head_size,
                )
            else:
                cache = torch.empty(
                    2, num_blocks, BLOCK_SIZE, layer.num_kv_heads,
                    layer.head_size,
                )
            layer.k_cache = cache[0]
            layer.v_cache = cache[1]
            self.kv_cache.append(cache)

        if self.rank == 0:
            print(
                f"  KV cache: {num_blocks} blocks x {BLOCK_SIZE} = "
                f"{num_blocks * BLOCK_SIZE} token slots (per-layer shapes)",
            )
            cfg = ATTN_BACKEND_CONFIG
            layouts = sorted({
                getattr(layer, "kv_layout", cfg.kv_layout)
                for layer in self._attn_layers
            })
            print(f"  Attention backend: {cfg.backend} "
                  f"(block_size={cfg.block_size}, "
                  f"kv_layout={'/'.join(layouts)})")

        if hasattr(self, '_warmup_encoder_cache'):
            del self._warmup_encoder_cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Mamba / SSM model support (slot-based recurrent state)
    # ------------------------------------------------------------------
    # Memory reserved for the CUDA-graph private pool (decode buckets).
    # Mirrors vLLM's ``profile_cudagraph_memory`` headroom, but without
    # the temp-graph capture pass: empirically ~1.0-1.5 GiB covers up to
    # 14 buckets at bs=256 for both Mamba v1 (2.8B) and Mamba2 (Codestral
    # 7B) on H200.  Tunable via env if a model needs more.
    _MAMBA_GRAPH_RESERVE_BYTES = int(
        os.environ.get("FASTKERNELS_MAMBA_GRAPH_RESERVE_BYTES", 1500 * 1024 * 1024)
    )

    def _compute_mamba_state_shapes(self):
        """Compute per-slot conv/ssm state shapes for the configured model.

        Factored out so both ``_profile_mamba_run`` (temp pool) and
        ``allocate_mamba_state_cache`` (final pool) use identical
        sizing -- mirrors how vLLM's MambaSpec is computed once and
        reused (``vllm/v1/attention/backends/mamba2_attn.py:get_kv_cache_shape``).

        Returns ``(conv_dim, ssm_state_shape, conv_kernel, per_slot_bytes)``.
        """
        cfg = self.config
        elem_size = torch.finfo(self.dtype).bits // 8

        if self.is_mamba2:
            n_groups = getattr(cfg, "n_groups", 1)
            tp = self.world_size
            if n_groups % tp != 0 and n_groups == 1:
                effective_groups = tp
            elif n_groups % tp != 0:
                raise ValueError(
                    f"Mamba2 n_groups={n_groups} not divisible by tp={tp}"
                )
            else:
                effective_groups = n_groups
            intermediate_size = getattr(
                cfg, "intermediate_size", cfg.num_heads * cfg.head_dim,
            )
            assert cfg.num_heads % tp == 0, (
                f"num_heads={cfg.num_heads} not divisible by tp={tp}"
            )
            num_heads_per_rank = cfg.num_heads // tp
            intermediate_per_rank = intermediate_size // tp
            groups_per_rank = max(1, effective_groups // tp)
            conv_dim = (
                intermediate_per_rank + 2 * groups_per_rank * cfg.state_size
            )
            ssm_state_shape = (
                num_heads_per_rank,
                cfg.head_dim,
                cfg.state_size,
            )
            per_layer_bytes = (
                conv_dim * cfg.conv_kernel
                + num_heads_per_rank * cfg.head_dim * cfg.state_size
            ) * elem_size
            conv_kernel = cfg.conv_kernel
        else:
            intermediate_size = cfg.intermediate_size
            conv_dim = intermediate_size
            state_size = getattr(cfg, "state_size", cfg.hidden_size)
            ssm_state_shape = (intermediate_size, state_size)
            conv_kernel = getattr(cfg, "conv_kernel", 4)
            per_layer_bytes = (
                conv_dim * conv_kernel + intermediate_size * state_size
            ) * elem_size

        return conv_dim, ssm_state_shape, conv_kernel, per_layer_bytes

    def _build_profile_mamba_metadata(self, n_seqs: int, tokens_per_seq: int):
        """Construct a synthetic prefill-only ``Mamba(2)Metadata`` for profiling.

        Mirrors ``_mamba_prepare_tensors`` for ``n_seqs`` prefill seqs
        of equal length, with state slot indices ``[0..n_seqs-1]``.
        """
        device = torch.device(f"cuda:{self.rank}")
        chunk_size = getattr(self.config, "chunk_size", 256)

        if self.is_mamba2:
            meta = Mamba2Metadata(chunk_size=chunk_size)
        else:
            meta = MambaMetadata()

        meta.num_prefill_tokens = n_seqs * tokens_per_seq
        meta.num_decode_tokens = 0
        meta.num_prefills = n_seqs
        meta.num_decodes = 0

        qsl = [i * tokens_per_seq for i in range(n_seqs + 1)]
        meta.query_start_loc_p = torch.tensor(
            qsl, dtype=torch.int32, device=device,
        )
        meta.state_indices_p = torch.arange(
            1, n_seqs + 1, dtype=torch.int32, device=device,
        )
        meta.has_initial_states_p = torch.zeros(
            n_seqs, dtype=torch.bool, device=device,
        )
        meta.prep_initial_states = False
        if not self.is_mamba2:
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(meta.query_start_loc_p)
            )
            meta.nums_dict = nums_dict
            meta.batch_ptr = batch_ptr
            meta.token_chunk_offset_ptr = token_chunk_offset_ptr

        if self.is_mamba2:
            num_computed = torch.zeros(n_seqs, dtype=torch.int32, device=device)
            cu_chunk, seq_idx, last_idx = build_chunk_metadata(
                meta.query_start_loc_p,
                chunk_size=chunk_size,
                num_computed_tokens_p=num_computed,
            )
            meta.cu_chunk_seqlen_p = cu_chunk
            meta.seq_idx_p = seq_idx
            meta.last_chunk_indices_p = last_idx

        return meta

    @torch.inference_mode()
    def _profile_mamba_run(self):
        """Measure peak activation memory of a worst-case Mamba prefill.

        Mirrors vLLM's ``GPUWorker.determine_available_memory`` ->
        ``GPUModelRunner.profile_run`` -> ``_dummy_run(max_num_tokens,
        is_profile=True)`` chain (``vllm/v1/worker/gpu_worker.py:401-441``,
        ``vllm/v1/worker/gpu_model_runner.py:5456-5528``).

        Allocates a *temporary* small Mamba state cache (just enough
        slots to host the synthetic batch) so the SSM kernels
        (``chunk_scan_combined_varlen``, ``selective_state_update``,
        ``causal_conv1d_fn``) are exercised on the real path -- skipping
        the mixer's profile/warmup branch which would otherwise
        underestimate the peak by ~10x for Mamba2 (the chunk-scan kernel,
        residual stack, and float32 ``silu(gate)`` cast in
        ``Mixer2RMSNormGated`` together dominate the activation
        working set at 16384 tokens).

        ``@torch.inference_mode`` is critical: without it the forward
        records an autograd graph, pinning every layer's intermediate
        activation and inflating the peak by 50-100x.

        Records ``self._profile_peak_bytes``.
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        n_tokens = int(self.max_num_batched_tokens)
        device = torch.device(f"cuda:{self.rank}")

        # Spread max_num_batched_tokens across multiple synthetic seqs so
        # the chunk-scan kernel sees a realistic ``query_start_loc``
        # layout.  vLLM's profile_run uses one synthetic seq per
        # max_num_seqs slot; we use a small constant since Mamba's
        # activation peak is dominated by total token count, not seq
        # count (chunk_scan operates over the flat token axis).
        n_seqs = max(1, min(8, n_tokens // 256))
        tokens_per_seq = n_tokens // n_seqs
        n_tokens = n_seqs * tokens_per_seq  # round to exact multiple

        # Allocate a temporary state pool (just n_seqs slots) so the
        # SSM kernels run on a real cache.  Mirrors vLLM's
        # ``_init_minimal_kv_cache_for_profiling``
        # (``vllm/v1/worker/gpu_model_runner.py:5531-5550``).
        conv_dim, ssm_state_shape, conv_kernel, _ = self._compute_mamba_state_shapes()
        temp_state = MambaStateManager(
            num_hidden_layers=self.config.num_hidden_layers,
            conv_dim=conv_dim,
            ssm_state_shape=ssm_state_shape,
            conv_kernel=conv_kernel,
            num_slots=n_seqs + 1,   # slot 0 is reserved
            dtype=self.dtype,
            device=device,
        )

        input_ids = torch.zeros(n_tokens, dtype=torch.int64, device=device)
        positions = torch.zeros(n_tokens, dtype=torch.int64, device=device)
        for i in range(n_seqs):
            positions[i * tokens_per_seq:(i + 1) * tokens_per_seq] = (
                torch.arange(tokens_per_seq, dtype=torch.int64, device=device)
            )

        meta = self._build_profile_mamba_metadata(n_seqs, tokens_per_seq)

        set_mamba_context(
            is_prefill=True,
            mamba_state=temp_state,
            mamba_metadata=meta,
        )
        try:
            hidden = self.model(input_ids, positions)
            # Run lm_head on the largest plausible logits batch so its
            # activation peak is also captured.  In real serving we
            # pick last-token-per-seq; here we just pass max_num_seqs
            # rows which is the upper bound.
            n_logits = min(self.max_num_seqs, n_tokens)
            logits_in = hidden[:n_logits]
            _ = self.model.compute_logits(logits_in)
            del logits_in
        finally:
            reset_context()
        torch.cuda.synchronize()

        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        self._profile_peak_bytes = max(0, int(peak - baseline_alloc))

        del input_ids, positions, hidden, meta, temp_state
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.rank == 0:
            print(
                f"  Mamba profile pass: activation peak "
                f"{self._profile_peak_bytes / (1<<20):.1f} MiB "
                f"(at {n_tokens} tokens, {n_seqs} synthetic prefill seqs)"
            )

    def _kimi_state_cache_bytes(self, num_slots: int) -> int:
        cfg = self.config
        local_heads = cfg.kda_num_heads // self.world_size
        local_proj = cfg.kda_num_heads * cfg.kda_head_dim // self.world_size
        kernel = cfg.short_conv_kernel_size
        dtype_bytes = torch.empty((), dtype=self.dtype).element_size()
        num_kda_layers = sum(
            1 for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)
        )
        conv_elems = 3 * num_kda_layers * num_slots * (kernel - 1) * local_proj
        recurrent_elems = (
            num_kda_layers
            * num_slots
            * local_heads
            * cfg.kda_head_dim
            * cfg.kda_head_dim
        )
        return conv_elems * dtype_bytes + recurrent_elems * 4

    def allocate_mamba_state_cache(self):
        """Size and allocate the global Mamba conv/ssm state pool.

        Tier 1A: profile-based sizing modeled on vLLM's
        ``determine_available_memory`` (``vllm/v1/worker/gpu_worker.py:
        349-441``):

        ``num_slots = (total*gpu_memory_utilization
                       - currently_allocated_bytes
                       - profile_activation_peak
                       - graph_reserve) / per_slot_bytes``

        ``currently_allocated_bytes`` covers model parameters; the
        profile peak (measured by ``_profile_mamba_run``) covers
        worst-case prefill activations; ``graph_reserve`` covers the
        CUDA-graph private pool.  Replaces the previous static
        ``state_cache_fraction`` heuristic.
        """
        if self.is_kimi_linear:
            usable_slots = max(1, self.max_num_seqs)
            use_decode_graph = not self.enforce_eager
            num_slots = usable_slots + (1 if use_decode_graph else 0)
            device = torch.device(f"cuda:{self.rank}")
            dtype = self.dtype
            mla_layer_count = sum(
                1
                for i in range(self.config.num_hidden_layers)
                if not self.config.is_kda_layer(i)
            )
            num_mla_blocks = self.num_blocks if mla_layer_count > 0 else 0

            self.mamba_state_manager = KimiLinearStateManager(
                config=self.config,
                num_slots=num_slots,
                block_size=BLOCK_SIZE,
                num_mla_blocks=num_mla_blocks,
                allocate_mla_kv_tensors=False,
                tp_size=self.world_size,
                device=device,
                dtype=dtype,
            )
            if use_decode_graph:
                self._kimi_pad_state_slot = usable_slots
                # ``range(1, ...)``: slot 0 is the null slot. The FLA recurrent
                # kernels and vLLM's causal-conv kernels both read state index 0
                # as "skip this sequence", so a sequence placed there gets NaN
                # output and a frozen recurrent state. Keep this in step with
                # ``KimiLinearStateManager.__init__``, which reserves it too --
                # this assignment replaces that deque wholesale.
                self.mamba_state_manager._free_slots = deque(range(1, usable_slots))
            else:
                self._kimi_pad_state_slot = self._KIMI_PAD_SLOT_ID
            self.num_state_slots = usable_slots
            self.max_num_seqs = min(self.max_num_seqs, usable_slots)
            if self.rank == 0:
                scratch = " + 1 scratch" if use_decode_graph else ""
                print(
                    f"  Kimi-Linear state cache: {usable_slots} sequence slots{scratch}, "
                    f"{num_mla_blocks} MLA blocks "
                    f"(x {BLOCK_SIZE} = {num_mla_blocks * BLOCK_SIZE} KV token slots, "
                    f"{mla_layer_count} MLA layers)"
                )
            return

        if self.is_qwen3_next:
            usable_slots = max(1, self.max_num_seqs)
            use_decode_graph = not self.enforce_eager
            num_slots = usable_slots + (1 if use_decode_graph else 0)
            device = torch.device(f"cuda:{self.rank}")
            dtype = self.dtype
            cfg = self.config
            qwen_block_size = self.block_size
            full_attn_layers = [
                i for i in range(cfg.num_hidden_layers)
                if not cfg.is_linear_attn_layer(i)
            ]

            local_kv_heads = (
                cfg.num_key_value_heads // self.world_size
                if cfg.num_key_value_heads % self.world_size == 0
                else cfg.num_key_value_heads
            )
            head_dim = getattr(
                cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads,
            )
            elem_size = torch.finfo(dtype).bits // 8
            block_bytes_total = (
                len(full_attn_layers)
                * qwen_block_size
                * 2
                * local_kv_heads
                * head_dim
                * elem_size
            )

            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            available_bytes = int(
                total * self.gpu_memory_utilization - used - peak + current
            )

            # Measure slotted GDN state before sizing the paged MHA block pool.
            self.mamba_state_manager = KimiLinearStateManager(
                config=cfg,
                num_slots=num_slots,
                block_size=qwen_block_size,
                num_mla_blocks=0,
                allocate_mla_kv_tensors=False,
                tp_size=self.world_size,
                device=device,
                dtype=dtype,
            )
            slotted_bytes = (
                torch.cuda.memory_stats()["allocated_bytes.all.current"] - current
            )
            del self.mamba_state_manager
            torch.cuda.empty_cache()

            kv_budget = max(0, available_bytes - slotted_bytes)
            num_mha_blocks = (
                max(1, kv_budget // block_bytes_total)
                if block_bytes_total > 0 else 0
            )
            self.mamba_state_manager = KimiLinearStateManager(
                config=cfg,
                num_slots=num_slots,
                block_size=qwen_block_size,
                num_mla_blocks=num_mha_blocks,
                allocate_mla_kv_tensors=True,
                tp_size=self.world_size,
                device=device,
                dtype=dtype,
            )
            if use_decode_graph:
                self._kimi_pad_state_slot = usable_slots
                # ``range(1, ...)``: slot 0 is the null slot. The FLA recurrent
                # kernels and vLLM's causal-conv kernels both read state index 0
                # as "skip this sequence", so a sequence placed there gets NaN
                # output and a frozen recurrent state. Keep this in step with
                # ``KimiLinearStateManager.__init__``, which reserves it too --
                # this assignment replaces that deque wholesale.
                self.mamba_state_manager._free_slots = deque(range(1, usable_slots))
                if self.mamba_state_manager._free_blocks is not None:
                    self._kimi_pad_block_id = self.mamba_state_manager._free_blocks.pop()
                else:
                    self._kimi_pad_block_id = self._KIMI_PAD_SLOT_ID
            else:
                self._kimi_pad_state_slot = self._KIMI_PAD_SLOT_ID
                self._kimi_pad_block_id = self._KIMI_PAD_SLOT_ID
            self.num_state_slots = usable_slots
            self.max_num_seqs = min(self.max_num_seqs, usable_slots)
            self.num_blocks = num_mha_blocks
            if self.rank == 0:
                scratch = " + 1 scratch" if use_decode_graph else ""
                kv_scratch = " (1 scratch block reserved)" if (
                    use_decode_graph and num_mha_blocks > 0
                ) else ""
                print(
                    f"  Qwen3-Next state cache: {usable_slots} sequence slots{scratch}, "
                    f"{num_mha_blocks} MHA blocks{kv_scratch} "
                    f"(x {qwen_block_size} = {num_mha_blocks * qwen_block_size} KV token slots, "
                    f"{len(full_attn_layers)} full-attn layers)"
                )
            return

        # ``free`` is what the OS sees as available; ``current_alloc`` is
        # what PyTorch's caching allocator reports as actively held
        # (including activations from previous transient ops that were
        # freed but cached).  We want to size against actually-needed
        # memory, not allocator cache; mirrors vLLM's approach which uses
        # ``allocated_bytes.all.peak`` rather than ``mem_get_info``
        # (``vllm/utils/mem_utils.py:252-281``,
        # ``vllm/v1/worker/gpu_worker.py:401-441``).
        free, total = torch.cuda.mem_get_info()
        current_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_layers = self.config.num_hidden_layers

        conv_dim, ssm_state_shape, conv_kernel, per_layer_bytes = (
            self._compute_mamba_state_shapes()
        )
        per_slot_bytes = num_layers * per_layer_bytes

        # Profile-based sizing (mirrors vLLM ``gpu_worker.py:437-441``):
        #   requested_bytes = total * gpu_memory_utilization
        #   non_kv_bytes    = current_alloc           # model params + persistent buffers
        #                   + non_torch_overhead       # NCCL workspace, custom AR, ...
        #                   + profile_peak_increase    # worst-case prefill activations
        #                   + graph_reserve            # CUDA-graph private pool
        #   budget          = requested_bytes - non_kv_bytes
        #
        # ``current_alloc`` (PyTorch allocator) is what the model + buffers
        # actually need; ``non_torch_overhead`` is the gap between
        # ``mem_get_info(used)`` and ``current_alloc``, which captures
        # things outside PyTorch's allocator (NCCL/Gloo workspaces,
        # fastsafetensors temp buffers, custom_ar staging, etc.).
        # ``profile_peak`` comes from ``_profile_mamba_run`` and is
        # already a *delta over baseline_alloc*, so it can be added on
        # top of ``current_alloc`` directly.
        profile_peak = getattr(self, "_profile_peak_bytes", 0)
        graph_reserve = self._MAMBA_GRAPH_RESERVE_BYTES
        requested_bytes = int(total * self.gpu_memory_utilization)
        non_torch_overhead = max(0, (total - free) - current_alloc)
        non_kv_bytes = (
            current_alloc + non_torch_overhead + profile_peak + graph_reserve
        )
        budget = max(0, requested_bytes - non_kv_bytes)
        # +1 because MambaStateManager reserves slot 0 (vLLM's prefill conv
        # kernel reads cache index 0 as "skip this sequence"), so a pool of N
        # tensors holds N-1 live sequences.
        capacity = budget // max(1, per_slot_bytes)
        num_slots = max(2, min(self.max_num_seqs + 1, capacity))

        if self.rank == 0:
            print(
                f"  Mamba memory budget: total={total / (1<<30):.1f} GiB, "
                f"requested={requested_bytes / (1<<30):.1f} GiB | "
                f"params={current_alloc / (1<<30):.2f} GiB, "
                f"non_torch={non_torch_overhead / (1<<30):.2f} GiB, "
                f"profile_peak={profile_peak / (1<<30):.2f} GiB, "
                f"graph_reserve={graph_reserve / (1<<30):.2f} GiB | "
                f"slot_budget={budget / (1<<30):.2f} GiB"
            )

        self.mamba_state_manager = MambaStateManager(
            num_hidden_layers=num_layers,
            conv_dim=conv_dim,
            ssm_state_shape=ssm_state_shape,
            conv_kernel=conv_kernel,
            num_slots=num_slots,
            dtype=self.dtype,
            device=torch.device(f"cuda:{self.rank}"),
        )
        self.num_state_slots = num_slots
        # Cap engine-level scheduling on slot count -- one slot per live seq,
        # minus the reserved slot 0.
        self.max_num_seqs = min(self.max_num_seqs, num_slots - 1)
        if self.rank == 0:
            print(f"  Mamba state cache: {num_slots - 1} sequence slots "
                  f"({per_slot_bytes / (1<<20):.1f} MiB/slot, slot 0 reserved)")

    def _init_fi_allreduce_fusion_workspace(self):
        """Create the FlashInfer fused-collective workspace before graph capture.

        ``fused_allreduce_add_gemma_rmsnorm`` creates it lazily on first use.
        That first use would otherwise be the eager warmup inside
        ``capture_kimi_cudagraph``, and the workspace (plus the zero-residual
        buffer it allocates) must not land in a graph's private memory pool.
        Doing it here also means a topology without NVSwitch multicast decides
        the fallback once, at startup, rather than per step.
        """
        if self.world_size <= 1:
            return
        from ..tasks.baseline.L2.flashinfer_allreduce_fusion import (
            get_fi_ar_workspace,
        )

        get_fi_ar_workspace(self.config.hidden_size, self.dtype)

    def can_allocate_mamba_state(self):
        return self.mamba_state_manager is not None and \
            self.mamba_state_manager.has_free_slot()

    def allocate_mamba_state(self, seq):
        return self.mamba_state_manager.allocate(seq)

    def deallocate_mamba_state(self, seq):
        if self.mamba_state_manager is not None:
            self.mamba_state_manager.deallocate(seq)

    def reset_kimi_state_cache(self):
        if (
            (self.is_kimi_linear or self.is_qwen3_next)
            and self.mamba_state_manager is not None
        ):
            sm = self.mamba_state_manager
            pad_slot = getattr(self, "_kimi_pad_state_slot", self._KIMI_PAD_SLOT_ID)
            pad_block = getattr(self, "_kimi_pad_block_id", self._KIMI_PAD_SLOT_ID)
            # Skip slot 0 as well as the pad slot: the FLA recurrent kernels and
            # vLLM's causal-conv kernels read state index 0 as "skip this
            # sequence" (``NULL_BLOCK_ID`` is 0), so a sequence placed there gets
            # NaN output and a recurrent state that never advances -- measured on
            # ``fused_recurrent_kda``: slot 0 -> NaN with zero state delta, slots
            # 1 and 3 -> clean with delta ~0.088. vLLM's block allocator reserves
            # block 0 for the same reason.
            #
            # The pristine order is fixed for the life of the engine, so it is
            # built once and copied. Re-running the generator expression cost
            # 8.0 ms per ``generate()`` at the long-context KV size (239k
            # blocks) against 1.4 ms to refill from a cached list, and this runs
            # on every rank on every call.
            key = (sm.num_slots, pad_slot, sm.num_mla_blocks, pad_block)
            if getattr(self, "_kimi_free_pool_key", None) != key:
                self._kimi_free_slot_pool = [
                    slot for slot in range(1, sm.num_slots) if slot != pad_slot
                ]
                self._kimi_free_block_pool = [
                    block for block in range(sm.num_mla_blocks)
                    if block != pad_block
                ]
                self._kimi_free_pool_key = key
            sm._free_slots = deque(self._kimi_free_slot_pool)
            sm._in_use.clear()
            if sm._free_blocks is not None:
                sm._free_blocks = deque(self._kimi_free_block_pool)

    def empty_cuda_cache(self):
        """Drop the PyTorch caching allocator's cached blocks.

        Called via ``mr.call`` between ``generate()`` invocations to
        mirror vLLM's per-step ``empty_cache`` and avoid cumulative
        fragmentation across benchmark scenarios.  Returns the freed
        bytes for diagnostics.
        """
        before = torch.cuda.memory_stats()["reserved_bytes.all.current"]
        torch.cuda.empty_cache()
        after = torch.cuda.memory_stats()["reserved_bytes.all.current"]
        return before - after

    def allocate_mamba_state_batch(self, n: int) -> list[int]:
        """Pop ``n`` free slots in deque order on every rank.

        Called from ``mr.call(...)`` so all TP ranks pop the same
        ``n`` slots in lock-step (the deque is identical across
        ranks).  Batched into a single SHM message to avoid the
        race conditions seen with per-seq ``mr.call`` invocations
        (``_pickle.UnpicklingError`` when the worker spin-loop
        outruns the writer).
        """
        sm = self.mamba_state_manager
        slots: list[int] = []
        for _ in range(n):
            if not sm._free_slots:
                raise RuntimeError("No free Mamba state slots")
            slot = sm._free_slots.popleft()
            sm._in_use.add(slot)
            slots.append(slot)
        # One indexed zero-fill per state tensor for the whole batch: a fresh
        # sequence's first chunk runs with ``has_initial_states=False`` so the
        # kernels overwrite rather than read, but zeroing on claim keeps the
        # "a live slot only ever holds its own sequence's state" invariant.
        sm.reset_slots(slots)
        return slots

    def deallocate_mamba_state_batch(self, slot_ids: list[int] | list[tuple[int, list[int]]]) -> None:
        """Return specific slots to the free pool on every rank.

        Batched counterpart of ``allocate_mamba_state_batch`` -- all
        ranks remove the same slot ids from ``_in_use`` and re-append
        them to ``_free_slots`` in lock-step.

        Freed slots are left dirty; ``allocate_mamba_state_batch`` zeroes them
        when they are claimed again, so zeroing here would just double the work
        on the hottest path (every finishing sequence).
        """
        sm = self.mamba_state_manager
        if sm is None:
            return
        for item in slot_ids:
            if isinstance(item, tuple):
                slot, block_table = item
            else:
                slot, block_table = item, []
            if slot in sm._in_use:
                sm._in_use.remove(slot)
                sm._free_slots.append(slot)
            if getattr(sm, "_free_blocks", None) is not None and block_table:
                sm._free_blocks.extend(block_table)

    def allocate_kimi_mla_blocks_batch(self, new_block_counts: list[int]) -> list[list[int]]:
        """Pop paged-attention KV cache blocks for hybrid decode on every rank."""
        sm = self.mamba_state_manager
        if (
            sm is None
            or not (self.is_kimi_linear or self.is_qwen3_next)
            or sm._free_blocks is None
        ):
            return [[] for _ in new_block_counts]
        out: list[list[int]] = []
        for count in new_block_counts:
            blocks: list[int] = []
            for _ in range(int(count)):
                if not sm._free_blocks:
                    raise RuntimeError("No free MLA KV cache blocks")
                blocks.append(sm._free_blocks.popleft())
            out.append(blocks)
        return out

    # ------------------------------------------------------------------
    # Hybrid recurrent decode CUDA-graph fast path
    # ------------------------------------------------------------------
    _KIMI_PAD_SLOT_ID = -1

    def _pad_page_table_cols(self, n_cols: int) -> int:
        """Round a page-table width up to the trtllm-gen MLA granularity.

        ``flashinfer``'s ``trtllm_batch_decode_with_kv_cache_mla`` rejects a
        page table whose column count is not a multiple of
        ``ceil(128 / page_size)`` ("Expected block_num % (128 / block_size)
        == 0").  The width follows ``max_model_len``, which the harness derives
        from the dataset, so without this a run trips the check for some
        prompt sets and not others.  The extra columns hold the pad block id
        and are never read -- the kernel bounds its walk by ``seq_lens``.
        """
        if not (self.is_kimi_linear or self.is_deepseek_mla):
            return n_cols
        block_size = self.block_size or 1
        gran = max(1, -(-128 // block_size))
        return ((n_cols + gran - 1) // gran) * gran

    def _init_kimi_decode_buffers(self):
        """Pre-allocate persistent buffers for hybrid decode CUDA graphs."""
        max_bs = self.max_num_seqs
        block_size = self.mamba_state_manager.block_size
        # max blocks any seq could possibly need at decode time
        max_blocks = self._pad_page_table_cols(
            (self.max_model_len + block_size - 1) // block_size,
        )
        dev = f"cuda:{self.rank}"

        self._kd_max_bs = max_bs
        self._kd_block_size = block_size
        self._kd_max_blocks = max_blocks
        pad_state_slot = getattr(
            self, "_kimi_pad_state_slot", self._KIMI_PAD_SLOT_ID,
        )

        # GPU side persistent buffers (graph captured)
        self._kd_input_ids = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._kd_positions = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._kd_state_indices = torch.full(
            (max_bs,), pad_state_slot, dtype=torch.int32, device=dev,
        )
        self._kd_seq_lens = torch.ones(max_bs, dtype=torch.int32, device=dev)
        self._kd_has_init = torch.ones(max_bs, dtype=torch.bool, device=dev)
        # MLA path uses int32 slot in KimiLinearMetadata, int64 in attention ctx
        self._kd_slot_mapping_int32 = torch.full(
            (max_bs,), self._KIMI_PAD_SLOT_ID, dtype=torch.int32, device=dev,
        )
        self._kd_slot_mapping_int64 = torch.full(
            (max_bs,), self._KIMI_PAD_SLOT_ID, dtype=torch.int64, device=dev,
        )
        self._kd_block_tables = torch.full(
            (max_bs, max_blocks), self._KIMI_PAD_SLOT_ID,
            dtype=torch.int32, device=dev,
        )
        # Decode is 1-token-per-seq, so cu_seqlens_q = arange(max_bs+1)
        self._kd_cu_seqlens_q = torch.arange(
            max_bs + 1, dtype=torch.int32, device=dev,
        )
        # Logit indices for decode: position of last token of each seq's slice
        # = arange(max_bs).
        self._kd_logit_idx = torch.arange(max_bs, dtype=torch.int64, device=dev)

        # CPU pinned staging for fast async copy
        self._kd_input_ids_cpu = torch.zeros(
            max_bs, dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._kd_positions_cpu = torch.zeros(
            max_bs, dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._kd_state_indices_cpu = torch.full(
            (max_bs,), pad_state_slot,
            dtype=torch.int32, device="cpu", pin_memory=True,
        )
        self._kd_seq_lens_cpu = torch.ones(
            max_bs, dtype=torch.int32, device="cpu", pin_memory=True,
        )
        self._kd_slot_mapping_int32_cpu = torch.full(
            (max_bs,), self._KIMI_PAD_SLOT_ID,
            dtype=torch.int32, device="cpu", pin_memory=True,
        )
        self._kd_slot_mapping_int64_cpu = torch.full(
            (max_bs,), self._KIMI_PAD_SLOT_ID,
            dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._kd_block_tables_cpu = torch.full(
            (max_bs, max_blocks), self._KIMI_PAD_SLOT_ID,
            dtype=torch.int32, device="cpu", pin_memory=True,
        )
        self._kd_input_ids_np = self._kd_input_ids_cpu.numpy()
        self._kd_positions_np = self._kd_positions_cpu.numpy()
        self._kd_state_indices_np = self._kd_state_indices_cpu.numpy()
        self._kd_seq_lens_np = self._kd_seq_lens_cpu.numpy()
        self._kd_slot_mapping_int32_np = self._kd_slot_mapping_int32_cpu.numpy()
        self._kd_slot_mapping_int64_np = self._kd_slot_mapping_int64_cpu.numpy()
        self._kd_block_tables_np = self._kd_block_tables_cpu.numpy()

        # Mark all as static for cudagraph
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "mark_static_address"):
            for t in [
                self._kd_input_ids, self._kd_positions, self._kd_state_indices,
                self._kd_seq_lens, self._kd_has_init, self._kd_slot_mapping_int32,
                self._kd_slot_mapping_int64, self._kd_block_tables,
                self._kd_cu_seqlens_q, self._kd_logit_idx,
            ]:
                torch._dynamo.mark_static_address(t)

        # Will be filled in by capture_kimi_cudagraph
        self._kimi_graphs: dict = {}
        self._kimi_graph_metas: dict = {}
        self._kimi_graph_pool = None
        self._kimi_graph_bs_list: list[int] = []
        self._kimi_graph_bs_for_n: list[int] | None = None
        # Output buffers: model produces hidden_states of shape (n, hidden);
        # then we extract last_hidden of shape (n, hidden). Persistent so
        # graphs can write to it.
        hidden_size = self.config.hidden_size
        self._kd_last_hidden = torch.zeros(
            max_bs, hidden_size,
            dtype=self.dtype, device=dev,
        )
        self._kd_lm_max_vals = torch.zeros(
            max_bs, dtype=torch.float32, device=dev,
        )
        self._kd_lm_max_idxs = torch.zeros(
            max_bs, dtype=torch.int64, device=dev,
        )
        self._kd_pinned_token_ids = torch.empty(
            max_bs, dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._kd_copy_stream = torch.cuda.Stream(device=dev)
        self._kd_copy_event = torch.cuda.Event()
        self._kd_greedy_info = torch.zeros(
            max_bs, 2, dtype=torch.float32, device=dev,
        )
        self._kd_greedy_gathered = [
            torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
            for _ in range(self.world_size)
        ]
        self._kd_greedy_all_info = torch.zeros(
            self.world_size, max_bs, 2, dtype=torch.float32, device=dev,
        )
        self._kd_greedy_arange = torch.arange(max_bs, device=dev)
        # Allocate before graph capture and reuse it for every bulk chunk. A
        # fresh allocation here can alias the CUDA graph private pool.
        self._kd_bulk_outputs = torch.zeros(
            _BULK_CHUNK_CAP,
            max_bs,
            dtype=torch.int64,
            device=dev,
        )

    def _prepare_kimi_decode_arrays(self, seqs, copy_block_tables: bool = True):
        """Fill pinned staging buffers for hybrid greedy decode fast path."""
        n = len(seqs)
        ids = self._kd_input_ids_np
        pos = self._kd_positions_np
        si = self._kd_state_indices_np
        sl = self._kd_seq_lens_np
        slot32 = self._kd_slot_mapping_int32_np
        slot64 = self._kd_slot_mapping_int64_np
        bt = self._kd_block_tables_np
        max_blocks = self._kd_max_blocks
        block_size = self._kd_block_size

        for i, seq in enumerate(seqs):
            ids[i] = seq.last_token
            start_pos = seq.num_computed_tokens
            pos[i] = start_pos
            si[i] = seq.state_slot
            seq_total = start_pos + 1
            sl[i] = seq_total
            block_idx = start_pos // block_size
            if block_idx < len(seq.block_table):
                block_id = seq.block_table[block_idx]
                slot = block_id * block_size + (start_pos % block_size)
            else:
                slot = self._KIMI_PAD_SLOT_ID
            slot32[i] = slot
            slot64[i] = slot
            if copy_block_tables:
                row = seq.block_table
                n_blocks = len(row)
                if n_blocks > 0:
                    bt[i, :n_blocks] = row
                if n_blocks < max_blocks:
                    bt[i, n_blocks:max_blocks] = self._KIMI_PAD_SLOT_ID

        return (
            n,
            ids[:n],
            pos[:n],
            si[:n],
            sl[:n],
            slot32[:n],
            bt[:n, :max_blocks if copy_block_tables else 0],
            copy_block_tables,
        )

    @torch.inference_mode()
    def capture_kimi_cudagraph(self):
        """Capture decode-only CUDA graphs for hybrid recurrent models.

        Mirrors capture_mamba_cudagraph. Each bucket records:
          model.forward(input_ids, positions, state_manager) ->
          last_hidden[arange(bs)] ->
          lm_head.project(last_hidden) -> local logits

        TP cross-rank gather + sampling happens OUTSIDE the graph in
        ``_run_kimi_decode_graph_greedy``.
        """
        from contextlib import nullcontext

        graph_max_default = self.max_num_seqs
        max_bs = min(self.max_num_seqs, graph_max_default)
        # Same bucket shape as vLLM's ``cudagraph_capture_sizes`` and as our own
        # ``graph_bs_list`` for the non-hybrid path: [1, 2, 4] then by 8 up to
        # 256, then by 16. The previous list stepped 16 -> 32 -> 48 -> 64 -> 96
        # -> 128 -> 160, so a decode batch of 100 padded to 128 (28% of the work
        # wasted) where vLLM pads to 104 (4%).
        bs_candidates = [i for i in (1, 2, 4) if i <= max_bs]
        if max_bs >= 8:
            bs_candidates += list(range(8, min(max_bs + 1, 256), 8))
        if max_bs >= 256:
            bs_candidates += list(range(256, max_bs + 1, 16))
        bs_list = sorted(set(bs_candidates))
        bs_list = [b for b in bs_list if b <= max_bs]
        if bs_list and bs_list[-1] != max_bs:
            bs_list.append(max_bs)
        if not bs_list:
            return
        self._kimi_graph_bs_list = bs_list

        sm_ = self.mamba_state_manager
        max_blocks = self._kd_max_blocks
        block_size = self._kd_block_size
        pad_state_slot = getattr(
            self, "_kimi_pad_state_slot", self._KIMI_PAD_SLOT_ID,
        )
        pad_block_id = getattr(
            self, "_kimi_pad_block_id", self._KIMI_PAD_SLOT_ID,
        )
        pad_kv_slot = (
            pad_block_id * block_size
            if pad_block_id != self._KIMI_PAD_SLOT_ID
            else self._KIMI_PAD_SLOT_ID
        )
        ar_ctx = (
            self.custom_ar.capture()
            if self.custom_ar is not None else nullcontext()
        )
        with ar_ctx:
            for bs in reversed(bs_list):
                # Prepare static views into the persistent buffers
                input_ids_v   = self._kd_input_ids[:bs]
                positions_v   = self._kd_positions[:bs]
                state_idx_v   = self._kd_state_indices[:bs]
                seq_lens_v    = self._kd_seq_lens[:bs]
                has_init_v    = self._kd_has_init[:bs]
                slot32_v      = self._kd_slot_mapping_int32[:bs]
                slot64_v      = self._kd_slot_mapping_int64[:bs]
                bt_v          = self._kd_block_tables[:bs, :max_blocks]
                cu_q_v        = self._kd_cu_seqlens_q[:bs + 1]
                logit_idx_v   = self._kd_logit_idx[:bs]

                # Reset to PAD so warmup doesn't index real sequence state.
                state_idx_v.fill_(pad_state_slot)
                slot32_v.fill_(pad_kv_slot)
                slot64_v.fill_(pad_kv_slot)
                bt_v.fill_(pad_block_id)
                input_ids_v.zero_()
                positions_v.zero_()
                seq_lens_v.fill_(1)
                has_init_v.fill_(True)

                md = KimiLinearMetadata(
                    num_actual_tokens=bs,
                    query_start_loc=cu_q_v,
                    max_query_len=1,
                    seq_lens=seq_lens_v,
                    max_seq_len=self.max_model_len,
                    state_indices=state_idx_v,
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decodes=bs,
                    num_decode_tokens=bs,
                    has_initial_state=has_init_v,
                    slot_mapping=slot32_v,
                    block_tables=bt_v,
                    # ``cu_q_v`` is already int32, so this only spares the
                    # layers a no-op ``.to()`` call; decode never needs the
                    # int64 state indices.
                    query_start_loc_int32=cu_q_v,
                    all_have_initial_state=True,
                    any_have_initial_state=True,
                )
                self._kimi_graph_metas[bs] = md

                set_context(
                    False,
                    cu_seqlens_q=cu_q_v,
                    cu_seqlens_k=cu_q_v,
                    max_seqlen_q=1,
                    max_seqlen_k=self.max_model_len,
                    slot_mapping=slot64_v,
                    context_lens=seq_lens_v,
                    block_tables=bt_v,
                    max_context_len=self.max_model_len,
                )
                ctx = get_context()
                ctx.kda_state = sm_
                ctx.kda_metadata = md
                self._refresh_fa3_decode_schedule(seq_lens_v)

                # Mark the batch dim dynamic BEFORE the first
                # compile-triggering forward. ``compile_model`` traces under
                # ``skip_all_guards_unsafe``, so without this the first bucket's
                # batch size is baked into the compiled subgraphs and silently
                # reused at every other size -- which surfaced as
                # "Expected batch size 1024 for query and block_table, got 1024
                # and 1008" when capture reached a different bucket. The mamba
                # and attention capture paths already do this; this one did not,
                # which is what blocked compiling Kimi-Linear.
                if self._compiled and not self._mark_dynamic_done:
                    # Every buffer sliced by ``bs`` has to be marked, not just
                    # input_ids: leaving any of them at a static size makes the
                    # shape solver reject the dynamic dim outright
                    # ("ConstraintViolationError ... L['input_ids'].size()[0]").
                    # ``cu_q_v`` is [:bs+1], a different size expression, so it
                    # needs marking too or the solver cannot relate s0+1 to s0.
                    for _t in (input_ids_v, positions_v, state_idx_v,
                               seq_lens_v, has_init_v, slot32_v, slot64_v,
                               bt_v, cu_q_v, logit_idx_v):
                        torch._dynamo.mark_dynamic(_t, 0)
                    self._mark_dynamic_done = True

                # Warmup eager so kernels autotune outside the graph
                hidden = self.model(input_ids_v, positions=positions_v, state_manager=sm_)
                last_h = hidden.index_select(0, logit_idx_v)
                self._kd_last_hidden[:bs].copy_(last_h)
                logits_local = self.model.lm_head.project(self._kd_last_hidden[:bs])
                max_vals, max_idxs = logits_local.max(dim=-1)
                self._kd_lm_max_vals[:bs].copy_(max_vals.float())
                self._kd_lm_max_idxs[:bs].copy_(max_idxs)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self._kimi_graph_pool):
                    hidden = self.model(input_ids_v, positions=positions_v, state_manager=sm_)
                    last_h = hidden.index_select(0, logit_idx_v)
                    self._kd_last_hidden[:bs].copy_(last_h)
                    logits_local = self.model.lm_head.project(
                        self._kd_last_hidden[:bs],
                    )
                    max_vals, max_idxs = logits_local.max(dim=-1)
                    self._kd_lm_max_vals[:bs].copy_(max_vals.float())
                    self._kd_lm_max_idxs[:bs].copy_(max_idxs)

                if self._kimi_graph_pool is None:
                    self._kimi_graph_pool = graph.pool()
                self._kimi_graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

        # Build bucket lookup: smallest bucket >= n
        max_bucket = self._kimi_graph_bs_list[-1]
        self._kimi_graph_bs_for_n = [0] * (self.max_num_seqs + 1)
        for n in range(self.max_num_seqs + 1):
            self._kimi_graph_bs_for_n[n] = next(
                (x for x in self._kimi_graph_bs_list if x >= n),
                max_bucket,
            )
        if self.rank == 0:
            graph_label = "Qwen3-Next" if self.is_qwen3_next else "Kimi"
            print(
                f"  {graph_label} CUDA graphs: {len(self._kimi_graphs)} buckets "
                f"(min={bs_list[0]}, max={bs_list[-1]})",
                flush=True,
            )

    def _stage_kimi_decode_graph_inputs(
        self, n: int, copy_block_tables: bool = True,
    ) -> int:
        """Copy CPU-staged hybrid decode metadata into graph input buffers."""
        bucket = self._kimi_graph_bs_for_n[n]
        max_blocks = self._kd_max_blocks
        block_size = self._kd_block_size
        pad_state_slot = getattr(
            self, "_kimi_pad_state_slot", self._KIMI_PAD_SLOT_ID,
        )
        pad_block_id = getattr(
            self, "_kimi_pad_block_id", self._KIMI_PAD_SLOT_ID,
        )
        pad_kv_slot = (
            pad_block_id * block_size
            if pad_block_id != self._KIMI_PAD_SLOT_ID
            else self._KIMI_PAD_SLOT_ID
        )

        if bucket > n:
            self._kd_input_ids_cpu[n:bucket].zero_()
            self._kd_positions_cpu[n:bucket].zero_()
            self._kd_state_indices_cpu[n:bucket].fill_(pad_state_slot)
            self._kd_seq_lens_cpu[n:bucket].fill_(1)
            self._kd_slot_mapping_int32_cpu[n:bucket].fill_(pad_kv_slot)
            self._kd_slot_mapping_int64_cpu[n:bucket].fill_(pad_kv_slot)
            if copy_block_tables:
                self._kd_block_tables_np[n:bucket, :max_blocks] = pad_block_id

        self._kd_input_ids[:bucket].copy_(
            self._kd_input_ids_cpu[:bucket], non_blocking=True,
        )
        self._kd_positions[:bucket].copy_(
            self._kd_positions_cpu[:bucket], non_blocking=True,
        )
        self._kd_state_indices[:bucket].copy_(
            self._kd_state_indices_cpu[:bucket], non_blocking=True,
        )
        self._kd_seq_lens[:bucket].copy_(
            self._kd_seq_lens_cpu[:bucket], non_blocking=True,
        )
        self._kd_slot_mapping_int32[:bucket].copy_(
            self._kd_slot_mapping_int32_cpu[:bucket], non_blocking=True,
        )
        self._kd_slot_mapping_int64[:bucket].copy_(
            self._kd_slot_mapping_int64_cpu[:bucket], non_blocking=True,
        )
        if copy_block_tables:
            self._kd_block_tables[:bucket, :max_blocks].copy_(
                self._kd_block_tables_cpu[:bucket, :max_blocks], non_blocking=True,
            )
        return bucket

    @torch.inference_mode()
    def _replay_kimi_decode_graph_greedy(
        self, n: int, return_all_ranks: bool = False,
    ):
        """Replay already-staged hybrid decode graph and return greedy token ids."""
        bucket = self._kimi_graph_bs_for_n[n]
        self._refresh_fa3_decode_schedule(self._kd_seq_lens[:bucket])
        self._kimi_graphs[bucket].replay()

        if self.world_size == 1:
            return self._kd_lm_max_idxs[:n]
        return self._kimi_greedy_gather(
            n, self._kd_lm_max_vals[:n], self._kd_lm_max_idxs[:n],
            return_all_ranks=return_all_ranks,
        )

    @torch.inference_mode()
    def _run_kimi_decode_graph_greedy(
        self, n: int, copy_block_tables: bool = True,
        return_all_ranks: bool = False,
    ):
        """Stage inputs, replay hybrid decode graph, and return greedy token ids."""
        self._stage_kimi_decode_graph_inputs(n, copy_block_tables)
        return self._replay_kimi_decode_graph_greedy(
            n, return_all_ranks=return_all_ranks,
        )

    def _kimi_greedy_gather(
        self, n: int, max_vals: torch.Tensor, max_idxs: torch.Tensor,
        return_all_ranks: bool = False,
    ):
        """TP cross-rank greedy gather without materializing full logits."""
        lm_head = self.model.lm_head
        info = self._kd_greedy_info[:n]
        info[:, 0] = max_vals
        info[:, 1] = max_idxs.float()
        info[:, 1] += lm_head.per_partition * self.rank

        gathered = [g[:n] for g in self._kd_greedy_gathered]
        dist.all_gather(gathered, info)
        if self.rank != 0 and not return_all_ranks:
            return None
        all_info = self._kd_greedy_all_info[:, :n]
        torch.stack(gathered, out=all_info)
        best_rank = all_info[:, :n, 0].argmax(dim=0)
        return all_info[best_rank, self._kd_greedy_arange[:n], 1].long()

    @torch.inference_mode()
    def run_kimi_decode_fast_async(self, decode_data):
        """Greedy hybrid decode graph + async D2H copy of token IDs."""
        n = decode_data[0]
        copy_block_tables = bool(decode_data[-1]) if len(decode_data) > 1 else True
        token_ids = self._run_kimi_decode_graph_greedy(n, copy_block_tables)
        if token_ids is None:
            return False, n
        main_stream = torch.cuda.current_stream()
        cs = self._kd_copy_stream
        with torch.cuda.stream(cs):
            cs.wait_stream(main_stream)
            self._kd_pinned_token_ids[:n].copy_(token_ids, non_blocking=True)
            self._kd_copy_event.record(cs)
        return True, n

    def _wait_async_kimi_tokens(self, n: int) -> list[int]:
        self._kd_copy_event.synchronize()
        return self._kd_pinned_token_ids[:n].tolist()

    @torch.inference_mode()
    def run_kimi_decode_many(self, seqs, steps: int):
        """Run fixed-batch greedy decode for ``steps`` tokens on all ranks."""
        if not seqs or steps <= 0:
            return [] if self.rank == 0 else None

        for seq in seqs:
            self.mamba_state_manager.ensure_blocks_for(
                seq,
                seq.num_computed_tokens + steps,
            )

        decode_data = self._prepare_kimi_decode_arrays(
            seqs, copy_block_tables=True,
        )
        n = decode_data[0]
        self._stage_kimi_decode_graph_inputs(n, copy_block_tables=True)

        outputs_dev = (
            self._kd_bulk_outputs[:steps, :n]
            if self.rank == 0 else None
        )
        arange = self._kd_greedy_arange[:n]
        block_size = int(self._kd_block_size)
        start_positions = [seq.num_computed_tokens for seq in seqs]
        uniform_positions = min(start_positions) == max(start_positions)
        start_pos0 = start_positions[0]
        for step in range(steps):
            # Keep the decode loop device-resident. Pulling token ids back to
            # CPU every step is a hard sync and dominates long fixed-batch decode.
            token_ids_t = self._replay_kimi_decode_graph_greedy(
                n, return_all_ranks=True,
            )
            if outputs_dev is not None:
                outputs_dev[step].copy_(token_ids_t)
            if step + 1 < steps:
                self._kd_input_ids[:n].copy_(token_ids_t)
                self._kd_positions[:n].add_(1)
                self._kd_seq_lens[:n].add_(1)
                if uniform_positions:
                    self._kd_slot_mapping_int32[:n].add_(1)
                    self._kd_slot_mapping_int64[:n].add_(1)
                    next_pos = start_pos0 + step + 1
                    if next_pos % block_size == 0:
                        block_idx = next_pos // block_size
                        block_ids = self._kd_block_tables[arange, block_idx]
                        slots = block_ids * block_size
                        self._kd_slot_mapping_int32[:n].copy_(
                            slots.to(torch.int32),
                        )
                        self._kd_slot_mapping_int64[:n].copy_(
                            slots.to(torch.int64),
                        )
                else:
                    block_idx = torch.div(
                        self._kd_positions[:n], block_size, rounding_mode="floor",
                    )
                    block_ids = self._kd_block_tables[arange, block_idx]
                    slots = block_ids * block_size + (
                        self._kd_positions[:n] - block_idx * block_size
                    )
                    self._kd_slot_mapping_int32[:n].copy_(slots.to(torch.int32))
                    self._kd_slot_mapping_int64[:n].copy_(slots.to(torch.int64))

        if self.rank == 0:
            outputs = outputs_dev.transpose(0, 1).cpu().tolist()
        else:
            outputs = None

        for i, seq in enumerate(seqs):
            if self.rank == 0:
                toks = outputs[i]
                if seq.token_ids is not None:
                    seq.token_ids.extend(toks)
                    seq.generated_ids.extend(toks)
                else:
                    seq._last_token = toks[-1]
                    seq._num_tokens += len(toks)
            else:
                if seq.token_ids is None:
                    seq._num_tokens += steps
            seq.num_computed_tokens += steps
        return outputs if self.rank == 0 else None

    def _write_kimi_decode_shm(
        self, n, ids_np, pos_np, si_np, sl_np, slot_np, bt_np,
        copy_block_tables: bool,
    ):
        """Pack a hybrid decode batch into SHM for the worker ranks."""
        max_blocks = bt_np.shape[1] if copy_block_tables else 0
        payload_bytes = (
            4
            + n * 8
            + n * 8
            + n * 4
            + n * 4
            + n * 4
            + (n * max_blocks * 4 if copy_block_tables else 0)
        )
        if payload_bytes > self._SHM_ACK_OFFSET:
            raise RuntimeError(
                f"Hybrid decode SHM payload too large: {payload_bytes} bytes",
            )

        buf = self.shm.buf
        buf[0:2] = n.to_bytes(2, "little")
        buf[2:4] = max_blocks.to_bytes(2, "little")
        off = 4
        arrays = (ids_np, pos_np, si_np, sl_np, slot_np)
        for arr in arrays:
            raw = arr.tobytes()
            buf[off:off + len(raw)] = raw
            off += len(raw)
        if copy_block_tables:
            raw = bt_np.tobytes()
            buf[off:off + len(raw)] = raw

    @torch.inference_mode()
    def _loop_kimi_decode_greedy(self, cur_seq):
        """Worker fast path for hybrid greedy decode."""
        buf = self.shm.buf
        n = int.from_bytes(buf[0:2], "little")
        max_blocks = int.from_bytes(buf[2:4], "little")
        off = 4
        ids = np.frombuffer(buf, dtype=np.int64, count=n, offset=off)
        off += n * 8
        pos = np.frombuffer(buf, dtype=np.int64, count=n, offset=off)
        off += n * 8
        si = np.frombuffer(buf, dtype=np.int32, count=n, offset=off)
        off += n * 4
        sl = np.frombuffer(buf, dtype=np.int32, count=n, offset=off)
        off += n * 4
        slot = np.frombuffer(buf, dtype=np.int32, count=n, offset=off)
        off += n * 4
        bt = None
        if max_blocks > 0:
            bt = np.frombuffer(
                buf, dtype=np.int32, count=n * max_blocks, offset=off,
            ).reshape(n, max_blocks)

        self._kd_input_ids_np[:n] = ids
        self._kd_positions_np[:n] = pos
        self._kd_state_indices_np[:n] = si
        self._kd_seq_lens_np[:n] = sl
        self._kd_slot_mapping_int32_np[:n] = slot
        self._kd_slot_mapping_int64_np[:n] = slot
        if max_blocks > 0:
            self._kd_block_tables_np[:n, :max_blocks] = bt
        self._ack_consumed(cur_seq)
        self.run_kimi_decode_fast_async((n, max_blocks > 0))

    @torch.inference_mode()
    def call_kimi_decode_async(self, decode_data):
        """Launch a greedy hybrid decode from precomputed staging arrays."""
        if self.world_size > 1 and self.rank == 0:
            self._write_kimi_decode_shm(*decode_data)
            self.shm.buf[self._SHM_FLAG_OFFSET] = 3
            self._signal_workers()
        return self.run_kimi_decode_fast_async(decode_data)

    @torch.inference_mode()
    def _run_kimi_linear_batch(self, seqs, is_prefill: bool,
                               prefill_chunk_lens=None):
        if not seqs:
            return None

        reset_context()

        device = torch.device(f"cuda:{self.rank}")
        sm_ = self.mamba_state_manager

        if is_prefill and prefill_chunk_lens is None:
            prefill_chunk_lens = [
                len(seq.token_ids) - seq.num_computed_tokens for seq in seqs
            ]

        for i, seq in enumerate(seqs):
            if seq.state_slot is None:
                raise RuntimeError("Kimi-Linear sequence has no allocated state slot")
            total_after = (
                seq.num_computed_tokens + prefill_chunk_lens[i] if is_prefill
                else seq.num_computed_tokens + 1
            )
            sm_.ensure_blocks_for(seq, total_after)

        ids: list[int] = []
        positions: list[int] = []
        cu: list[int] = [0]
        slot_mapping: list[int] = []
        state_indices: list[int] = []
        seq_lens: list[int] = []
        has_initial_state: list[bool] = []
        block_tables_rows: list[list[int]] = []
        max_query_len = 0
        max_seq_len = 0
        max_blocks = 0
        block_size = sm_.block_size

        for i, seq in enumerate(seqs):
            if is_prefill:
                # Chunked prefill: resume where the previous chunk stopped and
                # let the KDA conv/recurrent state continue via
                # ``has_initial_state`` (the same contract vLLM's GDN/KDA
                # metadata uses).
                start_pos = seq.num_computed_tokens
                tokens = list(
                    seq.token_ids[start_pos:start_pos + prefill_chunk_lens[i]],
                )
                init_state = start_pos > 0
            else:
                tokens = [seq.last_token]
                start_pos = seq.num_computed_tokens
                init_state = True

            n_tok = len(tokens)
            ids.extend(tokens)
            positions.extend(range(start_pos, start_pos + n_tok))
            cu.append(cu[-1] + n_tok)
            state_indices.append(seq.state_slot)
            seq_total = start_pos + n_tok
            seq_lens.append(seq_total)
            has_initial_state.append(init_state)
            max_query_len = max(max_query_len, n_tok)
            max_seq_len = max(max_seq_len, seq_total)

            for t in range(n_tok):
                global_pos = start_pos + t
                block_idx = global_pos // block_size
                if block_idx < len(seq.block_table):
                    block_id = seq.block_table[block_idx]
                    slot_mapping.append(
                        block_id * block_size + (global_pos % block_size),
                    )
                else:
                    slot_mapping.append(-1)
            block_tables_rows.append(list(seq.block_table))
            max_blocks = max(max_blocks, len(seq.block_table))

        n_tokens = len(ids)
        batch_size = len(seqs)

        ids_t = torch.tensor(ids, dtype=torch.int64, pin_memory=True).to(
            device, non_blocking=True,
        )
        pos_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(
            device, non_blocking=True,
        )
        cu_cpu = torch.tensor(cu, dtype=torch.int64, pin_memory=True)
        cu_gpu = cu_cpu.to(device, non_blocking=True)
        state_idx_t = torch.tensor(
            state_indices, dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
        seq_lens_t = torch.tensor(
            seq_lens, dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
        has_init_t = torch.tensor(
            has_initial_state, dtype=torch.bool, pin_memory=True,
        ).to(device, non_blocking=True)
        slot_t = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
        slot_t_mla = torch.tensor(
            slot_mapping, dtype=torch.int64, pin_memory=True,
        ).to(device, non_blocking=True)

        if max_blocks == 0:
            block_tables_t = torch.zeros(
                (batch_size, 1), dtype=torch.int32, device=device,
            )
        else:
            bt_cols = self._pad_page_table_cols(max_blocks)
            bt = np.full((batch_size, bt_cols), -1, dtype=np.int32)
            for i, row in enumerate(block_tables_rows):
                if row:
                    bt[i, :len(row)] = row
            block_tables_t = torch.from_numpy(bt).pin_memory().to(
                device, non_blocking=True,
            )

        logit_idx_cpu = (cu_cpu[1:] - 1).to(torch.int64)
        logit_idx_t = logit_idx_cpu.to(device, non_blocking=True)

        if is_prefill:
            num_decodes = 0
            num_decode_tokens = 0
            num_prefills = batch_size
            num_prefill_tokens = n_tokens
        else:
            num_decodes = batch_size
            num_decode_tokens = n_tokens
            num_prefills = 0
            num_prefill_tokens = 0

        md = KimiLinearMetadata(
            num_actual_tokens=n_tokens,
            query_start_loc=cu_gpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens_t,
            max_seq_len=max_seq_len,
            state_indices=state_idx_t,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            has_initial_state=has_init_t,
            slot_mapping=slot_t,
            block_tables=block_tables_t,
        )
        if num_prefills > 0:
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(md.non_spec_query_start_loc)
            )
            md.nums_dict = nums_dict
            md.batch_ptr = batch_ptr
            md.token_chunk_offset_ptr = token_chunk_offset_ptr

        # Kimi's full-attention layers are MLA: a prefill chunk that resumes
        # mid-prompt attends to its own new tokens with the dense causal
        # kernel and to the already-cached prefix through the gather+merge
        # context path (vLLM's MLACommonImpl._compute_prefill_context). Without
        # this metadata the chunk would silently drop the prefix.
        chunked_context = None
        if is_prefill and any(seq.num_computed_tokens > 0 for seq in seqs):
            chunked_context = self._build_chunked_context(
                seqs, prefill_chunk_lens, block_tables_t, device,
            )

        set_context(
            is_prefill,
            cu_seqlens_q=cu_gpu.to(torch.int32),
            cu_seqlens_k=cu_gpu.to(torch.int32),
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            slot_mapping=slot_t_mla,
            context_lens=seq_lens_t,
            block_tables=block_tables_t,
            max_context_len=max_seq_len,
            chunked_context=chunked_context,
        )
        ctx = get_context()
        ctx.kda_state = sm_
        ctx.kda_metadata = md
        try:
            hidden_states = self.model(
                ids_t, positions=pos_t, state_manager=sm_,
            )
        finally:
            reset_context()

        last_hidden = hidden_states.index_select(0, logit_idx_t)
        logits = self.model.compute_logits(last_hidden)

        for i, seq in enumerate(seqs):
            if is_prefill:
                seq.num_computed_tokens += prefill_chunk_lens[i]
            else:
                seq.num_computed_tokens += 1
        return logits

    @torch.inference_mode()
    def run_qwen3_next_mixed(self, prefill_seqs, decode_seqs,
                             prefill_chunk_lens=None):
        """Run Qwen3-Next with flat GDN state and paged MHA KV cache.

        If a batch contains any prefill sequence, all sequences in the step
        use the GDN chunk path. This mirrors vLLM's GDN metadata behavior for
        mixed batches and avoids a separate decode-first token layout.

        ``prefill_chunk_lens`` gives the number of prompt tokens to run for
        each entry of ``prefill_seqs`` this step (chunked prefill, matching
        vLLM's scheduler).  ``None`` means "the whole remainder", i.e. the
        single-shot behaviour.
        """
        seqs = list(prefill_seqs) + list(decode_seqs)
        if not seqs:
            return None

        if prefill_chunk_lens is None:
            prefill_chunk_lens = [
                len(seq.token_ids) - seq.num_computed_tokens
                for seq in prefill_seqs
            ]
        chunk_by_seq = {
            id(seq): chunk
            for seq, chunk in zip(prefill_seqs, prefill_chunk_lens)
        }

        reset_context()
        device = torch.device(f"cuda:{self.rank}")
        sm_ = self.mamba_state_manager
        block_size = sm_.block_size
        use_prefill_path = bool(prefill_seqs)

        cu: list[int] = [0]
        state_indices: list[int] = []
        seq_lens: list[int] = []
        has_initial_state: list[bool] = []
        block_tables_rows: list[list[int]] = []
        consumed: list[tuple["Sequence", int]] = []
        # Per-sequence numpy pieces, concatenated once at the end. A prefill
        # chunk is up to max_num_batched_tokens long, so building ids /
        # positions / slot_mapping with a Python loop per token costs several ms
        # of interpreter time per step on a path that is otherwise GPU-bound.
        ids_parts: list[np.ndarray] = []
        pos_parts: list[np.ndarray] = []
        slot_parts: list[np.ndarray] = []
        max_query_len = 0
        max_seq_len = 0
        max_blocks = 0

        for seq in seqs:
            if seq.state_slot is None:
                raise RuntimeError("Qwen3-Next sequence has no allocated state slot")
            chunk = chunk_by_seq.get(id(seq))
            if chunk is not None:
                start_pos = seq.num_computed_tokens
                n_tok = min(chunk, len(seq.token_ids) - start_pos)
                if n_tok <= 0:
                    continue
                seq_total = start_pos + n_tok
                token_ids = np.asarray(
                    seq.token_ids[start_pos:start_pos + n_tok], dtype=np.int64,
                )
            else:
                start_pos = len(seq) - 1
                n_tok = 1
                seq_total = len(seq)
                token_ids = np.asarray([seq.last_token], dtype=np.int64)

            sm_.ensure_blocks_for(seq, seq_total)

            consumed.append((seq, n_tok))
            ids_parts.append(token_ids)
            pos_parts.append(
                np.arange(start_pos, start_pos + n_tok, dtype=np.int64),
            )
            cu.append(cu[-1] + n_tok)
            state_indices.append(seq.state_slot)
            seq_lens.append(seq_total)
            has_initial_state.append(start_pos > 0)
            max_query_len = max(max_query_len, n_tok)
            max_seq_len = max(max_seq_len, seq_total)

            # slot = block_table[pos // block_size] * block_size + pos % block_size,
            # with -1 for positions past the allocated blocks (the kernels treat
            # that as "do not write").
            row = np.asarray(seq.block_table, dtype=np.int64)
            gpos = np.arange(start_pos, start_pos + n_tok, dtype=np.int64)
            blk = gpos // block_size
            in_range = blk < row.shape[0]
            slots = np.full(n_tok, -1, dtype=np.int64)
            if in_range.any():
                sel = blk[in_range]
                slots[in_range] = (
                    row[sel] * block_size + (gpos[in_range] - sel * block_size)
                )
            slot_parts.append(slots)
            block_tables_rows.append(seq.block_table)
            max_blocks = max(max_blocks, len(seq.block_table))

        if not ids_parts:
            return None

        ids_np = (
            ids_parts[0] if len(ids_parts) == 1 else np.concatenate(ids_parts)
        )
        pos_np = (
            pos_parts[0] if len(pos_parts) == 1 else np.concatenate(pos_parts)
        )
        slot_np = (
            slot_parts[0] if len(slot_parts) == 1 else np.concatenate(slot_parts)
        )

        n_tokens = ids_np.shape[0]
        batch_size = len(state_indices)

        ids_t = torch.from_numpy(ids_np).pin_memory().to(
            device, non_blocking=True,
        )
        pos_t = torch.from_numpy(pos_np).pin_memory().to(
            device, non_blocking=True,
        )
        cu_cpu = torch.tensor(cu, dtype=torch.int64, pin_memory=True)
        cu_gpu = cu_cpu.to(device, non_blocking=True)
        # One int32 copy for the whole step: the GDN layers, the conv kernel and
        # the attention context all want int32 cu_seqlens, and converting per
        # call was 3 launches per step plus 2 per GDN layer.
        cu_int32 = cu_gpu.to(torch.int32)
        state_idx_t = torch.tensor(
            state_indices, dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
        seq_lens_t = torch.tensor(
            seq_lens, dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
        has_init_t = torch.tensor(
            has_initial_state, dtype=torch.bool, pin_memory=True,
        ).to(device, non_blocking=True)
        slot_t_mha = torch.from_numpy(slot_np).pin_memory().to(
            device, non_blocking=True,
        )
        slot_t = slot_t_mha.to(torch.int32)

        if max_blocks == 0:
            block_tables_t = torch.zeros(
                (batch_size, 1), dtype=torch.int32, device=device,
            )
        else:
            bt = np.full((batch_size, max_blocks), -1, dtype=np.int32)
            for i, row in enumerate(block_tables_rows):
                if row:
                    bt[i, :len(row)] = row
            block_tables_t = torch.from_numpy(bt).pin_memory().to(
                device, non_blocking=True,
            )

        logit_idx_t = (cu_gpu[1:] - 1).to(torch.int64)
        if use_prefill_path:
            num_prefills = batch_size
            num_prefill_tokens = n_tokens
            num_decodes = 0
            num_decode_tokens = 0
        else:
            num_prefills = 0
            num_prefill_tokens = 0
            num_decodes = batch_size
            num_decode_tokens = n_tokens

        md = KimiLinearMetadata(
            num_actual_tokens=n_tokens,
            query_start_loc=cu_gpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens_t,
            max_seq_len=max_seq_len,
            state_indices=state_idx_t,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            has_initial_state=has_init_t,
            slot_mapping=slot_t,
            block_tables=block_tables_t,
            # Precomputed once per step: the GDN layers would otherwise convert
            # ``query_start_loc`` to int32 and ``state_indices`` to int64 on
            # every one of the 36 linear-attention layers.
            query_start_loc_int32=cu_int32,
            state_indices_long=(
                state_idx_t.long() if use_prefill_path else None
            ),
            all_have_initial_state=all(has_initial_state),
            any_have_initial_state=any(has_initial_state),
        )
        if num_prefills > 0:
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    md.non_spec_query_start_loc,
                    seqlens_cpu=[cu[i + 1] - cu[i] for i in range(batch_size)],
                )
            )
            md.nums_dict = nums_dict
            md.batch_ptr = batch_ptr
            md.token_chunk_offset_ptr = token_chunk_offset_ptr

        set_context(
            use_prefill_path,
            cu_seqlens_q=cu_int32,
            cu_seqlens_k=cu_int32,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            slot_mapping=slot_t_mha,
            context_lens=seq_lens_t,
            block_tables=block_tables_t,
            max_context_len=max_seq_len,
        )
        ctx = get_context()
        ctx.kda_state = sm_
        ctx.kda_metadata = md
        try:
            hidden_states = self.model(
                ids_t, positions=pos_t, state_manager=sm_,
            )
        finally:
            reset_context()

        last_hidden = hidden_states.index_select(0, logit_idx_t)
        logits = self.model.compute_logits(last_hidden)

        for seq, n_tok in consumed:
            seq.num_computed_tokens += n_tok
        return logits

    def _run_qwen3_next_batch(self, seqs, is_prefill: bool,
                              prefill_chunk_lens=None):
        if is_prefill:
            return self.run_qwen3_next_mixed(seqs, [], prefill_chunk_lens)
        return self.run_qwen3_next_mixed([], seqs)

    @torch.inference_mode()
    def _warmup_qwen3_next_prefill(self):
        """Warm Qwen3-Next prefill kernels at benchmark-relevant chunk sizes."""
        if not self.is_qwen3_next or self.mamba_state_manager is None:
            return

        warmup_lens = [512, 1024]
        warmup_lens = sorted({
            max(1, min(int(length), int(self.max_model_len)))
            for length in warmup_lens
            if int(length) > 0
        })

        total_tokens = int(self.max_num_batched_tokens)
        warmed: list[tuple[int, int]] = []
        for seq_len in warmup_lens:
            n_seqs = max(1, min(self.max_num_seqs, total_tokens // seq_len))
            if n_seqs <= 0:
                continue
            seqs = [
                Sequence([0] * seq_len, max_tokens=1, ignore_eos=True)
                for _ in range(n_seqs)
            ]
            slots = self.allocate_mamba_state_batch(n_seqs)
            for seq, slot in zip(seqs, slots):
                seq.state_slot = slot
            try:
                _ = self.run_qwen3_next_mixed(seqs, [])
                torch.cuda.synchronize()
                warmed.append((n_seqs, seq_len))
            finally:
                payloads = [
                    (seq.state_slot, list(seq.block_table))
                    for seq in seqs
                    if seq.state_slot is not None
                ]
                self.deallocate_mamba_state_batch(payloads)
                for seq in seqs:
                    seq.block_table = []
                    seq.state_slot = None

        if self.rank == 0 and warmed:
            desc = ", ".join(f"{n}x{l}" for n, l in warmed)
            print(f"  Qwen3-Next prefill warmup: {desc}", flush=True)

    def _mamba_prepare_tensors(self, prefill_seqs, decode_seqs, chunk_size,
                               prefill_chunk_lens=None):
        """Build flat input_ids / positions and the Mamba(2)Metadata for a
        mixed batch of prefill + decode sequences.

        ``prefill_chunk_lens`` (optional, one entry per prefill seq) caps how
        many tokens of each prefill sequence are processed this step, enabling
        chunked prefill: a long prompt is split across steps and the SSM/conv
        state is carried via ``has_initial_states_p`` + ``num_computed_tokens_p``
        (matching vLLM). When ``None`` each prefill seq processes all remaining
        prompt tokens in one step (the original single-shot behavior).

        Mixed layout is model-specific:
        - Mamba v1 keeps decode tokens first, then prefill tokens.
        - Mamba2 keeps the original prefill-first, decode-last order.
        Homogeneous prefill/decode batches keep their natural order.
        """
        device = torch.device(f"cuda:{self.rank}")
        input_ids: list[int] = []
        positions: list[int] = []
        prefill_state_indices: list[int] = []
        prefill_has_initial: list[bool] = []
        query_start_loc: list[int] = [0]
        decode_state_indices: list[int] = []
        has_mixed = bool(prefill_seqs) and bool(decode_seqs)
        mixed_decode_first = has_mixed and not self.is_mamba2

        def _chunk_len(i, seq):
            start = seq.num_computed_tokens
            if prefill_chunk_lens is not None:
                return min(prefill_chunk_lens[i], len(seq.token_ids) - start)
            return len(seq.token_ids) - start

        if mixed_decode_first:
            for seq in decode_seqs:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                decode_state_indices.append(seq.state_slot)
            # Round the decode row count up to the mixer's GEMM alignment.  The
            # chunk planner keeps the *prefill* token count aligned, but the
            # mixer also cares about the step *total* (prefill + decode), which
            # is the contiguous axis of the gate and out_proj operands -- and the
            # number of running sequences is arbitrary.  Padded rows carry the
            # null state index, which ``causal_conv1d_update`` and
            # ``selective_state_update`` skip: the same trick
            # ``capture_mamba_cudagraph`` uses to fill a bucket.  Unlike padding
            # the prefill side, this costs no extra scan chunk (decode does not
            # go through the varlen scan), just a handful of skipped rows.
            for _ in range(-len(decode_seqs) % _MAMBA_GEMM_ALIGN):
                input_ids.append(0)
                positions.append(0)
                decode_state_indices.append(self._MAMBA_PAD_SLOT_ID)
            for i, seq in enumerate(prefill_seqs):
                start = seq.num_computed_tokens
                chunk = _chunk_len(i, seq)
                input_ids.extend(seq.token_ids[start:start + chunk])
                positions.extend(range(start, start + chunk))
                query_start_loc.append(query_start_loc[-1] + chunk)
                prefill_state_indices.append(seq.state_slot)
                prefill_has_initial.append(start > 0)
        else:
            for i, seq in enumerate(prefill_seqs):
                start = seq.num_computed_tokens
                chunk = _chunk_len(i, seq)
                input_ids.extend(seq.token_ids[start:start + chunk])
                positions.extend(range(start, start + chunk))
                query_start_loc.append(query_start_loc[-1] + chunk)
                prefill_state_indices.append(seq.state_slot)
                prefill_has_initial.append(start > 0)
            for seq in decode_seqs:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                decode_state_indices.append(seq.state_slot)

        num_prefill_tokens = query_start_loc[-1]
        # Padded: the token count the kernels and the mixer see.
        num_decode_tokens = len(decode_state_indices)
        num_actual = num_prefill_tokens + num_decode_tokens

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64,
                                   pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64,
                                   pin_memory=True).cuda(non_blocking=True)

        # Build per-batch Mamba metadata
        if self.is_mamba2:
            meta = Mamba2Metadata(chunk_size=chunk_size)
        else:
            meta = MambaMetadata()
        meta.num_prefill_tokens = num_prefill_tokens
        meta.num_decode_tokens = num_decode_tokens
        meta.num_prefills = len(prefill_seqs)
        meta.num_decodes = len(decode_seqs)

        if prefill_seqs:
            meta.query_start_loc_p = torch.tensor(
                query_start_loc, dtype=torch.int32, device=device,
            )
            meta.state_indices_p = torch.tensor(
                prefill_state_indices, dtype=torch.int32, device=device,
            )
            meta.has_initial_states_p = torch.tensor(
                prefill_has_initial, dtype=torch.bool, device=device,
            )
            meta.prep_initial_states = any(prefill_has_initial)
            if not self.is_mamba2:
                nums_dict, batch_ptr, token_chunk_offset_ptr = (
                    compute_causal_conv1d_metadata(meta.query_start_loc_p)
                )
                meta.nums_dict = nums_dict
                meta.batch_ptr = batch_ptr
                meta.token_chunk_offset_ptr = token_chunk_offset_ptr

            if self.is_mamba2:
                num_computed = torch.tensor(
                    [s.num_computed_tokens for s in prefill_seqs],
                    dtype=torch.int32, device=device,
                )
                cu_chunk, seq_idx, last_idx = build_chunk_metadata(
                    meta.query_start_loc_p,
                    chunk_size=chunk_size,
                    num_computed_tokens_p=num_computed,
                )
                meta.cu_chunk_seqlen_p = cu_chunk
                meta.seq_idx_p = seq_idx
                meta.last_chunk_indices_p = last_idx

        if decode_seqs:
            meta.state_indices_d = torch.tensor(
                decode_state_indices, dtype=torch.int32, device=device,
            )

        return input_ids_t, positions_t, meta, num_actual

    def prepare_mamba_prefill(self, seqs):
        chunk_size = getattr(self.config, "chunk_size", 256)
        return self._mamba_prepare_tensors(seqs, [], chunk_size)

    def prepare_mamba_decode(self, seqs):
        chunk_size = getattr(self.config, "chunk_size", 256)
        return self._mamba_prepare_tensors([], seqs, chunk_size)

    def prepare_mamba_mixed(self, prefill_seqs, decode_seqs, prefill_chunk_lens=None):
        chunk_size = getattr(self.config, "chunk_size", 256)
        return self._mamba_prepare_tensors(
            prefill_seqs, decode_seqs, chunk_size,
            prefill_chunk_lens=prefill_chunk_lens,
        )

    @torch.inference_mode()
    def run_mamba(self, seqs, is_prefill: bool):
        """Run a Mamba/Mamba2 forward pass for a homogeneous batch.

        Mirrors the attention-side ``run`` entry point: builds metadata,
        installs it on the global Context, runs the model + LM head,
        returns logits (one per seq) on rank 0 or ``None`` on workers.
        """
        if is_prefill:
            input_ids, positions, meta, num_actual = self.prepare_mamba_prefill(seqs)
        else:
            input_ids, positions, meta, num_actual = self.prepare_mamba_decode(seqs)

        set_mamba_context(
            is_prefill=is_prefill,
            mamba_state=self.mamba_state_manager,
            mamba_metadata=meta,
        )
        try:
            hidden = self.model(input_ids, positions)
            # Pick last-token hidden per seq for logits
            if is_prefill:
                last_idx = (meta.query_start_loc_p[1:] - 1).to(torch.int64)
                hidden_last = hidden.index_select(0, last_idx)
            else:
                hidden_last = hidden  # one token per decode seq
            logits = self.model.compute_logits(hidden_last)
        finally:
            reset_context()
        return logits

    @torch.inference_mode()
    def run_mamba_mixed(self, prefill_seqs, decode_seqs, prefill_chunk_lens=None):
        """Run a mixed prefill+decode batch through the Mamba model.

        ``prefill_chunk_lens`` (optional) caps each prefill seq's tokens this
        step for chunked prefill; the returned logits are still one row per
        prefill seq (its chunk's last token) followed by one per decode seq.
        """
        input_ids, positions, meta, num_actual = self.prepare_mamba_mixed(
            prefill_seqs, decode_seqs, prefill_chunk_lens=prefill_chunk_lens,
        )
        set_mamba_context(
            is_prefill=True,
            mamba_state=self.mamba_state_manager,
            mamba_metadata=meta,
        )
        try:
            hidden = self.model(input_ids, positions)
            # Logits at end of each prefill seq + all decode tokens.
            indices: list[int] = []
            if prefill_seqs:
                qsl = meta.query_start_loc_p.to("cpu").tolist()
                if self.is_mamba2:
                    indices.extend(qsl[i + 1] - 1 for i in range(len(prefill_seqs)))
                else:
                    indices.extend(
                        meta.num_decode_tokens + qsl[i + 1] - 1
                        for i in range(len(prefill_seqs))
                    )
            if self.is_mamba2:
                indices.extend(
                    range(
                        meta.num_prefill_tokens,
                        meta.num_prefill_tokens + len(decode_seqs),
                    )
                )
            else:
                # ``num_decode_tokens`` may include alignment-pad rows; only the
                # real decode rows produce a token.
                indices.extend(range(len(decode_seqs)))
            idx_t = torch.tensor(indices, dtype=torch.int64,
                                 device=hidden.device)
            hidden_last = hidden.index_select(0, idx_t)
            logits = self.model.compute_logits(hidden_last)
        finally:
            reset_context()
        return logits

    # ------------------------------------------------------------------
    # Mamba decode fast path: pre-allocated buffers, GPU greedy argmax,
    # async D2H copy, and CUDA graph capture for decode-only steps.
    # Mirrors the attention engine's _init_greedy_buffers / capture_cudagraph
    # pattern (see ``vllm/v1/worker/gpu_model_runner.py`` and
    # ``vllm/v1/attention/backends/mamba_attn.py`` for the upstream
    # equivalent: vLLM only captures decode-only Mamba graphs and pads
    # state_indices_d with PAD_SLOT_ID = -1; the kernels skip those rows).
    # ------------------------------------------------------------------
    _MAMBA_PAD_SLOT_ID = -1

    def _init_mamba_decode_buffers(self):
        """Pre-allocate persistent buffers for the Mamba decode fast path.

        Creates GPU input buffers (input_ids, positions, state_indices_d),
        numpy staging buffers, greedy local-argmax outputs, async D2H
        plumbing, and TP cross-rank gather buffers.  These buffers are
        reused across all Mamba decode steps and are also the buffers
        that ``capture_mamba_cudagraph`` records.
        """
        max_bs = self.max_num_seqs
        dev = f"cuda:{self.rank}"

        self._md_input_ids = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._md_positions = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._md_state_indices = torch.full(
            (max_bs,), self._MAMBA_PAD_SLOT_ID,
            dtype=torch.int32, device=dev,
        )
        if hasattr(torch, "_dynamo") and hasattr(
            torch._dynamo, "mark_static_address"
        ):
            torch._dynamo.mark_static_address(self._md_input_ids)
            torch._dynamo.mark_static_address(self._md_positions)
            torch._dynamo.mark_static_address(self._md_state_indices)

        # Numpy views over pinned-CPU torch tensors so ``copy_(...,
        # non_blocking=True)`` into the GPU buffers is truly async.
        self._md_input_ids_cpu = torch.empty(
            max_bs, dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._md_positions_cpu = torch.empty(
            max_bs, dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._md_state_indices_cpu = torch.full(
            (max_bs,), self._MAMBA_PAD_SLOT_ID,
            dtype=torch.int32, device="cpu", pin_memory=True,
        )
        self._md_input_ids_np = self._md_input_ids_cpu.numpy()
        self._md_positions_np = self._md_positions_cpu.numpy()
        self._md_state_indices_np = self._md_state_indices_cpu.numpy()

        # Outputs of local greedy argmax (set every step / replay).
        self._md_lm_max_vals = torch.zeros(
            max_bs, dtype=torch.float32, device=dev,
        )
        self._md_lm_max_idxs = torch.zeros(
            max_bs, dtype=torch.int64, device=dev,
        )

        # Async D2H staging.
        self._md_pinned_token_ids = torch.empty(
            max_bs, dtype=torch.int64, device="cpu", pin_memory=True,
        )
        self._md_copy_stream = torch.cuda.Stream(device=dev)
        self._md_copy_event = torch.cuda.Event()

        # TP cross-rank greedy gather buffers (mirror _init_greedy_buffers).
        self._md_greedy_info = torch.zeros(
            max_bs, 2, dtype=torch.float32, device=dev,
        )
        self._md_greedy_gathered = [
            torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
            for _ in range(self.world_size)
        ]
        self._md_greedy_all_info = torch.zeros(
            self.world_size, max_bs, 2, dtype=torch.float32, device=dev,
        )
        self._md_greedy_arange = torch.arange(max_bs, device=dev)

        # Filled in by capture_mamba_cudagraph (None -> eager only).
        self._mamba_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._mamba_graph_metas: dict[int, object] = {}
        self._mamba_graph_bs_list: list[int] = []
        self._mamba_graph_bs_for_n: list[int] | None = None
        self._mamba_graph_pool = None

    def _prepare_mamba_decode_arrays(self, seqs):
        """Fill numpy staging buffers for a Mamba decode batch.

        Returns ``(n, ids_np, pos_np, si_np)`` where the arrays are
        sliced views of the persistent staging buffers.  Avoids the
        per-step ``torch.tensor(...)`` allocation that the slow
        ``_mamba_prepare_tensors`` path incurs.
        """
        n = len(seqs)
        ids = self._md_input_ids_np
        pos = self._md_positions_np
        si = self._md_state_indices_np
        for i, s in enumerate(seqs):
            tids = s.token_ids
            if tids is not None:
                ids[i] = tids[-1]
                pos[i] = len(tids) - 1
            else:
                ids[i] = s._last_token
                pos[i] = s._num_tokens - 1
            si[i] = s.state_slot
        return n, ids[:n], pos[:n], si[:n]

    def _mamba_make_decode_meta(self, n: int, state_indices: torch.Tensor):
        """Construct a per-step ``(Mamba|Mamba2)Metadata`` for decode.

        The metadata's tensors are *views into our persistent buffers*
        so a captured CUDA graph reads the same memory at replay time.
        """
        if self.is_mamba2:
            meta = Mamba2Metadata(
                chunk_size=getattr(self.config, "chunk_size", 256),
            )
        else:
            meta = MambaMetadata()
        meta.num_prefill_tokens = 0
        meta.num_decode_tokens = n
        meta.num_prefills = 0
        meta.num_decodes = n
        meta.state_indices_d = state_indices
        return meta

    @torch.inference_mode()
    def _mamba_graph_capacity(self) -> int:
        """Largest decode batch whose CUDA graph fits the reserved pool.

        A captured decode graph pins the working set of one decode forward, so
        the private pool is roughly that forward's activation peak.  Measure the
        peak eagerly at a probe batch and extrapolate linearly (the fixed part
        of the peak makes this an over-estimate, i.e. conservative) against the
        headroom ``allocate_mamba_state_cache`` already set aside.
        """
        max_bs = self.max_num_seqs
        if max_bs <= 1:
            return max(1, max_bs)

        probe = min(max_bs, 256)
        probe_ids = self._md_input_ids[:probe]
        probe_positions = self._md_positions[:probe]
        meta = self._mamba_make_decode_meta(
            probe, self._md_state_indices[:probe],
        )
        set_mamba_context(
            is_prefill=False,
            mamba_state=self.mamba_state_manager,
            mamba_metadata=meta,
        )
        # This probe -- not the capture loop below -- is the *first* forward
        # through the compiled Mamba2 model, so the batch dim has to be marked
        # dynamic here.  ``compile_model`` traces with ``dynamic=False`` and
        # drops every Dynamo guard, so whatever shape the first forward sees is
        # baked into the graph's buffers for good: a probe traced at 256 tokens
        # left every later batch size (including the first capture bucket at
        # ``max_num_seqs``) running against 256-row activations.
        if self._compiled and not self._mark_dynamic_done:
            torch._dynamo.mark_dynamic(probe_ids, 0)
            torch._dynamo.mark_dynamic(probe_positions, 0)
            self._mark_dynamic_done = True

        def _probe_forward():
            hidden = self.model(probe_ids, probe_positions)
            partial = self.model.lm_head.linear_op(
                hidden, self.model.lm_head.embedding_op.emb.weight,
            ).float()
            partial.max(dim=-1)

        try:
            # Warmup pass first: the compile this triggers (Inductor autotuning
            # allocates real buffers) is not part of a steady-state decode
            # forward and would inflate the per-seq estimate.
            _probe_forward()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            _probe_forward()
            torch.cuda.synchronize()
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        finally:
            reset_context()
        per_seq = max(1, (peak - baseline)) / probe
        affordable = int(self._MAMBA_GRAPH_RESERVE_BYTES / per_seq)
        return max(1, min(max_bs, affordable))

    @torch.inference_mode()
    def capture_mamba_cudagraph(self):
        """Capture decode-only CUDA graphs for Mamba/Mamba2 at bucket sizes.

        Mirrors vLLM's approach (see
        ``vllm/v1/worker/gpu_model_runner.py`` and
        ``BaseMambaAttentionMetadataBuilder.build_for_cudagraph_capture``):
        only decode-only steps are captured, and unused rows in the
        padded batch get ``state_indices_d = PAD_SLOT_ID (-1)`` so the
        ``causal_conv1d_update`` / ``selective_state_update`` /
        ``selective_scan_fn`` Triton kernels skip those slots.

        Each bucket records the model forward + the LM head's local
        ``linear_op`` + ``max(dim=-1)`` (just like the attention path).
        For TP > 1, the cross-rank ``(max_val, max_idx)`` gather happens
        outside the graph in ``_run_mamba_decode_graph``.
        """
        from contextlib import nullcontext

        # Largest batch worth capturing.  Capturing huge buckets eats CUDA-graph
        # private-pool memory, which for a big Mamba2 model (Codestral 7B
        # allocates ~4 GB of conv/ssm activations per layer at bs=1024) can OOM
        # alongside the slot pool -- vLLM caps at ``max_cudagraph_capture_size``
        # (512 by default) for the same reason.  The pool a decode graph needs is
        # about the eager decode forward's activation peak at that batch, so
        # measure it rather than hardcoding one number for every architecture: a
        # blanket cap of 256 sent every larger step down the eager path (~30 ms
        # versus ~7 ms replayed), and a 1000-prompt run spends its whole steady
        # state with 300-500 sequences decoding.
        max_bs = self._mamba_graph_capacity()
        if _is_hopper_gpu():
            # Same capture grid vLLM uses on Hopper
            # (``VllmConfig._set_cudagraph_sizes``): 1, 2, 4, step-8 to 256,
            # step-16 to min(max_num_seqs*2, 512). Leave the Blackwell bucket
            # list (including >512) unchanged.
            max_bs = min(max_bs, _VLLM_DEFAULT_MAX_CUDAGRAPH_CAPTURE_SIZE)
            buckets = _vllm_default_cudagraph_capture_sizes(max_bs)
        else:
            buckets = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
            # Above 256 the decode step is roughly linear in batch, so a coarser
            # stride costs only the padding up to the next bucket (<= 12%).
            buckets += list(range(288, 513, 32)) + list(range(576, 4097, 64))
        self._mamba_graph_bs_list = [b for b in buckets if b <= max_bs]
        if max_bs > 0 and (
            not self._mamba_graph_bs_list
            or self._mamba_graph_bs_list[-1] < max_bs
        ):
            # Always cover max_num_seqs exactly so no step falls back to eager.
            self._mamba_graph_bs_list.append(max_bs)
        if not self._mamba_graph_bs_list:
            self._mamba_graph_bs_for_n = None
            return

        lm_head = self.model.lm_head
        weight = lm_head.embedding_op.emb.weight
        graph_baseline = torch.cuda.memory_stats().get(
            "reserved_bytes.all.current", 0,
        )

        ar_ctx = (
            self.custom_ar.capture()
            if self.custom_ar is not None else nullcontext()
        )
        with ar_ctx:
            for bs in reversed(self._mamba_graph_bs_list):
                input_ids = self._md_input_ids[:bs]
                positions = self._md_positions[:bs]
                state_indices = self._md_state_indices[:bs]
                # Initialise to PAD so warmup is safe and any bucket-only
                # tail at runtime that we forget to fill stays a PAD.
                state_indices.fill_(self._MAMBA_PAD_SLOT_ID)
                input_ids.zero_()
                positions.zero_()

                meta = self._mamba_make_decode_meta(bs, state_indices)
                self._mamba_graph_metas[bs] = meta

                set_mamba_context(
                    is_prefill=False,
                    mamba_state=self.mamba_state_manager,
                    mamba_metadata=meta,
                )

                # Warmup forward (eager) so the CUDA-graph capture region
                # only records steady-state kernels.
                if self._compiled and not self._mark_dynamic_done:
                    torch._dynamo.mark_dynamic(input_ids, 0)
                    torch._dynamo.mark_dynamic(positions, 0)
                    self._mark_dynamic_done = True

                hidden = self.model(input_ids, positions)
                partial = lm_head.linear_op(hidden, weight).float()
                mv, mi = partial.max(dim=-1)
                self._md_lm_max_vals[:bs].copy_(mv)
                self._md_lm_max_idxs[:bs].copy_(mi)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self._mamba_graph_pool):
                    hidden = self.model(input_ids, positions)
                    partial = lm_head.linear_op(hidden, weight).float()
                    mv, mi = partial.max(dim=-1)
                    self._md_lm_max_vals[:bs].copy_(mv)
                    self._md_lm_max_idxs[:bs].copy_(mi)

                if self._mamba_graph_pool is None:
                    self._mamba_graph_pool = graph.pool()
                self._mamba_graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

        # ``self.max_num_seqs`` may exceed the largest captured bucket --
        # those steps fall back to eager.  We still build the lookup over
        # the full range but clamp to the largest bucket above it.
        max_bucket = self._mamba_graph_bs_list[-1]
        self._mamba_graph_bs_for_n = [0] * (self.max_num_seqs + 1)
        for n in range(self.max_num_seqs + 1):
            self._mamba_graph_bs_for_n[n] = next(
                (x for x in self._mamba_graph_bs_list if x >= n),
                max_bucket,
            )
        if self.rank == 0:
            pool_bytes = torch.cuda.memory_stats().get(
                "reserved_bytes.all.current", 0,
            ) - graph_baseline
            print(
                f"  Mamba CUDA graphs: {len(self._mamba_graphs)} buckets "
                f"(min={self._mamba_graph_bs_list[0]}, "
                f"max={self._mamba_graph_bs_list[-1]}, "
                f"pool={max(0, pool_bytes) / (1 << 20):.0f} MiB)"
            )

    @torch.inference_mode()
    def _run_mamba_decode_eager(self, n, ids_np, pos_np, si_np):
        """Eager-mode Mamba decode + greedy local argmax."""
        self._md_input_ids[:n].copy_(
            self._md_input_ids_cpu[:n], non_blocking=True,
        )
        self._md_positions[:n].copy_(
            self._md_positions_cpu[:n], non_blocking=True,
        )
        self._md_state_indices[:n].copy_(
            self._md_state_indices_cpu[:n], non_blocking=True,
        )

        meta = self._mamba_make_decode_meta(n, self._md_state_indices[:n])
        set_mamba_context(
            is_prefill=False,
            mamba_state=self.mamba_state_manager,
            mamba_metadata=meta,
        )
        try:
            hidden = self.model(
                self._md_input_ids[:n], self._md_positions[:n],
            )
            lm_head = self.model.lm_head
            partial = lm_head.linear_op(
                hidden, lm_head.embedding_op.emb.weight,
            ).float()
            max_vals, max_idxs = partial.max(dim=-1)
        finally:
            reset_context()

        if self.world_size == 1:
            return max_idxs

        return self._mamba_greedy_gather(n, max_vals, max_idxs)

    @torch.inference_mode()
    def _run_mamba_decode_graph(self, n, ids_np, pos_np, si_np):
        """Run a captured CUDA graph for Mamba decode at bucket >= n."""
        bucket = self._mamba_graph_bs_for_n[n]
        # Stage inputs into the persistent buffers the graph captured.
        self._md_input_ids[:n].copy_(
            self._md_input_ids_cpu[:n], non_blocking=True,
        )
        self._md_positions[:n].copy_(
            self._md_positions_cpu[:n], non_blocking=True,
        )
        self._md_state_indices[:n].copy_(
            self._md_state_indices_cpu[:n], non_blocking=True,
        )
        if bucket > n:
            # Pad tail rows so kernels skip them (PAD_SLOT_ID = -1).
            self._md_input_ids[n:bucket].zero_()
            self._md_positions[n:bucket].zero_()
            self._md_state_indices[n:bucket].fill_(self._MAMBA_PAD_SLOT_ID)

        self._mamba_graphs[bucket].replay()

        if self.world_size == 1:
            return self._md_lm_max_idxs[:n]

        return self._mamba_greedy_gather(
            n, self._md_lm_max_vals[:n], self._md_lm_max_idxs[:n],
        )

    def _mamba_greedy_gather(
        self, n: int, max_vals: torch.Tensor, max_idxs: torch.Tensor,
    ):
        """TP cross-rank greedy gather (mirrors attention's _greedy_from_hidden).

        Every rank participates in the all-gather (it's a collective),
        but only rank 0 returns the resulting token-id tensor; workers
        return ``None`` so ``run_mamba_decode_fast_async`` skips the
        async D2H copy on those ranks.
        """
        lm_head = self.model.lm_head
        info = self._md_greedy_info[:n]
        info[:, 0] = max_vals
        info[:, 1] = max_idxs.float()
        info[:, 1] += lm_head.per_partition * self.rank

        gathered = [g[:n] for g in self._md_greedy_gathered]
        dist.all_gather(gathered, info)
        if self.rank != 0:
            return None
        all_info = self._md_greedy_all_info[:, :n]
        torch.stack(gathered, out=all_info)
        best_rank = all_info[:, :n, 0].argmax(dim=0)
        return all_info[best_rank, self._md_greedy_arange[:n], 1].long()

    @torch.inference_mode()
    def run_mamba_decode_fast_async(self, decode_data):
        """Greedy Mamba decode step + async D2H copy of token IDs.

        Returns ``(has_result, n)`` -- caller must call
        ``_wait_async_mamba_tokens(n)`` later to get the token ID list.
        """
        n, ids_np, pos_np, si_np = decode_data
        use_graph = (
            not self.enforce_eager
            and self._mamba_graph_bs_for_n is not None
            and n <= self._mamba_graph_bs_list[-1]
        )
        if use_graph:
            token_ids = self._run_mamba_decode_graph(n, ids_np, pos_np, si_np)
        else:
            token_ids = self._run_mamba_decode_eager(n, ids_np, pos_np, si_np)

        if token_ids is None:
            # Non-rank-0 worker (TP > 1).
            return False, n
        main_stream = torch.cuda.current_stream()
        cs = self._md_copy_stream
        with torch.cuda.stream(cs):
            cs.wait_stream(main_stream)
            self._md_pinned_token_ids[:n].copy_(token_ids, non_blocking=True)
            self._md_copy_event.record(cs)
        return True, n

    def _wait_async_mamba_tokens(self, n: int) -> list[int]:
        """Wait for the async D2H copy and return the Python token list."""
        self._md_copy_event.synchronize()
        return self._md_pinned_token_ids[:n].tolist()

    def _write_mamba_decode_shm(self, n, ids_np, pos_np, si_np):
        """Pack a Mamba decode batch into SHM (TP > 1 dispatch).

        Layout: ``[n(2)][_(2)][ids(n*8)][pos(n*8)][si(n*4)]``.  The
        2-byte ``_`` slot mirrors the attention path's ``max_bt`` field.
        """
        buf = self.shm.buf
        buf[0:2] = n.to_bytes(2, "little")
        buf[2:4] = (0).to_bytes(2, "little")
        off = 4
        for arr in (ids_np, pos_np, si_np):
            nb = arr.nbytes
            buf[off:off + nb] = arr.tobytes()
            off += nb

    @torch.inference_mode()
    def _loop_mamba_decode_greedy(self, cur_seq):
        """Worker fast path for Mamba: read decode arrays from SHM into
        the pinned-CPU staging buffers, then dispatch the same fast path
        as rank 0 (the kernels read state_indices_d which we just wrote)."""
        buf = self.shm.buf
        n = int.from_bytes(buf[0:2], "little")
        off = 4
        ids = np.frombuffer(buf, dtype=np.int64, count=n, offset=off)
        off += n * 8
        pos = np.frombuffer(buf, dtype=np.int64, count=n, offset=off)
        off += n * 8
        si = np.frombuffer(buf, dtype=np.int32, count=n, offset=off)
        # Land into the persistent pinned buffers (numpy views).
        self._md_input_ids_np[:n] = ids
        self._md_positions_np[:n] = pos
        self._md_state_indices_np[:n] = si
        self._ack_consumed(cur_seq)
        self.run_mamba_decode_fast_async(
            (n,
             self._md_input_ids_np[:n],
             self._md_positions_np[:n],
             self._md_state_indices_np[:n]),
        )

    @torch.inference_mode()
    def call_mamba_decode_async(self, decode_data):
        """Launch a greedy Mamba decode from precomputed arrays + async D2H.

        Returns ``(has_result, n)``; rank 0 callers must follow up with
        ``_wait_async_mamba_tokens(n)``.
        """
        if self.world_size > 1 and self.rank == 0:
            self._write_mamba_decode_shm(*decode_data)
            self.shm.buf[self._SHM_FLAG_OFFSET] = 2  # mamba_decode_greedy
            self._signal_workers()
        return self.run_mamba_decode_fast_async(decode_data)
    def _refresh_indexer_schedule(self, context_lens: torch.Tensor) -> None:
        """Recompute the DSA indexer's paged-logits schedule for this step.

        Must run OUTSIDE CUDA graph capture/replay: with
        ``get_paged_mqa_logits_metadata`` inside the captured region the
        downstream ``fp8_fp4_paged_mqa_logits`` faults on the first replay, while
        a schedule computed outside captures and replays cleanly (the kernel is
        capturable -- verified standalone). vLLM builds the same tensor in its
        metadata builder, i.e. also outside the graph.

        ``context_lens`` is the [n] int32 decode lengths *including* the padded
        tail the captured graph covers, so the schedule matches the shape the
        graph was recorded at.
        """
        sched = self._indexer_sched
        if sched is None:
            return
        import deep_gemm as _dg

        # ``inference_mode`` explicitly: some decode sites (notably the CUDA
        # graph capture warmup) reach here with grad enabled, and the schedule is
        # computed from inference-mode buffers, which autograd refuses to track
        # ("Inference tensors cannot be saved for backward").
        with torch.inference_mode():
            cl_2d = context_lens.view(-1, 1)
            sched.copy_(
                _dg.get_paged_mqa_logits_metadata(
                    cl_2d, self._indexer_block_size, self._indexer_num_sms,
                ),
                non_blocking=True,
            )
        ctx = get_context()
        if ctx is not None:
            ctx.indexer_schedule = sched

    def _allocate_mla_kv_cache(self):
        """Allocate MLA KV cache + indexer K cache.

        The KV cache layout follows ``MLAAttention.kv_cache_dtype`` (set
        via ``FASTKERNELS_KV_CACHE_DTYPE``, default ``"auto"`` = BF16):

        * ``"auto"`` (default): BF16 cache, shape
          ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
          (1152 bytes/token for DeepSeek-V3.2). Matches stock vLLM's
          ``kv_cache_dtype=auto`` on DeepSeek-V3.2 so ``topk_indices``
          and MoE expert ids are bit-comparable.
        * ``"fp8_ds_mla"``: uint8 cache, shape
          ``[num_blocks, block_size, 656]`` (656 bytes/token).
        * ``"fp8_e4m3"``: float8_e4m3fn cache, shape
          ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
          (576 bytes/token) -- the plain per-tensor fp8 layout vLLM's
          FLASHINFER_MLA_SPARSE backend uses
          (``FlashInferMLASparseTRTLLMBackend.get_kv_cache_shape``). Same slot
          width as BF16, one byte per element instead of two.
        """
        from ..tasks.baseline.L2.mla_attention_impl import MLAAttention
        from ..tasks.baseline.L2.sparse_attn_indexer import SparseAttnIndexer

        _MLA_BLOCK_SIZE = 64  # FlashMLA uses block_size=64
        _INDEXER_CACHE_BYTES = 132
        _FP8_CACHE_BYTES = 656

        mla_layers = []
        indexer_layers = []
        for module in self.model.modules():
            if isinstance(module, MLAAttention):
                mla_layers.append(module)
            elif isinstance(module, SparseAttnIndexer):
                indexer_layers.append(module)

        num_layers = len(mla_layers)
        num_indexer_layers = len(indexer_layers)
        # Consumed by the decode scheduler (prefill-priority) and by the
        # prefill activation-arena reserve below, both of which are scoped to
        # DSA models because that is where they were measured.
        self._has_indexer_layers = num_indexer_layers > 0

        # All MLA layers must agree on the cache layout.
        if num_layers > 0:
            kv_cache_dtype = mla_layers[0].kv_cache_dtype
            for ml in mla_layers[1:]:
                assert ml.kv_cache_dtype == kv_cache_dtype, (
                    "Inconsistent kv_cache_dtype across MLA layers"
                )
        else:
            kv_cache_dtype = "auto"

        use_fp8_kv = kv_cache_dtype == "fp8_ds_mla"
        kv_lora_rank = mla_layers[0].kv_lora_rank if num_layers else 512
        rope_dim = mla_layers[0].qk_rope_head_dim if num_layers else 64
        if use_fp8_kv:
            cache_last_dim = _FP8_CACHE_BYTES
            cache_torch_dtype = torch.uint8
            bytes_per_slot = _FP8_CACHE_BYTES
            backend_desc = "FP8 KV cache"
        elif kv_cache_dtype == "fp8_e4m3":
            cache_last_dim = kv_lora_rank + rope_dim
            cache_torch_dtype = torch.float8_e4m3fn
            bytes_per_slot = cache_last_dim  # fp8 = 1 byte/element
            backend_desc = "FP8-E4M3 KV cache"
        else:
            # BF16: shape = (num_blocks, block_size, kv_lora_rank + rope_dim)
            cache_last_dim = kv_lora_rank + rope_dim
            cache_torch_dtype = torch.bfloat16
            bytes_per_slot = cache_last_dim * 2  # BF16 = 2 bytes/element
            backend_desc = "BF16 KV cache"

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        kv_bytes_per_block = num_layers * _MLA_BLOCK_SIZE * bytes_per_slot
        idx_bytes_per_block = num_indexer_layers * _MLA_BLOCK_SIZE * _INDEXER_CACHE_BYTES
        total_bytes_per_block = kv_bytes_per_block + idx_bytes_per_block

        reserve_bytes = 0
        if self.is_kimi_linear:
            reserve_bytes = self._kimi_state_cache_bytes(
                max(1, self.max_num_seqs),
            )

        # Prefill activation arena.
        #
        # ``peak`` above comes from the pre-KV warmup forward, which runs at
        # ``max_num_batched_tokens`` but BEFORE the KV cache exists -- so it never
        # exercises any path that needs the cache, notably the DSA indexer's
        # prefill gather + ``fp8_fp4_mqa_logits``, whose
        # [chunk_tokens x context] fp32 logits buffer is the largest transient in
        # a long prefill. KV sizing therefore hands the arena's memory to the KV
        # cache, and the caching allocator has to release blocks back to the
        # driver and re-acquire them every layer.
        #
        # Measured on GLM-5.2-NVFP4 (24,576-token prefill, tp=8): only 2.97 GiB
        # of transient above resident, but the allocator wants ~12 GiB of pool
        # slack to serve it without churning. With 9.2 GiB left over, one prefill
        # made 183 ``cudaFree`` (704 ms) + 186 ``cudaMalloc`` (334 ms) calls --
        # both device-synchronising, so peer ranks stalled and the wait surfaced
        # as multi-millisecond all-reduce durations on every other rank (median
        # 526 us, max 134 ms). Prefill throughput 13.8k tok/s against vLLM's
        # 27.4k; giving the arena room restored 27.0k.
        # 4 GiB restores full prefill throughput (26.9k tok/s, 0.98x of vLLM) at a
        # cost of 4.6% of the KV cache -- 29,177 blocks instead of 30,584, still
        # well clear of the 24,851 blocks long-context peaks at. Only models with a
        # DSA indexer have the un-measurable transient, so only they pay for it.
        default_arena = 4.0 if num_indexer_layers > 0 else 0.0
        arena_gib = float(os.environ.get("FASTKERNELS_PREFILL_ARENA_GIB",
                                         str(default_arena)))
        if arena_gib:
            reserve_bytes += int(arena_gib * (1 << 30))

        available_bytes = int(
            total * self.gpu_memory_utilization
            - used
            - peak
            + current
            - reserve_bytes
        )
        num_blocks = available_bytes // total_bytes_per_block
        assert num_blocks > 0, f"Not enough GPU memory for MLA KV cache on rank {self.rank}"
        self.num_blocks = num_blocks
        self.block_size = _MLA_BLOCK_SIZE

        # Update module-level BLOCK_SIZE so Sequence.num_blocks,
        # blocks_needed_for, BlockManager, prepare_prefill/decode, etc.
        # all use the MLA block size consistently.
        global BLOCK_SIZE
        BLOCK_SIZE = _MLA_BLOCK_SIZE
        # Recompute max_model_len with the new block size
        self.max_model_len = (
            (self.max_model_len + _MLA_BLOCK_SIZE - 1)
            // _MLA_BLOCK_SIZE * _MLA_BLOCK_SIZE
        )

        if self.rank == 0:
            print(f"  MLA KV cache: {num_blocks} blocks x {_MLA_BLOCK_SIZE} = "
                  f"{num_blocks * _MLA_BLOCK_SIZE} token slots ({num_layers} layers, "
                  f"{bytes_per_slot} bytes/token, dtype={kv_cache_dtype})")

        device = f"cuda:{self.rank}"
        for i, layer in enumerate(mla_layers):
            cache = torch.zeros(
                num_blocks, _MLA_BLOCK_SIZE, cache_last_dim,
                dtype=cache_torch_dtype, device=device,
            )
            layer.k_cache = cache
            layer.v_cache = cache
            # Resolve the fp8 attention scales to Python floats here: weights are
            # loaded, capture has not started, and the warmup forward ran against
            # an empty cache so it never touched the sparse decode path. Reading
            # them lazily on the first decode instead records a D2H copy into the
            # decode CUDA graph, and replay then fails with
            # ``cudaErrorInvalidAddressSpace``.
            if hasattr(layer, "finalize_kv_scales"):
                layer.finalize_kv_scales()

        # Materialize every lazily-allocated device workspace NOW, while the KV
        # cache is up but capture has not started. A workspace first allocated
        # inside a capture region belongs to that one graph's private memory
        # pool; the next captured batch size then records kernels reducing
        # atomically into another graph's pool, and replay faults
        # (``cudaErrorIllegalAddress`` / ``cudaErrorInvalidAddressSpace``). That
        # hit the DSA radix top-k and both trtllm-gen MLA workspaces -- the
        # sparse-MLA models were only runnable with ``enforce_eager`` because of
        # it. vLLM avoids the whole class by taking its workspaces from a
        # module-level buffer or its workspace manager at model init.
        _ws_device = torch.device(device)
        for module in self.model.modules():
            ensure = getattr(module, "ensure_workspaces", None)
            if ensure is not None:
                ensure(_ws_device)

        # Persistent DeepGEMM paged-MQA-logits schedule for the DSA indexer.
        # Shape is (num_sms + 1, 2) for every batch size and context length, so a
        # single buffer serves every captured decode graph. The engine refreshes
        # its CONTENTS each step, outside the graph -- see Context.indexer_schedule.
        self._indexer_sched = None
        if num_indexer_layers > 0:
            import deep_gemm as _dg

            self._indexer_block_size = _MLA_BLOCK_SIZE
            self._indexer_num_sms = torch.cuda.get_device_properties(
                _ws_device).multi_processor_count
            # ``inference_mode(False)``: KV cache allocation runs under
            # inference mode, and a buffer created there is an *inference
            # tensor*. The compiled model reads this one as an input, and the
            # CUDA-graph capture warmup traces with grad enabled, where autograd
            # rejects inference tensors ("Inference tensors cannot be saved for
            # backward"). Allocating it as a normal tensor sidesteps that; the
            # per-step ``copy_`` from an inference-mode source is still fine.
            with torch.inference_mode(False):
                probe = torch.ones(1, 1, dtype=torch.int32, device=_ws_device)
                self._indexer_sched = _dg.get_paged_mqa_logits_metadata(
                    probe, self._indexer_block_size, self._indexer_num_sms,
                ).clone()

        if num_indexer_layers > 0:
            if self.rank == 0:
                print(f"  Indexer K cache: {num_blocks} blocks x {_MLA_BLOCK_SIZE} = "
                      f"{num_blocks * _MLA_BLOCK_SIZE} token slots ({num_indexer_layers} layers, "
                      f"{_INDEXER_CACHE_BYTES} bytes/token)")
            for i, layer in enumerate(indexer_layers):
                layer.indexer_k_cache = torch.zeros(
                    num_blocks, _MLA_BLOCK_SIZE, _INDEXER_CACHE_BYTES,
                    dtype=torch.uint8, device=device,
                )
            # Shared, reused DSA-indexer prefill K-gather workspace (ONE buffer
            # for all indexer layers). Bounds the gather + logits memory so long
            # prefills chunk over it instead of allocating O(ΣN) per layer/step.
            # Sized like vLLM's get_max_prefill_buffer_size = max_model_len*40,
            # capped by the cache size and floored at max_model_len so any single
            # request always fits (see SparseAttnIndexer._prefill_topk).
            idx_head_dim = indexer_layers[0].head_dim
            idx_ws_tokens = min(40 * self.max_model_len,
                                num_blocks * _MLA_BLOCK_SIZE)
            idx_ws_tokens = max(idx_ws_tokens, self.max_model_len)
            idx_k_fp8_ws = torch.empty(
                idx_ws_tokens, idx_head_dim,
                dtype=torch.float8_e4m3fn, device=device,
            )
            idx_k_scale_ws = torch.empty(
                idx_ws_tokens, 4, dtype=torch.uint8, device=device,
            )
            for layer in indexer_layers:
                layer._gather_ws_k_fp8 = idx_k_fp8_ws
                layer._gather_ws_k_scale = idx_k_scale_ws
                layer._gather_ws_tokens = idx_ws_tokens

        if self.rank == 0:
            # The cache dtype picks the backend (see ``MLAAttention.__init__``),
            # so name the one actually in use rather than always "FlashMLA".
            backend_name = (
                "FlashInfer sparse MLA (trtllm-gen)"
                if kv_cache_dtype == "fp8_e4m3" else "FlashMLA"
            )
            print(f"  MLA attention backend: {backend_name} "
                  f"(block_size={_MLA_BLOCK_SIZE}, {backend_desc})")

        # Pre-allocate chunked prefill workspace for MLA context gathering.
        # Matches vllm's MLACommonMetadataBuilder workspace sizing.
        workspace_tokens = min(
            max(8 * self.max_model_len,
                4 * self.max_num_seqs * _MLA_BLOCK_SIZE),
            64 * 1024,
        )
        workspace_tokens = max(workspace_tokens,
                               self.max_num_seqs * _MLA_BLOCK_SIZE)
        # Workspace holds BF16 kv_c + k_pe (kv_lora_rank + qk_rope_head_dim
        # elements per token), regardless of on-disk cache layout.
        kv_lora_rank_ws = mla_layers[0].kv_lora_rank if num_layers else 512
        rope_dim_ws = mla_layers[0].qk_rope_head_dim if num_layers else 64
        self._mla_chunked_prefill_workspace = torch.empty(
            workspace_tokens, kv_lora_rank_ws + rope_dim_ws,
            dtype=torch.bfloat16, device=device,
        )
        self._mla_workspace_size = workspace_tokens

    def _build_chunked_context(self, prefill_seqs, prefill_chunk_sizes,
                               block_tables, device):
        """Build ChunkedContextMetadata for MLA prefill with prior context.

        When a prefill request has already-computed tokens (chunked prefill),
        those tokens live in the KV cache and must be gathered and attended to
        in workspace-sized chunks. This matches vllm's
        MLACommonMetadataBuilder.build() chunked context logic.
        """
        from .context import ChunkedContextMetadata

        num_prefills = len(prefill_seqs)
        context_lens = []
        for seq, chunk_size in zip(prefill_seqs, prefill_chunk_sizes):
            context_lens.append(seq.num_computed_tokens)
        context_lens_cpu = torch.tensor(context_lens, dtype=torch.int32)
        max_context_len = int(context_lens_cpu.max().item())

        if max_context_len == 0:
            return None

        num_prefills_with_context = int((context_lens_cpu > 0).sum().item())
        if num_prefills_with_context == 0:
            return None

        max_context_chunk = self._mla_workspace_size // num_prefills_with_context
        block_size = self.block_size
        max_context_chunk = (max_context_chunk // block_size) * block_size
        if max_context_chunk == 0:
            max_context_chunk = block_size
        num_chunks = (max_context_len + max_context_chunk - 1) // max_context_chunk

        chunk_starts = (
            torch.arange(num_chunks, dtype=torch.int32)
            .unsqueeze(1).expand(-1, num_prefills)
            * max_context_chunk
        )
        chunk_ends = torch.min(
            context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
        )
        chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

        cu_seq_lens_cpu = torch.zeros(
            num_chunks, num_prefills + 1, dtype=torch.int32)
        torch.cumsum(chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:],
                     dtype=torch.int32)
        chunk_total_token = cu_seq_lens_cpu[:, -1]

        max_token_num = int(chunk_total_token.max().item())
        token_to_seq_cpu = torch.zeros(
            num_chunks, max_token_num, dtype=torch.int32)
        range_idx = torch.arange(num_prefills, dtype=torch.int32)
        for i in range(num_chunks):
            chunk_t2s = torch.repeat_interleave(range_idx, chunk_seq_lens[i])
            clen = chunk_t2s.shape[0]
            token_to_seq_cpu[i, :clen] = chunk_t2s

        return ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
            starts=chunk_starts.to(device, non_blocking=True),
            seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
            seq_lens=chunk_seq_lens,
            workspace=self._mla_chunked_prefill_workspace,
            token_to_seq=token_to_seq_cpu.to(device, non_blocking=True),
            chunk_total_token=chunk_total_token.tolist(),
            has_empty_context=(chunk_seq_lens == 0).any(dim=1).tolist(),
        )
    def prepare_prefill(self, seqs):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_sq, max_sk = 0, 0
        slot_mapping = []
        max_bt = 0
        has_block_tables = False
        use_mrope = self.is_qwen_vl
        mrope_pos_list = [] if use_mrope else None

        for seq in seqs:
            sl = len(seq)
            input_ids.extend(seq.token_ids)
            if use_mrope and getattr(seq, 'mrope_positions', None) is not None:
                mrope_pos_list.append(seq.mrope_positions)
            else:
                positions.extend(range(sl))
                if use_mrope:
                    mrope_pos_list.append(
                        torch.arange(sl, dtype=torch.int64, device="cpu").unsqueeze(0).expand(3, -1)
                    )
            cu_seqlens_q.append(cu_seqlens_q[-1] + sl)
            cu_seqlens_k.append(cu_seqlens_k[-1] + sl)
            max_sq = max(sl, max_sq)
            max_sk = max(sl, max_sk)
            if not seq.block_table:  # warmup
                continue
            has_block_tables = True
            for i in range(seq.num_blocks):
                start = seq.block_table[i] * BLOCK_SIZE
                end = start + (BLOCK_SIZE if i != seq.num_blocks - 1
                               else seq.last_block_num_tokens)
                slot_mapping.extend(range(start, end))
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen

        block_tables = None
        if max_bt > 0:
            n = len(seqs)
            bt = np.full((n, max_bt), -1, dtype=np.int32)
            for i, seq in enumerate(seqs):
                if seq.block_table:
                    b = seq.block_table
                    bt[i, :len(b)] = b
            block_tables = torch.from_numpy(bt).pin_memory().cuda(non_blocking=True)

        # Per-token request id (row into block_tables). Pre-computing this
        # once per step avoids a Python for-loop + .item() sync inside every
        # MLA / DSA layer (see ``_forward_sparse_bf16`` fallback in
        # ``mla_attention_impl.py``).
        nseqs_pf = len(seqs)
        seq_lens_np = np.fromiter(
            (len(s) for s in seqs), dtype=np.int32, count=nseqs_pf,
        )
        req_id_np = np.repeat(
            np.arange(nseqs_pf, dtype=np.int32), seq_lens_np,
        )
        req_id_per_token = torch.from_numpy(req_id_np).pin_memory().cuda(
            non_blocking=True,
        )

        set_context(
            True,
            torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=(torch.int64 if self.is_deepseek_mla else torch.int32),
                         pin_memory=True).cuda(non_blocking=True),
            block_tables=block_tables,
            req_id_per_token=req_id_per_token,
        )

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        if use_mrope:
            positions_t = torch.cat(mrope_pos_list, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
        else:
            positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        return input_ids_t, positions_t

    def prepare_decode(self, seqs):
        n = len(seqs)
        ids = np.empty(n, dtype=np.int64)
        pos = np.empty(n, dtype=np.int64)
        sm = np.empty(n, dtype=np.int64)
        cl = np.empty(n, dtype=np.int32)
        use_mrope = self.is_qwen_vl
        if use_mrope:
            mrope_pos = np.empty((3, n), dtype=np.int64)
        max_bt = 0
        for i, seq in enumerate(seqs):
            ids[i] = seq.last_token
            base_pos = len(seq) - 1
            if use_mrope:
                # For decode, all 3 dims get the same position = context_len + delta
                decode_pos = base_pos + seq.mrope_position_delta
                mrope_pos[:, i] = decode_pos
            else:
                pos[i] = base_pos
            cl[i] = len(seq)
            sm[i] = seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen

        bt = np.full((n, max_bt), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            b = seq.block_table
            bt[i, :len(b)] = b
        max_cl = int(cl[:n].max())
        # Pure-decode: one token per request, so token i belongs to request i.
        # Reuse the persistent arange buffer allocated in
        # ``capture_cudagraph`` (also referenced by every captured decode
        # CUDA graph) to avoid allocating a fresh tensor each step.  Falls
        # back to an inline arange for eager runs where capture was skipped.
        req_id_per_token = getattr(self, "_decode_req_id_buf", None)
        if req_id_per_token is not None:
            req_id_per_token = req_id_per_token[:n]
        else:
            req_id_per_token = torch.arange(
                n, dtype=torch.int32, device=f"cuda:{self.rank}",
            )
        set_context(
            False,
            slot_mapping=torch.from_numpy(sm).pin_memory().cuda(non_blocking=True),
            context_lens=torch.from_numpy(cl).pin_memory().cuda(non_blocking=True),
            block_tables=torch.from_numpy(bt).pin_memory().cuda(non_blocking=True),
            max_context_len=max_cl,
            req_id_per_token=req_id_per_token,
        )
        self._refresh_indexer_schedule(get_context().context_lens)
        self._refresh_fa3_decode_schedule(get_context().context_lens)
        self._apply_pending_cross_ctx()
        if use_mrope:
            positions_t = torch.from_numpy(mrope_pos).pin_memory().cuda(non_blocking=True)
        else:
            positions_t = torch.from_numpy(pos).pin_memory().cuda(non_blocking=True)
        return (
            torch.from_numpy(ids).pin_memory().cuda(non_blocking=True),
            positions_t,
        )

    def prepare_mixed_batch(self, prefill_seqs, prefill_chunk_sizes, decode_seqs):
        """Prepare a unified mixed batch: [prefill_tokens... | decode_tokens...].

        The attention layer receives split metadata so it can dispatch
        prefill tokens to the prefill kernel and decode tokens to the
        decode kernel independently.
        """
        input_ids, positions = [], []
        slot_mapping = []
        block_size = self.block_size

        use_mrope = self.is_qwen_vl
        mrope_pos_list = [] if use_mrope else None

        # --- Prefill portion ---
        pf_cu_q, pf_cu_k = [0], [0]
        pf_max_sq, pf_max_sk = 0, 0
        pf_max_bt = 0

        for seq, chunk_size in zip(prefill_seqs, prefill_chunk_sizes):
            start_pos = seq.num_computed_tokens
            chunk_ids = seq.token_ids[start_pos:start_pos + chunk_size]
            input_ids.extend(chunk_ids)
            if use_mrope and getattr(seq, 'mrope_positions', None) is not None:
                mrope_pos_list.append(seq.mrope_positions[:, start_pos:start_pos + chunk_size])
            else:
                positions.extend(range(start_pos, start_pos + chunk_size))
                if use_mrope:
                    mrope_pos_list.append(
                        torch.arange(start_pos, start_pos + chunk_size,
                                     dtype=torch.int64, device="cpu").unsqueeze(0).expand(3, -1)
                    )

            kv_len = start_pos + chunk_size
            pf_cu_q.append(pf_cu_q[-1] + chunk_size)
            pf_cu_k.append(pf_cu_k[-1] + kv_len)
            pf_max_sq = max(chunk_size, pf_max_sq)
            pf_max_sk = max(kv_len, pf_max_sk)

            for p in range(start_pos, start_pos + chunk_size):
                slot_mapping.append(
                    seq.block_table[p // block_size] * block_size + (p % block_size)
                )
            blen = len(seq.block_table)
            if blen > pf_max_bt:
                pf_max_bt = blen

        num_prefill_tokens = len(input_ids)
        num_prefill_seqs = len(prefill_seqs)

        # Build prefill block table
        prefill_block_tables = None
        if pf_max_bt > 0 and num_prefill_seqs > 0:
            pbt = np.full((num_prefill_seqs, pf_max_bt), -1, dtype=np.int32)
            for i, seq in enumerate(prefill_seqs):
                b = seq.block_table
                pbt[i, :len(b)] = b
            prefill_block_tables = torch.from_numpy(pbt).pin_memory().cuda(non_blocking=True)

        # --- Decode portion ---
        nd = len(decode_seqs)
        dc_cl = np.empty(nd, dtype=np.int32)
        dc_max_bt = 0
        for i, seq in enumerate(decode_seqs):
            input_ids.append(seq.last_token)
            pos = len(seq) - 1
            if use_mrope:
                delta = getattr(seq, 'mrope_position_delta', 0)
                p = pos + delta
                mrope_pos_list.append(
                    torch.tensor([[p], [p], [p]], dtype=torch.int64, device="cpu"))
            else:
                positions.append(pos)
            dc_cl[i] = len(seq)
            slot_mapping.append(
                seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
            )
            blen = len(seq.block_table)
            if blen > dc_max_bt:
                dc_max_bt = blen

        dc_bt = np.full((nd, dc_max_bt), -1, dtype=np.int32) if nd > 0 else np.empty((0, 0), dtype=np.int32)
        for i, seq in enumerate(decode_seqs):
            b = seq.block_table
            dc_bt[i, :len(b)] = b
        dc_max_cl = int(dc_cl[:nd].max()) if nd > 0 else 0

        logit_idx = []
        for i in range(num_prefill_seqs):
            logit_idx.append(pf_cu_q[i + 1] - 1)
        for j in range(nd):
            logit_idx.append(num_prefill_tokens + j)

        chunked_context = None
        if self.is_deepseek_mla and num_prefill_seqs > 0:
            chunked_context = self._build_chunked_context(
                prefill_seqs, prefill_chunk_sizes, prefill_block_tables,
                device=torch.device(f"cuda:{self.rank}"),
            )

        # Per-token request id, aligned with the flat token layout
        # built above: ``[prefill_tokens..., decode_tokens...]``.
        # The unified block_table used downstream (see
        # ``_forward_sparse_bf16``) is laid out as
        # ``[decode_seqs..., prefill_seqs...]``; hence prefill tokens map
        # to rows ``num_decode_seqs + r`` and decode tokens to row ``j``.
        total_tokens = num_prefill_tokens + nd
        if total_tokens > 0:
            if num_prefill_tokens > 0:
                pf_ids_np = np.repeat(
                    (nd + np.arange(num_prefill_seqs, dtype=np.int32)),
                    np.asarray(prefill_chunk_sizes, dtype=np.int32),
                )
            else:
                pf_ids_np = np.empty(0, dtype=np.int32)
            dc_ids_np = np.arange(nd, dtype=np.int32)
            req_id_np = np.concatenate([pf_ids_np, dc_ids_np])
            req_id_per_token = torch.from_numpy(req_id_np).pin_memory().cuda(
                non_blocking=True,
            )
        else:
            req_id_per_token = None

        set_mixed_context(
            slot_mapping=torch.tensor(slot_mapping,
                                      dtype=(torch.int64 if self.is_deepseek_mla else torch.int32),
                                      pin_memory=True).cuda(non_blocking=True),
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=nd,
            num_prefill_seqs=num_prefill_seqs,
            prefill_cu_seqlens_q=torch.tensor(pf_cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            prefill_cu_seqlens_k=torch.tensor(pf_cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            prefill_max_seqlen_q=pf_max_sq,
            prefill_max_seqlen_k=pf_max_sk,
            prefill_block_tables=prefill_block_tables,
            decode_context_lens=torch.from_numpy(dc_cl).pin_memory().cuda(non_blocking=True) if nd > 0 else None,
            decode_block_tables=torch.from_numpy(dc_bt).pin_memory().cuda(non_blocking=True) if nd > 0 else None,
            decode_max_context_len=dc_max_cl,
            logit_indices=torch.tensor(logit_idx, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            chunked_context=chunked_context,
            req_id_per_token=req_id_per_token,
        )

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if use_mrope:
            positions_t = torch.cat(mrope_pos_list, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
        else:
            positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids_t, positions_t

    @torch.inference_mode()
    def _compute_logits(self, hidden_states, skip_final_softcap=False):
        if skip_final_softcap:
            raw_logits_fn = getattr(self.model, "compute_logits_no_softcap", None)
            if raw_logits_fn is not None:
                return raw_logits_fn(hidden_states)
        return self.model.compute_logits(hidden_states)

    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill, inputs_embeds=None,
                  deepstack_embeds=None, encoder_outputs=None,
                  skip_final_softcap=False):
        # Any non-replay forward (prefill, mixed, eager decode, capture warmup)
        # breaks the device token handoff: it takes its ids from the argument, not
        # from graph_vars, so a staged value would be stale for the next replay.
        self._staged_ids_n = -1
        # One traced signature for the compiled inner model. DeepStack adds a
        # residual at the first few decoder layers; vLLM keeps that inside its
        # compiled graph, reading from persistent buffers that it zeroes after
        # each multimodal step (_clear_deepstack_input_embeds). We used to route
        # any forward carrying deepstack_embeds to an *uncompiled* model instead,
        # which cost ~27% of Qwen3-VL's prefill throughput (52k tok/s against
        # 81.5k for compiled Qwen2-VL on the same run, ~71k after normalising for
        # model size). Passing zeroed buffer slices when there is no vision keeps
        # the graph structurally identical between decode and multimodal prefill,
        # so there is exactly one artifact and no guard-dispatch hazard -- the
        # earlier attempt to just hand deepstack_embeds to the graph traced
        # without them silently reused the wrong graph and dropped the residual.
        # The no-op adds touch only len(deepstack_visual_indexes)=3 layers, i.e.
        # 3 adds of (num_tokens, hidden) per step, negligible against the ~113GB
        # of KV a full decode step reads.
        if deepstack_embeds is None:
            _ds_bufs = getattr(self, "_deepstack_bufs", None)
            if _ds_bufs:
                _n = input_ids.shape[0]
                deepstack_embeds = [b[:_n] for b in _ds_bufs]
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs_list[-1]:
            model = self.model
            if is_prefill and self.is_bitnet and self._compiled:
                # BitNet's BitLinear switches between bf16 fake-quant prefill
                # and int2 decode by reading the runtime Context. torch.compile
                # specializes that Python branch during decode graph capture,
                # so compiled prefill would incorrectly use the int2 path.
                model = getattr(self, "_eager_model", self.model)
            # For compiled VL models, always compute inputs_embeds outside
            # the compiled graph (matching vLLM).  The compiled inner
            # Qwen3Model was traced with inputs_embeds, so it must always
            # receive one.
            if self.is_qwen_vl and self._compiled and inputs_embeds is None:
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if inputs_embeds is not None:
                kwargs = {"inputs_embeds": inputs_embeds}
                if deepstack_embeds is not None:
                    kwargs["deepstack_embeds"] = deepstack_embeds
                return self._compute_logits(
                    model(input_ids, positions, **kwargs),
                    skip_final_softcap=skip_final_softcap,
                )
            if encoder_outputs is not None:
                return self._compute_logits(
                    model(input_ids, positions, encoder_outputs=encoder_outputs),
                    skip_final_softcap=skip_final_softcap,
                )
            return self._compute_logits(
                model(input_ids, positions),
                skip_final_softcap=skip_final_softcap,
            )
        # Decode path: CUDA graph replay.
        # For VL models, embed_fn is recorded inside the graph — updating
        # input_ids in graph_vars is sufficient; the graph replays embed_fn
        # on the new ids to produce inputs_embeds internally.
        bs = input_ids.size(0)
        ctx = get_context()
        graph_bs = self._graph_bs_for_n[bs]
        gv = self.graph_vars
        gv["input_ids"][:bs] = input_ids
        if self.is_qwen_vl:
            gv["positions"][:, :bs] = positions
        else:
            gv["positions"][:bs] = positions
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        if bs < graph_bs:
            gv["slot_mapping"][bs:graph_bs].fill_(-1)
            gv["context_lens"][bs:graph_bs].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        bt = ctx.block_tables
        gv["block_tables"][:bs, :bt.size(1)] = bt
        self._refresh_fa3_decode_schedule(gv["context_lens"][:graph_bs])
        self.graphs[graph_bs].replay()
        return self.model.compute_logits(gv["outputs"][:bs])

    @torch.inference_mode()
    def run_decode_greedy(self, seqs):
        """Fused decode path for greedy sampling with TP.
        Returns GPU tensor (rank 0) or list (TP=1).
        """
        decode_data = self._prepare_decode_arrays(seqs)
        return self.run_decode_greedy_fast(decode_data)

    @torch.inference_mode()
    def run_decode_greedy_fast(self, decode_data):
        """Fast decode: receives precomputed arrays instead of Sequence objects.
        
        Returns GPU tensor (rank 0) or None (other ranks).
        Does NOT call .tolist() -- caller is responsible for syncing.
        """
        n, ids_np, pos_np, sm_np, cl_np, bt_np = decode_data

        if self.enforce_eager:
            return self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)
        if n > self.graph_bs_list[-1]:
            return self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)

        self._run_graph_from_numpy(n, ids_np, pos_np, sm_np, cl_np, bt_np)
        return self._greedy_from_hidden(n)

    def _run_graph_from_numpy(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        """Copy numpy arrays into graph vars and replay the CUDA graph."""
        gv = self.graph_vars
        graph_bs = self._graph_bs_for_n[n]
        prev_n = getattr(self, '_prev_decode_n', -1)

        # Skip the token H2D copy when the previous step staged them on device
        # (see _stage_next_input_ids). ``ids_np`` is then stale by design -- the
        # host may not have read the token yet.
        if self._staged_ids_n == n:
            self._staged_ids_n = -1
        else:
            gv["input_ids"][:n].copy_(torch.from_numpy(ids_np), non_blocking=True)
        if self.is_qwen_vl:
            gv["positions"][:, :n].copy_(torch.from_numpy(pos_np), non_blocking=True)
        else:
            gv["positions"][:n].copy_(torch.from_numpy(pos_np), non_blocking=True)
        gv["slot_mapping"][:n].copy_(torch.from_numpy(sm_np), non_blocking=True)
        if n < graph_bs and n != prev_n:
            gv["slot_mapping"][n:graph_bs].fill_(-1)
            gv["context_lens"][n:graph_bs].zero_()
        gv["context_lens"][:n].copy_(torch.from_numpy(cl_np), non_blocking=True)
        gv["block_tables"][:n, :bt_np.shape[1]].copy_(
            torch.from_numpy(bt_np), non_blocking=True
        )
        self._prev_decode_n = n
        # Outside the graph, and after context_lens for this step is in place.
        self._refresh_indexer_schedule(gv["context_lens"][:graph_bs])
        self._refresh_fa3_decode_schedule(gv["context_lens"][:graph_bs])
        self.graphs[graph_bs].replay()

    @torch.inference_mode()
    def run_decode_greedy_fast_async(self, decode_data):
        """Like run_decode_greedy_fast but starts async D2H copy.

        Returns (has_result, n) -- caller must call _wait_async_tokens(n)
        later to get the Python list of token IDs.
        """
        n, ids_np, pos_np, sm_np, cl_np, bt_np = decode_data

        if self.enforce_eager:
            result = self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)
            if result is not None:
                self._issue_token_copy(n, result)
                return True, n
            return False, n
        if n > self.graph_bs_list[-1]:
            result = self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)
            if result is not None:
                self._issue_token_copy(n, result)
                return True, n
            return False, n

        self._run_graph_from_numpy(n, ids_np, pos_np, sm_np, cl_np, bt_np)
        has_result = self._greedy_from_hidden_async(n)
        return has_result, n

    def _run_decode_greedy_eager(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        # Eager decode reads its tokens from ``ids_np``, so any device handoff is
        # unusable here and must not leak into the next graph replay.
        self._invalidate_staged_ids()
        """Eager decode path for greedy sampling with TP (no CUDA graphs)."""
        self._eager_input_ids[:n].copy_(torch.from_numpy(ids_np), non_blocking=True)
        if self.is_qwen_vl:
            self._eager_positions[:, :n].copy_(torch.from_numpy(pos_np), non_blocking=True)
        else:
            self._eager_positions[:n].copy_(torch.from_numpy(pos_np), non_blocking=True)
        bt_cols = bt_np.shape[1]
        self._eager_slot_mapping[:n].copy_(torch.from_numpy(sm_np), non_blocking=True)
        self._eager_context_lens[:n].copy_(torch.from_numpy(cl_np), non_blocking=True)
        self._eager_block_tables[:n, :bt_cols].copy_(
            torch.from_numpy(bt_np), non_blocking=True)

        input_ids = self._eager_input_ids[:n]
        if self.is_qwen_vl:
            positions = self._eager_positions[:, :n]
        else:
            positions = self._eager_positions[:n]
        slot_mapping = self._eager_slot_mapping[:n]
        context_lens = self._eager_context_lens[:n]
        if ATTN_BACKEND_CONFIG.use_trtllm:
            # Full width, NOT ``[:n, :bt_cols]``. FlashInfer's trtllm-gen launcher
            # derives the page-table row stride from ``size(-1)``, not ``stride(0)``,
            # so a column slice of a wider allocation makes every row but row 0 read
            # the wrong KV pages. The consuming ops defensively call
            # ``block_table.contiguous()``, which repairs the stride but pays a copy
            # per layer per step; handing over the full width makes that a no-op and
            # protects any consumer that lacks the guard. The kernel walks only
            # ``ceil(seq_len/page)`` columns per row -- exactly the ones copied above
            # -- so the untouched tail is never read.
            block_tables = self._eager_block_tables[:n]
        else:
            block_tables = self._eager_block_tables[:n, :bt_cols]

        req_id_per_token = getattr(self, "_decode_req_id_buf", None)
        if req_id_per_token is not None:
            req_id_per_token = req_id_per_token[:n]
        else:
            # Eager decode without a captured CUDA graph never allocated the
            # persistent arange buffer. Pure-decode is one token per request
            # (token i -> request i); without this the sparse-MLA
            # ``convert_indices`` sees an all-zeros req map and every decode
            # token past the first indexes request 0's (shorter) block table,
            # truncating its sparse-attention KV. Mirrors the fallback at the
            # non-eager ``prepare_decode`` above.
            req_id_per_token = torch.arange(
                n, dtype=torch.int32, device=f"cuda:{self.rank}",
            )
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            max_context_len=int(cl_np.max()),
            req_id_per_token=req_id_per_token,
        )
        self._refresh_indexer_schedule(context_lens)
        self._refresh_fa3_decode_schedule(context_lens)
        self._apply_pending_cross_ctx()
        use_bitnet_eager_model = self.is_bitnet and self._compiled
        model = getattr(self, "_eager_model", self.model) if use_bitnet_eager_model else self.model
        if use_bitnet_eager_model:
            disable_custom_ops()
        try:
            if self.is_qwen_vl and self._compiled:
                inputs_embeds = model.get_input_embeddings()(input_ids)
                # Same uniform signature as run_model: the compiled graph was
                # traced with deepstack_embeds present, so every entry point has
                # to supply them (zeroed here -> no-op adds).
                _ds_bufs = getattr(self, "_deepstack_bufs", None)
                _kw = ({"deepstack_embeds":
                        [b[:input_ids.shape[0]] for b in _ds_bufs]}
                       if _ds_bufs else {})
                hidden = model(input_ids, positions,
                               inputs_embeds=inputs_embeds, **_kw)
            else:
                hidden = model(input_ids, positions)
            lm_head = model.lm_head
            logits = lm_head.linear_op(hidden, lm_head.embedding_op.emb.weight).float()
            max_vals, max_idxs = logits.max(dim=-1)
        finally:
            if use_bitnet_eager_model:
                enable_custom_ops()
        reset_context()

        if self.world_size > 1:
            info = self._greedy_info
            info[:n, 0] = max_vals
            info[:n, 1] = max_idxs.float()
            vocab_offset = lm_head.per_partition * self.rank
            info[:n, 1] += vocab_offset
            dist.all_gather(self._greedy_gathered, info)
            all_info = self._greedy_all_info
            torch.stack(self._greedy_gathered, out=all_info)
            best_rank = all_info[:, :n, 0].argmax(dim=0)
            return all_info[best_rank, self._greedy_arange[:n], 1].to(torch.int64)
        else:
            return max_idxs

    def _init_greedy_buffers(self):
        """Pre-allocate buffers for gather_greedy to avoid per-step allocation."""
        max_bs = self.max_num_seqs
        dev = f"cuda:{self.rank}"
        self._greedy_info = torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
        self._greedy_gathered = [
            torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
            for _ in range(self.world_size)
        ]
        self._greedy_all_info = torch.zeros(self.world_size, max_bs, 2, dtype=torch.float32, device=dev)
        self._greedy_arange = torch.arange(max_bs, device=dev)

        max_num_blocks = (self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        # Back the per-step decode staging arrays with *pinned* host memory.
        #
        # These are written as numpy and then handed to
        # ``device_tensor.copy_(torch.from_numpy(arr), non_blocking=True)``.
        # ``cudaMemcpyAsync`` from *pageable* memory is synchronous, so
        # ``non_blocking=True`` was a no-op and every per-step H2D copy blocked
        # the host. A batch-64 x 8192 decode trace measured 2114 ms across 15318
        # ``cudaMemcpyAsync`` calls (~138 us each) against vLLM's 13 ms across
        # 1848 calls (~7 us) -- vLLM stages through pinned buffers, so its copies
        # are genuinely async. Keeping the torch tensor alive and exposing a
        # ``.numpy()`` view leaves every existing call site unchanged while
        # making the copies actually asynchronous.
        def _pinned(shape, torch_dtype):
            # device="cpu" is required: the engine sets a CUDA default device, and
            # pin_memory only applies to CPU tensors.
            t = torch.empty(shape, dtype=torch_dtype, device="cpu",
                            pin_memory=True)
            return t, t.numpy()

        self._pin_ids, self._np_ids = _pinned(max_bs, torch.int64)
        if self.is_qwen_vl:
            self._pin_pos, self._np_pos = _pinned((3, max_bs), torch.int64)
        else:
            self._pin_pos, self._np_pos = _pinned(max_bs, torch.int64)
        # DeepSeek MLA FP8 KV cache stores require int64 slot_mapping;
        # the FA3 path only needs int32.
        sm_np_dtype = np.int64 if self.is_deepseek_mla else np.int32
        sm_torch_dtype = torch.int64 if self.is_deepseek_mla else torch.int32
        self._pin_sm, self._np_sm = _pinned(max_bs, sm_torch_dtype)
        self._pin_cl, self._np_cl = _pinned(max_bs, torch.int32)
        self._pin_bt, self._np_bt = _pinned((max_bs, max_num_blocks), torch.int32)
        self._np_bt.fill(-1)

        self._eager_input_ids = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        if self.is_qwen_vl:
            self._eager_positions = torch.zeros(3, max_bs, dtype=torch.int64, device=dev)
        else:
            self._eager_positions = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._eager_slot_mapping = torch.zeros(max_bs, dtype=sm_torch_dtype, device=dev)
        self._eager_context_lens = torch.zeros(max_bs, dtype=torch.int32, device=dev)
        self._eager_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=dev)

        # Async D2H: pinned buffers + copy stream for pipelined decode.
        # A ring, not a single slot: with host run-ahead (see the decode fast path
        # in generate) step N+1's copy is issued while step N's tokens have not
        # been read off the host yet, so a single buffer/event pair would be
        # overwritten before it was consumed. Two outstanding copies is the most
        # the loop ever has; three slots leaves headroom.
        self._N_COPY_SLOTS = 3
        self._pinned_token_ids_ring = [
            torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(self._N_COPY_SLOTS)
        ]
        self._copy_events = [torch.cuda.Event()
                             for _ in range(self._N_COPY_SLOTS)]
        self._copy_stream = torch.cuda.Stream(device=dev)
        # Issue/harvest cursors into the ring; the difference is the number of
        # copies in flight. Harvest order is FIFO, matching issue order, so the
        # caller does not have to carry a slot id around.
        self._copy_issued = 0
        self._copy_harvested = 0

    def _issue_token_copy(self, n: int, gpu_ids: torch.Tensor) -> None:
        """Start the D2H copy of this step's sampled tokens into a free ring slot.

        On the MAIN stream, deliberately. ``gpu_ids`` is usually a view of a
        persistent graph output buffer (``lm_max_idxs``), which the next replay
        overwrites; a copy on a side stream only survived because the host used to
        block on it before launching that replay. With run-ahead there is no such
        block, so a side-stream copy would race the next replay. Enqueuing it on
        the main stream orders it between the two replays for free -- it is a
        few hundred bytes, and the event recorded after it lets the host poll for
        completion without waiting on anything that follows.
        """
        if self._copy_issued - self._copy_harvested >= self._N_COPY_SLOTS:
            # Should not happen with the current loop; drain rather than corrupt.
            self._wait_async_tokens(n)
        slot = self._copy_issued % self._N_COPY_SLOTS
        stream = torch.cuda.current_stream()
        self._pinned_token_ids_ring[slot][:n].copy_(gpu_ids, non_blocking=True)
        self._copy_events[slot].record(stream)
        self._copy_issued += 1

    def _stage_next_input_ids(self, n: int, gpu_ids: torch.Tensor) -> None:
        """Hand this step's sampled tokens to the next graph replay ON DEVICE.

        The decode graph reads its tokens from ``graph_vars["input_ids"]``, which
        was filled by an H2D copy of a host numpy array -- so the host had to
        know the sampled token before it could launch the next step, and the
        measured cost of that round trip is 1.56 ms of GPU idle per step at bs=1
        (vLLM: 0.74 ms). Writing the token device-to-device here removes the
        dependency: the copy is issued on the same stream as the replay, so it is
        ordered after the step that produced it and before the step that consumes
        it, with no host involvement.

        ``_staged_ids_n`` is the batch size the staged tokens correspond to; a
        consumer must check it matches, and any non-graph forward invalidates it.
        """
        self.graph_vars["input_ids"][:n].copy_(gpu_ids, non_blocking=True)
        self._staged_ids_n = n

    def _invalidate_staged_ids(self) -> None:
        """Drop the device token handoff (a non-graph forward ran in between)."""
        self._staged_ids_n = -1

    def _greedy_from_hidden(self, n):
        """Use CUDA-graph-captured LM head + local argmax, then allgather.
        
        Returns GPU tensor of token IDs (rank 0) or None (other ranks).
        Caller must call .tolist() to sync.
        """
        gv = self.graph_vars

        if self.world_size == 1:
            ids = gv["lm_max_idxs"][:n]
            self._stage_next_input_ids(n, ids)
            return ids

        local_max_vals = gv["lm_max_vals"][:n]
        local_max_idxs = gv["lm_max_idxs"][:n] + self.model.lm_head.vocab_start

        info = self._greedy_info[:n]
        info[:, 0] = local_max_vals
        info[:, 1] = local_max_idxs.float()

        gathered = [g[:n] for g in self._greedy_gathered]
        dist.all_gather(gathered, info)

        all_info = self._greedy_all_info
        torch.stack(gathered, out=all_info[:, :n])
        best_rank = all_info[:, :n, 0].argmax(dim=0)
        token_ids = all_info[:, :n, 1].long()[best_rank, self._greedy_arange[:n]]

        # Stage on EVERY rank, before the rank-0 gate: the all_gather above makes
        # ``token_ids`` bit-identical across ranks, so each rank can fill its own
        # graph input buffer without rank 0 broadcasting the host value.
        self._stage_next_input_ids(n, token_ids)

        if self.rank == 0:
            return token_ids
        return None

    def _greedy_from_hidden_async(self, n):
        """Like _greedy_from_hidden but starts async D2H copy.

        After calling this, the caller must eventually call
        _wait_async_tokens(n) to get the Python list of token IDs.
        Between the two calls, the CPU is free to do other work.
        """
        gpu_ids = self._greedy_from_hidden(n)
        if gpu_ids is not None:
            self._issue_token_copy(n, gpu_ids)
        return gpu_ids is not None

    def _wait_async_tokens(self, n):
        """Wait for the oldest outstanding D2H copy and return its token list."""
        slot = self._copy_harvested % self._N_COPY_SLOTS
        self._copy_events[slot].synchronize()
        self._copy_harvested += 1
        return self._pinned_token_ids_ring[slot][:n].tolist()

    def _prepare_decode_arrays(self, seqs):
        """Precompute numpy arrays for decode - uses pre-allocated buffers."""
        # Full rebuild means the caller has real host tokens (it runs after a
        # finish, or on the first decode after prefill), so use them and drop any
        # device handoff.
        self._invalidate_staged_ids()
        n = len(seqs)
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        max_bt = 0
        bs = BLOCK_SIZE
        for i, seq in enumerate(seqs):
            tids = seq.token_ids
            if tids is not None:
                slen = len(tids)
                ids_np[i] = tids[-1]
            else:
                slen = seq._num_tokens
                ids_np[i] = seq._last_token
            if self.is_qwen_vl:
                decode_pos = slen - 1 + seq.mrope_position_delta
                pos_np[:, i] = decode_pos
            else:
                pos_np[i] = slen - 1
            cl_np[i] = slen
            bt = seq.block_table
            blen = len(bt)
            r = slen % bs
            sm_np[i] = bt[-1] * bs + (r - 1 if r else bs - 1)
            if blen > max_bt:
                max_bt = blen
        bt_np = self._np_bt
        for i, seq in enumerate(seqs):
            b = seq.block_table
            blen = len(b)
            bt_np[i, :blen] = b
            if blen < max_bt:
                bt_np[i, blen:max_bt] = -1
        self._prev_max_bt = max_bt
        if self.is_qwen_vl:
            return (n, ids_np[:n], pos_np[:, :n], sm_np[:n], cl_np[:n], bt_np[:n, :max_bt])
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n], bt_np[:n, :max_bt])

    def _update_decode_arrays_incremental(self, n, token_ids, decode_seqs):
        """Update pre-allocated decode arrays incrementally after a decode step.

        Much faster than _prepare_decode_arrays: vectorized numpy ops +
        only touches block table rows that crossed a block boundary.
        """
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        bt_np = self._np_bt
        bs = BLOCK_SIZE

        # ``token_ids is None`` means the previous step's tokens were handed to
        # the graph on device (see _stage_next_input_ids) and the host has not read
        # them yet -- everything else in this update is token-independent.
        if token_ids is not None:
            ids_np[:n] = token_ids
        if self.is_qwen_vl:
            pos_np[:, :n] += 1
        else:
            pos_np[:n] += 1
        cl_np[:n] += 1
        sm_np[:n] += 1

        boundary_mask = cl_np[:n] % bs == 1
        if boundary_mask.any():
            max_bt = 0
            for i in np.where(boundary_mask)[0]:
                seq = decode_seqs[i]
                bt = seq.block_table
                blen = len(bt)
                sm_np[i] = bt[-1] * bs
                bt_np[i, :blen] = bt
                if blen > max_bt:
                    max_bt = blen
            if max_bt == 0:
                max_bt = self._prev_max_bt
            else:
                for i in np.where(~boundary_mask)[0]:
                    blen = len(decode_seqs[i].block_table)
                    if blen > max_bt:
                        max_bt = blen
                self._prev_max_bt = max_bt
        else:
            max_bt = self._prev_max_bt
        if self.is_qwen_vl:
            return (n, ids_np[:n], pos_np[:, :n], sm_np[:n], cl_np[:n],
                    bt_np[:n, :max_bt])
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n],
                bt_np[:n, :max_bt])

    def _write_decode_shm(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        """Write decode arrays directly into SHM with binary layout.

        Layout: [n(2)][max_bt(2)][staged(1)][pad(3)][ids(n*8)][pos(n*8 or 3*n*8)][sm(n*4)][cl(n*4)][bt(n*max_bt*4)]

        The header is padded to 8 bytes so the int64 fields that follow stay
        8-byte aligned; an odd header makes every ``np.frombuffer`` on the reader
        side unaligned.
        pos_np is (n,) for standard models or (3, n) for MRoPE models.

        ``staged`` carries rank 0's decision about the device token handoff. The
        workers must not decide for themselves: they would skip the ids copy
        whenever their own last replay had the same batch size, which is not the
        same question -- a batch that was reordered (preemption, a swap) has the
        same n but different tokens per slot, and rank 0 rebuilds the host array
        in that case. Taking the flag from rank 0 keeps every rank reading the
        tokens from the same place.
        """
        max_bt = bt_np.shape[1]
        buf = self.shm.buf
        arrays = (ids_np, pos_np.ravel(), sm_np, cl_np, bt_np)
        payload_bytes = 8 + sum(arr.nbytes for arr in arrays)
        if payload_bytes > self._SHM_ACK_OFFSET:
            # Assigning past the data region would silently truncate the
            # lvalue slice and raise an opaque memoryview structure error,
            # so fail with the numbers needed to diagnose it.
            raise RuntimeError(
                f"Decode SHM payload too large: {payload_bytes} bytes for "
                f"n={n}, max_bt={max_bt} (limit {self._SHM_ACK_OFFSET}); "
                f"_init_shm_layout sized for max_num_seqs="
                f"{self.max_num_seqs}, max_model_len={self.max_model_len}"
            )
        buf[0:2] = n.to_bytes(2, "little")
        buf[2:4] = max_bt.to_bytes(2, "little")
        buf[4] = 1 if self._staged_ids_n == n else 0
        off = 8
        for arr in arrays:
            nb = arr.nbytes
            buf[off:off+nb] = arr.tobytes()
            off += nb

    def _loop_decode_greedy(self, cur_seq):
        """Worker fast path: read decode arrays from SHM without pickle.

        Must mirror :meth:`_write_decode_shm` exactly. In particular, MLA
        models write ``slot_mapping`` as ``int64`` (8 bytes per element)
        because the FP8 paged KV cache stores require it; non-MLA models
        use ``int32``. Reading the wrong dtype here both garbles ``sm``
        and shifts every following field, which produces inconsistent
        decode metadata across ranks and deadlocks TP collectives.
        """
        buf = self.shm.buf
        n = int.from_bytes(buf[0:2], "little")
        max_bt = int.from_bytes(buf[2:4], "little")
        # Mirror rank 0's device-handoff decision (see _write_decode_shm).
        self._staged_ids_n = n if buf[4] else -1
        off = 8
        ids_np = np.frombuffer(buf, dtype=np.int64, count=n, offset=off).copy(); off += n * 8
        if self.is_qwen_vl:
            pos_np = np.frombuffer(buf, dtype=np.int64, count=3*n, offset=off).copy().reshape(3, n); off += 3 * n * 8
        else:
            pos_np = np.frombuffer(buf, dtype=np.int64, count=n, offset=off).copy(); off += n * 8
        if self.is_deepseek_mla:
            sm_np = np.frombuffer(buf, dtype=np.int64, count=n, offset=off).copy(); off += n * 8
        else:
            sm_np = np.frombuffer(buf, dtype=np.int32, count=n, offset=off).copy(); off += n * 4
        cl_np = np.frombuffer(buf, dtype=np.int32, count=n, offset=off).copy(); off += n * 4
        bt_np = np.frombuffer(buf, dtype=np.int32, count=n*max_bt, offset=off).copy().reshape(n, max_bt)
        self._ack_consumed(cur_seq)
        self.run_decode_greedy_fast((n, ids_np, pos_np, sm_np, cl_np, bt_np))

    def call_decode_greedy(self, seqs):
        """Optimized call for greedy decode: uses SHM spin-wait signaling.
        
        Returns GPU tensor of token IDs (doesn't sync).
        Caller must call .tolist() to get Python list.
        """
        if self.world_size > 1 and self.rank == 0:
            if _PROFILE:
                _t0 = time.perf_counter()
            decode_data = self._prepare_decode_arrays(seqs)
            self._write_decode_shm(*decode_data)
            if _PROFILE:
                _t1 = time.perf_counter()
            self.shm.buf[self._SHM_FLAG_OFFSET] = 1  # mark as decode_greedy
            self._signal_workers()
            if _PROFILE:
                _t2 = time.perf_counter()
            result = self.run_decode_greedy_fast(decode_data)
            if _PROFILE:
                torch.cuda.synchronize()
                _t3 = time.perf_counter()
                pd = getattr(self, '_call_profile', None)
                if pd is None:
                    pd = {"prepare": 0.0, "signal": 0.0, "gpu_exec": 0.0, "n_calls": 0}
                    self._call_profile = pd
                pd["prepare"] += _t1 - _t0
                pd["signal"] += _t2 - _t1
                pd["gpu_exec"] += _t3 - _t2
                pd["n_calls"] += 1
            return result
        return self.run_decode_greedy(seqs)

    def call_decode_greedy_async(self, decode_data):
        """Launch greedy decode from precomputed arrays and start async D2H.

        Returns (has_result, n). Caller must call
        model_runner._wait_async_tokens(n) to get token IDs.
        """
        n = decode_data[0]
        if self.world_size > 1 and self.rank == 0:
            self._write_decode_shm(*decode_data)
            self.shm.buf[self._SHM_FLAG_OFFSET] = 1
            self._signal_workers()
            return self.run_decode_greedy_fast_async(decode_data)
        return self.run_decode_greedy_fast_async(decode_data)

    # ``@torch.inference_mode()`` is load-bearing on every method reachable from
    # ``loop()``. Rank 0 calls these from inside ``generate()``, which is already
    # decorated, so they inherit the ambient inference mode; worker ranks reach
    # them from ``loop()`` with *no* grad context, and since the model's weights
    # are plain nn.Parameters (requires_grad=True), the forward then records an
    # autograd graph and pins every layer's intermediate activation. Measured on
    # tp=2 Qwen2-VL: rank 1's vision encoder grew ~46G over its post-capture
    # baseline and OOM'd in the patch merger after all 32 blocks, while rank 0's
    # identical warmup forward peaked at 1.4G. See the same warning on
    # _profile_activation_peak.
    @torch.inference_mode()
    def run(self, seqs, is_prefill, prefill_chunk_lens=None):
        if self.is_kimi_linear:
            return self._run_kimi_linear_batch(
                seqs, is_prefill, prefill_chunk_lens,
            )
        if self.is_qwen3_next:
            return self._run_qwen3_next_batch(
                seqs, is_prefill, prefill_chunk_lens,
            )
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill
            else self.prepare_decode(seqs)
        )
        result = self.run_model(input_ids, positions, is_prefill)
        reset_context()
        return result

    @torch.inference_mode()
    def run_mixed(
        self,
        prefill_seqs,
        prefill_chunk_sizes,
        decode_seqs,
        skip_final_softcap=False,
    ):
        input_ids, positions = self.prepare_mixed_batch(
            prefill_seqs, prefill_chunk_sizes, decode_seqs,
        )
        result = self.run_model(
            input_ids, positions, True,
            skip_final_softcap=skip_final_softcap,
        )
        reset_context()
        return result

    def _strip_mm_tensors(self, seqs):
        """Return lightweight copies of sequences for SHM dispatch (no large tensors)."""
        if self.world_size == 1:
            # ``call()`` invokes the method in-process at tp=1, so there is no
            # pickling to strip for -- and a steady-state mixed multimodal step
            # carries hundreds of decode seqs, so copying every __dict__ here
            # was pure per-step overhead.
            return seqs
        stripped = []
        for s in seqs:
            c = Sequence.__new__(Sequence)
            c.__dict__.update(s.__dict__)
            c.pixel_values = None
            c.video_pixel_values = None
            c.input_audio_features = None
            c.audio_feature_lengths = None
            c.encoder_features = None
            stripped.append(c)
        return stripped

    @torch.inference_mode()
    def _run_mm_lm(self, prefill_seqs, prefill_chunk_sizes, decode_seqs,
                   vis_cache_map):
        """Run LM forward with vision embeddings from _vis_cache.

        All ranks have vision outputs in self._vis_cache from prior
        _broadcast_visual calls. This method:
        1. Computes text embeddings via embed_fn (TP allreduce)
        2. Merges cached vision embeddings into text embeddings
        3. Runs LM forward pass

        vis_cache_map: list of dicts per prefill seq:
          [{"cache_idx": int, "modality": str, "thw": [[t,h,w],...]}]
          None entries mean no vision for that seq.
        """
        input_ids, positions = self.prepare_mixed_batch(
            prefill_seqs, prefill_chunk_sizes, decode_seqs,
        )

        if self.is_qwen_vl and self.world_size > 1:
            if self.rank == 0:
                if positions.ndim == 1:
                    shape_flag = torch.zeros(1, dtype=torch.int64, device="cuda")
                else:
                    shape_flag = torch.tensor([positions.shape[0]], dtype=torch.int64, device="cuda")
                dist.broadcast(shape_flag, src=0)
                pos_gpu = positions.cuda() if not positions.is_cuda else positions
                dist.broadcast(pos_gpu, src=0)
                positions = pos_gpu
            else:
                shape_flag = torch.zeros(1, dtype=torch.int64, device="cuda")
                dist.broadcast(shape_flag, src=0)
                ndim0 = shape_flag.item()
                if ndim0 > 0:
                    n_tokens = input_ids.shape[0]
                    positions = torch.empty(int(ndim0), n_tokens, dtype=torch.int64, device="cuda")
                    dist.broadcast(positions, src=0)
                else:
                    pos_gpu = positions.cuda() if not positions.is_cuda else positions
                    dist.broadcast(pos_gpu, src=0)
                    positions = pos_gpu
        device = input_ids.device
        model = self.model
        embed_fn = model.get_input_embeddings()

        has_deepstack = hasattr(model.visual, 'deepstack_merger_list')
        if has_deepstack:
            visual_dim = model.config.vision.out_hidden_size
        merge_size = model.config.vision.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = getattr(self.config, "audio_token_id", None)

        # One embedding call for the whole step. ``input_ids`` from
        # prepare_mixed_batch is already laid out as
        # ``[prefill chunk tokens... | one token per decode seq]`` in the same
        # sequence order this method iterates, so the batch's text embeddings
        # are exactly ``embed_fn(input_ids)``.
        #
        # This replaces a per-sequence loop that did, every step: a pageable
        # H2D copy of each prefill seq's *entire* token_ids list, one embed_fn
        # launch per prefill seq, and -- the expensive part -- one 1-element
        # H2D copy plus one embed_fn launch per *decode* seq. A steady-state
        # mixed step here carries ~500 decode seqs, so that was ~1000 tiny
        # dispatches and a 500-way torch.cat per step, and those steps were
        # measured at 137ms against ~20ms of real GPU work.
        inputs_embeds = embed_fn(input_ids)

        # Vision rows are scattered in with a single index_copy_ per step.
        # Row indices come from the placeholder positions precomputed on each
        # Sequence, so nothing here reads back from the GPU -- the old path
        # spent four syncs per prefill seq (mask.any(), two .item() calls, and
        # another mask.any() per DeepStack level).
        vis_rows: list[np.ndarray] = []
        vis_embeds: list[torch.Tensor] = []
        ds_rows: list[np.ndarray] = []
        ds_level_embeds: list[list[torch.Tensor]] = []
        row_offset = 0

        for seq_idx, seq in enumerate(prefill_seqs):
            if prefill_chunk_sizes is not None:
                start = seq.num_computed_tokens
                chunk_size = prefill_chunk_sizes[seq_idx]
            else:
                start = 0
                chunk_size = len(seq.token_ids)
            end = start + chunk_size

            info = vis_cache_map[seq_idx] if seq_idx < len(vis_cache_map) else None
            if info is not None:
                vis_out = self._vis_cache[info["cache_idx"]]
                modality = info["modality"]
                if modality == "image":
                    tok_id = image_token_id
                elif modality == "video":
                    tok_id = video_token_id
                else:
                    tok_id = audio_token_id

                if modality != "audio" and has_deepstack:
                    all_vis_embeds = vis_out[:, :visual_dim]
                    ds_cat = vis_out[:, visual_dim:]
                    all_ds_features = list(ds_cat.split(visual_dim, dim=1))
                else:
                    all_vis_embeds = vis_out
                    all_ds_features = []

                if "embed_start" in info:
                    es = info["embed_start"]
                    ec = info["embed_count"]
                    embeds = all_vis_embeds[es:es + ec]
                    ds_features = [d[es:es + ec] for d in all_ds_features]
                else:
                    embeds = all_vis_embeds
                    ds_features = all_ds_features

                positions_np = getattr(seq, "vis_positions", None)
                if (positions_np is None
                        or getattr(seq, "vis_token_id", None) != tok_id):
                    # Sequences that reached this rank through __setstate__ (any
                    # rank but 0) carry no position cache -- recompute it here.
                    ids_np = np.asarray(seq.token_ids, dtype=np.int64)
                    positions_np = np.flatnonzero(ids_np == tok_id).astype(np.int64)
                    seq.vis_positions = positions_np
                    seq.vis_token_id = tok_id

                lo = int(np.searchsorted(positions_np, start))
                hi = int(np.searchsorted(positions_np, end))
                if hi > lo:
                    rows = positions_np[lo:hi] - start + row_offset
                    vis_rows.append(rows)
                    vis_embeds.append(embeds[lo:hi])
                    if has_deepstack and ds_features:
                        ds_rows.append(rows)
                        ds_level_embeds.append(
                            [d[lo:hi] for d in ds_features])

            row_offset += chunk_size

        if vis_rows:
            rows_np = (vis_rows[0] if len(vis_rows) == 1
                       else np.concatenate(vis_rows))
            rows_t = torch.from_numpy(rows_np).pin_memory().to(
                device, non_blocking=True)
            src = (vis_embeds[0] if len(vis_embeds) == 1
                   else torch.cat(vis_embeds, dim=0))
            inputs_embeds.index_copy_(0, rows_t, src.to(inputs_embeds.dtype))

        deepstack_embeds = None
        if has_deepstack and ds_level_embeds:
            num_levels = max(len(d) for d in ds_level_embeds)
            if num_levels > 0:
                total_tokens = inputs_embeds.shape[0]
                hidden_dim = inputs_embeds.shape[-1]
                ds_rows_np = (ds_rows[0] if len(ds_rows) == 1
                              else np.concatenate(ds_rows))
                ds_rows_t = torch.from_numpy(ds_rows_np).pin_memory().to(
                    device, non_blocking=True)
                bufs = getattr(self, "_deepstack_bufs", None)
                if (bufs is None or len(bufs) < num_levels
                        or bufs[0].shape[0] < total_tokens
                        or bufs[0].shape[1] != hidden_dim):
                    # Fallback / grow: shapes beyond what _init_deepstack_buffers
                    # sized for (only reachable if max_num_batched_tokens was
                    # raised after init).
                    bufs = [
                        torch.zeros(total_tokens, hidden_dim, device=device,
                                    dtype=inputs_embeds.dtype)
                        for _ in range(num_levels)
                    ]
                    self._deepstack_bufs = bufs
                deepstack_embeds = []
                for level in range(num_levels):
                    level_dst = bufs[level][:total_tokens]
                    level_dst.zero_()
                    parts = [d[level] for d in ds_level_embeds
                             if level < len(d)]
                    if parts:
                        src = parts[0] if len(parts) == 1 else torch.cat(
                            parts, dim=0)
                        level_dst.index_copy_(
                            0, ds_rows_t, src.to(inputs_embeds.dtype))
                    deepstack_embeds.append(level_dst)

        self._vis_cache = []

        result = self.run_model(input_ids, positions, True,
                                inputs_embeds=inputs_embeds,
                                deepstack_embeds=deepstack_embeds)
        if deepstack_embeds is not None:
            # Zero the rows we just filled, mirroring vLLM's
            # _clear_deepstack_input_embeds. The decode CUDA graphs were captured
            # reading these same persistent buffers, so anything left behind here
            # would be added as a stale residual on the next decode replay.
            for buf in deepstack_embeds:
                buf.zero_()
        reset_context()
        return result

    @torch.inference_mode()
    def _broadcast_visual(self, pv_shape, thw_shape):
        """Broadcast pixel values and run vision encoder on all ranks.

        Rank 0 must set self._mm_pv and self._mm_thw before calling.
        Other ranks receive via NCCL broadcast.
        All ranks participate in model.visual() which uses TP allreduce.
        Returns the vision encoder output tensor.
        """
        device = torch.device("cuda")
        vis_dtype = self.model.visual.patch_embed.proj.weight.dtype
        if self.rank == 0:
            # _dispatch_vision_encoder normally hands this over already on the
            # device, cast to vis_dtype by _stage_pixel_values, so there is
            # nothing left to do. The host path is for callers that build
            # _mm_pv directly, e.g. _warmup_vision_encoder.
            src = self._mm_pv
            if src.is_cuda:
                bpv = src if src.dtype == vis_dtype else src.to(dtype=vis_dtype)
            else:
                if src.dtype != vis_dtype:
                    src = src.to(dtype=vis_dtype)
                if not src.is_pinned():
                    staged = torch.empty_like(src, pin_memory=True)
                    staged.copy_(src)
                    src = staged
                bpv = src.to(device=device, non_blocking=True)
            bthw = (self._mm_thw.clone() if isinstance(self._mm_thw, torch.Tensor)
                    else torch.tensor(self._mm_thw, dtype=torch.long)).to(device)
            self._mm_pv = None
            self._mm_thw = None
        else:
            bpv = torch.empty(pv_shape, dtype=vis_dtype, device=device)
            bthw = torch.empty(thw_shape, dtype=torch.long, device=device)
        if self.world_size > 1:
            dist.broadcast(bpv, src=0)
            dist.broadcast(bthw, src=0)
        vis_out = self.model.visual(bpv, grid_thw=bthw.cpu())
        del bpv
        if not hasattr(self, '_vis_cache'):
            self._vis_cache = []
        self._vis_cache.append(vis_out)
        return vis_out

    @torch.inference_mode()
    def _broadcast_audio(self, feature_shape, lengths_shape):
        """Broadcast Qwen-Omni audio features and run the audio tower."""
        device = torch.device("cuda")
        audio_dtype = next(self.model.audio_tower.parameters()).dtype
        if self.rank == 0:
            feats = self._mm_audio_features.to(device=device, dtype=audio_dtype)
            lengths = self._mm_audio_lengths.to(device=device, dtype=torch.long)
            self._mm_audio_features = None
            self._mm_audio_lengths = None
        else:
            feats = torch.empty(feature_shape, dtype=audio_dtype, device=device)
            lengths = torch.empty(lengths_shape, dtype=torch.long, device=device)
        if self.world_size > 1:
            dist.broadcast(feats, src=0)
            dist.broadcast(lengths, src=0)
        feat_lens, output_lens = (
            self.model.audio_tower._get_feat_extract_output_lengths(lengths)
        )
        audio_out = self.model.audio_tower(
            feats, feature_lens=lengths, aftercnn_lens=feat_lens,
        ).last_hidden_state
        if not hasattr(self, '_vis_cache'):
            self._vis_cache = []
        self._vis_cache.append(audio_out)
        return audio_out, output_lens

    # ------------------------------------------------------------------
    # Whisper cross-attention context helpers
    # ------------------------------------------------------------------

    def _set_cross_attn_context_prefill(self, prefill_seqs, prefill_chunk_sizes,
                                        encoder_seqs):
        """Set cross-attention metadata on the global Context for prefill.

        For sequences with new encoder outputs: builds slot_mapping for
        writing encoder K/V to paged cache, and cu_seqlens for Q (decoder
        tokens) x K (encoder tokens) non-causal attention.

        For decode sequences in a mixed batch, cross-attn context is also
        needed (handled by the mixed path caller).
        """
        ctx = get_context()
        block_size = BLOCK_SIZE

        # Cross-attn slot mapping: for NEW encoder outputs being written to cache
        cross_slot_mapping = []
        for seq in encoder_seqs:
            enc_len = seq.encoder_seq_len
            for p in range(enc_len):
                block_idx = seq.cross_block_table[p // block_size]
                cross_slot_mapping.append(block_idx * block_size + (p % block_size))

        # Q = decoder tokens (for each prefill seq, chunk_size tokens)
        # K = encoder tokens (encoder_seq_len per seq)
        cross_cu_q, cross_cu_k = [0], [0]
        cross_max_sq, cross_max_sk = 0, 0
        cross_max_bt = 0
        for seq, chunk in zip(prefill_seqs, prefill_chunk_sizes):
            cross_cu_q.append(cross_cu_q[-1] + chunk)
            enc_len = seq.encoder_seq_len
            cross_cu_k.append(cross_cu_k[-1] + enc_len)
            cross_max_sq = max(chunk, cross_max_sq)
            cross_max_sk = max(enc_len, cross_max_sk)
            blen = len(seq.cross_block_table)
            if blen > cross_max_bt:
                cross_max_bt = blen

        cross_bt = None
        if cross_max_bt > 0:
            n = len(prefill_seqs)
            bt_arr = np.full((n, cross_max_bt), -1, dtype=np.int32)
            for i, seq in enumerate(prefill_seqs):
                b = seq.cross_block_table
                bt_arr[i, :len(b)] = b
            cross_bt = torch.from_numpy(bt_arr).pin_memory().cuda(non_blocking=True)

        ctx.cross_slot_mapping = (
            torch.tensor(cross_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            if cross_slot_mapping else None
        )
        ctx.cross_cu_seqlens_q = torch.tensor(
            cross_cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.cross_cu_seqlens_k = torch.tensor(
            cross_cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.cross_max_seqlen_q = cross_max_sq
        ctx.cross_max_seqlen_k = cross_max_sk
        ctx.cross_block_tables = cross_bt

    def _init_cross_attn_decode_buffers(self):
        """Persistent cross-attention decode metadata, for CUDA graph capture.

        ``WhisperCrossAttention._forward_decode`` reads ``cross_context_lens``
        and ``cross_block_tables`` off the global Context, so a captured decode
        graph bakes whatever tensors were live at capture time. The old
        per-step ``torch.from_numpy(...).cuda()`` in
        ``_set_cross_attn_context_decode`` therefore cannot be captured -- the
        graph would replay against a freed pointer. These buffers are written
        in place instead, exactly like ``graph_vars``' self-attention metadata.

        Two deliberate choices about the rows past the real batch size, which a
        graph captured at ``graph_bs > n`` still executes:

        * Block ids start at 0, a valid page, not -1. FlashAttention's launcher
          walks ``ceil(max_seqlen_k / page_size)`` columns of every row in the
          table, so -1 would be a real out-of-bounds read.
        * Context lens start at ``max_encoder_tokens`` and are never zeroed, so
          those rows attend to block 0 and produce finite garbage that
          ``_greedy_from_hidden`` slices off. Zeroing instead (what
          self-attention does with ``context_lens[n:graph_bs]``) would hand
          FlashAttention ``seqused_k == 0`` and a NaN row. Captured sizes are
          dense up to 256, so this wastes at most 7 rows of encoder attention.
        """
        if not getattr(self, "_cross_attn_layers", None):
            return
        max_bs = self.max_num_seqs
        dev = f"cuda:{self.rank}"
        blocks = self.cross_blocks_per_seq
        self._cross_cl_buf = torch.full(
            (max_bs,), self.max_encoder_tokens, dtype=torch.int32, device=dev,
        )
        self._cross_bt_buf = torch.zeros(
            (max_bs, blocks), dtype=torch.int32, device=dev,
        )
        # Pinned staging + numpy views over the same memory: the per-step fill
        # is a Python loop over sequences, and writing it through numpy keeps
        # that loop off the GPU and off the allocator.
        self._cross_cl_host = torch.empty(
            (max_bs,), dtype=torch.int32, device="cpu",
        ).pin_memory()
        self._cross_bt_host = torch.zeros(
            (max_bs, blocks), dtype=torch.int32, device="cpu",
        ).pin_memory()
        self._cross_cl_np = self._cross_cl_host.numpy()
        self._cross_bt_np = self._cross_bt_host.numpy()

    def _set_cross_attn_context_for_capture(self, bs):
        """Point the Context's cross-attn decode fields at the graph buffers.

        ``set_context`` builds a fresh Context, so capture has to re-apply
        these after every call, the same way ``_run_decode_greedy_eager`` calls
        ``_apply_pending_cross_ctx``.
        """
        if not getattr(self, "_cross_attn_layers", None):
            return
        ctx = get_context()
        ctx.cross_slot_mapping = None
        ctx.cross_context_lens = self._cross_cl_buf[:bs]
        ctx.cross_block_tables = self._cross_bt_buf[:bs]
        ctx.cross_max_context_len = self.max_encoder_tokens

    def _set_cross_attn_context_decode(self, decode_seqs):
        """Set cross-attention metadata on the global Context for decode.

        Each decode token attends to the full encoder output via paged cache.
        Also stores the tensors as instance state so they can be applied to
        any Context created later (e.g., by the greedy eager decode path).

        The metadata goes into the persistent buffers from
        ``_init_cross_attn_decode_buffers`` so that a captured decode graph
        replays against stable pointers. Encoder K/V is written once at
        prefill and never moves, so callers only need to invoke this when the
        running set changes, not every step.
        """
        n = len(decode_seqs)
        cl_buf = getattr(self, "_cross_cl_buf", None)
        if cl_buf is None:
            # No cross-attention layers (or buffers not built): nothing to do.
            return
        bt_buf = self._cross_bt_buf
        cl_np, bt_np = self._cross_cl_np, self._cross_bt_np

        max_bt = 0
        for i, seq in enumerate(decode_seqs):
            cl_np[i] = seq.encoder_seq_len
            b = seq.cross_block_table
            lb = len(b)
            bt_np[i, :lb] = b
            if lb > max_bt:
                max_bt = lb

        cl_buf[:n].copy_(self._cross_cl_host[:n], non_blocking=True)
        if max_bt:
            bt_buf[:n, :max_bt].copy_(
                self._cross_bt_host[:n, :max_bt], non_blocking=True,
            )

        self._pending_cross_ctx = {
            "cross_context_lens": cl_buf[:n],
            "cross_block_tables": bt_buf[:n],
            "cross_max_context_len": self.max_encoder_tokens,
        }
        self._apply_pending_cross_ctx()

    def _apply_pending_cross_ctx(self):
        """Apply pending cross-attention context to the current global Context."""
        pending = getattr(self, '_pending_cross_ctx', None)
        if pending is None:
            return
        ctx = get_context()
        ctx.cross_slot_mapping = None
        ctx.cross_context_lens = pending["cross_context_lens"]
        ctx.cross_block_tables = pending["cross_block_tables"]
        ctx.cross_max_context_len = pending["cross_max_context_len"]

    def _clear_pending_cross_ctx(self):
        self._pending_cross_ctx = None

    def _set_cross_attn_context_mixed(self, prefill_seqs, prefill_chunk_sizes,
                                      decode_seqs, encoder_seqs):
        """Set cross-attention metadata for mixed prefill+decode batch.

        Prefill seqs: Q = chunked decoder tokens, K = encoder tokens (from paged cache).
        Decode seqs: Q = 1 token each, K = encoder tokens (from paged cache).
        Both use non-causal attention via the prefill kernel with block tables.
        """
        ctx = get_context()
        block_size = BLOCK_SIZE

        # Cross-attn slot mapping for NEW encoder outputs
        cross_slot_mapping = []
        for seq in encoder_seqs:
            enc_len = seq.encoder_seq_len
            for p in range(enc_len):
                block_idx = seq.cross_block_table[p // block_size]
                cross_slot_mapping.append(block_idx * block_size + (p % block_size))

        # Build unified cu_seqlens: prefill seqs first, then decode seqs
        all_seqs = list(prefill_seqs) + list(decode_seqs)
        cross_cu_q, cross_cu_k = [0], [0]
        cross_max_sq, cross_max_sk = 0, 0
        cross_max_bt = 0

        for idx, seq in enumerate(all_seqs):
            if idx < len(prefill_seqs):
                q_tokens = prefill_chunk_sizes[idx]
            else:
                q_tokens = 1
            enc_len = seq.encoder_seq_len
            cross_cu_q.append(cross_cu_q[-1] + q_tokens)
            cross_cu_k.append(cross_cu_k[-1] + enc_len)
            cross_max_sq = max(q_tokens, cross_max_sq)
            cross_max_sk = max(enc_len, cross_max_sk)
            blen = len(seq.cross_block_table)
            if blen > cross_max_bt:
                cross_max_bt = blen

        cross_bt = None
        if cross_max_bt > 0:
            n = len(all_seqs)
            bt_arr = np.full((n, cross_max_bt), -1, dtype=np.int32)
            for i, seq in enumerate(all_seqs):
                b = seq.cross_block_table
                bt_arr[i, :len(b)] = b
            cross_bt = torch.from_numpy(bt_arr).pin_memory().cuda(non_blocking=True)

        ctx.cross_slot_mapping = (
            torch.tensor(cross_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            if cross_slot_mapping else None
        )
        ctx.cross_cu_seqlens_q = torch.tensor(
            cross_cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.cross_cu_seqlens_k = torch.tensor(
            cross_cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.cross_max_seqlen_q = cross_max_sq
        ctx.cross_max_seqlen_k = cross_max_sk
        ctx.cross_block_tables = cross_bt

    def _compile_model(self):
        """Apply torch.compile with FastKernelsBackend (mirrors vLLM).

        The backend:
        1. Splits the graph at attention custom-op boundaries
        2. Compiles each subgraph with symbolic shapes (one compile per
           unique subgraph structure, not per batch size)
        3. Drops all Dynamo guards so no re-tracing occurs

        After this, the model works for any batch size.  The subsequent
        ``capture_cudagraph`` call records the compiled kernels into
        per-batch-size CUDA graphs for decode replay.

        For VL models, only the inner Qwen3Model is compiled (not the
        outer Qwen3VLForConditionalGeneration).  This keeps embed_tokens
        and lm_head outside the compiled boundary, matching vLLM's
        architecture where @support_torch_compile is applied only to the
        inner LLM model.  The engine always computes inputs_embeds
        outside the compiled graph and passes it in, so the compiled
        graph only ever traces the inputs_embeds branch.
        """
        from .compilation import compile_model, configure_post_grad_passes

        configure_post_grad_passes(model_dtype=self.dtype)
        # Mamba2 owns a dedicated decode-cudagraph path, so keep the
        # compile stack's generic cudagraph wrapper off for it.
        cudagraph_enabled = not self.is_mamba2
        if self.is_whisper:
            # Compile the decoder only, matching vLLM, which puts
            # ``@support_torch_compile`` on ``WhisperDecoder`` and leaves the
            # encoder and the outer wrapper eager. The encoder runs once per
            # request from ``get_multimodal_embeddings`` (not through forward),
            # so it has nothing to gain and would only lengthen the traced
            # graph. ``set_compiled_decoder`` keeps the uncompiled module
            # reachable for the prefill call.
            self.model.set_compiled_decoder(
                compile_model(
                    self.model.decoder,
                    cudagraph_enabled=cudagraph_enabled,
                )
            )
        elif self.is_qwen_vl:
            # Save the uncompiled inner model for multimodal prefill
            # (which needs deepstack_embeds that the compiled graph
            # doesn't trace).
            self._eager_inner_model = self.model.model
            self.model.model = compile_model(
                self.model.model,
                cudagraph_enabled=cudagraph_enabled,
            )
        else:
            self._eager_model = self.model
            self.model = compile_model(
                self.model,
                cudagraph_enabled=cudagraph_enabled,
            )
        self._compiled = True
        self._mark_dynamic_done = False

    @torch.inference_mode()
    def _supports_cudagraphs(self) -> bool:
        """Whether a decode step of this model can be recorded into a CUDA graph.

        DSA models (DeepSeek-V3.2, GLM-5.2) need the indexer's paged-logits
        schedule to be built OUTSIDE the graph. With
        ``get_paged_mqa_logits_metadata`` called from inside the captured region,
        capture succeeds but the first REPLAY dies in
        ``deep_gemm.fp8_fp4_paged_mqa_logits`` -- ``cudaErrorInvalidAddressSpace``
        on GLM-5.2-NVFP4, ``cudaErrorIllegalAddress`` on GLM-5.2-FP8, and nothing
        at all under compute-sanitizer. The kernel itself IS capturable: fed a
        schedule built beforehand it captures and replays fine standalone. vLLM
        builds the same tensor in its metadata builder, i.e. also outside the
        graph.

        So the engine keeps one persistent ``_indexer_sched`` buffer and refreshes
        it per step via ``_refresh_indexer_schedule`` at every decode site,
        including immediately before ``replay()``. This returns False only if that
        buffer is missing, which would mean capture is unsafe.
        """
        if not getattr(self, "is_deepseek_mla", False):
            return True
        from ..tasks.baseline.L2.sparse_attn_indexer import SparseAttnIndexer

        return not any(
            isinstance(m, SparseAttnIndexer) for m in self.model.modules()
        ) or self._indexer_sched is not None

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Trace, warm up, and capture the decode CUDA graphs.

        ``inference_mode`` on the whole thing: this is where the compiled model
        is traced for the first time, and tracing with grad enabled makes
        AOTAutograd build a training graph wrapped in an ``autograd.Function``.
        Every real forward afterwards runs under inference mode, so that graph
        then rejects its own inputs ("Inference tensors cannot be saved for
        backward") the first time a mixed prefill+decode batch takes the eager
        compiled path.
        """
        import gc
        from contextlib import nullcontext
        max_bs = self.max_num_seqs
        max_num_blocks = (self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        if self.is_qwen_vl:
            positions = torch.zeros(3, max_bs + 1, dtype=torch.int64)
        else:
            positions = torch.zeros(max_bs, dtype=torch.int64)
        sm_torch_dtype = torch.int64 if self.is_deepseek_mla else torch.int32
        slot_mapping = torch.full((max_bs,), -1, dtype=sm_torch_dtype)
        # One cached token per dummy request, not zero: vLLM's uniform-decode
        # dummy run sets ``seq_lens = max_query_len`` (== 1) and only zeroes the
        # padded tail (``gpu_model_runner._dummy_run``).  A zero context length
        # is not a state any real decode step reaches, so capturing against it
        # warms a shape the kernels never see again.
        context_lens = torch.ones(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # Persistent arange buffer for per-token request id mapping during
        # pure-decode (token i -> sequence i).  Captured into decode CUDA
        # graphs and reused by ``prepare_decode``; kept on-device so the
        # sparse MLA ``convert_indices`` kernel never falls back to an
        # all-zeros buffer (see ``_forward_sparse_bf16``).
        decode_req_id = torch.arange(max_bs, dtype=torch.int32).cuda()
        self._decode_req_id_buf = decode_req_id

        # Match vLLM's default ``cudagraph_capture_sizes`` shape:
        # [1, 2, 4, 8, 16, 24, ..., 256, 272, ..., max_capture].
        env_cap = os.environ.get("FASTKERNELS_MAX_CUDAGRAPH_BS")
        if env_cap:
            max_capture_limit = int(env_cap)
        else:
            # Capture up to ``max_num_seqs`` when the scheduler can actually form
            # batches beyond 512, because anything past the largest captured size
            # falls through ``run_decode_greedy_fast`` to the EAGER path, and eager
            # decode is catastrophically slower for a DSA model.
            #
            # Measured on GLM-5.2-NVFP4, mixed at 1000 sequences (max_num_seqs
            # 1024): the decode batch histogram was
            # 512-640:83  640-768:97  768-896:79  896-1024:44, i.e. 303 of 1022
            # steps -- and ~59% of all decode tokens, since these are the *large*
            # batches -- ran eager. Raising the limit to 1024 took the scenario
            # from 4,210 to 7,942 tok/s. Startup cost is 83 graphs instead of 63
            # (96s vs 75s of capture); configs that cannot exceed 512 concurrent
            # sequences are unaffected.
            # Scoped to DSA models along with prefill-priority, and to MoE models
            # generally. The eager-above-the-largest-captured-size problem is
            # general, but it bites MoE hardest: the trtllm-gen fused MoE call
            # costs 412-676 us of CPU dispatch (autotuner lookup + routing config
            # + cooperative launch), 2.3-3.7x the Triton path's ~182 us, so at 32
            # layers an eager decode step burns ~21 ms of host time that a graph
            # replay pays once. Measured on Mixtral-8x7B tp=2, mixed at 1000
            # sequences: raising the cap 512 -> 1024 cut the profiled generate
            # 14.4 s -> 12.1 s and lifted GPU-busy 77.7% -> 92.9% (idle 3.2 s ->
            # 0.9 s) with the kernel sum unchanged -- pure idle removal from the
            # bs>512 decode steps that had run eager. gpt-oss (also this-path MoE)
            # already got 1024; ``self.is_moe`` folds Mixtral in. Dense models
            # below 512 concurrent sequences are unaffected (~0.3 GiB and ~21 s of
            # extra capture only when max_num_seqs actually exceeds 512).
            # ``_has_indexer_layers`` is set in allocate_kv_cache, before capture.
            max_capture_limit = (
                1024
                if (self.is_gpt_oss or self.is_gemma4
                    or (self.max_num_seqs > 512
                        and (self.is_moe
                             or getattr(self, "_has_indexer_layers", False))))
                else 512
            )
        max_capture = min(max_bs, max_capture_limit)
        self.graph_bs_list = [i for i in [1, 2, 4] if i <= max_capture]
        if max_capture >= 8:
            self.graph_bs_list += list(range(8, min(max_capture + 1, 256), 8))
        if max_capture >= 256:
            self.graph_bs_list += list(range(256, max_capture + 1, 16))
        if self.graph_bs_list[-1] != max_capture:
            self.graph_bs_list.append(max_capture)
        self.graphs = {}
        self.graph_pool = None

        outputs = torch.zeros(max_bs, self.config.hidden_size)

        # For VL models, allocate an inputs_embeds buffer so the compiled
        # inner model is always traced with inputs_embeds (matching vLLM).
        # deepstack_embeds is NOT passed here — it is only needed during
        # multimodal prefill, which uses the eager model.  Keeping deepstack
        # out of the compiled graph avoids unnecessary zero-tensor additions
        # on every decode step.
        vl_inputs_embeds = None
        vl_embed_fn = None
        if self.is_qwen_vl:
            hidden_size = self.config.hidden_size
            vl_inputs_embeds = torch.zeros(max_bs, hidden_size,
                                           dtype=self.dtype)
            vl_embed_fn = self.model.get_input_embeddings()

        lm_head = self.model.lm_head
        vocab_per_rank = lm_head.per_partition
        lm_logits = torch.zeros(max_bs, vocab_per_rank)
        lm_max_vals = torch.zeros(max_bs)
        lm_max_idxs = torch.zeros(max_bs, dtype=torch.int64)

        ar_ctx = self.custom_ar.capture() if self.custom_ar is not None else nullcontext()
        _graph_list = list(reversed(self.graph_bs_list))

        # Single warmup at the largest batch size to trigger all Triton/CUDA
        # kernel JIT compilation. Subsequent captures reuse compiled kernels.
        largest_bs = _graph_list[0]
        set_context(
            False, slot_mapping=slot_mapping[:largest_bs],
            context_lens=context_lens[:largest_bs],
            block_tables=block_tables[:largest_bs],
            max_context_len=self.max_model_len,
            req_id_per_token=decode_req_id[:largest_bs],
        )
        self._refresh_indexer_schedule(context_lens[:largest_bs])
        self._refresh_fa3_decode_schedule(context_lens[:largest_bs])
        self._set_cross_attn_context_for_capture(largest_bs)
        # Mark batch dim as dynamic BEFORE the first compile-triggering forward
        # so Dynamo / Inductor produce a single symbolic-shape compiled graph
        # that works for every batch size we will subsequently capture, instead
        # of hard-coding the warmup batch size into the compiled subgraphs
        # (which under ``skip_all_guards_unsafe`` would silently get reused at
        # the wrong size and trip Inductor's ``assert_size_stride`` checks).
        warmup_ids = input_ids[:largest_bs]
        warmup_pos = (positions[:, :largest_bs]
                      if self.is_qwen_vl else positions[:largest_bs])
        warmup_ie = (vl_inputs_embeds[:largest_bs]
                     if vl_inputs_embeds is not None else None)
        # DeepStack buffers must be part of the traced graph, so the same graph
        # serves decode (buffers zeroed -> no-op adds) and multimodal prefill
        # (buffers filled). See run_model.
        _ds_all = getattr(self, "_deepstack_bufs", None)

        def _ds(n):
            return [b[:n] for b in _ds_all] if _ds_all else None

        warmup_ds = _ds(largest_bs)
        if self._compiled and not self._mark_dynamic_done:
            torch._dynamo.mark_dynamic(warmup_ids, 0)
            if self.is_qwen_vl:
                torch._dynamo.mark_dynamic(warmup_pos, 1)
            else:
                torch._dynamo.mark_dynamic(warmup_pos, 0)
            if warmup_ie is not None:
                torch._dynamo.mark_dynamic(warmup_ie, 0)
            if warmup_ds is not None:
                # Same token dim as inputs_embeds. Without marking these, the
                # slices come in as a *static* size and the solver reports a
                # ConstraintViolationError against the dynamic inputs_embeds dim.
                for _t in warmup_ds:
                    torch._dynamo.mark_dynamic(_t, 0)
            self._mark_dynamic_done = True

        if self.is_qwen_vl:
            warmup_ie.copy_(vl_embed_fn(warmup_ids))
            _kw = {"inputs_embeds": warmup_ie}
            if warmup_ds is not None:
                _kw["deepstack_embeds"] = warmup_ds
            outputs[:largest_bs] = self.model(
                warmup_ids, warmup_pos, **_kw,
            )
        else:
            outputs[:largest_bs] = self.model(warmup_ids, warmup_pos)
        lm_logits[:largest_bs] = lm_head.linear_op(
            outputs[:largest_bs], lm_head.embedding_op.emb.weight)
        lm_max_vals[:largest_bs], lm_max_idxs[:largest_bs] = \
            lm_logits[:largest_bs].max(dim=-1)
        reset_context()
        torch.cuda.synchronize()

        def _eager_decode(bs: int) -> tuple:
            """One decode-shaped forward at ``bs``. Used to prime cubins /
            workspaces outside CUDA-graph capture, and as the per-size warmup
            immediately before ``torch.cuda.graph``.
            """
            set_context(
                False, slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                max_context_len=self.max_model_len,
                req_id_per_token=decode_req_id[:bs],
            )
            self._refresh_indexer_schedule(context_lens[:bs])
            self._refresh_fa3_decode_schedule(context_lens[:bs])
            self._set_cross_attn_context_for_capture(bs)
            ids_slice = input_ids[:bs]
            pos_slice = (positions[:, :bs] if self.is_qwen_vl
                         else positions[:bs])
            ie_slice = (vl_inputs_embeds[:bs]
                        if vl_inputs_embeds is not None else None)
            if ie_slice is not None:
                ie_slice.copy_(vl_embed_fn(ids_slice))
                outputs[:bs] = self.model(
                    ids_slice, pos_slice,
                    inputs_embeds=ie_slice,
                    **({"deepstack_embeds": _ds(bs)} if _ds_all else {}),
                )
            else:
                outputs[:bs] = self.model(ids_slice, pos_slice)
            lm_logits[:bs] = lm_head.linear_op(
                outputs[:bs], lm_head.embedding_op.emb.weight)
            lm_max_vals[:bs], lm_max_idxs[:bs] = lm_logits[:bs].max(dim=-1)
            return ids_slice, pos_slice, ie_slice

        # GLM-5.2 only: prime remaining capture sizes *before* custom-AR /
        # CUDA-graph so FlashInfer cubins and workspaces for smaller CTA tiles
        # (bs=128 PerCta128, etc.) are not first-touched inside the graph pool
        # -- that path has hung with no traceback on a cold cubin cache.
        _is_glm = getattr(self.config, "model_type", "") == "glm_moe_dsa"
        if _is_glm:
            if self.rank == 0 and len(_graph_list) > 1:
                print(f"    Eager-priming {len(_graph_list) - 1} remaining "
                      f"capture sizes (outside CUDA graph)", flush=True)
            for bs in _graph_list[1:]:
                _eager_decode(bs)
                torch.cuda.synchronize()
            reset_context()

        # Freeze GC during capture to avoid Python GC stalls (matches vllm).
        gc.collect()
        gc.freeze()

        with ar_ctx:
            for _gi, bs in enumerate(_graph_list):
                if self.rank == 0 and (
                    _is_glm
                    or _gi % max(1, len(_graph_list) // 5) == 0
                    or _gi == len(_graph_list) - 1
                ):
                    print(f"    CUDA graph {_gi+1}/{len(_graph_list)} (bs={bs})",
                          flush=True)
                graph = torch.cuda.CUDAGraph()
                ids_slice, pos_slice, ie_slice = _eager_decode(bs)

                if self._compiled and not self._mark_dynamic_done:
                    torch._dynamo.mark_dynamic(ids_slice, 0)
                    if self.is_qwen_vl:
                        torch._dynamo.mark_dynamic(pos_slice, 1)
                    else:
                        torch._dynamo.mark_dynamic(pos_slice, 0)
                    if ie_slice is not None:
                        torch._dynamo.mark_dynamic(ie_slice, 0)
                    self._mark_dynamic_done = True

                with torch.cuda.graph(graph, self.graph_pool):
                    if ie_slice is not None:
                        ie_slice.copy_(vl_embed_fn(ids_slice))
                        outputs[:bs] = self.model(
                            ids_slice, pos_slice,
                            inputs_embeds=ie_slice,
                            **({"deepstack_embeds": _ds(bs)} if _ds_all else {}),
                        )
                    else:
                        outputs[:bs] = self.model(ids_slice, pos_slice)
                    lm_logits[:bs] = lm_head.linear_op(
                        outputs[:bs], lm_head.embedding_op.emb.weight)
                    lm_max_vals[:bs], lm_max_idxs[:bs] = lm_logits[:bs].max(dim=-1)

                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

        gc.unfreeze()
        gc.collect()

        self.graph_vars = dict(
            input_ids=input_ids, positions=positions,
            slot_mapping=slot_mapping, context_lens=context_lens,
            block_tables=block_tables, outputs=outputs,
            lm_logits=lm_logits, lm_max_vals=lm_max_vals,
            lm_max_idxs=lm_max_idxs,
        )
        # Pre-compute lookup table: _graph_bs_for_n[n] = smallest graph_bs >= n
        self._graph_bs_for_n = [0] * (max_bs + 1)
        for n in range(max_bs + 1):
            for x in self.graph_bs_list:
                if x >= n:
                    self._graph_bs_for_n[n] = x
                    break
            else:
                self._graph_bs_for_n[n] = max_bs


# ---------------------------------------------------------------------------
# LlamaEngine — only runs on rank 0
# ---------------------------------------------------------------------------
class LlamaEngine:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        seed: int = 42,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_layers: int | None = None,
        kv_cache_dtype: str | None = None,
    ):
        self.model_name = model_name
        self.seed = seed
        # Gemma4 used to be pinned to 2048 here. vLLM schedules it at the
        # B200 default of 16384 (``Chunked prefill is enabled with
        # max_num_batched_tokens=16384``), and 2048 costs 8x the prefill
        # chunks: 41,280 kernel launches for a 256x512 prefill against vLLM's
        # 6,651, and 28% of prefill wall time spent host-bound rather than on
        # the GPU. Measured on the mixed scenario, lifting the cap takes
        # prefill from 6.00s to 4.57s.
        _env_mnbt = os.environ.get("FASTKERNELS_MAX_NUM_BATCHED_TOKENS")
        if _env_mnbt:
            max_num_batched_tokens = int(_env_mnbt)
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else _DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        self._set_seeds(seed)

        # Unique shared memory name to avoid collisions
        shm_name = f"sllama_{uuid.uuid4().hex[:8]}"

        mr_kwargs = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_layers=max_layers,
            kv_cache_dtype=kv_cache_dtype,
        )

        # Launch non-rank-0 workers
        self.workers = []
        self.events = []
        # Register cleanup BEFORE spawning workers so that if construction fails
        # partway (e.g. rank 0 OOMs after the workers are already spawned) the
        # orphaned workers are still torn down instead of leaking their GPU
        # memory and hanging the process at exit.
        atexit.register(self._cleanup)
        ctx = mp.get_context("spawn")
        try:
            for i in range(1, tensor_parallel_size):
                event = ctx.Event()
                p = ctx.Process(
                    target=ModelRunner,
                    args=(model_name, i, tensor_parallel_size, dtype,
                          enforce_eager, event, shm_name),
                    kwargs=mr_kwargs,
                )
                p.start()
                self.workers.append(p)
                self.events.append(event)

            # Rank 0 model runner (events is a list for rank 0)
            self.model_runner = ModelRunner(
                model_name, 0, tensor_parallel_size, dtype,
                enforce_eager, self.events, shm_name,
                **mr_kwargs,
            )
        except BaseException:
            # Free workers/GPU immediately so the caller can recover (and the
            # next engine is not starved of memory), then re-raise.
            self._cleanup()
            raise
        self.is_mamba = self.model_runner.is_mamba
        self.is_kimi_linear = self.model_runner.is_kimi_linear
        self.is_qwen3_next = self.model_runner.is_qwen3_next
        if self.is_mamba:
            # Mamba uses MambaStateManager (slot-based) not paged KV blocks.
            self.block_manager = BlockManager(0)
        else:
            self.block_manager = BlockManager(self.model_runner.num_blocks)
        if hasattr(self.model_runner, '_cross_free_block_ids_init'):
            n = self.model_runner._cross_free_block_ids_init
            self.block_manager.cross_free_block_ids = deque(range(n))
            self.block_manager._num_cross_blocks = n
        if hasattr(self.model_runner, 'cross_blocks_per_seq'):
            self.cross_blocks_per_seq = self.model_runner.cross_blocks_per_seq
        self.max_num_seqs = self.model_runner.max_num_seqs
        self.max_num_batched_tokens = self.model_runner.max_num_batched_tokens
        # Mirrored for the scheduler's admission gate, which caps a request's
        # reservation at the context window the way vLLM's does.
        self.max_model_len = self.model_runner.max_model_len
        print(f"  Scheduling: max_num_seqs={self.max_num_seqs}, "
              f"max_num_batched_tokens={self.max_num_batched_tokens}")

        self.tokenizer = _load_tokenizer(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.is_qwen_vl = self.model_runner.is_qwen_vl
        self.is_qwen3_vl = self.model_runner.is_qwen3_vl
        self.is_whisper = self.model_runner.is_whisper
        self.processor = None
        if self.is_qwen_vl:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True,
            )

        self.encoder_cache: dict[int, tuple] = {}

        if self.is_whisper:
            from transformers import WhisperProcessor
            self.whisper_processor = WhisperProcessor.from_pretrained(model_name)

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _cleanup(self):
        # Idempotent: safe to call explicitly between scenarios and again at
        # interpreter exit (via atexit). Tears the model down and, crucially,
        # guarantees every spawned worker is dead so Python's built-in
        # multiprocessing atexit handler (which joins non-daemon children with
        # NO timeout) cannot hang the whole process at shutdown.
        mr = getattr(self, "model_runner", None)
        signalled = False
        if mr is not None:
            try:
                mr.call("exit")
                signalled = True
            except Exception:
                pass
            self.model_runner = None
        for p in getattr(self, "workers", []):
            try:
                # Only wait for a graceful exit if we actually signalled one;
                # otherwise (e.g. failed construction) go straight to kill.
                if signalled:
                    p.join(timeout=10)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
            except Exception:
                pass
        self.workers = []
        # Drop the module-level registry that pins this model's attention/MoE
        # layers (and the KV-cache slices they reference); otherwise the model
        # and its KV cache cannot be garbage-collected between engines.
        try:
            clear_no_compile_layers()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sample_greedy(self, logits):
        return logits.argmax(dim=-1).tolist()

    def _sample(self, logits, params):
        if logits is None:
            return []
        if params.temperature == 0.0:
            return self._sample_greedy(logits)
        logits = logits / params.temperature
        if params.top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(torch.softmax(sl, -1), -1)
            mask = cp - torch.softmax(sl, -1) >= params.top_p
            sl[mask] = float("-inf")
            logits = logits.scatter(1, si, sl)
        probs = torch.softmax(logits, -1)
        return torch.multinomial(probs, 1).squeeze(-1).tolist()

    def _preprocess_multimodal(self, prompt, images=None, videos=None, audios=None):
        """Preprocess a multimodal prompt with image/video/audio inputs.

        A video item may be either a bare frame sequence (list[PIL.Image] or a
        (T,H,W,3) ndarray) or, matching vLLM's native video item, a
        ``(frames, metadata)`` pair. Pass the pair whenever metadata is
        available: without it the HF video processor re-samples the frames
        against a *default* fps, which for Qwen3-VL on the MMVU scenario keeps
        only 4 of 32 supplied frames (grid t=2 instead of t=16) and so produces
        ~490 video tokens where vLLM produces ~3520. That is a 7x difference in
        the work being compared, and it also loses the per-frame timestamps
        Qwen3-VL derives from ``frames_indices``/``fps``.

        Returns (token_ids, pixel_values, image_grid_thw, video_pixel_values,
                 video_grid_thw, video_second_per_grid, input_audio_features,
                 audio_feature_lengths).
        """
        messages = [{"role": "user", "content": []}]
        if audios:
            for audio in audios:
                messages[0]["content"].append({"type": "audio", "audio": audio})
        if images:
            for img in images:
                messages[0]["content"].append({"type": "image", "image": img})

        video_frames = None
        video_metadata = None
        do_sample_frames = None
        if videos:
            video_frames = []
            video_metadata = []
            for vid in videos:
                meta = None
                if (isinstance(vid, tuple) and len(vid) == 2
                        and isinstance(vid[1], dict)):
                    vid, meta = vid
                video_frames.append(vid)
                if meta is not None:
                    # HF's VideoMetadata carries no do_sample_frames field; it
                    # travels as a processor kwarg instead (as in vLLM).
                    if do_sample_frames is None:
                        do_sample_frames = bool(
                            meta.get("do_sample_frames", False))
                    meta = {k: v for k, v in meta.items()
                            if k != "do_sample_frames"}
                video_metadata.append(meta)
                messages[0]["content"].append({"type": "video", "video": vid})
            if all(m is None for m in video_metadata):
                video_metadata = None

        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        processor_kwargs = dict(
            text=[text],
            images=images,
            videos=video_frames,
            return_tensors="pt",
            padding=True,
        )
        if video_metadata is not None:
            processor_kwargs["video_metadata"] = video_metadata
            if do_sample_frames is not None:
                processor_kwargs["do_sample_frames"] = do_sample_frames
        if audios is not None:
            processor_kwargs["audio"] = audios

        # Image-only fast path: run the image processor, tokenize the *unexpanded*
        # chat text, and splice the placeholder token ids ourselves.
        #
        # The full processor call expands each placeholder inside the prompt
        # string and re-tokenizes the result, so its cost grows with the number
        # of vision tokens: measured 13.4ms/item on VisionArena against 6.4ms for
        # the spliced path, i.e. over half of preprocessing. That matters because
        # preprocessing paces admission -- 42 of 66 admission steps on the
        # Qwen3-VL image scenario ended starved, waiting on the producer -- and
        # vLLM does the same thing at token level (its PromptUpdate machinery)
        # rather than re-tokenizing.
        #
        # Verified to produce byte-identical input_ids on 24/24 VisionArena
        # images. Restricted to the image-only case: video and audio carry
        # per-item extras (Qwen3-VL interleaves per-frame timestamp text) whose
        # expansion is not a plain placeholder repeat.
        if (video_frames is None and not audios and images
                and getattr(self, "_mm_splice_ok", True)):
            try:
                ip = self.processor.image_processor(
                    images=images, return_tensors="pt")
                grid = ip["image_grid_thw"]
                merge = self.model_runner.model.config.vision.spatial_merge_size
                img_tok = self.model_runner.config.image_token_id
                counts = [int(g.prod() // (merge ** 2)) for g in grid]
                base = self.tokenizer(text, return_tensors=None)["input_ids"]
                ids = []
                k = 0
                for tid in base:
                    if tid == img_tok and k < len(counts):
                        ids.extend([img_tok] * counts[k])
                        k += 1
                    else:
                        ids.append(tid)
                if k == len(counts):
                    return (ids, ip["pixel_values"], grid,
                            None, None, None, None, None)
                self._mm_splice_ok = False
            except Exception:
                # Any processor whose expansion is not a placeholder repeat
                # falls back permanently rather than per-item.
                self._mm_splice_ok = False

        inputs = self.processor(**processor_kwargs)
        token_ids = inputs["input_ids"][0].tolist()
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        video_pixel_values = inputs.get("pixel_values_videos", None)
        video_grid_thw = inputs.get("video_grid_thw", None)
        video_second_per_grid = inputs.get("video_second_per_grid", None)
        input_audio_features = inputs.get("input_audio_features", None)
        if input_audio_features is None and inputs.get("input_features", None) is not None:
            input_audio_features = inputs["input_features"]
            feature_mask = inputs.get("feature_attention_mask", None)
            if feature_mask is not None:
                input_audio_features = input_audio_features.permute(0, 2, 1)[
                    feature_mask.bool()
                ].permute(1, 0)
        audio_feature_lengths = inputs.get("audio_feature_lengths", None)
        if audio_feature_lengths is None and inputs.get("feature_attention_mask", None) is not None:
            audio_feature_lengths = inputs["feature_attention_mask"].sum(-1)

        return (token_ids, pixel_values, image_grid_thw,
                video_pixel_values, video_grid_thw, video_second_per_grid,
                input_audio_features, audio_feature_lengths)

    def _dispatch_vision_encoder(self, seqs, chunk_sizes=None):
        """Dispatch vision encoder to all TP ranks and build vis_cache_map.

        For each sequence with images/videos, broadcasts pixel values via NCCL
        and runs the vision encoder on all ranks (required for TP allreduce).

        Returns (vis_cache_map, cache_count) where vis_cache_map is a list
        with one entry per seq: None for text-only seqs, or a dict with
        cache_idx/modality for vision seqs.
        """
        mr = self.model_runner
        model = mr.model
        merge_size = model.config.vision.spatial_merge_size
        vis_dtype = model.visual.patch_embed.proj.weight.dtype

        def _stage_pixel_values(pv_list):
            """Concatenate + cast + upload the step's pixel values in one pass.

            The HF processor hands back float32, and the encoder wants bf16 on
            the GPU. Doing that as cat -> .to(dtype) -> pinned copy walks the
            data three times on the host: for the MMVU video scenario that is
            ~83MB of float32 per clip and ~3 clips per step, so ~390GB of host
            memcpy over the run -- measured as ~15ms per clip of pure CPU time
            on top of a 44ms encoder forward. Copying each sequence straight
            into its slice of a persistent pinned bf16 buffer casts and packs in
            a single pass, and reuses the (expensive) cudaHostAlloc across steps.

            Result is bit-identical: copy_ applies the same float32->bf16
            rounding the old .to(dtype) did.

            Two buffers, each with an event recorded after its upload, so a
            buffer is never refilled while its own async H2D is still in flight.
            In practice the wait is free -- per-step sampling reads token ids
            back, so the host cannot run more than a step ahead of the GPU --
            but the events make that independent of the sampling path.
            """
            rows = sum(t.shape[0] for t in pv_list)
            feat = pv_list[0].shape[1]
            bufs = getattr(self, "_mm_stage_bufs", None)
            if bufs is None:
                bufs = self._mm_stage_bufs = [None, None]
                self._mm_stage_evts = [None, None]
                self._mm_stage_idx = 0
            idx = self._mm_stage_idx
            self._mm_stage_idx = 1 - idx
            evt = self._mm_stage_evts[idx]
            if evt is not None:
                evt.synchronize()
            buf = bufs[idx]
            if (buf is None or buf.shape[0] < rows or buf.shape[1] != feat
                    or buf.dtype != vis_dtype):
                buf = torch.empty((rows, feat), dtype=vis_dtype,
                                  pin_memory=True)
                bufs[idx] = buf
            staged = buf[:rows]
            off = 0
            for t in pv_list:
                n = t.shape[0]
                staged[off:off + n].copy_(t)
                off += n
            gpu = staged.to(device=f"cuda:{mr.rank}", non_blocking=True)
            evt = torch.cuda.Event()
            evt.record()
            self._mm_stage_evts[idx] = evt
            return gpu

        mr._vis_cache = []
        vis_cache_map = []
        cache_idx = 0
        seq_entries = []

        # Reuse encoder output across a sequence's chunked prefill steps, as
        # vLLM's encoder_cache does. Without this, any prefill that spans more
        # than one step re-runs the whole vision tower: with staggered admission
        # that turned 37 admission steps into 69 encoder calls and cost 3.5s on
        # the Qwen3-VL image scenario. Entries are dropped by _finish_seq and on
        # preemption.
        #
        # tp=1 only: at tp>1 every rank keeps its own _vis_cache, populated by
        # _broadcast_visual, so skipping the encode on rank 0 would desynchronise
        # the ranks. Chunked multimodal prefill is rare without the stagger, so
        # tp>1 keeps re-encoding rather than growing the SHM protocol.
        _enc_reuse = (mr.world_size == 1)
        if _enc_reuse:
            for i, seq in enumerate(seqs):
                hit = self.encoder_cache.get(id(seq))
                if hit is None:
                    continue
                embeds, modality = hit
                mr._vis_cache.append(embeds)
                seq_entries.append((i, cache_idx, modality, 0, embeds.shape[0]))
                cache_idx += 1

        image_pv_list = []
        image_thw_list = []
        image_entries = []
        image_embed_start = 0
        for i, seq in enumerate(seqs):
            if seq.pixel_values is not None and not (
                    _enc_reuse and id(seq) in self.encoder_cache):
                thw = seq.image_grid_thw
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                image_pv_list.append(seq.pixel_values)
                image_thw_list.append(thw)
                embed_count = int((thw.prod(-1) // (merge_size ** 2)).sum().item())
                image_entries.append((i, image_embed_start, embed_count))
                image_embed_start += embed_count

        if image_pv_list:
            batched_pv = _stage_pixel_values(image_pv_list)
            batched_thw = torch.cat(image_thw_list, dim=0).cpu()
            mr._mm_pv = batched_pv
            mr._mm_thw = batched_thw
            mr.call(
                "_broadcast_visual",
                list(batched_pv.shape),
                list(batched_thw.shape),
            )
            for i, embed_start, embed_count in image_entries:
                seq_entries.append(
                    (i, cache_idx, "image", embed_start, embed_count),
                )
            cache_idx += 1

        video_pv_list = []
        video_thw_list = []
        video_entries = []
        video_embed_start = 0
        for i, seq in enumerate(seqs):
            if seq.video_pixel_values is not None and not (
                    _enc_reuse and id(seq) in self.encoder_cache):
                grid_thw = seq.video_grid_thw
                if not isinstance(grid_thw, torch.Tensor):
                    grid_thw = torch.tensor(grid_thw, dtype=torch.long)
                video_pv_list.append(seq.video_pixel_values)
                video_thw_list.append(grid_thw)
                embed_count = int(
                    (grid_thw.prod(-1) // (merge_size ** 2)).sum().item()
                )
                video_entries.append((i, video_embed_start, embed_count))
                video_embed_start += embed_count

        if video_pv_list:
            batched_pv = _stage_pixel_values(video_pv_list)
            batched_thw = torch.cat(video_thw_list, dim=0).cpu()
            mr._mm_pv = batched_pv
            mr._mm_thw = batched_thw
            mr.call(
                "_broadcast_visual",
                list(batched_pv.shape),
                list(batched_thw.shape),
            )
            for i, embed_start, embed_count in video_entries:
                seq_entries.append(
                    (i, cache_idx, "video", embed_start, embed_count),
                )
            cache_idx += 1

        if _enc_reuse:
            # Populate the reuse cache with this step's fresh encodes, sliced per
            # sequence -- but only for sequences whose prefill continues into a
            # later step, which is the only case that would re-encode.
            #
            # Caching unconditionally is a memory leak, not just waste: the entry
            # would then live until the sequence *finishes decoding*, and a
            # 32-frame clip's embeddings are ~145MB (4423 merged tokens x 4096 x
            # 2 bytes, x3 DeepStack levels). With hundreds of clips in flight
            # that is tens of GB, and it OOM'd the video scenario outright.
            for i, c_idx, modality, e_start, e_count in seq_entries:
                seq = seqs[i]
                if id(seq) in self.encoder_cache:
                    continue
                if chunk_sizes is None or i >= len(chunk_sizes):
                    continue
                if seq.num_computed_tokens + chunk_sizes[i] >= len(seq.token_ids):
                    continue  # prefill completes this step; nothing to reuse
                full = mr._vis_cache[c_idx]
                self.encoder_cache[id(seq)] = (
                    full[e_start:e_start + e_count], modality,
                )

        if _enc_reuse and chunk_sizes is not None:
            # Drop entries whose sequence finishes prefilling on this step -- the
            # placeholders are all consumed, so nothing will read them again.
            for i, seq in enumerate(seqs):
                if i < len(chunk_sizes) and id(seq) in self.encoder_cache \
                        and seq.num_computed_tokens + chunk_sizes[i] \
                        >= len(seq.token_ids):
                    self.encoder_cache.pop(id(seq), None)

        audio_seqs = [
            (i, seq) for i, seq in enumerate(seqs)
            if getattr(seq, "input_audio_features", None) is not None
        ]
        audio_output_lens = None
        if audio_seqs:
            feats = torch.cat(
                [seq.input_audio_features for _, seq in audio_seqs], dim=1,
            )
            lengths = torch.cat([
                seq.audio_feature_lengths.to(torch.long) for _, seq in audio_seqs
            ])
            mr._mm_audio_features = feats
            mr._mm_audio_lengths = lengths
            _, output_lens = mr.call(
                "_broadcast_audio", list(feats.shape), list(lengths.shape),
            )
            audio_output_lens = output_lens.detach().cpu().tolist()
            for i, _seq in audio_seqs:
                seq_entries.append((i, cache_idx, "audio", None, None))
            cache_idx += 1

        per_seq_map = [None] * len(seqs)
        audio_embed_start = 0
        audio_idx = 0
        for si, ci, modality, embed_start, embed_count in seq_entries:
            seq = seqs[si]
            if modality == "image":
                per_seq_map[si] = {
                    "cache_idx": ci,
                    "modality": "image",
                    "embed_start": embed_start,
                    "embed_count": embed_count,
                }
            elif modality == "video":
                per_seq_map[si] = {
                    "cache_idx": ci,
                    "modality": "video",
                    "embed_start": embed_start,
                    "embed_count": embed_count,
                }
            elif modality == "audio":
                assert audio_output_lens is not None
                embed_count = audio_output_lens[audio_idx]
                per_seq_map[si] = {
                    "cache_idx": ci,
                    "modality": "audio",
                    "embed_start": audio_embed_start,
                    "embed_count": embed_count,
                }
                audio_embed_start += embed_count
                audio_idx += 1

        return per_seq_map, cache_idx

    @torch.inference_mode()
    def _run_vision_encoder(self, seqs, chunk_sizes=None):
        """Run vision encoder with batching and caching.

        Images are batched into a single model.visual() call (concatenated
        pixel_values, stacked grid_thw). Videos are processed one-by-one to
        avoid OOM. Results are cached by id(seq) for reuse across steps.

        If chunk_sizes is provided, only produces embeddings for the chunk
        range [num_computed_tokens : num_computed_tokens + chunk_size] per seq,
        enabling chunked multimodal prefill.

        Returns (inputs_embeds, deepstack_embeds) where deepstack_embeds is a
        list of tensors for Qwen3-VL DeepStack, or None for Qwen2-VL.
        """
        model = self.model_runner.model
        has_deepstack = hasattr(model.visual, 'deepstack_merger_list')
        if has_deepstack:
            visual_dim = model.config.vision.out_hidden_size
            deepstack_num_levels = len(model.visual.deepstack_visual_indexes)
        merge_size = model.config.vision.spatial_merge_size
        image_token_id = self.model_runner.config.image_token_id
        video_token_id = self.model_runner.config.video_token_id

        # --- Phase 1: Run encoder for uncached sequences ---
        img_seqs = []
        img_pv_list = []
        img_thw_list = []
        for seq in seqs:
            if seq.pixel_values is not None and id(seq) not in self.encoder_cache:
                img_seqs.append(seq)
                img_pv_list.append(seq.pixel_values.cuda())
                thw = seq.image_grid_thw
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                img_thw_list.append(thw)

        if img_seqs:
            batched_pv = torch.cat(img_pv_list, dim=0)
            batched_thw = torch.cat(img_thw_list, dim=0).cpu()
            pv_shape = list(batched_pv.shape)
            thw_shape = list(batched_thw.shape)
            self.model_runner._mm_pv = batched_pv
            self.model_runner._mm_thw = batched_thw
            vis_out = self.model_runner.call(
                "_broadcast_visual", pv_shape, thw_shape,
            )

            sizes = (batched_thw.prod(-1) // (merge_size ** 2)).tolist()
            if has_deepstack and deepstack_num_levels > 0:
                all_img_embeds = vis_out[:, :visual_dim]
                ds_cat = vis_out[:, visual_dim:]
                all_ds_features = list(ds_cat.split(visual_dim, dim=1))
            else:
                all_img_embeds = vis_out
                all_ds_features = None

            per_seq_embeds = all_img_embeds.split(sizes)
            if all_ds_features is not None:
                per_seq_ds = [ds.split(sizes) for ds in all_ds_features]
            else:
                per_seq_ds = None

            embed_idx = 0
            for i, seq in enumerate(img_seqs):
                thw = seq.image_grid_thw
                n_items = len(thw) if isinstance(thw, list) else thw.shape[0]
                seq_embeds = torch.cat(
                    per_seq_embeds[embed_idx:embed_idx + n_items], dim=0
                ) if n_items > 1 else per_seq_embeds[embed_idx]
                if per_seq_ds is not None:
                    seq_ds = [
                        torch.cat(ds[embed_idx:embed_idx + n_items], dim=0)
                        if n_items > 1 else ds[embed_idx]
                        for ds in per_seq_ds
                    ]
                else:
                    seq_ds = []
                self.encoder_cache[id(seq)] = (seq_embeds, seq_ds, "image")
                embed_idx += n_items

            del batched_pv, batched_thw

        for seq in seqs:
            if seq.video_pixel_values is not None and id(seq) not in self.encoder_cache:
                video_pv = seq.video_pixel_values.cuda()
                grid_thw = seq.video_grid_thw
                if not isinstance(grid_thw, torch.Tensor):
                    grid_thw = torch.tensor(grid_thw, dtype=torch.long)
                grid_thw = grid_thw.cpu()
                vpv_shape = list(video_pv.shape)
                vthw_shape = list(grid_thw.shape)
                self.model_runner._mm_pv = video_pv
                self.model_runner._mm_thw = grid_thw
                vis_out = self.model_runner.call(
                    "_broadcast_visual", vpv_shape, vthw_shape,
                )

                if has_deepstack and deepstack_num_levels > 0:
                    video_embeds = vis_out[:, :visual_dim]
                    ds_cat = vis_out[:, visual_dim:]
                    ds_features = list(ds_cat.split(visual_dim, dim=1))
                else:
                    video_embeds = vis_out
                    ds_features = []
                self.encoder_cache[id(seq)] = (video_embeds, ds_features, "video")
                del video_pv

        # --- Phase 2: Merge vision embeddings into text embeddings ---
        all_inputs_embeds = []
        all_deepstack = [] if has_deepstack else None
        embed_fn = model.get_input_embeddings()

        for seq_idx, seq in enumerate(seqs):
            full_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")

            if chunk_sizes is not None:
                start = seq.num_computed_tokens
                end = start + chunk_sizes[seq_idx]
                chunk_ids = full_ids[start:end]
            else:
                chunk_ids = full_ids
                start = 0
                end = len(full_ids)

            text_embeds = embed_fn(chunk_ids)
            seq_deepstack = [] if has_deepstack else None

            cached = self.encoder_cache.get(id(seq))
            if cached is not None:
                embeds, ds_features, modality = cached
                tok_id = image_token_id if modality == "image" else video_token_id

                mask = chunk_ids == tok_id
                if mask.any():
                    if chunk_sizes is not None:
                        full_mask = full_ids == tok_id
                        chunk_vis_start = full_mask[:start].sum().item()
                        n_vis_in_chunk = mask.sum().item()
                        chunk_embeds = embeds[chunk_vis_start:chunk_vis_start + n_vis_in_chunk]
                        text_embeds[mask] = chunk_embeds.to(text_embeds.dtype)
                    else:
                        text_embeds[mask] = embeds.to(text_embeds.dtype)

                if has_deepstack and ds_features:
                    for i_ds, ds_feat in enumerate(ds_features):
                        ds_expanded = torch.zeros_like(text_embeds)
                        if mask.any():
                            if chunk_sizes is not None:
                                ds_expanded[mask] = ds_feat[chunk_vis_start:chunk_vis_start + n_vis_in_chunk].to(text_embeds.dtype)
                            else:
                                ds_expanded[mask] = ds_feat.to(text_embeds.dtype)
                        if modality == "video" and i_ds < len(seq_deepstack):
                            seq_deepstack[i_ds] = seq_deepstack[i_ds] + ds_expanded
                        else:
                            seq_deepstack.append(ds_expanded)

            all_inputs_embeds.append(text_embeds)
            if has_deepstack:
                all_deepstack.append(seq_deepstack)

        inputs_embeds = torch.cat(all_inputs_embeds, dim=0)

        if has_deepstack and all_deepstack:
            num_levels = max((len(ds) for ds in all_deepstack), default=0)
            if num_levels > 0:
                deepstack_embeds = []
                for level in range(num_levels):
                    level_parts = []
                    for ds in all_deepstack:
                        if level < len(ds):
                            level_parts.append(ds[level])
                        else:
                            level_parts.append(
                                torch.zeros_like(all_inputs_embeds[0]))
                    deepstack_embeds.append(torch.cat(level_parts, dim=0))
                return inputs_embeds, deepstack_embeds

        return inputs_embeds, None

    # ------------------------------------------------------------------
    # Mamba / SSM scheduling loop
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_kimi_linear(
        self,
        prompts,
        sp_list,
        collect_logits: bool = False,
        use_tqdm: bool = False,
    ):
        eos = self.tokenizer.eos_token_id
        mr = self.model_runner
        max_num_seqs = self.max_num_seqs
        max_batched_tokens = self.max_num_batched_tokens
        mr.call("reset_kimi_state_cache")

        all_seqs: list[Sequence] = []
        seq_sp: dict[int, "SamplingParams"] = {}
        for i, prompt in enumerate(prompts):
            sp = sp_list[i]
            ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
            seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
            all_seqs.append(seq)
            seq_sp[id(seq)] = sp

        seq_logits: dict[int, list[torch.Tensor]] = {
            id(s): [] for s in all_seqs
        } if collect_logits else {}

        waiting: deque[Sequence] = deque(all_seqs)
        running: list[Sequence] = []
        all_greedy = (
            not collect_logits
            and all(sp.temperature == 0.0 for sp in sp_list)
        )

        def _ensure_decode_blocks(seqs: list[Sequence]) -> bool:
            block_size = mr.mamba_state_manager.block_size
            new_block_counts: list[int] = []
            needs_blocks = False
            for seq in seqs:
                total_after = seq.num_computed_tokens + 1
                blocks_needed = (total_after + block_size - 1) // block_size
                count = max(0, blocks_needed - len(seq.block_table))
                new_block_counts.append(count)
                needs_blocks = needs_blocks or count > 0
            if not needs_blocks:
                return False
            new_blocks = mr.call("allocate_kimi_mla_blocks_batch", new_block_counts)
            for seq, blocks in zip(seqs, new_blocks):
                if blocks:
                    seq.block_table.extend(blocks)
            return True

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(
                total=len(prompts), desc="Processed prompts", dynamic_ncols=True,
            )
        decode_bt_dirty = True

        state_manager = mr.mamba_state_manager
        block_size = getattr(state_manager, "block_size", 0) or 1
        total_kv_blocks = getattr(state_manager, "num_mla_blocks", 0) or 0
        block_budget = max(1, total_kv_blocks - 2) if total_kv_blocks else 0

        def _worst_case_blocks(seq: Sequence) -> int:
            prompt_len = len(seq.token_ids) - len(seq.generated_ids)
            final_len = prompt_len + seq.max_tokens
            return (final_len + block_size - 1) // block_size

        # Admitted but not yet fully prefilled (chunked prefill in flight).
        prefilling: list[Sequence] = []

        while waiting or running or prefilling:
            # --- Admission -------------------------------------------------
            # Sequences move waiting -> prefilling (claiming one recurrent
            # state slot each) gated only by slot availability and the
            # worst-case KV block reservation; the per-step token budget is
            # handled by the chunk planner below, so a prompt longer than the
            # budget spans several steps instead of being admitted whole.
            blocked_by_kv = False
            if waiting:
                committed_blocks = (
                    sum(
                        _worst_case_blocks(seq)
                        for seq in itertools.chain(running, prefilling)
                    )
                    if total_kv_blocks
                    else 0
                )
                admitted: list[Sequence] = []
                while (
                    waiting
                    and len(running) + len(prefilling) + len(admitted) < max_num_seqs
                    and mr.can_allocate_mamba_state()
                ):
                    if total_kv_blocks:
                        candidate_blocks = _worst_case_blocks(waiting[0])
                        must_admit = not (running or prefilling or admitted)
                        if (
                            not must_admit
                            and committed_blocks + candidate_blocks > block_budget
                        ):
                            blocked_by_kv = True
                            break
                        committed_blocks += candidate_blocks
                    admitted.append(waiting.popleft())
                if admitted:
                    slots = mr.call("allocate_mamba_state_batch", len(admitted))
                    for seq, slot in zip(admitted, slots):
                        seq.state_slot = slot
                    prefilling.extend(admitted)

            # --- Chunk planner ---------------------------------------------
            # Fill max_num_batched_tokens with prefill chunks in FIFO order.
            # A prompt longer than the budget is split across steps; its
            # GDN/KDA conv + recurrent state continues via has_initial_state
            # and num_computed_tokens (vLLM's chunked-prefill contract), so a
            # 128K prompt never materializes a full-sequence chunk-scan
            # transient.
            #
            # The sequence that runs into the budget is *split* rather than
            # deferred, so every step is filled exactly. Ending the step short
            # instead looks harmless but is not: on the LongBench-v2 length mix
            # (5.8K..131.6K) it left 21% of each step's budget unused and took
            # 126 forwards where 100 carry the same 1.63M tokens, and a forward
            # costs the same launch overhead however few tokens are in it. This
            # is what vLLM's scheduler does -- it fills
            # ``max_num_batched_tokens`` to the token.
            prefill_seqs: list[Sequence] = []
            prefill_chunk_lens: list[int] = []
            budget_left = max_batched_tokens
            for seq in prefilling:
                remaining = seq.num_remaining_prefill
                if remaining <= 0:
                    continue
                chunk = remaining if remaining <= budget_left else budget_left
                prefill_seqs.append(seq)
                prefill_chunk_lens.append(chunk)
                budget_left -= chunk
                if budget_left <= 0:
                    break

            if prefill_seqs:
                decode_bt_dirty = True
                logits = mr.call(
                    "run", prefill_seqs, True, prefill_chunk_lens,
                )
                # The runner advanced num_computed_tokens by each chunk, so a
                # sequence with prefill left keeps its carried state and waits
                # for the next step; only a finished prompt yields a token.
                scheduled = {id(seq) for seq in prefill_seqs}
                prefilling = [
                    seq for seq in prefill_seqs if seq.num_remaining_prefill > 0
                ] + [seq for seq in prefilling if id(seq) not in scheduled]
                if logits is not None:
                    finished_payloads: list[tuple[int, list[int]]] = []
                    next_running: list[Sequence] = list(running)
                    sampled: list[Sequence] = []
                    for i, seq in enumerate(prefill_seqs):
                        if seq.num_remaining_prefill > 0:
                            continue
                        if collect_logits:
                            seq_logits[id(seq)].append(logits[i:i + 1].cpu())
                        tid = self._sample(logits[i:i + 1], seq_sp[id(seq)])[0]
                        seq.append_token(tid)
                        sampled.append(seq)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            finished_payloads.append((seq.state_slot, list(seq.block_table)))
                        else:
                            next_running.append(seq)
                    if finished_payloads:
                        mr.call("deallocate_mamba_state_batch", finished_payloads)
                        finished_slots = {slot for slot, _ in finished_payloads}
                        for seq in sampled:
                            if seq.state_slot in finished_slots:
                                seq.block_table = []
                                seq.state_slot = None
                                if pbar is not None:
                                    pbar.update(1)
                    running = next_running

            if not running:
                continue

            if (
                (waiting or prefilling)
                and not blocked_by_kv
                and all_greedy
                and all(seq.ignore_eos for seq in running)
                and len(running) < max_num_seqs
            ):
                # Throughput mode: admit/prefill all fixed-length requests
                # first so the steady-state decode batch has uniform length
                # and can use the bulk graph replay path below.
                continue

            if (
                (not waiting or len(running) >= max_num_seqs)
                and running
                and all_greedy
                and not mr.enforce_eager
                and getattr(mr, "_kimi_graph_bs_for_n", None) is not None
                and all(seq.ignore_eos for seq in running)
            ):
                remaining = [
                    seq.max_tokens - len(seq.generated_ids)
                    for seq in running
                ]
                min_remaining = min(remaining) if remaining else 0
                if min_remaining > 1:
                    chunk = min(min_remaining, _BULK_CHUNK_CAP)
                    graph_max = mr._kimi_graph_bs_list[-1]
                    for start in range(0, len(running), graph_max):
                        mr.call(
                            "run_kimi_decode_many",
                            running[start:start + graph_max],
                            chunk,
                        )
                    finished_payloads = [
                        (seq.state_slot, list(seq.block_table))
                        for seq in running
                        if len(seq.generated_ids) >= seq.max_tokens
                    ]
                    if finished_payloads:
                        mr.call(
                            "deallocate_mamba_state_batch",
                            finished_payloads,
                        )
                    next_running = []
                    for seq in running:
                        if len(seq.generated_ids) >= seq.max_tokens:
                            seq.block_table = []
                            seq.state_slot = None
                            if pbar is not None:
                                pbar.update(1)
                        else:
                            next_running.append(seq)
                    running = next_running
                    decode_bt_dirty = True
                    continue

            decode_seqs = list(running)
            if (
                all_greedy
                and not mr.enforce_eager
                and getattr(mr, "_kimi_graph_bs_for_n", None) is not None
                and len(decode_seqs) <= mr._kimi_graph_bs_list[-1]
            ):
                if _ensure_decode_blocks(decode_seqs):
                    decode_bt_dirty = True
                decode_data = mr._prepare_kimi_decode_arrays(
                    decode_seqs, copy_block_tables=decode_bt_dirty,
                )
                has_result, async_n = mr.call_kimi_decode_async(decode_data)
                if has_result:
                    token_ids = mr._wait_async_kimi_tokens(async_n)
                    finished_payloads: list[tuple[int, list[int]]] = []
                    next_running = []
                    any_finished = False
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        seq.num_computed_tokens += 1
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            any_finished = True
                            finished_payloads.append(
                                (seq.state_slot, list(seq.block_table)),
                            )
                        else:
                            next_running.append(seq)
                    if finished_payloads:
                        mr.call("deallocate_mamba_state_batch", finished_payloads)
                        finished_slots = {slot for slot, _ in finished_payloads}
                        for seq in decode_seqs:
                            if seq.state_slot in finished_slots:
                                seq.block_table = []
                                seq.state_slot = None
                                if pbar is not None:
                                    pbar.update(1)
                    running = next_running
                    decode_bt_dirty = any_finished
                    continue

            logits = mr.call("run", decode_seqs, False)
            if logits is None:
                continue
            decode_bt_dirty = True
            if collect_logits:
                for i, seq in enumerate(decode_seqs):
                    seq_logits[id(seq)].append(logits[i:i + 1].cpu())

            finished_payloads: list[tuple[int, list[int]]] = []
            next_running = []
            for i, seq in enumerate(decode_seqs):
                tid = self._sample(logits[i:i + 1], seq_sp[id(seq)])[0]
                seq.append_token(tid)
                done = len(seq.generated_ids) >= seq.max_tokens
                if not seq.ignore_eos:
                    done = done or tid == eos
                if done:
                    finished_payloads.append((seq.state_slot, list(seq.block_table)))
                else:
                    next_running.append(seq)

            if finished_payloads:
                mr.call("deallocate_mamba_state_batch", finished_payloads)
                finished_slots = {slot for slot, _ in finished_payloads}
                for seq in decode_seqs:
                    if seq.state_slot in finished_slots:
                        seq.block_table = []
                        seq.state_slot = None
                        if pbar is not None:
                            pbar.update(1)
            running = next_running

        if pbar is not None:
            pbar.close()

        return [
            GenerationOutput(
                prompt=(prompts[i] if isinstance(prompts[i], str) else ""),
                generated_text=(
                    self.tokenizer.decode(
                        all_seqs[i].generated_ids, skip_special_tokens=True,
                    )
                    if (
                        isinstance(prompts[i], str)
                    )
                    else ""
                ),
                token_ids=all_seqs[i].generated_ids,
                logits_history=(
                    seq_logits.get(id(all_seqs[i])) if collect_logits else None
                ),
            )
            for i in range(len(prompts))
        ]

    @torch.inference_mode()
    def _generate_mamba(
        self,
        prompts,
        sp_list,
        collect_logits: bool = False,
        use_tqdm: bool = False,
        decode_text: bool = True,
    ):
        """Scheduler for Mamba / Mamba2 models.

        Mamba state lives in a slot pool (one slot per live sequence) so
        scheduling reduces to: while there is a free slot and a waiting
        seq, admit it; each step runs a mixed prefill + decode batch
        through the model, with per-batch ``(Mamba|Mamba2)Metadata``
        carrying the slot indices and prefill/decode split.

        Hot loop optimisations (mirroring vLLM's GPUModelRunner):
          - When the step is *pure decode* and all sequences are greedy,
            we use a captured CUDA graph + GPU local argmax + async D2H
            copy of the next token IDs (``run_mamba_decode_fast_async``)
            and **pipeline** the next step's CPU prep with the previous
            step's tokens still in flight.  This is the steady state for
            most of generation once all prompts have been admitted.
          - When the step needs prefill (admitting new sequences) we
            fall back to ``run_mamba_mixed`` which builds the heavier
            varlen metadata once.
          - We pre-build a ``seq -> (sp, idx)`` lookup so we never call
            the O(N) ``all_seqs.index(s)`` per token per step.
        """
        eos = self.tokenizer.eos_token_id
        mr = self.model_runner

        # Optional per-step instrumentation -- enable with
        # ``FASTKERNELS_PROFILE_MAMBA=1``.  Records wall-clock time spent
        # in each phase (admit, decode-array prep, GPU dispatch, D2H
        # wait, finalize/dealloc) and prints a summary at the end so
        # we can see which phase dominates without resorting to a full
        # CUDA profiler.
        _profile = os.environ.get("FASTKERNELS_PROFILE_MAMBA", "0") == "1"
        _stats = {
            "fast_steps": 0,
            "fast_admit": 0.0,
            "fast_prep": 0.0,
            "fast_dispatch": 0.0,
            "fast_wait": 0.0,
            "fast_finalize": 0.0,
            "fast_pbar": 0.0,
            "slow_steps": 0,
            "slow_admit": 0.0,
            "slow_call": 0.0,
            "slow_finalize": 0.0,
            "slow_pbar": 0.0,
            "total_decode_tokens": 0,
            "total_prefill_tokens": 0,
        }

        # Build sequences in input order, plus a seq -> sp lookup (avoids
        # O(N^2) ``all_seqs.index(s)`` calls that dominated the old
        # scheduler at 1000 prompts).
        all_seqs: list[Sequence] = []
        seq_sp: dict[int, "SamplingParams"] = {}
        for i, prompt in enumerate(prompts):
            sp = sp_list[i]
            ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
            seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
            all_seqs.append(seq)
            seq_sp[id(seq)] = sp

        seq_logits: dict[int, list[torch.Tensor]] = {
            id(s): [] for s in all_seqs
        } if collect_logits else {}

        waiting: deque[Sequence] = deque(all_seqs)
        running: list[Sequence] = []
        # Sequences whose prompt is being consumed across multiple steps
        # (chunked prefill). Each holds a Mamba state slot; its SSM/conv state
        # is carried between chunks via has_initial_states + num_computed_tokens.
        prefilling: list[Sequence] = []

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=len(prompts), desc="Processed prompts",
                         dynamic_ncols=True)
        _pbar_pending = 0

        # Whether we can use the GPU greedy fast path for decode steps.
        # Multi-/non-greedy sampling falls back to the slow CPU path.
        all_greedy = (
            not collect_logits
            and all(sp.temperature == 0.0 for sp in sp_list)
        )

        def _admit():
            """Move waiting seqs into ``prefilling`` (allocating one Mamba state
            slot each) while the slot pool and ``max_num_seqs`` allow.

            Token budgeting is handled per step by the chunk planner below -- a
            single prompt may span many steps (chunked prefill), so admission is
            gated only by slot availability, not prompt length.

            Allocation must happen on every TP rank so each rank's local
            ``MambaStateManager`` agrees on slot ownership; ``mr.call``
            broadcasts via SHM and runs locally on rank 0. Batched into a single
            message to avoid ``_pickle.UnpicklingError`` race conditions.
            """
            admitted: list[Sequence] = []
            max_seqs = self.max_num_seqs
            while (
                waiting
                and mr.can_allocate_mamba_state()
                and len(running) + len(prefilling) + len(admitted) < max_seqs
            ):
                admitted.append(waiting.popleft())
            if admitted:
                slots = mr.call("allocate_mamba_state_batch", len(admitted))
                for s, slot in zip(admitted, slots):
                    s.state_slot = slot
                prefilling.extend(admitted)
            return admitted

        def _sample(logits_row: torch.Tensor, sp) -> int:
            if sp.temperature == 0.0:
                return int(logits_row.argmax().item())
            probs = torch.softmax(logits_row.float() / sp.temperature, dim=-1)
            top_k = getattr(sp, "top_k", None)
            if top_k is not None and top_k > 0:
                top_v, top_i = torch.topk(probs, k=min(top_k, probs.numel()))
                probs = torch.zeros_like(probs).scatter_(0, top_i, top_v)
                probs = probs / probs.sum()
            return int(torch.multinomial(probs, num_samples=1).item())

        def _finalize(seq, tok_id):
            """Append a token to ``seq`` and report whether it finished."""
            seq.append_token(tok_id)
            seq.num_computed_tokens = len(seq)
            done = len(seq.generated_ids) >= seq.max_tokens
            if not seq.ignore_eos:
                done = done or tok_id == eos
            return done

        while waiting or running or prefilling:
            _t0 = time.perf_counter() if _profile else 0.0
            _admit()
            decode_seqs = list(running)

            # Chunk planner: fill the per-step token budget
            # (max_num_batched_tokens, minus one token per running decode) with
            # prefill chunks in FIFO order over the ``prefilling`` queue. A
            # prompt longer than the budget is split across steps; its SSM/conv
            # state continues via has_initial_states + num_computed_tokens
            # (matching vLLM's chunked-prefill scheduler), so a single long
            # prefill never materializes a full-sequence chunk-scan transient.
            #
            # Charged against the *unpadded* decode count even though
            # ``_mamba_prepare_tensors`` rounds Mamba v1's decode rows up: the
            # <=7 extra rows are a rounding error against a 16K budget, and
            # keeping the divisor here means the alignment padding does not move
            # where long prompts get cut.  Chunk boundaries change which tokens
            # share a scan accumulation, so moving them reshuffles bf16
            # tie-breaks and costs ~6% of the reference token agreement for no
            # throughput gain.
            token_budget = max(
                getattr(mr, "max_num_batched_tokens", 16384) - len(decode_seqs), 0,
            )
            # Mamba2 SSD chunk size: a *continuing* (non-final) prefill chunk
            # must end on a chunk_size boundary, else the carried SSM state
            # diverges from a single-shot prefill (verified: sub-chunk_size steps
            # mismatch, >=chunk_size match). vLLM enforces the same alignment.
            # Mamba v1 has no such constraint and only needs the 16-byte GEMM
            # alignment its channel-major mixer implies, so it keeps the whole
            # budget instead of rounding down to 256.
            chunk_align = getattr(mr, "_mamba_prefill_align", 256)
            prefill_seqs: list[Sequence] = []
            prefill_chunk_lens: list[int] = []
            budget_left = token_budget
            for s in prefilling:
                remaining = s.num_remaining_prefill
                if remaining <= 0:
                    continue
                if remaining <= budget_left:
                    chunk = remaining                       # final chunk: exact
                else:
                    # Continuing chunk: largest aligned block that fits.
                    aligned = (budget_left // chunk_align) * chunk_align
                    if aligned <= 0:
                        if prefill_seqs:
                            break                           # no room; next step
                        aligned = chunk_align               # guarantee progress
                    chunk = aligned
                prefill_seqs.append(s)
                prefill_chunk_lens.append(chunk)
                budget_left -= chunk
                if budget_left <= 0:
                    break
            # A *final* chunk is a prompt's exact remainder, so the step's
            # prefill token count can still land unaligned -- that count is the
            # contiguous axis of the x_proj / out_proj operands (see the mixer),
            # and an odd one costs ~3 ms per layer.  Shaving the tail off the
            # last chunk keeps the total aligned.  Skipped for Mamba2, whose
            # token-major mixer is immune and whose chunk boundaries are
            # load-bearing.
            if not mr.is_mamba2 and prefill_chunk_lens:
                excess = sum(prefill_chunk_lens) % _MAMBA_GEMM_ALIGN
                # Only trim while prefill work outlives this step regardless, so
                # the trimmed tail rides along in a step that was already going
                # to run.  If this step would have drained the queue, trimming
                # would buy a whole extra forward pass instead, and the mixer's
                # cheap pad is the better trade.
                if 0 < excess < prefill_chunk_lens[-1] and (
                    waiting
                    or len(prefilling) > len(prefill_seqs)
                    or any(
                        c < s.num_remaining_prefill
                        for s, c in zip(prefill_seqs, prefill_chunk_lens)
                    )
                ):
                    prefill_chunk_lens[-1] -= excess
            _t_admit = (time.perf_counter() - _t0) if _profile else 0.0

            if not prefill_seqs and not decode_seqs:
                break

            # =========================================================
            # FAST PATH: pure decode + greedy + CUDA-graph capture set up.
            # Steady-state for the bulk of decode-heavy / balanced runs.
            # =========================================================
            if (
                not prefill_seqs
                and decode_seqs
                and all_greedy
                and mr.max_num_batched_tokens >= len(decode_seqs)
            ):
                _t1 = time.perf_counter() if _profile else 0.0
                decode_data = mr._prepare_mamba_decode_arrays(decode_seqs)
                _t_prep = (time.perf_counter() - _t1) if _profile else 0.0

                _t1 = time.perf_counter() if _profile else 0.0
                has_result, async_n = mr.call_mamba_decode_async(decode_data)
                _t_dispatch = (time.perf_counter() - _t1) if _profile else 0.0

                _t_wait = 0.0
                _t_finalize = 0.0
                # Drain any waiting prompts admitted between steps while
                # we still keep pipelining decodes -- but only on the
                # rank-0 path that actually owns the result.
                if has_result:
                    _t1 = time.perf_counter() if _profile else 0.0
                    token_ids = mr._wait_async_mamba_tokens(async_n)
                    _t_wait = (time.perf_counter() - _t1) if _profile else 0.0

                    _t1 = time.perf_counter() if _profile else 0.0
                    finished_now: list[Sequence] = []
                    new_running: list[Sequence] = []
                    for s, tok_id in zip(decode_seqs, token_ids):
                        if _finalize(s, tok_id):
                            finished_now.append(s)
                        else:
                            new_running.append(s)
                    if finished_now:
                        # Broadcast to all TP ranks so every rank
                        # returns the same slots to its local free pool
                        # in lock-step.
                        slot_ids = [s.state_slot for s in finished_now]
                        mr.call("deallocate_mamba_state_batch", slot_ids)
                        for s in finished_now:
                            s.state_slot = None
                            if pbar is not None:
                                _pbar_pending += 1
                    running = new_running
                    _t_finalize = (time.perf_counter() - _t1) if _profile else 0.0
                else:
                    # Worker rank or graph fell through; treat as no-op.
                    running = decode_seqs

                _t1 = time.perf_counter() if _profile else 0.0
                if pbar is not None and _pbar_pending:
                    pbar.update(_pbar_pending)
                    _pbar_pending = 0
                _t_pbar = (time.perf_counter() - _t1) if _profile else 0.0

                if _profile:
                    _stats["fast_steps"] += 1
                    _stats["fast_admit"] += _t_admit
                    _stats["fast_prep"] += _t_prep
                    _stats["fast_dispatch"] += _t_dispatch
                    _stats["fast_wait"] += _t_wait
                    _stats["fast_finalize"] += _t_finalize
                    _stats["fast_pbar"] += _t_pbar
                    _stats["total_decode_tokens"] += len(decode_seqs)
                continue

            # =========================================================
            # SLOW PATH: any mixed prefill+decode step (or non-greedy).
            # =========================================================
            _t1 = time.perf_counter() if _profile else 0.0
            logits = mr.call(
                "run_mamba_mixed", prefill_seqs, decode_seqs, prefill_chunk_lens,
            )
            _t_call = (time.perf_counter() - _t1) if _profile else 0.0

            _t1 = time.perf_counter() if _profile else 0.0
            # Greedy sampling for the whole batch in one argmax + one D2H copy.
            # Per-row ``argmax().item()`` costs a launch and a device sync each,
            # which at ~500 rows per mixed step dominated the step's host time.
            greedy_ids = logits.argmax(dim=-1).tolist() if all_greedy else None
            row = 0
            new_running: list[Sequence] = []
            finished_now: list[Sequence] = []
            still_prefilling: list[Sequence] = []
            # Prefill rows come first (one per prefill seq's chunk last token).
            # Advance each seq by its chunk; only a seq whose prompt is now fully
            # consumed produces a real token (its chunk's last token is the final
            # prompt token). Partial seqs keep their carried state and continue.
            for s, chunk in zip(prefill_seqs, prefill_chunk_lens):
                s.num_computed_tokens += chunk
                if s.num_remaining_prefill == 0:
                    sp = seq_sp[id(s)]
                    if greedy_ids is not None:
                        tok_id = greedy_ids[row]
                    else:
                        logit_row = logits[row]
                        tok_id = _sample(logit_row, sp)
                        if collect_logits:
                            seq_logits[id(s)].append(logit_row.detach().cpu())
                    if _finalize(s, tok_id):
                        finished_now.append(s)
                    else:
                        new_running.append(s)
                else:
                    still_prefilling.append(s)
                row += 1
            # Decode rows follow.
            for s in decode_seqs:
                sp = seq_sp[id(s)]
                if greedy_ids is not None:
                    tok_id = greedy_ids[row]
                else:
                    logit_row = logits[row]
                    tok_id = _sample(logit_row, sp)
                    if collect_logits:
                        seq_logits[id(s)].append(logit_row.detach().cpu())
                row += 1
                if _finalize(s, tok_id):
                    finished_now.append(s)
                else:
                    new_running.append(s)

            scheduled_prefill_ids = {id(s) for s in prefill_seqs}
            prefilling[:] = still_prefilling + [
                s for s in prefilling if id(s) not in scheduled_prefill_ids
            ]
            if finished_now:
                slot_ids = [s.state_slot for s in finished_now]
                mr.call("deallocate_mamba_state_batch", slot_ids)
                for s in finished_now:
                    s.state_slot = None
                    if pbar is not None:
                        _pbar_pending += 1
            running = new_running
            _t_finalize_slow = (time.perf_counter() - _t1) if _profile else 0.0

            _t1 = time.perf_counter() if _profile else 0.0
            if pbar is not None and _pbar_pending:
                pbar.update(_pbar_pending)
                _pbar_pending = 0
            _t_pbar = (time.perf_counter() - _t1) if _profile else 0.0

            if _profile:
                _stats["slow_steps"] += 1
                _stats["slow_admit"] += _t_admit
                _stats["slow_call"] += _t_call
                _stats["slow_finalize"] += _t_finalize_slow
                _stats["slow_pbar"] += _t_pbar
                _stats["total_prefill_tokens"] += sum(prefill_chunk_lens)
                _stats["total_decode_tokens"] += len(decode_seqs)

        if pbar is not None:
            pbar.close()

        if _profile:
            def _fmt(t):
                return f"{t * 1000:>9.1f} ms"
            print("\n=== _generate_mamba per-phase timing ===")
            n_fast = _stats["fast_steps"]
            n_slow = _stats["slow_steps"]
            print(f"  Steps: fast_decode={n_fast}  slow_mixed={n_slow}  "
                  f"decode_tokens={_stats['total_decode_tokens']}  "
                  f"prefill_tokens={_stats['total_prefill_tokens']}")
            if n_fast:
                tot_fast = sum([
                    _stats["fast_admit"], _stats["fast_prep"],
                    _stats["fast_dispatch"], _stats["fast_wait"],
                    _stats["fast_finalize"], _stats["fast_pbar"],
                ])
                print(f"  FAST PATH ({n_fast} steps, total {_fmt(tot_fast)}):")
                print(f"    admit         {_fmt(_stats['fast_admit'])}  "
                      f"({100*_stats['fast_admit']/max(tot_fast,1e-9):5.1f}%)")
                print(f"    decode_prep   {_fmt(_stats['fast_prep'])}  "
                      f"({100*_stats['fast_prep']/max(tot_fast,1e-9):5.1f}%)")
                print(f"    gpu_dispatch  {_fmt(_stats['fast_dispatch'])}  "
                      f"({100*_stats['fast_dispatch']/max(tot_fast,1e-9):5.1f}%)")
                print(f"    gpu+d2h_wait  {_fmt(_stats['fast_wait'])}  "
                      f"({100*_stats['fast_wait']/max(tot_fast,1e-9):5.1f}%)")
                print(f"    finalize      {_fmt(_stats['fast_finalize'])}  "
                      f"({100*_stats['fast_finalize']/max(tot_fast,1e-9):5.1f}%)")
                print(f"    pbar          {_fmt(_stats['fast_pbar'])}  "
                      f"({100*_stats['fast_pbar']/max(tot_fast,1e-9):5.1f}%)")
                print(f"    avg/step      {_fmt(tot_fast/n_fast)}")
            if n_slow:
                tot_slow = sum([
                    _stats["slow_admit"], _stats["slow_call"],
                    _stats["slow_finalize"], _stats["slow_pbar"],
                ])
                print(f"  SLOW PATH ({n_slow} steps, total {_fmt(tot_slow)}):")
                print(f"    admit         {_fmt(_stats['slow_admit'])}  "
                      f"({100*_stats['slow_admit']/max(tot_slow,1e-9):5.1f}%)")
                print(f"    mr.call(mix)  {_fmt(_stats['slow_call'])}  "
                      f"({100*_stats['slow_call']/max(tot_slow,1e-9):5.1f}%)")
                print(f"    finalize      {_fmt(_stats['slow_finalize'])}  "
                      f"({100*_stats['slow_finalize']/max(tot_slow,1e-9):5.1f}%)")
                print(f"    pbar          {_fmt(_stats['slow_pbar'])}  "
                      f"({100*_stats['slow_pbar']/max(tot_slow,1e-9):5.1f}%)")
                print(f"    avg/step      {_fmt(tot_slow/n_slow)}")
            print("=" * 50)

        # Release any cached transient activations back to the OS so the
        # next ``generate()`` call (e.g. the next benchmark scenario)
        # starts with a clean allocator.  Mirrors how vLLM's
        # ``LLMEngine`` calls ``empty_cache`` after each batch finishes
        # to avoid cumulative fragmentation across requests
        # (``vllm/v1/engine/core.py:_step``).  Without this, Mamba2's
        # 16384-token prefill activations stay cached on rank 0 and
        # the second scenario can OOM looking for a contiguous block.
        if mr.world_size > 1:
            mr.call("empty_cuda_cache")
        else:
            torch.cuda.empty_cache()

        return [
            GenerationOutput(
                prompt=(prompts[i] if isinstance(prompts[i], str) else ""),
                generated_text=(
                    self.tokenizer.decode(
                        all_seqs[i].generated_ids, skip_special_tokens=True,
                    )
                    if decode_text else ""
                ),
                token_ids=all_seqs[i].generated_ids,
                logits_history=(
                    seq_logits.get(id(all_seqs[i])) if collect_logits else None
                ),
            )
            for i in range(len(prompts))
        ]

    def _vision_merge_size(self) -> int:
        try:
            return int(
                self.model_runner.model.config.vision.spatial_merge_size
            )
        except AttributeError:
            return 1

    def _max_encoder_tokens(self) -> int:
        value = os.environ.get("FASTKERNELS_MAX_ENCODER_TOKENS")
        if value:
            return max(1, int(value))
        return int(self.max_num_batched_tokens)

    @torch.inference_mode()
    def generate(self, prompts, sampling_params, collect_logits: bool = False,
                 images=None, videos=None, audio_features=None,
                 use_tqdm: bool = False,
                 decode_text: bool = True):
        """Generate completions for a batch of prompts.

        Uses unified chunked-prefill scheduling: every GPU step processes
        both decode tokens (for running seqs) and prefill chunks (for
        new/continuing seqs) in a single forward pass, matching vLLM's
        approach.

        For Whisper (encoder-decoder), audio_features is a list of
        [num_mel_bins, T] log-mel spectrogram tensors. The encoder runs
        during the first prefill step and cross-attention KV is written
        to paged cache. No chunked prefill for encoder-decoder (matching
        vLLM).
        """
        if isinstance(sampling_params, list):
            sp_list = sampling_params
        else:
            sp_list = [sampling_params] * len(prompts)

        seed = sp_list[0].seed
        if seed is not None:
            self._set_seeds(seed)

        if self.is_kimi_linear or self.is_qwen3_next:
            return self._generate_kimi_linear(
                prompts, sp_list, collect_logits=collect_logits,
                use_tqdm=use_tqdm,
            )

        if self.is_mamba:
            return self._generate_mamba(
                prompts, sp_list, collect_logits=collect_logits,
                use_tqdm=use_tqdm, decode_text=decode_text,
            )

        eos = self.tokenizer.eos_token_id
        waiting: deque[Sequence] = deque()
        running: deque[Sequence] = deque()
        prefilling: deque[Sequence] = deque()

        seq_logits: dict[int, list[torch.Tensor]] = {}

        # Handle multimodal inputs
        if images is None:
            images = [None] * len(prompts)
        if videos is None:
            videos = [None] * len(prompts)
        if audio_features is None:
            audio_features = [None] * len(prompts)

        _preprocess_t0 = time.perf_counter()

        # Whisper: compute encoder output length from mel spectrogram length.
        # Two conv layers with stride 2 => T_enc = T_mel // 4 (rounded down
        # by convolutions, but padding ensures it's close to T_mel // 2 per layer).
        # More precisely: after conv1 with kernel=3, stride=1, padding=1: T_out = T_mel
        # after conv2 with kernel=3, stride=2, padding=1: T_out = (T_mel + 1) // 2
        # Then the max is clamped to max_source_positions (1500).
        def _whisper_encoder_tokens(mel_T):
            return min((mel_T + 1) // 2, getattr(self.model_runner.config, 'max_source_positions', 1500))

        def _make_seq(i):
            prompt = prompts[i]
            sp = sp_list[i]
            img = images[i] if i < len(images) else None
            vid = videos[i] if i < len(videos) else None
            aud = audio_features[i] if i < len(audio_features) else None

            if self.is_whisper and aud is not None:
                ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
                max_toks = min(sp.max_tokens,
                               self.model_runner.max_model_len - len(ids))
                seq = Sequence(ids, max_tokens=max_toks, ignore_eos=sp.ignore_eos)
                seq.encoder_features = aud
                mel_T = aud.shape[-1]
                seq.encoder_seq_len = _whisper_encoder_tokens(mel_T)
                return seq
            elif self.is_qwen_vl and (
                img is not None or vid is not None or aud is not None
            ):
                (ids, pixel_values, image_grid_thw,
                 video_pv, video_grid_thw, video_second_per_grid,
                 input_audio_features, audio_feature_lengths) = (
                    self._preprocess_multimodal(
                        prompt, images=img, videos=vid, audios=aud,
                    )
                )
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
                seq.pixel_values = pixel_values
                seq.image_grid_thw = image_grid_thw.tolist() if image_grid_thw is not None else None
                seq.video_pixel_values = video_pv
                seq.video_grid_thw = video_grid_thw.tolist() if video_grid_thw is not None else None
                seq.video_second_per_grid = (
                    video_second_per_grid.tolist()
                    if video_second_per_grid is not None else None
                )
                seq.input_audio_features = input_audio_features
                seq.audio_feature_lengths = audio_feature_lengths

                model = self.model_runner.model
                merge_size = model.config.vision.spatial_merge_size
                image_token_id = self.model_runner.config.image_token_id
                video_token_id = self.model_runner.config.video_token_id
                image_offsets = []
                video_offsets = []
                img_idx = 0
                vid_idx = 0
                i_tok = 0
                while i_tok < len(ids):
                    tid = ids[i_tok]
                    if tid == image_token_id and seq.image_grid_thw and img_idx < len(seq.image_grid_thw):
                        image_offsets.append(i_tok)
                        t, h, w = seq.image_grid_thw[img_idx]
                        num_tokens = t * (h // merge_size) * (w // merge_size)
                        i_tok += num_tokens
                        img_idx += 1
                    elif tid == video_token_id and seq.video_grid_thw and vid_idx < len(seq.video_grid_thw):
                        t, h, w = seq.video_grid_thw[vid_idx]
                        tokens_per_frame = (h // merge_size) * (w // merge_size)
                        if self.is_qwen3_vl:
                            frames_found = 0
                            j = i_tok
                            while j < len(ids) and frames_found < t:
                                if ids[j] == video_token_id:
                                    video_offsets.append(j)
                                    j += tokens_per_frame
                                    frames_found += 1
                                else:
                                    j += 1
                            vid_idx += 1
                            i_tok = j
                        else:
                            video_offsets.append(i_tok)
                            num_tokens = t * tokens_per_frame
                            i_tok += num_tokens
                            vid_idx += 1
                    else:
                        i_tok += 1

                mrope_positions, delta = model.get_mrope_input_positions(
                    ids,
                    image_grid_thw=seq.image_grid_thw,
                    video_grid_thw=seq.video_grid_thw,
                    image_offsets=image_offsets if image_offsets else None,
                    video_offsets=video_offsets if video_offsets else None,
                    video_second_per_grid=seq.video_second_per_grid,
                    audio_feature_lengths=audio_feature_lengths,
                )
                seq.mrope_positions = mrope_positions
                seq.mrope_position_delta = delta

                # Placeholder-token positions for the per-step vision merge.
                # Done here, on the producer thread, so the engine loop never
                # has to derive them from GPU masks.
                if seq.pixel_values is not None:
                    seq.vis_token_id = image_token_id
                elif seq.video_pixel_values is not None:
                    seq.vis_token_id = video_token_id
                elif input_audio_features is not None:
                    seq.vis_token_id = getattr(
                        self.model_runner.config, "audio_token_id", None)
                if seq.vis_token_id is not None:
                    ids_np = np.asarray(ids, dtype=np.int64)
                    seq.vis_positions = np.flatnonzero(
                        ids_np == seq.vis_token_id).astype(np.int64)
            elif self.is_qwen_vl:
                ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
                seq.mrope_positions = torch.arange(len(ids), dtype=torch.int64).unsqueeze(0).expand(3, -1)
                seq.mrope_position_delta = 0
            else:
                ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
            return seq

        num_prompts = len(prompts)
        all_seqs: list = [None] * num_prompts

        # Preprocessing is real work for the multimodal scenarios: the HF
        # processor plus mrope positions cost ~9ms per VisionArena image and
        # ~140ms per 32-frame MMVU clip, i.e. 9s / 140s over a 1000-prompt run.
        # vLLM never pays that in front of its engine -- the frontend process
        # preprocesses request N while the EngineCore process is already running
        # GPU steps for requests 1..N-1, which is why its ``Processed prompts``
        # bar starts several seconds into a timed generate(). Mirror that: a
        # producer thread walks the prompts in order and publishes each Sequence
        # to ``waiting`` as soon as it is ready, so the scheduler can issue GPU
        # work after the first prompt instead of after the last.
        #
        # One producer thread driving a 2-worker pool, not a wide pool. The HF
        # processor scales *negatively* past two threads -- measured on this
        # host over VisionArena images through Qwen2VLProcessor: 8.7ms/item
        # serial, 6.7ms at 2 threads, 9.3ms at 3, 15.3ms at 4, 32.6ms at 8,
        # 50.7ms at 28 -- because each item is mostly GIL-bound Python around
        # OpenMP regions that convoy when several threads enter them at once.
        # torch.set_num_threads(1) in the workers recovers <10% of it. So the
        # ThreadPoolExecutor(max_workers=8) that used to sit here was 3.7x
        # slower than two workers, and cost ~25s of the 66.5s Qwen2-VL image
        # scenario on its own.
        #
        # ``pool.map`` yields in submission order, so ``waiting`` is still filled
        # strictly left-to-right and the scheduler admits prompts FCFS as before.
        #
        # Text-only batches keep the pre-loop barrier: _make_seq is then just a
        # tokenizer.encode (or nothing at all, when the caller passed token ids),
        # so there is no cost to hide, and handing the scheduler the whole queue
        # up front lets it pack full max_num_batched_tokens prefill steps from
        # step 0 instead of ramping. Single-prompt generates (the latency
        # scenarios) keep it too -- there is no other work to overlap with, so a
        # thread would only add startup and wake-up latency.
        _streaming_preprocess = (
            self.is_qwen_vl
            and num_prompts > 1
            and any(x is not None
                    for x in (*images, *videos, *audio_features))
        )
        _produce_done = threading.Event()
        _produce_error: list[BaseException] = []

        def _produce_seqs():
            try:
                if _streaming_preprocess:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        it = enumerate(pool.map(_make_seq, range(num_prompts)))
                        for i, seq in it:
                            all_seqs[i] = seq
                            # Admission index: MLA/DSA decode keeps ``running``
                            # sorted by it so the packed batch layout matches
                            # vLLM's persistent batch step-for-step.
                            seq._req_idx = i
                            if collect_logits:
                                seq_logits[id(seq)] = []
                            waiting.append(seq)
                else:
                    for i in range(num_prompts):
                        seq = _make_seq(i)
                        all_seqs[i] = seq
                        seq._req_idx = i
                        if collect_logits:
                            seq_logits[id(seq)] = []
                        waiting.append(seq)
            except BaseException as exc:  # surfaced by the caller below
                _produce_error.append(exc)
            finally:
                _produce_done.set()

        _producer = None
        if _streaming_preprocess:
            _producer = threading.Thread(
                target=_produce_seqs, name="fk-preprocess", daemon=True,
            )
            _producer.start()
        else:
            _produce_seqs()
            if _produce_error:
                raise _produce_error[0]

        _preprocess_time = time.perf_counter() - _preprocess_t0
        if os.environ.get("FASTKERNELS_STEP_PROFILE") == "1":
            _how = ("dispatched to producer thread" if _streaming_preprocess
                    else "serial")
            print(f"[Profile] Preprocessing {num_prompts} seqs ({_how}): "
                  f"{_preprocess_time:.3f}s")

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=num_prompts, desc="Processed prompts",
                         dynamic_ncols=True,
                         postfix="est. speed input: 0.00 toks/s, "
                                 "output: 0.00 toks/s")
        num_finished = 0
        total_in_toks = 0
        total_out_toks = 0

        use_greedy = (sp_list[0].temperature == 0.0
                      and not collect_logits)
        is_bitnet = getattr(
            self.model_runner.config, "model_type", "",
        ) == "bitnet"
        block_size = BLOCK_SIZE
        bm = self.block_manager
        num_blocks = bm._num_blocks
        watermark_blocks = max(int(num_blocks * 0.01), 1)

        _pbar_pending = 0
        _pbar_pending_in = 0
        _pbar_pending_out = 0

        step_profile = {
            "pure_decode": 0, "decode_tokens": 0, "decode_time": 0.0,
            "pure_mm_prefill": 0, "mm_prefill_tokens": 0, "mm_prefill_time": 0.0,
            "pure_text_prefill": 0, "text_prefill_tokens": 0,
            "mixed_mm": 0, "mixed_mm_pf_tokens": 0, "mixed_mm_dc_tokens": 0, "mixed_mm_time": 0.0,
            "mixed_text": 0, "mixed_text_pf_tokens": 0, "mixed_text_dc_tokens": 0,
            "preempted": 0, "preempted_tokens": 0,
        }
        _step_profile_active = os.environ.get("FASTKERNELS_STEP_PROFILE") == "1"

        _admit_stride_cache: list = [False, 1, False]

        def _reserve_full() -> bool:
            _admit_stride()
            return _admit_stride_cache[2]

        def _admit_stride() -> int:
            """Admit a full batch every Nth step instead of every step.

            When a batch cannot be held at full length, admitting it as fast as
            the producer can supply it is the worst schedule: every request
            reaches its final size at about the same step, KV hits 100%, and the
            scheduler evicts the overflow, which re-prefills and runs as a
            second decode wave 512 steps long at a handful of sequences. On the
            Qwen3-VL image scenario that wave is 491 of 1030 decode steps for
            ~10% of the tokens.

            Staggering admission fixes the peak, because at the peak step a
            request admitted at step t holds ceil((prompt + max_tokens - t)/block)
            rather than the full amount. Summed over a spread of S steps the peak
            is N*F - N*S/(2*block), so the spread that just fits is
            S >= 2*block*(N*F - usable)/N.

            Caveat on that derivation: N*F, the sum of final lengths, is not the
            demand capacity has to cover. Requests retire while others grow, so
            the real peak is lower -- by 0.54x on gemma-4's mixed workload, which
            is enough to invert the regime choice below (see the scope guard).
            ``tests/debug/workload_batch_profile.py`` computes both from a
            workload; the peak it reports matched the engine's own kv_peak
            counter to 0.3% there.

            Spreading by admitting *fewer per step* was measured and is a net
            loss (0.83x -> 0.79x): the vision encoder batches per step, so going
            from 25 images per call to 7 raised per-image encoder cost 27% and
            tripled the number of decode-bearing prefill steps. Striding instead
            keeps every encoder call full and simply leaves gaps between them.

            vLLM gets the same stagger for free -- its frontend renders
            multimodal prompts serially, ~9s of a 28s run, so admission spreads
            over ~320 steps and its KV peaks at 97.9% having never preempted.

            Multimodal/encoder-decoder models only; see the scope guard below.
            Returns 1 (admit every step) when the batch already fits.
            """
            if _admit_stride_cache[0]:
                return _admit_stride_cache[1]
            _env = os.environ.get("FASTKERNELS_MM_ADMIT_STRIDE")
            if _env:
                _admit_stride_cache[0] = True
                _admit_stride_cache[1] = max(1, int(_env))
                return _admit_stride_cache[1]
            # Both this stagger and the full-length reservation it selects
            # between are corrections for one multimodal-only cost: a preempted
            # multimodal request re-runs its *vision encoder* as well as its
            # prefill, so evictions are dearer for us than for vLLM, which
            # reserves only the current length in every regime and eats the
            # preemption. Neither divergence has a text-model justification (no
            # encoder to redo, and no serial-frontend counterpart in vLLM to
            # imitate) or a text-model measurement, and on gemma-4-26B-A4B the
            # reservation regime misfired badly: N*F/usable is only 1.52x
            # (47,000 blocks of final length against 30,868 usable), which the
            # spread test below reads as "far over" and answers with a full
            # reservation -- yet with the reservation off the run peaks at 81.6%
            # of the pool and preempts zero times. It cost 1496 steps at mean
            # decode batch 301 against 1044 at 391, i.e. 9,793 -> 10,997 tok/s
            # on the mixed scenario (1044 steps at mean 391 is the workload's
            # floor: max output_len is 1024 and sum/max is 392). End to end
            # against vLLM 0.26 on B200 tp=1, full validate config, that row is
            # 0.822x -> 0.997x and 1.039x on two idle-host passes, with
            # long-context, single-request and fixed-batch-32 unchanged at
            # 1.01x / 1.12-1.13x / 1.14-1.15x and identical output token counts.
            # This is the same fault 1c9268f removed for gemma-4 once already,
            # reintroduced from the multimodal side, so gate the whole mechanism
            # on the cost that motivates it instead of leaving it tree-wide.
            #
            # Gemma-4 is also the only text model on this loop that can reach the
            # gate, because its pool is the smallest by a wide margin -- 31,179
            # blocks, where the other four are 70,892 (Llama-3.1-8B), 110,463
            # (Mixtral-8x7B tp=2), 216,987 (gpt-oss-120b tp=2) and 29,177 at
            # block_size 64 (GLM-5.2-NVFP4 tp=8). All four were replayed under
            # FASTKERNELS_MM_ADMIT_SCOPE=all on the mixed workload: none ever
            # stops on "reserved", none takes a stride, and all already run at
            # their workload floor (KV peaks 34.6% / 25.6% / 11.9% / 21.3%), so
            # this guard is a measured no-op for them. Gemma-4's pool is small
            # only because 25 of its 30 layers are sliding_attention whose
            # out-of-window blocks we do not free -- see
            # _allocate_variable_kv_cache.
            #
            # FASTKERNELS_MM_ADMIT_SCOPE=all restores the tree-wide behaviour,
            # which is how a text model is A/B'd against it.
            if (not (self.is_qwen_vl or self.is_whisper)
                    and os.environ.get("FASTKERNELS_MM_ADMIT_SCOPE") != "all"):
                _admit_stride_cache[0] = True
                _admit_stride_cache[1] = 1
                _admit_stride_cache[2] = False
                return 1
            # On by default, but the sign of this trade-off depends on how
            # expensive a prefill step is, so it has moved twice under
            # measurement. Striding holds KV under 100% and drives preemptions
            # to 0, paying for it with ~50% more mixed prefill steps, each
            # carrying its own encoder call and decode pass. With DeepStack
            # prefill still running eagerly that was a net loss (0.83x -> 0.80x);
            # once prefill went through the compiled graph (+40% prefill
            # throughput) it became a clear win (0.78x -> 0.89x on the Qwen3-VL
            # image scenario). Note this has no counterpart in vLLM -- its
            # admission spread is an incidental consequence of a serial frontend,
            # not a policy -- so it is a deliberate divergence, justified only by
            # the measurement above and by the guard below that keeps it off
            # where a modest stagger cannot help.
            # 256, not fewer. A smaller sample resolves the regime sooner --
            # which does help video reach zero preemptions -- but it estimates the
            # mean block count too noisily and flips the regime decision on the
            # image scenarios, which sit near the boundary: at 32 both regressed
            # (Qwen2-VL image 0.96x -> 0.92x, Qwen3-VL image 0.89x -> 0.85x) while
            # video moved only 0.94x -> 0.93x. The residual preemptions video
            # keeps at 256 are not what limits it.
            _MIN_SAMPLE = 256
            if len(waiting) < _MIN_SAMPLE and not _produce_done.is_set():
                return 1
            sample = list(itertools.islice(waiting, _MIN_SAMPLE))
            if not sample or num_prompts <= 1:
                return 1
            merge = self._vision_merge_size()
            f_blocks = sum(
                (min(s.num_prompt_tokens + s.max_tokens, self.max_model_len)
                 + block_size - 1) // block_size
                for s in sample
            ) / len(sample)
            mean_prompt = sum(s.num_prompt_tokens for s in sample) / len(sample)
            mean_enc = max(1.0, sum(_seq_encoder_tokens(s, merge)
                                    for s in sample) / len(sample))
            usable = num_blocks - watermark_blocks
            stride = 1
            deficit = num_prompts * f_blocks - usable
            spread = 2 * block_size * max(0.0, deficit) / num_prompts
            # Only stagger when a *modest* stagger is enough. If the batch needs
            # far more KV than exists -- MMVU clips want ~254k blocks against
            # 63k, i.e. several unavoidable decode waves -- no admission order
            # avoids preemption, and spreading would merely delay all the work
            # (the derivation asks for a 6000-step spread there). Requiring the
            # spread to fit inside half the decode length keeps this to the case
            # it was measured on, where it is worth 1030 steps -> 658.
            mean_max_tokens = sum(s.max_tokens for s in sample) / len(sample)
            # 1.25x margin on the required spread. The bound is derived from a
            # uniform-admission model, and reality is noisier: the same derived
            # stride of 4 landed at 98.0% KV with 0 preemptions on one run and
            # 100% with 18 preemptions on the next (0.89x vs 0.86x). Since
            # crossing 100% costs a 512-step low-occupancy wave while a slightly
            # longer stagger costs only a few extra prefill steps, the asymmetry
            # is worth paying for.
            # 2x margin on the derived spread. The bound comes from a uniform
            # admission model and consistently under-predicts the real peak by
            # ~1600 blocks, because admission is bursty (a full batch every Nth
            # step) and chunked prefill lengthens lifetimes. Measured on the
            # Qwen3-VL image scenario: the 1x spread (stride 4-5) landed at 100%
            # KV with 11-18 preemptions and 0.86x, while ~2x (stride 10) held
            # 87.5% with zero preemptions and 0.89x. Crossing 100% costs a full
            # 512-step low-occupancy wave even for 11 evictions, so err long.
            # This also lands close to vLLM's own effective spread (~320 steps,
            # from its serial frontend), which is where its KV peaks at 97.9%
            # having never preempted.
            spread *= 2.0
            _admit_stride_cache[2] = bool(
                deficit > 0 and spread > 0.75 * mean_max_tokens)
            if deficit > 0 and spread <= 0.75 * mean_max_tokens:
                # How many requests one step can actually take, i.e. what keeps
                # the encoder calls full.
                per_step = max(1.0, min(
                    self.max_num_batched_tokens / max(1.0, mean_prompt),
                    self._max_encoder_tokens() / mean_enc,
                ))
                steps_needed = max(1.0, num_prompts / per_step)
                stride = max(1, int(round(spread / steps_needed)))
            _admit_stride_cache[0] = True
            _admit_stride_cache[1] = stride
            if stride > 1 and self.model_runner.rank == 0 \
                    and os.environ.get("FASTKERNELS_STEP_PROFILE") == "1":
                print(f"  mm_admit_stride:   mean {f_blocks:.1f} blocks/seq x "
                      f"{num_prompts} = {num_prompts * f_blocks:.0f} vs "
                      f"{usable} usable -> spread {spread:.0f} steps, "
                      f"{per_step:.0f}/batch -> admit every {stride} steps",
                      flush=True)
            return stride
        def _seq_final_blocks(seq) -> int:
            """Blocks this sequence will hold once it has generated max_tokens."""
            return (min(seq.num_prompt_tokens + seq.max_tokens,
                        self.max_model_len) + block_size - 1) // block_size

        # Reserve final length at admission, but only when the batch cannot fit
        # no matter how it is ordered. Which of the three regimes applies is
        # decided by the same sample as the stagger (see _admit_stride):
        #
        #   N*F <= capacity        no constraint. Qwen2-VL image peaks at 52% of
        #                          its cache; gating it is a pure loss
        #                          (measured 0.96x -> 0.92x).
        #   N*F slightly over      a stagger fixes the peak while keeping every
        #                          request in flight -> use the stagger.
        #                          Reserving here just converts preemption into
        #                          an equivalent hold-back queue and costs more
        #                          (Qwen3-VL image 0.85x -> 0.76x).
        #   N*F far over           several decode waves are unavoidable, so
        #                          holding requests back costs nothing that
        #                          capacity was not already costing -- and it
        #                          buys back everything preemption wastes.
        #                          Qwen3-VL video: 85 preemptions -> 0, prefill
        #                          3.84M -> 3.54M tokens, 0.91x -> 0.94x.
        #
        # vLLM reserves only the current length in all three regimes and eats the
        # preemption; that is cheaper for it than for us because an evicted
        # multimodal request re-runs its vision encoder as well as its prefill.
        # That cost is also the whole reason to diverge, so _reserve_full() is
        # false for text models regardless of regime (see _admit_stride) -- and
        # note that "N*F far over" is a poor proxy for the real peak whenever
        # output lengths are heterogeneous, because requests retire while others
        # grow: gemma-4 mixed classifies as 1.52x over and still peaks at 81.6%.
        _committed = [0]

        # Track unconditionally, enforce conditionally. Gating the *accounting*
        # on _reserve_full() undercounts badly: the flag needs a 256-sequence
        # sample to resolve, and on the video scenarios several hundred requests
        # are admitted before that, so the running total could never reach the
        # threshold and the gate never bound (86 preemptions, unchanged 0.91x).
        def _commit(seq) -> None:
            _committed[0] += _seq_final_blocks(seq)

        def _uncommit(seq) -> None:
            _committed[0] = max(0, _committed[0] - _seq_final_blocks(seq))

        def _record_kv_usage() -> None:
            """Peak KV blocks in use. Preemption starts when this hits
            num_blocks, so it says whether the cache is genuinely full or
            whether blocks are being held/leaked somewhere."""
            u = num_blocks - len(bm.free_block_ids)
            if u > step_profile.get("kv_peak_blocks", 0):
                step_profile["kv_peak_blocks"] = u

        # No per-band time accumulation. It was attempted and removed: a step's
        # band is recorded before the forward and its duration after, but the
        # untimed step categories break the pairing, so the split dropped or
        # misassigned time (403 steps in the top occupancy band were credited
        # 1.1s while fast_decode alone measured 16.07s over 511 steps). The batch
        # histogram below is sound and is what localised the preemption tail; the
        # per-category totals are what to reason from for time.

        def _record_decode_batch(n: int) -> None:
            """Histogram of decode batch sizes.

            Total output tokens are fixed, so wall clock is driven by how many
            *steps* they are spread over, i.e. by occupancy. A run whose average
            batch is well under what the KV cache can hold is paying for a long
            low-occupancy tail rather than for slow kernels.
            """
            h = step_profile.setdefault("dc_hist", [0] * 8)
            cap = max(1, self.max_num_seqs)
            b = min(7, (n * 8) // cap)
            h[b] += 1
            _record_kv_usage()
            step_profile["dc_max"] = max(step_profile.get("dc_max", 0), n)

        def _record_preemption(seq: Sequence) -> None:
            """Count recompute preemptions and the decode work they discard.

            ``Sequence.preempt()`` resets to the prompt and drops
            ``generated_ids``, so every preemption costs a re-prefill *and* a
            replay of every decode step the sequence had already taken. Greedy
            decoding makes the replayed tokens identical, so output is
            unaffected -- but a nonzero count here means wasted work, and is the
            signal to make preempt() resume from prompt+generated the way
            vLLM's ``_preempt_request`` does (it keeps ``_all_token_ids`` and
            only resets ``num_computed_tokens``).
            """
            step_profile["preempted"] += 1
            step_profile["preempted_tokens"] += len(seq.generated_ids)

        def _finish_seq(seq: Sequence) -> None:
            nonlocal _pbar_pending, _pbar_pending_in, _pbar_pending_out
            seq.status = SeqStatus.FINISHED
            _uncommit(seq)
            bm.deallocate(seq)
            bm.deallocate_cross(seq)
            self.encoder_cache.pop(id(seq), None)
            if pbar is not None:
                _pbar_pending += 1
                _pbar_pending_in += seq.num_prompt_tokens
                _pbar_pending_out += len(seq.generated_ids)

        def _flush_pbar() -> None:
            nonlocal _pbar_pending, _pbar_pending_in, _pbar_pending_out
            nonlocal total_in_toks, total_out_toks
            if _pbar_pending == 0:
                return
            total_in_toks += _pbar_pending_in
            total_out_toks += _pbar_pending_out
            elapsed = pbar.format_dict["elapsed"]
            if elapsed > 0:
                pbar.postfix = (
                    f"est. speed input: {total_in_toks / elapsed:.2f}"
                    f" toks/s, output: "
                    f"{total_out_toks / elapsed:.2f} toks/s")
            pbar.update(_pbar_pending)
            _pbar_pending = 0
            _pbar_pending_in = 0
            _pbar_pending_out = 0

        def _prefill_blocked_by_capacity() -> bool:
            """Whether scheduler would be unable to add prefill work now.

            This lets the pure-decode fast path keep using async D2H and
            incremental metadata while prefill queues are blocked by the
            same capacity checks used by the scheduler below. As soon as a
            decode finish frees enough capacity, fast decode stops and the
            normal scheduler admits or continues prefill work.
            """
            if not waiting and not prefilling:
                return True
            if self.is_qwen_vl or self.is_whisper:
                return False

            decode_count = min(len(running), self.max_num_seqs)
            if decode_count == 0:
                return False

            decode_need_blocks = 0
            for i, seq in enumerate(running):
                if i < decode_count and len(seq) % block_size == 1:
                    decode_need_blocks += 1

            free_after_decode = len(bm.free_block_ids) - decode_need_blocks
            if free_after_decode < 0:
                return False

            token_budget = self.max_num_batched_tokens - decode_count
            if token_budget <= 0:
                return True

            for seq in prefilling:
                remaining = seq.num_remaining_prefill
                if remaining <= 0:
                    continue
                chunk = min(remaining, token_budget)
                if chunk <= 0:
                    break
                if seq.blocks_needed_for(chunk) <= free_after_decode:
                    return False

            if not waiting:
                return True

            seq = waiting[0]
            prompt_len = seq.num_prompt_tokens
            chunk = min(prompt_len, token_budget)
            blocks_needed = (chunk + block_size - 1) // block_size
            if free_after_decode < blocks_needed + watermark_blocks:
                return True

            # vLLM's ``full_sequence_must_fit`` admission gate (on by default
            # via ``scheduler_reserve_full_isl``) requires the request's
            # *current* length -- ``min(request.num_tokens, max_model_len)``,
            # i.e. prompt + tokens generated so far -- to fit in the free
            # blocks. It exists so chunked prefill cannot admit a 100k prompt
            # on the strength of its first 2k chunk alone. It deliberately does
            # *not* reserve the tokens the request has yet to generate, and it
            # is a per-request check against free blocks rather than a global
            # sum over everything in flight. Reserving the final length
            # (prompt + max_tokens) for every in-flight sequence instead caps
            # concurrency at a fraction of max_num_seqs; growth past this point
            # is handled by preemption in ``_schedule_decode_tokens``, exactly
            # as vLLM handles it.
            full_blocks = (min(prompt_len, self.max_model_len)
                           + block_size - 1) // block_size
            if free_after_decode < full_blocks + watermark_blocks:
                return True

            if decode_count + 1 > self.max_num_seqs:
                return True

            return False

        # Diagnostic gate: force every decode step through the fresh-rebuild
        # path (_prepare_decode_arrays reads live seq state). The async fast
        # path reuses the prior step's sampled ``token_ids`` to build the next
        # input incrementally (6664), so token substitution at append_token
        # would NOT propagate there. Off by default; used only by the
        # forced-decode agreement harness.
        _force_sync_decode = os.environ.get(
            "FASTKERNELS_FORCE_SYNC_DECODE", "0") != "0"

        # Prefill-priority scheduling: keep prefill steps PURE instead of adding
        # decode tokens to them.
        #
        # Mixing costs prefill throughput badly on a DSA model. Measured on
        # long-context (64 x 24,576 tokens): the identical prefill work runs at
        # 26,829 tok/s as pure prefill steps but only ~20,700 tok/s when the
        # scheduler folds a handful of decode tokens in (96 mixed steps at 792 ms
        # each against 608 ms for the same token count). vLLM absorbs that cost by
        # graph-capturing the mixed prefill-decode shapes (PIECEWISE); we run them
        # eager, so we pay it.
        #
        # This diverges from vLLM's batching, which mixes freely. It is on by
        # default anyway, because measurement says it improves fidelity rather
        # than trading it away: on the mixed workload the average matching prefix
        # against vLLM went from 13.03 to 26.83 tokens (and throughput 1.562x ->
        # 1.634x), while long-context went 0.843x -> 0.948x with alignment 30.02
        # -> 27.23, inside its 25.6-43.1 run-to-run band. Latency is unchanged.
        # The reading is that our *unified mixed-batch* path is itself a source of
        # divergence from vLLM, so keeping prefill steps pure removes an error
        # source as well as the eager-step cost. Set the env var to 0 to restore
        # vLLM-style mixed batching.
        # Scoped to DSA/indexer models: that is the only family it was measured
        # on, and it changes batch composition for every request, so enabling it
        # tree-wide on one model's evidence would put every other architecture's
        # throughput and alignment at risk untested. Set the env var explicitly
        # to force it on or off anywhere.
        _pp_env = os.environ.get("FASTKERNELS_PREFILL_PRIORITY")
        _prefill_priority = (
            _pp_env != "0" if _pp_env is not None
            else bool(getattr(self.model_runner, "_has_indexer_layers", False))
        )

        def _can_enter_decode_fast_path() -> bool:
            if _force_sync_decode:
                return False
            # Note: the stagger in ``_admit_stride`` blocks admission outright on
            # non-stride steps, so those steps carry decode and nothing else, and
            # it is tempting to route them here (94 of 673 steps on the Qwen3-VL
            # image scenario take the scheduler path only to admit nothing).
            # Measured: catastrophic. Skipping the scheduler also skips
            # ``_schedule_decode_tokens``, which is what promotes sequences whose
            # prefill just finished into ``running`` and grows their block
            # tables, so the fast path replays against a nearly empty batch --
            # 673 steps became 7714 at 66 decode tokens per step instead of 760,
            # KV peaked at 19.8% instead of 97.9%, and the scenario went
            # 0.93x -> 0.51x. The scheduler pass on those steps is not waste.
            return (
                running
                and use_greedy
                and _prefill_blocked_by_capacity()
            )

        _loop_t0 = time.perf_counter()
        _step_index = -1
        # Kernel-level profile over a window of engine steps:
        # FASTKERNELS_TORCH_PROFILE="<first_step>:<num_steps>". Diffing this
        # table against vLLM's is what located the vision encoder's excess
        # ``direct_copy`` traffic (55% of a 21ms/call deficit), so keep it
        # available rather than re-deriving it each time.
        _tp_win = os.environ.get("FASTKERNELS_TORCH_PROFILE")
        _torch_prof = None
        if _tp_win and num_prompts >= 100:
            # Warmup and alignment call generate() with a handful of prompts and
            # reset the step index each time, so an ungated window profiles those
            # instead of the timed run.
            _tp_start, _tp_count = (int(x) for x in _tp_win.split(":"))
        else:
            _tp_win = None
        while (waiting or running or prefilling
               or not _produce_done.is_set()):
            _step_index += 1
            # Keep the decode (running) queue in admission order so the packed
            # batch layout matches vLLM's persistent-batch order step-for-step
            # (offline determinism vs vLLM). fk otherwise rotates decoded seqs
            # to the tail of ``running`` each step; vLLM keeps them in place.
            # DSA sparse attention's decode split-KV reduction is batch-order-
            # sensitive, so this ordering is what lets fk reproduce vLLM's
            # long-context batched tokens. Gated to DSA/MLA to avoid perturbing
            # other models' batch shapes.
            if getattr(getattr(self, "model_runner", None),
                       "is_deepseek_mla", False) and len(running) > 1:
                running = deque(
                    sorted(running, key=lambda s: getattr(s, "_req_idx", 0))
                )
            if _tp_win:
                if _step_index == _tp_start:
                    _torch_prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False, with_stack=False,
                    )
                    _torch_prof.__enter__()
                elif _torch_prof is not None and \
                        _step_index == _tp_start + _tp_count:
                    _torch_prof.__exit__(None, None, None)
                    print(f"\n[Torch Profile] steps "
                          f"[{_tp_start}, {_step_index}) "
                          f"({_tp_count} steps)")
                    print(_torch_prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=45))
                    _torch_prof = None
            if _produce_error:
                break
            if pbar is not None:
                _flush_pbar()
            # =============================================================
            # FAST PATH: pure decode (most common steady-state)
            # Skip the full scheduler when no prefill can be admitted.
            # =============================================================
            if _can_enter_decode_fast_path():
                if is_bitnet and len(running) == 1:
                    seq = running[0]
                    remaining = seq.max_tokens - len(seq.generated_ids)
                    mr = self.model_runner
                    # Microsoft BitNet's ladder decode kernel is fastest under
                    # CUDA graph, but for 1024-token prefills the captured graph
                    # diverges from the official direct decode in the first few
                    # autoregressive steps. Seed the KV state with a tiny
                    # uncompiled prefix, then return to graph replay.
                    if (
                        seq.ignore_eos
                        and remaining > 0
                        and seq.num_prompt_tokens >= 1024
                        and len(seq.generated_ids) <= 4
                        and mr.world_size == 1
                        and not mr.enforce_eager
                    ):
                        if len(seq) % block_size != 1 or bm.free_block_ids:
                            if len(seq) % block_size == 1:
                                seq.block_table.append(
                                    bm.free_block_ids.popleft())
                            decode_data = mr._prepare_decode_arrays([seq])
                            gpu_ids = mr._run_decode_greedy_eager(*decode_data)
                            if gpu_ids is not None:
                                tid = int(gpu_ids[:1].tolist()[0])
                                seq.append_token(tid)
                                done = len(seq.generated_ids) >= seq.max_tokens
                                if not seq.ignore_eos:
                                    done = done or tid == eos
                                if done:
                                    _finish_seq(seq)
                                    running.clear()
                                continue

                need_blocks = 0
                for seq in running:
                    if len(seq) % block_size == 1:
                        need_blocks += 1
                if need_blocks <= len(bm.free_block_ids):
                    if _step_profile_active:
                        _spt0 = time.perf_counter()
                        step_profile["fast_decode"] = step_profile.get("fast_decode", 0) + 1
                        step_profile["fast_decode_tokens"] = step_profile.get("fast_decode_tokens", 0) + len(running)
                        _record_decode_batch(len(running))
                    if _PROFILE:
                        _fp_t0 = time.perf_counter()
                    decode_seqs = list(running)
                    for seq in decode_seqs:
                        if len(seq) % block_size == 1:
                            seq.block_table.append(bm.free_block_ids.popleft())

                    mr = self.model_runner
                    n_dc = len(decode_seqs)
                    if self.is_whisper:
                        mr._set_cross_attn_context_decode(decode_seqs)
                    decode_data = mr._prepare_decode_arrays(decode_seqs)
                    if _PROFILE:
                        _fp_t1 = time.perf_counter()
                    if mr.world_size > 1:
                        mr._write_decode_shm(*decode_data)
                        mr.shm.buf[mr._SHM_FLAG_OFFSET] = 1
                        mr._signal_workers()
                    has_result, _async_n = mr.run_decode_greedy_fast_async(decode_data)
                    if _PROFILE:
                        _fp_t2 = time.perf_counter()
                    if has_result:
                        token_ids = mr._wait_async_tokens(_async_n)
                        if _PROFILE:
                            _fp_t3 = time.perf_counter()
                        any_finished = False
                        for seq, tid in zip(decode_seqs, token_ids):
                            seq.append_token(tid)
                            done = len(seq.generated_ids) >= seq.max_tokens
                            if not seq.ignore_eos:
                                done = done or tid == eos
                            if done:
                                _finish_seq(seq)
                                any_finished = True
                        if any_finished:
                            running = deque(s for s in running
                                            if s.status != SeqStatus.FINISHED)
                        if _PROFILE:
                            _fp_t4 = time.perf_counter()
                            _fp = getattr(self, '_fast_path_profile', None)
                            if _fp is None:
                                _fp = {'prep': 0., 'gpu': 0., 'tolist': 0.,
                                       'post': 0., 'n': 0}
                                self._fast_path_profile = _fp
                            _fp['prep'] += _fp_t1 - _fp_t0
                            _fp['gpu'] += _fp_t2 - _fp_t1
                            _fp['tolist'] += _fp_t3 - _fp_t2
                            _fp['post'] += _fp_t4 - _fp_t3
                            _fp['n'] += 1

                        _whisper_fast = self.is_whisper
                        use_incr = True

                        # --- host run-ahead, one step deep -------------------
                        # Without this the host waits for the sampled token
                        # before it can launch the next step, so the GPU drains
                        # between steps: measured 1.56 ms of idle per step at
                        # bs=1 against vLLM's 0.74 ms. ``_pending`` holds a step
                        # that is still on the GPU; its sequences carry a
                        # placeholder token that ``_retire_pending`` overwrites
                        # once the D2H copy lands. The next step can be prepared
                        # and launched in the meantime because the graph reads
                        # its tokens straight from the sampler's device tensor
                        # (see _stage_next_input_ids).
                        _pending = None      # (seqs, n) with tokens in flight

                        def _retire_pending():
                            """Patch in the real tokens for the in-flight step."""
                            nonlocal _pending, any_finished, running
                            seqs_p, n_p = _pending
                            _pending = None
                            tids = mr._wait_async_tokens(n_p)
                            for seq_p, tid_p in zip(seqs_p, tids):
                                seq_p.token_ids[-1] = tid_p
                                seq_p.generated_ids[-1] = tid_p
                                # The run-ahead guard below only defers steps
                                # that cannot terminate a sequence, so this is
                                # dead code -- kept so the two paths are
                                # observably identical if the guard ever widens.
                                done_p = len(seq_p.generated_ids) >= seq_p.max_tokens
                                if not seq_p.ignore_eos:
                                    done_p = done_p or tid_p == eos
                                if done_p:
                                    _finish_seq(seq_p)
                                    any_finished = True
                            if any_finished:
                                running = deque(
                                    s for s in running
                                    if s.status != SeqStatus.FINISHED)

                        while _can_enter_decode_fast_path():
                            if any_finished:
                                if _pending is not None:
                                    _retire_pending()
                                decode_seqs = list(running)
                                n_dc = len(decode_seqs)
                                any_finished = False
                                use_incr = False
                                if _whisper_fast:
                                    mr._set_cross_attn_context_decode(decode_seqs)

                            need_blocks = 0
                            for seq in decode_seqs:
                                if len(seq) % block_size == 1:
                                    need_blocks += 1
                            if need_blocks > len(bm.free_block_ids):
                                break
                            if _step_profile_active:
                                step_profile["fast_decode"] = step_profile.get("fast_decode", 0) + 1
                                step_profile["fast_decode_tokens"] = step_profile.get("fast_decode_tokens", 0) + n_dc
                                _record_decode_batch(n_dc)
                            for seq in decode_seqs:
                                if len(seq) % block_size == 1:
                                    seq.block_table.append(
                                        bm.free_block_ids.popleft())
                            if _PROFILE:
                                _fp_t0 = time.perf_counter()
                            if use_incr:
                                decode_data = \
                                    mr._update_decode_arrays_incremental(
                                        n_dc, token_ids, decode_seqs)
                            else:
                                decode_data = mr._prepare_decode_arrays(
                                    decode_seqs)
                                use_incr = True
                            if _PROFILE:
                                _fp_t1 = time.perf_counter()
                            if mr.world_size > 1:
                                mr._write_decode_shm(*decode_data)
                                mr.shm.buf[mr._SHM_FLAG_OFFSET] = 1
                                mr._signal_workers()
                            has_result, _async_n = mr.run_decode_greedy_fast_async(decode_data)
                            if _PROFILE:
                                _fp_t2 = time.perf_counter()

                            # This step is on the GPU now, so retiring the
                            # previous one costs only the copy that has already
                            # had a full step to complete.
                            if _pending is not None:
                                _retire_pending()

                            # Defer this step's host read too, and let the next
                            # iteration launch on top of it -- but only when the
                            # step provably cannot finish a sequence. With greedy
                            # sampling and ignore_eos, termination is purely a
                            # length question, so the next step's batch is
                            # already known and the pipeline is exactly
                            # equivalent to the blocking loop. Anything else
                            # (an EOS-terminated request, the last token of a
                            # request, an eager step with no device handoff)
                            # falls through to the blocking path unchanged.
                            if (has_result and not any_finished
                                    and mr._staged_ids_n == _async_n
                                    and all(s.ignore_eos
                                            and len(s.generated_ids) + 1
                                            < s.max_tokens
                                            for s in decode_seqs)):
                                if _PROFILE:
                                    _fp_t3 = time.perf_counter()
                                for seq in decode_seqs:
                                    seq.append_token(_RUNAHEAD_PLACEHOLDER)
                                _pending = (decode_seqs, _async_n)
                                # The graph takes its tokens from the device, so
                                # the incremental prep must not write the host
                                # array -- the value there is a placeholder.
                                token_ids = None
                                if _PROFILE:
                                    _fp_t4 = time.perf_counter()
                                    _fp['prep'] += _fp_t1 - _fp_t0
                                    _fp['gpu'] += _fp_t2 - _fp_t1
                                    _fp['tolist'] += _fp_t3 - _fp_t2
                                    _fp['post'] += _fp_t4 - _fp_t3
                                    _fp['n'] += 1
                                continue

                            if has_result:
                                token_ids = mr._wait_async_tokens(_async_n)
                                if _PROFILE:
                                    _fp_t3 = time.perf_counter()
                                for seq, tid in zip(decode_seqs, token_ids):
                                    seq.append_token(tid)
                                    done = (len(seq.generated_ids)
                                            >= seq.max_tokens)
                                    if not seq.ignore_eos:
                                        done = done or tid == eos
                                    if done:
                                        _finish_seq(seq)
                                        any_finished = True
                                if any_finished:
                                    running = deque(
                                        s for s in running
                                        if s.status != SeqStatus.FINISHED)
                                if _PROFILE:
                                    _fp_t4 = time.perf_counter()
                                    _fp['prep'] += _fp_t1 - _fp_t0
                                    _fp['gpu'] += _fp_t2 - _fp_t1
                                    _fp['tolist'] += _fp_t3 - _fp_t2
                                    _fp['post'] += _fp_t4 - _fp_t3
                                    _fp['n'] += 1
                            else:
                                # No tokens came back. ``token_ids`` may be None
                                # from an earlier deferral and there is no device
                                # handoff to fall back on, so the next step has to
                                # rebuild its input array from the sequences.
                                use_incr = False

                        # Leaving the fast path (out of blocks, or a prefill is
                        # ready): the in-flight step must be retired before any
                        # other code reads generated_ids.
                        if _pending is not None:
                            _retire_pending()
                    if _step_profile_active:
                        step_profile["decode_time"] += time.perf_counter() - _spt0
                    continue

            elif running and not waiting and not prefilling:
                decode_seqs = list(running)
                need_blocks = 0
                for seq in decode_seqs:
                    if len(seq) % block_size == 1:
                        need_blocks += 1
                if need_blocks <= len(bm.free_block_ids):
                    for seq in decode_seqs:
                        if len(seq) % block_size == 1:
                            seq.block_table.append(bm.free_block_ids.popleft())
                    if self.is_whisper:
                        self.model_runner._set_cross_attn_context_decode(decode_seqs)
                    result = self.model_runner.call("run", decode_seqs, False)
                    if result is not None:
                        if collect_logits:
                            for i, seq in enumerate(decode_seqs):
                                seq_logits[id(seq)].append(result[i:i+1].cpu())
                        token_ids = self._sample(result, sp_list[0])
                        finished_set = set()
                        for seq, tid in zip(decode_seqs, token_ids):
                            seq.append_token(tid)
                            done = len(seq.generated_ids) >= seq.max_tokens
                            if not seq.ignore_eos:
                                done = done or tid == eos
                            if done:
                                _finish_seq(seq)
                                finished_set.add(id(seq))
                        if finished_set:
                            running = deque(s for s in running if id(s) not in finished_set)
                    continue

            # =============================================================
            # SCHEDULE: one unified step
            # =============================================================
            token_budget = self.max_num_batched_tokens

            # --- 1. Allocate blocks for decode seqs that need a new block ---
            decode_seqs: list[Sequence] = []
            new_running: deque[Sequence] = deque()
            def _preempt_newest(exclude: Sequence) -> bool:
                """Free the newest in-flight sequence to make room. vLLM's
                ``schedule()`` does ``preempted_req = self.running.pop()`` --
                the *last* running request -- and loops until the allocation
                fits, giving up only when the request needing blocks is itself
                the newest. Preempting the newest discards the least computed
                work and keeps the oldest sequences advancing, so the queue
                always drains. Returns False when nothing else can be freed.
                """
                # ``running`` is ordered oldest-to-newest and this loop pops from
                # its left, so whatever is left in ``running`` is the newest;
                # ``new_running`` (deferred by max_num_seqs) and ``decode_seqs``
                # (already scheduled this step) hold progressively older ones.
                for pool in (running, new_running):
                    while pool:
                        victim = pool.pop()
                        if victim is exclude:
                            pool.append(victim)
                            break
                        _record_preemption(victim)
                        _uncommit(victim)
                        bm.deallocate(victim)
                        bm.deallocate_cross(victim)
                        self.encoder_cache.pop(id(victim), None)
                        victim.preempt()
                        waiting.appendleft(victim)
                        return True
                for i in range(len(decode_seqs) - 1, -1, -1):
                    victim = decode_seqs[i]
                    if victim is exclude:
                        continue
                    del decode_seqs[i]
                    _record_preemption(victim)
                    _uncommit(victim)
                    bm.deallocate(victim)
                    bm.deallocate_cross(victim)
                    self.encoder_cache.pop(id(victim), None)
                    victim.preempt()
                    waiting.appendleft(victim)
                    return True
                return False

            def _schedule_decode_tokens() -> None:
                nonlocal token_budget, running, decode_seqs, new_running
                while running:
                    seq = running.popleft()
                    if len(decode_seqs) >= self.max_num_seqs:
                        new_running.append(seq)
                        continue
                    needs_block = (len(seq) % block_size == 1)
                    if needs_block:
                        while not bm.free_block_ids:
                            if not _preempt_newest(seq):
                                break
                        if not bm.free_block_ids:
                            # Nothing left to evict: this sequence is the only
                            # one in flight, so preempting it would livelock.
                            _record_preemption(seq)
                            _uncommit(seq)
                            bm.deallocate(seq)
                            bm.deallocate_cross(seq)
                            self.encoder_cache.pop(id(seq), None)
                            seq.preempt()
                            waiting.appendleft(seq)
                            continue
                        seq.block_table.append(bm.free_block_ids.popleft())
                    decode_seqs.append(seq)
                running = new_running
                token_budget -= len(decode_seqs)

            if (is_bitnet or _prefill_priority) and (waiting or prefilling):
                # BitNet uses bf16 fake-quant weights for prefill and int2
                # weights for decode. Since BitLinear dispatches per forward,
                # mixed prefill+decode batches would run decode tokens through
                # the prefill path and break alignment. Try prefill first; if
                # no prefill work can be scheduled this step, fall back to
                # pure decode below to avoid scheduler empty-spin/deadlock.
                pass
            else:
                _schedule_decode_tokens()

            # --- 2. Continue prefilling seqs already mid-prefill ---
            prefill_seqs: list[Sequence] = []
            prefill_chunk_sizes: list[int] = []
            still_prefilling: deque[Sequence] = deque()
            while prefilling and token_budget > 0:
                seq = prefilling.popleft()
                remaining = seq.num_remaining_prefill
                chunk = min(remaining, token_budget)
                blocks_needed = seq.blocks_needed_for(chunk)
                if blocks_needed > 0:
                    if len(bm.free_block_ids) < blocks_needed:
                        still_prefilling.append(seq)
                        continue
                    bm.allocate_n(seq, blocks_needed)
                prefill_seqs.append(seq)
                prefill_chunk_sizes.append(chunk)
                token_budget -= chunk
            while prefilling:
                still_prefilling.append(prefilling.popleft())
            prefilling = still_prefilling

            # --- 3. Admit new seqs from waiting queue ---
            encoder_budget = self._max_encoder_tokens()
            merge_size = self._vision_merge_size()
            mm_admitted = 0
            _admitted = 0
            _admit_stop = None
            _stride = _admit_stride()
            # Note: check decode_seqs, not running -- _schedule_decode_tokens
            # has already drained running into decode_seqs by this point, so
            # ``not running`` is true every step and would disable the stride.
            _admit_ok = (_stride == 1 or (_step_index % _stride) == 0
                         or not (decode_seqs or prefilling))
            while waiting and token_budget > 0 and _admit_ok:
                seq = waiting[0]
                prompt_len = seq.num_prompt_tokens

                has_mm = self.is_qwen_vl and (
                    getattr(seq, 'pixel_values', None) is not None
                    or getattr(seq, 'video_pixel_values', None) is not None
                    or getattr(seq, 'input_audio_features', None) is not None)

                is_whisper_seq = (seq.encoder_features is not None)

                if is_whisper_seq:
                    # No chunked prefill for encoder-decoder (matching vLLM).
                    chunk = prompt_len
                    if chunk > token_budget:
                        break
                    # Check cross-attn block availability
                    cross_blocks_needed = getattr(self, 'cross_blocks_per_seq', 0)
                    cross_free = len(getattr(bm, 'cross_free_block_ids', []))
                    if cross_blocks_needed > 0 and cross_free < cross_blocks_needed:
                        break
                elif has_mm:
                    chunk = min(prompt_len, token_budget)
                    # Refusing to split a prompt that would have fit in a full
                    # step was tried here and reverted. It does what it says --
                    # under the stagger the remainder otherwise gets a forward
                    # pass of its own, so Qwen3-VL image went from 67 mixed steps
                    # averaging 8.2k prefill tokens to 36 averaging 15.2k, and
                    # total steps 673 -> 638 -- but the time simply moved from
                    # prefill into decode: throughput fell 16,689 -> 16,330 tok/s
                    # against the same reference. vLLM splits freely; so do we.
                    seq_encoder_tokens = _seq_encoder_tokens(seq, merge_size)
                    if mm_admitted and seq_encoder_tokens > encoder_budget:
                        _admit_stop = "encoder_budget"
                        break
                else:
                    chunk = min(prompt_len, token_budget)

                blocks_needed = (chunk + block_size - 1) // block_size
                free = len(bm.free_block_ids)
                if free < blocks_needed + watermark_blocks:
                    _admit_stop = "kv_blocks"
                    break
                # vLLM's ``full_sequence_must_fit`` gate: the request's
                # *current* length must fit in the free blocks, so a chunked
                # prefill cannot admit a long prompt on the strength of its
                # first chunk. Tokens not yet generated are not reserved --
                # see the matching comment in ``_prefill_blocked_by_capacity``.
                full_blocks = (min(prompt_len, self.max_model_len)
                               + block_size - 1) // block_size
                if free < full_blocks + watermark_blocks:
                    _admit_stop = "kv_full_seq"
                    break
                if (_committed[0]
                        and _committed[0] + _seq_final_blocks(seq)
                        > num_blocks - watermark_blocks
                        and _reserve_full()):
                    _admit_stop = "reserved"
                    break
                num_scheduled = len(prefill_seqs) + len(decode_seqs)
                if is_bitnet:
                    # BitNet may intentionally leave running decode seqs
                    # unscheduled while it tries to drain pure-prefill work.
                    # Count those active seqs against max_num_seqs so
                    # --kb-bsz=1 really means one in-flight request, not one
                    # prefill plus hidden running decode requests.
                    num_scheduled += len(running) + len(prefilling)
                if num_scheduled >= self.max_num_seqs:
                    _admit_stop = "max_num_seqs"
                    break
                waiting.popleft()
                _commit(seq)
                bm.allocate_n(seq, blocks_needed)
                if is_whisper_seq and cross_blocks_needed > 0:
                    for _ in range(cross_blocks_needed):
                        seq.cross_block_table.append(bm.cross_free_block_ids.popleft())
                seq.status = SeqStatus.PREFILLING
                prefill_seqs.append(seq)
                prefill_chunk_sizes.append(chunk)
                token_budget -= chunk
                _admitted += 1
                if has_mm:
                    encoder_budget -= seq_encoder_tokens
                    mm_admitted += 1

            if _step_profile_active and _admitted:
                # Why admission stopped: "starved" means the waiting queue ran
                # dry, i.e. preprocessing -- not KV capacity or the token
                # budget -- is what limits how many prompts enter per step.
                # Counted for every model, not just multimodal ones: gating this
                # on ``mm_admitted`` made the whole breakdown invisible on text
                # models, which is what hid gemma-4 stopping on "reserved".
                if _admit_stop is None and token_budget > 0:
                    _admit_stop = "starved"
                key = f"admit_stop_{_admit_stop or 'token_budget'}"
                step_profile[key] = step_profile.get(key, 0) + 1
                step_profile["admitted_total"] = (
                    step_profile.get("admitted_total", 0) + _admitted)
                step_profile["admit_steps"] = (
                    step_profile.get("admit_steps", 0) + 1)

            if ((is_bitnet or _prefill_priority)
                    and not prefill_seqs and not decode_seqs and running):
                _schedule_decode_tokens()

            if not decode_seqs and not prefill_seqs:
                if not _produce_done.is_set() and not waiting:
                    # Nothing admissible yet and the producer still owes us
                    # sequences: yield instead of spinning the scheduler.
                    time.sleep(0.0002)
                continue

            # =============================================================
            # EXECUTE: single forward pass
            # =============================================================
            n_pf = len(prefill_seqs)
            n_dc = len(decode_seqs)

            if _step_profile_active:
                _spt0 = time.perf_counter()
                _sp = step_profile
                has_mm_step = self.is_qwen_vl and any(
                    s.pixel_values is not None
                    or s.video_pixel_values is not None
                    or getattr(s, 'input_audio_features', None) is not None
                    for s in prefill_seqs
                )
                _sp_cat = None
                if n_pf == 0:
                    _sp["pure_decode"] += 1
                    _sp["decode_tokens"] += n_dc
                    _record_decode_batch(n_dc)
                    _sp_cat = "decode_time"
                elif n_dc == 0 and has_mm_step:
                    _sp["pure_mm_prefill"] += 1
                    _sp["mm_prefill_tokens"] += sum(prefill_chunk_sizes)
                    _sp_cat = "mm_prefill_time"
                elif n_dc == 0:
                    _sp["pure_text_prefill"] += 1
                    _sp["text_prefill_tokens"] += sum(prefill_chunk_sizes)
                elif has_mm_step:
                    _sp["mixed_mm"] += 1
                    _sp["mixed_mm_pf_tokens"] += sum(prefill_chunk_sizes)
                    _sp["mixed_mm_dc_tokens"] += n_dc
                    _record_decode_batch(n_dc)
                    _sp_cat = "mixed_mm_time"
                else:
                    _sp["mixed_text"] += 1
                    _sp["mixed_text_pf_tokens"] += sum(prefill_chunk_sizes)
                    _sp["mixed_text_dc_tokens"] += n_dc

            # ----- Whisper: run encoder for new prefill seqs -----
            encoder_outputs = None
            encoder_seqs = []
            if self.is_whisper and n_pf > 0:
                encoder_seqs = [s for s in prefill_seqs
                                if s.encoder_features is not None and not s.encoder_computed]
                if encoder_seqs:
                    model = self.model_runner.model
                    dev = f"cuda:{self.model_runner.rank}"
                    # Run the encoder in chunks rather than stacking every
                    # admitted sequence into one forward. The encoder's cost is
                    # linear in the batch and nothing reserves memory for it:
                    # allocate_kv_cache hands the KV cache everything
                    # gpu_memory_utilization allows minus the (peak - current)
                    # transient it measured during warmup, and _warmup_whisper
                    # runs the encoder at batch 1. So the reserve was ~1/338 of
                    # the real peak, while a step admitting max_num_seqs=338
                    # sequences needs 338 x 1500 x 5120 x 2 = 4.84 GiB for the
                    # FFN intermediate alone and ~15 GiB live -- against the
                    # 18.1 GiB left over, which OOM'd in the encoder MLP on the
                    # full librispeech scenario.
                    #
                    # Chunking bounds the peak independently of max_num_seqs
                    # instead of shrinking the KV cache for every whisper run.
                    # 64 x 1500 = 96k encoder tokens per forward already
                    # saturates a B200, so this is not paying throughput for the
                    # headroom. The per-chunk output tensors stay alive either
                    # way (encoder_outputs holds unbind views of them), so what
                    # this actually frees is the transient FFN activation, which
                    # is the term that scales.
                    encoder_outputs = []
                    for i in range(0, len(encoder_seqs), _WHISPER_ENCODER_CHUNK):
                        chunk = encoder_seqs[i:i + _WHISPER_ENCODER_CHUNK]
                        features_batch = torch.stack([
                            s.encoder_features.to(
                                device=dev, dtype=self.model_runner.dtype,
                            ) for s in chunk
                        ], dim=0)
                        encoder_outputs.extend(
                            model.get_multimodal_embeddings(features_batch))
                        del features_batch
                    for seq, enc_out in zip(encoder_seqs, encoder_outputs):
                        seq.encoder_computed = True
                        seq.encoder_seq_len = enc_out.shape[0]

            if n_pf == 0 and use_greedy:
                # Pure decode with CUDA graphs (fast path)
                if self.is_whisper:
                    self.model_runner._set_cross_attn_context_decode(decode_seqs)
                if (
                    is_bitnet
                    and len(decode_seqs) == 1
                    and decode_seqs[0].ignore_eos
                    and decode_seqs[0].num_prompt_tokens >= 1024
                    and len(decode_seqs[0].generated_ids) <= 4
                    and self.model_runner.world_size == 1
                    and not self.model_runner.enforce_eager
                ):
                    decode_data = self.model_runner._prepare_decode_arrays(
                        decode_seqs)
                    gpu_result = self.model_runner._run_decode_greedy_eager(
                        *decode_data)
                else:
                    gpu_result = self.model_runner.call_decode_greedy(decode_seqs)
                if gpu_result is not None:
                    token_ids = gpu_result.tolist()
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            _finish_seq(seq)
                            finished_set.add(id(seq))
                        else:
                            running.append(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                else:
                    running.extend(decode_seqs)
            elif n_pf == 0:
                # Pure decode, non-greedy
                if self.is_whisper:
                    self.model_runner._set_cross_attn_context_decode(decode_seqs)
                result = self.model_runner.call("run", decode_seqs, False)
                if result is not None:
                    if collect_logits:
                        for i, seq in enumerate(decode_seqs):
                            seq_logits[id(seq)].append(result[i:i+1].cpu())
                    token_ids = self._sample(result, sp_list[0])
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            _finish_seq(seq)
                            finished_set.add(id(seq))
                        else:
                            running.append(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                else:
                    running.extend(decode_seqs)
            elif n_dc == 0:
                # Pure prefill (no running decode seqs)
                has_mm = self.is_qwen_vl and any(
                    s.pixel_values is not None
                    or s.video_pixel_values is not None
                    or getattr(s, 'input_audio_features', None) is not None
                    for s in prefill_seqs
                )
                if has_mm:
                    if _step_profile_active:
                        _vt0 = time.perf_counter()
                    vis_cache_map, _ = self._dispatch_vision_encoder(prefill_seqs, prefill_chunk_sizes)
                    if _step_profile_active:
                        torch.cuda.synchronize()
                        step_profile["vis_time"] = (
                            step_profile.get("vis_time", 0.0)
                            + time.perf_counter() - _vt0)
                        step_profile["vis_calls"] = (
                            step_profile.get("vis_calls", 0) + 1)
                    _stripped_pf = self.model_runner._strip_mm_tensors(prefill_seqs)
                    logits = self.model_runner.call(
                        "_run_mm_lm", _stripped_pf, prefill_chunk_sizes, [],
                        vis_cache_map,
                    )
                    if _step_profile_active:
                        torch.cuda.synchronize()
                        step_profile["mm_prefill_time"] += time.perf_counter() - _spt0
                elif self.is_whisper and encoder_outputs:
                    input_ids_t, positions_t = self.model_runner.prepare_mixed_batch(
                        prefill_seqs, prefill_chunk_sizes, [],
                    )
                    self.model_runner._set_cross_attn_context_prefill(
                        prefill_seqs, prefill_chunk_sizes, encoder_seqs)
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        encoder_outputs=encoder_outputs,
                    )
                    reset_context()
                else:
                    logits = self.model_runner.call(
                        "run_mixed", prefill_seqs, prefill_chunk_sizes, [],
                        use_greedy and not collect_logits,
                    )
                if logits is not None:
                    self._process_prefill_logits(
                        logits, prefill_seqs, prefill_chunk_sizes,
                        sp_list[0], eos, collect_logits, seq_logits,
                        running, prefilling, bm, block_size,
                        finish_seq=_finish_seq,
                    )
            else:
                # Mixed batch: prefill + decode together
                has_mm = self.is_qwen_vl and any(
                    s.pixel_values is not None
                    or s.video_pixel_values is not None
                    or getattr(s, 'input_audio_features', None) is not None
                    for s in prefill_seqs
                )
                if has_mm:
                    if _step_profile_active:
                        _vt0 = time.perf_counter()
                    vis_cache_map, _ = self._dispatch_vision_encoder(prefill_seqs, prefill_chunk_sizes)
                    if _step_profile_active:
                        torch.cuda.synchronize()
                        step_profile["vis_time"] = (
                            step_profile.get("vis_time", 0.0)
                            + time.perf_counter() - _vt0)
                        step_profile["vis_calls"] = (
                            step_profile.get("vis_calls", 0) + 1)
                    _stripped_pf = self.model_runner._strip_mm_tensors(prefill_seqs)
                    _stripped_dc = self.model_runner._strip_mm_tensors(decode_seqs)
                    logits = self.model_runner.call(
                        "_run_mm_lm", _stripped_pf, prefill_chunk_sizes,
                        _stripped_dc, vis_cache_map,
                    )
                    if _step_profile_active:
                        torch.cuda.synchronize()
                        step_profile["mixed_mm_time"] += time.perf_counter() - _spt0
                elif self.is_whisper:
                    input_ids_t, positions_t = self.model_runner.prepare_mixed_batch(
                        prefill_seqs, prefill_chunk_sizes, decode_seqs,
                    )
                    self.model_runner._set_cross_attn_context_mixed(
                        prefill_seqs, prefill_chunk_sizes, decode_seqs,
                        encoder_seqs)
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        encoder_outputs=encoder_outputs if encoder_outputs else [],
                    )
                    reset_context()
                else:
                    logits = self.model_runner.call(
                        "run_mixed", prefill_seqs, prefill_chunk_sizes, decode_seqs,
                        use_greedy and not collect_logits,
                    )
                if logits is not None:
                    pf_logits = logits[:n_pf]
                    dc_logits = logits[n_pf:]

                    self._process_prefill_logits(
                        pf_logits, prefill_seqs, prefill_chunk_sizes,
                        sp_list[0], eos, collect_logits, seq_logits,
                        running, prefilling, bm, block_size,
                        finish_seq=_finish_seq,
                    )

                    if collect_logits:
                        for i, seq in enumerate(decode_seqs):
                            seq_logits[id(seq)].append(dc_logits[i:i+1].cpu())
                    dc_token_ids = self._sample(dc_logits, sp_list[0])
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, dc_token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            _finish_seq(seq)
                            finished_set.add(id(seq))
                        else:
                            running.append(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                else:
                    running.extend(decode_seqs)

        _loop_t0_elapsed = time.perf_counter() - _loop_t0
        if _producer is not None:
            _producer.join()
        if _produce_error:
            raise _produce_error[0]

        if pbar is not None:
            _flush_pbar()
            pbar.close()

        if _step_profile_active:
            print(f"  engine_loop_total: {_loop_t0_elapsed:.3f}s")
            sp = step_profile
            fd = sp.get("fast_decode", 0)
            fdt = sp.get("fast_decode_tokens", 0)
            slow = (sp["pure_decode"] + sp["pure_mm_prefill"]
                    + sp["pure_text_prefill"] + sp["mixed_mm"] + sp["mixed_text"])
            total = fd + slow
            print(f"\n[Step Profile] total_steps={total} (fast_decode={fd}, scheduled={slow})")
            print(f"  fast_decode:       {fd:6d} steps, {fdt:10d} tokens, {sp['decode_time']:.3f}s")
            print(f"  pure_decode:       {sp['pure_decode']:6d} steps, {sp['decode_tokens']:10d} tokens")
            print(f"  pure_text_prefill: {sp['pure_text_prefill']:6d} steps, {sp['text_prefill_tokens']:10d} tokens")
            print(f"  pure_mm_prefill:   {sp['pure_mm_prefill']:6d} steps, {sp['mm_prefill_tokens']:10d} tokens, {sp['mm_prefill_time']:.3f}s")
            print(f"  mixed_text:        {sp['mixed_text']:6d} steps, pf={sp['mixed_text_pf_tokens']:10d} dc={sp['mixed_text_dc_tokens']:10d}")
            print(f"  mixed_mm:          {sp['mixed_mm']:6d} steps, pf={sp['mixed_mm_pf_tokens']:10d} dc={sp['mixed_mm_dc_tokens']:10d}, {sp['mixed_mm_time']:.3f}s")
            print(f"  preemptions:       {sp['preempted']:6d} seqs, "
                  f"{sp['preempted_tokens']:10d} decode tokens discarded")
            if sp.get("kv_peak_blocks"):
                print(f"  kv_peak:           {sp['kv_peak_blocks']} of "
                      f"{num_blocks} blocks "
                      f"({sp['kv_peak_blocks'] / num_blocks:.1%}), "
                      f"{num_prompts} seqs")
            if sp.get("dc_hist"):
                cap = max(1, self.max_num_seqs)
                bands = " ".join(
                    f"{i*cap//8}-{(i+1)*cap//8}:{c}"
                    for i, c in enumerate(sp["dc_hist"]) if c
                )
                tot_dc = (sp.get("fast_decode_tokens", 0) + sp["decode_tokens"]
                          + sp["mixed_mm_dc_tokens"] + sp["mixed_text_dc_tokens"])
                nsteps = sum(sp["dc_hist"])
                print(f"  decode_batch:      avg={tot_dc / max(1, nsteps):.0f} "
                      f"max={sp.get('dc_max', 0)} over {nsteps} steps "
                      f"(cap {cap}); bands {bands}")
            if sp.get("vis_calls"):
                print(f"  vision_encoder:    {sp['vis_calls']:6d} calls, "
                      f"{sp['vis_time']:.3f}s "
                      f"({sp['vis_time'] / sp['vis_calls'] * 1e3:.1f}ms/call)")
            if self.is_qwen_vl:
                # How much runtime headroom the multimodal steps actually used,
                # against the reserve carved out of the KV cache for them.
                mr = self.model_runner
                base = getattr(mr, "_post_capture_alloc", 0)
                reserve = getattr(mr, "_mm_reserve_bytes", 0)
                used = max(0, torch.cuda.max_memory_allocated() - base)
                msg = f"  mm_runtime_peak:   {used / 2**30:.1f}G used"
                if reserve:
                    msg += (f" of {reserve / 2**30:.1f}G reserved "
                            f"({used / reserve:.0%})")
                print(msg)
            if sp.get("admit_steps"):
                stops = " ".join(
                    f"{k[len('admit_stop_'):]}={v}"
                    for k, v in sorted(sp.items())
                    if k.startswith("admit_stop_")
                )
                print(f"  admission:         {sp['admitted_total']} seqs over "
                      f"{sp['admit_steps']} steps "
                      f"({sp['admitted_total'] / sp['admit_steps']:.1f}/step); "
                      f"stopped on: {stops}")
            fp = getattr(self, "_fast_path_profile", None)
            if fp is not None and fp["n"]:
                print(
                    "  fast_path_detail: "
                    f"prep={fp['prep']:.3f}s "
                    f"gpu_dispatch={fp['gpu']:.3f}s "
                    f"d2h_wait={fp['tolist']:.3f}s "
                    f"post={fp['post']:.3f}s "
                    f"steps={fp['n']}"
                )
                self._fast_path_profile = {
                    "prep": 0., "gpu": 0., "tolist": 0., "post": 0., "n": 0,
                }

        # Return in original order
        _detok_t0 = time.perf_counter()
        outputs = [
            GenerationOutput(
                prompt=(prompts[i] if isinstance(prompts[i], str) else ""),
                generated_text=(
                    self.tokenizer.decode(
                        all_seqs[i].generated_ids, skip_special_tokens=True,
                    )
                    if decode_text else ""
                ),
                token_ids=all_seqs[i].generated_ids,
                logits_history=(
                    seq_logits.get(id(all_seqs[i])) if collect_logits else None
                ),
            )
            for i in range(len(prompts))
        ]
        if _step_profile_active:
            # Detokenization is a serial post-pass here, inside the caller's
            # timed region; vLLM detokenizes incrementally in its frontend while
            # the engine core keeps stepping, so it pays ~none of this.
            print(f"  output_detokenize: {time.perf_counter() - _detok_t0:.3f}s "
                  f"for {len(outputs)} seqs")
        return outputs

    def _process_prefill_logits(
        self, logits, prefill_seqs, prefill_chunk_sizes,
        sp, eos, collect_logits, seq_logits,
        running, prefilling, bm, block_size,
        finish_seq=None,
    ):
        """Handle output from prefill sequences after a forward pass.

        For sequences whose prefill is complete, sample the first decode
        token. For sequences still mid-prefill, update num_computed_tokens
        and move them to the prefilling queue.
        """
        # Separate seqs into "done prefilling" vs "still prefilling"
        sample_seqs = []
        sample_logits = []
        for i, (seq, chunk) in enumerate(zip(prefill_seqs, prefill_chunk_sizes)):
            seq.num_computed_tokens += chunk
            if seq.num_remaining_prefill == 0:
                # Prefill complete — sample first decode token
                sample_seqs.append(seq)
                sample_logits.append(logits[i:i+1])
            else:
                # More prefill chunks needed
                prefilling.append(seq)

        if not sample_seqs:
            return

        sample_logits_t = torch.cat(sample_logits, dim=0)
        if collect_logits:
            for i, seq in enumerate(sample_seqs):
                seq_logits[id(seq)].append(sample_logits_t[i:i+1].cpu())
        token_ids = self._sample(sample_logits_t, sp)
        for seq, tid in zip(sample_seqs, token_ids):
            seq.append_token(tid)
            seq.status = SeqStatus.RUNNING
            done = len(seq.generated_ids) >= seq.max_tokens
            if not seq.ignore_eos:
                done = done or tid == eos
            if done:
                if finish_seq is not None:
                    finish_seq(seq)
                else:
                    seq.status = SeqStatus.FINISHED
                    bm.deallocate(seq)
            else:
                running.append(seq)
