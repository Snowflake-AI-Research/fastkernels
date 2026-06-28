"""TP-aware embedding and LM head (L2 operators)."""


from __future__ import annotations


# Inlined from infra/context.py
import enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


class CUDAGraphMode(enum.IntEnum):
    """Runtime mode for CUDA graph dispatch (mirrors vLLM CUDAGraphMode)."""
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class AttnBackendConfig:
    """Selects attention backend and associated KV cache parameters.

    Blackwell (sm_100+) uses TRTLLM-gen kernels via FlashInfer (HND layout,
    block_size=16).  Hopper and below use flash_attn (NHD layout,
    block_size=256).  Auto-detection picks the optimal backend for the
    current GPU.
    """
    backend: str = "flash_attn"
    block_size: int = 256
    kv_layout: str = "NHD"

    @classmethod
    def auto_detect(cls) -> "AttnBackendConfig":
        if not torch.cuda.is_available():
            return cls()
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 10:
            try:
                from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: F401
                return cls(backend="trtllm", block_size=16, kv_layout="HND")
            except ImportError:
                pass
        return cls()

    @property
    def use_trtllm(self) -> bool:
        return self.backend == "trtllm"


_ATTN_BACKEND_CONFIG: AttnBackendConfig | None = None


def get_attn_backend_config() -> AttnBackendConfig:
    global _ATTN_BACKEND_CONFIG
    if _ATTN_BACKEND_CONFIG is None:
        _ATTN_BACKEND_CONFIG = AttnBackendConfig.auto_detect()
    return _ATTN_BACKEND_CONFIG


def set_attn_backend_config(config: AttnBackendConfig) -> None:
    global _ATTN_BACKEND_CONFIG
    _ATTN_BACKEND_CONFIG = config


@dataclass
class ChunkedContextMetadata:
    """Metadata for chunked prefill context processing (MLA).

    When a prefill request has prior computed tokens in the KV cache,
    those tokens must be gathered and attended to in chunks to bound
    workspace memory.  Matches vllm's MLACommonPrefillMetadata.ChunkedContextMetadata.
    """
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    workspace: torch.Tensor
    token_to_seq: torch.Tensor
    chunk_total_token: list[int]


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int = 0

    # Chunked prefill: mixed batch with both prefill and decode tokens.
    # Token layout: [prefill_tokens... | decode_tokens...]
    is_mixed: bool = False
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_seqs: int = 0

    # Prefill-specific metadata (indexed over prefill seqs only)
    prefill_cu_seqlens_q: torch.Tensor | None = None
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_max_seqlen_q: int = 0
    prefill_max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None

    # Decode-specific metadata (indexed over decode seqs only)
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    decode_max_context_len: int = 0

    # Flat indices into concatenated input for extracting one logit per seq
    logit_indices: torch.Tensor | None = None

    # MLA chunked prefill context (for requests with prior computed tokens)
    chunked_context: ChunkedContextMetadata | None = None

    # Per-token request ID mapping (for sparse indexer index conversion)
    req_id_per_token: torch.Tensor | None = None

    # Cross-attention metadata (encoder-decoder models like Whisper)
    # Slot mapping for writing encoder K/V to paged cache
    cross_slot_mapping: torch.Tensor | None = None
    # Prefill: cu_seqlens for decoder Q and encoder K
    cross_cu_seqlens_q: torch.Tensor | None = None
    cross_cu_seqlens_k: torch.Tensor | None = None
    cross_max_seqlen_q: int = 0
    cross_max_seqlen_k: int = 0
    cross_block_tables: torch.Tensor | None = None
    # Decode: context lens = encoder sequence lengths per request
    cross_context_lens: torch.Tensor | None = None
    cross_max_context_len: int = 0

    # --- Compilation / CUDA-graph fields (mirror vLLM ForwardContext) ---
    # Maps layer prefix -> live nn.Module for custom-op runtime lookup.
    no_compile_layers: dict[str, "nn.Module"] = field(default_factory=dict)
    # Runtime mode for CUDAGraphWrapper dispatch.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    # Batch size key used by CUDAGraphWrapper for per-shape graph caching.
    batch_size_for_graph: int = 0

    # --- Mamba / SSM fields (mirror vLLM ForwardContext.attn_metadata
    # for Mamba layers).  ``mamba_state`` owns the global conv/ssm state
    # tensors; ``mamba_metadata`` is a per-batch dataclass (Mamba2Metadata
    # or MambaMetadata) carrying state slot indices and prefill/decode
    # metadata read by every Mamba mixer in its forward pass.
    mamba_state: object = None
    mamba_metadata: object = None


# Global module registry populated once at model init; copied into each
# Context so compiled custom ops can resolve their target modules.
_STATIC_NO_COMPILE_LAYERS: dict[str, "nn.Module"] = {}


def register_no_compile_layers(layers: dict[str, "nn.Module"]) -> None:
    """Register attention/MoE modules for custom-op lookup during compiled
    execution.  Called once after model construction."""
    _STATIC_NO_COMPILE_LAYERS.update(layers)


def get_no_compile_layers() -> dict[str, "nn.Module"]:
    return _STATIC_NO_COMPILE_LAYERS


def auto_register_no_compile_layers(model: "nn.Module") -> None:
    """Walk *model* and register every MoE and Attention sub-module by its
    fully-qualified prefix so custom ops can find them at runtime.

    Recognized types (by class name to avoid circular imports):
      - ``Qwen3MoE``, ``MixtralMoE``, ``GptOssMoE``, ``DeepSeekMoE`` (MoE blocks)
      - ``Attention``, ``MLAAttention``, ``SparseAttnIndexer``       (attention impls)
      - ``Mamba2Mixer``                                              (Mamba2 compile boundary)

    Also sets ``_layer_name`` on each module so it knows its own key.
    ``_use_custom_op`` remains ``False`` until compilation is enabled.
    """
    _TARGET_NAMES = {
        "Qwen3MoE", "MixtralMoE", "GptOssMoE", "DeepSeekMoE",
        "Attention", "MLAAttention", "SparseAttnIndexer",
        "Mamba2Mixer",
    }
    layers: dict[str, "nn.Module"] = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ in _TARGET_NAMES:
            layers[name] = mod
            mod._layer_name = name  # type: ignore[attr-defined]
    register_no_compile_layers(layers)


def enable_custom_ops() -> None:
    """Switch all registered no-compile layers to dispatch through custom ops.
    Called once after torch.compile is applied to the model."""
    for mod in _STATIC_NO_COMPILE_LAYERS.values():
        mod._use_custom_op = True  # type: ignore[attr-defined]


def disable_custom_ops() -> None:
    """Revert to eager dispatch (used for testing/fallback)."""
    for mod in _STATIC_NO_COMPILE_LAYERS.values():
        mod._use_custom_op = False  # type: ignore[attr-defined]


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None,
                max_context_len=0, chunked_context=None,
                req_id_per_token=None):
    global _CONTEXT
    # For pure-decode batches (``is_prefill=False`` with no mixed fields),
    # mirror the generic ``context_lens`` / ``block_tables`` / ``max_context_len``
    # into the decode-specific fields so that DSA indexer and other
    # decode-specialised paths (which consult ``decode_context_lens`` /
    # ``decode_block_tables`` — matching vLLM's FlashInfer metadata) can
    # find them.  Without this, ``SparseAttnIndexer._decode_topk`` would
    # early-return all -1 indices and attention would degenerate.
    dc_cl = context_lens if not is_prefill else None
    dc_bt = block_tables if not is_prefill else None
    dc_max = max_context_len if not is_prefill else 0
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables, max_context_len,
                       chunked_context=chunked_context,
                       req_id_per_token=req_id_per_token,
                       no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
                       decode_context_lens=dc_cl,
                       decode_block_tables=dc_bt,
                       decode_max_context_len=dc_max)


def set_mixed_context(
    slot_mapping,
    num_prefill_tokens, num_decode_tokens, num_prefill_seqs,
    prefill_cu_seqlens_q, prefill_cu_seqlens_k,
    prefill_max_seqlen_q, prefill_max_seqlen_k,
    prefill_block_tables,
    decode_context_lens, decode_block_tables, decode_max_context_len,
    logit_indices,
    chunked_context=None,
    req_id_per_token=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=True, is_mixed=True,
        slot_mapping=slot_mapping,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_seqs=num_prefill_seqs,
        prefill_cu_seqlens_q=prefill_cu_seqlens_q,
        prefill_cu_seqlens_k=prefill_cu_seqlens_k,
        prefill_max_seqlen_q=prefill_max_seqlen_q,
        prefill_max_seqlen_k=prefill_max_seqlen_k,
        prefill_block_tables=prefill_block_tables,
        decode_context_lens=decode_context_lens,
        decode_block_tables=decode_block_tables,
        decode_max_context_len=decode_max_context_len,
        logit_indices=logit_indices,
        chunked_context=chunked_context,
        req_id_per_token=req_id_per_token,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
    )


def set_mamba_context(
    is_prefill: bool,
    mamba_state,
    mamba_metadata,
):
    """Install per-batch Mamba state + metadata in the global Context.

    Used by ``ModelRunner.run_mamba`` -- mirrors how the attention path
    uses ``set_context`` / ``set_mixed_context``.  Mamba mixers read
    ``ctx.mamba_state`` and ``ctx.mamba_metadata`` in their forward.
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        mamba_state=mamba_state,
        mamba_metadata=mamba_metadata,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context(no_compile_layers=_STATIC_NO_COMPILE_LAYERS)


@contextmanager
def set_forward_context(
    is_prefill: bool = False,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_size_for_graph: int = 0,
    **ctx_kwargs,
):
    """Context manager that sets both KV-cache metadata and compile/graph
    fields for the duration of a forward pass.

    ``no_compile_layers`` is always populated from the global registry so
    custom ops can resolve modules without the caller threading it through.
    """
    global _CONTEXT
    prev = _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_size_for_graph=batch_size_for_graph,
        **ctx_kwargs,
    )
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = prev


# Inlined from infra/tp.py
import torch.distributed as dist


def _tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def _tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# Inlined from tasks/reference/L1/linear.py
import torch.nn as nn
import torch.nn.functional as F


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


# Inlined from tasks/reference/L1/embedding.py


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)


# Inlined from tasks/reference/L1/allreduce.py
from contextlib import nullcontext
from typing import Optional

from torch.distributed import ProcessGroup


_CUSTOM_AR: Optional["CustomAllreduce"] = None


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


class AllReduce(nn.Module):
    def forward(self, tensor):
        dist.all_reduce(tensor)
        return tensor


class CustomAllreduce:
    """Compatibility shim for callers expecting the baseline custom AR API."""

    disabled = True

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        del group, device, max_size

    def capture(self):
        return nullcontext()

    def custom_all_reduce(self, input: torch.Tensor) -> None:
        del input
        return None

    def close(self) -> None:
        pass

__all__ = ["AllReduce", "CustomAllreduce", "get_custom_ar", "set_custom_ar"]


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__()
        tp, rank = _tp_size(), _tp_rank()
        assert num_embeddings % tp == 0
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size
        self.embedding_dim = embedding_dim
        self.per_partition = num_embeddings // tp
        self.vocab_start = self.per_partition * rank
        self.vocab_end = self.vocab_start + self.per_partition
        self.tp_size = tp
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.embedding_op = Embedding(self.per_partition, embedding_dim)
        self.embedding_op.emb.weight.weight_loader = self._weight_loader
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
        y = self.embedding_op(x)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            y = self.allreduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 bias: bool = False,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__(num_embeddings, embedding_dim,
                         params_dtype=params_dtype,
                         org_num_embeddings=org_num_embeddings,
                         padding_size=padding_size)
        self.linear_op = Matmul()

    def project(self, x):
        """Linear projection only (no gather). Used inside CUDA graph."""
        ctx = get_context()
        if ctx.is_mixed:
            x = x[ctx.logit_indices].contiguous()
        elif ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return self.linear_op(x, self.embedding_op.emb.weight)

    def gather_logits(self, logits):
        """Gather partial logits from all ranks. Used outside CUDA graph."""
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if _tp_rank() == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if _tp_rank() == 0 else logits
        return logits

    def gather_greedy(self, logits):
        """Fast path for greedy: local argmax + small allgather.

        Instead of gathering full vocab logits (~31MB/rank), gather only
        the (max_val, max_idx) per sequence (~2KB/rank).
        Returns token IDs directly on rank 0, None on other ranks.
        """
        if self.tp_size <= 1:
            return None

        rank = _tp_rank()
        local_max_vals, local_max_idxs = logits.max(dim=-1)
        local_max_idxs = local_max_idxs + self.vocab_start

        info = torch.stack([local_max_vals, local_max_idxs.float()], dim=-1)
        gathered = [torch.empty_like(info) for _ in range(self.tp_size)]
        dist.all_gather(gathered, info)
        if rank == 0:
            all_info = torch.stack(gathered, dim=0)
            all_vals = all_info[:, :, 0]
            all_idxs = all_info[:, :, 1].long()
            best_rank = all_vals.argmax(dim=0)
            bs = logits.size(0)
            token_ids = all_idxs[best_rank, torch.arange(bs, device=logits.device)]
            return token_ids
        return None

    def forward(self, x):
        logits = self.project(x)
        return self.gather_logits(logits)
