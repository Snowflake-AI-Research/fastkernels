"""Standalone torch.compile integration adapted from vLLM's compilation stack.

Consolidates custom op registration, CUDA graph capture/replay, Inductor
post-grad passes, and the piecewise compilation backend into a single module.

Compilation flow:

1. ``mark_dynamic`` marks batch dimensions as symbolic before the first
   ``torch.compile`` call.
2. Dynamo traces the model **once** (guards dropped via
   ``skip_all_guards_unsafe``).
3. ``FastKernelsBackend`` receives the FX graph, splits it at attention
   custom-op boundaries.
4. Each non-splitting subgraph is compiled **once** with ``compile_fx``
   using fake/symbolic args extracted from graph placeholder metadata.
5. Deduplication: structurally identical subgraphs (repeated transformer
   layers) reuse the same compiled artifact via ``autograd_cache_key``.
6. Each compiled subgraph is wrapped with ``CUDAGraphWrapper`` for
   per-batch-size CUDA graph capture/replay during decode.
7. At runtime, no recompilation occurs — Dynamo guards are dropped and
   the compiled subgraphs handle any batch size.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import operator
import os
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from typing import Any
from unittest.mock import patch

import torch
import torch.fx as fx
import torch._inductor.custom_graph_pass

from .context import CUDAGraphMode, enable_custom_ops, get_context, get_no_compile_layers

logger = logging.getLogger(__name__)


# ===================================================================
# Custom op registrations for torch.compile boundaries
# ===================================================================
#
# Registers opaque custom ops for attention so that torch.compile
# (Inductor) does not trace into paged-KV attention kernels.  At
# runtime, the ops look up the actual nn.Module from the global
# ``no_compile_layers`` registry and call its implementation.
#
# Matching vLLM's default for Qwen3-VL-235B-FP8: splitting_ops
# contains only attention ops.  MoE is NOT a splitting op — the MoE
# forward is transparent to Inductor (it appears as opaque nodes within
# a compiled piece, not as a graph boundary).  This lets Inductor
# optimize the code around MoE (norms, linears) within the same
# compiled subgraph.
#
# MoE custom ops are still registered (for use when MoE needs to be
# opaque, e.g. expert parallelism), but they are not in SPLITTING_OPS
# by default.

SPLITTING_OPS: list[str] = [
    "fastkernels::unified_attention",
    "fastkernels::whisper_cross_attention",
    "fastkernels::mamba2_conv_ssm_forward",
    "fastkernels::unified_mla_attention",
    "fastkernels::sparse_attn_indexer",
    "fastkernels::kda_attention",
]


def _moe_forward_impl(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(hidden_states)


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _gemma4_moe_forward_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(hidden_states, router_logits)


def _gemma4_moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _unified_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(query, key, value)


def _unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query)


def _whisper_cross_attention_impl(
    query: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(query)


def _whisper_cross_attention_fake(
    query: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    # ``query`` arrives as [num_tokens, num_heads, head_dim]; the op returns the
    # flattened [num_tokens, num_heads * head_dim] that out_proj consumes.
    return query.new_empty((query.shape[0], query.shape[1] * query.shape[2]))


def _mamba2_conv_ssm_forward_impl(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_no_compile_layers()[layer_name]
    layer.conv_ssm_forward(projected_states, output)


def _mamba2_conv_ssm_forward_fake(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


def _unified_mla_attention_impl(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    topk_indices: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(q, kv_c_normed, k_pe, topk_indices)


def _unified_mla_attention_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    topk_indices: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    # Output shape is (N, num_heads * v_head_dim) where N is the (possibly
    # symbolic) batch dim of ``q``. Using ``q.new_empty`` propagates the
    # symbolic dim so torch.compile can keep the batch dim dynamic.
    return q.new_empty((q.shape[0], layer.num_heads * layer.v_head_dim))


def _sparse_attn_indexer_impl(
    hidden_states: torch.Tensor,
    q_latent: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(hidden_states, q_latent, positions)


def _sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    q_latent: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    M = hidden_states.shape[0]
    return torch.empty(
        (M, layer.topk_tokens), dtype=torch.int32, device=hidden_states.device,
    )


def _custom_all_reduce_impl(t: torch.Tensor) -> torch.Tensor:
    """All-reduce that survives torch.compile, preferring the custom IPC kernel.

    ``AllReduce.forward`` previously took ``dist.all_reduce`` unconditionally when
    ``torch.compiler.is_compiling()``, because the custom all-reduce does Python-side
    IPC pointer work that a compiled graph cannot trace. That made every collective
    in the compiled decode graph go through NCCL, at one-token message sizes (~5.7 KB
    for gpt-oss) where NCCL is far off its efficient range. Measured effect on
    gpt-oss-120b decode: tp=1 2.527 ms/step, tp=2 3.026 ms/step -- i.e. adding a
    second GPU made decode *slower*, where vLLM goes 2.789 -> 2.291.

    Registering it as a custom op is how vLLM handles the same problem: inductor
    treats the call as opaque, so the IPC work happens at runtime instead of being
    traced. Falls back to NCCL when the buffer does not qualify (too large,
    misaligned, non-contiguous) exactly as the eager path does.
    """
    from ..tasks.baseline.L1.allreduce import get_custom_ar

    ar = get_custom_ar()
    if ar is not None:
        out = ar.custom_all_reduce(t)
        if out is not None:
            return out
    out = t.clone()
    torch.distributed.all_reduce(out)
    return out


def _custom_all_reduce_fake(t: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(t)


def _fused_ar_add_rmsnorm_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One FlashInfer kernel for all-reduce + residual-add + RMSNorm.

    See ``L1.flashinfer_allreduce_fusion`` for why this is a single kernel and
    what it falls back to when the batch outgrows the workspace.
    """
    from ..tasks.baseline.L1.flashinfer_allreduce_fusion import (
        fused_allreduce_add_rmsnorm,
    )

    return fused_allreduce_add_rmsnorm(x, residual, weight, eps)


def _fused_ar_add_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


def _fused_ar_rmsnorm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused all-reduce + RMSNorm with no residual (layer 0's input_layernorm).

    Returns ``(normed, all_reduced)``: layer 0 carries the un-normalised
    all-reduce result forward as the residual, so both outputs are live.
    """
    from ..tasks.baseline.L1.flashinfer_allreduce_fusion import (
        fused_allreduce_rmsnorm,
    )

    return fused_allreduce_rmsnorm(x, weight, eps)


def _fused_ar_rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


def _kda_attention_impl(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_no_compile_layers()[layer_name]
    layer.forward_impl(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        raw_g=raw_g,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def _kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


_registered = False


def ensure_custom_ops_registered() -> None:
    """Register the custom ops with torch.library (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    lib = torch.library.Library("fastkernels", "DEF")

    lib.define(
        "moe_forward(Tensor hidden_states, str layer_name) -> Tensor"
    )
    lib.impl("moe_forward", _moe_forward_impl, "CUDA")
    lib.impl("moe_forward", _moe_forward_impl, "CPU")

    abstract_lib = torch.library.Library("fastkernels", "IMPL", "Meta")
    abstract_lib.impl("moe_forward", _moe_forward_fake)

    lib.define(
        "gemma4_moe_forward(Tensor hidden_states, Tensor router_logits, "
        "str layer_name) -> Tensor"
    )
    lib.impl("gemma4_moe_forward", _gemma4_moe_forward_impl, "CUDA")
    lib.impl("gemma4_moe_forward", _gemma4_moe_forward_impl, "CPU")
    abstract_lib.impl("gemma4_moe_forward", _gemma4_moe_forward_fake)

    lib.define(
        "unified_attention(Tensor query, Tensor key, Tensor value, "
        "str layer_name) -> Tensor"
    )
    lib.impl("unified_attention", _unified_attention_impl, "CUDA")
    lib.impl("unified_attention", _unified_attention_impl, "CPU")
    abstract_lib.impl("unified_attention", _unified_attention_fake)

    lib.define(
        "whisper_cross_attention(Tensor query, str layer_name) -> Tensor"
    )
    lib.impl("whisper_cross_attention", _whisper_cross_attention_impl, "CUDA")
    lib.impl("whisper_cross_attention", _whisper_cross_attention_impl, "CPU")
    abstract_lib.impl("whisper_cross_attention", _whisper_cross_attention_fake)

    lib.define(
        "mamba2_conv_ssm_forward(Tensor projected_states, Tensor(a!) output, "
        "str layer_name) -> ()"
    )
    lib.impl("mamba2_conv_ssm_forward", _mamba2_conv_ssm_forward_impl, "CUDA")
    lib.impl("mamba2_conv_ssm_forward", _mamba2_conv_ssm_forward_impl, "CPU")
    abstract_lib.impl("mamba2_conv_ssm_forward", _mamba2_conv_ssm_forward_fake)

    lib.define(
        "unified_mla_attention(Tensor q, Tensor kv_c_normed, Tensor k_pe, "
        "Tensor? topk_indices, str layer_name) -> Tensor"
    )
    lib.impl("unified_mla_attention", _unified_mla_attention_impl, "CUDA")
    lib.impl("unified_mla_attention", _unified_mla_attention_impl, "CPU")
    abstract_lib.impl("unified_mla_attention", _unified_mla_attention_fake)

    lib.define(
        "sparse_attn_indexer(Tensor hidden_states, Tensor q_latent, "
        "Tensor positions, str layer_name) -> Tensor"
    )
    lib.impl("sparse_attn_indexer", _sparse_attn_indexer_impl, "CUDA")
    lib.impl("sparse_attn_indexer", _sparse_attn_indexer_impl, "CPU")
    abstract_lib.impl("sparse_attn_indexer", _sparse_attn_indexer_fake)

    # Kimi Delta Attention (Kimi-Linear's linear-attention layers).  Mirrors
    # vLLM's ``vllm::kda_attention`` splitting op: the recurrence writes into
    # a preallocated ``core_attn_out`` and returns nothing.
    lib.define("custom_all_reduce(Tensor t) -> Tensor")
    lib.impl("custom_all_reduce", _custom_all_reduce_impl, "CUDA")
    abstract_lib.impl("custom_all_reduce", _custom_all_reduce_fake)

    # Fused all-reduce + residual-add + RMSNorm (FlashInfer). Opaque on purpose:
    # the workspace lookup and the token-count fallback are Python-side work that
    # a traced graph cannot express. ``AllReduceFusedAddRMSNormPass`` rewrites the
    # unfused triple into this op post-grad.
    lib.define(
        "fused_allreduce_add_rmsnorm(Tensor x, Tensor residual, "
        "Tensor weight, float eps) -> (Tensor, Tensor)"
    )
    lib.impl("fused_allreduce_add_rmsnorm", _fused_ar_add_rmsnorm_impl, "CUDA")
    abstract_lib.impl("fused_allreduce_add_rmsnorm", _fused_ar_add_rmsnorm_fake)

    # No-residual variant: layer 0's input_layernorm consumes the vocab-parallel
    # embedding's all-reduce and has no residual yet. One site per step, but the
    # first collective in the decode graph, so it absorbs the step's launch skew.
    # vLLM fuses it as ``AllReduceRMSNormPattern``.
    lib.define(
        "fused_allreduce_rmsnorm(Tensor x, Tensor weight, float eps) "
        "-> (Tensor, Tensor)"
    )
    lib.impl("fused_allreduce_rmsnorm", _fused_ar_rmsnorm_impl, "CUDA")
    abstract_lib.impl("fused_allreduce_rmsnorm", _fused_ar_rmsnorm_fake)

    lib.define(
        "kda_attention(Tensor q_proj_states, Tensor k_proj_states, "
        "Tensor v_proj_states, Tensor raw_g, Tensor beta, "
        "Tensor(a!) core_attn_out, str layer_name) -> ()"
    )
    lib.impl("kda_attention", _kda_attention_impl, "CUDA")
    lib.impl("kda_attention", _kda_attention_impl, "CPU")
    abstract_lib.impl("kda_attention", _kda_attention_fake)

    # Keep references alive for the lifetime of the process.
    ensure_custom_ops_registered._lib = lib  # type: ignore[attr-defined]
    ensure_custom_ops_registered._abstract_lib = abstract_lib  # type: ignore[attr-defined]


# ===================================================================
# CUDA graph capture and replay
# ===================================================================
#
# ``CUDAGraphWrapper`` wraps a callable (typically a compiled subgraph)
# and transparently captures / replays CUDA graphs keyed by batch size.
# Dispatch is controlled by ``Context.cudagraph_runtime_mode`` and the
# wrapper's own ``runtime_mode``:
#
#   - If context mode is ``NONE`` -> fall through (no graph).
#   - If context mode **does not match** ``self.runtime_mode`` -> fall through.
#   - If context mode **matches** -> capture (first time) or replay (cached).
#
# This mode-matching is critical for FULL_AND_PIECEWISE operation
# (vLLM's default for decode): piecewise wrappers only activate for
# PIECEWISE mode, while the engine's full-model graph only activates
# for FULL mode.

@dataclasses.dataclass
class _CUDAGraphEntry:
    cudagraph: torch.cuda.CUDAGraph
    output: torch.Tensor


class CUDAGraphWrapper(torch.nn.Module):
    """Wrap a callable with per-batch-size CUDA graph capture/replay.

    Inherits ``nn.Module`` so it can be assigned as a submodule of the
    FX split graph (``setattr(split_gm, submod_name, wrapper)``).

    Parameters
    ----------
    runnable : callable
        The function to capture (e.g. compiled model forward).
    runtime_mode : CUDAGraphMode
        Which mode this wrapper responds to.  A PIECEWISE wrapper only
        captures/replays when context says PIECEWISE; a FULL wrapper only
        when context says FULL.  Defaults to PIECEWISE (for subgraph wrapping).
    graph_pool : optional
        Shared ``torch.cuda.graph_pool_handle()`` for memory reuse.
    capture_context : optional
        Context manager to enter during capture (e.g. ``custom_ar.capture()``).
    """

    def __init__(
        self,
        runnable,
        runtime_mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE,
        graph_pool=None,
        capture_context=None,
    ):
        super().__init__()
        self.runnable = runnable
        self.runtime_mode = runtime_mode
        self.graph_pool = graph_pool
        self.capture_context = capture_context
        self._cache: dict[int, _CUDAGraphEntry] = {}

    @property
    def captured_sizes(self) -> list[int]:
        return sorted(self._cache.keys())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        ctx = get_context()
        mode = ctx.cudagraph_runtime_mode

        if mode == CUDAGraphMode.NONE or mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        bs = ctx.batch_size_for_graph
        entry = self._cache.get(bs)

        if entry is None:
            entry = self._capture(bs, *args, **kwargs)
            return entry.output

        entry.cudagraph.replay()
        return entry.output

    def _capture(self, bs: int, *args, **kwargs) -> _CUDAGraphEntry:
        logger.debug("Capturing %s CUDA graph for batch_size=%d",
                     self.runtime_mode.name, bs)
        cap_ctx = self.capture_context or nullcontext()

        with cap_ctx:
            self.runnable(*args, **kwargs)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                output = self.runnable(*args, **kwargs)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

        entry = _CUDAGraphEntry(cudagraph=graph, output=output)
        self._cache[bs] = entry
        torch.cuda.synchronize()
        return entry


# ===================================================================
# Inductor post-grad passes
# ===================================================================
#
# With the dual-dispatch architecture (forward_native for compiled
# path, forward_cuda for eager), Inductor sees pure PyTorch code for
# norms and activations.  This means Inductor's own fusion engine
# handles most optimizations automatically (e.g., fusing RMSNorm with
# adjacent quant, eliminating intermediate tensors).
#
# The pass manager provides:
#   - ``NoopEliminationPass`` -- removes identity reshape/view/expand/
#     slice ops that can block Inductor's automatic fusion
#   - Infrastructure for custom passes when needed

class InductorPass:
    """Base class for post-grad Inductor passes."""

    name: str = "base"

    def __call__(self, graph: torch.fx.Graph) -> None:
        raise NotImplementedError


class NoopEliminationPass(InductorPass):
    """Remove redundant reshape/view/expand/slice ops.

    These identity ops are inserted by functionalization and prevent
    Inductor's built-in fusion from matching adjacent ops.  Mirrors
    vLLM's ``NoOpEliminationPass``.
    """

    name = "noop_elimination"

    _IDENTITY_TARGETS = frozenset({
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor,
    })

    def __call__(self, graph: torch.fx.Graph) -> None:
        count = 0
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in self._IDENTITY_TARGETS:
                continue
            if self._is_identity(node):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                count += 1
        if count > 0:
            logger.debug("NoopElimination: removed %d identity ops", count)

    @staticmethod
    def _is_identity(node: torch.fx.Node) -> bool:
        inp = node.args[0]
        if not isinstance(inp, torch.fx.Node):
            return False
        out_val = node.meta.get("val")
        inp_val = inp.meta.get("val")
        if out_val is None or inp_val is None:
            return False
        if hasattr(out_val, "shape") and hasattr(inp_val, "shape"):
            return (list(out_val.shape) == list(inp_val.shape)
                    and out_val.dtype == inp_val.dtype)
        return False


class AllReduceFusedAddRMSNormPass(InductorPass):
    """Rewrite ``all_reduce -> add(residual) -> rms_norm`` into one kernel.

    Mirrors vLLM's ``AllReduceFusionPass``. Every tensor-parallel region in a
    transformer layer ends with this triple -- 36 layers x 2 sites for
    gpt-oss-120b -- and collapsing each one from two kernels and two HBM round
    trips into one, launched with PDL, is worth 0.39 ms/step of decode at tp=2 on
    B200. (Measured on vLLM by toggling its own ``fuse_allreduce_rms``:
    2.684 -> 2.292 ms/step.)

    Matched structurally rather than with ``pattern_matcher.register_replacement``.
    vLLM can use the matcher because their RMSNorm is a single IR-level op, so
    both of the subgraph's outputs are ``getitem``s off one node. Ours is
    :meth:`RMSNorm.forward_native`, which decomposes into two independent aten
    chains that merely share the residual add -- and ``MultiOutputPattern``
    cannot bridge two separately-traced chains, so it rejects every candidate
    with "no anchor found". Walking the chain by hand also allows rewiring both
    consumers, which the matcher does not support.

    The chain, as it appears post-grad (``add`` has exactly three users):

        ar   = fastkernels.custom_all_reduce(x)
        c1   = convert(ar, f32);  c2 = convert(residual, f32)
        add  = c1 + c2
        rs   = rsqrt(mean(add ** 2, -1, keepdim) + eps)
        norm = convert(add * rs, dtype) * weight
        res  = convert(add, dtype)

    Any deviation aborts that site and leaves it untouched, so a failed match
    costs performance, never correctness.
    """

    name = "allreduce_fused_add_rmsnorm"

    _CVT = torch.ops.prims.convert_element_type.default
    _ADD = torch.ops.aten.add.Tensor
    _MUL = torch.ops.aten.mul.Tensor
    _POW = torch.ops.aten.pow.Tensor_Scalar
    _MEAN = torch.ops.aten.mean.dim
    _RSQRT = torch.ops.aten.rsqrt.default

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.matched_count = 0
        # The fastkernels namespace is populated at runtime, after this module is
        # imported, so these cannot be class attributes.
        ensure_custom_ops_registered()
        self._AR = torch.ops.fastkernels.custom_all_reduce.default
        self._FUSED = torch.ops.fastkernels.fused_allreduce_add_rmsnorm.default
        self._FUSED_NR = torch.ops.fastkernels.fused_allreduce_rmsnorm.default

    # -- small helpers over fx nodes ----------------------------------------

    @staticmethod
    def _is(node: object, target: object) -> bool:
        return (isinstance(node, torch.fx.Node) and node.op == "call_function"
                and node.target is target)

    @classmethod
    def _sole_user(cls, node: torch.fx.Node, target: object) -> torch.fx.Node | None:
        """The node's only user, if it is a call to *target*."""
        if len(node.users) != 1:
            return None
        user = next(iter(node.users))
        return user if cls._is(user, target) else None

    @classmethod
    def _cvt_to(cls, node: object, dtype: torch.dtype) -> torch.fx.Node | None:
        """*node* if it converts to *dtype*, else None."""
        if cls._is(node, cls._CVT) and node.args[1] == dtype:
            return node  # type: ignore[return-value]
        return None

    @classmethod
    def _binary_operands(cls, node: torch.fx.Node, one: torch.fx.Node):
        """Return the other operand of a 2-arg node known to take *one*."""
        a, b = node.args[0], node.args[1]
        if a is one:
            return b
        if b is one:
            return a
        return None

    # -- the pass -----------------------------------------------------------

    def __call__(self, graph: torch.fx.Graph) -> None:
        count = 0
        candidates = [n for n in graph.nodes
                      if n.op == "call_function" and n.target is self._AR]
        for ar in candidates:
            if self._fuse_site(graph, ar) or self._fuse_site_no_residual(graph, ar):
                count += 1
        if count:
            graph.eliminate_dead_code()
        self.matched_count += count
        logger.info("AllReduceFusedAddRMSNorm: fused %d of %d all-reduce site(s) "
                    "in this graph (%d total)", count, len(candidates),
                    self.matched_count)

    @staticmethod
    def _skip(ar: torch.fx.Node, reason: str) -> bool:
        """Record why a site was left alone. Not fusing is always safe, so these
        are debug-level -- but a run that unexpectedly fuses nothing needs them."""
        logger.debug("AllReduceFusedAddRMSNorm: skipped %s: %s", ar.name, reason)
        return False

    def _fuse_site(self, graph: torch.fx.Graph, ar: torch.fx.Node) -> bool:
        dtype = self.dtype

        c1 = self._sole_user(ar, self._CVT)
        if c1 is None or c1.args[1] != torch.float32:
            return self._skip(ar, "all-reduce output is not a lone convert-to-f32")
        add = self._sole_user(c1, self._ADD)
        if add is None:
            return self._skip(ar, "convert-to-f32 does not feed a lone add")

        # The residual must arrive in the activation dtype, because that is what
        # the fused kernel reads -- and what vLLM's residual stream carries.
        # Inductor's ``pointless_convert`` collapses the f32 -> dtype -> f32
        # round-trip between layers, so an unfused site downstream of another
        # unfused site sees a raw f32 operand instead. Sites are visited in graph
        # order and each rewrite reinstates a ``convert(residual, f32)`` for its
        # consumers, so the next site down the residual stream then matches.
        widened = self._cvt_to(self._binary_operands(add, c1), torch.float32)
        if widened is None:
            return self._skip(ar, "add's other operand is not convert(residual, f32)")
        residual = widened.args[0]
        if not isinstance(residual, torch.fx.Node):
            return self._skip(ar, "residual operand is not a node")
        residual_val = residual.meta.get("val")
        if residual_val is None or residual_val.dtype != dtype:
            return self._skip(ar, "residual is not in the activation dtype")

        # Variance branch: add -> pow -> mean -> +eps -> rsqrt.
        pow_n = next((u for u in add.users
                      if self._is(u, self._POW) and u.args[1] == 2), None)
        if pow_n is None:
            return self._skip(ar, "add does not feed a squaring op")
        mean = self._sole_user(pow_n, self._MEAN)
        if mean is None or list(mean.args[1]) != [-1] or mean.args[2] is not True:
            return self._skip(ar, "mean is not over the last dim with keepdim")
        eps_add = self._sole_user(mean, self._ADD)
        if eps_add is None:
            return self._skip(ar, "mean does not feed a lone add(eps)")
        eps = eps_add.args[1]
        if not isinstance(eps, float):
            return self._skip(ar, "eps is not a float constant")
        rsqrt = self._sole_user(eps_add, self._RSQRT)
        if rsqrt is None:
            return self._skip(ar, "eps-add does not feed a lone rsqrt")

        # Scale branch: identified by taking rsqrt, not merely by being a mul --
        # a residual consumer can be a mul too.
        scale_mul = next((u for u in add.users if self._is(u, self._MUL)
                          and self._binary_operands(u, rsqrt) is add), None)
        if scale_mul is None:
            return self._skip(ar, "no multiply of the sum by rsqrt")
        norm_cvt = self._sole_user(scale_mul, self._CVT)
        if norm_cvt is None or norm_cvt.args[1] != dtype:
            return self._skip(ar, "scale multiply does not feed a lone convert back")
        weight_mul = self._sole_user(norm_cvt, self._MUL)
        if weight_mul is None:
            return self._skip(ar, "convert-back does not feed a lone multiply")
        weight = self._binary_operands(weight_mul, norm_cvt)
        if not isinstance(weight, torch.fx.Node):
            return self._skip(ar, "gamma operand is not a node")
        weight_val = weight.meta.get("val")
        if weight_val is None or weight_val.dim() != 1:
            return self._skip(ar, "gamma is not 1-D")
        # The kernel reads gamma in the activation dtype. A model with an fp32
        # norm weight (Gemma-style) would be silently misread, so require a
        # match -- vLLM guards the same case with a separate pattern.
        if weight_val.dtype != dtype:
            return self._skip(ar, "gamma is not in the activation dtype")

        norm_val = weight_mul.meta.get("val")
        add_val = add.meta.get("val")
        if norm_val is None or add_val is None:
            return self._skip(ar, "missing fake-tensor metadata")

        # A fresh fake tensor for the residual output: reusing the *input*
        # residual's ``val`` would leave a surviving node and a new node sharing
        # one metadata object, which Inductor's aliasing analysis reads.
        try:
            with residual_val.fake_mode:
                res_val = torch.empty_like(residual_val)
        except AttributeError:
            res_val = residual_val

        # Everything downstream of the sum that is not the norm itself is a
        # residual consumer, and gets the fused op's residual output instead.
        residual_consumers = [u for u in add.users if u not in (pow_n, scale_mul)]

        with graph.inserting_before(weight_mul):
            fused = graph.call_function(
                self._FUSED, args=(ar.args[0], residual, weight, eps))
            fused.meta["val"] = (norm_val, res_val)
            norm_out = graph.call_function(operator.getitem, args=(fused, 0))
            norm_out.meta["val"] = norm_val
            res_out = graph.call_function(operator.getitem, args=(fused, 1))
            res_out.meta["val"] = res_val

        weight_mul.replace_all_uses_with(norm_out)

        for consumer in residual_consumers:
            if self._cvt_to(consumer, dtype) is not None:
                # Already narrowing the sum to the activation dtype -- that is
                # exactly the fused op's residual output.
                consumer.replace_all_uses_with(res_out)
                continue
            # Consumer wants the f32 sum. Hand it the bf16 residual widened back,
            # which is the value vLLM carries between layers, and which lets the
            # next site down the stream match this same shape.
            with graph.inserting_before(consumer):
                w = graph.call_function(
                    self._CVT, args=(res_out, torch.float32))
                w.meta["val"] = add_val
            consumer.replace_input_with(add, w)
        return True


    def _fuse_site_no_residual(self, graph: torch.fx.Graph,
                               ar: torch.fx.Node) -> bool:
        """The residual-free form: layer 0's ``input_layernorm``.

        ``forward_native`` with ``residual=None`` skips the add, so the variance
        branch and the scale multiply hang off the f32 cast of the all-reduce
        rather than off a sum:

            c1   = convert(ar, f32)
            rs   = rsqrt(mean(c1 ** 2, -1, keepdim) + eps)
            norm = convert(c1 * rs, dtype) * weight

        Unlike the with-residual form, ``ar`` legitimately has a second consumer:
        layer 0 carries the un-normalised all-reduce result forward as the
        residual. Those consumers are rewired to the op's second output.
        """
        dtype = self.dtype

        c1 = next((u for u in ar.users
                   if self._cvt_to(u, torch.float32) is not None), None)
        if c1 is None:
            return self._skip(ar, "[nr] all-reduce output has no convert-to-f32 user")

        pow_n = next((u for u in c1.users
                      if self._is(u, self._POW) and u.args[1] == 2), None)
        if pow_n is None:
            return self._skip(ar, "[nr] no squaring op on the cast")
        mean = self._sole_user(pow_n, self._MEAN)
        if mean is None or list(mean.args[1]) != [-1] or mean.args[2] is not True:
            return self._skip(ar, "[nr] mean is not over the last dim with keepdim")
        eps_add = self._sole_user(mean, self._ADD)
        if eps_add is None:
            return self._skip(ar, "[nr] mean does not feed a lone add(eps)")
        eps = eps_add.args[1]
        if not isinstance(eps, float):
            return self._skip(ar, "[nr] eps is not a float constant")
        rsqrt = self._sole_user(eps_add, self._RSQRT)
        if rsqrt is None:
            return self._skip(ar, "[nr] eps-add does not feed a lone rsqrt")

        scale_mul = next((u for u in c1.users if self._is(u, self._MUL)
                          and self._binary_operands(u, rsqrt) is c1), None)
        if scale_mul is None:
            return self._skip(ar, "[nr] no multiply of the cast by rsqrt")
        if set(c1.users) - {pow_n, scale_mul}:
            return self._skip(ar, "[nr] cast has a consumer outside {pow, mul}")

        norm_cvt = self._sole_user(scale_mul, self._CVT)
        if norm_cvt is None or norm_cvt.args[1] != dtype:
            return self._skip(ar, "[nr] scale mul does not feed a lone convert back")
        weight_mul = self._sole_user(norm_cvt, self._MUL)
        if weight_mul is None:
            return self._skip(ar, "[nr] convert-back does not feed a lone multiply")
        weight = self._binary_operands(weight_mul, norm_cvt)
        if not isinstance(weight, torch.fx.Node):
            return self._skip(ar, "[nr] gamma operand is not a node")
        weight_val = weight.meta.get("val")
        if weight_val is None or weight_val.dim() != 1 or weight_val.dtype != dtype:
            return self._skip(ar, "[nr] gamma is not 1-D in the activation dtype")
        norm_val = weight_mul.meta.get("val")
        ar_val = ar.meta.get("val")
        if norm_val is None or ar_val is None:
            return self._skip(ar, "[nr] missing fake-tensor metadata")

        carriers = [u for u in ar.users if u is not c1]
        with graph.inserting_before(weight_mul):
            fused = graph.call_function(
                self._FUSED_NR, args=(ar.args[0], weight, eps))
            fused.meta["val"] = (norm_val, ar_val)
            norm_out = graph.call_function(operator.getitem, args=(fused, 0))
            norm_out.meta["val"] = norm_val
            ar_out = graph.call_function(operator.getitem, args=(fused, 1))
            ar_out.meta["val"] = ar_val

        weight_mul.replace_all_uses_with(norm_out)
        for carrier in carriers:
            carrier.replace_input_with(ar, ar_out)
        return True


class PostGradPassManager(torch._inductor.custom_graph_pass.CustomGraphPass):
    """Orchestrates post-grad Inductor passes.

    Wired into ``torch._inductor.config.post_grad_custom_post_pass`` to run
    after Inductor's own optimizations.

    Inherits ``CustomGraphPass`` so Inductor's cache machinery can type-check
    and hash the pass correctly.
    """

    def __init__(self, model_dtype: torch.dtype | None = None) -> None:
        self.passes: list[InductorPass] = [
            NoopEliminationPass(),
        ]
        ar_fusion = _maybe_allreduce_fusion_pass(model_dtype)
        if ar_fusion is not None:
            self.passes.append(ar_fusion)

    def __call__(self, graph: torch.fx.Graph) -> None:
        for p in self.passes:
            p(graph)

    def uuid(self):
        """Identity of this pass set, for Inductor's FX graph cache key.

        Returning None makes Inductor cache compiled graphs *without* accounting
        for these passes, so a cache written before a pass existed -- or before it
        was edited -- is replayed verbatim and the pass silently never runs. That
        is how the AR+RMSNorm fusion first appeared to do nothing: it fired on a
        toy model and matched 0 sites on gpt-oss, purely because the 120B graph
        came back from a stale cache. Hashing this file's source plus the active
        pass names makes any change to a pass invalidate the artifacts.
        """
        import hashlib

        from torch._inductor.custom_graph_pass import get_hash_for_files

        h = hashlib.sha256()
        h.update(get_hash_for_files((__file__,)))
        for p in self.passes:
            h.update(p.name.encode())
        return h.hexdigest()

    def add(self, pass_: InductorPass) -> None:
        self.passes.append(pass_)


def _maybe_allreduce_fusion_pass(
    model_dtype: torch.dtype | None = None,
) -> InductorPass | None:
    """Build the AR+RMSNorm fusion pass, or None if it cannot run here.

    Every precondition is a real runtime property, not a config choice: there are
    no collectives to fuse at tp=1, FlashInfer may not be installed, and the
    mnnvl backend needs NVSwitch multicast that only exists on some topologies.
    Returning None leaves the graph exactly as it was.
    """
    import torch.distributed as dist

    from ..tasks.baseline.L1.flashinfer_allreduce_fusion import (
        fi_ar_fusion_available,
    )

    if not (dist.is_available() and dist.is_initialized()):
        return None
    if dist.get_world_size() <= 1:
        logger.debug("AllReduce fusion disabled: world_size <= 1")
        return None
    if not fi_ar_fusion_available():
        logger.info("AllReduce fusion disabled: FlashInfer allreduce_fusion "
                    "unavailable")
        return None
    dtype = model_dtype or torch.get_default_dtype()
    return AllReduceFusedAddRMSNormPass(dtype=dtype)


def configure_post_grad_passes(model_dtype: torch.dtype | None = None) -> None:
    """Install the fastkernels post-grad pass manager into Inductor config.

    ``model_dtype`` is the activation dtype the graph will actually carry. The
    AR+RMSNorm pattern bakes it in, so it has to be the model's dtype rather
    than ``torch.get_default_dtype()`` -- the engine restores the process
    default before compiling.
    """
    pm = PostGradPassManager(model_dtype=model_dtype)
    torch._inductor.config.post_grad_custom_post_pass = pm
    logger.info("Installed PostGradPassManager with passes: %s",
                [p.name for p in pm.passes])


def remove_post_grad_passes() -> None:
    """Remove fastkernels post-grad passes from Inductor config."""
    torch._inductor.config.post_grad_custom_post_pass = None


# ===================================================================
# Inductor config alignment (mirrors vLLM's inductor_compile_config)
# ===================================================================
#
# vLLM 0.26 passes these to ``compile_fx`` as ``config_patches`` for every
# piecewise subgraph (``VllmBackend.inductor_config``).  Reproducing them
# matters for both numerics and throughput:
#
#   enable_auto_functionalized_v2=False
#       Custom post-grad fusion passes are written against
#       auto-functionalization V1; V2 silently stops them matching.
#
#   size_asserts / alignment_asserts / scalar_asserts = False
#       Inductor emits an assert_size_stride / assert_alignment call per
#       buffer.  vLLM measured ~2 ms per forward pass on large models and
#       disables them on torch < 2.12 unless VLLM_LOGGING_LEVEL=DEBUG.
#
#   combo_kernels / benchmark_combo_kernel = True  (torch >= 2.9)
#       Horizontal fusion, which vLLM enables specifically to fuse qk-norm
#       and qk-rope where query and key have different shapes.

def _torch_at_least(version: str) -> bool:
    from torch.torch_version import TorchVersion
    return TorchVersion(torch.__version__) >= version


def vllm_aligned_inductor_config() -> dict[str, Any]:
    """Build the ``compile_fx`` config patches vLLM 0.26 would use."""
    cfg: dict[str, Any] = {
        "enable_auto_functionalized_v2": False,
    }

    if not _torch_at_least("2.12.0.dev"):
        # torch >= 2.12 asserts once instead of per-buffer, so the
        # workaround is only needed below that.
        enable_asserts = (
            os.environ.get("FASTKERNELS_LOGGING_LEVEL", "").upper() == "DEBUG"
        )
        cfg["size_asserts"] = enable_asserts
        cfg["alignment_asserts"] = enable_asserts
        cfg["scalar_asserts"] = enable_asserts

    if _torch_at_least("2.9.0.dev"):
        cfg["combo_kernels"] = True
        cfg["benchmark_combo_kernel"] = True

    return cfg


# ===================================================================
# FX graph splitting (adapted from vllm/compilation/backends.py)
# ===================================================================

@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def _should_split(node: fx.Node, splitting_ops: list[str]) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, torch._ops.OpOverloadPacket):
        return target._qualified_op_name in splitting_ops
    if isinstance(target, torch._ops.OpOverload):
        packet_name = target.name()
        overload_name = f"{packet_name}.{target._overloadname}"
        return overload_name in splitting_ops or packet_name in splitting_ops
    return False


def _is_empty_allocation_node(node: fx.Node) -> bool:
    if node.op == "call_method":
        return node.target == "new_empty"
    if node.op != "call_function":
        return False
    target = node.target
    if target in (torch.empty, torch.empty_like, torch.empty_strided):
        return True
    if isinstance(target, torch._ops.OpOverloadPacket):
        pname = target._qualified_op_name
    elif isinstance(target, torch._ops.OpOverload):
        pname = target.name()
    else:
        return False
    return pname.startswith("aten::empty") or pname.startswith("aten::new_empty")


def _merge_empty_only_subgraphs(
    node_to_subgraph_id: dict[fx.Node, int],
    split_op_graphs: list[int],
) -> None:
    nodes_by_sgid: dict[int, list[fx.Node]] = defaultdict(list)
    for node, sgid in node_to_subgraph_id.items():
        nodes_by_sgid[sgid].append(node)

    splitting_set = set(split_op_graphs)
    prev_ns: int | None = None
    max_sgid = max(node_to_subgraph_id.values(), default=-1)

    for sgid in range(max_sgid + 1):
        nodes = nodes_by_sgid.get(sgid, [])
        if not nodes:
            continue
        is_ns = sgid not in splitting_set
        is_eo = len(nodes) == 1 and _is_empty_allocation_node(nodes[0])
        merged = False
        if is_eo and prev_ns is not None:
            empty_node = nodes[0]
            if all(
                inp.op == "placeholder"
                or node_to_subgraph_id[inp] <= prev_ns
                for inp in empty_node.all_input_nodes
            ):
                node_to_subgraph_id[empty_node] = prev_ns
                merged = True
        if not merged and is_ns:
            prev_ns = sgid


def split_graph(
    graph: fx.GraphModule,
    splitting_ops: list[str],
) -> tuple[fx.GraphModule, list[SplitItem]]:
    """Split an FX graph at custom-op boundaries."""
    subgraph_id = 0
    node_to_subgraph_id: dict[fx.Node, int] = {}
    split_op_graphs: list[int] = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue

        if node.op == "call_function" and node.target == operator.getitem:
            input_node = node.args[0]
            if input_node.op != "placeholder":
                assert input_node in node_to_subgraph_id
                node_to_subgraph_id[node] = node_to_subgraph_id[input_node]
                continue

        if _should_split(node, splitting_ops):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            if _should_split(node.next, splitting_ops):
                subgraph_id -= 1
            else:
                subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    _merge_empty_only_subgraphs(node_to_subgraph_id, split_op_graphs)

    split_gm = torch.fx.passes.split_module.split_module(
        graph, None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True,
    )

    outputs: list[SplitItem] = []
    for name, module in split_gm.named_modules():
        if "." in name or name == "":
            continue
        gid = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, gid, gid in split_op_graphs, module))
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# ===================================================================
# AlwaysHitShapeEnv (from vLLM compiler_interface.py)
# ===================================================================

class AlwaysHitShapeEnv:
    """Dummy shape environment that makes Inductor cache lookups always hit.

    When compiling subgraphs outside of Dynamo's tracing context, there's
    no ShapeEnv to provide. This dummy makes the cache work anyway.
    """

    def __init__(self) -> None:
        self.guards: list[Any] = []
        # Newer PyTorch (>=2.6) Inductor codecache reads this off the ShapeEnv
        # when building the FxGraph cache key. The real ShapeEnv defaults it to
        # an empty dict; mirror that so cache-key construction stays a no-op.
        self.var_to_hint_override: dict[Any, Any] = {}

    def evaluate_guards_expression(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def get_pruned_guards(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def produce_guards_expression(self, *args: Any, **kwargs: Any) -> str:
        return ""


# ===================================================================
# Helpers for extracting fake args from graph (mirrors vLLM)
# ===================================================================

def _get_fake_args(graph: fx.GraphModule) -> list:
    """Get fake/symbolic args from placeholder metadata.

    This is the key mechanism: Inductor receives fake tensors with symbolic
    shapes, so it generates code that works for ANY concrete batch size —
    no per-shape recompilation needed.
    """
    fake_args = []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("example_value")
            if val is None:
                val = node.meta.get("val")
            if val is not None:
                fake_args.append(val)
        else:
            break
    return fake_args


# ===================================================================
# PiecewiseBackend (mirrors vLLM's PiecewiseBackend)
# ===================================================================

class _StopCompiling(BaseException):
    pass


class PiecewiseBackend:
    """Compiled backend for a single non-splitting subgraph.

    Compiles the subgraph once with symbolic/fake args from the graph,
    then dispatches at runtime. Identical layers are deduplicated via
    autograd_cache_key normalization.
    """

    # Class-level cache: autograd_cache_key -> compiled callable.
    # Shared across all PiecewiseBackend instances to enable dedup.
    _loaded_artifacts: dict[str, Any] = {}

    def __init__(
        self,
        graph: fx.GraphModule,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        returns_tuple: bool,
        fake_args: list[Any] | None = None,
    ):
        self.graph = graph
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sym_shape_indices = sym_shape_indices
        self.returns_tuple = returns_tuple

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1
        )

        self.runnable = self._compile(fake_args)

    def _compile(self, fake_args: list[Any] | None = None) -> Callable[..., Any]:
        """Compile this subgraph once with symbolic args."""
        from torch._inductor.compile_fx import compile_fx

        if fake_args is None:
            fake_args = _get_fake_args(self.graph)

        graph_copy = copy.deepcopy(self.graph)

        from torch._subclasses.fake_tensor import FakeTensor
        input_fake_mode = None
        for x in fake_args:
            if isinstance(x, FakeTensor):
                input_fake_mode = x.fake_mode
                break

        cache_key = None
        orig_cache_key_fn = (
            torch._functorch._aot_autograd.autograd_cache.autograd_cache_key
        )

        def patched_autograd_cache_key(*args, **kwargs):
            result = orig_cache_key_fn(*args, **kwargs)
            if result is None:
                return None
            nonlocal cache_key
            cache_key = result[0]
            if cache_key in PiecewiseBackend._loaded_artifacts:
                raise _StopCompiling()
            return result

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        def _check_can_cache(*args, **kwargs) -> None:
            return

        with ExitStack() as stack:
            stack.enter_context(
                torch._functorch.config.patch(
                    autograd_cache_normalize_inputs=True
                )
            )
            stack.enter_context(
                patch(
                    "torch._functorch._aot_autograd.autograd_cache"
                    ".autograd_cache_key",
                    patched_autograd_cache_key,
                )
            )
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    _get_shape_env,
                )
            )
            from torch._functorch._aot_autograd.autograd_cache import (
                AOTAutogradCache,
            )
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache"
                        ".AOTAutogradCache._get_shape_env",
                        _get_shape_env,
                    )
                )
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache,
                )
            )
            stack.enter_context(
                torch._inductor.config.patch(fx_graph_remote_cache=False)
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )
            stack.enter_context(
                torch._functorch.config.patch(
                    enable_remote_autograd_cache=False
                )
            )

            if hasattr(torch._dynamo, "utils"):
                ctx = torch._dynamo.utils.get_metrics_context()
                stack.enter_context(ctx)

            tracing_ctx = torch._guards.TracingContext.try_get()
            old_tracing_fake_mode = None
            if tracing_ctx is not None and input_fake_mode is not None:
                old_tracing_fake_mode = tracing_ctx.fake_mode
                tracing_ctx.fake_mode = input_fake_mode

            try:
                compiled = compile_fx(
                    graph_copy,
                    fake_args,
                    config_patches={
                        "fx_graph_cache": True,
                        "fx_graph_remote_cache": False,
                        **vllm_aligned_inductor_config(),
                    },
                )
            except _StopCompiling:
                assert cache_key is not None
                logger.debug(
                    "Subgraph %d/%d deduplicated (cache_key hit)",
                    self.piecewise_compile_index,
                    self.total_piecewise_compiles,
                )
                return PiecewiseBackend._loaded_artifacts[cache_key]
            finally:
                if tracing_ctx is not None and old_tracing_fake_mode is not None:
                    tracing_ctx.fake_mode = old_tracing_fake_mode

        if cache_key is not None and compiled is not None:
            PiecewiseBackend._loaded_artifacts[cache_key] = compiled

        logger.debug(
            "Compiled subgraph %d/%d",
            self.piecewise_compile_index,
            self.total_piecewise_compiles,
        )
        return compiled

    def __call__(self, *args: Any) -> Any:
        graph_output = self.runnable(*args)
        if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
            return graph_output
        return graph_output[0]


# ===================================================================
# PiecewiseCompileInterpreter (mirrors vLLM)
# ===================================================================

class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Interpreter that replaces compilable submodules with PiecewiseBackend
    instances, optionally wrapped with CUDAGraphWrapper.

    Runs the split graph with fake/symbolic args to drive compilation of
    each subgraph. After this, the split graph's submodules are compiled
    callables.
    """

    def __init__(
        self,
        module: fx.GraphModule,
        compile_submod_names: list[str],
        cudagraph_enabled: bool = True,
    ):
        super().__init__(module)
        self.compile_submod_names = compile_submod_names
        self.cudagraph_enabled = cudagraph_enabled
        self.extra_traceback = False

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)

        gm = getattr(self.module, target)
        outputs = gm.graph.output_node().args[0]
        output = fx.map_arg(outputs, lambda node: node.meta["example_value"])

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            from torch._inductor.compile_fx import graph_returns_tuple

            piecewise = PiecewiseBackend(
                submod,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                graph_returns_tuple(submod),
                fake_args=list(args),
            )

            if self.cudagraph_enabled:
                wrapper = CUDAGraphWrapper(
                    runnable=piecewise,
                    runtime_mode=CUDAGraphMode.PIECEWISE,
                )
                self.module.__dict__[target] = wrapper
            else:
                self.module.__dict__[target] = piecewise

        return output


# ===================================================================
# FastKernels Dynamo backend (mirrors vLLM's VllmBackend)
# ===================================================================

class FastKernelsBackend:
    """Custom Dynamo backend that mirrors vLLM's VllmBackend.

    Called **exactly once** by Dynamo (guards are dropped). It:
    1. Splits the FX graph at attention custom-op boundaries
    2. Creates PiecewiseBackend for each non-splitting subgraph
    3. Compiles each with ``compile_fx`` using symbolic/fake args
    4. Deduplicates identical layers via ``autograd_cache_key``
    5. Wraps with CUDAGraphWrapper for PIECEWISE capture/replay
    """

    def __init__(
        self,
        splitting_ops: list[str] | None = None,
        cudagraph_enabled: bool = True,
    ):
        self.splitting_ops = splitting_ops or SPLITTING_OPS
        self.cudagraph_enabled = cudagraph_enabled
        self._called = False

    def __call__(
        self,
        graph: fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Any:
        assert not self._called, "FastKernelsBackend should only be called once"
        self._called = True

        logger.info("FastKernelsBackend: splitting graph at %s", self.splitting_ops)

        split_gm, piecewise_graphs = split_graph(graph, self.splitting_ops)

        compile_submod_names = [
            item.submod_name
            for item in piecewise_graphs
            if not item.is_splitting_graph
        ]

        logger.info(
            "FastKernelsBackend: %d subgraphs (%d compilable, %d splitting ops)",
            len(piecewise_graphs),
            len(compile_submod_names),
            len(piecewise_graphs) - len(compile_submod_names),
        )

        all_fake_values = []
        for node in graph.graph.find_nodes(op="placeholder"):
            all_fake_values.append(node.meta["example_value"])

        fake_args = [
            all_fake_values[i]
            if isinstance(t, torch.Tensor)
            else t
            for i, t in enumerate(example_inputs)
        ]

        PiecewiseCompileInterpreter(
            split_gm,
            compile_submod_names,
            cudagraph_enabled=self.cudagraph_enabled,
        ).run(*fake_args)

        logger.info("FastKernelsBackend: compilation complete")

        return split_gm


# ===================================================================
# Model compilation entry point
# ===================================================================

def compile_model(
    model: torch.nn.Module,
    cudagraph_enabled: bool = True,
) -> torch.nn.Module:
    """Apply torch.compile with the FastKernels backend.

    Mirrors vLLM's compilation flow:
    1. Register and enable custom ops for attention/MoE
    2. ``mark_dynamic`` on batch dimensions so Dynamo traces with symbolic
       shapes
    3. ``fullgraph=True`` — single graph, no graph breaks
    4. ``skip_all_guards_unsafe`` — Dynamo never re-traces
    5. ``FastKernelsBackend`` — splits, compiles with symbolic shapes, deduplicates

    The model is compiled once, then reused for all batch sizes.
    """
    ensure_custom_ops_registered()
    enable_custom_ops()

    PiecewiseBackend._loaded_artifacts.clear()

    backend = FastKernelsBackend(
        cudagraph_enabled=cudagraph_enabled,
    )

    options: dict[str, Any] = {}
    if hasattr(torch.compiler, "skip_all_guards_unsafe"):
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
    else:
        options["guard_filter_fn"] = lambda x: [False for _ in x]

    compiled = torch.compile(
        model,
        fullgraph=True,
        dynamic=False,
        backend=backend,
        options=options,
    )

    original_cache_size = torch._dynamo.config.cache_size_limit
    original_accumulated = torch._dynamo.config.accumulated_cache_size_limit
    torch._dynamo.config.cache_size_limit = 2048
    torch._dynamo.config.accumulated_cache_size_limit = 8192

    model._fastkernels_compiled = compiled  # type: ignore[attr-defined]
    model._fastkernels_cache_restore = (  # type: ignore[attr-defined]
        original_cache_size, original_accumulated
    )
    model._fastkernels_first_call = True  # type: ignore[attr-defined]

    logger.info(
        "Model wrapped with FastKernelsBackend (piecewise compile, "
        "symbolic shapes, autograd_cache_key dedup)"
    )
    return compiled
