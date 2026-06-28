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
    "fastkernels::mamba2_conv_ssm_forward",
    "fastkernels::unified_mla_attention",
    "fastkernels::sparse_attn_indexer",
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


class PostGradPassManager(torch._inductor.custom_graph_pass.CustomGraphPass):
    """Orchestrates post-grad Inductor passes.

    Wired into ``torch._inductor.config.post_grad_custom_post_pass`` to run
    after Inductor's own optimizations.

    Inherits ``CustomGraphPass`` so Inductor's cache machinery can type-check
    and hash the pass correctly.
    """

    def __init__(self) -> None:
        self.passes: list[InductorPass] = [
            NoopEliminationPass(),
        ]

    def __call__(self, graph: torch.fx.Graph) -> None:
        for p in self.passes:
            p(graph)

    def uuid(self):
        return None

    def add(self, pass_: InductorPass) -> None:
        self.passes.append(pass_)


def configure_post_grad_passes() -> None:
    """Install the fastkernels post-grad pass manager into Inductor config."""
    pm = PostGradPassManager()
    torch._inductor.config.post_grad_custom_post_pass = pm
    logger.info("Installed PostGradPassManager with passes: %s",
                [p.name for p in pm.passes])


def remove_post_grad_passes() -> None:
    """Remove fastkernels post-grad passes from Inductor config."""
    torch._inductor.config.post_grad_custom_post_pass = None


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
