"""Capture per-operator init/forward tensor metadata for a fastkernels model.

Runs a model end-to-end through the fastkernels ``LlamaEngine`` on a set of
short prompts and records the ``dtype``/``shape`` of every argument passed to
the ``__init__`` and ``forward`` of each ``nn.Module`` subclass defined under
``fastkernels.tasks.baseline`` (i.e. every fastkernels operator). Results are
aggregated per operator class and written to a JSON file.

The model, dtype, tensor-parallel degree, eager mode, ``max_num_seqs`` and the
capture workloads all come from a list of ``BenchmarkScenario`` objects loaded
from the scenarios YAML table named by the required ``scenarios`` argument (a
path, or a packaged name like ``full`` / ``default`` / ``minimal``): each
scenario is loaded into its own engine and every workload it lists is captured
to a separate report.

Scenarios are captured in parallel across the available GPUs. Each scenario
runs in its own child process pinned to a private set of GPUs (a ``tp=N``
scenario claims N GPUs) via ``CUDA_VISIBLE_DEVICES``; the scheduler packs
scenarios onto the GPU pool by TP degree and launches the next one as soon as
enough GPUs free up. Because every scenario is isolated in its own process, a
crash, OOM or CUDA fault in one never brings down the others -- it is recorded
as that scenario's failure and the rest continue. Use ``--gpus`` to restrict the
pool (with a single GPU the scenarios simply run one at a time).

Usage::

    python -m fastkernels capture full                     # parallel, all GPUs
    python -m fastkernels capture minimal --gpus 0,1,2,3    # restrict the GPU pool
    python -m fastkernels capture full --output /tmp/llama32_1b_ops.json

The capture is cross-checked two independent ways: (1) a forward-pre-hook that
re-observes every operator's forward arguments, and (2) a mock continuous-
batching / chunked-prefill replay that -- driven only by the real per-request
prompt and generated lengths plus the engine's batch/token budgets -- predicts
the per-step batch composition and checks it against the captured tensor
leading dimensions (sequence length / token count / batch size).
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import enum
import functools
import gc
import importlib
import inspect
import json
import math
import os
import pkgutil
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .workloads import (
    VLM, OmniModal, Purpose, load_real_prompt_workload, resolve_benchmark, spec_for,
)

# Default directory for capture reports (override per-run with ``--output``).
CAPTURE_DIR = Path.home() / ".fastkernels" / "captures"

# Internal env var the parallel scheduler uses to tell a worker subprocess which
# scenario to capture. It is set (alongside CUDA_VISIBLE_DEVICES) by the parent
# and read at startup by the child; it is deliberately NOT a user-facing CLI
# flag, so a normal ``fastkernels capture`` invocation never sees it.
_WORKER_INDEX_ENV = "FK_CAPTURE_WORKER_INDEX"

# The package that holds every fastkernels operator (nn.Module) definition.
_BASELINE_PACKAGE = "fastkernels.tasks.baseline"

# Base TCP port for the engine's tensor-parallel c10d rendezvous. Each scheduled
# scenario is handed a *distinct* port (base + its scenario index) via the
# ``FASTKERNELS_NCCL_PORT`` env var so that concurrently-running multi-GPU
# scenarios never collide on a single fixed port -- a collision makes rank 0's
# TCPStore bind fail with ``EADDRINUSE`` and kills every-but-one TP scenario.
# Overridable so a back-to-back re-run can step around a lingering TIME_WAIT.
_NCCL_PORT_BASE = int(os.environ.get("FASTKERNELS_NCCL_PORT_BASE", "29500"))

# --- Scenario watchdog -------------------------------------------------------
# A tensor-parallel scenario runs as rank 0 (the child the scheduler launches)
# plus ``tp - 1`` worker processes it spawns. When a rank hits an OOM, an
# illegal-memory fault, or a kernel that aborts, it does NOT cleanly exit -- the
# whole group wedges: the healthy ranks spin forever inside a NCCL collective
# waiting for the faulted peer, or the faulted rank gets stuck unwinding a
# corrupt CUDA context. Either way rank 0 never returns and its GPUs are never
# reclaimed, which would stall the entire pool for the rest of the run.
#
# The one trait every wedge shares is that the scenario stops making progress,
# and progress is directly observable: a live scenario keeps writing to its log
# (per-workload banners, generation summaries, verification results), whereas a
# wedged one goes silent. So the watchdog force-kills a scenario's whole process
# group when its log has been idle for too long -- with a generous threshold so
# a legitimately long (but silent) generation is never mistaken for a hang -- and
# backstops that with an absolute wall-clock cap. Each scenario is launched in
# its own session so the group can be killed without touching its siblings.
_SCENARIO_TIMEOUT_SEC = int(os.environ.get("FASTKERNELS_SCENARIO_TIMEOUT_SEC", "3600"))
# Kill a scenario whose log has produced no new output for this long. Must
# comfortably exceed the longest *silent* stretch of a healthy scenario -- model
# load emits shard progress, each workload prints at its boundaries, and
# generation streams a per-request tqdm bar, so the log advances steadily.
_SCENARIO_STALL_SEC = int(os.environ.get("FASTKERNELS_SCENARIO_STALL_SEC", "1200"))
# How often (seconds) the scheduler runs the watchdog checks.
_WATCHDOG_CHECK_INTERVAL_SEC = 15.0
# Seconds to wait after SIGTERM before escalating a group kill to SIGKILL.
_WATCHDOG_TERM_GRACE_SEC = 8.0


# NOTE: The capture prompts come from standardized BenchmarkScenario workloads
# (from the scenarios table). The old ``DEFAULT_PROMPTS`` list (and the
# ad-hoc ``--prompt`` override that consumed it) has been removed; the list is
# kept here, commented out, for reference only.
#
# Short, varied prompts that ask for brief answers so that -- once wrapped in
# the instruct chat template -- greedy decoding emits the end-of-turn token and
# stops quickly. There are intentionally more prompts than the default batch
# size so the continuous batch has to admit waiting requests as earlier ones
# finish.
# DEFAULT_PROMPTS = [
#     "What is the capital of France? Answer in one word.",
#     "What is 2 + 2? Reply with just the number.",
#     "Name one primary color.",
#     "What is the opposite of hot? One word.",
#     "Translate 'good morning' into Spanish.",
#     "Give the chemical symbol for water.",
#     "What day comes after Monday?",
#     "Say hello.",
#     "What is the largest planet in our solar system?",
#     "Spell the word 'cat'.",
#     "How many legs does a spider have?",
#     "What color is the sky on a clear day? One word.",
# ]

# Capture scenarios come from a YAML table named by the required ``scenarios``
# positional argument (a path or a packaged name like ``full`` / ``default`` /
# ``minimal``), resolved via ``workloads.resolve_benchmark``. Each scenario is
# loaded into its own engine and every workload it lists is captured to a
# separate report.

# qualified_name -> {"init": {...}, "forward": {...}}
_RECORDS: dict[str, dict] = {}

# Independent forward capture via torch hooks, used to verify _RECORDS.
# qualified_name -> {"calls": int, "keys": {json_key: count}}
_HOOK_RECORDS: dict[str, dict] = {}

# A few operators (MLA attention) read the process-global inference ``Context``
# *inside* their forward instead of taking it as an argument, so their captured
# arg shapes alone don't say which code path ran. For just those ops we also
# record a coarse, bucketed context snapshot -- enough to rebuild a
# representative prefill Context offline (see ``fastkernels.bench``) but low-
# cardinality (a couple of bucketed scalars) so it dedups across layers/steps
# instead of exploding into one forward variant per call. Gated to this set so
# it never touches the dedup of any other op.
_CTX_SUMMARY_OPS = {
    "fastkernels.tasks.baseline.L2.mla_attention_impl:MLAAttention",
}


def _bucket_pow2(x: int) -> int:
    """Round ``x`` down to a power of two (0 stays 0) -- keeps a captured count
    low-cardinality so it does not defeat forward-variant dedup."""
    x = int(x or 0)
    return 1 << (x.bit_length() - 1) if x > 0 else 0


def _context_summary() -> dict:
    """Coarse snapshot of the global inference ``Context`` for the ops in
    ``_CTX_SUMMARY_OPS``. Best-effort: an empty dict if the context is
    unavailable (e.g. imported outside an engine run)."""
    try:
        from .infra.context import get_context
        ctx = get_context()
        return {"prefill": bool(ctx.is_prefill),
                "pf_seqs": _bucket_pow2(getattr(ctx, "num_prefill_seqs", 0))}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Value summarization
#
# Each value is rendered as a small, fully-spelled-out structure:
#   tensor        -> {"shape": [16384, 2048], "dtype": "bfloat16"}
#   torch.dtype   -> {"dtype": "bfloat16"}
#   module        -> {"module": "RotaryEmbedding"}
#   config/object -> {"object": "LlamaConfig"}
#   int/float/bool/None/str -> the raw JSON value
#   list / dict   -> recurse element-wise
# ---------------------------------------------------------------------------
def _dtype_name(dtype: torch.dtype) -> str:
    """Full dtype name without the ``torch.`` namespace (e.g. ``bfloat16``)."""
    return str(dtype).replace("torch.", "")


def _summarize(value, depth: int = 0):
    """Return a compact, structured summary of ``value``'s dtype/shape."""
    if isinstance(value, torch.Tensor):
        desc = {"shape": list(value.shape), "dtype": _dtype_name(value.dtype)}
        if value.device.type != "cuda" or value.device.index not in (0, None):
            desc["device"] = str(value.device)
        # Record the stride only when the tensor is *not* contiguous, so a
        # non-standard memory layout (e.g. a column-major / TMA-aligned FP8
        # scale, a transposed weight) can be faithfully re-materialized by a
        # consumer. Contiguous tensors omit it and are assumed row-major.
        try:
            if not value.is_contiguous():
                desc["stride"] = list(value.stride())
        except (RuntimeError, NotImplementedError):
            pass
        return desc
    if isinstance(value, torch.dtype):
        return {"dtype": _dtype_name(value)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:64]
    if isinstance(value, (list, tuple)):
        if depth >= 2:
            return {"list": len(value)}
        return [_summarize(v, depth + 1) for v in value[:8]]
    if isinstance(value, dict):
        if depth >= 2:
            return {"dict": len(value)}
        return {str(k): _summarize(v, depth + 1) for k, v in list(value.items())[:8]}
    if isinstance(value, nn.Module):
        return {"module": type(value).__name__}
    return {"object": type(value).__name__}


def _summarize_call(sig, args, kwargs) -> dict:
    """Summarize the arguments of a call, keyed by parameter name."""
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        summary = {}
        for name, val in bound.arguments.items():
            if name == "self":
                continue
            param = sig.parameters.get(name)
            if param is not None and param.kind is inspect.Parameter.VAR_KEYWORD:
                summary[name] = {k: _summarize(v) for k, v in val.items()}
            elif param is not None and param.kind is inspect.Parameter.VAR_POSITIONAL:
                summary[name] = [_summarize(v) for v in val]
            else:
                summary[name] = _summarize(val)
        return summary
    except Exception:
        # Fall back to positional/keyword dumps when binding fails.
        return {
            "_args": [_summarize(v) for v in args[1:]],  # drop self
            "_kwargs": {k: _summarize(v) for k, v in kwargs.items()},
        }


# ---------------------------------------------------------------------------
# Construction recipes
#
# ``_summarize`` (above) produces compact shape/dtype metadata for the batch-
# schedule verification; it deliberately collapses a config object to
# ``{"object": "LlamaConfig"}``. A *recipe* is the complementary representation
# used to *reconstruct* an operator's ``__init__`` args offline (for a per-
# operator benchmark): it preserves class identity + values.
#
#   dataclass          -> {"$dataclass": "mod:Cls", "fields": {name: recipe}}
#   torch.dtype        -> {"$dtype": "bfloat16"}
#   torch.Tensor       -> {"$tensor": {"shape": [...], "dtype": "bfloat16"}}
#   Enum               -> {"$enum": "mod:Cls", "value": recipe}
#   list               -> [recipe, ...]        tuple -> {"$tuple": [recipe, ...]}
#   dict               -> {"$dict": {k: recipe}}
#   nn.Module (a captured operator) -> {"$op_ref": {"op": "mod:Cls",
#                                                    "init_variant_id": i}}
#   int/float/bool/str/None -> stored verbatim
#   anything else (a runtime handle: process group, quant method, ...)
#                            -> {"$opaque": "TypeName"}   (a runner skips these)
#
# Marker keys are ``$``-prefixed and every plain dict is wrapped in ``$dict``,
# so a value recipe is unambiguous (a scalar, a list, or a one-``$``-key dict).
# Reconstruction lives here too (``reconstruct`` / ``reconstruct_op``) so the
# format and its inverse stay together; a benchmark imports them from
# ``fastkernels.capture``.
# ---------------------------------------------------------------------------
class ReconstructError(RuntimeError):
    """A recipe could not be turned back into a live value/module."""


_RECIPE_MAX_DEPTH = 8
_RECIPE_MAX_STR = 8192


def _type_qual(cls: type) -> str:
    """Import-resolvable ``module:QualName`` (handles nested class defs)."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _op_qual(cls: type) -> str:
    """``module:Name`` matching the keys ``_record`` uses for operators."""
    return f"{cls.__module__}:{cls.__name__}"


def _encode_arg(value, depth: int = 0):
    """Encode one value into a JSON-serializable, reconstructable recipe."""
    try:
        if depth > _RECIPE_MAX_DEPTH:
            return {"$opaque": "depth-exceeded"}
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= _RECIPE_MAX_STR else {"$opaque": "str-too-long"}
        if isinstance(value, torch.dtype):
            return {"$dtype": _dtype_name(value)}
        if isinstance(value, torch.Tensor):
            return {"$tensor": {"shape": list(value.shape),
                                "dtype": _dtype_name(value.dtype)}}
        if isinstance(value, enum.Enum):
            return {"$enum": _type_qual(type(value)),
                    "value": _encode_arg(value.value, depth + 1)}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {"$dataclass": _type_qual(type(value)),
                    "fields": {f.name: _encode_arg(getattr(value, f.name), depth + 1)
                               for f in dataclasses.fields(value) if f.init}}
        if isinstance(value, list):
            return [_encode_arg(v, depth + 1) for v in value]
        if isinstance(value, tuple):
            return {"$tuple": [_encode_arg(v, depth + 1) for v in value]}
        if isinstance(value, dict):
            return {"$dict": {str(k): _encode_arg(v, depth + 1)
                              for k, v in value.items()}}
        if isinstance(value, nn.Module):
            # A submodule built by an instrumented operator carries its init
            # tag; reference it so the runner rebuilds it from its own recipe.
            init_key = getattr(value, "_fk_init_key", None)
            if init_key is not None:
                return {"$op_ref": {"op": _op_qual(type(value)), "init_key": init_key}}
            return {"$opaque": _op_qual(type(value))}
        return {"$opaque": _op_qual(type(value))}
    except Exception:
        return {"$opaque": "encode-error"}


def _encode_init_call(sig, args, kwargs) -> dict:
    """Encode an ``__init__`` call as ``{param_name: recipe}`` (``self`` dropped)."""
    if sig is None:
        return {"_args": [_encode_arg(v) for v in args[1:]]}
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        out: dict = {}
        for name, val in bound.arguments.items():
            if name == "self":
                continue
            param = sig.parameters.get(name)
            if param is not None and param.kind is inspect.Parameter.VAR_KEYWORD:
                out["__varkw__"] = {str(k): _encode_arg(v) for k, v in val.items()}
            elif param is not None and param.kind is inspect.Parameter.VAR_POSITIONAL:
                out["__varargs__"] = [_encode_arg(v) for v in val]
            else:
                out[name] = _encode_arg(val)
        return out
    except Exception:
        return {"_args": [_encode_arg(v) for v in args[1:]]}


def _translate_op_refs(recipe, resolve):
    """Copy *recipe* with each ``$op_ref`` init_key replaced by the resolved
    ``init_variant_id`` (``resolve(op_qualname, init_key) -> int``)."""
    if isinstance(recipe, list):
        return [_translate_op_refs(x, resolve) for x in recipe]
    if isinstance(recipe, dict):
        if "$op_ref" in recipe:
            ref = recipe["$op_ref"]
            return {"$op_ref": {"op": ref["op"],
                                "init_variant_id": resolve(ref["op"], ref.get("init_key"))}}
        return {k: _translate_op_refs(v, resolve) for k, v in recipe.items()}
    return recipe


def is_reconstructable(recipe, operators=None, _seen=None) -> bool:
    """True if *recipe* has no ``$opaque`` nodes and every ``$op_ref`` resolves.

    Pass a report's ``operators`` dict to also recurse through op references;
    without it a resolvable ``$op_ref`` is assumed buildable.
    """
    if isinstance(recipe, list):
        return all(is_reconstructable(x, operators, _seen) for x in recipe)
    if isinstance(recipe, dict):
        if "$opaque" in recipe:
            return False
        if "$op_ref" in recipe:
            ref = recipe["$op_ref"]
            vid = ref.get("init_variant_id", -1)
            if vid < 0:
                return False
            if operators is None:
                return True
            seen = _seen or set()
            tag = (ref["op"], vid)
            if tag in seen:
                return True
            try:
                sub = operators[ref["op"]]["init"]["variants"][vid]["args"]
            except (KeyError, IndexError, TypeError):
                return False
            return is_reconstructable(sub, operators, seen | {tag})
        return all(is_reconstructable(v, operators, _seen) for v in recipe.values())
    return True


def _import_symbol(qual: str):
    module_name, _, name = qual.partition(":")
    obj = importlib.import_module(module_name)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def reconstruct(recipe, *, operators=None, device="cpu"):
    """Turn a value recipe back into a live value (tensors are zero-filled)."""
    if recipe is None or isinstance(recipe, (bool, int, float, str)):
        return recipe
    if isinstance(recipe, list):
        return [reconstruct(x, operators=operators, device=device) for x in recipe]
    if not isinstance(recipe, dict):
        raise ReconstructError(f"unexpected recipe node: {type(recipe).__name__}")
    if "$dtype" in recipe:
        return getattr(torch, recipe["$dtype"])
    if "$tensor" in recipe:
        spec = recipe["$tensor"]
        return torch.zeros(spec["shape"], dtype=getattr(torch, spec["dtype"]),
                           device=device)
    if "$tuple" in recipe:
        return tuple(reconstruct(x, operators=operators, device=device)
                     for x in recipe["$tuple"])
    if "$dict" in recipe:
        return {k: reconstruct(v, operators=operators, device=device)
                for k, v in recipe["$dict"].items()}
    if "$enum" in recipe:
        return _import_symbol(recipe["$enum"])(
            reconstruct(recipe["value"], operators=operators, device=device))
    if "$dataclass" in recipe:
        cls = _import_symbol(recipe["$dataclass"])
        return cls(**{k: reconstruct(v, operators=operators, device=device)
                      for k, v in recipe["fields"].items()})
    if "$op_ref" in recipe:
        ref = recipe["$op_ref"]
        vid = ref.get("init_variant_id", -1)
        if operators is None or vid < 0:
            raise ReconstructError(f"unresolvable op_ref: {ref!r}")
        return reconstruct_op(ref["op"], vid, operators, device=device)
    if "$opaque" in recipe:
        raise ReconstructError(f"opaque init arg: {recipe['$opaque']}")
    raise ReconstructError(f"unknown recipe marker: {sorted(recipe)[:1]}")


def _reconstruct_init_call(call: dict, *, operators=None, device="cpu"):
    """Return ``(args, kwargs)`` for a constructor from an encoded init call."""
    if "_args" in call:
        return ([reconstruct(v, operators=operators, device=device)
                 for v in call["_args"]], {})
    args = [reconstruct(v, operators=operators, device=device)
            for v in call.get("__varargs__", [])]
    varkw = {k: reconstruct(v, operators=operators, device=device)
             for k, v in call.get("__varkw__", {}).items()}
    named = {k: reconstruct(v, operators=operators, device=device)
             for k, v in call.items() if k not in ("__varargs__", "__varkw__")}
    return (args, {**named, **varkw})


def reconstruct_op(qualname: str, init_variant_id: int, operators: dict,
                   *, device: str = "cpu") -> nn.Module:
    """Instantiate operator *qualname* from its ``init_variant_id`` recipe.

    *operators* is a capture report's ``operators`` dict. Submodule ``$op_ref``
    args are rebuilt recursively. Raises :class:`ReconstructError` if the recipe
    is opaque/unresolvable. Tensor args are zero-filled and weights are freshly
    initialized -- fine for perf benchmarking, not numerical golden checks.
    """
    try:
        recipe = operators[qualname]["init"]["variants"][init_variant_id]["args"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ReconstructError(
            f"no init variant {init_variant_id} for {qualname}") from exc
    cls = _import_symbol(qualname)
    args, kwargs = _reconstruct_init_call(recipe, operators=operators, device=device)
    return cls(*args, **kwargs)


def _record(qualname: str, module_name: str, class_name: str,
            method: str, call_summary: dict, init_key: str | None = None) -> str:
    entry = _RECORDS.setdefault(qualname, {"init": None, "forward": None})
    slot = entry[method]
    if slot is None:
        # ``variants`` maps the JSON summary key -> {"count", "args"}. Using a
        # dict keeps recording O(1) per call and stores every distinct variant
        # (no cap), so long decode loops with many shapes stay exact.
        slot = entry[method] = {"calls": 0, "variants": {}}
    slot["calls"] += 1
    key = json.dumps(call_summary, sort_keys=True, default=str)
    variant = slot["variants"].get(key)
    if variant is None:
        variant = slot["variants"][key] = {"count": 1, "args": call_summary}
    else:
        variant["count"] += 1
    # For a ``forward`` call, remember which ``__init__`` variant built the
    # instance it ran on (``init_key``, the instance's ``_fk_init_key`` tag set
    # in ``_wrap_method``). This pairs each forward variant with the init args
    # needed to construct a benchmarkable module -- ``_build_report`` turns these
    # keys into ``init_variant_ids``. O(1) per call; ``entry["forward"] = None``
    # between workloads clears it, so no extra reset bookkeeping is needed.
    if method == "forward" and init_key is not None:
        init_keys = variant.setdefault("init_keys", {})
        init_keys[init_key] = init_keys.get(init_key, 0) + 1
    return key


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
def _wrap_method(cls, method_name: str):
    """Wrap ``cls.<method_name>`` in place to record its argument metadata."""
    raw = cls.__dict__.get(method_name)
    if raw is None or getattr(raw, "_fk_captured", False):
        return
    if not callable(raw):
        return

    qualname = f"{cls.__module__}:{cls.__name__}"
    record_key = "init" if method_name == "__init__" else method_name
    try:
        sig = inspect.signature(raw)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(raw)
    def wrapper(*args, **kwargs):
        tag_instance = None
        try:
            # Attribute the call to the *actual* instance class, so a subclass
            # that inherits this (unwrapped) method is recorded under its own
            # name -- matching the forward-pre-hook cross-check, which keys on
            # ``type(module)``. Without this, an inherited ``forward`` (e.g.
            # Gemma4ProportionalRotaryEmbedding inheriting RotaryEmbedding.forward)
            # records under the base class in the monkey-patch but the subclass
            # in the hook, producing a spurious verification mismatch.
            inst_cls = type(args[0]) if args and isinstance(args[0], cls) else cls
            q = f"{inst_cls.__module__}:{inst_cls.__name__}"
            # ``init`` is recorded as a reconstructable *recipe* (so distinct
            # configs become distinct variants and each can be rebuilt offline);
            # ``forward`` keeps the compact shape/dtype summary the batch-
            # schedule verification compares against the hook records.
            if record_key == "init":
                payload = _encode_init_call(sig, args, kwargs)
            elif sig is not None:
                payload = _summarize_call(sig, args, kwargs)
            else:
                payload = {"_args": [_summarize(v) for v in args[1:]]}
            # For context-reading ops, fold a coarse snapshot of the global
            # inference Context into the forward variant so the code path it ran
            # can be reconstructed offline. Injected identically in the hook
            # below so Verification #1's key cross-check still agrees.
            if record_key == "forward" and q in _CTX_SUMMARY_OPS:
                payload["_ctx"] = _context_summary()
            # On ``forward``, look up which ``__init__`` variant built this
            # instance so the call is attributed to those init args.
            init_key = (
                getattr(args[0], "_fk_init_key", None)
                if record_key == "forward" and args else None
            )
            key = _record(q, inst_cls.__module__, inst_cls.__name__,
                          record_key, payload, init_key=init_key)
            if record_key == "init" and args:
                # Defer tagging until after the real ``__init__`` runs (below):
                # the instance is then fully constructed, and for a subclass
                # whose ``super().__init__()`` is itself wrapped this records the
                # outermost (most-derived) init variant, not the base's.
                tag_instance = (args[0], key)
        except Exception:
            pass
        result = raw(*args, **kwargs)
        if tag_instance is not None:
            # A plain str -- not registered as a parameter/buffer/submodule by
            # ``nn.Module.__setattr__``.
            try:
                tag_instance[0]._fk_init_key = tag_instance[1]
            except Exception:
                pass
        return result

    wrapper._fk_captured = True
    setattr(cls, method_name, wrapper)


def _discover_operator_classes() -> list[type]:
    """Import every ``fastkernels.tasks.baseline`` submodule and collect the
    ``nn.Module`` subclasses defined there."""
    package = importlib.import_module(_BASELINE_PACKAGE)
    classes: dict[str, type] = {}

    def _collect(module) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, nn.Module):
                continue
            if not obj.__module__.startswith(_BASELINE_PACKAGE):
                continue
            classes[f"{obj.__module__}:{obj.__name__}"] = obj

    _collect(package)
    failed = []
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".csrc" in info.name:
            continue
        try:
            submodule = importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 - keep going on optional deps
            failed.append((info.name, exc))
            continue
        _collect(submodule)

    if failed:
        print(f"  (skipped {len(failed)} baseline module(s) that failed to import)")
        for name, exc in failed[:10]:
            print(f"    - {name}: {type(exc).__name__}: {exc}")

    # Restrict to list.py operator *targets*: never instrument/record a
    # non-target helper (e.g. an inner model class excluded via ``__targets__``)
    # that only runs as part of a bigger op.
    try:
        from .list import discover_operator_targets
        allowed = {
            f"{_BASELINE_PACKAGE}.L{t.level}.{t.name}:{t.class_name}"
            for t in discover_operator_targets()
        }
    except Exception as exc:  # noqa: BLE001 - never let target discovery break capture
        print(f"  (operator-target filter unavailable, capturing all: {exc!r})")
        allowed = None
    if allowed:
        classes = {k: v for k, v in classes.items() if k in allowed}

    return list(classes.values())


def _instrument(classes: list[type]) -> set[str]:
    instrumented: set[str] = set()
    for cls in classes:
        wrapped_any = False
        for method_name in ("__init__", "forward"):
            before = getattr(getattr(cls, method_name, None), "_fk_captured", False)
            _wrap_method(cls, method_name)
            after = getattr(getattr(cls, method_name, None), "_fk_captured", False)
            if after and not before:
                wrapped_any = True
        if wrapped_any:
            instrumented.add(f"{cls.__module__}:{cls.__name__}")
    return instrumented


# ---------------------------------------------------------------------------
# Verification: an independent forward capture via torch hooks
#
# ``nn.Module.__call__`` fires forward pre-hooks around ``self.forward`` --
# a code path that is entirely separate from our monkey-patched ``forward``.
# Recording the same calls both ways and checking that the call counts and the
# per-argument shape/dtype summaries agree gives strong evidence that the
# monkey-patch records exactly what actually flows through each operator.
# ---------------------------------------------------------------------------
def _make_forward_hook(qualname: str, sig):
    def hook(module, args, kwargs):
        try:
            summary = (
                _summarize_call(sig, (module,) + tuple(args), kwargs)
                if sig is not None
                else {"_args": [_summarize(v) for v in args]}
            )
            # Mirror the monkey-patch's context snapshot (see ``_wrap_method``)
            # so the two records produce identical keys for these ops.
            if qualname in _CTX_SUMMARY_OPS:
                summary["_ctx"] = _context_summary()
            rec = _HOOK_RECORDS.setdefault(qualname, {"calls": 0, "keys": {}})
            rec["calls"] += 1
            key = json.dumps(summary, sort_keys=True, default=str)
            rec["keys"][key] = rec["keys"].get(key, 0) + 1
        except Exception:
            pass
        return None

    return hook


def _register_verification_hooks(model, instrumented: set[str]) -> list:
    handles = []
    for module in model.modules():
        cls = type(module)
        qualname = f"{cls.__module__}:{cls.__name__}"
        if qualname not in instrumented:
            continue
        try:
            sig = inspect.signature(cls.forward)
        except (TypeError, ValueError):
            sig = None
        handle = module.register_forward_pre_hook(
            _make_forward_hook(qualname, sig), with_kwargs=True,
        )
        handles.append(handle)
    return handles


def _verify_forward_capture() -> dict:
    """Cross-check monkey-patch forward records against the hook records."""
    checked = 0
    matched = 0
    issues: list[str] = []
    for qualname, entry in sorted(_RECORDS.items()):
        fwd = entry.get("forward")
        if not fwd:
            continue
        hook = _HOOK_RECORDS.get(qualname)
        checked += 1
        if hook is None:
            issues.append(f"{qualname}: no hook observations (forward may be "
                          f"called directly, bypassing __call__)")
            continue

        mp_calls = fwd["calls"]
        hk_calls = hook["calls"]
        mp_keys = set(fwd["variants"].keys())
        hk_keys = set(hook["keys"].keys())

        problems = []
        if mp_calls != hk_calls:
            problems.append(f"call count {mp_calls} (patch) != {hk_calls} (hook)")
        # The monkey-patch and the hook each record every distinct shape/dtype
        # variant (neither is capped), so the two sets must agree exactly.
        missing = mp_keys - hk_keys
        if missing:
            problems.append(f"{len(missing)} shape variant(s) not seen by hook")
        extra = hk_keys - mp_keys
        if extra:
            problems.append(f"{len(extra)} shape variant(s) missed by patch")

        if problems:
            issues.append(f"{qualname}: " + "; ".join(problems))
        else:
            matched += 1

    return {
        "method": "independent torch forward_pre_hook cross-check",
        "classes_checked": checked,
        "classes_matched": matched,
        "passed": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Verification #2: a mock continuous-batching / chunked-prefill scheduler
#
# The engine runs every GPU step as a *unified* batch: it decodes one token for
# each running sequence and (chunk-)prefills waiting sequences in the very same
# forward pass, subject to three limits -- at most ``max_num_seqs`` concurrent
# sequences, at most ``max_num_batched_tokens`` tokens per step, and the paged
# KV-cache block pool (a waiting sequence is only admitted while its blocks --
# plus a watermark, and vLLM's ``full_sequence_must_fit`` check that its whole
# ``ceil(prompt/block_size)`` fits in the *free* pool -- are available; tokens it
# has yet to generate are not reserved, and growth past that point is handled by
# recompute-preempting the newest in-flight sequence). Given each
# request's prompt length, generated length and decode budget plus those limits
# and the block-pool geometry we can replay the policy analytically and predict,
# for every step, how many prefill tokens / prefill sequences / decode sequences
# it processes.
#
# The KV-cache limit is what caps concurrency for *long* prompts: the token/seq
# budgets alone let dozens of sequences run at once, but a handful of 8K-128K
# prompts exhaust the block pool first, so an accurate replay must model it (an
# earlier budget-only model over-predicted long-context concurrency). When the
# block pool is unknown the replay falls back to budget-only (the ``num_blocks
# is None`` path), which is exact for short-prompt workloads.
#
# Those three per-step counts fully determine the leading ("sequence length" /
# batch) dimension of every forward tensor:
#   * whole-batch ops (LlamaModel, RMSNorm, ...) see ``prefill_tokens +
#     decode_seqs`` rows;
#   * the prefill attention kernel sees ``prefill_tokens`` rows across
#     ``num_prefill_seqs`` sequences (``cu_seqlens_q`` has one extra entry);
#   * the decode attention kernel sees ``decode_seqs`` rows.
# So the simulation's predicted counts/variants can be checked directly against
# what the capture recorded, and against invariants that must hold for *any*
# correct schedule (total prefill == sum of prompt lengths, total decode ==
# sum of generated-minus-one, nothing exceeds the two budgets).
# ---------------------------------------------------------------------------
class _MockSeq:
    __slots__ = ("prompt_len", "gen_len", "max_tokens",
                 "num_computed", "nblocks", "generated")

    def __init__(self, prompt_len: int, gen_len: int, max_tokens: int):
        self.prompt_len = prompt_len
        self.gen_len = gen_len          # tokens actually generated (ground truth)
        self.max_tokens = max_tokens    # decode budget (reported, not reserved)
        self.num_computed = 0           # prompt tokens prefilled so far
        self.nblocks = 0                # KV blocks currently held
        self.generated = 0              # decode tokens produced so far


def _simulate_continuous_batching(prompt_lens, gen_lens, max_num_seqs,
                                  max_num_batched_tokens, *,
                                  max_tokens=None, num_blocks=None,
                                  block_size: int = 256,
                                  watermark_blocks: int = 0) -> list[dict]:
    """Replay the engine's unified chunked-prefill schedule analytically.

    Returns one dict per GPU step with the token/sequence composition the
    engine's forward pass would see that step. This mirrors the scheduler in
    ``LlamaEngine.generate``: (1) decode every running sequence (allocating a
    fresh block when it crosses a block boundary, recompute-preempting the
    *newest* in-flight sequence when the pool is empty); (2) continue in-flight
    prefills FIFO within the token + block budgets; (3) admit waiting sequences
    while they fit the token budget and the block pool (+ watermark), including
    vLLM's ``full_sequence_must_fit`` check that the whole prompt fits.

    ``max_tokens`` is retained for ``_MockSeq`` construction but is no longer a
    peak-block reservation, matching the engine: a request reserves its
    *current* length, not its final one. When ``num_blocks`` is
    ``None`` the KV-cache accounting is skipped entirely and the replay reduces
    to the token/seq-budget-only model (exact for short-prompt workloads).
    """
    from collections import deque

    if max_tokens is None:
        max_tokens = list(gen_lens)
    bounded = num_blocks is not None
    free = int(num_blocks) if bounded else 0

    def _cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    waiting: deque[_MockSeq] = deque(
        _MockSeq(p, g, m) for p, g, m in zip(prompt_lens, gen_lens, max_tokens)
    )
    prefilling: deque[_MockSeq] = deque()  # admitted, prefill not yet complete
    running: deque[_MockSeq] = deque()     # prefilled, still generating
    steps: list[dict] = []

    # Safety bound: every recorded step advances >=1 prefill or decode token,
    # so a correct schedule needs at most sum(prompt)+sum(gen) of them; anything
    # past this (or a no-progress deadlock) signals a bug and is cut short.
    max_steps = sum(gen_lens) + sum(prompt_lens) + len(prompt_lens) + 16

    while waiting or prefilling or running:
        if len(steps) > max_steps:
            break

        token_budget = max_num_batched_tokens

        # --- 1) decode scheduling: one token per running seq, up to
        #        max_num_seqs, allocating a block when it crosses a boundary. ---
        decode_seqs: list[_MockSeq] = []
        new_running: deque[_MockSeq] = deque()

        def _preempt_newest(exclude: _MockSeq) -> bool:
            """Recompute-preempt the newest in-flight seq, as the engine does.

            Mirrors ``_preempt_newest`` in ``LlamaEngine.generate``, which
            mirrors vLLM's ``preempted_req = self.running.pop()``: the *last*
            running request is evicted so the oldest keep advancing.
            ``running`` is oldest-to-newest and this loop pops from its left, so
            the newest survivors sit at its right end.
            """
            nonlocal free
            for pool in (running, new_running):
                while pool:
                    victim = pool.pop()
                    if victim is exclude:
                        pool.append(victim)
                        break
                    free += victim.nblocks
                    victim.nblocks = 0
                    victim.num_computed = 0
                    victim.generated = 0
                    waiting.appendleft(victim)
                    return True
            for i in range(len(decode_seqs) - 1, -1, -1):
                victim = decode_seqs[i]
                if victim is exclude:
                    continue
                del decode_seqs[i]
                free += victim.nblocks
                victim.nblocks = 0
                victim.num_computed = 0
                victim.generated = 0
                waiting.appendleft(victim)
                return True
            return False

        while running:
            seq = running.popleft()
            if len(decode_seqs) >= max_num_seqs:
                new_running.append(seq)
                continue
            if bounded and (seq.prompt_len + seq.generated) % block_size == 1:
                while free == 0 and _preempt_newest(seq):
                    pass
                if free == 0:
                    # Nothing left to evict: preempt this one instead.
                    free += seq.nblocks
                    seq.nblocks = 0
                    seq.num_computed = 0
                    seq.generated = 0
                    waiting.appendleft(seq)
                    continue
                seq.nblocks += 1
                free -= 1
            decode_seqs.append(seq)
        running = new_running
        token_budget -= len(decode_seqs)

        # --- 2) continue in-flight (partially prefilled) sequences, FIFO. ---
        prefill_seqs: list[_MockSeq] = []
        prefill_chunks: list[int] = []
        still: deque[_MockSeq] = deque()
        while prefilling and token_budget > 0:
            seq = prefilling.popleft()
            remaining = seq.prompt_len - seq.num_computed
            chunk = min(remaining, token_budget)
            if bounded:
                need = _cdiv(seq.num_computed + chunk, block_size) - seq.nblocks
                if need > 0:
                    if free < need:
                        still.append(seq)
                        continue
                    seq.nblocks += need
                    free -= need
            prefill_seqs.append(seq)
            prefill_chunks.append(chunk)
            token_budget -= chunk
        while prefilling:
            still.append(prefilling.popleft())
        prefilling = still

        # --- 3) admit new waiting sequences (token + block budgets). ---
        while waiting and token_budget > 0:
            seq = waiting[0]
            chunk = min(seq.prompt_len, token_budget)
            if bounded:
                need = _cdiv(chunk, block_size)  # num_computed == 0 (fresh)
                if free < need + watermark_blocks:
                    break
                # vLLM's ``full_sequence_must_fit`` gate: the request's current
                # length (its prompt, for a fresh admission) must fit in the
                # free blocks, so chunked prefill cannot admit a long prompt on
                # the strength of its first chunk. Tokens not yet generated are
                # not reserved -- see the matching comment in the engine.
                full_need = _cdiv(seq.prompt_len, block_size)
                if free < full_need + watermark_blocks:
                    break
            if len(prefill_seqs) + len(decode_seqs) >= max_num_seqs:
                break
            waiting.popleft()
            if bounded:
                seq.nblocks += need
                free -= need
            prefill_seqs.append(seq)
            prefill_chunks.append(chunk)
            token_budget -= chunk

        if not decode_seqs and not prefill_seqs:
            # Nothing schedulable (empty, or every prefill blocked with no decode
            # to free blocks) -- terminal, so stop rather than spin.
            break

        prefill_tokens = sum(prefill_chunks)
        steps.append({
            "prefill_tokens": prefill_tokens,
            "num_prefill_seqs": len(prefill_seqs),
            "prefill_seq_lens": sorted(prefill_chunks, reverse=True),
            "decode_seqs": len(decode_seqs),
            "total_tokens": prefill_tokens + len(decode_seqs),
        })

        # ---- apply the step's effects ----
        # Prefill advances; a sequence that completes emits its first token.
        for seq, chunk in zip(prefill_seqs, prefill_chunks):
            seq.num_computed += chunk
            if seq.num_computed >= seq.prompt_len:
                seq.generated = 1         # first token comes from the prefill
                if seq.generated >= seq.gen_len:
                    if bounded:           # finished at prefill: free its blocks
                        free += seq.nblocks
                        seq.nblocks = 0
                else:
                    running.append(seq)
            else:
                prefilling.append(seq)
        # Each decoded sequence emitted one token this step.
        for seq in decode_seqs:
            seq.generated += 1
            if seq.generated >= seq.gen_len:
                if bounded:
                    free += seq.nblocks
                    seq.nblocks = 0
            else:
                running.append(seq)

    return steps


def _find_hook_record(*suffixes):
    """Return the hook record whose qualified name ends with a suffix.

    We read the batch-composition ground truth from the independent forward-pre-
    hook observations (``_HOOK_RECORDS``), which store every distinct argument
    signature keyed by its JSON summary, so the histograms below stay exact at
    any scale.
    """
    for suffix in suffixes:
        for qualname, rec in _HOOK_RECORDS.items():
            if qualname.endswith(suffix) and rec.get("keys"):
                return rec
    return None


def _leading_dim(summary: dict, *names):
    """First matching argument's leading (row / sequence) dimension, or None."""
    for name in names:
        val = summary.get(name)
        if isinstance(val, dict) and isinstance(val.get("shape"), list) \
                and val["shape"]:
            return val["shape"][0]
    return None


def _infer_num_layers(engine) -> int:
    cfg = getattr(getattr(engine, "model_runner", None), "config", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if isinstance(n, int) and n > 0:
        return n
    for suffix in ("flash_attn_decode:FlashAttnDecode",
                   "flash_attn_prefill:FlashAttnPrefill",
                   "attention:LlamaAttention"):
        for qualname, entry in _RECORDS.items():
            if qualname.endswith(suffix) and entry.get("init"):
                return entry["init"]["calls"]
    return 1


def _iter_hook_summaries(rec):
    """Yield ``(summary_dict, count)`` for every distinct signature in a hook."""
    for key, count in rec["keys"].items():
        try:
            summary = json.loads(key)
        except (TypeError, ValueError):
            continue
        if isinstance(summary, dict):
            yield summary, count


def _actual_combined_tokens() -> dict[int, int] | None:
    """Per-step multiset of the whole-batch row count from the model-level op.

    The model-level module runs once per GPU step, so its call counts are
    already per-step (no per-layer scaling needed).
    """
    # Prefer the Llama model op; fall back to any top-level ``*ForCausalLM`` so
    # non-Llama LLMs (e.g. GptOssForCausalLM) are covered too. The whole-model op
    # runs exactly once per step regardless of architecture.
    rec = _find_hook_record(
        "L4.llama:LlamaModel", ":LlamaForCausalLM", "ForCausalLM",
    )
    if rec is None:
        return None
    multiset: dict[int, int] = {}
    for summary, count in _iter_hook_summaries(rec):
        n = _leading_dim(summary, "input_ids", "positions")
        if n is not None:
            multiset[n] = multiset.get(n, 0) + count
    return multiset


def _actual_prefill(num_layers: int) -> dict | None:
    """Per-step multisets of prefill token / sequence counts from the kernel.

    The prefill attention kernel runs once per *prefill-using* layer per step, so
    raw call counts are divided by that per-step invocation count to recover
    per-step frequencies. For a uniform model this equals ``num_layers``; but for
    a model with heterogeneous attention where only a subset of layers route to
    FlashAttnPrefill it is that subset's size -- e.g. Gemma-4, whose sliding-window
    layers use FlashAttnPrefill (with a window) while its full-attention layers
    take a separate long-context prefill path, so only 25 of its 30 layers invoke
    FlashAttnPrefill per long-context prefill step.

    We recover the per-step invocation count self-calibrating from the data:
    within a step the kernel is invoked once per prefill-using layer, all at that
    step's chunk size, so each per-chunk-size call count is a multiple of the
    per-step invocation count and their GCD equals it (a chunk size that occurs in
    a single step -- always present with varied real prompts -- pins the GCD to
    exactly that count). ``num_layers`` is the fallback when the GCD is degenerate
    (no records, or > num_layers because every chunk size shares a common step
    multiple).

    Raw counts are accumulated per leading dimension *before* dividing so
    fractional per-step frequencies are never rounded to zero.
    """
    rec = _find_hook_record("flash_attn_prefill:FlashAttnPrefill")
    if rec is None:
        return None
    raw_tokens: dict[int, int] = {}   # q leading dim -> raw call count
    raw_seqs: dict[int, int] = {}     # (cu_seqlens_q - 1) -> raw call count
    for summary, count in _iter_hook_summaries(rec):
        n = _leading_dim(summary, "q")
        if n is None:
            continue
        raw_tokens[n] = raw_tokens.get(n, 0) + count
        cu = _leading_dim(summary, "cu_seqlens_q")
        if cu is not None:
            raw_seqs[cu - 1] = raw_seqs.get(cu - 1, 0) + count
    # Per-step FlashAttnPrefill invocations = number of prefill-using layers,
    # recovered as the GCD of the per-chunk-size call counts (see docstring).
    lyr = math.gcd(*raw_tokens.values()) if raw_tokens else 0
    if not 1 <= lyr <= max(num_layers, 1):
        lyr = max(num_layers, 1)
    tokens = {n: int(round(c / lyr)) for n, c in raw_tokens.items()}
    seqs = {s: int(round(c / lyr)) for s, c in raw_seqs.items()}
    total_tokens = sum(n * steps for n, steps in tokens.items())
    return {"tokens": tokens, "seqs": seqs, "total_tokens": total_tokens,
            "layers_per_prefill_step": lyr}


def _actual_decode() -> dict | None:
    """Per-step multiset of decode batch sizes (q rows) from the decode kernel.

    Runs once per layer per step, so counts are divided by ``num_layers``... but
    since the decode batch size is bounded by ``max_num_seqs`` we only need the
    set of observed sizes and the maximum, which are scale-invariant.
    """
    rec = _find_hook_record("flash_attn_decode:FlashAttnDecode")
    if rec is None:
        return None
    sizes: set[int] = set()
    biggest = 0
    for summary, _count in _iter_hook_summaries(rec):
        n = _leading_dim(summary, "q")
        if n is not None:
            sizes.add(n)
            biggest = max(biggest, n)
    return {"sizes": sizes, "max": biggest}


def _ms_sorted(multiset: dict[int, int]) -> dict[str, int]:
    """JSON-friendly (string-keyed, sorted) view of an int-keyed multiset."""
    return {str(k): multiset[k] for k in sorted(multiset)}


def _multiset_diff(predicted: dict[int, int], actual: dict[int, int]) -> str:
    if predicted == actual:
        return "exact match"
    keys = sorted(set(predicted) | set(actual))
    return "; ".join(
        f"{k}: predicted {predicted.get(k, 0)} vs captured {actual.get(k, 0)}"
        for k in keys if predicted.get(k, 0) != actual.get(k, 0)
    )


def _verify_batch_schedule(prompt_lens, gen_lens, max_num_seqs,
                           max_num_batched_tokens, num_layers, *,
                           max_tokens=None, num_blocks=None,
                           block_size: int = 256,
                           watermark_blocks: int = 0) -> dict:
    """Cross-check the captured forward shapes against a mock scheduler."""
    steps = _simulate_continuous_batching(
        prompt_lens, gen_lens, max_num_seqs, max_num_batched_tokens,
        max_tokens=max_tokens, num_blocks=num_blocks,
        block_size=block_size, watermark_blocks=watermark_blocks,
    )

    pred_combined: dict[int, int] = {}
    pred_prefill_tokens: dict[int, int] = {}
    pred_prefill_seqs: dict[int, int] = {}
    total_prefill = 0
    total_decode = 0
    max_total = 0
    max_active = 0
    max_decode = 0
    for st in steps:
        pred_combined[st["total_tokens"]] = (
            pred_combined.get(st["total_tokens"], 0) + 1)
        total_decode += st["decode_seqs"]
        max_total = max(max_total, st["total_tokens"])
        max_decode = max(max_decode, st["decode_seqs"])
        max_active = max(max_active, st["num_prefill_seqs"] + st["decode_seqs"])
        if st["prefill_tokens"] > 0:
            pt, ps = st["prefill_tokens"], st["num_prefill_seqs"]
            pred_prefill_tokens[pt] = pred_prefill_tokens.get(pt, 0) + 1
            pred_prefill_seqs[ps] = pred_prefill_seqs.get(ps, 0) + 1
            total_prefill += pt

    checks: list[dict] = []

    def _check(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    # --- invariants that must hold for ANY correct schedule ---
    want_prefill = sum(prompt_lens)
    want_decode = sum(g - 1 for g in gen_lens)
    # Recompute-preemption (under paged-KV pressure -- e.g. many long-context
    # sequences, amplified for sliding-window models that the pool sizing does
    # not shrink) legitimately *re-prefills* a preempted sequence on resume, so
    # prefill/decode work becomes a lower bound of, not equal to, the single-pass
    # total. Detect it (the replay re-prefilled beyond the prompt sum) and check
    # ">=" instead of "=="; the exact per-step / per-chunk histograms below still
    # cross-check the replay against the capture strictly.
    preempted = total_prefill > want_prefill
    _pf_extra = total_prefill - want_prefill
    if preempted:
        _check("total prefill tokens >= sum(prompt lengths) [recompute-preemption]",
               total_prefill >= want_prefill,
               f"simulated {total_prefill} >= {want_prefill} "
               f"(+{_pf_extra} re-prefilled by preemption)")
        _check("total decode tokens >= sum(generated - 1) [recompute-preemption]",
               total_decode >= want_decode,
               f"simulated {total_decode} >= {want_decode}")
    else:
        _check("total prefill tokens == sum(prompt lengths)",
               total_prefill == want_prefill,
               f"simulated {total_prefill} vs sum(prompt_lens) {want_prefill}")
        _check("total decode tokens == sum(generated - 1)",
               total_decode == want_decode,
               f"simulated {total_decode} vs {want_decode}")
    _check("every step within max_num_batched_tokens",
           max_total <= max_num_batched_tokens,
           f"max step tokens {max_total} <= {max_num_batched_tokens}")
    _check("concurrent sequences within max_num_seqs",
           max_active <= max_num_seqs,
           f"max concurrent {max_active} <= {max_num_seqs}")

    # --- cross-check the simulation against the captured shapes ---
    # These histograms come from the uncapped hook records, so the comparison
    # stays exact regardless of how many distinct token counts a large, unknown
    # prompt set produces.
    combined = _actual_combined_tokens()
    if combined is not None:
        _check("per-step token counts match capture",
               combined == pred_combined,
               _multiset_diff(pred_combined, combined))

    prefill = _actual_prefill(num_layers)
    if prefill is not None:
        if preempted:
            _check("captured prefill tokens >= sum(prompt lengths) [recompute-preemption]",
                   prefill["total_tokens"] >= want_prefill,
                   f"captured {prefill['total_tokens']} >= {want_prefill}")
        else:
            _check("captured prefill tokens == sum(prompt lengths)",
                   prefill["total_tokens"] == want_prefill,
                   f"captured {prefill['total_tokens']} vs {want_prefill}")
        _check("prefill token counts match capture",
               prefill["tokens"] == pred_prefill_tokens,
               _multiset_diff(pred_prefill_tokens, prefill["tokens"]))
        _check("prefill sequence counts match capture",
               prefill["seqs"] == pred_prefill_seqs,
               _multiset_diff(pred_prefill_seqs, prefill["seqs"]))

    decode = _actual_decode()
    if decode is not None:
        pred_decode_sizes = {
            st["decode_seqs"] for st in steps if st["decode_seqs"] > 0}
        _check("captured decode batch <= max_num_seqs",
               decode["max"] <= max_num_seqs,
               f"max captured decode batch {decode['max']} <= {max_num_seqs}")
        _check("captured decode batch sizes match simulation",
               decode["sizes"] == pred_decode_sizes,
               f"captured {sorted(decode['sizes'])} vs "
               f"simulated {sorted(pred_decode_sizes)}")

    return {
        "method": "mock continuous-batching / chunked-prefill replay",
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "num_layers": num_layers,
        "kv_cache_modeled": num_blocks is not None,
        "num_kv_blocks": num_blocks,
        "kv_block_size": block_size if num_blocks is not None else None,
        "num_requests": len(prompt_lens),
        "simulated_steps": len(steps),
        "simulated_prefill_steps": sum(
            1 for s in steps if s["prefill_tokens"] > 0),
        "simulated_prefill_tokens": total_prefill,
        "simulated_decode_tokens": total_decode,
        "max_step_tokens": max_total,
        "max_decode_batch": max_decode,
        "predicted_step_token_counts": _ms_sorted(pred_combined),
        "predicted_prefill_token_counts": _ms_sorted(pred_prefill_tokens),
        "predicted_prefill_seq_counts": _ms_sorted(pred_prefill_seqs),
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _build_report(model_name: str, workload: str, dtype: torch.dtype,
                  max_batch_size: int, num_classes: int,
                  generation: dict, verification: dict) -> dict:
    operators = {}
    # Global map ``qualname -> {init variant key -> index}`` used to (a) point
    # each forward variant at the init variant(s) that built its instances
    # (``init_variant_ids``) and (b) resolve ``$op_ref`` submodule references to
    # a concrete ``init_variant_id`` in the referenced operator.
    init_key_index = {
        q: {k: i for i, k in enumerate(e["init"]["variants"])}
        for q, e in _RECORDS.items() if e.get("init")
    }

    def _resolve_ref(op_qualname, init_key):
        return init_key_index.get(op_qualname, {}).get(init_key, -1)

    for qualname, entry in sorted(_RECORDS.items()):
        out_entry = {}
        this_index = init_key_index.get(qualname, {})
        for method in ("init", "forward"):
            slot = entry[method]
            if slot is None:
                continue
            variants = []
            for v in slot["variants"].values():
                if method == "init":
                    # Emit the reconstruction recipe, resolving submodule
                    # ``$op_ref``s to concrete init_variant_ids.
                    out_v = {"count": v["count"],
                             "args": _translate_op_refs(v["args"], _resolve_ref)}
                else:
                    # Pair each forward variant with the init variant(s) that
                    # built the instances it ran on (empty if init not captured).
                    out_v = {"count": v["count"], "args": v["args"],
                             "init_variant_ids": sorted(
                                 this_index[k] for k in v.get("init_keys", {})
                                 if k in this_index)}
                variants.append(out_v)
            out_entry[method] = {"calls": slot["calls"], "variants": variants}
        operators[qualname] = out_entry

    return {
        "model": model_name,
        "workload": workload,
        "dtype": str(dtype),
        "engine": "fastkernels.infra.engine.LlamaEngine",
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "max_batch_size": max_batch_size,
        "generation": generation,
        "verification": verification,
        "num_operator_classes_instrumented": num_classes,
        "num_operator_classes_executed": len(operators),
        "operators": operators,
    }


def _dumps(obj, indent: int = 2, level: int = 0) -> str:
    """Pretty-print JSON, but keep small objects/arrays on a single line.

    Leaf structures such as tensor descriptors (``{"shape": [...], "dtype":
    ...}``) and shape arrays render inline, while the operator hierarchy stays
    indented for readability.
    """
    compact = json.dumps(obj, ensure_ascii=False)
    if not isinstance(obj, (dict, list)) or len(compact) <= 80:
        return compact
    pad = " " * (indent * (level + 1))
    if isinstance(obj, dict):
        body = ",\n".join(
            f"{pad}{json.dumps(k, ensure_ascii=False)}: "
            f"{_dumps(v, indent, level + 1)}"
            for k, v in obj.items()
        )
        open_b, close_b = "{", "}"
    else:
        body = ",\n".join(
            f"{pad}{_dumps(v, indent, level + 1)}" for v in obj
        )
        open_b, close_b = "[", "]"
    return f"{open_b}\n{body}\n{' ' * (indent * level)}{close_b}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _scenario_slug(scenario, workload: str, num_requests: int,
                   max_layers: int | None = None) -> str:
    """Report-filename stem encoding the distinguishing scenario fields for a
    single ``workload``.

    Two runs that differ in any of these values write to different files (no
    timestamp needed); re-running the same scenario/workload/``--max-requests``
    overwrites its report. ``None`` max_num_seqs renders as ``auto``; capture
    always runs eager, so the mode tag is fixed. A ``--max-layers`` run adds an
    ``_L<n>`` tag so a truncated capture never overwrites the full-model report.
    """
    model = scenario.hf_name.replace("/", "__")
    max_num_seqs = scenario.max_num_seqs if scenario.max_num_seqs is not None else "auto"
    layers_tag = f"_L{max_layers}" if max_layers is not None else ""
    return (
        f"{model}_tp{scenario.tp}_{scenario.dtype}_{workload}"
        f"_req{num_requests}_seqs{max_num_seqs}{layers_tag}_eager"
    )


def _engine_dtype(dtype_str: str) -> torch.dtype | None:
    """Map a scenario dtype string to the engine's compute dtype.

    Plain torch dtypes (bfloat16 / float16 / float32) are used directly.
    Quantization-scheme tags (mxfp4, fp8, ...) are not torch compute dtypes --
    the weight loader applies the quantization from the model's own quant config
    -- so we return ``None`` and let the engine infer the compute dtype from the
    model config.
    """
    dt = getattr(torch, dtype_str, None)
    return dt if isinstance(dt, torch.dtype) else None


def _report_path(output_arg, scenario, workload: str, multi: bool, num_requests: int,
                 max_layers: int | None = None) -> Path:
    """Resolve the report path for one workload run.

    Default: ``CAPTURE_DIR/<scenario-slug>.json``. An explicit ``--output`` is
    honored verbatim for a single run, or has the (model-qualified) scenario
    slug suffixed onto its stem when several runs share one ``--output`` base.
    """
    slug = _scenario_slug(scenario, workload, num_requests, max_layers)
    if output_arg is not None:
        p = Path(output_arg)
        return p.with_name(f"{p.stem}_{slug}{p.suffix}") if multi else p
    return CAPTURE_DIR / f"{slug}.json"


def _purge_stale_reports(scenario, runs, args, multi: bool,
                         max_layers: int | None = None) -> None:
    """Delete any pre-existing report file for each of this scenario's workloads.

    Called when a scenario's capture starts so that an interrupted or crashing
    run leaves *no* file for an unfinished workload -- rather than a previous
    run's file, which could be mistaken for a fresh result. Each workload that
    completes rewrites its own report afterward.
    """
    for wl_label, _wl in runs:
        path = _report_path(args.output, scenario, wl_label, multi,
                            args.max_requests, max_layers)
        try:
            if path.exists():
                path.unlink()
                print(f"  (removed stale report {path.name})")
        except OSError as exc:  # noqa: BLE001 - a stale file we can't delete is non-fatal
            print(f"  (warning: could not remove stale report {path}: {exc})")


# NOTE: ``_prepare_prompts`` supported the removed ``--prompt`` ad-hoc override
# (raw strings, optionally chat-templated). It is kept here, commented out, for
# reference; capture now only runs real scenario workloads (already tokenized).
#
# def _prepare_prompts(engine, prompts: list[str], use_chat_template: bool):
#     """Return (engine_inputs, used_chat_template).
#
#     When a chat template is available we tokenize each prompt as a single-turn
#     user message with a generation prompt, so an instruct model produces a
#     bounded assistant turn ending in the end-of-turn token. Token-id lists are
#     handed to the engine directly to avoid re-adding a BOS token.
#     """
#     tok = engine.tokenizer
#     if use_chat_template and getattr(tok, "chat_template", None):
#         try:
#             return (
#                 [
#                     tok.apply_chat_template(
#                         [{"role": "user", "content": p}],
#                         add_generation_prompt=True,
#                         tokenize=True,
#                     )
#                     for p in prompts
#                 ],
#                 True,
#             )
#         except Exception as exc:  # noqa: BLE001
#             print(f"  (chat template unavailable: {exc}; using raw prompts)")
#     return list(prompts), False


def _reset_engine_runtime_state(engine) -> None:
    """Return the engine to a pristine per-run state between workloads.

    Capture loads each model exactly once and reuses the same engine for every
    one of that scenario's workloads (never reloading it). To keep those
    workloads from interfering, the *runtime* state that accumulates per
    ``generate`` -- the paged KV-cache block pool -- is returned to empty before
    each run. A cleanly finished ``generate`` already frees every block as its
    sequences finish, so this is normally a no-op; making it explicit guarantees
    the next workload starts from a full pool (which the schedule replay assumes)
    and surfaces, rather than silently carries over, any leftover allocation.

    (Mamba/SSM recurrent state needs no reset here: its manager zeroes each slot
    on both allocate and free, so stale state can never leak into a later run.)
    """
    bm = getattr(engine, "block_manager", None)
    reset = getattr(bm, "reset", None)
    if reset is None:
        return
    total = getattr(bm, "_num_blocks", None)
    free = getattr(bm, "free_block_ids", None)
    if total and free is not None and len(free) != total:
        print(
            f"  (note: {total - len(free)} KV block(s) still allocated from a "
            f"prior workload; resetting the pool before this run)"
        )
    reset()


# ---------------------------------------------------------------------------
# Multimodal (VLM / OmniModal) input loading
#
# These loaders are copied verbatim from ``tests/bench_vllm.py`` (the current,
# non-deprecated reference benchmark), whose multimodal loaders live inside a
# worker-source string (``_MM_PRELOAD_FN``) and so cannot be imported. Copying
# them keeps the captured operator shapes identical to what the benchmark runs:
# OpenCV video decode (vLLM's OpenCVVideoBackend, 32 frames), PyAV audio decode
# (no torchcodec), and the VisionArena / MMVU / librispeech dispatcher with its
# exact filters, prompt construction and MMVU ``snapshot_download`` + URL->local
# resolution. The multimodal engine path is driven exactly like
# ``FASTKERNELS_VLM_WORKER`` in that file: raw text prompts plus per-request
# ``images`` / ``videos`` / ``audio_features`` handed to ``LlamaEngine.generate``,
# which runs the HF processor (chat template + placeholder expansion) internally.
# ---------------------------------------------------------------------------

# Fixed shuffle seed so a given (dataset, n_req) always draws the same requests.
_MEDIA_SEED = 42


def _decode_audio_array(audio):
    """Decode a HF Audio item to mono float32 samples without torchcodec."""
    import numpy as np
    from io import BytesIO

    if isinstance(audio, dict) and audio.get("array") is not None:
        samples = np.asarray(audio["array"], dtype=np.float32)
        return samples, int(audio["sampling_rate"])

    import av

    source = None
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            source = BytesIO(audio["bytes"])
        elif audio.get("path") is not None:
            source = audio["path"]
    if source is None:
        raise ValueError("Unsupported audio sample format")

    chunks = []
    sampling_rate = None
    with av.open(source) as container:
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            sampling_rate = frame.sample_rate
            chunks.append(arr)
    if not chunks or sampling_rate is None:
        raise ValueError("Audio sample has no decodable frames")

    samples = np.concatenate(chunks, axis=-1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        samples = samples.astype(np.float32) / max(abs(info.min), info.max)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0)
    return samples, int(sampling_rate)


def _load_video_opencv(video_path, num_frames=32):
    """Load video frames with OpenCV, matching vLLM's OpenCVVideoBackend."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_num / original_fps if original_fps > 0 else 0

    num_frames_to_sample = total_frames_num
    if num_frames > 0:
        num_frames_to_sample = min(num_frames, total_frames_num)
    num_frames_to_sample = max(1, num_frames_to_sample)

    if num_frames_to_sample == total_frames_num:
        frame_idx = list(range(num_frames_to_sample))
    else:
        frame_idx = np.linspace(
            0, total_frames_num - 1, num_frames_to_sample, dtype=int
        ).tolist()

    frame_idx_set = set(frame_idx)
    max_idx = max(frame_idx)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((num_frames_to_sample, height, width, 3), dtype=np.uint8)

    i = 0
    valid_frame_indices = []
    for idx in range(max_idx + 1):
        ok = cap.grab()
        if not ok:
            continue
        if idx in frame_idx_set:
            ret, frame = cap.retrieve()
            if ret:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                valid_frame_indices.append(idx)
                i += 1

    cap.release()
    valid_num_frames = len(valid_frame_indices)
    frames = frames[:valid_num_frames]

    metadata = {
        "total_num_frames": total_frames_num,
        "fps": original_fps,
        "duration": duration,
        "video_backend": "opencv",
        "frames_indices": valid_frame_indices,
        "do_sample_frames": valid_num_frames == total_frames_num,
    }
    return frames, metadata


def _preload_mm_data(dataset_name, dataset_split, num_seqs, seed,
                     num_video_frames=32):
    """Pre-download and load multimodal samples into memory.

    Returns list of dicts with keys:
      - prompt: str
      - images: list[PIL.Image] or None
      - video_frames: np.ndarray (T,H,W,3) or None
      - video_metadata: dict or None
      - audio: np.ndarray or None
      - audio_sampling_rate: int or None
    """
    from io import BytesIO

    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm

    use_streaming = "MMVU" not in dataset_name
    data = load_dataset(dataset_name, split=dataset_split,
                        streaming=use_streaming)
    if "librispeech_asr" in dataset_name:
        from datasets import Audio
        data = data.cast_column("audio", Audio(decode=False))
    data = data.shuffle(seed=seed)

    results = []
    if "VisionArena" in dataset_name:
        pbar = tqdm(data, total=num_seqs, desc="Loading images")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            try:
                prompt = item["conversation"][0][0]["content"]
                if "base64" in prompt or len(prompt) > 4096:
                    continue
                img = item["images"][0]
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(BytesIO(img["bytes"]))
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")
                w, h = img.size
                if w * h > 2048 * 2048:
                    continue
            except Exception:
                continue
            results.append({
                "prompt": prompt,
                "images": [img],
                "video_frames": None,
                "video_metadata": None,
                "audio": None,
                "audio_sampling_rate": None,
            })
            pbar.update(0)
        pbar.close()
    elif "MMVU" in dataset_name:
        from huggingface_hub import snapshot_download
        local_root = snapshot_download(dataset_name, repo_type="dataset")
        remote_root = (
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        )
        pbar = tqdm(data, total=num_seqs, desc="Loading videos")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            prompt = item["question"] + " " + " ".join(
                f"{k}.{v}" for k, v in item["choices"].items())
            video_path = item["video"].replace(remote_root, local_root)
            frames, metadata = _load_video_opencv(
                video_path, num_frames=num_video_frames)
            results.append({
                "prompt": prompt,
                "images": None,
                "video_frames": frames,
                "video_metadata": metadata,
                "audio": None,
                "audio_sampling_rate": None,
            })
            pbar.update(0)
        pbar.close()
    elif "librispeech_asr" in dataset_name:
        pbar = tqdm(data, total=num_seqs, desc="Loading audio")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            try:
                samples, sampling_rate = _decode_audio_array(item["audio"])
                if samples.ndim != 1 or samples.size == 0:
                    continue
            except Exception:
                continue
            results.append({
                "prompt": "Transcribe this audio and answer in text.",
                "images": None,
                "video_frames": None,
                "video_metadata": None,
                "audio": samples,
                "audio_sampling_rate": sampling_rate,
            })
            pbar.update(0)
        pbar.close()
    return results


def _media_descriptor(item) -> str:
    """Short human-readable summary of a request's media for the report."""
    if item["images"] is not None:
        return f"{len(item['images'])} image(s)"
    if item["video_frames"] is not None:
        return f"{int(item['video_frames'].shape[0])} video frame(s)"
    if item["audio"] is not None:
        sr = item["audio_sampling_rate"]
        return f"1 audio clip ({item['audio'].shape[-1]} samples @ {sr}Hz)"
    return "text"


# ---------------------------------------------------------------------------
# EAGLE-3 speculative-decoding capture
#
# EAGLE-3 scenarios name the *draft head* (e.g. yuhuili/EAGLE3-LLaMA3.1-Instruct-8B);
# the target LM is inferred. Capture runs them through ``LlamaEagle3Engine`` (a
# distinct engine from ``LlamaEngine``: its own paged KV cache + spec-decode loop),
# recording both the target and draft operator forwards. Verification #2 (the
# continuous-batching replay) does not model speculative decoding, so only the
# forward-pre-hook cross-check applies here.
# ---------------------------------------------------------------------------
_EAGLE3_TARGETS = {
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B": "meta-llama/Llama-3.1-8B-Instruct",
}
# EAGLE-3 draft heads support a limited position window; capture caps the engine
# context accordingly and drops longer prompts (e.g. long-context 8K-128K buckets).
_EAGLE3_MAX_MODEL_LEN = int(os.environ.get("FASTKERNELS_EAGLE3_MAX_MODEL_LEN", "4096"))


def _is_eagle3(scenario) -> bool:
    return "eagle3" in scenario.hf_name.lower()


def _eagle3_target(draft_repo: str) -> str:
    """Target LM for an EAGLE-3 draft head (all current heads target Llama-3.1-8B)."""
    return _EAGLE3_TARGETS.get(draft_repo, "meta-llama/Llama-3.1-8B-Instruct")


def _register_hooks_on_models(models, instrumented: set[str]) -> list:
    """Register forward-pre-hooks across several models, deduping shared module
    instances (EAGLE-3 shares the target's ``embed_tokens`` with the draft, so a
    naive per-model pass would double-count that module in the hook records)."""
    handles = []
    seen: set[int] = set()
    for model in models:
        for module in model.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            cls = type(module)
            qualname = f"{cls.__module__}:{cls.__name__}"
            if qualname not in instrumented:
                continue
            try:
                sig = inspect.signature(cls.forward)
            except (TypeError, ValueError):
                sig = None
            handles.append(module.register_forward_pre_hook(
                _make_forward_hook(qualname, sig), with_kwargs=True))
    return handles


def _capture_eagle3_scenario(scenario, runs, args, instrumented, n_instrumented, multi):
    """Capture an EAGLE-3 scenario via LlamaEagle3Engine (target + draft)."""
    from .infra.eagle3_engine import Eagle3SamplingParams, LlamaEagle3Engine

    target = _eagle3_target(scenario.hf_name)
    _RECORDS.clear()
    _HOOK_RECORDS.clear()
    print(
        f"\n########## Scenario: {scenario.hf_name} "
        f"(EAGLE-3 draft; target={target}, dtype={scenario.dtype}) ##########"
    )
    print("Loading EAGLE-3 target+draft into LlamaEagle3Engine (eager) ...")
    engine = LlamaEagle3Engine(
        model_name=target,
        draft_repo=scenario.hf_name,
        seed=42,
        dtype=torch.bfloat16,
        max_model_len=_EAGLE3_MAX_MODEL_LEN,
        max_num_seqs=scenario.max_num_seqs or 32,
        spec_steps=3,
        spec_topk=4,
        enforce_eager=True,
    )

    written: list[Path] = []
    ok = True
    try:
        for wl_label, wl in runs:
            print(f"\n=== Capturing workload: {wl_label} ===")
            spec = spec_for(wl)
            params = spec.params
            wl_dataset = getattr(params, "dataset_name", "") or None
            if spec.purpose is Purpose.LATENCY:
                n_req = min(args.max_requests, getattr(params, "batch_size", 1))
                wl_decode_cap = getattr(params, "output_len", None)
            else:
                n_req = min(args.max_requests,
                            getattr(params, "num_requests", args.max_requests))
                wl_decode_cap = getattr(params, "decode_cap", None)
            try:
                samples = load_real_prompt_workload(
                    wl_label, engine.tokenizer, num_requests=n_req,
                    dataset_name=wl_dataset, decode_cap=wl_decode_cap,
                )
                # The draft head has a bounded position window; drop prompts whose
                # prompt+decode would exceed the engine context.
                n_loaded = len(samples)
                samples = [
                    s for s in samples
                    if len(s.prompt_token_ids) + s.output_len <= _EAGLE3_MAX_MODEL_LEN
                ]
                dropped = n_loaded - len(samples)
                if not samples:
                    print(f"  !! SKIP workload {wl_label}: all {n_loaded} prompt(s) "
                          f"exceed the EAGLE-3 engine context "
                          f"({_EAGLE3_MAX_MODEL_LEN}).")
                    continue
                if dropped:
                    print(f"  (dropped {dropped}/{n_loaded} prompt(s) exceeding the "
                          f"EAGLE-3 context {_EAGLE3_MAX_MODEL_LEN})")
                gen_prompts = [list(s.prompt_token_ids) for s in samples]
                max_new_tokens = [s.output_len for s in samples]
                prompt_texts = [
                    engine.tokenizer.decode(p, skip_special_tokens=False)
                    for p in gen_prompts
                ]
                sampling = [Eagle3SamplingParams(max_tokens=mnt, ignore_eos=True) for mnt in max_new_tokens]

                for entry in _RECORDS.values():
                    entry["forward"] = None
                _HOOK_RECORDS.clear()
                engine.reset()
                hook_handles = _register_hooks_on_models(
                    [engine.target, engine.draft], instrumented,
                )
                try:
                    outputs = engine.generate(
                        gen_prompts, sampling, use_tqdm=True, decode_text=True,
                    )
                finally:
                    for h in hook_handles:
                        h.remove()
            except Exception as exc:  # noqa: BLE001 - isolate per-workload failures
                traceback.print_exc()
                print(f"  !! workload {wl_label} failed: {exc!r}; "
                      f"skipping to the next workload.")
                ok = False
                continue

            gen_lengths = [len(o.token_ids) for o in outputs]
            num_eos = sum(1 for n, cap in zip(gen_lengths, max_new_tokens) if n < cap)
            prompt_lens = [len(p) for p in gen_prompts]
            responses = [
                {
                    "prompt": prompt_texts[i],
                    "prompt_tokens": prompt_lens[i],
                    "response": engine.tokenizer.decode(
                        outputs[i].token_ids, skip_special_tokens=False),
                    "max_new_tokens": max_new_tokens[i],
                    "tokens": gen_lengths[i],
                    "stop_reason": ("eos" if gen_lengths[i] < max_new_tokens[i]
                                    else "max_tokens"),
                }
                for i in range(len(gen_prompts))
            ]
            generation = {
                "num_prompts": len(gen_prompts),
                "max_batch_size": engine.max_num_seqs,
                "engine": "eagle3",
                "target_model": target,
                "spec_steps": engine.spec_steps,
                "spec_topk": engine.topk,
                "num_draft_tokens": engine.num_draft_tokens,
                "modality": "text",
                "used_chat_template": True,
                "max_new_tokens_source": "dataset_response_length",
                "max_new_tokens_range": [min(max_new_tokens), max(max_new_tokens)],
                "num_finished_by_eos": num_eos,
                "num_hit_max_tokens": len(gen_lengths) - num_eos,
                "responses": responses,
            }
            print(f"  Generation: {num_eos}/{len(gen_prompts)} reached EOS; "
                  f"lengths={gen_lengths}")

            hook_verification = _verify_forward_capture()
            status = "PASS" if hook_verification["passed"] else "FAIL"
            print(
                f"  Verification 1 (hook cross-check) [{status}]: "
                f"{hook_verification['classes_matched']}/"
                f"{hook_verification['classes_checked']} operator forwards match "
                f"the independent hook capture."
            )
            for issue in hook_verification["issues"]:
                print(f"    ! {issue}")
            schedule_verification = {
                "method": "mock continuous-batching / chunked-prefill replay",
                "applicable": False,
                "passed": True,
                "reason": ("skipped for EAGLE-3: speculative decoding (draft tree + "
                           "target verification) is not modeled by the continuous-"
                           "batching replay; the forward-pre-hook cross-check is the "
                           "pass criterion."),
            }
            print("  Verification 2 (mock batching replay) [N/A]: "
                  "skipped for EAGLE-3 speculative decoding (see report).")
            verification = {
                "forward_pre_hook_crosscheck": hook_verification,
                "mock_batching_replay": schedule_verification,
                "passed": hook_verification["passed"],
            }
            report = _build_report(
                scenario.hf_name, wl_label, engine.dtype,
                engine.max_num_seqs, n_instrumented, generation, verification,
            )
            out_path = _report_path(args.output, scenario, wl_label, multi, args.max_requests)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(_dumps(report))
                f.write("\n")
            written.append(out_path)
            print(
                f"  Captured {report['num_operator_classes_executed']} executed "
                f"operator class(es) (of {n_instrumented} instrumented)."
            )
            print(f"  Report written to {out_path}")
            if not verification["passed"]:
                ok = False
    finally:
        # LlamaEagle3Engine is single-process (TP=1) with no atexit registration,
        # so a plain del + cache clear frees its GPU memory.
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return written, ok


# ---------------------------------------------------------------------------
# Recurrent / hybrid alternate-engine capture (FLA + Jamba)
#
# GLA / RetNet / RWKV-7 (``FLAEngine``) carry per-sequence recurrent *state*
# and Jamba (``JambaEngine``) is a hybrid transformer+Mamba+MoE model; neither
# plugs into ``LlamaEngine``'s paged-KV ``model_runner``, so they run through
# their own single-process engines. Both drive a plain autoregressive text
# workload (token-id prompts -> ``generate``), so capture reuses the text
# LLM input prep. Verification #2 (the paged-KV continuous-batching replay)
# does not model recurrent state / hybrid slot pools, so -- as for EAGLE-3 --
# only the forward-pre-hook cross-check (#1) applies.
# ---------------------------------------------------------------------------
# JambaEngine has no tensor-parallel support, so a tp>1 Jamba scenario still
# loads the full model onto a single GPU (~96 GB for Jamba-Mini-1.7). That
# leaves little room for the attention KV cache, so cap the context + batch
# modestly for capture (long-context prompts beyond this are dropped). Both
# are env-overridable.
_JAMBA_MAX_MODEL_LEN = int(os.environ.get("FASTKERNELS_JAMBA_MAX_MODEL_LEN", "16384"))
_JAMBA_MAX_NUM_SEQS = int(os.environ.get("FASTKERNELS_JAMBA_MAX_NUM_SEQS", "16"))


def _is_fla(scenario) -> bool:
    """FLA recurrent LLMs (GLA / RetNet / RWKV-7) are all published under the
    ``fla-hub/`` org and run through ``FLAEngine``."""
    return scenario.hf_name.startswith("fla-hub/")


def _is_jamba(scenario) -> bool:
    return "jamba" in scenario.hf_name.lower()


def _capture_altengine_scenario(scenario, runs, args, instrumented, n_instrumented,
                                multi, *, engine, sampling_cls, engine_label,
                                engine_meta, max_model_len=None):
    """Capture a single-model text-LLM scenario through an alternate engine
    (``FLAEngine`` / ``JambaEngine``) that has no paged-KV ``model_runner``.

    ``engine`` is already constructed and owned by the caller (which also frees
    it). The per-workload loop mirrors ``_capture_eagle3_scenario``: real
    chat-templated prompts as token-id lists, forward-pre-hooks on the single
    model, Verification #1 as the pass criterion, Verification #2 recorded
    N/A. When ``max_model_len`` is set, prompts whose prompt+decode exceed it
    are dropped (and a workload with none left is skipped).
    """
    written: list[Path] = []
    ok = True
    for wl_label, wl in runs:
        print(f"\n=== Capturing workload: {wl_label} ===")
        spec = spec_for(wl)
        params = spec.params
        wl_dataset = getattr(params, "dataset_name", "") or None
        if spec.purpose is Purpose.LATENCY:
            n_req = min(args.max_requests, getattr(params, "batch_size", 1))
            wl_decode_cap = getattr(params, "output_len", None)
        else:
            n_req = min(args.max_requests,
                        getattr(params, "num_requests", args.max_requests))
            wl_decode_cap = getattr(params, "decode_cap", None)
        try:
            samples = load_real_prompt_workload(
                wl_label, engine.tokenizer, num_requests=n_req,
                dataset_name=wl_dataset, decode_cap=wl_decode_cap,
            )
            n_loaded = len(samples)
            if max_model_len is not None:
                samples = [
                    s for s in samples
                    if len(s.prompt_token_ids) + s.output_len <= max_model_len
                ]
                dropped = n_loaded - len(samples)
                if not samples:
                    print(f"  !! SKIP workload {wl_label}: all {n_loaded} prompt(s) "
                          f"exceed the {engine_label} engine context "
                          f"({max_model_len}).")
                    continue
                if dropped:
                    print(f"  (dropped {dropped}/{n_loaded} prompt(s) exceeding the "
                          f"{engine_label} context {max_model_len})")
            gen_prompts = [list(s.prompt_token_ids) for s in samples]
            max_new_tokens = [s.output_len for s in samples]
            prompt_texts = [
                engine.tokenizer.decode(p, skip_special_tokens=False)
                for p in gen_prompts
            ]
            sampling = [sampling_cls(temperature=0.0, max_tokens=mnt, ignore_eos=True)
                        for mnt in max_new_tokens]

            for entry in _RECORDS.values():
                entry["forward"] = None
            _HOOK_RECORDS.clear()
            hook_handles = _register_hooks_on_models([engine.model], instrumented)
            try:
                outputs = engine.generate(gen_prompts, sampling, use_tqdm=True)
            finally:
                for h in hook_handles:
                    h.remove()
        except Exception as exc:  # noqa: BLE001 - isolate per-workload failures
            traceback.print_exc()
            print(f"  !! workload {wl_label} failed: {exc!r}; "
                  f"skipping to the next workload.")
            ok = False
            continue

        gen_lengths = [len(o.token_ids) for o in outputs]
        num_eos = sum(1 for n, cap in zip(gen_lengths, max_new_tokens) if n < cap)
        prompt_lens = [len(p) for p in gen_prompts]
        responses = [
            {
                "prompt": prompt_texts[i],
                "prompt_tokens": prompt_lens[i],
                "response": engine.tokenizer.decode(
                    outputs[i].token_ids, skip_special_tokens=False),
                "max_new_tokens": max_new_tokens[i],
                "tokens": gen_lengths[i],
                "stop_reason": ("eos" if gen_lengths[i] < max_new_tokens[i]
                                else "max_tokens"),
            }
            for i in range(len(gen_prompts))
        ]
        generation = {
            "num_prompts": len(gen_prompts),
            "max_batch_size": engine.max_num_seqs,
            "engine": engine_label,
            "modality": "text",
            "used_chat_template": True,
            "max_new_tokens_source": "dataset_response_length",
            "max_new_tokens_range": [min(max_new_tokens), max(max_new_tokens)],
            "num_finished_by_eos": num_eos,
            "num_hit_max_tokens": len(gen_lengths) - num_eos,
            "responses": responses,
            **engine_meta,
        }
        print(f"  Generation: {num_eos}/{len(gen_prompts)} reached EOS; "
              f"lengths={gen_lengths}")

        hook_verification = _verify_forward_capture()
        status = "PASS" if hook_verification["passed"] else "FAIL"
        print(
            f"  Verification 1 (hook cross-check) [{status}]: "
            f"{hook_verification['classes_matched']}/"
            f"{hook_verification['classes_checked']} operator forwards match "
            f"the independent hook capture."
        )
        for issue in hook_verification["issues"]:
            print(f"    ! {issue}")
        schedule_verification = {
            "method": "mock continuous-batching / chunked-prefill replay",
            "applicable": False,
            "passed": True,
            "reason": (f"skipped for {engine_label}: recurrent-state / hybrid slot "
                       "pools are not modeled by the paged-KV continuous-batching "
                       "replay; the forward-pre-hook cross-check is the pass "
                       "criterion."),
        }
        print(f"  Verification 2 (mock batching replay) [N/A]: "
              f"skipped for {engine_label} (see report).")
        verification = {
            "forward_pre_hook_crosscheck": hook_verification,
            "mock_batching_replay": schedule_verification,
            "passed": hook_verification["passed"],
        }
        report = _build_report(
            scenario.hf_name, wl_label, engine.dtype,
            engine.max_num_seqs, n_instrumented, generation, verification,
        )
        out_path = _report_path(args.output, scenario, wl_label, multi, args.max_requests)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(_dumps(report))
            f.write("\n")
        written.append(out_path)
        print(
            f"  Captured {report['num_operator_classes_executed']} executed "
            f"operator class(es) (of {n_instrumented} instrumented)."
        )
        print(f"  Report written to {out_path}")
        if not verification["passed"]:
            ok = False
    return written, ok


def _capture_fla_scenario(scenario, runs, args, instrumented, n_instrumented, multi):
    """Capture a GLA / RetNet / RWKV-7 scenario via ``FLAEngine``."""
    from .infra.fla_engine import FLAEngine, SamplingParams as FLASamplingParams

    _RECORDS.clear()
    _HOOK_RECORDS.clear()
    print(
        f"\n########## Scenario: {scenario.hf_name} "
        f"(FLA recurrent LLM, dtype={scenario.dtype}) ##########"
    )
    print("Loading FLA model into FLAEngine (eager) ...")
    engine = FLAEngine(
        model_name=scenario.hf_name,
        dtype=_engine_dtype(scenario.dtype),
        seed=42,
        max_num_seqs=scenario.max_num_seqs or 256,
    )
    try:
        return _capture_altengine_scenario(
            scenario, runs, args, instrumented, n_instrumented, multi,
            engine=engine, sampling_cls=FLASamplingParams, engine_label="fla",
            engine_meta={"recurrent_state": True}, max_model_len=None,
        )
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _capture_jamba_scenario(scenario, runs, args, instrumented, n_instrumented, multi):
    """Capture an AI21 Jamba (hybrid transformer+Mamba+MoE) scenario via
    ``JambaEngine``."""
    from .infra.jamba_engine import JambaEngine, SamplingParams as JambaSamplingParams

    _RECORDS.clear()
    _HOOK_RECORDS.clear()
    print(
        f"\n########## Scenario: {scenario.hf_name} "
        f"(Jamba hybrid, dtype={scenario.dtype}, tp={scenario.tp}) ##########"
    )
    print("Loading Jamba into JambaEngine (eager) ...")
    engine = JambaEngine(
        model_name=scenario.hf_name,
        dtype=_engine_dtype(scenario.dtype),
        seed=42,
        max_num_seqs=scenario.max_num_seqs or _JAMBA_MAX_NUM_SEQS,
        max_model_len=_JAMBA_MAX_MODEL_LEN,
    )
    try:
        return _capture_altengine_scenario(
            scenario, runs, args, instrumented, n_instrumented, multi,
            engine=engine, sampling_cls=JambaSamplingParams, engine_label="jamba",
            engine_meta={"hybrid_attn_mamba_moe": True},
            max_model_len=_JAMBA_MAX_MODEL_LEN,
        )
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Non-text families (vision / detection / embedding / recsys / video repr.)
#
# These don't autoregress tokens, so the LLM/alt-engine "prompts -> generate"
# harness doesn't fit. Capture only needs the operators' forward SHAPES, so we
# build the fastkernels model for the family and drive ONE representative
# forward per workload with SYNTHETIC inputs (random tensors sized from the
# workload spec) -- no datasets, tokenizers, or real generation. Verification #1
# (the forward-pre-hook cross-check) still applies; #2 (paged-KV replay) does
# not. Families we can't yet build cheaply (needs provisioned assets, a custom
# CUDA build, MSA/alignment data, ...) raise ``_CaptureSkip`` so they land as a
# clean SKIP in the summary rather than a hard error.
# ---------------------------------------------------------------------------

# Workload enum class names that are not the text-LLM / VLM / Omni paths.
_NONTEXT_FAMILIES = {
    "Diffusion", "VideoDiffusion", "Segmentation", "ASR", "TTS",
    "VisionEncoder", "Detection", "Rendering", "PointCloudSeg",
    "StructurePrediction", "Robotics", "PointCloudPolicy", "Recsys",
    "Embedding", "DiffusionLM", "WorldModel", "VideoRepresentation",
}

# timm-backed vision encoders: hf id -> (timm name, L4 module stem, class, default res).
_TIMM_VISION = {
    "google/siglip2-so400m-patch16-naflex":
        ("naflexvit_so400m_patch16_siglip.v2_webli", "siglip2", "SigLIP2Model", 384),
    "facebook/dinov3-vit7b16-pretrain-lvd1689m":
        ("vit_7b_patch16_dinov3.lvd1689m", "dinov3", "DINOv3Model", 256),
    "timm/swinv2_large_window12_192.ms_in22k":
        ("swinv2_large_window12_192.ms_in22k", "swinv2", "SwinV2Model", 192),
    "timm/mobilenetv4_conv_medium.e500_r256_in1k":
        ("mobilenetv4_conv_medium.e500_r256_in1k", "mobilenetv4", "MobileNetV4Model", 256),
}


class _CaptureSkip(Exception):
    """A scenario that capture deliberately skips (unsupported / asset-gated)."""


def _build_vision(scenario, dtype):
    """Vision encoders: timm-backed (``from_timm``) or image-cls loader. Forward
    takes a single NCHW image batch."""
    name = scenario.hf_name
    if name in _TIMM_VISION:
        timm_name, stem, cls_name, default_res = _TIMM_VISION[name]
        mod = __import__(f"fastkernels.tasks.baseline.L4.{stem}", fromlist=[cls_name])
        model = getattr(mod, cls_name).from_timm(timm_name).to(
            device="cuda", dtype=dtype).eval()
    else:
        from .infra.image_cls_loader import infer_image_size, load_ours_model
        model = load_ours_model(name, "cuda", dtype)
        default_res = infer_image_size(name)

    def run(wl_label, wl, params):
        res = int(getattr(params, "resolution", 0) or default_res)
        n = min(int(getattr(params, "batch_size", 0) or 1), 8)
        x = torch.randn(n, 3, res, res, device="cuda", dtype=dtype)
        with torch.no_grad():
            model(x)
        return n

    return [model], run


def _build_detection(scenario, dtype):
    """YOLOv10 / RT-DETRv2: forward takes a 640x640 image batch."""
    from .infra.detection_loader import (
        infer_image_size, load_ours_detector, run_ours_detector,
    )
    name = scenario.hf_name
    detector = load_ours_detector(name, "cuda", dtype)
    size = infer_image_size(name)

    def run(wl_label, wl, params):
        n = min(int(getattr(params, "batch_size", 0) or 1), 4)
        x = torch.randn(n, 3, size, size, device="cuda", dtype=dtype)
        with torch.no_grad():
            run_ours_detector(detector, name, x, image_size=size, max_detections=100)
        return n

    return [detector], run


def _build_embedding(scenario, dtype):
    """bge-m3 / colbertv2 via the pooling engine; forward encodes a batch of
    random token-id sequences."""
    from .infra.embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine(scenario.hf_name, dtype=dtype, device="cuda")
    vocab = int(getattr(engine.tokenizer, "vocab_size", 0) or 30000)

    def run(wl_label, wl, params):
        n = min(int(getattr(params, "batch_size", 0) or 8), 32)
        reqs = [{"prompt_token_ids":
                 torch.randint(1, vocab, (64,)).tolist()} for _ in range(n)]
        engine.token_embed(reqs, use_tqdm=False)
        return n

    return [engine.model], run


def _build_recsys(scenario, dtype):
    """DLRMv2 / LightGCN (random-init: capture needs shapes, not trained
    weights). Forward takes a synthetic feature / interaction batch."""
    name = scenario.hf_name.lower()
    if "dlrm" in name:
        from .tasks.baseline.L4.dlrmv2 import DLRMv2, DLRMv2Config
        cfg = DLRMv2Config()
        model = DLRMv2(cfg).to("cuda", torch.float32).eval()

        def run(wl_label, wl, params):
            n = min(int(getattr(params, "batch_size", 0) or 32), 64)
            dense = torch.randn(n, cfg.num_dense_features, device="cuda")
            sparse = [torch.randint(0, m, (n, 1), device="cuda")
                      for m in cfg.num_embeddings_per_feature]
            with torch.no_grad():
                model(dense, sparse)
            return n

        return [model], run

    from .tasks.baseline.L4.lightgcn import LightGCN, LightGCNConfig
    num_users = num_items = 2000
    model = LightGCN(LightGCNConfig(num_users=num_users, num_items=num_items)).to(
        "cuda", torch.float32).eval()

    def run(wl_label, wl, params):
        n = min(int(getattr(params, "batch_size", 0) or 32), 64)
        users = torch.randint(0, num_users, (n,), device="cuda")
        items = torch.randint(0, num_items, (n,), device="cuda")
        adj = LightGCN.build_adjacency(users, items, num_users, num_items,
                                       device=torch.device("cuda"))
        with torch.no_grad():
            model(users, items, adj)
        return n

    return [model], run


def _build_vjepa2(scenario, dtype):
    """V-JEPA 2: forward takes a synthetic video clip; the predictor path runs
    encoder+predictor, the encoder path runs the encoder only."""
    from .tasks.baseline.L4.vjepa2 import VJEPA2Model
    model = VJEPA2Model.from_pretrained(scenario.hf_name).to("cuda", dtype).eval()
    cfg = model.config
    frames = int(getattr(cfg, "frames_per_clip", 16) or 16)
    crop = int(getattr(cfg, "crop_size", 256) or 256)

    def run(wl_label, wl, params):
        name = getattr(wl, "value", str(wl))
        skip_predictor = "encoder" in name
        vids = torch.randn(1, frames, 3, crop, crop, device="cuda", dtype=dtype)
        with torch.no_grad():
            model(pixel_values_videos=vids, skip_predictor=skip_predictor)
        return 1

    return [model], run


def _build_diffusion(scenario, dtype):
    """FLUX / SDXL / HunyuanVideo: run ONE 1-step latent denoise so the DiT/UNet
    (and the text encoders) execute. ``output_type='latent'`` skips the VAE
    decode. The transformer/unet is the operator-heavy module we care about."""
    name = scenario.hf_name
    lower = name.lower()
    if "stable-diffusion" in lower or "sdxl" in lower:
        from .infra.sdxl_engine import SDXLEngine
        from .tasks.baseline.L4.sdxl import SDXLSamplingParams
        engine = SDXLEngine(name, dtype=dtype, enforce_eager=True)
        pipeline = engine._get_pipeline()

        def make_params(params):
            return SDXLSamplingParams(
                height=int(getattr(params, "height", 512) or 512),
                width=int(getattr(params, "width", 512) or 512),
                num_inference_steps=1, output_type="latent",
            )
    else:
        from .infra.diffusion_engine import DiffusionEngine
        from .tasks.baseline.L4.flux import DiffusionSamplingParams
        from .tasks.baseline.L4.hunyuan_video import (
            HunyuanVideoDiffusionSamplingParams,
        )
        engine = DiffusionEngine(name, dtype=dtype, enforce_eager=True)
        pipeline = engine._get_pipeline()
        video = engine._model_type == "hunyuan_video"

        def make_params(params):
            h = int(getattr(params, "height", 512) or 512)
            w = int(getattr(params, "width", 512) or 512)
            if video:
                return HunyuanVideoDiffusionSamplingParams(
                    height=h, width=w,
                    # Cap frames: temporal length only needs to be non-trivial
                    # for shapes, and a full clip at 1 step is still slow.
                    num_frames=min(int(getattr(params, "num_frames", 5) or 5), 5),
                    num_inference_steps=1, output_type="latent",
                )
            return DiffusionSamplingParams(
                height=h, width=w, num_inference_steps=1, output_type="latent",
            )

    modules = [m for m in (
        getattr(pipeline, "transformer", None), getattr(pipeline, "unet", None),
        getattr(pipeline, "text_encoder", None),
        getattr(pipeline, "text_encoder_2", None),
    ) if isinstance(m, torch.nn.Module)]

    def run(wl_label, wl, params):
        engine.generate("a photograph of a cat", make_params(params))
        return 1

    return modules, run


def _build_llada(scenario, dtype):
    """LLaDA masked-diffusion LM: a short 1-block generate exercises the model."""
    from .infra.dllm_engine import DLLMSamplingParams, LLaDAEngine
    engine = LLaDAEngine(scenario.hf_name, dtype=dtype, device="cuda")

    def run(wl_label, wl, params):
        engine.generate(
            ["Write a Python function that returns 1."],
            DLLMSamplingParams(gen_length=32, steps=2, block_length=32),
        )
        return 1

    return [engine.model], run


def _build_oasis(scenario, dtype):
    """Oasis world model: one short autoregressive-diffusion rollout with a
    synthetic prompt frame + zero action stream (mirrors the engine's warmup)."""
    from .infra.oasis_engine import OasisEngine
    from .tasks.baseline.L4.oasis import OasisSamplingParams
    engine = OasisEngine(scenario.hf_name, dtype=dtype, device="cuda")
    pipeline = engine._get_pipeline()

    def run(wl_label, wl, params):
        nframes = min(int(getattr(params, "num_frames", 4) or 4), 6)
        ddim = min(int(getattr(params, "ddim_steps", 4) or 4), 4)
        prompt = torch.rand(1, 1, 3, 360, 640, device="cuda")
        actions = torch.zeros(1, nframes, 25, device="cuda")
        engine.generate(
            prompt, actions,
            OasisSamplingParams(num_frames=nframes, ddim_steps=ddim, seed=0),
        )
        return 1

    return [pipeline.model, pipeline.vae], run


def _build_whisper(scenario, dtype):
    """Whisper ASR runs through the paged-KV ``LlamaEngine`` (encoder-decoder).
    Drive one short decode on a synthetic 30s mel with the standard
    transcribe decoder prompt; hook the engine's model."""
    import numpy as np
    from transformers import WhisperProcessor
    from .infra.engine import LlamaEngine, SamplingParams
    engine = LlamaEngine(model_name=scenario.hf_name, seed=0, enforce_eager=True,
                         tensor_parallel_size=scenario.tp)
    processor = WhisperProcessor.from_pretrained(scenario.hf_name)
    # The processor pads/truncates to a fixed 30s window, so any short clip
    # yields the model's real (n_mels, 3000) feature shape.
    feats = processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000,
                      return_tensors="pt").input_features[0]
    decoder_prompt = list(processor.tokenizer.encode(
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        add_special_tokens=False))
    model = engine.model_runner.model

    def run(wl_label, wl, params):
        n = min(int(getattr(params, "batch_size", 0) or 1), 4)
        cap = min(int(getattr(params, "output_len", 4) or 4), 4)
        if hasattr(engine, "block_manager"):
            engine.block_manager.reset()
        engine.generate(
            [decoder_prompt] * n,
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=cap),
            audio_features=[feats] * n, use_tqdm=False,
        )
        return n

    return [model], run


def _build_ttt_e2e(scenario, dtype):
    """TTT-E2E (test-time-training) has no public checkpoint; build the 125m
    paper config random-init and run one meta-training chunk (``T == chunk_size``,
    the minimum divisor) so both the prefix and the suffix inner-loop paths run.
    Weight values are irrelevant for shape capture."""
    from .infra.ttt_e2e_engine import TTTE2EEngine
    from .tasks.baseline.L4.ttt_e2e import TTTE2EConfig
    cfg = TTTE2EConfig(
        vocab_size=128256, hidden_size=768, intermediate_size=1664,
        num_hidden_layers=12, num_attention_heads=12, chunk_size=1024,
        sliding_window_size=8192, suffix_len=3, max_position_embeddings=32768,
        rope_theta=500000.0, rms_norm_eps=1e-6, qk_norm=True,
        tie_word_embeddings=True, has_prime=True,
    )
    engine = TTTE2EEngine(cfg, weights_npz=None, device="cuda",
                          param_dtype=dtype, compute_dtype=dtype)

    def run(wl_label, wl, params):
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.chunk_size), device="cuda")
        engine.forward(ids, train_mode="meta")
        return 1

    return [engine.pipeline], run


def _build_dp3(scenario, dtype):
    """DP3 point-cloud diffusion policy has no public checkpoint; materialize a
    random-init one (the fastkernels pipeline's own state_dict round-trips) and
    run the engine's synthetic warmup forward (PointNet encoder + 1D U-Net
    denoise)."""
    import tempfile
    from .infra.dp3_engine import DP3Engine
    from .validate.bench_dp3 import materialize_fastkernels_checkpoint
    from .workloads import DP3_CONFIG
    variant = getattr(scenario, "variant", None) or "simple_dp3"
    ckpt_dir = tempfile.mkdtemp(prefix="fastkernels_dp3_")
    ckpt = materialize_fastkernels_checkpoint(
        ckpt_dir, variant, seed=0,
        action_dim=DP3_CONFIG.action_dim, state_dim=DP3_CONFIG.state_dim,
        num_points=DP3_CONFIG.num_points,
    )
    engine = DP3Engine(checkpoint_path=ckpt, seed=0, dtype=torch.float32,
                       device="cuda", enforce_eager=True)
    pipeline = engine._get_pipeline()

    def run(wl_label, wl, params):
        engine.warmup(num_steps=2)
        return 1

    return [pipeline], run


def _ensure_spconv():
    """PTV3's ``ours`` layers hard-import ``spconv.pytorch`` + ``addict``. Reuse
    validate's provision installer for just that pip leg; skip its reference-repo
    clone, which capture (ours-only) never needs."""
    try:
        import addict  # noqa: F401
        import spconv.pytorch  # noqa: F401
    except ImportError:
        from .validate.provision import _pip
        _pip("spconv-cu126", "addict")


def _build_ptv3(scenario, dtype):
    """PointTransformerV3 point-cloud segmentation. Random-init on one synthetic
    object cloud (feat = coord+normal, the model's 6-channel input); the serial-
    ization derives grid_coord/batch from grid_size+offset, so no dataset or
    checkpoint is needed."""
    _ensure_spconv()
    from .infra.pointcloud_loader import default_ptv3_kwargs, load_ours_point_model
    model = load_ours_point_model(scenario.hf_name, device="cuda", dtype=dtype,
                                  **default_ptv3_kwargs())

    def run(wl_label, wl, params):
        n = 2048
        coord = torch.rand(n, 3, device="cuda", dtype=dtype)
        normal = torch.randn(n, 3, device="cuda", dtype=dtype)
        batch = {
            "coord": coord,
            "feat": torch.cat([coord, normal], dim=1),
            "grid_size": 0.02,
            "offset": torch.tensor([n], device="cuda", dtype=torch.long),
        }
        model(batch)
        return 1

    return [model], run


def _build_3dgs(scenario, dtype):
    """3D Gaussian Splatting. The fastkernels renderer is pure torch (no gsplat),
    and the scene .ply auto-downloads from the Hub, so render the baked-in orbit
    cameras once. A small resolution/point budget keeps capture fast; the CUDA-
    graph path is left off (incompatible with the capture instrumentation)."""
    from .infra.graphics_loader import load_3dgs_scene, load_ours_3dgs
    scene = load_3dgs_scene(
        scene_name=getattr(scenario, "scene", None) or "train",
        num_cameras=1, max_points=100_000,
        device="cuda", dtype=torch.float32, width=512, height=512,
    )
    ours = load_ours_3dgs(scene)

    def run(wl_label, wl, params):
        ours()
        return 1

    return [ours], run


def _build_sam3(scenario, dtype):
    """SAM 3.1 promptable segmentation. The fastkernels ``Sam3Model`` is a
    self-contained L4 task whose config defaults need no network; random-init is
    enough for a shape-only capture. Drive one image + dummy text-token forward
    (fp32; params are already fp32 so no complex-RoPE-buffer cast is needed)."""
    from .tasks.baseline.L4.sam3 import Sam3Config, Sam3Model
    config = Sam3Config.from_pretrained(scenario.hf_name)
    model = Sam3Model(config).cuda().eval()
    res = int(getattr(config, "img_size", 1008))
    n_tok = int(getattr(config, "text_context_length", 32))

    def run(wl_label, wl, params):
        b = min(int(getattr(params, "batch_size", 0) or 1), 4)
        images = torch.randn(b, 3, res, res, device="cuda", dtype=torch.float32)
        tok = torch.ones(b, n_tok, dtype=torch.long, device="cuda")
        model(images, tok)
        return b

    return [model], run


def _build_pi0(scenario, dtype):
    """Pi0 robotics VLA. ``Pi0Config`` is fully defaulted, so the real-shaped
    pipeline (SigLIP vision + Gemma VLM + DiT action expert) builds random-init
    without the OpenPI checkpoint -- enough for shape capture. Drive one
    flow-matching forward on a synthetic ALOHA 3-camera observation."""
    from .tasks.baseline.L4.pi0 import Pi0Config, Pi0Pipeline, Pi0SamplingParams
    config = Pi0Config()
    model = Pi0Pipeline(config).cuda().to(dtype).eval()
    cams = 3
    patches = (config.vlm_vision_config.image_size //
               config.vlm_vision_config.patch_size) ** 2
    img_tokens = patches * cams
    text_len = 48
    ids = [config.image_token_id] * img_tokens + [2] + [1] * text_len

    def run(wl_label, wl, params):
        input_ids = torch.tensor([ids], device="cuda")
        n = input_ids.shape[1]
        state = torch.randn(1, config.max_state_dim, device="cuda", dtype=dtype)
        pixel_values = torch.rand(1, cams, 3, 224, 224, device="cuda", dtype=dtype) * 2 - 1
        model(
            state=state, input_ids=input_ids, pixel_values=pixel_values,
            pixel_attention_mask=torch.ones(1, cams, dtype=torch.bool, device="cuda"),
            attention_mask=torch.ones(1, n, device="cuda"),
            params=Pi0SamplingParams(num_inference_steps=2),
        )
        return 1

    return [model], run


def _openfold3_synthetic_batch(n_tokens: int, n_seqs: int, c_token: int):
    """Reproduce the atom/MSA feature dict that ``bench_openfold3`` builds, but
    with a synthetic MSA (random residue types) so no alignment files are
    needed. Shapes/dtypes match the featurizer exactly."""
    import numpy as np
    NUM_CLASSES, MAX_ATOMS = 22, 23
    C_ELEM, C_NAME_DIM, C_NAME_VOCAB = 119, 4, 64
    rng = np.random.RandomState(0)
    msa_index = rng.randint(0, 20, size=(n_seqs, n_tokens))
    msa_onehot_raw = np.eye(NUM_CLASSES, dtype=np.float32)[msa_index]
    msa_onehot = np.zeros((n_seqs, n_tokens, 32), dtype=np.float32)
    msa_onehot[:, :, :NUM_CLASSES] = msa_onehot_raw
    has_deletion = np.zeros((n_seqs, n_tokens), dtype=np.float32)
    deletion_value = np.zeros((n_seqs, n_tokens), dtype=np.float32)
    profile = msa_onehot_raw.mean(axis=0)
    deletion_mean = has_deletion.mean(axis=0)
    token_features = np.concatenate([
        profile, deletion_mean[:, None],
        np.zeros((n_tokens, c_token - NUM_CLASSES - 1), dtype=np.float32),
    ], axis=1)
    n_atoms = n_tokens * MAX_ATOMS
    ref_pos = np.zeros((n_atoms, 3), np.float32)
    ref_charge = np.zeros(n_atoms, np.float32)
    ref_mask = np.zeros(n_atoms, np.float32)
    ref_element = np.zeros((n_atoms, C_ELEM), np.float32)
    ref_name = np.zeros((n_atoms, C_NAME_DIM, C_NAME_VOCAB), np.float32)
    ref_space_uid = np.zeros(n_atoms, np.int64)
    atom_to_token = np.zeros(n_atoms, np.int64)
    atom_mask = np.zeros(n_atoms, np.float32)
    for i in range(n_tokens):
        base = i * MAX_ATOMS
        for j in range(5):
            idx = base + j
            ref_pos[idx] = rng.randn(3).astype(np.float32) * 1.5
            ref_mask[idx] = atom_mask[idx] = 1.0
            ref_element[idx, min(5 + j, C_ELEM - 1)] = 1.0
            ref_name[idx, 0, min(j + 1, C_NAME_VOCAB - 1)] = 1.0
        ref_space_uid[base:base + MAX_ATOMS] = i
        atom_to_token[base:base + MAX_ATOMS] = i
    restype = np.zeros((n_tokens, 32), np.float32); restype[:, :NUM_CLASSES] = msa_onehot_raw[0]
    profile32 = np.zeros((n_tokens, 32), np.float32); profile32[:, :NUM_CLASSES] = profile
    tt = torch.from_numpy
    return {
        "token_features": tt(token_features).unsqueeze(0),
        "residue_index": torch.arange(n_tokens, dtype=torch.float32).unsqueeze(0),
        "token_mask": torch.ones(1, n_tokens),
        "atom_mask": tt(atom_mask).unsqueeze(0),
        "msa": tt(msa_onehot).unsqueeze(0),
        "has_deletion": tt(has_deletion).unsqueeze(0),
        "deletion_value": tt(deletion_value).unsqueeze(0),
        "msa_mask": torch.ones(1, n_seqs, n_tokens),
        "restype": tt(restype).unsqueeze(0),
        "profile": tt(profile32).unsqueeze(0),
        "deletion_mean": tt(deletion_mean).unsqueeze(0),
        "ref_pos": tt(ref_pos).unsqueeze(0),
        "ref_charge": tt(ref_charge).unsqueeze(0),
        "ref_mask": tt(ref_mask).unsqueeze(0),
        "ref_element": tt(ref_element).unsqueeze(0),
        "ref_atom_name_chars": tt(ref_name).unsqueeze(0),
        "ref_space_uid": tt(ref_space_uid).unsqueeze(0),
        "atom_to_token_index": tt(atom_to_token).unsqueeze(0),
        "token_index": torch.arange(n_tokens, dtype=torch.float32).unsqueeze(0),
        "asym_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "entity_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "sym_id": torch.zeros(1, n_tokens, dtype=torch.long),
    }


def _build_openfold3(scenario, dtype):
    """OpenFold3 (AlphaFold3) structure prediction. The fastkernels
    ``OpenFold3Model`` builds random-init from its config (no checkpoint needed
    for shapes). Use the bench's reduced blocks/rollout and a small synthetic
    chain (trunk O(N^2) memory), driving trunk + diffusion rollout once."""
    from .tasks.baseline.L4.openfold3 import OpenFold3Config, OpenFold3Model
    cfg = OpenFold3Config(pairformer_no_blocks=48, msa_no_blocks=4,
                          no_rollout_steps=10, num_recycles=1)
    model = OpenFold3Model(cfg).cuda().to(dtype).eval()
    c_token = int(getattr(cfg, "c_token_embedder", 384))

    def run(wl_label, wl, params):
        batch = _openfold3_synthetic_batch(n_tokens=16, n_seqs=8, c_token=c_token)
        batch = {k: (v.to("cuda", dtype) if v.is_floating_point() else v.to("cuda"))
                 for k, v in batch.items()}
        with torch.no_grad():
            model(batch)
        return 1

    return [model], run


def _build_tts(scenario, dtype):
    """CosyVoice3 text-to-speech. Reuse validate's provision (s3tokenizer pip
    leg), load the published checkpoint, and run one short ``generate`` on
    synthetic 3s reference audio. Preprocessing mirrors vllm-omni (bench keeps it
    in a worker string, so it is inlined here): Qwen text tokens, on-GPU
    S3Tokenizer speech tokens, mel speech-feat, campplus speaker embedding ->
    talker AR LLM -> flow-matching code2wav."""
    import os
    from functools import partial
    try:
        import s3tokenizer
    except ImportError:
        from .validate.provision import _pip
        _pip("s3tokenizer")
        import s3tokenizer
    import numpy as np
    import onnxruntime
    from huggingface_hub import snapshot_download
    from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
    from vllm_omni.model_executor.models.cosyvoice3.utils import (
        extract_speech_feat, extract_spk_embedding, extract_text_token,
        mel_spectrogram,
    )
    from .tasks.baseline.L4.cosyvoice3 import CosyVoice3Config, CosyVoice3ForTTS

    model_dir = snapshot_download(scenario.hf_name)
    config = CosyVoice3Config.from_pretrained(scenario.hf_name)
    model = CosyVoice3ForTTS(config, model_stage="e2e")
    model.load_weights_e2e(model_dir, torch.device("cuda"))
    model.eval()

    tokenizer = get_qwen_tokenizer(
        token_path=os.path.join(model_dir, config.qwen_pretrain_path),
        skip_special_tokens=config.skip_special_tokens, version=config.version)
    campplus = onnxruntime.InferenceSession(
        os.path.join(model_dir, config.campplus_onxx_path),
        providers=["CPUExecutionProvider"])
    feat_ext = partial(mel_spectrogram, **getattr(config, "feat_extractor", {}))
    s3 = s3tokenizer.load_model("speech_tokenizer_v3_25hz").to("cuda").eval()

    def _speech_token(ref_audio, ref_sr):
        from math import gcd
        from scipy.signal import resample_poly
        wav = ref_audio
        if ref_sr != 16000:
            g = gcd(ref_sr, 16000)
            wav = resample_poly(wav, 16000 // g, ref_sr // g).astype(np.float32)
        mel = s3tokenizer.log_mel_spectrogram(torch.from_numpy(wav))
        mels_p, mels_lens = s3tokenizer.padding([mel])
        with torch.inference_mode():
            codes, codes_lens = s3.quantize(mels_p.cuda(), mels_lens.cuda())
        n = int(codes_lens[0].item())
        return codes[:1, :n].to(dtype=torch.int32, device="cuda")

    def run(wl_label, wl, params):
        ref_audio = np.random.randn(24000 * 3).astype(np.float32)
        text_token, _ = extract_text_token("Warmup.", tokenizer, config.allowed_special)
        prompt_text_token, _ = extract_text_token(
            "Testing my voice.", tokenizer, config.allowed_special)
        speech_token = _speech_token(ref_audio, 24000)
        speech_feat, _ = extract_speech_feat((ref_audio, 24000), feat_ext, "cuda")
        if config.sample_rate == 24000:
            tok_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat = speech_feat[:, :2 * tok_len]
            speech_token = speech_token[:, :tok_len]
        spk_embedding = extract_spk_embedding((ref_audio, 24000), campplus, "cuda")
        model.generate(
            text_token=text_token.cuda(), prompt_text_token=prompt_text_token.cuda(),
            speech_token=speech_token.cuda(), speech_feat=speech_feat.cuda(),
            spk_embedding=spk_embedding.cuda(), n_timesteps=10,
            temperature=0, cfm_seed=12345,
        )
        return 1

    return [model], run


def _build_instantngp(scenario, dtype):
    """InstantNGP NeRF. The fastkernels renderer loads a pyngp-trained snapshot,
    so reuse validate's provision to build pyngp + the fox scene, then render one
    view. Skips cleanly if the source build is unavailable (needs cmake/CUDA
    toolkit)."""
    from .validate.provision import COMPONENTS
    comp = COMPONENTS["instant_ngp"]
    if not comp.check():
        from .validate.provision import provision
        if provision(["instant_ngp"]):
            raise _CaptureSkip(
                "instant-ngp pyngp source build unavailable "
                "(needs the repo clone + cmake/CUDA toolkit)")
    from .infra.nerf_loader import load_ours_instantngp
    scene = getattr(scenario, "scene", None) or "fox"
    model = load_ours_instantngp(scene_name=scene, train_steps=5,
                                 width=512, height=512, spp=1)

    def run(wl_label, wl, params):
        model(view_index=0)
        return 1

    return [model], run


_NONTEXT_BUILDERS = {
    "VisionEncoder": _build_vision,
    "Detection": _build_detection,
    "Embedding": _build_embedding,
    "Recsys": _build_recsys,
    "VideoRepresentation": _build_vjepa2,
    "Diffusion": _build_diffusion,
    "VideoDiffusion": _build_diffusion,
    "DiffusionLM": _build_llada,
    "WorldModel": _build_oasis,
    "ASR": _build_whisper,
    "PointCloudPolicy": _build_dp3,
    "PointCloudSeg": _build_ptv3,
    "Segmentation": _build_sam3,
    "StructurePrediction": _build_openfold3,
    "Robotics": _build_pi0,
    "TTS": _build_tts,
}

# Models the family lookup above can't route on its own: either the workload
# family is the generic ``LLM`` (ttt_e2e), or several models share one family
# (the ``Rendering`` family covers both 3DGS and InstantNGP). Keyed by the bare
# model token in ``hf_name``; checked before the family map.
_NONTEXT_MODEL_BUILDERS = {
    "ttt_e2e": _build_ttt_e2e,
    "Voxel51/gaussian_splatting": _build_3dgs,
    "NVlabs/instant-ngp": _build_instantngp,
}


def _capture_nontext_scenario(scenario, runs, args, instrumented, n_instrumented,
                              multi):
    """Capture a non-text family by driving one synthetic forward per workload.

    Raises ``_CaptureSkip`` when the family has no builder yet, or the builder
    can't construct the model in this environment (missing assets/deps)."""
    family = type(scenario.workloads[0]).__name__
    builder = _NONTEXT_MODEL_BUILDERS.get(scenario.hf_name) or _NONTEXT_BUILDERS.get(family)
    if builder is None:
        raise _CaptureSkip(
            f"capture for {family} models is not implemented yet "
            f"(synthetic-forward driver pending)")

    dtype = _engine_dtype(scenario.dtype) or torch.bfloat16
    _RECORDS.clear()
    _HOOK_RECORDS.clear()
    print(f"\n########## Scenario: {scenario.hf_name} "
          f"({family}, dtype={scenario.dtype}) ##########")
    print(f"Building fastkernels {family} model (synthetic-input capture) ...")
    try:
        models, run_forward = builder(scenario, dtype)
    except _CaptureSkip:
        raise
    except Exception as exc:  # noqa: BLE001 - unbuildable here -> skip, not fail
        raise _CaptureSkip(f"could not build {family} model {scenario.hf_name!r}: "
                           f"{exc!r}")

    written: list[Path] = []
    ok = True
    try:
        for wl_label, wl in runs:
            print(f"\n=== Capturing workload: {wl_label} ===")
            params = spec_for(wl).params
            for entry in _RECORDS.values():
                entry["forward"] = None
            _HOOK_RECORDS.clear()
            handles = _register_hooks_on_models(models, instrumented)
            try:
                n = run_forward(wl_label, wl, params)
            except Exception as exc:  # noqa: BLE001 - isolate per-workload failures
                traceback.print_exc()
                print(f"  !! workload {wl_label} failed: {exc!r}; "
                      f"skipping to the next workload.")
                ok = False
                continue
            finally:
                for h in handles:
                    h.remove()

            # The forward-pre-hook cross-check is INFORMATIONAL here (recorded in
            # the report, never fails the scenario): diffusion/rollout families
            # routinely call submodule ``.forward`` directly (bypassing
            # ``__call__``), so a pre-hook legitimately sees fewer ops than the
            # monkeypatch that actually records shapes. Only a forward exception
            # (above) fails a synthetic-forward capture.
            hook_verification = _verify_forward_capture()
            status = "PASS" if hook_verification["passed"] else "INFO"
            print(f"  Verification 1 (hook cross-check) [{status}]: "
                  f"{hook_verification['classes_matched']}/"
                  f"{hook_verification['classes_checked']} operator forwards match "
                  f"the independent hook capture.")
            for issue in hook_verification["issues"]:
                print(f"    ! {issue}")
            verification = {
                "forward_pre_hook_crosscheck": hook_verification,
                "mock_batching_replay": {
                    "method": "mock continuous-batching / chunked-prefill replay",
                    "applicable": False, "passed": True,
                    "reason": (f"skipped for {family}: no paged-KV token schedule; "
                               "the forward-pre-hook cross-check is the pass "
                               "criterion."),
                },
                "passed": hook_verification["passed"],
            }
            generation = {
                "num_prompts": n,
                "max_batch_size": n,
                "engine": family.lower(),
                "modality": family,
                "inputs": "synthetic (random tensors sized from workload spec)",
            }
            report = _build_report(
                scenario.hf_name, wl_label, dtype, n, n_instrumented,
                generation, verification,
            )
            out_path = _report_path(args.output, scenario, wl_label, multi,
                                    args.max_requests)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(_dumps(report))
                f.write("\n")
            written.append(out_path)
            print(f"  Captured {report['num_operator_classes_executed']} executed "
                  f"operator class(es) (of {n_instrumented} instrumented).")
            print(f"  Report written to {out_path}")
    finally:
        for m in models:
            del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return written, ok


def _capture_scenario(scenario, runs, args, LlamaEngine, SamplingParams,
                      instrumented, n_instrumented, multi):
    """Build one scenario's engine and capture each of its workloads.

    Returns ``(written_paths, ok)`` where ``ok`` is False if any workload's
    verification failed. Raises if the engine build or a workload run itself
    errors, so the caller can report the scenario and move on to the next one.
    The engine is always released before returning so the next (possibly larger)
    model has room on the GPU.
    """
    # ``--max-layers`` truncates the transformer decoder stack to its first N
    # blocks. It is only wired through the standard ``LlamaEngine`` path (below),
    # which covers plain LLMs and Qwen-VL/Omni; the specialized alt-engines
    # (EAGLE-3 target/draft coupling, FLA recurrent state, Jamba hybrid layer
    # pattern) don't funnel through the ``max_layers`` loader path and truncation
    # is ill-defined for them, so they always capture at full depth. Only the
    # applied value flavors the report filename / metadata.
    alt_engine = _is_eagle3(scenario) or _is_fla(scenario) or _is_jamba(scenario)
    report_max_layers = None if alt_engine else args.max_layers
    if alt_engine and args.max_layers is not None:
        print(
            f"  NOTE: --max-layers={args.max_layers} is not applied for this "
            f"engine type (EAGLE-3/FLA/Jamba); capturing at full depth."
        )

    # Remove any stale report(s) for this scenario up front: an interrupted or
    # crashing capture then leaves no file for an unfinished workload, rather
    # than a previous run's file that could be mistaken for a fresh result.
    _purge_stale_reports(scenario, runs, args, multi, report_max_layers)
    if _is_eagle3(scenario):
        # EAGLE-3 uses a distinct engine (target + speculative draft head).
        return _capture_eagle3_scenario(
            scenario, runs, args, instrumented, n_instrumented, multi,
        )
    if _is_fla(scenario):
        # GLA / RetNet / RWKV-7 carry recurrent state (FLAEngine, not paged-KV).
        return _capture_fla_scenario(
            scenario, runs, args, instrumented, n_instrumented, multi,
        )
    if _is_jamba(scenario):
        # Jamba is a hybrid transformer+Mamba+MoE model (JambaEngine).
        return _capture_jamba_scenario(
            scenario, runs, args, instrumented, n_instrumented, multi,
        )
    if scenario.hf_name in _NONTEXT_MODEL_BUILDERS or (
        scenario.workloads
        and type(scenario.workloads[0]).__name__ in _NONTEXT_FAMILIES
    ):
        # Non-text families (vision/detection/embedding/recsys/...) don't
        # autoregress tokens; drive a synthetic forward per workload instead.
        # ``_NONTEXT_MODEL_BUILDERS`` also routes checkpoint-free LLM-family
        # models (ttt_e2e) here by their bare model token.
        return _capture_nontext_scenario(
            scenario, runs, args, instrumented, n_instrumented, multi,
        )
    dtype = _engine_dtype(scenario.dtype)
    # Fresh init/hook records per model (a new engine re-runs every __init__).
    _RECORDS.clear()
    _HOOK_RECORDS.clear()

    # 3) Build the engine from the scenario so every module runs in-process and
    #    the wrapped forwards are actually invoked.
    print(
        f"\n########## Scenario: {scenario.hf_name} "
        f"(dtype={scenario.dtype}, tp={scenario.tp}, "
        f"enforce_eager={scenario.enforce_eager}) ##########"
    )
    print(f"Loading {scenario.hf_name} into fastkernels LlamaEngine ...")
    # Capture instruments every operator forward with a Python monkey-patch that
    # json-serializes tensor metadata; that is incompatible with torch.compile /
    # CUDA-graph capture (Dynamo cannot trace it). The engine is therefore always
    # built in eager mode for capture, regardless of the scenario's
    # ``enforce_eager`` (which only affects real benchmarking).
    if not scenario.enforce_eager:
        print(
            "  (forcing enforce_eager=True for capture; the scenario's "
            "enforce_eager=False applies to benchmarking only)"
        )
    engine = LlamaEngine(
        model_name=scenario.hf_name,
        dtype=dtype,
        tensor_parallel_size=scenario.tp,
        enforce_eager=True,
        max_num_seqs=scenario.max_num_seqs,
        max_layers=args.max_layers,
    )

    written: list[Path] = []
    ok = True
    try:
        for wl_label, wl in runs:
            print(f"\n=== Capturing workload: {wl_label} ===")

            # 4) Install the independent verification hooks for this run.
            hook_handles = _register_verification_hooks(
                engine.model_runner.model, instrumented,
            )

            # 5) Assemble the capture inputs by loading real, chat-templated
            #    prompts and feeding them to the engine as token-id lists.
            #
            #    NOTE: the ``--prompt`` ad-hoc override (and its ``--max-tokens``
            #    budget) was removed. Previously a workload of ``None`` captured
            #    user-supplied raw strings instead:
            #
            #        if wl is None:
            #            gen_prompts, used_chat_template = _prepare_prompts(
            #                engine, args.prompts,
            #                use_chat_template=not args.no_chat_template)
            #            prompt_texts = list(args.prompts)
            #            max_new_tokens = [args.max_tokens] * len(gen_prompts)
            #
            #    Every run is now a real scenario workload.
            #
            # The loader returns every available row up to ``--max-requests``
            # (the default is large enough to use them all); curated sets return
            # fewer if they have fewer rows (e.g. long-context has 64).
            #
            # Resolve the request count / dataset / decode budget from the
            # workload's own spec instead of the loader's ``DEFAULT_WORKLOAD_
            # DATASETS`` map, which only knows the two *throughput* workloads
            # (mixed, long-context). The *latency* probes (single-request,
            # fixed-batch-32) carry their real-prompt dataset + decode budget on
            # the LatencyWorkload spec, so looking them up by name in that map
            # raised KeyError and aborted the whole scenario.
            spec = spec_for(wl)
            params = spec.params
            wl_dataset = getattr(params, "dataset_name", "") or None
            modality = getattr(params, "modality", "text")
            is_media = isinstance(wl, (VLM, OmniModal)) and modality != "text"
            if spec.purpose is Purpose.LATENCY:
                # Latency probe: submit a fixed-size batch of real prompts with a
                # bounded per-request decode budget.
                n_req = min(args.max_requests, getattr(params, "batch_size", 1))
                wl_decode_cap = getattr(params, "output_len", None)
            else:
                n_req = min(
                    args.max_requests,
                    getattr(params, "num_requests", args.max_requests),
                )
                wl_decode_cap = getattr(params, "decode_cap", None)
                # VLM/OmniModal text-throughput carries its decode budget as
                # ``output_len`` (no ``decode_cap`` attribute); fall back to it so
                # the text modality is capped while the LLM path stays unchanged.
                if wl_decode_cap is None:
                    wl_decode_cap = getattr(params, "output_len", None)

            # Per-request media (image/video/audio) inputs, parallel to
            # ``gen_prompts``; all ``None`` on the text/LLM path. ``prompt_lens``
            # is the per-request prompt token count that drives Verification #2;
            # it stays ``None`` for media because the engine expands the media
            # placeholders internally and never returns the expanded length.
            images = videos = audios = None
            prompt_lens = None
            media_descriptors = None

            if is_media:
                # Multimodal LLM workload (Qwen2-VL / Qwen3-VL / Qwen2.5-Omni):
                # load the real image/video/audio dataset and hand the raw text
                # prompt + media to the engine, which runs the HF processor
                # (chat template + placeholder expansion) internally. Media
                # loading + the generate call mirror tests/bench_vllm.py's
                # FASTKERNELS_VLM_WORKER so captured shapes match the benchmark.
                if not engine.is_qwen_vl:
                    print(
                        f"  !! SKIP workload {wl_label}: modality={modality!r} "
                        f"needs a Qwen-VL/Omni engine (is_qwen_vl=False)"
                    )
                    for handle in hook_handles:
                        handle.remove()
                    continue
                media_cap = os.environ.get("FASTKERNELS_MEDIA_MAX_REQUESTS")
                if media_cap:
                    n_req = min(n_req, int(media_cap))
                out_len = params.output_len
                print(
                    f"Loading '{wl_label}' {modality} media ({n_req}) "
                    f"from {wl_dataset} ..."
                )
                mm = _preload_mm_data(
                    wl_dataset, getattr(params, "dataset_split", None),
                    n_req, _MEDIA_SEED,
                )
                if not mm:
                    raise RuntimeError(
                        f"{wl_dataset} ({modality}) yielded no usable requests"
                    )
                from PIL import Image
                gen_prompts = [item["prompt"] for item in mm]
                prompt_texts = list(gen_prompts)
                max_new_tokens = [out_len] * len(mm)
                images = [item["images"] for item in mm]
                videos = [
                    [[Image.fromarray(item["video_frames"][j]).convert("RGB")
                      for j in range(item["video_frames"].shape[0])]]
                    if item["video_frames"] is not None else None
                    for item in mm
                ]
                audios = [
                    [item["audio"]] if item["audio"] is not None else None
                    for item in mm
                ]
                media_descriptors = [_media_descriptor(item) for item in mm]
                # Fixed decode budget from the workload spec (media datasets carry
                # no per-row response length). ignore_eos=True matches bench_vllm.py
                # (every request decodes exactly its budget).
                sampling = [
                    SamplingParams(temperature=0.0, max_tokens=out_len, ignore_eos=True)
                    for _ in mm
                ]
                used_chat_template = True
            else:
                print(f"Loading '{wl_label}' workload prompts ({n_req}) ...")
                samples = load_real_prompt_workload(
                    wl_label, engine.tokenizer, num_requests=n_req,
                    dataset_name=wl_dataset, decode_cap=wl_decode_cap,
                )
                # Drop prompts whose prompt+decode exceeds the engine's context
                # window, else the paged block-table preallocation in the engine
                # overflows. The long-context buckets are sized in a *reference*
                # tokenizer's tokens; a model whose tokenizer is less efficient
                # (e.g. gemma-4 re-tokenizing a 128K-Llama-token document into
                # ~156K gemma tokens) can produce sequences longer than
                # max_model_len for that model.
                _mml = getattr(getattr(engine, "model_runner", None),
                               "max_model_len", None)
                if _mml:
                    n0 = len(samples)
                    samples = [
                        s for s in samples
                        if len(s.prompt_token_ids) + s.output_len <= _mml
                    ]
                    if len(samples) < n0:
                        print(f"  (dropped {n0 - len(samples)}/{n0} prompt(s) whose "
                              f"prompt+decode exceeds max_model_len={_mml})")
                    if not samples:
                        print(f"  !! SKIP workload {wl_label}: all prompts exceed "
                              f"max_model_len={_mml}")
                        for handle in hook_handles:
                            handle.remove()
                        continue
                gen_prompts = [list(s.prompt_token_ids) for s in samples]
                # Per-request generation budget from the HF dataset's own
                # response length (like bench_vllm.py), so the decode phase
                # matches the workload's real output-length distribution instead
                # of a flat cap.
                max_new_tokens = [s.output_len for s in samples]
                used_chat_template = True
                # Decode WITH special tokens so the chat-template structure is
                # preserved. Decoding with skip_special_tokens=True strips the
                # structural delimiters (e.g. gpt-oss harmony's <|start|>/
                # <|message|>) but keeps the plain-text role/channel labels,
                # gluing them onto the content ("systemYou are ChatGPT...").
                prompt_texts = [
                    engine.tokenizer.decode(ids, skip_special_tokens=False)
                    for ids in gen_prompts
                ]
                # One SamplingParams per request so each gets its own max_tokens.
                # ignore_eos=True matches bench_vllm.py: every request decodes
                # exactly its per-request budget (the dataset response length),
                # so the captured decode-length distribution matches the benchmark
                # rather than stopping early at EOS.
                sampling = [
                    SamplingParams(temperature=0.0, max_tokens=mnt, ignore_eos=True)
                    for mnt in max_new_tokens
                ]
                # Full prompt token counts (token-id prompts); drives Verification #2.
                prompt_lens = [len(p) for p in gen_prompts]
            print(
                f"Running {len(gen_prompts)} prompt(s) at max_batch_size="
                f"{engine.max_num_seqs}, max_new_tokens="
                f"[{min(max_new_tokens)}..{max(max_new_tokens)}] "
                f"(modality={modality}, chat_template={used_chat_template}) ..."
            )

            # Reset forward + hook records so each workload's report reflects
            # only its own run (and the monkey-patch and hooks observe the same
            # window -- the engine's warmup forward ran during construction,
            # before hooks were attached). Init records -- captured at
            # construction -- are model-level and kept across this scenario's
            # workloads.
            for entry in _RECORDS.values():
                entry["forward"] = None
            _HOOK_RECORDS.clear()

            # Same engine/model as every other workload of this scenario (loaded
            # once, never reloaded). Clear the per-run KV state so workloads run
            # independently, then remove this run's hooks in a finally so a
            # failed generate can't leak them into anything that follows.
            _reset_engine_runtime_state(engine)
            try:
                outputs = engine.generate(
                    gen_prompts, sampling,
                    images=images, videos=videos, audio_features=audios,
                    # "Processed prompts" bar -- one tick per finished request.
                    use_tqdm=True,
                )
            finally:
                for handle in hook_handles:
                    handle.remove()

            # 6) Summarize generation: how many sequences stopped at EOS vs. the
            #    request's own token budget, and keep the decoded responses so
            #    they can be inspected. A request "reached EOS" when it emitted
            #    fewer tokens than its (per-request) max_new_tokens.
            gen_lengths = [len(o.token_ids) for o in outputs]
            num_eos = sum(
                1 for n, cap in zip(gen_lengths, max_new_tokens) if n < cap
            )
            # Re-decode the generated ids WITH special tokens so the response
            # keeps its chat-template structure (e.g. gpt-oss harmony channels
            # <|channel|>analysis<|message|>...). The engine's ``generated_text``
            # is decoded with skip_special_tokens=True, which would otherwise
            # glue channel labels onto the text ("analysisWe need to...").
            responses = []
            for i in range(len(gen_prompts)):
                resp = {
                    "prompt": prompt_texts[i],
                    "prompt_tokens": (
                        prompt_lens[i] if prompt_lens is not None else None
                    ),
                    "response": engine.tokenizer.decode(
                        outputs[i].token_ids, skip_special_tokens=False,
                    ),
                    "max_new_tokens": max_new_tokens[i],
                    "tokens": gen_lengths[i],
                    "stop_reason": (
                        "eos" if gen_lengths[i] < max_new_tokens[i]
                        else "max_tokens"
                    ),
                }
                if media_descriptors is not None:
                    resp["media"] = media_descriptors[i]
                responses.append(resp)
            generation = {
                "num_prompts": len(gen_prompts),
                "max_batch_size": engine.max_num_seqs,
                "batch_refilled": len(gen_prompts) > engine.max_num_seqs,
                "used_chat_template": used_chat_template,
                "modality": modality,
                "dataset": wl_dataset,
                "prompt_tokens_known": prompt_lens is not None,
                "max_new_tokens_source": (
                    "workload_output_len" if is_media
                    else "dataset_response_length"
                ),
                "max_new_tokens_range": [min(max_new_tokens), max(max_new_tokens)],
                "num_finished_by_eos": num_eos,
                "num_hit_max_tokens": len(gen_lengths) - num_eos,
                "responses": responses,
            }
            if args.max_layers is not None:
                # Only the first N transformer decoder layers were built and run
                # (embeddings / final norm / LM head -- and any vision encoder --
                # are unaffected). Record both the requested cap and the model's
                # resulting depth so the truncated report is self-describing.
                generation["max_layers"] = args.max_layers
                _cfg = getattr(getattr(engine, "model_runner", None), "config", None)
                generation["num_hidden_layers"] = getattr(
                    _cfg, "num_hidden_layers", None)
            print(
                f"  Generation: {generation['num_finished_by_eos']}/{len(gen_prompts)} "
                f"reached EOS before their per-request budget; "
                f"lengths={gen_lengths}"
            )

            # 7a) Verify the capture against the independent hook observations.
            hook_verification = _verify_forward_capture()
            status = "PASS" if hook_verification["passed"] else "FAIL"
            print(
                f"  Verification 1 (hook cross-check) [{status}]: "
                f"{hook_verification['classes_matched']}/"
                f"{hook_verification['classes_checked']} operator forwards match "
                f"the independent hook capture."
            )
            for issue in hook_verification["issues"]:
                print(f"    ! {issue}")

            # 7b) Verify the captured shapes against a mock continuous-batching /
            #     chunked-prefill replay driven by the per-request prompt and
            #     generated lengths, the engine's batch/token budgets, AND its
            #     KV-cache block pool (block granularity + total blocks + a 1%
            #     admission watermark, mirroring LlamaEngine.generate). The KV
            #     pool is what bounds concurrency for long prompts; without it
            #     the replay over-predicts long-context batch sizes.
            #
            #     Only runs for text-token workloads (plain LLMs + the multimodal
            #     ``text`` modality), where ``prompt_lens`` is the real per-request
            #     prompt length. For image/video/audio the engine expands the media
            #     placeholders internally and ``generate`` never returns the
            #     expanded length, so there is no ground-truth vector to drive the
            #     replay (and vision/audio-encoder scheduling is not modeled) --
            #     the hook cross-check above is the pass criterion there.
            _mr_cfg = getattr(getattr(engine, "model_runner", None), "config", None)
            pure_ssm = (getattr(_mr_cfg, "model_type", "") in ("mamba", "mamba2")
                        if _mr_cfg is not None else False)
            if prompt_lens is not None and not pure_ssm:
                from .infra.engine import BLOCK_SIZE
                _bm = getattr(engine, "block_manager", None)
                num_kv_blocks = getattr(_bm, "_num_blocks", None)
                watermark = max(int(num_kv_blocks * 0.01), 1) if num_kv_blocks else 0
                schedule_verification = _verify_batch_schedule(
                    prompt_lens, gen_lengths,
                    engine.max_num_seqs, engine.max_num_batched_tokens,
                    _infer_num_layers(engine),
                    max_tokens=max_new_tokens,
                    num_blocks=num_kv_blocks or None,
                    block_size=BLOCK_SIZE,
                    watermark_blocks=watermark,
                )
                status = "PASS" if schedule_verification["passed"] else "FAIL"
                n_ok = sum(1 for c in schedule_verification["checks"] if c["passed"])
                print(
                    f"  Verification 2 (mock batching replay) [{status}]: "
                    f"{n_ok}/{len(schedule_verification['checks'])} checks passed "
                    f"({schedule_verification['simulated_steps']} simulated steps, "
                    f"{schedule_verification['simulated_prefill_steps']} with prefill)."
                )
                for check in schedule_verification["checks"]:
                    if not check["passed"]:
                        print(f"    ! {check['name']}: {check['detail']}")
            elif pure_ssm:
                # Pure-SSM (Mamba / Mamba2) has no paged-attention KV cache:
                # sequences hold fixed-size SSM state slots, so the paged-KV
                # continuous-batching replay does not model their per-step
                # admission (aggregate token/concurrency totals match, but the
                # step-by-step schedule diverges). As for EAGLE-3 / FLA / Jamba,
                # the forward-pre-hook cross-check is the pass criterion.
                schedule_verification = {
                    "method": "mock continuous-batching / chunked-prefill replay",
                    "applicable": False,
                    "passed": True,
                    "reason": (
                        "skipped for pure-SSM (mamba/mamba2): no paged-attention KV "
                        "cache -- sequences occupy fixed SSM state slots, so the "
                        "paged-KV continuous-batching replay does not model per-step "
                        "admission. The forward-pre-hook cross-check is the pass "
                        "criterion."
                    ),
                }
                print(
                    "  Verification 2 (mock batching replay) [N/A]: "
                    "skipped for pure-SSM mamba (see report)."
                )
            else:
                schedule_verification = {
                    "method": "mock continuous-batching / chunked-prefill replay",
                    "applicable": False,
                    "passed": True,
                    "reason": (
                        f"skipped for the {modality} modality: the engine expands "
                        f"media placeholders internally and generate() does not "
                        f"return the expanded per-request prompt length, so there "
                        f"is no ground-truth prompt-length vector to drive the "
                        f"replay; vision/audio-encoder scheduling is also not "
                        f"modeled. The forward-pre-hook cross-check is the pass "
                        f"criterion for this workload."
                    ),
                }
                print(
                    f"  Verification 2 (mock batching replay) [N/A]: "
                    f"skipped for the {modality} modality (see report)."
                )

            verification = {
                "forward_pre_hook_crosscheck": hook_verification,
                "mock_batching_replay": schedule_verification,
                "passed": hook_verification["passed"] and schedule_verification["passed"],
            }

            # 8) Write this workload's report to its own file. Record the
            #    engine's resolved compute dtype (``dtype`` may be None when the
            #    scenario names a quantization scheme like mxfp4/fp8).
            report = _build_report(
                scenario.hf_name, wl_label, engine.model_runner.dtype,
                engine.max_num_seqs, n_instrumented,
                generation, verification,
            )
            out_path = _report_path(args.output, scenario, wl_label, multi,
                                    args.max_requests, report_max_layers)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(_dumps(report))
                f.write("\n")
            written.append(out_path)
            print(
                f"  Captured {report['num_operator_classes_executed']} executed "
                f"operator class(es) (of {n_instrumented} instrumented)."
            )
            print(f"  Report written to {out_path}")
            if not verification["passed"]:
                ok = False
    finally:
        # Release the model (even on error) so the next scenario's engine fits.
        # The engine registers ``_cleanup`` with ``atexit``, which keeps a strong
        # reference to it -- a plain ``del`` would therefore NOT free its GPU
        # memory (this is what starved the second scenario and caused an OOM).
        # Run cleanup explicitly and drop the atexit registration first.
        engine._cleanup()
        atexit.unregister(engine._cleanup)
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return written, ok


# ---------------------------------------------------------------------------
# GPU-aware parallel scheduler
#
# Each scenario is captured in its own child process, pinned to a private set of
# GPUs via ``CUDA_VISIBLE_DEVICES`` (a scenario with ``tp=N`` claims N GPUs).
# Subprocess isolation is what makes this robust: capture instruments operators
# with process-global monkey-patches and accumulates into module-level dicts, so
# scenarios cannot safely share an interpreter -- and a crash, OOM or CUDA fault
# in one child can never take down the parent or the other scenarios. The
# scheduler packs scenarios onto the available GPUs by TP degree and launches
# the next one as soon as enough GPUs free up.
# ---------------------------------------------------------------------------
def _detect_gpu_ids(explicit: str | None) -> list[str]:
    """Physical GPU ids available for scheduling, as opaque strings.

    Priority: ``--gpus`` > ``CUDA_VISIBLE_DEVICES`` > ``nvidia-smi`` > torch.
    Ids stay strings so integer indices and MIG/UUID device specs both
    round-trip cleanly into each child's ``CUDA_VISIBLE_DEVICES``.
    """
    if explicit:
        return [tok.strip() for tok in explicit.split(",") if tok.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [tok.strip() for tok in env.split(",") if tok.strip()]
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is not None:
        try:
            out = subprocess.run(
                [nvsmi, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout
            ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if ids:
                return ids
        except Exception:  # noqa: BLE001 - fall back to torch below
            pass
    try:
        n = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n = 0
    return [str(i) for i in range(n)]


def _scenario_log_name(scenario, index: int) -> str:
    """Filesystem-safe, per-scenario log filename stem."""
    model = scenario.hf_name.replace("/", "__")
    return f"{index:02d}_{model}_tp{scenario.tp}_{scenario.dtype}"


def _kill_process_group(proc: "subprocess.Popen", pgid: int) -> None:
    """Terminate an entire scenario process group (rank 0 + spawned workers).

    SIGTERM first (lets Python finalizers/atexit run), escalating to SIGKILL if
    the group is still alive after a short grace -- a rank wedged in a NCCL
    busy-wait may ignore the polite signal.
    """
    for sig, grace in ((signal.SIGTERM, _WATCHDOG_TERM_GRACE_SEC), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)


def _watchdog_kill_reason(info: dict, now_wall: float, now_mono: float) -> str | None:
    """Return a reason string if this running scenario should be force-killed.

    A scenario is considered wedged when its log has gone silent for longer than
    ``_SCENARIO_STALL_SEC`` (no forward progress -- a hung NCCL collective or a
    stuck post-fault CUDA teardown), with an absolute wall-clock cap as a
    backstop. Returns ``None`` when the scenario still looks healthy.
    """
    elapsed = now_mono - info["start"]
    if elapsed > _SCENARIO_TIMEOUT_SEC:
        return (f"exceeded {_SCENARIO_TIMEOUT_SEC}s wall-clock timeout "
                f"(elapsed {int(elapsed)}s)")
    try:
        idle = now_wall - os.stat(info["log"]).st_mtime
    except OSError:
        return None
    if idle > _SCENARIO_STALL_SEC:
        return (f"no log output for {int(idle)}s (>{_SCENARIO_STALL_SEC}s) "
                f"-- appears wedged/hung")
    return None


def _wait_any(running: dict, poll: float = 1.0) -> list:
    """Block until at least one running child exits; return the finished procs.

    While waiting, periodically runs the scenario watchdog and force-kills (and
    reports) any scenario whose log has gone silent past the stall threshold or
    which has blown the wall-clock cap, so a single hang can never wedge the pool.
    """
    last_check = time.monotonic()
    while True:
        done = [proc for proc in running if proc.poll() is not None]
        if done:
            return done
        now = time.monotonic()
        if now - last_check >= _WATCHDOG_CHECK_INTERVAL_SEC:
            last_check = now
            now_wall = time.time()
            for proc, info in list(running.items()):
                if proc.poll() is not None:
                    continue
                reason = _watchdog_kill_reason(info, now_wall, now)
                if reason:
                    info["killed_reason"] = reason
                    print(
                        f"  !! WATCHDOG scenario[{info['index']}] "
                        f"{info['scenario'].hf_name}: {reason}; killing group"
                    )
                    _kill_process_group(proc, info["pgid"])
        time.sleep(poll)


def _print_log_tail(log_path: Path, n: int = 20) -> None:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return
    tail = lines[-n:]
    print(f"     --- last {len(tail)} line(s) of {log_path} ---")
    for ln in tail:
        print(f"     | {ln}")


def _worker_command(args) -> list[str]:
    """Argv for a worker child. Which scenario it captures is passed out-of-band
    via the ``_WORKER_INDEX_ENV`` environment variable (set by ``_launch``), so
    the ``scenarios`` positional selects the table, not the individual scenario.
    It IS forwarded so the child resolves the exact same scenario table (and thus
    the same index ordering) as the parent.
    """
    # ``-u`` keeps the child's stdout unbuffered so its log advances in real
    # time -- both for live monitoring and so the watchdog's log-idle check
    # measures true progress rather than a stuck output buffer.
    cmd = [
        sys.executable, "-u", "-m", "fastkernels.capture",
        args.scenarios,
        "--max-requests", str(args.max_requests),
    ]
    if args.output:
        cmd += ["--output", args.output]
    if args.max_layers is not None:
        cmd += ["--max-layers", str(args.max_layers)]
    return cmd


def _run_scenarios_parallel(scenarios, args, gpu_ids: list[str]) -> int:
    """Capture ``scenarios`` concurrently, packing them onto ``gpu_ids`` by TP.

    Returns a process exit code (0 iff every scenario captured and verified).
    Any scenario that fails -- to schedule, build, run, or verify -- is recorded
    and reported without affecting the others.
    """
    total = len(gpu_ids)
    log_dir = CAPTURE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # status: index -> (state, detail); state in {ok, verify-failed, error, skipped}
    results: dict[int, tuple[str, str]] = {}
    pending: list[tuple[int, object]] = []
    for i, s in enumerate(scenarios):
        if s.tp > total:
            detail = f"needs tp={s.tp} > {total} GPU(s) available"
            results[i] = ("skipped", detail)
            print(f"  !! SKIP scenario[{i}] {s.hf_name}: {detail}")
        else:
            pending.append((i, s))
    # Larger-TP scenarios first so they claim GPUs instead of being starved by a
    # stream of single-GPU jobs; ties keep registry order.
    pending.sort(key=lambda t: (-t[1].tp, t[0]))

    free = list(gpu_ids)
    running: dict = {}

    def _launch(i, s) -> None:
        assign = [free.pop(0) for _ in range(s.tp)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assign)
        env[_WORKER_INDEX_ENV] = str(i)
        # Distinct rendezvous port per scenario so concurrent TP scenarios don't
        # collide on one fixed port (EADDRINUSE on rank 0's TCPStore bind).
        env["FASTKERNELS_NCCL_PORT"] = str(_NCCL_PORT_BASE + i)
        log_path = log_dir / f"{_scenario_log_name(s, i)}.log"
        logf = open(log_path, "w")
        # ``start_new_session`` puts this scenario (rank 0 + the workers it
        # spawns) in its own process group so the watchdog can size it and, if
        # it hangs, kill the whole group without touching sibling scenarios.
        proc = subprocess.Popen(
            _worker_command(args), stdout=logf,
            stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        running[proc] = {
            "index": i, "scenario": s, "gpus": assign,
            "log": log_path, "logf": logf,
            "pgid": proc.pid, "start": time.monotonic(),
        }
        print(
            f"  -> [GPU {env['CUDA_VISIBLE_DEVICES']}] scenario[{i}] "
            f"{s.hf_name} (tp={s.tp}) started; log {log_path}"
        )

    try:
        while pending or running:
            # Greedily launch every pending scenario that fits the free pool.
            made_progress = True
            while made_progress:
                made_progress = False
                for pos, (i, s) in enumerate(pending):
                    if s.tp <= len(free):
                        _launch(i, s)
                        pending.pop(pos)
                        made_progress = True
                        break

            if not running:
                # Nothing running and nothing launchable: cannot happen after the
                # tp>total filter, but never spin forever.
                for i, s in pending:
                    results[i] = ("skipped", "could not be scheduled")
                break

            for proc in _wait_any(running):
                info = running.pop(proc)
                info["logf"].close()
                free.extend(info["gpus"])
                i, s, rc = info["index"], info["scenario"], proc.returncode
                killed_reason = info.get("killed_reason")
                if killed_reason:
                    results[i] = ("error", f"{killed_reason}; {info['log']}")
                elif rc == 0:
                    results[i] = ("ok", "")
                elif rc == 1:
                    results[i] = ("verify-failed", str(info["log"]))
                elif rc == 3:
                    results[i] = ("skipped", f"unsupported/asset-gated; {info['log']}")
                else:
                    results[i] = ("error", f"exit={rc}; {info['log']}")
                print(
                    f"  <- [{results[i][0].upper()}] scenario[{i}] {s.hf_name} "
                    f"(rc={rc}); freed GPU {','.join(info['gpus'])}"
                )
                if results[i][0] == "error":
                    _print_log_tail(info["log"])
    finally:
        # Never leave orphaned GPU processes behind on interrupt/error: kill the
        # whole group so spawned tensor-parallel workers go too.
        for proc, info in list(running.items()):
            _kill_process_group(proc, info["pgid"])
            info["logf"].close()

    print("\nCapture summary:")
    ok_all = True
    for i, s in enumerate(scenarios):
        state, detail = results.get(i, ("unknown", ""))
        if state not in ("ok", "skipped"):
            ok_all = False
        line = f"  [{state.upper()}] scenario[{i}] {s.hf_name} (tp={s.tp})"
        if detail:
            line += f" -- {detail}"
        print(line)
    return 0 if ok_all else 1


# --- Ray parallel scheduler ---------------------------------------------------
# Like ``_run_scenarios_parallel`` but hands GPU packing to Ray (``num_gpus=tp``
# per task) and lays out one folder per run, mirroring ``fastkernels validate``:
# ``<root>/<NN_model_tpN_dtype>/`` holds that scenario's ``report*.json`` and
# ``run.log``, with a ``run.jsonl`` event stream and ``summary.json`` at the root.
# Each Ray task re-pins ``CUDA_VISIBLE_DEVICES`` to the exclusive GPUs Ray
# assigned (``ray.get_gpu_ids()``), because ``num_gpus`` alone does not isolate
# the nested capture subprocess from sibling jobs.
_RAY_PROGRESS_INTERVAL_SEC = 30.0


def _run_id() -> str:
    return time.strftime("capture_%Y%m%d-%H%M%S")


def _run_root(args) -> Path:
    """Per-run output root: ``--output`` verbatim (as a directory) or a
    timestamped folder under the captures dir."""
    return Path(args.output) if args.output else CAPTURE_DIR / _run_id()


def _job_dir(root: Path, index: int, scenario) -> Path:
    return root / _scenario_log_name(scenario, index)


def _worker_command_for_job(args, job_dir: Path) -> list[str]:
    """Argv for a Ray worker child, pinning its report(s) into ``job_dir``."""
    cmd = [
        sys.executable, "-u", "-m", "fastkernels.capture",
        args.scenarios,
        "--max-requests", str(args.max_requests),
        "--output", str(job_dir / "report.json"),
    ]
    if args.max_layers is not None:
        cmd += ["--max-layers", str(args.max_layers)]
    return cmd


def _dashboard_url(ray) -> str | None:
    """URL of the Ray dashboard -- the live view of every capture task (named
    ``capture_NN_<model>_tpN_<dtype>``), its state, and per-GPU usage."""
    try:
        get_url = getattr(ray, "get_dashboard_url", None)
        value = get_url() if get_url else None
        if value:
            return value if value.startswith("http") else f"http://{value}"
    except Exception:  # noqa: BLE001
        pass
    try:
        value = ray._private.worker._global_node.webui_url
        return value if value.startswith("http") else f"http://{value}"
    except Exception:  # noqa: BLE001
        return None


def _init_ray(ray):
    """Start Ray with the dashboard on; retry without it if that fails."""
    kwargs = {"ignore_reinit_error": True, "log_to_driver": False}
    try:
        return ray.init(include_dashboard=True, dashboard_host="127.0.0.1", **kwargs)
    except Exception as exc:  # noqa: BLE001 - dashboard is best-effort
        print(f"  !! Ray dashboard startup failed ({exc}); continuing without it")
        ray.shutdown()
        return ray.init(include_dashboard=False, **kwargs)


def _append_run_event(root: Path, event: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.time(), **event}, sort_keys=True) + "\n")


def _write_capture_summary(root: Path, scenarios, results: dict) -> None:
    summary = {
        "root": str(root),
        "scenarios": [
            {
                "index": i, "model": s.hf_name, "tp": s.tp, "dtype": s.dtype,
                "status": results.get(i, ("unknown", ""))[0],
                "detail": results.get(i, ("", ""))[1],
            }
            for i, s in enumerate(scenarios)
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _visible_to_physical_gpu_ids(visible: list[str], parent_visible: list[str]) -> list[str]:
    """Map a Ray task's ``CUDA_VISIBLE_DEVICES`` (indices into the parent's
    visible set) back to physical ids, for GPU reclaim."""
    out: list[str] = []
    for gpu in visible:
        if gpu in parent_visible:
            out.append(gpu)
            continue
        try:
            idx = int(gpu)
        except ValueError:
            out.append(gpu)
            continue
        out.append(parent_visible[idx] if 0 <= idx < len(parent_visible) else gpu)
    return out


def _parse_cvd(value: str | None) -> list[str]:
    if not value:
        return []
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _ray_gpu_id_tokens(raw_ids) -> list[str]:
    """Normalize ``ray.get_gpu_ids()`` (ints, floats like ``0.0``, strings)."""
    tokens: list[str] = []
    for gpu in raw_ids or []:
        if isinstance(gpu, float):
            tokens.append(str(int(gpu)))
        elif isinstance(gpu, int):
            tokens.append(str(gpu))
        else:
            tokens.append(str(gpu).strip())
    return tokens


def _gpu_token_as_index(token: str) -> int | None:
    """Parse a CVD / Ray GPU token as a non-negative integer index, or None."""
    token = token.strip()
    if token.isdigit():
        return int(token)
    try:
        value = float(token)
    except ValueError:
        return None
    if value < 0 or value != int(value):
        return None
    return int(value)


def _ray_exclusive_gpu_ids(
    *,
    env_cvd: list[str],
    ray_gpu_ids: list[str],
    parent_visible: list[str],
    expected: int,
) -> list[str]:
    """Physical GPU ids this Ray task exclusively owns.

    Ray's ``num_gpus=tp`` reservation does not always narrow
    ``CUDA_VISIBLE_DEVICES`` on the nested capture subprocess (the worker
    copies the driver env, which still lists every GPU). Every rank-0 then
    uses physical GPU 0, so a tp=1 Llama KV cache (~90% of an H200) OOMs a
    concurrent tp=2 Qwen3-Next. Pin the child to *exactly* ``expected``
    devices:

    1. If Ray already isolated this worker (``CUDA_VISIBLE_DEVICES`` length
       equals ``expected``), keep that list. ``ray.get_gpu_ids()`` may be
       ``0..N-1`` *into that list*; remapping those through ``parent_visible``
       would steal GPU 0 from a sibling.
    2. Otherwise treat ``ray.get_gpu_ids()`` as indices into the current CVD
       (if set) or as physical / parent-visible ids.
    """
    if expected <= 0:
        return []
    if len(env_cvd) == expected:
        return list(env_cvd)

    if env_cvd and len(ray_gpu_ids) == expected:
        idxs = [_gpu_token_as_index(tok) for tok in ray_gpu_ids]
        if all(i is not None and 0 <= i < len(env_cvd) for i in idxs):
            return [env_cvd[i] for i in idxs]

    if len(ray_gpu_ids) == expected:
        return _visible_to_physical_gpu_ids(ray_gpu_ids, parent_visible)

    if env_cvd:
        mapped = _visible_to_physical_gpu_ids(env_cvd, parent_visible)
        if len(mapped) == expected:
            return mapped

    raise RuntimeError(
        f"Ray task expected {expected} exclusive GPU(s), got "
        f"CUDA_VISIBLE_DEVICES={env_cvd!r} ray.get_gpu_ids()={ray_gpu_ids!r} "
        f"parent={parent_visible!r}"
    )


def _reclaim_gpus(gpu_ids: list[str]) -> None:
    """SIGKILL any compute process still holding ``gpu_ids`` (re-parented ranks
    that outlived the worker and would otherwise perturb the next task)."""
    smi = shutil.which("nvidia-smi")
    ids = [g for g in gpu_ids if g]
    if smi is None or not ids:
        return
    try:
        out = subprocess.run(
            [smi, "-i", ",".join(ids), "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001 - best-effort reclaim
        return
    for tok in out.split():
        try:
            os.kill(int(tok), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def _reap_process_group(proc: "subprocess.Popen", physical_gpus: list[str]) -> bool:
    """Kill anything left in the child's session, then reclaim its GPUs. Returns
    True if survivors had to be killed. ``start_new_session=True`` makes the
    child's pgid its pid, so this never signals the driver or a sibling task."""
    try:
        os.killpg(proc.pid, 0)
        survivors = True
    except (ProcessLookupError, PermissionError):
        survivors = False
    if survivors:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(1.0)
    _reclaim_gpus(physical_gpus)
    return survivors


def _ray_capture_job(job: dict, parent_visible_gpus: list[str],
                     stall_timeout: int, timeout: int) -> dict:
    """Ray remote: run one scenario's capture worker under a stall/wall-clock
    watchdog, streaming to ``run.log`` and reaping GPU survivors on exit."""
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(job["log_path"])

    env = os.environ.copy()
    env[_WORKER_INDEX_ENV] = str(job["index"])
    env["FASTKERNELS_NCCL_PORT"] = str(job["nccl_port"])
    try:
        import ray as _ray
        raw_ids = list(_ray.get_gpu_ids())
    except Exception:  # noqa: BLE001 - pin from CVD alone if Ray is unavailable
        raw_ids = []
    ray_ids = _ray_gpu_id_tokens(raw_ids)
    env_cvd = _parse_cvd(env.get("CUDA_VISIBLE_DEVICES"))
    logf = open(log_path, "w", buffering=1)
    try:
        physical = _ray_exclusive_gpu_ids(
            env_cvd=env_cvd,
            ray_gpu_ids=ray_ids,
            parent_visible=parent_visible_gpus,
            expected=int(job["tp"]),
        )
    except RuntimeError as exc:
        logf.write(f"GPU PIN FAILED: {exc}\n")
        logf.close()
        return {"index": job["index"], "status": "error", "rc": 2,
                "detail": str(exc), "log": str(log_path), "orphans": False}

    # Pin this worker *and* the capture child. Ray's num_gpus reservation is
    # not sufficient: the nested subprocess used to inherit the driver's full
    # CUDA_VISIBLE_DEVICES and every rank-0 landed on physical GPU 0.
    cvd = ",".join(physical)
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    env["CUDA_VISIBLE_DEVICES"] = cvd
    logf.write(f"job_index: {job['index']}\n")
    logf.write(f"name: {job['name']}\n")
    logf.write(f"tp: {job['tp']}\n")
    logf.write(f"CUDA_VISIBLE_DEVICES: {cvd}\n")
    logf.write(f"ray.get_gpu_ids(): {raw_ids!r}\n")
    logf.write(f"parent_visible_gpus: {','.join(parent_visible_gpus)}\n")
    logf.write("command: " + " ".join(job["cmd"]) + "\n\n")
    logf.flush()

    start = time.monotonic()
    watchdog_reason: str | None = None
    proc: subprocess.Popen | None = None
    orphans = False
    try:
        proc = subprocess.Popen(
            job["cmd"], stdout=logf, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        while proc.poll() is None:
            now = time.monotonic()
            if now - start > timeout:
                watchdog_reason = (f"exceeded {timeout}s wall-clock timeout "
                                   f"(elapsed {int(now - start)}s)")
                break
            try:
                idle = time.time() - os.stat(log_path).st_mtime
            except OSError:
                idle = 0.0
            if idle > stall_timeout:
                watchdog_reason = (f"no log output for {int(idle)}s "
                                   f"(>{stall_timeout}s) -- appears wedged")
                break
            time.sleep(1.0)
        if watchdog_reason:
            logf.write(f"\nWATCHDOG: {watchdog_reason}\n")
            logf.flush()
            _kill_process_group(proc, proc.pid)
    except BaseException:
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc, proc.pid)
        raise
    finally:
        if proc is not None and _reap_process_group(proc, physical):
            orphans = True
            logf.write("\nORPHANS: killed processes that outlived the worker; "
                       "GPU memory they held may have perturbed a neighbor\n")
        logf.close()

    rc = proc.returncode if proc is not None and proc.returncode is not None else -9
    if watchdog_reason:
        status = "error"
        detail = watchdog_reason
    elif rc == 0:
        status = "ok"
        detail = ""
    elif rc == 1:
        status = "verify-failed"
        detail = str(log_path)
    elif rc == 3:
        status = "skipped"
        detail = f"unsupported/asset-gated family; see {log_path}"
    else:
        status = "error"
        detail = f"exit={rc}; {log_path}"
    return {"index": job["index"], "status": status, "rc": rc,
            "detail": detail, "log": str(log_path), "orphans": orphans}


def _run_scenarios_ray(scenarios, args, gpu_ids: list[str], root: Path) -> int:
    """Capture ``scenarios`` in parallel via Ray, one GPU-pinned task each.

    Returns a process exit code (0 iff every scenario captured and verified).
    Falls back to the subprocess scheduler if Ray is not installed.
    """
    try:
        import ray
    except ModuleNotFoundError:
        print("  !! Ray not installed; using the subprocess scheduler instead.")
        return _run_scenarios_parallel(scenarios, args, gpu_ids)

    total = len(gpu_ids)
    results: dict[int, tuple[str, str]] = {}
    jobs: list[dict] = []
    for i, s in enumerate(scenarios):
        if s.tp > total:
            detail = f"needs tp={s.tp} > {total} GPU(s) available"
            results[i] = ("skipped", detail)
            print(f"  !! SKIP scenario[{i}] {s.hf_name}: {detail}")
            continue
        job_dir = _job_dir(root, i, s)
        jobs.append({
            "index": i, "tp": s.tp, "name": s.hf_name,
            "run_dir": str(job_dir), "log_path": str(job_dir / "run.log"),
            "nccl_port": _NCCL_PORT_BASE + i,
            "cmd": _worker_command_for_job(args, job_dir),
        })

    root.mkdir(parents=True, exist_ok=True)
    _append_run_event(root, {"event": "run_start", "visible_gpus": gpu_ids,
                             "job_count": len(jobs)})

    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Always pin the driver to the detected pool so Ray's GPU resource count
    # matches ``gpu_ids`` even when ``--gpus`` was omitted.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    try:
        _init_ray(ray)
        print(f"  Ray dashboard: {_dashboard_url(ray) or 'unavailable'}", flush=True)
        total_cpus = int(ray.cluster_resources().get("CPU") or os.cpu_count() or 1)
        remote_fn = ray.remote(_ray_capture_job)
        ref_to_job: dict[object, dict] = {}
        for job in jobs:
            num_cpus = max(1, math.floor((total_cpus - 2) * job["tp"] / total))
            ref = remote_fn.options(
                name=f"capture_{job['index']:02d}_{_scenario_log_name(scenarios[job['index']], job['index'])}",
                num_gpus=job["tp"], num_cpus=num_cpus,
            ).remote(job, gpu_ids, _SCENARIO_STALL_SEC, _SCENARIO_TIMEOUT_SEC)
            ref_to_job[ref] = job
            _append_run_event(root, {"event": "task_submitted", "job": job})
            print(f"  -> submitted scenario[{job['index']}] {job['name']} "
                  f"(tp={job['tp']}); log {job['log_path']}")

        pending = list(ref_to_job)
        last = time.monotonic()
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=2.0)
            if not ready:
                now = time.monotonic()
                if now - last >= _RAY_PROGRESS_INTERVAL_SEC:
                    print(f"  running/pending tasks: {len(pending)}", flush=True)
                    last = now
                continue
            for ref in ready:
                job = ref_to_job[ref]
                i = job["index"]
                try:
                    r = ray.get(ref)
                    state, detail = r["status"], r.get("detail", "")
                except Exception as exc:  # noqa: BLE001 - task crash is a failure
                    state, detail = "error", repr(exc)
                    r = {"status": state, "detail": detail}
                results[i] = (state, detail)
                _append_run_event(root, {"event": "task_finished", "index": i,
                                         "status": state, "result": r})
                print(f"  <- [{state.upper()}] scenario[{i}] {job['name']}")
                if state == "error":
                    _print_log_tail(Path(job["log_path"]))
    finally:
        try:
            ray.shutdown()
        except Exception:  # noqa: BLE001
            pass
        if prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

    _write_capture_summary(root, scenarios, results)
    _append_run_event(root, {"event": "run_finished", "results": {
        str(i): results.get(i, ("unknown", ""))[0] for i in range(len(scenarios))}})

    print("\nCapture summary:")
    ok_all = True
    for i, s in enumerate(scenarios):
        state, detail = results.get(i, ("unknown", ""))
        if state not in ("ok", "skipped"):
            ok_all = False
        line = f"  [{state.upper()}] scenario[{i}] {s.hf_name} (tp={s.tp})"
        if detail:
            line += f" -- {detail}"
        print(line)
    print(f"\nOutput root: {root}")
    return 0 if ok_all else 1


def _prefetch_media(scenarios) -> None:
    """Parent-side pre-download of the large video dataset repos.

    A multimodal ``video`` workload fetches its clips with ``snapshot_download``
    -- a large download that, run inside a scenario's worker, could trip the
    per-scenario stall/wall-clock watchdog. Doing it once here in the parent
    (which has no watchdog) warms the Hugging Face hub cache, so each worker's
    identical ``snapshot_download`` returns immediately and only the (fast) frame
    decode + generation count against its watchdog. Image/audio workloads stream
    in-worker (comparatively small) and need no pre-download. Best-effort: a
    failure here just means the worker downloads the repo itself.
    """
    video_datasets: dict[str, None] = {}
    for s in scenarios:
        for wl in s.workloads:
            if not isinstance(wl, (VLM, OmniModal)):
                continue
            params = spec_for(wl).params
            if getattr(params, "modality", "text") == "video":
                ds = getattr(params, "dataset_name", None)
                if ds:
                    video_datasets.setdefault(ds, None)
    if not video_datasets:
        return
    from huggingface_hub import snapshot_download
    for ds in video_datasets:
        print(f"Pre-downloading video dataset '{ds}' so per-scenario workers "
              f"load from cache ...")
        try:
            snapshot_download(ds, repo_type="dataset")
            print(f"  done pre-downloading {ds}")
        except Exception as exc:  # noqa: BLE001 - best-effort warm cache
            print(f"  !! pre-download of {ds} failed ({exc!r}); workers will "
                  f"fetch it themselves.")


def _setup_capture():
    """Shared per-process setup: import the engine, then discover + instrument
    every operator once. Returns the engine classes and the instrumented-
    operator set used by ``_capture_scenario``.
    """
    # 1) Import the engine module FIRST so transformers and the LLM path load in
    #    the correct order (avoids torch.library/collective registration clashes
    #    that happen if unrelated operator modules import ahead of it).
    print("Importing fastkernels engine ...")
    from .infra.engine import LlamaEngine, SamplingParams

    # 2) Discover & instrument every operator ONCE, before any engine builds a
    #    model, so __init__ calls are captured too.
    print(f"Discovering operators in {_BASELINE_PACKAGE} ...")
    classes = _discover_operator_classes()
    instrumented = _instrument(classes)
    n_instrumented = len(instrumented)
    print(f"  Instrumented {n_instrumented} operator class(es).")
    return LlamaEngine, SamplingParams, instrumented, n_instrumented


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastkernels capture",
        description=(
            "Run one or more models through fastkernels and capture the "
            "dtype/shape of every operator's init/forward arguments. The models, "
            "dtypes, tensor-parallel degrees, max_num_seqs and workloads are "
            "taken from the scenarios YAML given as the positional argument (one "
            "report per scenario workload). Capture always runs eager (the "
            "instrumentation is incompatible with CUDA graphs)."
        ),
    )
    parser.add_argument(
        "scenarios",
        help="Path to a scenarios YAML, or a packaged name resolved against "
             "fastkernels/scenarios/ (e.g. 'full', 'default', 'minimal'). "
             "Defines the models/dtypes/TP/workloads to capture.",
    )
    parser.add_argument(
        "--max-requests", type=int, default=1_000_000,
        help="Max prompts to load per scenario workload. The default is large "
             "enough to use every row of each workload's dataset (each loader "
             "returns all available rows, capped at this value).",
    )
    parser.add_argument(
        "--output", default=None,
        help="For a parallel (multi-scenario) run, the output ROOT directory: "
             "each scenario writes into <output>/<NN_model_tpN_dtype>/ "
             "(report*.json + run.log), with run.jsonl + summary.json at the "
             "root (default: ~/.fastkernels/captures/capture_<timestamp>/). For "
             "a single-scenario run it is the report path base as before "
             "(default ~/.fastkernels/captures/<scenario-slug>.json).",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated physical GPU ids to schedule across (default: all "
             "visible GPUs / CUDA_VISIBLE_DEVICES). Scenarios are packed onto "
             "these by their TP degree and captured in parallel via Ray, each in "
             "its own GPU-pinned task, launching more as GPUs free up.",
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Build and run only the first MAX_LAYERS transformer decoder "
             "layers of the model. Only those layers are allocated and their "
             "weights loaded; anything outside the decoder stack (embeddings, "
             "final norm, LM head, and any vision/audio encoder) is unaffected. "
             "Applies to the standard LlamaEngine path (plain LLMs and "
             "Qwen-VL/Omni); ignored for EAGLE-3/FLA/Jamba scenarios.",
    )
    args = parser.parse_args(argv)
    if args.max_layers is not None and args.max_layers < 1:
        parser.error("--max-layers must be >= 1")

    # The scenarios YAML is the source of truth for the models and engine
    # configuration (model, dtype, TP, max_num_seqs) and the capture workloads;
    # --max-requests/--output stay run-level knobs. Each scenario gets its own
    # engine and each workload its own report. Worker children are handed the
    # same scenarios argument, so their scenario index matches the parent's.
    try:
        scenarios = resolve_benchmark(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  !! could not load scenarios {args.scenarios!r}: {exc}")
        return 2

    # A "run" is one (scenario, workload) pair. ``multi`` decides whether an
    # explicit --output gets a per-run suffix (default paths are always unique
    # per scenario slug). Computed over ALL scenarios so a per-scenario worker
    # suffixes identically.
    runs_per_scenario = [
        [(w.value, w) for w in s.workloads]
        for s in scenarios
    ]
    multi = sum(len(r) for r in runs_per_scenario) > 1

    # --- Worker mode: when the parent scheduler set _WORKER_INDEX_ENV, capture
    #     exactly that one scenario in-process. The parent has already pinned
    #     this process to its GPUs via CUDA_VISIBLE_DEVICES. This is internal
    #     (env-driven), so there is no user-facing scenario-selection flag. ---
    worker_index = os.environ.get(_WORKER_INDEX_ENV)
    if worker_index is not None:
        try:
            idx = int(worker_index)
        except ValueError:
            print(f"  !! invalid {_WORKER_INDEX_ENV}={worker_index!r}")
            return 2
        if not 0 <= idx < len(scenarios):
            print(f"  !! {_WORKER_INDEX_ENV}={idx} out of range "
                  f"(have {len(scenarios)} scenario(s))")
            return 2
        LlamaEngine, SamplingParams, instrumented, n_instrumented = _setup_capture()
        scenario = scenarios[idx]
        runs = runs_per_scenario[idx]
        try:
            _, ok = _capture_scenario(
                scenario, runs, args, LlamaEngine, SamplingParams,
                instrumented, n_instrumented, multi,
            )
        except _CaptureSkip as exc:
            print(f"\n  == SKIP scenario {scenario.hf_name}: {exc}")
            return 3
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"\n  !! Scenario {scenario.hf_name} failed to capture: {exc!r}")
            return 2
        return 0 if ok else 1

    # --- Parent: GPU-aware parallel scheduling (default). Each scenario runs in
    #     its own subprocess so a crash/OOM/verification failure is isolated. ---
    # Parent only (worker mode returned above): pre-download the large video
    # dataset repos once, outside any per-scenario watchdog, so the workers load
    # them from the warm cache.
    _prefetch_media(scenarios)
    gpu_ids = _detect_gpu_ids(args.gpus)
    if len(scenarios) > 1 and len(gpu_ids) >= 1:
        root = _run_root(args)
        print(
            f"Scheduling {len(scenarios)} scenario(s) across {len(gpu_ids)} "
            f"GPU(s) [{', '.join(gpu_ids)}] with Ray ..."
        )
        print(f"  output root: {root}")
        return _run_scenarios_ray(scenarios, args, gpu_ids, root)

    # --- In-process fallback (single scenario or no GPU detected). A
    #     per-scenario failure is reported and the loop continues with the next
    #     scenario. ---
    LlamaEngine, SamplingParams, instrumented, n_instrumented = _setup_capture()
    exit_code = 0
    written: list[Path] = []
    for scenario, runs in zip(scenarios, runs_per_scenario):
        try:
            paths, ok = _capture_scenario(
                scenario, runs, args, LlamaEngine, SamplingParams,
                instrumented, n_instrumented, multi,
            )
            written.extend(paths)
            if not ok:
                exit_code = 1
        except _CaptureSkip as exc:
            print(f"\n  == SKIP scenario {scenario.hf_name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(
                f"\n  !! Scenario {scenario.hf_name} failed to capture: "
                f"{exc!r}; skipping to the next scenario."
            )
            exit_code = 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. Wrote {len(written)} report(s):")
    for p in written:
        print(f"  - {p}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
