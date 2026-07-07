"""Capture per-operator init/forward tensor metadata for a fastkernels model.

Runs a model end-to-end through the fastkernels ``LlamaEngine`` on a set of
short prompts and records the ``dtype``/``shape`` of every argument passed to
the ``__init__`` and ``forward`` of each ``nn.Module`` subclass defined under
``fastkernels.tasks.baseline`` (i.e. every fastkernels operator). Results are
aggregated per operator class and written to a JSON file.

The model, dtype, tensor-parallel degree, eager mode, ``max_num_seqs`` and the
capture workloads all come from a list of ``BenchmarkScenario`` objects
(``CAPTURE_SCENARIOS`` below): each scenario is loaded into its own engine and
every workload it lists is captured to a separate report.

Scenarios are captured in parallel across the available GPUs. Each scenario
runs in its own child process pinned to a private set of GPUs (a ``tp=N``
scenario claims N GPUs) via ``CUDA_VISIBLE_DEVICES``; the scheduler packs
scenarios onto the GPU pool by TP degree and launches the next one as soon as
enough GPUs free up. Because every scenario is isolated in its own process, a
crash, OOM or CUDA fault in one never brings down the others -- it is recorded
as that scenario's failure and the rest continue. Use ``--gpus`` to restrict the
pool (with a single GPU the scenarios simply run one at a time).

Usage::

    python -m fastkernels capture                          # parallel, all GPUs
    python -m fastkernels capture --gpus 0,1,2,3           # restrict the GPU pool
    python -m fastkernels capture --output /tmp/llama32_1b_ops.json
    python -m fastkernels.capture --selftest   # offline scheduler fuzz, no GPU

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
import functools
import gc
import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .registry import FULL_BENCHMARK
from .workloads import load_real_prompt_workload

# Default directory for capture reports (override per-run with ``--output``).
CAPTURE_DIR = Path.home() / ".fastkernels" / "captures"

# Internal env var the parallel scheduler uses to tell a worker subprocess which
# scenario to capture. It is set (alongside CUDA_VISIBLE_DEVICES) by the parent
# and read at startup by the child; it is deliberately NOT a user-facing CLI
# flag, so a normal ``fastkernels capture`` invocation never sees it.
_WORKER_INDEX_ENV = "FK_CAPTURE_WORKER_INDEX"

# The package that holds every fastkernels operator (nn.Module) definition.
_BASELINE_PACKAGE = "fastkernels.tasks.baseline"

# Optional, architecture-specific GPU kernel libraries. ``deep_gemm`` is
# imported eagerly by ``infra.weight_loader`` (via the DeepSeek chain) but is
# only *invoked* by non-Llama architectures. When it is absent we register a
# stub so the Llama path can still be loaded and captured; the stub raises if
# any of its symbols are ever actually called, so it can never silently affect
# a real computation. (We deliberately do NOT stub ``fla`` and friends: those
# only affect linear-attention operators, and stubbing them makes their
# submodule imports partially succeed and pollute global torch state.)
_OPTIONAL_KERNEL_DEPS = ("deep_gemm",)


class _MissingDependencyStub(types.ModuleType):
    """A placeholder module whose attributes raise only when *called*.

    Dunder attributes (``__file__``, ``__path__``, ``__spec__`` ...) must behave
    like a normal module's, so we raise ``AttributeError`` for them; otherwise
    ``inspect``/``importlib`` machinery scanning ``sys.modules`` would receive a
    function and crash.
    """

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        module_name = self.__name__

        def _unavailable(*args, **kwargs):
            raise RuntimeError(
                f"Optional dependency '{module_name}' is not installed; "
                f"'{module_name}.{name}' cannot be used. This stub only exists "
                f"so unrelated operators can be imported for metadata capture."
            )

        return _unavailable


def _install_optional_dependency_stubs(names=_OPTIONAL_KERNEL_DEPS) -> list[str]:
    """Register stubs for any of ``names`` that are not importable."""
    stubbed = []
    for name in names:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError, ModuleNotFoundError):
            pass
        sys.modules[name] = _MissingDependencyStub(name)
        stubbed.append(name)
    return stubbed

# NOTE: The capture prompts come from standardized BenchmarkScenario workloads
# (``CAPTURE_SCENARIOS`` below). The old ``DEFAULT_PROMPTS`` list (and the
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

# Capture scenarios: each entry is loaded into its own engine and every workload
# it lists is captured to a separate report. We capture the full standardized
# benchmark set from the registry, each scenario with its own workloads.
CAPTURE_SCENARIOS = FULL_BENCHMARK[:]

# qualified_name -> {"init": {...}, "forward": {...}}
_RECORDS: dict[str, dict] = {}

# Independent forward capture via torch hooks, used to verify _RECORDS.
# qualified_name -> {"calls": int, "keys": {json_key: count}}
_HOOK_RECORDS: dict[str, dict] = {}


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


def _record(qualname: str, module_name: str, class_name: str,
            method: str, call_summary: dict) -> None:
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
        slot["variants"][key] = {"count": 1, "args": call_summary}
    else:
        variant["count"] += 1


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
        try:
            summary = (
                _summarize_call(sig, args, kwargs)
                if sig is not None
                else {"_args": [_summarize(v) for v in args[1:]]}
            )
            _record(qualname, cls.__module__, cls.__name__, record_key, summary)
        except Exception:
            pass
        return raw(*args, **kwargs)

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
# forward pass, subject to two budgets -- at most ``max_num_seqs`` concurrent
# sequences and at most ``max_num_batched_tokens`` tokens per step. Given only
# each request's prompt length and generated length (plus those two budgets) we
# can replay that policy analytically and predict, for every step, how many
# prefill tokens / prefill sequences / decode sequences it processes.
#
# Those three numbers fully determine the leading ("sequence length" / batch)
# dimension of every forward tensor:
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
    __slots__ = ("prompt_len", "gen_len", "remaining_prefill", "generated")

    def __init__(self, prompt_len: int, gen_len: int):
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.remaining_prefill = prompt_len
        self.generated = 0


def _simulate_continuous_batching(prompt_lens, gen_lens, max_num_seqs,
                                  max_num_batched_tokens) -> list[dict]:
    """Replay the engine's unified chunked-prefill schedule analytically.

    Returns one dict per GPU step with the token/sequence composition the
    engine's forward pass would see that step.
    """
    from collections import deque

    waiting = deque(_MockSeq(p, g) for p, g in zip(prompt_lens, gen_lens))
    prefilling: list[_MockSeq] = []   # admitted, prefill not yet complete
    running: list[_MockSeq] = []      # prefilled, still generating
    steps: list[dict] = []

    # Safety bound: a correct schedule needs one step per generated token plus
    # a bounded number of prefill steps; anything past this signals a bug.
    max_steps = sum(gen_lens) + sum(prompt_lens) + len(prompt_lens) + 16

    while waiting or prefilling or running:
        if len(steps) > max_steps:
            break

        num_decode = len(running)
        token_budget = max_num_batched_tokens - num_decode
        chunks: list[tuple[_MockSeq, int]] = []

        # 1) continue in-flight (partially prefilled) sequences first, FIFO.
        for seq in prefilling:
            if token_budget <= 0:
                break
            chunk = min(seq.remaining_prefill, token_budget)
            if chunk <= 0:
                continue
            chunks.append((seq, chunk))
            token_budget -= chunk

        # 2) admit new waiting sequences up to the concurrency + token budgets.
        live = num_decode + len(prefilling)
        while waiting and live < max_num_seqs and token_budget > 0:
            seq = waiting.popleft()
            chunk = min(seq.remaining_prefill, token_budget)
            chunks.append((seq, chunk))
            token_budget -= chunk
            prefilling.append(seq)
            live += 1

        prefill_tokens = sum(c for _, c in chunks)
        steps.append({
            "prefill_tokens": prefill_tokens,
            "num_prefill_seqs": len(chunks),
            "prefill_seq_lens": sorted((c for _, c in chunks), reverse=True),
            "decode_seqs": num_decode,
            "total_tokens": prefill_tokens + num_decode,
        })

        # ---- apply the step's effects ----
        for seq in running:               # each running seq emits one token
            seq.generated += 1
        for seq, chunk in chunks:         # advance prefill cursors
            seq.remaining_prefill -= chunk
        completed = [s for s in prefilling if s.remaining_prefill == 0]
        if completed:
            for seq in completed:
                seq.generated = 1         # first token comes from the prefill
            prefilling = [s for s in prefilling if s.remaining_prefill > 0]
            running.extend(completed)
        running = [s for s in running if s.generated < s.gen_len]

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

    The prefill attention kernel runs once per layer per step, so raw call
    counts are divided by ``num_layers`` to recover per-step frequencies.

    Raw counts are accumulated per leading dimension *before* dividing: models
    with heterogeneous per-layer attention (e.g. GPT-OSS alternates sliding-
    window and full attention) emit several distinct forward signatures per step
    that share the same ``q`` / ``cu_seqlens_q`` leading dims but split the layer
    count between them. Dividing each signature's count individually would round
    fractional per-step frequencies to zero; aggregating first keeps them whole.
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
    lyr = max(num_layers, 1)
    tokens = {n: int(round(c / lyr)) for n, c in raw_tokens.items()}
    seqs = {s: int(round(c / lyr)) for s, c in raw_seqs.items()}
    total_tokens = sum(n * steps for n, steps in tokens.items())
    return {"tokens": tokens, "seqs": seqs, "total_tokens": total_tokens}


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
                           max_num_batched_tokens, num_layers) -> dict:
    """Cross-check the captured forward shapes against a mock scheduler."""
    steps = _simulate_continuous_batching(
        prompt_lens, gen_lens, max_num_seqs, max_num_batched_tokens,
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
    for qualname, entry in sorted(_RECORDS.items()):
        out_entry = {}
        for method in ("init", "forward"):
            slot = entry[method]
            if slot is None:
                continue
            out_entry[method] = {
                "calls": slot["calls"],
                "variants": [
                    {"count": v["count"], "args": v["args"]}
                    for v in slot["variants"].values()
                ],
            }
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
def _scenario_slug(scenario, workload: str, num_requests: int) -> str:
    """Report-filename stem encoding the distinguishing scenario fields for a
    single ``workload``.

    Two runs that differ in any of these values write to different files (no
    timestamp needed); re-running the same scenario/workload/``--num-requests``
    overwrites its report. ``None`` max_num_seqs renders as ``auto``; capture
    always runs eager, so the mode tag is fixed.
    """
    model = scenario.hf_name.replace("/", "__")
    max_num_seqs = scenario.max_num_seqs if scenario.max_num_seqs is not None else "auto"
    return (
        f"{model}_tp{scenario.tp}_{scenario.dtype}_{workload}"
        f"_req{num_requests}_seqs{max_num_seqs}_eager"
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


def _report_path(output_arg, scenario, workload: str, multi: bool, num_requests: int) -> Path:
    """Resolve the report path for one workload run.

    Default: ``CAPTURE_DIR/<scenario-slug>.json``. An explicit ``--output`` is
    honored verbatim for a single run, or has the (model-qualified) scenario
    slug suffixed onto its stem when several runs share one ``--output`` base.
    """
    slug = _scenario_slug(scenario, workload, num_requests)
    if output_arg is not None:
        p = Path(output_arg)
        return p.with_name(f"{p.stem}_{slug}{p.suffix}") if multi else p
    return CAPTURE_DIR / f"{slug}.json"


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


def _capture_scenario(scenario, runs, args, LlamaEngine, SamplingParams,
                      instrumented, n_instrumented, multi):
    """Build one scenario's engine and capture each of its workloads.

    Returns ``(written_paths, ok)`` where ``ok`` is False if any workload's
    verification failed. Raises if the engine build or a workload run itself
    errors, so the caller can report the scenario and move on to the next one.
    The engine is always released before returning so the next (possibly larger)
    model has room on the GPU.
    """
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
            # The loader returns every available row up to ``--num-requests``
            # (the default is large enough to use them all); curated sets return
            # fewer if they have fewer rows (e.g. long-context has 64).
            n_req = args.num_requests
            print(f"Loading '{wl_label}' workload prompts ({n_req}) ...")
            samples = load_real_prompt_workload(
                wl_label, engine.tokenizer, num_requests=n_req,
            )
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
            # ignore_eos is left False so short answers still stop naturally at
            # EOS (keeping stored responses clean); the dataset length is an
            # upper bound. bench_vllm.py forces ignore_eos=True for throughput
            # determinism -- flip it here if exact-length decode is wanted.
            sampling = [
                SamplingParams(temperature=0.0, max_tokens=mnt)
                for mnt in max_new_tokens
            ]
            print(
                f"Running {len(gen_prompts)} prompt(s) at max_batch_size="
                f"{engine.max_num_seqs}, max_new_tokens="
                f"[{min(max_new_tokens)}..{max(max_new_tokens)}] "
                f"(chat_template={used_chat_template}) ..."
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

            outputs = engine.generate(gen_prompts, sampling, use_tqdm=False)

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
            # Full prompt token counts (the model always sees the complete
            # chat-templated prompt). Reused below for schedule verification.
            prompt_lens = [
                len(p) if isinstance(p, (list, tuple))
                else len(engine.tokenizer(p).input_ids)
                for p in gen_prompts
            ]
            # Re-decode the generated ids WITH special tokens so the response
            # keeps its chat-template structure (e.g. gpt-oss harmony channels
            # <|channel|>analysis<|message|>...). The engine's ``generated_text``
            # is decoded with skip_special_tokens=True, which would otherwise
            # glue channel labels onto the text ("analysisWe need to...").
            responses = [
                {
                    "prompt": prompt_texts[i],
                    "prompt_tokens": prompt_lens[i],
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
                for i in range(len(gen_prompts))
            ]
            generation = {
                "num_prompts": len(gen_prompts),
                "max_batch_size": engine.max_num_seqs,
                "batch_refilled": len(gen_prompts) > engine.max_num_seqs,
                "used_chat_template": used_chat_template,
                "max_new_tokens_source": "dataset_response_length",
                "max_new_tokens_range": [min(max_new_tokens), max(max_new_tokens)],
                "num_finished_by_eos": num_eos,
                "num_hit_max_tokens": len(gen_lengths) - num_eos,
                "responses": responses,
            }
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
            #     chunked-prefill replay driven only by the per-request prompt
            #     and generated lengths plus the engine's batch/token budgets.
            schedule_verification = _verify_batch_schedule(
                prompt_lens, gen_lengths,
                engine.max_num_seqs, engine.max_num_batched_tokens,
                _infer_num_layers(engine),
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
            out_path = _report_path(args.output, scenario, wl_label, multi, args.num_requests)
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


def _wait_any(running: dict, poll: float = 1.0) -> list:
    """Block until at least one running child exits; return the finished procs."""
    while True:
        done = [proc for proc in running if proc.poll() is not None]
        if done:
            return done
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
    there is no user-facing scenario-selection flag.
    """
    cmd = [
        sys.executable, "-m", "fastkernels.capture",
        "--num-requests", str(args.num_requests),
    ]
    if args.output:
        cmd += ["--output", args.output]
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
        log_path = log_dir / f"{_scenario_log_name(s, i)}.log"
        logf = open(log_path, "w")
        proc = subprocess.Popen(
            _worker_command(args), stdout=logf,
            stderr=subprocess.STDOUT, env=env,
        )
        running[proc] = {
            "index": i, "scenario": s, "gpus": assign,
            "log": log_path, "logf": logf,
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
                if rc == 0:
                    results[i] = ("ok", "")
                elif rc == 1:
                    results[i] = ("verify-failed", str(info["log"]))
                else:
                    results[i] = ("error", f"exit={rc}; {info['log']}")
                print(
                    f"  <- [{results[i][0].upper()}] scenario[{i}] {s.hf_name} "
                    f"(rc={rc}); freed GPU {','.join(info['gpus'])}"
                )
                if rc != 0:
                    _print_log_tail(info["log"])
    finally:
        # Never leave orphaned GPU processes behind on interrupt/error.
        for proc, info in list(running.items()):
            proc.terminate()
            info["logf"].close()

    print("\nCapture summary:")
    ok_all = True
    for i, s in enumerate(scenarios):
        state, detail = results.get(i, ("unknown", ""))
        if state != "ok":
            ok_all = False
        line = f"  [{state.upper()}] scenario[{i}] {s.hf_name} (tp={s.tp})"
        if detail:
            line += f" -- {detail}"
        print(line)
    return 0 if ok_all else 1


def _setup_capture():
    """Shared per-process setup: stub optional deps, import the engine, then
    discover + instrument every operator once. Returns the engine classes and
    the instrumented-operator set used by ``_capture_scenario``.
    """
    # 0) Stub out optional GPU kernel libs (deep_gemm) if missing so the engine
    #    and the DeepSeek operator chain can be imported.
    stubbed = _install_optional_dependency_stubs()
    if stubbed:
        print(f"  Stubbed missing optional deps: {', '.join(stubbed)}")

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
            "taken from CAPTURE_SCENARIOS (one report per scenario workload). "
            "Capture always runs eager (the instrumentation is incompatible with "
            "CUDA graphs)."
        ),
    )
    parser.add_argument(
        "--num-requests", type=int, default=1_000_000,
        help="Max prompts to load per scenario workload. The default is large "
             "enough to use every row of each workload's dataset (each loader "
             "returns all available rows, capped at this value).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Base path for the JSON report(s) (default: "
             "~/.fastkernels/captures/<scenario-slug>.json per scenario "
             "workload). With multiple runs, the scenario slug is suffixed onto "
             "this base to keep the paths distinct.",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run the offline mock-scheduler self-check and exit (no model).",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated physical GPU ids to schedule across (default: all "
             "visible GPUs / CUDA_VISIBLE_DEVICES). Scenarios are packed onto "
             "these by their TP degree and captured in parallel, each in its own "
             "GPU-pinned subprocess, launching more as GPUs free up.",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest_simulator()

    # CAPTURE_SCENARIOS is the source of truth for the models and engine
    # configuration (model, dtype, TP, max_num_seqs) and the capture workloads;
    # --num-requests/--output stay run-level knobs. Each scenario gets its own
    # engine and each workload its own report.
    scenarios = CAPTURE_SCENARIOS

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
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"\n  !! Scenario {scenario.hf_name} failed to capture: {exc!r}")
            return 2
        return 0 if ok else 1

    # --- Parent: GPU-aware parallel scheduling (default). Each scenario runs in
    #     its own subprocess so a crash/OOM/verification failure is isolated. ---
    gpu_ids = _detect_gpu_ids(args.gpus)
    if len(scenarios) > 1 and len(gpu_ids) >= 1:
        print(
            f"Scheduling {len(scenarios)} scenario(s) across {len(gpu_ids)} "
            f"GPU(s) [{', '.join(gpu_ids)}] by TP degree ..."
        )
        return _run_scenarios_parallel(scenarios, args, gpu_ids)

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


def _selftest_simulator(trials: int = 2000, seed: int = 0) -> int:
    """Offline, property-based check of the mock scheduler (no GPU/model).

    Instead of comparing against baked-in numbers (which are meaningless once
    the prompts come from an arbitrary dataset), this fuzzes the simulator with
    randomized request sets and asserts the structural invariants that must hold
    for *any* input and *any* budgets:

      * every prompt token is prefilled exactly once
        -> sum of prefill tokens == sum(prompt lengths);
      * every generated token after the first comes from a decode step
        -> sum of decode participations == sum(generated - 1);
      * no step exceeds ``max_num_batched_tokens``;
      * no step runs more than ``max_num_seqs`` concurrent sequences;
      * a single prompt longer than the token budget is chunked (never dropped)
        and still sums back to its prompt length.

    Run with ``python -m fastkernels.capture --selftest``.
    """
    import random

    rng = random.Random(seed)
    failures: list[str] = []
    for t in range(trials):
        n = rng.randint(1, 40)
        prompt_lens = [rng.randint(1, 200) for _ in range(n)]
        gen_lens = [rng.randint(1, 60) for _ in range(n)]
        max_num_seqs = rng.randint(1, 8)
        # Sometimes force a tiny token budget so chunked prefill actually kicks
        # in (prompt longer than one step's budget).
        max_batched = rng.choice([16, 32, max(prompt_lens), 4096, 16384])

        steps = _simulate_continuous_batching(
            prompt_lens, gen_lens, max_num_seqs, max_batched)

        total_prefill = sum(s["prefill_tokens"] for s in steps)
        total_decode = sum(s["decode_seqs"] for s in steps)
        problems = []
        if total_prefill != sum(prompt_lens):
            problems.append(
                f"prefill {total_prefill} != sum(prompt) {sum(prompt_lens)}")
        if total_decode != sum(g - 1 for g in gen_lens):
            problems.append(
                f"decode {total_decode} != sum(gen-1) "
                f"{sum(g - 1 for g in gen_lens)}")
        if steps and max(s["total_tokens"] for s in steps) > max_batched:
            problems.append("a step exceeded the token budget")
        if steps and max(
                s["num_prefill_seqs"] + s["decode_seqs"] for s in steps
        ) > max_num_seqs:
            problems.append("a step exceeded max_num_seqs")
        if problems:
            failures.append(
                f"trial {t} (seqs={max_num_seqs}, budget={max_batched}, "
                f"prompts={prompt_lens}, gens={gen_lens}): "
                + "; ".join(problems))

    ok = not failures
    print(f"selftest: {trials} randomized trials -> "
          f"{'PASS' if ok else 'FAIL'}")
    for msg in failures[:10]:
        print(f"  ! {msg}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
