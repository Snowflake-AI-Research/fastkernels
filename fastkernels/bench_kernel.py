"""Kernel-level benchmarking: score candidate kernels against their baseline.

``fastkernels bench`` benchmarks every candidate kernel under
``fastkernels/tasks/candidate/L{1..4}/`` against the matching baseline operator
in ``fastkernels/tasks/baseline/`` -- for both **numerical correctness** and
**performance**. The shapes/dtypes/init args a kernel is exercised with come
straight from the capture reports written by ``fastkernels capture``
(``~/.fastkernels/captures/*.json``): each report records, per operator, the
``__init__`` recipes and ``forward`` argument shapes/dtypes observed in real
model runs. This tool is the *consumer* of those reports; it replaces the
retired ``fastkernels/bench/kernels`` suite (removed in commit c26874a) whose
bespoke shape/golden "InputRegistry" was superseded by ``capture.py``.

Design (adapted from NVIDIA's SOL-ExecBench ``core/bench`` + fastkernels'
``capture.py`` scheduler):

* **Isolation** -- one operator per subprocess, at most one subprocess per GPU
  (pinned via ``CUDA_VISIBLE_DEVICES`` so each child sees its device as
  ``cuda:0``). A crash / OOM / illegal-memory fault in one candidate cannot take
  down the others, monkey-patch reward-hack checks start from a clean
  interpreter, and GPU memory is fully reclaimed between operators. A parent
  scheduler packs operators onto the GPU pool and watches for hangs.
* **Correctness** -- the baseline is reconstructed from its captured ``__init__``
  recipe; the candidate is built with the *same* args and given the baseline's
  weights (``load_state_dict(..., strict=False)``); the *same* materialized
  inputs are fed to both and their outputs compared with per-dtype tolerances
  over several rounds of fresh inputs.
* **Performance** -- CUDA-event timing with warmup, an L2-cache flush and a
  shifting memory pool (distinct ``data_ptr`` per iteration, no in-loop
  ``cudaMalloc``); the median of many iterations. Baseline is timed too, giving
  a speedup factor.
* **Reward-hack detection** -- the timing primitive's identity, this module's
  own critical functions, background-thread counts and output-tensor types are
  all checked. Any tampering is reported as ``REWARD_HACK``. (Runtime kernel
  compilation is *not* blocked -- fastkernels kernels are legitimately built
  from source via ``cpp_extension.load`` at import.)

Usage::

    fastkernels bench --list                       # what would run
    fastkernels bench --dry-run --self-test        # full plan (shapes), run nothing
    fastkernels bench                              # all candidates vs baseline
    fastkernels bench --self-test --level 1        # baselines vs themselves
    fastkernels bench --target rms_norm --gpus 0   # one operator, one GPU

Known limitations (honest by design):

* Correctness uses shared **random** weights (captures don't store weights) --
  a candidate-vs-baseline equivalence check, not a golden check against a spec.
* Inputs are materialized from shape+dtype plus name heuristics. Operators whose
  ``forward`` args include live module handles, opaque runtime objects or
  deeply-nested/collapsed values are reported ``SKIPPED``, not benchmarked. A
  case whose generated inputs the baseline itself rejects is also ``SKIPPED``.
* One GPU per operator (tp=1) -- matches the L1/L2 kernel scope; multi-GPU
  operators would need the tp-packing logic from ``capture.py``.

No existing top-level ``fastkernels`` command conflicts with ``bench`` (the
``fastkernels/bench/{e2e,eval}`` packages are only reachable via
``python -m fastkernels.bench.*``), so nothing needs to be disabled.

Code map (see the ``PART N`` banners below):

* PART 1  Setup -- imports, worker stdout hardening, constants, exceptions.
* PART 2  Discovery & captures (shared) -- operator identity, candidate
          loading, report loading, per-operator case selection.
* PART 3  Benchmark primitives (worker-side) -- module reconstruction, input
          materialization, tensor-tree helpers, correctness, reward-hack, timing.
* PART 4  Results & reporting -- the ScenarioResult/BenchResult model, the
          table printer and the JSON writer.
* PART 5  Worker -- benchmark one operator (all its cases) on a single GPU.
* PART 6  Parent scheduler -- GPU-pool packing, watchdog, clock locking.
* PART 7  CLI -- argument parsing and the ``main`` entry point.
"""

# ###########################################################################
# PART 1  ·  SETUP -- imports, worker stdout hardening, constants, exceptions
# ###########################################################################

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Worker stdout hardening -- MUST happen before ``import torch``.
#
# A worker child emits its JSONL results on the *original* stdout fd while every
# other byte (torch/JIT/Triton chatter, our own progress prints) is redirected
# to stderr, so the result stream can never be corrupted by library noise. We
# save a dup of the real stdout, then point fd 1 at fd 2 (stderr). ``_emit``
# writes to the saved fd; ``print`` and native writes go to the log.
# ---------------------------------------------------------------------------
_IS_WORKER = ("--worker" in sys.argv) or bool(os.environ.get("FK_BENCH_WORKER"))
if _IS_WORKER:
    _REAL_STDOUT_FD = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(1, "w", buffering=1)
else:
    _REAL_STDOUT_FD = 1

import torch  # noqa: E402  -- imported after the fd redirect above
import torch.nn as nn  # noqa: E402

from fastkernels import CANDIDATE_DIR, RESULTS_DIR, run_output_path  # noqa: E402
from fastkernels import capture  # noqa: E402  (reconstruct_op / _reconstruct_init_call / CAPTURE_DIR)

# Captured at import, before any candidate code is imported: the identity of the
# CUDA-event timing primitive. If a candidate monkey-patches it to fake timings,
# the id changes and we catch it (see ``_check_monkey_patch``).
_ELAPSED_ADDR = id(torch.cuda.Event.elapsed_time)


# ---------------------------------------------------------------------------
# Constants / defaults (mirroring SOL-ExecBench BenchmarkConfig + the old
# fastkernels per-dtype tolerances).
# ---------------------------------------------------------------------------
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50
DEFAULT_ROUNDS = 3
DEFAULT_MAX_SHAPES = 5
DEFAULT_SEED = 200

# 99% of elements must fall within the per-dtype allclose bound (SOL rule).
REQUIRED_MATCHED_RATIO = 0.99

# Per-dtype (atol, rtol). Tighter for fp32, loose for low precision.
_FP8_TYPES = tuple(
    t for t in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if t is not None
)
_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-5, 1e-3),
    torch.float64: (1e-5, 1e-3),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
    **{t: (0.125, 0.125) for t in _FP8_TYPES},
}
_DEFAULT_TOL = (1e-2, 1e-2)

# Watchdog / subprocess bounds (parent side).
_EVAL_TIMEOUT_SEC = int(os.environ.get("FK_BENCH_TIMEOUT_SEC", "1200"))
_STALL_SEC = int(os.environ.get("FK_BENCH_STALL_SEC", "600"))
_WATCHDOG_INTERVAL_SEC = 10.0
_TERM_GRACE_SEC = 8.0

# GPU clock-lock presets (name substring -> (graphics MHz, memory MHz)).
_CLOCK_PRESETS = {
    "NVIDIA B200": (1500, 3996),
    "NVIDIA H100": (1410, 1593),
    "NVIDIA A100": (1065, 1215),
}

# Result statuses.
PASSED = "PASSED"
INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
INCORRECT_SHAPE = "INCORRECT_SHAPE"
INCORRECT_DTYPE = "INCORRECT_DTYPE"
RUNTIME_ERROR = "RUNTIME_ERROR"
REWARD_HACK = "REWARD_HACK"
SKIPPED = "SKIPPED"
_PASS_STATUSES = {PASSED}
_FAIL_STATUSES = {INCORRECT_NUMERICAL, INCORRECT_SHAPE, INCORRECT_DTYPE,
                  RUNTIME_ERROR, REWARD_HACK}


class _RewardHack(RuntimeError):
    """A candidate tried to tamper with timing / eval integrity."""


class _UnsupportedInput(Exception):
    """A captured forward argument cannot be materialized (skip the case)."""


# ###########################################################################
# PART 2  ·  DISCOVERY & CAPTURES (shared by parent + worker)
# ###########################################################################

# ---------------------------------------------------------------------------
# Operator identity (parsed from capture keys) + candidate loading.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Operator:
    qualname: str          # e.g. fastkernels.tasks.baseline.L1.embedding:Embedding
    level: int             # 1..4
    stem: str              # module file stem, e.g. "embedding"
    class_name: str        # e.g. "Embedding"

    @property
    def module_path(self) -> str:
        return self.qualname.split(":", 1)[0]


_LEVEL_RE = re.compile(r"^L([1-4])$")


def _parse_operator(qualname: str) -> Operator | None:
    """Turn a capture ``module:Class`` key into an :class:`Operator` (or None)."""
    if ":" not in qualname:
        return None
    module_path, class_name = qualname.split(":", 1)
    parts = module_path.split(".")
    level = None
    for p in parts:
        m = _LEVEL_RE.match(p)
        if m:
            level = int(m.group(1))
    if level is None or not parts:
        return None
    return Operator(qualname, level, parts[-1], class_name)


def _import_symbol(qualname: str):
    """``module:QualName`` -> the live object (mirrors ``capture._import_symbol``)."""
    module_name, _, name = qualname.partition(":")
    obj = importlib.import_module(module_name)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _candidate_file(op: Operator) -> Path:
    return CANDIDATE_DIR / f"L{op.level}" / f"{op.stem}.py"


def _load_candidate_class(op: Operator) -> type | None:
    """Load the candidate class for *op* from ``tasks/candidate/L{level}/{stem}.py``.

    Prefers a proper dotted import (``fastkernels.tasks.candidate.L{level}.{stem}``,
    a PEP-420 namespace subpackage) so intra-candidate relative imports work;
    falls back to loading the file by path when the candidate dir has been moved
    via ``FASTKERNELS_CANDIDATE_DIR``. Returns ``None`` if the file or the class
    named identically to the baseline is absent.
    """
    if not _candidate_file(op).exists():
        return None
    dotted = op.module_path.replace(".tasks.baseline.", ".tasks.candidate.")
    module = None
    try:
        module = importlib.import_module(dotted)
    except Exception:
        module = None
    if module is None:
        try:
            spec = importlib.util.spec_from_file_location(
                f"_fk_candidate_L{op.level}_{op.stem}", _candidate_file(op))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception:
            return None
    return getattr(module, op.class_name, None)


# ---------------------------------------------------------------------------
# Capture loading + case selection.
# ---------------------------------------------------------------------------
def _load_reports(captures_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(captures_dir.glob("*.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return reports


def _discover_operators(reports: list[dict]) -> dict[str, Operator]:
    """All baseline operators (with at least one forward variant) seen in captures."""
    ops: dict[str, Operator] = {}
    for rep in reports:
        for qualname, entry in rep.get("operators", {}).items():
            if not entry.get("forward") or not entry["forward"].get("variants"):
                continue
            if ".tasks.baseline." not in qualname:
                continue
            op = _parse_operator(qualname)
            if op is not None:
                ops.setdefault(qualname, op)
    return ops


def _collect_cases(op: Operator, reports: list[dict], max_shapes: int) -> list[dict]:
    """Select up to *max_shapes* distinct (init, forward) cases for *op*.

    Each case pairs a captured ``forward`` variant with the ``__init__`` variant
    that built the instance it ran on (``init_variant_ids``). Cases are
    deduplicated across reports by (init args, forward args) and ranked by
    observed call ``count`` so the hottest real shapes are benchmarked first.
    """
    cases: list[dict] = []
    seen: set[tuple] = set()
    for rep in reports:
        operators = rep.get("operators", {})
        entry = operators.get(op.qualname)
        if not entry or not entry.get("forward"):
            continue
        init_slot = entry.get("init")
        n_init = len(init_slot["variants"]) if init_slot else 0
        for fv in entry["forward"]["variants"]:
            ivids = [v for v in (fv.get("init_variant_ids") or []) if 0 <= v < n_init]
            vid = ivids[0] if ivids else (0 if n_init else None)
            fwd_args = fv.get("args", {})
            init_args_json = ""
            if vid is not None:
                init_args_json = json.dumps(init_slot["variants"][vid]["args"], sort_keys=True)
            sig = (vid, init_args_json, json.dumps(fwd_args, sort_keys=True))
            if sig in seen:
                continue
            seen.add(sig)
            cases.append({
                "operators": operators,
                "vid": vid,
                "fwd_args": fwd_args,
                "count": int(fv.get("count", 0)),
            })
    cases.sort(key=lambda c: -c["count"])
    return cases[:max_shapes]


# ###########################################################################
# PART 3  ·  BENCHMARK PRIMITIVES (worker-side)
#            reconstruction -> inputs -> tensor helpers -> correctness
#            -> reward-hack -> timing
# ###########################################################################

# ---------------------------------------------------------------------------
# Module reconstruction + weight sharing.
# ---------------------------------------------------------------------------
def _scalar_init_args(operators: dict, qualname: str, vid: int | None) -> dict:
    """Top-level scalar ``__init__`` args (num_embeddings, hidden_size, ...)
    used as bounds for integer-input heuristics."""
    if vid is None:
        return {}
    try:
        call = operators[qualname]["init"]["variants"][vid]["args"]
    except (KeyError, IndexError, TypeError):
        return {}
    return {k: v for k, v in call.items() if isinstance(v, (int, float, bool, str))}


def _build_modules(op: Operator, baseline_cls: type, candidate_cls: type,
                   operators: dict, vid: int | None, device: str):
    """Instantiate baseline + candidate with identical (reconstructed) init args."""
    if vid is not None:
        call = operators[op.qualname]["init"]["variants"][vid]["args"]
        if not capture.is_reconstructable(call, operators):
            raise _UnsupportedInput("unreconstructable __init__ recipe")
        b_args, b_kwargs = capture._reconstruct_init_call(call, operators=operators, device=device)
        c_args, c_kwargs = capture._reconstruct_init_call(call, operators=operators, device=device)
    else:
        b_args, b_kwargs, c_args, c_kwargs = [], {}, [], {}
    baseline = baseline_cls(*b_args, **b_kwargs)
    candidate = candidate_cls(*c_args, **c_kwargs)
    return baseline, candidate


def _prepare_module(module: nn.Module, dtype: torch.dtype, device: str) -> nn.Module:
    """Move to *device*, cast floating parameters to *dtype* (fp16/bf16/fp32
    only -- never fp8, which the op stores explicitly), and switch to eval."""
    module = module.to(device)
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        for p in module.parameters():
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype)
    module.eval()
    return module


def _case_dtype(fwd_args: dict) -> torch.dtype:
    """The dominant floating compute dtype of a case (first fp16/bf16/fp32
    tensor input); defaults to bfloat16."""
    for v in _iter_tensor_specs(fwd_args):
        dt = getattr(torch, v["dtype"], None)
        if isinstance(dt, torch.dtype) and dt in (torch.float16, torch.bfloat16, torch.float32):
            return dt
    return torch.bfloat16


def _iter_tensor_specs(obj):
    """Yield every ``{"shape","dtype"}`` tensor spec nested in a forward-args dict."""
    if isinstance(obj, dict):
        if isinstance(obj.get("dtype"), str) and isinstance(obj.get("shape"), list):
            yield obj
            return
        for v in obj.values():
            yield from _iter_tensor_specs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensor_specs(v)


# ---------------------------------------------------------------------------
# Input materialization.
# ---------------------------------------------------------------------------
def _materialize_tensor(name: str, shape, dtype_str: str, device: str, init_args: dict):
    dt = getattr(torch, dtype_str, None)
    if not isinstance(dt, torch.dtype):
        raise _UnsupportedInput(f"{name}: unknown dtype {dtype_str!r}")
    shape = tuple(int(s) for s in shape)
    if dt in _FP8_TYPES:
        return torch.randn(shape, device=device, dtype=torch.float32).clamp_(-2, 2).to(dt)
    if dt.is_floating_point:
        return torch.randn(shape, device=device, dtype=dt)
    if dt == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    return _materialize_int(name, shape, dt, device, init_args)


def _materialize_int(name: str, shape, dt: torch.dtype, device: str, init_args: dict):
    """Best-effort valid integer tensors (indices / positions / cu_seqlens).

    Structured tensors get a semantically-valid shape; everything else is a
    bounded ``randint`` so the baseline does not index out of range. If the
    guess is still invalid the baseline will raise and the case is SKIPPED.
    """
    lname = name.lower().rsplit(".", 1)[-1]
    if "cu_seqlen" in lname or "cu_seq" in lname:
        # Monotonic non-decreasing cumulative lengths starting at 0.
        n = int(shape[-1])
        cu = torch.zeros(shape, device=device, dtype=dt)
        if n > 1:
            chunks = torch.randint(1, 64, (n - 1,), device=device, dtype=dt)
            cu.reshape(-1)[:n][1:] = torch.cumsum(chunks, 0)
        return cu
    bound = 128
    if "input_id" in lname or "token" in lname:
        bound = int(init_args.get("num_embeddings")
                    or init_args.get("vocab_size") or 1024)
    elif "position" in lname:
        bound = int(init_args.get("max_position_embeddings") or 2048)
    elif ("slot" in lname or "block_table" in lname or "cache_seqlen" in lname
          or "seqlen" in lname or "seq_len" in lname):
        bound = 256
    elif "grid" in lname:
        bound = 32
    return torch.randint(0, max(2, bound), shape, device=device, dtype=dt)


def _materialize_value(name: str, v, device: str, init_args: dict):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, list):
        return [_materialize_value(f"{name}[{i}]", x, device, init_args)
                for i, x in enumerate(v)]
    if isinstance(v, dict):
        if isinstance(v.get("dtype"), str) and isinstance(v.get("shape"), list):
            return _materialize_tensor(name, v["shape"], v["dtype"], device, init_args)
        # A bare torch.dtype summary, e.g. {"dtype": "bfloat16"}.
        if set(v) == {"dtype"} and isinstance(v["dtype"], str):
            dt = getattr(torch, v["dtype"], None)
            if isinstance(dt, torch.dtype):
                return dt
            raise _UnsupportedInput(f"{name}: unknown dtype {v['dtype']!r}")
        # A torch.device handle, summarized as {"object": "device"}.
        if v.get("object") == "device":
            return torch.device(device)
        # A collapsed / opaque / handle summary -- cannot be materialized.
        if any(k in v for k in ("module", "object", "list", "dict", "$opaque", "$tensor")):
            raise _UnsupportedInput(f"{name}: {sorted(v)[:1]}")
        # A plain nested dict (e.g. **kwargs): recurse.
        return {k: _materialize_value(f"{name}.{k}", x, device, init_args)
                for k, x in v.items()}
    raise _UnsupportedInput(f"{name}: {type(v).__name__}")


def _partition_cu(total: int, nseg: int, device, dtype) -> torch.Tensor:
    """A valid cumulative-sequence-length vector: ``nseg`` roughly-equal
    segments of ``total`` tokens, as ``[0, s1, s1+s2, ..., total]``."""
    if nseg <= 0:
        return torch.zeros(1, device=device, dtype=dtype)
    base, rem = divmod(int(total), nseg)
    cu = [0]
    for i in range(nseg):
        cu.append(cu[-1] + base + (1 if i < rem else 0))
    return torch.tensor(cu, device=device, dtype=dtype)


def _fix_structured_inputs(kwargs: dict) -> None:
    """Rewrite ``cu_seqlens*`` tensors so they are consistent with the query/key
    row counts of the same call (varlen attention needs a valid partition that
    sums to the number of tokens). Best-effort, keyed by conventional names.
    """
    pairs = [("cu_seqlens_q", "q"), ("cu_seqlens_q", "query"),
             ("cu_seqlens_k", "k"), ("cu_seqlens_k", "key"),
             ("cu_seqlens", "q"), ("cu_seqlens", "query"), ("cu_seqlens", "hidden_states")]
    for cn, qn in pairs:
        cu = kwargs.get(cn)
        q = kwargs.get(qn)
        if (isinstance(cu, torch.Tensor) and cu.dim() == 1 and cu.numel() >= 2
                and isinstance(q, torch.Tensor) and q.dim() >= 1):
            kwargs[cn] = _partition_cu(q.shape[0], cu.numel() - 1, cu.device, cu.dtype)


def _build_call(cls: type, fwd_args: dict, device: str, init_args: dict):
    """Materialize a callable ``(args, kwargs)`` for ``cls.forward`` from a
    captured forward-args summary, honoring the forward signature's arg kinds."""
    if set(fwd_args) == {"_args"}:  # signature-binding fell back to positional
        vals = [_materialize_value(f"arg{i}", v, device, init_args)
                for i, v in enumerate(fwd_args["_args"])]
        return vals, {}
    try:
        params = list(inspect.signature(cls.forward).parameters.values())[1:]  # drop self
    except (TypeError, ValueError):
        params = None
    if not params:
        # No signature: pass everything by keyword.
        kw = {k: _materialize_value(k, v, device, init_args) for k, v in fwd_args.items()}
        _fix_structured_inputs(kw)
        return [], kw
    call_args: list = []
    call_kwargs: dict = {}
    for p in params:
        if p.name not in fwd_args:
            if p.default is not inspect.Parameter.empty:
                continue
            raise _UnsupportedInput(f"missing required forward arg {p.name!r}")
        val = _materialize_value(p.name, fwd_args[p.name], device, init_args)
        if p.kind is p.VAR_POSITIONAL:
            call_args.extend(val if isinstance(val, (list, tuple)) else [val])
        elif p.kind is p.VAR_KEYWORD:
            call_kwargs.update(val if isinstance(val, dict) else {})
        elif p.kind is p.POSITIONAL_ONLY:
            call_args.append(val)
        else:
            call_kwargs[p.name] = val
    _fix_structured_inputs(call_kwargs)
    return call_args, call_kwargs


# ---------------------------------------------------------------------------
# Tensor-tree helpers (clone / collect / map) shared by correctness + timing.
# ---------------------------------------------------------------------------
def _clone_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, list):
        return [_clone_tree(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tree(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone_tree(v) for k, v in obj.items()}
    return obj


def _map_tensors(obj, it):
    if isinstance(obj, torch.Tensor):
        return next(it)
    if isinstance(obj, list):
        return [_map_tensors(x, it) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_map_tensors(x, it) for x in obj)
    if isinstance(obj, dict):
        return {k: _map_tensors(v, it) for k, v in obj.items()}
    return obj


def _tensor_leaves(obj) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for x in obj for t in _tensor_leaves(x)]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _tensor_leaves(v)]
    return []


# ---------------------------------------------------------------------------
# Correctness (adapts SOL-ExecBench core/bench/correctness.py).
# ---------------------------------------------------------------------------
def _compare_tensor(out: torch.Tensor, ref: torch.Tensor):
    """Return (ok, max_abs, max_rel, matched_ratio, detail) for one tensor pair."""
    atol, rtol = _TOLERANCES.get(ref.dtype, _DEFAULT_TOL)
    x = out.detach().to(torch.float32)
    y = ref.detach().to(torch.float32)
    # Sanity: NaN / Inf / degenerate all-zeros.
    if torch.isnan(x).any() or torch.isnan(y).any():
        return False, float("inf"), float("inf"), 0.0, "NaN in output"
    if torch.isinf(x).any() or torch.isinf(y).any():
        return False, float("inf"), float("inf"), 0.0, "Inf in output"
    ref_norm = torch.linalg.vector_norm(y).item()
    if ref_norm > 0 and torch.linalg.vector_norm(x).item() == 0:
        return False, ref_norm, ref_norm, 0.0, "output is all-zeros"
    if x.numel() == 0:
        return True, 0.0, 0.0, 1.0, ""
    abs_err = (x - y).abs()
    max_abs = abs_err.max().item()
    bound = atol + rtol * y.abs()
    exceed = (abs_err > bound) | ~torch.isfinite(abs_err)
    matched = 1.0 - (exceed.sum().item() / exceed.numel())
    max_rel = (abs_err / y.abs().clamp_min(atol)).max().item()
    ok = matched >= REQUIRED_MATCHED_RATIO
    detail = "" if ok else f"only {matched:.4f} within atol={atol},rtol={rtol}"
    return ok, max_abs, max_rel, matched, detail


def _compare(out, ref, check_dtype: bool):
    """Structurally compare candidate vs reference outputs.

    Returns (status, max_abs, max_rel, matched_ratio, detail).
    """
    out_leaves = _tensor_leaves(out)
    ref_leaves = _tensor_leaves(ref)
    if len(out_leaves) != len(ref_leaves):
        return (INCORRECT_SHAPE, float("inf"), float("inf"), 0.0,
                f"output tensor count {len(out_leaves)} != {len(ref_leaves)}")
    worst_matched = 1.0
    max_abs = max_rel = 0.0
    for o, r in zip(out_leaves, ref_leaves):
        if tuple(o.shape) != tuple(r.shape):
            return (INCORRECT_SHAPE, float("inf"), float("inf"), 0.0,
                    f"shape {tuple(o.shape)} != {tuple(r.shape)}")
        if check_dtype and o.dtype != r.dtype:
            return (INCORRECT_DTYPE, float("inf"), float("inf"), 0.0,
                    f"dtype {o.dtype} != {r.dtype}")
        ok, a, rl, matched, detail = _compare_tensor(o, r)
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, rl)
        worst_matched = min(worst_matched, matched)
        if not ok:
            return INCORRECT_NUMERICAL, max_abs, max_rel, worst_matched, detail
    return PASSED, max_abs, max_rel, worst_matched, ""


# ---------------------------------------------------------------------------
# Reward-hack detection (ported from SOL-ExecBench core/bench/reward_hack.py).
# ---------------------------------------------------------------------------
_CRITICAL_NAMES = [
    "_time_module", "_compare", "_compare_tensor", "_run_forward",
    "_check_monkey_patch", "_check_threads", "_check_lazy_outputs",
    "_check_integrity", "_build_call", "_materialize_value",
]


def _check_monkey_patch() -> None:
    if id(torch.cuda.Event.elapsed_time) != _ELAPSED_ADDR:
        raise _RewardHack("torch.cuda.Event.elapsed_time was monkey-patched")


def _check_threads(before: int, after: int) -> None:
    if after > before:
        raise _RewardHack(
            f"background thread(s) injected during timing ({before} -> {after})")


def _check_lazy_outputs(outputs) -> None:
    for t in _tensor_leaves(outputs):
        if type(t) is not torch.Tensor:  # strict: rejects FakeTensor / meta / lazy
            raise _RewardHack(f"output is a non-materialized tensor: {type(t).__name__}")


def _snapshot_integrity() -> dict[str, int]:
    g = globals()
    return {n: id(g[n]) for n in _CRITICAL_NAMES if n in g}


def _check_integrity(snapshot: dict[str, int]) -> None:
    g = globals()
    for name, ident in snapshot.items():
        if name not in g or id(g[name]) != ident:
            raise _RewardHack(f"eval function was monkey-patched: {name}")


# ---------------------------------------------------------------------------
# Timing (CUDA events + L2 flush + shifting memory pool; adapts SOL timing.py).
# ---------------------------------------------------------------------------
def _l2_flush_buffer(device: str) -> torch.Tensor:
    try:
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
    except Exception:
        l2 = 50 * 1024 * 1024
    return torch.empty(int(2 * l2), dtype=torch.int8, device=device)


class _ShiftingPool:
    """Distinct ``data_ptr`` per iteration with ~1x memory and no in-loop malloc.

    Each source tensor gets one flat pool sized ``span + (iters-1)*step``; call
    ``i`` copies the (pristine) source into ``pool[i*step : i*step+span]`` and
    returns a view there, so consecutive iterations only shift the base address.
    Copying from a private clone means in-place kernels never corrupt later
    iterations. Adapted from SOL-ExecBench's ``ShiftingMemoryPoolAllocator``.
    """

    def __init__(self, tensors: list[torch.Tensor], total: int):
        self.entries = []
        self.total = total
        self.i = 0
        for t in tensors:
            t = t.contiguous()
            step = max(1, 256 // t.element_size())
            span = t.numel()
            pool = torch.empty(span + (total - 1) * step, dtype=t.dtype, device=t.device)
            src = t.reshape(-1).clone()
            self.entries.append((pool, src, span, step, tuple(t.shape)))

    def next(self) -> list[torch.Tensor]:
        idx = min(self.i, self.total - 1)
        self.i += 1
        out = []
        for pool, src, span, step, shape in self.entries:
            off = idx * step
            slot = pool.narrow(0, off, span)
            slot.copy_(src)
            out.append(slot.view(shape))
        return out


def _time_module(module: nn.Module, base_call, warmup: int, iters: int,
                 device: str) -> float:
    """Median CUDA-event latency (ms) of ``module(*args, **kwargs)``."""
    args, kwargs = base_call
    leaves = _tensor_leaves((args, kwargs))
    pool = _ShiftingPool(leaves, warmup + iters) if leaves else None
    l2 = _l2_flush_buffer(device)

    def call_once():
        if pool is None:
            return module(*args, **kwargs)
        it = iter(pool.next())
        a, k = _map_tensors((args, kwargs), it)
        return module(*a, **k)

    with torch.no_grad():
        for _ in range(warmup):
            l2.zero_()
            call_once()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            l2.zero_()
            starts[i].record()
            call_once()
            ends[i].record()
        torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.median(times)


def _run_forward(module: nn.Module, args, kwargs):
    with torch.no_grad():
        out = module(*args, **kwargs)
    torch.cuda.synchronize()
    return out


# ###########################################################################
# PART 4  ·  RESULTS & REPORTING (shared)
# ###########################################################################

# ---------------------------------------------------------------------------
# Result data model + reporting.
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    """One benchmarked (operator, shape) case; carries its own op + level."""
    op: str
    level: int
    shape: str
    dtype: str
    status: str
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    matched_ratio: float = 1.0
    candidate_ms: float = 0.0
    baseline_ms: float = 0.0
    speedup: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class BenchResult:
    """Every scenario from a run, plus table/JSON rendering."""
    mode: str
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def _count(self, statuses) -> int:
        return sum(1 for s in self.scenarios if s.status in statuses)

    def all_passed(self) -> bool:
        return not any(s.status in _FAIL_STATUSES for s in self.scenarios)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "passed": self._count(_PASS_STATUSES),
                "failed": self._count(_FAIL_STATUSES), "skipped": self._count({SKIPPED}),
                "all_passed": self.all_passed(),
                "scenarios": [s.to_dict() for s in self.scenarios]}

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def print_table(self) -> None:
        by_op: dict = {}
        for s in self.scenarios:
            by_op.setdefault((s.level, s.op), []).append(s)
        for (level, op), rows in sorted(by_op.items()):
            print(f"\n=== {op}  (L{level}) ===")
            print(f"  {'STATUS':<20} {'SHAPE / DTYPE':<34} {'ERR':>10} "
                  f"{'CAND ms':>10} {'BASE ms':>10} {'SPEEDUP':>8}")
            for s in rows:
                err = "-" if s.status == SKIPPED else f"{s.max_abs_error:.2e}"
                cms = f"{s.candidate_ms:.4f}" if s.candidate_ms else "-"
                bms = f"{s.baseline_ms:.4f}" if s.baseline_ms else "-"
                spd = f"{s.speedup:.2f}x" if s.speedup else "-"
                print(f"  {s.status:<20} {f'{s.shape} {s.dtype}'[:34]:<34} {err:>10} "
                      f"{cms:>10} {bms:>10} {spd:>8}")
                if s.detail and s.status != PASSED:
                    print(f"      -> {s.detail}")
        p, f, sk = (self._count(_PASS_STATUSES), self._count(_FAIL_STATUSES),
                    self._count({SKIPPED}))
        print(f"\n{'=' * 70}")
        print(f"TOTAL: {p} passed, {f} failed, {sk} skipped  "
              f"({'ALL PASSED' if self.all_passed() else 'FAILURES PRESENT'})")


# ###########################################################################
# PART 5  ·  WORKER -- benchmark one operator (all its cases) on one GPU
# ###########################################################################

# ---------------------------------------------------------------------------
# Worker: benchmark one operator (all its cases) on a single GPU.
# ---------------------------------------------------------------------------
def _json_safe(obj):
    """Replace non-finite floats with a large finite sentinel so our own result
    dicts (which use inf as an error marker) stay valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return 1e30
        if math.isinf(obj):
            return 1e30 if obj > 0 else -1e30
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _emit(obj: dict) -> None:
    """Write one strict-JSON result line to the saved real stdout fd."""
    line = json.dumps(_json_safe(obj), allow_nan=False) + "\n"
    os.write(_REAL_STDOUT_FD, line.encode())


def _shape_repr(fwd_args: dict) -> str:
    shapes = [tuple(v["shape"]) for v in _iter_tensor_specs(fwd_args)]
    return ",".join(str(list(s)) for s in shapes[:4]) or "scalar"


def _bench_one_case(op: Operator, baseline_cls, candidate_cls, case: dict,
                    device: str, warmup: int, iters: int, rounds: int,
                    seed: int, integrity: dict) -> ScenarioResult:
    operators = case["operators"]
    vid = case["vid"]
    fwd_args = case["fwd_args"]
    dtype = _case_dtype(fwd_args)
    init_args = _scalar_init_args(operators, op.qualname, vid)
    res = ScenarioResult(op=op.qualname, level=op.level,
                         shape=_shape_repr(fwd_args), dtype=str(dtype).replace("torch.", ""),
                         status=SKIPPED)

    # 1) Build both modules with identical init; share weights.
    try:
        baseline, candidate = _build_modules(
            op, baseline_cls, candidate_cls, operators, vid, device)
    except _UnsupportedInput as exc:
        res.detail = f"skip: {exc}"
        return res
    except capture.ReconstructError as exc:
        res.detail = f"skip: unreconstructable init ({exc})"
        return res
    except Exception as exc:  # noqa: BLE001 - building the module itself failed
        res.status = RUNTIME_ERROR
        res.detail = f"module construction failed: {exc!r}"
        return res

    baseline = _prepare_module(baseline, dtype, device)
    candidate = _prepare_module(candidate, dtype, device)
    try:
        candidate.load_state_dict(baseline.state_dict(), strict=False)
    except Exception:
        pass  # best-effort weight sharing (old runner does the same)

    # 2) Correctness over N rounds of fresh inputs.
    worst = None
    for r in range(rounds):
        torch.manual_seed(seed + r)
        try:
            base_call = _build_call(baseline_cls, fwd_args, device, init_args)
        except _UnsupportedInput as exc:
            res.detail = f"skip: {exc}"
            return res
        args, kwargs = base_call
        try:
            ref = _run_forward(baseline, *(_clone_tree((args, kwargs))))
        except Exception as exc:  # noqa: BLE001 - baseline rejects generated inputs
            res.detail = f"skip: baseline failed on generated inputs ({exc!r})"
            return res
        try:
            out = _run_forward(candidate, *(_clone_tree((args, kwargs))))
        except Exception as exc:  # noqa: BLE001
            res.status = RUNTIME_ERROR
            res.detail = f"candidate forward raised: {exc!r}"
            return res
        if r == 0:
            try:
                _check_integrity(integrity)
                _check_lazy_outputs(out)
            except _RewardHack as exc:
                res.status = REWARD_HACK
                res.detail = str(exc)
                return res
        status, max_abs, max_rel, matched, detail = _compare(out, ref, check_dtype=(r == 0))
        if worst is None or matched < worst[3]:
            worst = (status, max_abs, max_rel, matched, detail)
        res.max_abs_error = max(res.max_abs_error, max_abs)
        res.max_rel_error = max(res.max_rel_error, max_rel)
        if status != PASSED:
            res.status = status
            res.matched_ratio = matched
            res.detail = detail
            return res
        del ref, out
    res.matched_ratio = worst[3] if worst else 1.0

    # 3) Reward-hack guard immediately before timing.
    try:
        _check_monkey_patch()
    except _RewardHack as exc:
        res.status = REWARD_HACK
        res.detail = str(exc)
        return res

    # 4) Timing: candidate (guarded) and baseline (trusted) -> speedup.
    torch.manual_seed(seed)
    try:
        base_call = _build_call(baseline_cls, fwd_args, device, init_args)
        threads_before = threading.active_count()
        res.candidate_ms = _time_module(candidate, base_call, warmup, iters, device)
        _check_threads(threads_before, threading.active_count())
        res.baseline_ms = _time_module(baseline, base_call, warmup, iters, device)
    except _RewardHack as exc:
        res.status = REWARD_HACK
        res.detail = str(exc)
        return res
    except Exception as exc:  # noqa: BLE001
        res.status = RUNTIME_ERROR
        res.detail = f"timing failed: {exc!r}"
        return res

    if res.candidate_ms > 0:
        res.speedup = res.baseline_ms / res.candidate_ms
    res.status = PASSED
    return res


def _worker_main(args) -> int:
    device = "cuda:0"
    op = _parse_operator(args.op)
    if op is None:
        _emit({"op": args.op, "status": RUNTIME_ERROR, "detail": "bad operator name"})
        return 2

    # Harden BEFORE importing candidate code.
    integrity = _snapshot_integrity()

    try:
        baseline_cls = _import_symbol(op.qualname)
    except Exception as exc:  # noqa: BLE001
        _emit({"op": op.qualname, "level": op.level, "shape": "-", "dtype": "-",
               "status": RUNTIME_ERROR, "detail": f"cannot import baseline: {exc!r}"})
        return 1
    if args.self_test:
        candidate_cls = baseline_cls
    else:
        candidate_cls = _load_candidate_class(op)
        if candidate_cls is None:
            _emit({"op": op.qualname, "level": op.level, "shape": "-", "dtype": "-",
                   "status": SKIPPED, "detail": "no candidate class found"})
            return 0
    # Re-check integrity after importing candidate/baseline modules: catches an
    # eval-function or timing-primitive patch installed at candidate import time,
    # regardless of whether any case reaches the timing stage.
    try:
        _check_integrity(integrity)
        _check_monkey_patch()
    except _RewardHack as exc:
        _emit({"op": op.qualname, "level": op.level, "shape": "-", "dtype": "-",
               "status": REWARD_HACK, "detail": str(exc)})
        return 1

    reports = _load_reports(Path(args.captures))
    cases = _collect_cases(op, reports, args.max_shapes)
    print(f"[worker] {op.qualname}: {len(cases)} case(s)", flush=True)
    if not cases:
        _emit({"op": op.qualname, "level": op.level, "shape": "-", "dtype": "-",
               "status": SKIPPED, "detail": "no forward cases in captures"})
        return 0

    for n, case in enumerate(cases):
        print(f"[worker] case {n + 1}/{len(cases)}  shape={_shape_repr(case['fwd_args'])}",
              flush=True)
        try:
            res = _bench_one_case(op, baseline_cls, candidate_cls, case, device,
                                  args.warmup, args.iters, args.rounds, DEFAULT_SEED,
                                  integrity)
        except _RewardHack as exc:
            res = ScenarioResult(op=op.qualname, level=op.level,
                                 shape=_shape_repr(case["fwd_args"]), dtype="-",
                                 status=REWARD_HACK, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - never let one case kill the worker
            traceback.print_exc()
            res = ScenarioResult(op=op.qualname, level=op.level,
                                 shape=_shape_repr(case["fwd_args"]), dtype="-",
                                 status=RUNTIME_ERROR, detail=f"{exc!r}")
        _emit(res.to_dict())
        # Reclaim between cases.
        torch.cuda.empty_cache()
    return 0


# ###########################################################################
# PART 6  ·  PARENT SCHEDULER -- GPU-pool packing, watchdog, clock locking
# ###########################################################################

# ---------------------------------------------------------------------------
# Parent: GPU-pool scheduler (mirrors capture.py, simplified to one GPU per op).
# ---------------------------------------------------------------------------
def _detect_gpu_ids(explicit: str | None) -> list[str]:
    """Physical GPU ids: --gpus > CUDA_VISIBLE_DEVICES > nvidia-smi > torch."""
    if explicit:
        return [t.strip() for t in explicit.split(",") if t.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            out = subprocess.run([nvsmi, "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
            ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass
    try:
        n = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n = 0
    return [str(i) for i in range(n)]


def _kill_group(proc: subprocess.Popen, pgid: int) -> None:
    for sig, grace in ((signal.SIGTERM, _TERM_GRACE_SEC), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)


def _worker_command(op: Operator, args) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "fastkernels.bench_kernel", "--worker",
           "--op", op.qualname, "--captures", str(args.captures),
           "--max-shapes", str(args.max_shapes), "--warmup", str(args.warmup),
           "--iters", str(args.iters), "--rounds", str(args.rounds)]
    if args.self_test:
        cmd.append("--self-test")
    return cmd


def _read_scenarios(jsonl_path: Path, op: Operator) -> list[ScenarioResult]:
    out: list[ScenarioResult] = []
    try:
        text = jsonl_path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        d.setdefault("level", op.level)
        # Keep only ScenarioResult fields.
        fields = {f.name for f in dataclasses.fields(ScenarioResult)}
        out.append(ScenarioResult(**{k: v for k, v in d.items() if k in fields}))
    return out


def _run_parallel(ops: list[Operator], args, gpu_ids: list[str], mode: str) -> BenchResult:
    log_dir = RESULTS_DIR / "kernel_bench_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(ops)
    free = list(gpu_ids)
    running: dict = {}
    collected: dict[str, list[ScenarioResult]] = {}

    def _launch(op: Operator) -> None:
        gpu = free.pop(0)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["FK_BENCH_WORKER"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        stem = f"{op.level}_{op.stem}_{op.class_name}"
        jsonl_path = log_dir / f"{stem}.jsonl"
        log_path = log_dir / f"{stem}.log"
        jf = open(jsonl_path, "w")
        lf = open(log_path, "w")
        proc = subprocess.Popen(_worker_command(op, args), stdout=jf, stderr=lf,
                                env=env, start_new_session=True)
        running[proc] = {"op": op, "gpu": gpu, "jsonl": jsonl_path, "log": log_path,
                         "jf": jf, "lf": lf, "pgid": proc.pid, "start": time.monotonic()}
        print(f"  -> [GPU {gpu}] {op.qualname} started (log {log_path})")

    def _watchdog() -> None:
        now, wall = time.monotonic(), time.time()
        for proc, info in list(running.items()):
            if proc.poll() is not None:
                continue
            reason = None
            if now - info["start"] > _EVAL_TIMEOUT_SEC:
                reason = f"exceeded {_EVAL_TIMEOUT_SEC}s wall-clock cap"
            else:
                try:
                    idle = wall - os.stat(info["log"]).st_mtime
                    if idle > _STALL_SEC:
                        reason = f"no output for {int(idle)}s (>{_STALL_SEC}s)"
                except OSError:
                    pass
            if reason:
                info["killed_reason"] = reason
                print(f"  !! WATCHDOG {info['op'].qualname}: {reason}; killing")
                _kill_group(proc, info["pgid"])

    try:
        while pending or running:
            while pending and free:
                _launch(pending.pop(0))
            # Wait for at least one child to finish (with watchdog).
            last = time.monotonic()
            while True:
                done = [p for p in running if p.poll() is not None]
                if done:
                    break
                if time.monotonic() - last >= _WATCHDOG_INTERVAL_SEC:
                    last = time.monotonic()
                    _watchdog()
                time.sleep(0.5)
            for proc in done:
                info = running.pop(proc)
                info["jf"].close()
                info["lf"].close()
                free.append(info["gpu"])
                op = info["op"]
                scenarios = _read_scenarios(info["jsonl"], op)
                rc = proc.returncode
                if info.get("killed_reason"):
                    scenarios.append(ScenarioResult(
                        op=op.qualname, level=op.level, shape="-", dtype="-",
                        status=RUNTIME_ERROR, detail=info["killed_reason"]))
                elif rc != 0 and not scenarios:
                    scenarios.append(ScenarioResult(
                        op=op.qualname, level=op.level, shape="-", dtype="-",
                        status=RUNTIME_ERROR,
                        detail=f"worker exited rc={rc}; see {info['log']}"))
                collected[op.qualname] = scenarios
                np = sum(1 for s in scenarios if s.status == PASSED)
                nf = sum(1 for s in scenarios if s.status in _FAIL_STATUSES)
                print(f"  <- [GPU {info['gpu']}] {op.qualname} done "
                      f"(rc={rc}; {np} passed, {nf} failed); freed GPU {info['gpu']}")
    finally:
        for proc, info in list(running.items()):
            _kill_group(proc, info["pgid"])
            info["jf"].close()
            info["lf"].close()

    return BenchResult(
        mode=mode,
        scenarios=[s for op in ops for s in collected.get(op.qualname, [])])


# ---------------------------------------------------------------------------
# Optional GPU clock locking (best-effort; needs passwordless sudo nvidia-smi).
# ---------------------------------------------------------------------------
def _lock_clocks() -> bool:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return False
    try:
        name = subprocess.run([nvsmi, "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.splitlines()
        name = name[0].strip() if name else ""
    except Exception:  # noqa: BLE001
        return False
    preset = next((v for k, v in _CLOCK_PRESETS.items() if k in name), None)
    if preset is None:
        print(f"  (no clock preset for '{name}'; skipping clock lock)")
        return False
    gpu_mhz, dram_mhz = preset
    try:
        subprocess.run(["sudo", "-n", nvsmi, "-lgc", str(gpu_mhz)], check=True,
                       capture_output=True, timeout=30)
        subprocess.run(["sudo", "-n", nvsmi, "-lmc", str(dram_mhz)], check=True,
                       capture_output=True, timeout=30)
        print(f"  locked clocks: graphics={gpu_mhz}MHz memory={dram_mhz}MHz")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not lock clocks: {exc}; continuing unlocked)")
        return False


def _unlock_clocks() -> None:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return
    for flag in ("-rgc", "-rmc"):
        try:
            subprocess.run(["sudo", "-n", nvsmi, flag], check=False,
                           capture_output=True, timeout=30)
        except Exception:  # noqa: BLE001
            pass


# ###########################################################################
# PART 7  ·  CLI -- argument parsing and the ``main`` entry point
# ###########################################################################

# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fastkernels bench",
        description="Benchmark candidate kernels against their baseline for "
                    "correctness and performance, using captured shapes/dtypes.")
    p.add_argument("--target", default=None,
                   help="Only this operator (module stem or class name).")
    p.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=None,
                   help="Only operators at this level.")
    p.add_argument("--self-test", action="store_true",
                   help="Benchmark each baseline against itself (identity check).")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids to use (default: all visible).")
    p.add_argument("--max-shapes", type=int, default=DEFAULT_MAX_SHAPES,
                   help=f"Max captured shapes per operator (default {DEFAULT_MAX_SHAPES}).")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="Correctness rounds with fresh inputs (default 3).")
    p.add_argument("--captures", default=str(capture.CAPTURE_DIR),
                   help="Directory of capture reports (default ~/.fastkernels/captures).")
    p.add_argument("--lock-clocks", action="store_true",
                   help="Lock GPU clocks for stable timing (needs sudo nvidia-smi).")
    p.add_argument("--output", default=None, help="Path for the JSON results file.")
    p.add_argument("--json", action="store_true", help="Print results as JSON to stdout.")
    p.add_argument("--list", action="store_true",
                   help="List the operators that would be benchmarked and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the full execution plan (operators, selected "
                        "shapes/dtypes and config) without running anything.")
    p.add_argument("-v", "--verbose", action="store_true")
    # Worker-only (internal) flags.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--op", default=None, help=argparse.SUPPRESS)
    return p


def _resolve_operators(args) -> list[Operator]:
    captures_dir = Path(args.captures)
    reports = _load_reports(captures_dir)
    ops = list(_discover_operators(reports).values())
    if args.level is not None:
        ops = [o for o in ops if o.level == args.level]
    if args.target is not None:
        ops = [o for o in ops if o.stem == args.target or o.class_name == args.target]
    if not args.self_test:
        ops = [o for o in ops if _candidate_file(o).exists()]
    ops.sort(key=lambda o: (o.level, o.stem))
    return ops


def _print_plan(ops: list[Operator], args, gpu_ids: list[str], mode: str) -> int:
    """Dry run: show what would be benchmarked -- operators, the exact
    shapes/dtypes/init-variants selected from captures, and the run config --
    without building any module, spawning a worker or touching a GPU."""
    reports = _load_reports(Path(args.captures))
    gpu_disp = ",".join(gpu_ids) if gpu_ids else "none detected"
    print("DRY RUN -- no modules are built and no kernels are executed.")
    print(f"Mode: {mode}  |  GPUs: {gpu_disp} (one operator per GPU)")
    print(f"Config: warmup={args.warmup} iters={args.iters} rounds={args.rounds} "
          f"max-shapes={args.max_shapes}")
    print(f"Captures: {args.captures} ({len(reports)} report(s))\n")
    total_cases = 0
    for op in ops:
        cases = _collect_cases(op, reports, args.max_shapes)
        total_cases += len(cases)
        if args.self_test:
            tag = "self-test (baseline vs itself)"
        else:
            try:
                tag = f"candidate: {_candidate_file(op).relative_to(CANDIDATE_DIR.parent)}"
            except ValueError:
                tag = f"candidate: {_candidate_file(op)}"
        print(f"L{op.level}  {op.stem}  ({op.class_name})  [{tag}]")
        if not cases:
            print("      (no forward cases in captures)")
        for i, c in enumerate(cases, 1):
            dt = str(_case_dtype(c["fwd_args"])).replace("torch.", "")
            vtag = f"init#{c['vid']}" if c["vid"] is not None else "no-init"
            print(f"      #{i}  count={c['count']:<6} {dt:<9} {vtag:<8} "
                  f"{_shape_repr(c['fwd_args'])}")
    print(f"\nPlan: {len(ops)} operator(s), {total_cases} case(s) total. "
          "(dry run -- nothing executed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.worker:
        if not args.op:
            print("--worker requires --op", file=sys.stderr)
            return 2
        return _worker_main(args)

    ops = _resolve_operators(args)
    mode = "self-test" if args.self_test else "candidate"

    if args.list:
        print(f"Operators to benchmark ({mode} mode): {len(ops)}")
        for o in ops:
            tag = "" if args.self_test else "  [candidate present]"
            print(f"  L{o.level}  {o.stem:<28} {o.class_name}{tag}")
        if not ops and not args.self_test:
            print("  (none -- add kernels under tasks/candidate/L*/, "
                  "or use --self-test)")
        return 0

    gpu_ids = _detect_gpu_ids(args.gpus)

    if args.dry_run:
        return _print_plan(ops, args, gpu_ids, mode)

    if not ops:
        if args.self_test:
            print(f"No operators found in captures at {args.captures}. "
                  "Run `fastkernels capture` first.")
        else:
            print("No candidates to benchmark. Add kernels under "
                  "tasks/candidate/L*/ (matching a baseline operator name), "
                  "or run with --self-test.")
        return 0

    if not gpu_ids:
        print("No GPUs available.", file=sys.stderr)
        return 2
    print(f"Benchmarking {len(ops)} operator(s) in {mode} mode on GPU(s) "
          f"{','.join(gpu_ids)} (one operator per GPU).")

    locked = _lock_clocks() if args.lock_clocks else False
    try:
        result = _run_parallel(ops, args, gpu_ids, mode)
    finally:
        if locked:
            _unlock_clocks()

    result.print_table()
    out_path = Path(args.output) if args.output else run_output_path("bench_kernel")
    result.save_json(out_path)
    print(f"\nResults written to {out_path}")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
