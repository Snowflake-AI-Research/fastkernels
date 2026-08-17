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
  case whose generated inputs the baseline itself rejects, or whose baseline
  output is all-zeros (untestable — a stub would match), is also ``SKIPPED``.
* One GPU per operator (tp=1) -- matches the L1/L2 kernel scope; multi-GPU
  operators would need the tp-packing logic from ``capture.py``.

This module is ``fastkernels/bench.py`` (invoked as ``fastkernels bench`` /
``python -m fastkernels.bench``); importing it has no side effects.

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
import gc
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
    # Recursive: capture writes each run into a per-run/per-scenario subfolder
    # (``<run>/<NN_model_tpN_dtype>/report*.json``); flat top-level reports from
    # older runs are still matched.
    for path in sorted(captures_dir.rglob("*.json")):
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


def _case_size(case: dict) -> int:
    """Total captured input elements of a case (its problem size)."""
    total = 0
    for spec in _iter_tensor_specs(case["fwd_args"]):
        n = 1
        for s in spec["shape"]:
            n *= int(s)
        total += n
    return total


def _select_spread(cases: list[dict], max_shapes: int) -> list[dict]:
    """Pick a representative spread of at most *max_shapes* cases: the smallest,
    largest and median by problem size, then fill the rest with the hottest
    (highest ``count``). Covers the size distribution instead of only the hot
    decode shape, which matters for roofline-style operator benchmarking."""
    if len(cases) <= max_shapes:
        return sorted(cases, key=lambda c: -c["count"])
    by_size = sorted(cases, key=_case_size)
    priority = [0, len(by_size) - 1, len(by_size) // 2]  # min, max, median
    priority += sorted(range(len(by_size)), key=lambda i: -by_size[i]["count"])
    picked, seen = [], set()
    for i in priority:
        if i in seen:
            continue
        seen.add(i)
        picked.append(by_size[i])
        if len(picked) >= max_shapes:
            break
    return picked


def _collect_cases(op: Operator, reports: list[dict], max_shapes: int) -> list[dict]:
    """Select up to *max_shapes* distinct (init, forward) cases for *op*.

    Each case pairs a captured ``forward`` variant with the ``__init__`` variant
    that built the instance it ran on (``init_variant_ids``). Cases are
    deduplicated across reports by (init args, forward args), then a
    representative spread by problem size (smallest / largest / median) plus the
    hottest shapes is selected (see :func:`_select_spread`).
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
    return _select_spread(cases, max_shapes)


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


def _resolve_opaque_ops(recipe, operators: dict):
    """Rewrite ``$opaque`` nodes that name a constructible fastkernels operator
    into ``$op_ref`` so reconstruction can build them.

    Capture emits ``$opaque`` for a pre-built operator instance passed into
    another op's ``__init__`` when that instance carries no init tag -- e.g. a
    stateless ``GELU`` ``act_fn`` handed to ``VisionMLP``/``VisionBlock``. When
    the named class is present in the report with an init variant, it *is*
    reconstructable, so point an ``$op_ref`` at its first variant. Opaque nodes
    that don't resolve are left as-is (the case still SKIPs)."""
    if isinstance(recipe, list):
        return [_resolve_opaque_ops(x, operators) for x in recipe]
    if isinstance(recipe, dict):
        if "$opaque" in recipe:
            qual = recipe["$opaque"]
            entry = operators.get(qual) if isinstance(qual, str) else None
            if entry and entry.get("init") and entry["init"].get("variants"):
                return {"$op_ref": {"op": qual, "init_variant_id": 0}}
            return recipe
        return {k: _resolve_opaque_ops(v, operators) for k, v in recipe.items()}
    return recipe


def _build_modules(op: Operator, baseline_cls: type, candidate_cls: type,
                   operators: dict, vid: int | None, device: str):
    """Instantiate baseline + candidate with identical (reconstructed) init args."""
    if vid is not None:
        call = operators[op.qualname]["init"]["variants"][vid]["args"]
        call = _resolve_opaque_ops(call, operators)
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
    """Move to *device*, cast **high-precision** floating parameters to *dtype*
    (fp16/bf16/fp32 only), and switch to eval. FP8 (and other low-bit) params
    are left untouched -- casting a quantized weight to bf16 would break the
    kernel that expects it (DeepGEMM asserts the weight is float8_e4m3fn)."""
    _HIGH_PREC = (torch.float16, torch.bfloat16, torch.float32)
    module = module.to(device)
    if dtype in _HIGH_PREC:
        for p in module.parameters():
            if p.dtype in _HIGH_PREC and p.dtype != dtype:
                p.data = p.data.to(dtype)
    module.eval()
    return module


def _init_fp8_module_weights(module: nn.Module) -> None:
    """Give every FP8 linear submodule a *valid* block-scaled weight.

    Reconstructed fp8 wrapper linears (``ColumnParallelLinear`` etc.) allocate
    their ``weight``/``weight_scale_inv`` params with ``torch.empty`` and rely on
    the weight loader to fill + layout-transform them; without that their scale
    is uninitialized and in the wrong (checkpoint) layout, so the internal
    DeepGEMM GEMM produces NaN or asserts. Here we regenerate each such param
    from a reference weight -- exactly like the FP8 input builder does for
    ``Fp8Linear`` -- so the module runs. Both baseline and candidate are init'd
    (matching param shapes), then weights are shared via ``load_state_dict``.
    """
    from fastkernels.tasks.baseline.L1.fp8_linear import Fp8Linear
    from fastkernels.infra.weight_loader import _postprocess_moe_fp8_weights
    for sub in module.modules():
        lin = getattr(sub, "linear_op", None)
        w = getattr(sub, "weight", None)
        if (isinstance(lin, Fp8Linear) and isinstance(w, torch.nn.Parameter)
                and w.dtype == torch.float8_e4m3fn and w.dim() == 2):
            n, k = w.shape
            weight_fp8, weight_scale_inv = _fp8_block_quant_weight(n, k, str(w.device))
            sub.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
            sub.weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        # Block-scaled MoE experts (Qwen3MoE): raw 3D fp8 ``w13``/``w2`` params.
        # torch.empty leaves NaN fp8 bytes, and the DeepGEMM MoE path needs the
        # scale in its own layout (``w13_scale_dg``/``w2_scale_dg``) or it asserts
        # on the raw scale. Fill valid fp8 and build the DeepGEMM-layout scales.
        w13, w13s = getattr(sub, "w13", None), getattr(sub, "w13_scale", None)
        if (isinstance(w13, torch.nn.Parameter) and w13.dim() == 3
                and w13.dtype == torch.float8_e4m3fn
                and isinstance(w13s, torch.nn.Parameter)):
            with torch.no_grad():
                for name in ("w13", "w2"):
                    p = getattr(sub, name, None)
                    if (isinstance(p, torch.nn.Parameter)
                            and p.dtype == torch.float8_e4m3fn):
                        ref = torch.randn(p.shape, device=p.device,
                                          dtype=torch.float32).clamp_(-2, 2)
                        p.data.copy_(ref.to(torch.float8_e4m3fn))
                # ``_prepare_module`` downcast the float32 block scales to the run
                # dtype (bf16); DeepGEMM requires them float32, so restore.
                for name in ("w13_scale", "w2_scale"):
                    sp = getattr(sub, name, None)
                    if isinstance(sp, torch.nn.Parameter) and sp.dtype != torch.float32:
                        sp.data = sp.data.float()
            try:
                _postprocess_moe_fp8_weights(sub)
            except Exception:  # noqa: BLE001 - no DeepGEMM here; forward uses Triton
                pass


def _init_nvfp4_module_weights(module: nn.Module) -> None:
    """Give every NVFP4 MoE submodule *valid* TRTLLM-gen fp4 expert weights.

    The NVFP4 analogue of :func:`_init_fp8_module_weights`. A reconstructed
    NVFP4 MoE allocates its pre-quant expert weights with ``torch.empty`` and
    relies on the checkpoint loader to fill them + ``prepare_fp4_weights()`` to
    transform them into the kernel's block-scaled layout; fresh, those params
    hold garbage. Here we fill the high-precision weights with a small random
    reference and run the module's own ``prepare_fp4_weights()`` (exactly what
    ``weight_loader._postprocess_nvfp4_weights`` does after loading) so the op
    runs. Duck-typed (no import) and best-effort -- a GPU without the TRTLLM-gen
    kernel just leaves the case to skip. Both baseline + candidate are init'd,
    then weights are shared via ``load_state_dict``."""
    _HIGH_PREC = (torch.float16, torch.bfloat16, torch.float32)
    for sub in module.modules():
        prepare = getattr(sub, "prepare_fp4_weights", None)
        if not (callable(prepare) and getattr(sub, "use_nvfp4", False)):
            continue
        with torch.no_grad():
            for p in sub.parameters(recurse=True):
                if p.dtype in _HIGH_PREC and p.numel() > 0 and (
                        not torch.isfinite(p).all()
                        or p.detach().abs().max().item() > 1e4):
                    p.normal_(0, 0.02)
        try:
            prepare()
        except Exception:  # noqa: BLE001 - no TRTLLM-gen kernel here; case skips
            pass


def _sanitize_float_params(module: nn.Module) -> None:
    """Re-initialize any high-precision float param that is *uninitialized
    garbage* to a small normal.

    Many modules allocate weights with ``torch.empty`` and rely on a weight
    loader to fill them; reconstructed fresh, those params hold arbitrary memory
    (often non-finite or huge), which overflows to NaN once run through e.g.
    attention. We only touch params that look like garbage (non-finite or
    extreme magnitude), leaving legitimately-initialized weights (norm ones,
    embeddings) alone. Applied to baseline + candidate, then shared via
    ``load_state_dict``."""
    _HIGH_PREC = (torch.float16, torch.bfloat16, torch.float32)
    with torch.no_grad():
        for p in module.parameters():
            if p.dtype in _HIGH_PREC and p.numel() > 0:
                if not torch.isfinite(p).all() or p.detach().abs().max().item() > 1e4:
                    p.normal_(0, 0.02)


# --- DSA sparse indexer decode reconstruction --------------------------------
# The indexer's meaningful path scores queries against a paged KV cache and
# selects a top-k. To bench it standalone we synthesize a small, self-consistent
# decode state: a valid FP8 indexer cache (populated by the real store kernel so
# the packing/scale format is exact) plus decode metadata that all agree on the
# same block geometry. The cached length is > topk_tokens so the top-k actually
# selects (a real scoring test rather than "return everything").
_INDEXER_BLOCK_SIZE = 64


def _indexer_ctx_len(topk_tokens: int) -> int:
    """Per-request cached KV length for the indexer decode bench: > topk (so the
    top-k genuinely selects), rounded up to a whole number of blocks."""
    ctx = 2 * max(1, int(topk_tokens))
    return -(-ctx // _INDEXER_BLOCK_SIZE) * _INDEXER_BLOCK_SIZE


def _prep_sparse_attn_indexer(module: nn.Module, init_args: dict, device: str) -> None:
    """Populate ``SparseAttnIndexer`` module state its decode forward needs but
    that is neither a constructor arg nor a captured weight: a valid FP8 indexer
    K cache, plus flags that make the top-k comparison deterministic.

    The cache is filled with the real ``IndexerKCacheStore`` kernel (so the fp8
    key + UE8M0 scale packing is byte-exact) from a fixed seed, so baseline and
    candidate get an identical cache (else their logits -- hence top-k -- would
    differ). ``_sort_topk`` forces the canonical ascending index order (the
    kernels are otherwise order-nondeterministic under real sparsity), and
    ``max_model_len`` is shrunk to bound the decode logits buffer width."""
    from fastkernels.tasks.baseline.L1.indexer_k_cache import IndexerKCacheStore

    try:
        topk = int(init_args["topk_tokens"])
        head_dim = int(init_args["head_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _UnsupportedInput(f"SparseAttnIndexer prep: init dims ({exc!r})")

    ctx_len = _indexer_ctx_len(topk)
    max_blocks = ctx_len // _INDEXER_BLOCK_SIZE
    total_slots = max_blocks * _INDEXER_BLOCK_SIZE
    cache = torch.zeros(max_blocks, _INDEXER_BLOCK_SIZE, head_dim + 4,
                        dtype=torch.uint8, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    keys = torch.randn(total_slots, head_dim, generator=gen, device=device,
                       dtype=torch.float32).clamp_(-2, 2).to(torch.bfloat16)
    slots = torch.arange(total_slots, device=device, dtype=torch.int64)
    IndexerKCacheStore()(keys, cache, slots)

    module.indexer_k_cache = cache
    module._sort_topk = True
    module.max_model_len = ctx_len


def _prep_process_weights(module: nn.Module, init_args: dict, device: str) -> None:
    """Run every submodule's ``process_weights_after_loading``. The trtllm-gen
    BF16 MoE kernels (KimiMoE / SharedExpertMoE / GptOssMoE) read expert weights
    in a 4D BlockMajorK layout this produces; without it they index a still-3D
    weight ('Index 3 out of bounds for tensor with 3 dimensions'). The methods
    guard themselves, so this is safe on modules that need no conversion."""
    for sub in module.modules():
        fn = getattr(sub, "process_weights_after_loading", None)
        if callable(fn):
            fn()


def _prep_cast_buffers(module: nn.Module, init_args: dict, device: str) -> None:
    """Cast floating buffers to the module's (single) param dtype. Real loading
    casts the whole module to the run dtype; ``_prepare_module`` only casts
    params, leaving e.g. AlphaFold3's ``FourierEmbedding`` float32 ``w``/``b``
    buffers to clash with the bf16 Linear that consumes them (float != bf16)."""
    dtypes = {p.dtype for p in module.parameters() if p.is_floating_point()}
    if len(dtypes) != 1:
        return
    target = next(iter(dtypes))
    high_prec = (torch.float16, torch.bfloat16, torch.float32)
    for sub in module.modules():
        for name, buf in list(sub._buffers.items()):
            if (buf is not None and buf.is_floating_point()
                    and buf.dtype in high_prec and buf.dtype != target):
                sub._buffers[name] = buf.to(target)


# --- Kimi-Linear / Qwen3-Next recurrent state (KDA / GDN) --------------------
# The linear-attention layers read their conv + recurrent state from a
# ``KimiLinearStateManager`` on the global Context (plus, for GDN, the same
# object as a forward arg) rather than from their inputs. Rebuilding the real
# manager needs the full model config (an opaque ``$dataclass`` we don't get in
# a prep); instead a prep sizes just the per-layer buffers the forwards index
# from the *module*, stashes them here, and the matching builder attaches them
# to a fresh prefill Context with a single-sequence ``KimiLinearMetadata``.
_KIMI_RECURRENT_SM: object | None = None
_KIMI_STATE_SLOTS = 8


class _KimiRecurrentState:
    """Minimal duck-typed ``KimiLinearStateManager``: only the per-layer conv /
    recurrent buffers the KDA and GDN forwards index (slot 0 reserved as null,
    matching the real manager; our single sequence uses slot 1)."""

    def __init__(self, n_layers: int):
        self.q_conv_states = [None] * n_layers
        self.k_conv_states = [None] * n_layers
        self.v_conv_states = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.gdn_conv = [None] * n_layers
        self.recurrent = [None] * n_layers


def _locate_recurrent_attn(module: nn.Module):
    """Return ``("kda"|"gdn", attn_module)`` for the recurrent attention inside
    ``module`` (the module itself for the L2 ops, a submodule for the decoders),
    or ``(None, None)``."""
    from fastkernels.tasks.baseline.L2.kimi_delta_attention import KimiDeltaAttention
    from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import (
        Qwen3NextGDNAttention,
    )
    for sub in module.modules():
        if isinstance(sub, KimiDeltaAttention):
            return "kda", sub
        if isinstance(sub, Qwen3NextGDNAttention):
            return "gdn", sub
    return None, None


def _prep_kimi_recurrent(module: nn.Module, init_args: dict, device: str) -> None:
    """Build the per-layer recurrent state buffers (sized from the module) and
    stash them for the builder. Also run ``process_weights_after_loading`` so
    GDN's aliased input projection and Qwen3-Next's trtllm-gen MoE get their
    expected layouts."""
    global _KIMI_RECURRENT_SM
    kind, attn = _locate_recurrent_attn(module)
    if attn is None:
        raise _UnsupportedInput("no KDA/GDN submodule found")
    for sub in module.modules():
        fn = getattr(sub, "process_weights_after_loading", None)
        if callable(fn):
            fn()
    dt = torch.bfloat16
    for p in module.parameters():
        if p.is_floating_point() and p.dtype in (torch.float16, torch.bfloat16):
            dt = p.dtype
            break
    S = _KIMI_STATE_SLOTS
    li = attn.layer_idx
    sm = _KimiRecurrentState(li + 1)
    if kind == "kda":
        proj = attn.q_conv1d.weight.shape[0]
        k = max(attn.conv_size - 1, 1)
        for lst in (sm.q_conv_states, sm.k_conv_states, sm.v_conv_states):
            lst[li] = torch.zeros(S, k, proj, device=device, dtype=dt)
        sm.recurrent_states[li] = torch.zeros(
            S, attn.local_num_heads, attn.head_dim, attn.head_dim,
            device=device, dtype=torch.float32)
    else:  # gdn
        k = max(attn.conv_kernel_size - 1, 1)
        conv_dim = (2 * attn.local_k_heads * attn.head_k_dim
                    + attn.local_v_heads * attn.head_v_dim)
        sm.gdn_conv[li] = torch.zeros(
            S, k, conv_dim, device=device, dtype=dt).transpose(-1, -2)
        sm.recurrent[li] = torch.zeros(
            S, attn.local_v_heads, attn.head_k_dim, attn.head_v_dim,
            device=device, dtype=dt)
    _KIMI_RECURRENT_SM = sm


# qualname -> prep(module, init_args, device): set up op-specific module state
# (caches / flags) a valid forward depends on but that isn't a constructor arg
# or a captured weight. Applied to baseline AND candidate before timing.
_MODULE_PREP = {
    "fastkernels.tasks.baseline.L2.sparse_attn_indexer:SparseAttnIndexer":
        _prep_sparse_attn_indexer,
    # trtllm-gen BF16 MoE expert weights need the 4D layout conversion first.
    "fastkernels.tasks.baseline.L2.kimi_moe:KimiMoE": _prep_process_weights,
    "fastkernels.tasks.baseline.L2.shared_expert_moe:SharedExpertMoE":
        _prep_process_weights,
    "fastkernels.tasks.baseline.L3.gpt_oss_decoder:GptOssDecoderLayer":
        _prep_process_weights,
    # AlphaFold3 diffusion: match FourierEmbedding's float32 buffers to bf16.
    "fastkernels.tasks.baseline.L2.alphafold3_diffusion_conditioning:DiffusionConditioning":
        _prep_cast_buffers,
    "fastkernels.tasks.baseline.L3.alphafold3_diffusion_module:DiffusionModule":
        _prep_cast_buffers,
    "fastkernels.tasks.baseline.L3.alphafold3_diffusion_module:SampleDiffusion":
        _prep_cast_buffers,
    # Kimi-Linear / Qwen3-Next linear attention: build the recurrent conv/state
    # buffers the forwards read off the Context (paired with the builder below).
    "fastkernels.tasks.baseline.L2.kimi_delta_attention:KimiDeltaAttention":
        _prep_kimi_recurrent,
    "fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention:Qwen3NextGDNAttention":
        _prep_kimi_recurrent,
    "fastkernels.tasks.baseline.L3.kimi_linear_decoder:KimiLinearDecoderLayer":
        _prep_kimi_recurrent,
    "fastkernels.tasks.baseline.L3.qwen3_next_decoder:Qwen3NextDecoderLayer":
        _prep_kimi_recurrent,
}


# qualname -> dtype override for cases whose captured inputs mislead _case_dtype.
_FORCE_DTYPE = {
    # Ran in a float32 scenario, but its only input (t) is int64 so the heuristic
    # defaults to bf16 -- clashing with the always-fp32 sinusoidal embedding.
    "fastkernels.tasks.baseline.L2.oasis_timestep_embedder:OasisTimestepEmbedder":
        torch.float32,
}


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
def _contiguous_stride(shape: tuple) -> tuple:
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return tuple(reversed(stride))


def _materialize_tensor(name: str, shape, dtype_str: str, device: str, init_args: dict,
                        stride=None):
    dt = getattr(torch, dtype_str, None)
    if not isinstance(dt, torch.dtype):
        raise _UnsupportedInput(f"{name}: unknown dtype {dtype_str!r}")
    shape = tuple(int(s) for s in shape)
    if dt in _FP8_TYPES:
        t = torch.randn(shape, device=device, dtype=torch.float32).clamp_(-2, 2).to(dt)
    elif dt.is_floating_point:
        t = torch.randn(shape, device=device, dtype=dt)
    elif dt == torch.bool:
        t = torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    else:
        t = _materialize_int(name, shape, dt, device, init_args)
    # Honor a captured non-contiguous layout (e.g. a column-major FP8 scale):
    # allocate with the recorded strides and copy the values in.
    if stride is not None:
        stride = tuple(int(s) for s in stride)
        if stride != _contiguous_stride(shape):
            strided = torch.empty_strided(shape, stride, dtype=t.dtype, device=device)
            strided.copy_(t)
            return strided
    return t


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
    elif "expert" in lname or "topk_id" in lname:
        # Expert indices: bound by the captured expert count so routing stays
        # in range. ``slot_mapping`` / ``block_table`` are bounded against the
        # sibling cache tensor in ``_fix_structured_inputs`` instead.
        bound = int(init_args.get("num_experts")
                    or init_args.get("n_routed_experts")
                    or init_args.get("num_local_experts") or 8)
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
            return _materialize_tensor(name, v["shape"], v["dtype"], device, init_args,
                                       stride=v.get("stride"))
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
    sums to the number of tokens). Best-effort, keyed by conventional activation
    names -- ``cu_seqlens_q``/``cu_seqlens_k`` pair with q/k, a bare
    ``cu_seqlens`` pairs with whatever activation is present (``x`` for vision
    attention, ``hidden_states``/``query`` elsewhere)."""
    def _activation_for(cu_name: str):
        if cu_name.endswith("_q"):
            names = ("q", "query")
        elif cu_name.endswith("_k"):
            names = ("k", "key")
        else:
            names = ("q", "query", "x", "hidden_states", "key", "k")
        for n in names:
            t = kwargs.get(n)
            if isinstance(t, torch.Tensor) and t.dim() >= 1:
                return t
        return None

    for cn in list(kwargs):
        if cn != "cu_seqlens" and not cn.startswith("cu_seqlens"):
            continue
        cu = kwargs.get(cn)
        if not (isinstance(cu, torch.Tensor) and cu.dim() == 1 and cu.numel() >= 2):
            continue
        act = _activation_for(cn)
        if act is not None:
            kwargs[cn] = _partition_cu(act.shape[0], cu.numel() - 1, cu.device, cu.dtype)

    # General safety net for ops without a dedicated builder: bound any
    # ``slot_mapping`` / ``block_table`` against a sibling ``*cache*`` tensor so
    # the indices stay in range -- slots index the flat (num_blocks *
    # block_size) space, block tables index blocks (num_blocks). Ops with a
    # registered builder (the KV stores) never reach here.
    cache = next((t for k, t in kwargs.items()
                  if "cache" in k.lower() and isinstance(t, torch.Tensor)
                  and t.dim() >= 3), None)
    if cache is not None:
        num_blocks = int(cache.shape[0])
        num_slots = num_blocks * int(cache.shape[1])
        for k, t in kwargs.items():
            lk = k.lower()
            if not (isinstance(t, torch.Tensor) and t.numel() > 0
                    and not t.is_floating_point()):
                continue
            if "block_table" in lk:
                kwargs[k] = _valid_slot_mapping(cache, num_blocks, t)
            elif "slot" in lk:
                kwargs[k] = _valid_slot_mapping(cache, num_slots, t)

    # Bound request-id vectors to the block table's request count. The
    # convert-indices kernel indexes ``block_table[req_id]``; an out-of-range
    # req_id reads adjacent GPU memory -> nondeterministic global slots.
    bt = next((t for k, t in kwargs.items()
               if "block_table" in k.lower() and isinstance(t, torch.Tensor)
               and t.dim() == 2), None)
    if bt is not None:
        num_reqs = max(1, int(bt.shape[0]))
        for k, t in kwargs.items():
            if ("req_id" in k.lower() and isinstance(t, torch.Tensor)
                    and t.numel() > 0 and not t.is_floating_point()):
                kwargs[k] = torch.randint(0, num_reqs, tuple(t.shape),
                                          device=t.device, dtype=t.dtype)


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
# Per-operator input builders (registry).
#
# Some operators take *pre-quantized* tensors whose values, dtype AND memory
# layout are tied together by a quantization scheme -- e.g. a block-scaled FP8
# GEMM wants an fp8 weight plus a UE8M0, column-major (``stride(-2)==1``) scale.
# The generic shape/dtype materializer cannot synthesize a valid (weight, scale)
# pair, so for those ops we register a builder that generates a fresh reference
# bf16 weight and quantizes it with the *same* routine the model uses. Weights
# stay fresh-random and are shared identically by baseline + candidate (built
# once per round, cloned to each). Everything else falls through to the generic
# ``_build_call``.
# ---------------------------------------------------------------------------
def _fp8_block_quant_weight(N: int, K: int, device: str):
    """Random reference weight -> ``(weight_fp8[N,K], weight_scale_inv)`` in the
    UE8M0 column-major layout ``deep_gemm.fp8_gemm_nt`` expects. Mirrors
    ``fp8_linear.postprocess_fp8_weights`` but starting from a bf16 reference
    rather than a checkpoint's fp8 weight."""
    import deep_gemm
    from fastkernels.tasks.baseline.L1.fp8_grouped_gemm_contiguous import (
        _is_deep_gemm_e8m0_used,
    )
    from fastkernels.tasks.baseline.L1.fp8_linear import Fp8Linear

    bs = Fp8Linear.BLOCK_SIZE
    use_ue8m0 = _is_deep_gemm_e8m0_used()
    w = torch.randn(N, K, device=device, dtype=torch.float32) * 0.02
    weight_fp8, scale = deep_gemm.per_block_cast_to_fp8(w, use_ue8m0)
    weight_scale_inv = deep_gemm.transform_sf_into_required_layout(
        sf=scale.unsqueeze(0), mn=N, k=K, recipe=(1, bs, bs),
        num_groups=1, is_sfa=False, disable_ue8m0_cast=not use_ue8m0,
    ).squeeze(0)
    return weight_fp8, weight_scale_inv


def _build_fp8_linear_inputs(fwd_args, device, init_args):
    """Inputs for ``Fp8Linear.forward(input_bf16, weight_fp8, weight_scale_inv,
    bias)``. The captured shapes fix M/K/N; the fp8 weight + scale are derived
    by quantizing a reference weight (the captured weight_scale_inv shape/dtype
    is intentionally ignored -- it is a packed, layout-specific artifact)."""
    def _shape(key):
        spec = fwd_args.get(key)
        if not (isinstance(spec, dict) and isinstance(spec.get("shape"), list)):
            raise _UnsupportedInput(f"Fp8Linear: missing tensor arg {key!r}")
        return [int(s) for s in spec["shape"]]

    in_shape = _shape("input_bf16")
    N, wK = _shape("weight_fp8")
    if wK != in_shape[-1]:
        raise _UnsupportedInput(f"Fp8Linear: weight K={wK} != input K={in_shape[-1]}")
    input_bf16 = torch.randn(in_shape, device=device, dtype=torch.bfloat16)
    weight_fp8, weight_scale_inv = _fp8_block_quant_weight(N, in_shape[-1], device)
    bias = None
    bspec = fwd_args.get("bias")
    if isinstance(bspec, dict) and isinstance(bspec.get("shape"), list):
        bias = torch.randn([int(s) for s in bspec["shape"]], device=device,
                           dtype=torch.bfloat16)
    return [], {"input_bf16": input_bf16, "weight_fp8": weight_fp8,
                "weight_scale_inv": weight_scale_inv, "bias": bias}


# ---------------------------------------------------------------------------
# Index-based ops: KV-cache stores + MoE grouped GEMM.
#
# Their integer inputs are *not* free: a ``slot_mapping`` must point at a real
# paged-cache slot, and a fused-MoE launch needs mutually-consistent routing
# metadata. The generic ``randint`` materializer would index out of range
# (illegal memory) or feed the kernel garbage. We regenerate those tensors from
# the *captured shapes* (cache size, expert count) so every index is valid --
# the same idea as SOL-ExecBench's ``custom_inputs_entrypoint``.
# ---------------------------------------------------------------------------
def _materialize_all(fwd_args: dict, device: str, init_args: dict, skip=()) -> dict:
    """Materialize every forward arg by keyword, skipping the given names
    (which the caller regenerates itself)."""
    return {k: _materialize_value(k, v, device, init_args)
            for k, v in fwd_args.items() if k not in skip}


def _valid_slot_mapping(cache: torch.Tensor, num_slots: int, ref: torch.Tensor):
    """Distinct in-bounds slot indices (or ``randint`` when the mapping is
    longer than the cache), shaped/typed like the captured ``slot_mapping``."""
    n = ref.numel()
    num_slots = max(1, int(num_slots))
    if n <= num_slots:
        slots = torch.randperm(num_slots, device=cache.device)[:n]
    else:
        slots = torch.randint(0, num_slots, (n,), device=cache.device)
    return slots.to(ref.dtype).reshape(ref.shape)


def _build_store_kvcache_inputs(cache_key: str, num_slots_fn):
    """Factory for a KV-cache-store builder: materialize generically, then
    rebind ``slot_mapping`` to valid slots derived from the cache tensor's
    shape (``num_slots_fn`` encodes the NHD/HND/MLA layout)."""
    def _builder(fwd_args, device, init_args):
        kw = _materialize_all(fwd_args, device, init_args)
        cache = kw.get(cache_key)
        sm = kw.get("slot_mapping")
        if not (isinstance(cache, torch.Tensor) and isinstance(sm, torch.Tensor)):
            raise _UnsupportedInput("KV store: missing cache/slot_mapping")
        kw["slot_mapping"] = _valid_slot_mapping(cache, num_slots_fn(cache.shape), sm)
        return [], kw
    return _builder


def _fill_indexer_k_cache(cache: torch.Tensor, block_table: torch.Tensor) -> None:
    """Pack finite fp8 keys + UE8M0 scales into pages *block_table* walks.

    Random uint8 bytes decode to NaN/Inf scales; a zeroed cache makes gather
    return zeros and a stub candidate pass. Use the store kernel so the
    layout matches what gather reads.
    """
    if cache.ndim < 2 or int(cache.shape[-1]) <= 4:
        raise _UnsupportedInput("indexer cache: unexpected layout")
    try:
        from fastkernels.tasks.baseline.L1.indexer_k_cache import IndexerKCacheStore
    except Exception as exc:  # noqa: BLE001
        raise _UnsupportedInput(f"indexer cache fill: {exc!r}")
    block_size = int(cache.shape[1])
    head_dim = int(cache.shape[-1]) - 4
    pages = torch.unique(block_table.reshape(-1).to(torch.int64))
    n = int(pages.numel()) * block_size
    keys = (torch.randn(n, head_dim, device=cache.device, dtype=torch.float32)
            .clamp_(-2, 2).to(torch.bfloat16))
    slots = (pages.unsqueeze(1) * block_size
             + torch.arange(block_size, device=cache.device, dtype=torch.int64)
             ).reshape(-1)
    try:
        IndexerKCacheStore()(keys, cache, slots)
    except Exception as exc:  # noqa: BLE001
        raise _UnsupportedInput(f"indexer cache fill: {exc!r}")


def _build_indexer_k_cache_gather_inputs(fwd_args, device, init_args):
    """Inputs for ``IndexerKCacheGather.forward(kv_cache, block_table,
    cu_seq_lens, ...)``. The paged cache is uint8 ``[num_blocks, block_size,
    132]`` holding fp8 keys plus a float32 scale per token; random bytes decode
    to NaN/Inf. Pack referenced pages via the store kernel, bound
    ``block_table`` to the cache's block count, and drop the captured output
    workspace / ``total_tokens`` so they stay consistent with the generated
    cu_seq_lens."""
    kw = _materialize_all(fwd_args, device, init_args,
                          skip=("total_tokens", "out_k_fp8", "out_k_scale"))
    cache = kw.get("kv_cache")
    bt = kw.get("block_table")
    if not (isinstance(cache, torch.Tensor) and isinstance(bt, torch.Tensor)):
        raise _UnsupportedInput("IndexerKCacheGather: missing kv_cache/block_table")
    kw["block_table"] = _valid_slot_mapping(cache, int(cache.shape[0]), bt)
    _fill_indexer_k_cache(cache, kw["block_table"])
    return [], kw


def _build_moe_grouped_gemm_inputs(fwd_args, device, init_args):
    """Consistent inputs for ``MoeGroupedGemm.forward``.

    The routing metadata (``sorted_token_ids`` / ``expert_ids`` /
    ``num_tokens_post_padded``) is regenerated from a random top-k assignment
    via the model's own ``MoeAlign`` so every block maps to a valid expert and
    every token id stays ``< num_tokens * top_k``. ``A``/``B``/``C`` and the
    routing weights keep their captured shapes."""
    try:
        from fastkernels.tasks.baseline.L1.moe_align import MoeAlign
        from fastkernels.tasks.baseline.L1.moe_grouped_gemm import _get_default_config
    except Exception as exc:  # noqa: BLE001
        raise _UnsupportedInput(f"MoeGroupedGemm: {exc!r}")

    def _shape(key):
        spec = fwd_args.get(key)
        if not (isinstance(spec, dict) and isinstance(spec.get("shape"), list)):
            raise _UnsupportedInput(f"MoeGroupedGemm: missing tensor arg {key!r}")
        return [int(s) for s in spec["shape"]]

    a_shape, b_shape = _shape("A"), _shape("B")
    top_k = fwd_args.get("top_k")
    if not (isinstance(top_k, int) and top_k >= 1):
        raise _UnsupportedInput("MoeGroupedGemm: missing/invalid top_k")
    num_tokens, num_experts = a_shape[0], b_shape[0]
    meta = {"sorted_token_ids", "expert_ids", "num_tokens_post_padded", "config"}
    kw = _materialize_all(fwd_args, device, init_args, skip=meta)
    try:
        config = _get_default_config(num_tokens, N=b_shape[1])
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k),
                                 device=device, dtype=torch.int32)
        sorted_ids, expert_ids, num_padded = MoeAlign().to(device)(
            topk_ids, config["BLOCK_SIZE_M"], num_experts)
    except Exception as exc:  # noqa: BLE001
        raise _UnsupportedInput(f"MoeGroupedGemm: routing metadata ({exc!r})")
    kw.update(sorted_token_ids=sorted_ids, expert_ids=expert_ids,
              num_tokens_post_padded=num_padded, config=config)
    return [], kw


def _set_mla_prefill_context(n: int, device: str, ctx_summary: dict | None) -> None:
    """Publish a prefill inference ``Context`` for the MLA family: split the N
    tokens into varlen segments (across the captured bucketed prefill seq count
    when available, else one) and set ``cu_seqlens``. ``block_tables`` /
    ``slot_mapping`` are left unset so the sparse/decode branches -- and the
    indexer's paged-cache gather -- are skipped: the reconstructable dense-prefill
    path (``MLAAttention._forward_mha``)."""
    import itertools
    from fastkernels.infra.context import set_context
    ctx = ctx_summary or {}
    s = int(ctx.get("pf_seqs") or 0) if ctx.get("prefill") else 0
    s = max(1, min(s or 1, n))
    base, rem = divmod(n, s)
    sizes = [base + (1 if i < rem else 0) for i in range(s)]
    cu = torch.tensor([0, *itertools.accumulate(sizes)], device=device,
                      dtype=torch.int32)
    set_context(is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max(sizes), max_seqlen_k=max(sizes))


def _build_mla_attention_inputs(fwd_args, device, init_args):
    """Inputs for ``MLAAttention.forward`` driving its reconstructable *prefill*
    path (``_forward_mha``).

    The captured path is sparse decode, which needs a populated paged KV cache
    (block tables, slot mapping, valid top-k into cached tokens) we do not
    capture. Its dense-prefill sibling needs none of that: with the module's
    default empty ``k_cache`` and a prefill ``Context``, the decode/sparse
    branches (guarded by ``kv_cache.ndim >= 2``) are skipped and forward runs the
    ``kv_b_proj`` up-projection + varlen attention over the new tokens. We wire a
    fresh bf16 ``kv_b_proj`` (the projection the parent ``DeepSeekMLAAttention``
    shares into ``_kv_b_proj``); ``cu_seqlens`` come from the shared prefill-ctx
    helper (split across the captured bucketed prefill seq count in ``_ctx``)."""
    from fastkernels.tasks.baseline.L2.parallel_linear import ColumnParallelLinear

    kw = _materialize_all(fwd_args, device, init_args,
                          skip=("kv_b_proj", "topk_indices", "output_shape", "_ctx"))
    q, kv_c = kw.get("q"), kw.get("kv_c_normed")
    if not (isinstance(q, torch.Tensor) and isinstance(kv_c, torch.Tensor)):
        raise _UnsupportedInput("MLAAttention: missing q/kv_c_normed")
    try:
        kv_lora = int(init_args["kv_lora_rank"])
        out = int(init_args["num_heads"]) * (
            int(init_args["qk_nope_head_dim"]) + int(init_args["v_head_dim"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise _UnsupportedInput(f"MLAAttention: init dims ({exc!r})")

    # bf16 kv_b_proj (local per-rank width matches the captured num_heads); a
    # forward arg, not a submodule, so its weights are filled here rather than by
    # the module-level sanitizers. Shared by baseline + candidate (not cloned).
    kvbp = ColumnParallelLinear(kv_lora, out, bias=False, quant_config=None)
    with torch.no_grad():
        kvbp.weight.normal_(0, 0.02)
    kw["kv_b_proj"] = kvbp.to(device=device, dtype=kv_c.dtype)

    _set_mla_prefill_context(int(q.shape[0]), device, fwd_args.get("_ctx"))
    return [], kw


def _build_sparse_attn_indexer_inputs(fwd_args, device, init_args):
    """Inputs for ``SparseAttnIndexer.forward`` driving its decode path
    (``_decode_topk``) against the cache built by ``_prep_sparse_attn_indexer``.

    Wires a fresh ``rope_emb`` (indexer RoPE; ``_wk_wp_fused`` stays ``None`` so
    the separate ``wk``/``weights_proj`` run -- no ``compute_absorbed_weights``
    needed), and sets a decode ``Context`` whose ``block_tables`` /
    ``context_lens`` match the prep cache's block geometry: every request shares
    blocks ``[0, max_blocks)`` and reads ``ctx_len`` cached tokens. ``positions``
    are re-drawn within the RoPE cache."""
    from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YarnRotaryEmbedding
    from fastkernels.infra.context import set_context

    kw = _materialize_all(fwd_args, device, init_args, skip=("rope_emb", "positions"))
    hs = kw.get("hidden_states")
    if not isinstance(hs, torch.Tensor):
        raise _UnsupportedInput("SparseAttnIndexer: missing hidden_states")
    m = int(hs.shape[0])
    try:
        rope_dim = int(init_args["rope_dim"])
        topk = int(init_args["topk_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _UnsupportedInput(f"SparseAttnIndexer: init dims ({exc!r})")

    max_pos = 8192
    rope = YarnRotaryEmbedding(head_dim=rope_dim, max_position_embeddings=max_pos,
                               rope_theta=10000.0, scaling_factor=1.0,
                               is_plain=True, cache_dtype=hs.dtype).to(device)
    kw["rope_emb"] = rope
    kw["positions"] = torch.randint(0, max_pos, (m,), device=device, dtype=torch.int64)

    # Decode Context consistent with _prep_sparse_attn_indexer's cache geometry.
    ctx_len = _indexer_ctx_len(topk)
    max_blocks = ctx_len // _INDEXER_BLOCK_SIZE
    block_tables = torch.arange(max_blocks, device=device, dtype=torch.int32) \
        .view(1, max_blocks).expand(m, max_blocks).contiguous()
    context_lens = torch.full((m,), ctx_len, device=device, dtype=torch.int32)
    set_context(is_prefill=False, context_lens=context_lens,
                block_tables=block_tables, max_context_len=ctx_len)
    return [], kw


def _build_deepseek_mla_composite_inputs(fwd_args, device, init_args):
    """Inputs for the DeepSeek MLA composites -- ``DeepSeekMLAAttention`` and the
    ``DeepSeekDecoderLayer`` wrapping it -- driving the reconstructable prefill
    path end to end.

    Construction already wires the submodules (the parent ``__init__`` sets
    ``self.attn._kv_b_proj`` and ``self.indexer._rope_emb``), so the only missing
    piece is a prefill ``Context``. With one, ``self.attn`` runs ``_forward_mha``
    over the new tokens (no absorbed weights or paged cache) and the indexer runs
    its projection+RoPE+quant front half then early-returns (``block_tables``
    unset), while ``o_proj``/``down_proj`` skip their allreduce at ``tp=1``. All
    args (incl. ``positions``, ``residual``) come from the generic materializer;
    weights are filled by the module-level bf16/fp8/nvfp4 sanitizers."""
    kw = _materialize_all(fwd_args, device, init_args)
    hs = kw.get("hidden_states")
    if not isinstance(hs, torch.Tensor):
        raise _UnsupportedInput("DeepSeek MLA composite: missing hidden_states")
    _set_mla_prefill_context(int(hs.shape[0]), device, fwd_args.get("_ctx"))
    return [], kw


def _fix_paged_decode(kw: dict, cache_keys: tuple[str, ...]) -> None:
    """Make a paged-attention decode's inputs finite and in-range, in place.

    A random ``block_table`` / ``cache_seqlens`` walks pages that don't exist
    -> NaN. Draw page ids within the cache's block count and cap context
    lengths to what the page table can address (``max_pages * page_size``).
    Leave the cache values from the generic materializer (randn / clamped
    fp8) — zeroing them made the baseline output zeros, so a stub passed.
    Geometry comes from the first cache: ``[num_blocks, ..., page_size,
    head_dim]``."""
    caches = [kw.get(k) for k in cache_keys]
    bt, sl = kw.get("block_table"), kw.get("cache_seqlens")
    if not (all(isinstance(c, torch.Tensor) for c in caches)
            and isinstance(bt, torch.Tensor) and isinstance(sl, torch.Tensor)):
        raise _UnsupportedInput("paged decode: missing cache/block_table/cache_seqlens")
    c0 = caches[0]
    num_blocks, page_size = int(c0.shape[0]), int(c0.shape[-2])
    max_pages = int(bt.shape[-1])
    kw["block_table"] = torch.randint(0, max(1, num_blocks), tuple(bt.shape),
                                      device=c0.device, dtype=bt.dtype)
    max_len = max(1, max_pages * page_size)
    sl = torch.randint(1, max_len + 1, tuple(sl.shape), device=c0.device, dtype=sl.dtype)
    kw["cache_seqlens"] = sl
    if "max_seq_len" in kw:
        kw["max_seq_len"] = int(sl.max().item())


def _build_trtllm_decode_inputs(fwd_args, device, init_args):
    """Inputs for ``TRTLLMDecode.forward`` with a finite paged K/V cache
    and consistent block_table / cache_seqlens (see ``_fix_paged_decode``)."""
    kw = _materialize_all(fwd_args, device, init_args)
    _fix_paged_decode(kw, ("k_cache", "v_cache"))
    return [], kw


def _build_flashinfer_mla_decode_inputs(fwd_args, device, init_args):
    """Inputs for ``FlashInferMLADecode.forward`` with a finite paged
    latent cache and consistent block_table / cache_seqlens."""
    kw = _materialize_all(fwd_args, device, init_args)
    _fix_paged_decode(kw, ("kv_cache",))
    return [], kw


def _build_chunk_gla_inputs(fwd_args, device, init_args):
    """Inputs for ``ChunkGLA.forward``. ``g`` is a log-space forget gate: a
    random (possibly positive) gate makes the chunk scan's ``exp(cumsum(g))``
    overflow to NaN, so squash it to <= 0 with logsigmoid."""
    kw = _materialize_all(fwd_args, device, init_args)
    g = kw.get("g")
    if isinstance(g, torch.Tensor):
        kw["g"] = torch.nn.functional.logsigmoid(g.float()).to(g.dtype)
    return [], kw


def _build_attention_prefill_inputs(fwd_args, device, init_args):
    """Drive the reconstructable dense-prefill path for the attention family
    (``Attention`` / ``LlamaAttention``) and the decoder layers wrapping it
    (Llama / Qwen3-MoE / GPT-OSS).

    Their forward reads paged KV-cache metadata from the global inference
    ``Context``; with none set they deref ``None`` ('NoneType has no attribute
    contiguous'). Publish a prefill ``Context`` (the module's default empty KV
    cache -> the dense branch, block_tables None) so attention runs over the new
    tokens. A rotary passed as a *forward* arg is an opaque captured module we
    can't rebuild, so pass ``None``: the op then uses its own (``$op_ref``-
    reconstructed) init rotary, or skips RoPE when the rotary is model-level
    (GPT-OSS) -- fine for a self-consistent baseline-vs-baseline check."""
    kw = _materialize_all(fwd_args, device, init_args, skip=("rotary_emb",))
    n = None
    for key in ("hidden_states", "query"):
        t = kw.get(key)
        if isinstance(t, torch.Tensor):
            n = int(t.shape[0])
            break
    if n is None:
        raise _UnsupportedInput("attention prefill: no hidden_states/query")
    if "rotary_emb" in fwd_args:
        kw["rotary_emb"] = None
    _set_mla_prefill_context(n, device, fwd_args.get("_ctx"))
    return [], kw


def _build_recurrent_prefill_inputs(fwd_args, device, init_args):
    """GLA / GLADecoderLayer (linear-attention recurrent state): drop the opaque
    captured ``RecurrentCache`` so forward runs its cacheless prefill path
    (``initial_state=None``, no cache write). The packed ``cu_seqlens`` lives
    under the captured ``**kwargs`` catch-all, so ``module(**kw)`` re-nests it
    out of reach -- the whole [1, T] row is treated as one dense sequence, which
    is a valid GLA forward for a self-consistent baseline-vs-baseline check."""
    kw = _materialize_all(fwd_args, device, init_args, skip=("past_key_values",))
    return [], kw


def _set_kimi_prefill_context(n: int, sm, device: str) -> None:
    """Publish a single-sequence prefill Context for the Kimi-Linear / Qwen3-Next
    recurrent layers: a fresh state manager (``sm``) plus a ``KimiLinearMetadata``
    for one prompt of ``n`` tokens on slot 1. ``has_initial_state`` is all-False
    so both the conv and the recurrent state start from zero on every call --
    which keeps baseline and candidate deterministic even though they share the
    (stateful) manager instance."""
    from fastkernels.infra.context import (
        KimiLinearMetadata, get_context, set_context,
    )
    from fastkernels.infra.mamba_state import compute_causal_conv1d_metadata
    cu = torch.tensor([0, n], device=device, dtype=torch.int32)
    state_idx = torch.tensor([1], device=device, dtype=torch.int32)
    md = KimiLinearMetadata(
        num_actual_tokens=n,
        query_start_loc=cu, max_query_len=n,
        seq_lens=torch.tensor([n], device=device, dtype=torch.int32), max_seq_len=n,
        state_indices=state_idx,
        num_prefills=1, num_prefill_tokens=n, num_decodes=0, num_decode_tokens=0,
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
        all_have_initial_state=False, any_have_initial_state=False,
        query_start_loc_int32=cu, state_indices_long=state_idx.long(),
    )
    md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = (
        compute_causal_conv1d_metadata(cu)
    )
    set_context(is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=n, max_seqlen_k=n)
    ctx = get_context()
    ctx.kda_state = sm
    ctx.kda_metadata = md


def _build_kimi_recurrent_inputs(fwd_args, device, init_args):
    """Inputs for the Kimi-Linear KDA and Qwen3-Next GDN linear-attention ops (and
    the decoder layers wrapping them). Drop the opaque captured state manager and
    rotary module, then attach the prep-built recurrent state + a single-sequence
    prefill Context. Captured ``positions`` is materialized but unused on the
    linear path."""
    sm = _KIMI_RECURRENT_SM
    if sm is None:
        raise _UnsupportedInput("kimi recurrent state not prepared")
    kw = _materialize_all(fwd_args, device, init_args,
                          skip=("state_manager", "rotary_emb"))
    hs = kw.get("hidden_states")
    if not isinstance(hs, torch.Tensor):
        raise _UnsupportedInput("kimi recurrent: missing hidden_states")
    _set_kimi_prefill_context(int(hs.shape[0]), sm, device)
    if "rotary_emb" in fwd_args:
        kw["rotary_emb"] = None
    kw["state_manager"] = sm
    return [], kw


def _build_row_parallel_inputs(fwd_args, device, init_args):
    """RowParallelLinear row-shards its input to ``input_size // tp``, but the
    captured activation is the full (pre-shard) width. Regenerate ``x`` at the
    runtime per-rank width so it matches the sharded weight the module built."""
    from fastkernels.infra.tp import _tp_size
    kw = _materialize_all(fwd_args, device, init_args)
    x = kw.get("x")
    isize = init_args.get("input_size")
    if isinstance(x, torch.Tensor) and isize:
        local = int(isize) // max(1, _tp_size())
        if x.shape[-1] != local:
            kw["x"] = torch.randn(*x.shape[:-1], local, device=device, dtype=x.dtype)
    return [], kw


def _build_gather_dequant_mla_inputs(fwd_args, device, init_args):
    """Consistent chunked-context gather metadata for GatherAndDequantKVCacheMLA:
    partition ``total_tokens`` across ``num_seqs`` (block_table rows) with a
    monotone ``cu_seq_lens``, matching ``token_to_seq``/``workspace_starts``, and
    an in-range ``block_table``. The captured random metadata walks unallocated
    pages -> illegal memory access."""
    import itertools
    kw = _materialize_all(fwd_args, device, init_args)
    kv, bt, tts = kw.get("kv_cache"), kw.get("block_table"), kw.get("token_to_seq")
    if not (isinstance(kv, torch.Tensor) and isinstance(bt, torch.Tensor)
            and isinstance(tts, torch.Tensor)):
        raise _UnsupportedInput("gather-dequant MLA: missing cache/block_table/token_to_seq")
    num_blocks, num_seqs, total = int(kv.shape[0]), int(bt.shape[0]), int(tts.shape[0])
    base, rem = divmod(total, max(1, num_seqs))
    lens = [base + (1 if i < rem else 0) for i in range(num_seqs)]
    cu = [0, *itertools.accumulate(lens)]
    kw["cu_seq_lens"] = torch.tensor(cu, device=device, dtype=torch.int32)
    kw["token_to_seq"] = torch.repeat_interleave(
        torch.arange(num_seqs, device=device),
        torch.tensor(lens, device=device)).to(torch.int32)
    kw["workspace_starts"] = torch.tensor(cu[:-1], device=device, dtype=torch.int32)
    kw["block_table"] = torch.randint(0, num_blocks, tuple(bt.shape),
                                      device=device, dtype=bt.dtype)
    kw["total_tokens"] = total
    return [], kw


def _build_kimi_mla_inputs(fwd_args, device, init_args):
    """KimiMLAAttention: its forward ignores ``positions``/``state_manager`` and
    delegates to an ``MLAAttention`` (``self.attn``, ``_kv_b_proj`` already wired
    in ``__init__``). With a prefill ``Context`` and the module's default empty
    KV cache it runs the dense-prefill MHA path (no paged cache / absorbed
    weights), exactly like the standalone ``MLAAttention`` builder."""
    kw = _materialize_all(fwd_args, device, init_args,
                          skip=("state_manager", "positions", "_ctx"))
    hs = kw.get("hidden_states")
    if not isinstance(hs, torch.Tensor):
        raise _UnsupportedInput("KimiMLAAttention: missing hidden_states")
    _set_mla_prefill_context(int(hs.shape[0]), device, fwd_args.get("_ctx"))
    return [], kw


class _Qwen3NextKV:
    """Minimal state manager: paged HND K/V cache indexed by ``layer_idx``."""

    def __init__(self, layer_idx: int, kc: torch.Tensor, vc: torch.Tensor):
        self.k_cache = [None] * (layer_idx + 1)
        self.v_cache = [None] * (layer_idx + 1)
        self.k_cache[layer_idx] = kc
        self.v_cache[layer_idx] = vc


def _build_qwen3_next_attention_inputs(fwd_args, device, init_args):
    """Qwen3-Next full attention: rebuild the paged-prefill harness the engine
    normally supplies. One prefill sequence over an empty (then just-stored) HND
    KV cache with contiguous slot_mapping/block_table, a fresh partial-NeoX RoPE
    (the captured ref carries no init args), and a ``KimiLinearMetadata`` on the
    Context + a duck-typed state manager the forward reads its cache from."""
    from fastkernels.infra.tp import _tp_size
    from fastkernels.infra.context import KimiLinearMetadata, get_context, set_context
    from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding

    kw = _materialize_all(fwd_args, device, init_args,
                          skip=("state_manager", "rotary_emb", "positions"))
    hs = kw.get("hidden_states")
    if not isinstance(hs, torch.Tensor):
        raise _UnsupportedInput("Qwen3NextAttention: missing hidden_states")
    try:
        nkv = int(init_args["num_key_value_heads"])
        head_dim = int(init_args["head_dim"])
        layer_idx = int(init_args["layer_idx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _UnsupportedInput(f"Qwen3NextAttention: init dims ({exc!r})")

    N = int(hs.shape[0])
    tp = _tp_size()
    num_kv_heads = nkv // tp if nkv % tp == 0 else nkv
    block = 16  # B200 HND page size (ATTN_BACKEND_CONFIG.block_size)
    nblocks = (N + block - 1) // block + 1
    kc = torch.zeros(nblocks, num_kv_heads, block, head_dim,
                     device=device, dtype=hs.dtype)
    vc = torch.zeros_like(kc)

    cu = torch.tensor([0, N], device=device, dtype=torch.int32)
    md = KimiLinearMetadata(
        num_actual_tokens=N,
        query_start_loc=cu, max_query_len=N,
        seq_lens=torch.tensor([N], device=device, dtype=torch.int32), max_seq_len=N,
        num_prefills=1, num_prefill_tokens=N, num_decodes=0, num_decode_tokens=0,
        slot_mapping=torch.arange(N, device=device, dtype=torch.int32),
        block_tables=torch.arange(nblocks, device=device, dtype=torch.int32)
        .view(1, nblocks),
    )
    set_context(is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=N, max_seqlen_k=N)
    get_context().kda_metadata = md

    rope = RotaryEmbedding(head_dim=head_dim // 4, max_position_embeddings=N + 1,
                           rope_theta=10000.0, is_neox_style=True).to(device)
    kw["rotary_emb"] = rope
    kw["positions"] = torch.arange(N, device=device, dtype=torch.int64)
    kw["state_manager"] = _Qwen3NextKV(layer_idx, kc, vc)
    return [], kw


def _build_af3_inputs(fwd_args, device, init_args):
    """AlphaFold3 ops take a ``batch`` feature dict, but the capture only recorded
    the keys the outer model read -- the atom-attention / relpos paths inside
    reach for many more (``ref_pos``, ``atom_to_token_index``, chain/entity ids,
    ...), so forward hits ``KeyError``. Fill the missing keys with finite,
    self-consistent features sized from the captured token/atom masks. Single
    chain/entity; atoms are spread monotonically across tokens."""
    kw = _materialize_all(fwd_args, device, init_args)
    batch = kw.get("batch")
    if not isinstance(batch, dict):
        raise _UnsupportedInput("AF3: missing batch dict")
    tm, am = batch.get("token_mask"), batch.get("atom_mask")
    if not (isinstance(tm, torch.Tensor) and isinstance(am, torch.Tensor)):
        raise _UnsupportedInput("AF3: missing token_mask/atom_mask")
    B, T, A, dt = tm.shape[0], tm.shape[1], am.shape[-1], am.dtype
    ar_t = torch.arange(T, device=device, dtype=dt).unsqueeze(0).expand(B, T)
    a2t = ((torch.arange(A, device=device) * T) // max(A, 1)).clamp_(0, T - 1)
    a2t = a2t.unsqueeze(0).expand(B, A)
    c_elem = int(init_args.get("c_atom_ref_element") or 119)
    c_name = int(init_args.get("c_atom_ref_name_chars") or 256)
    defaults = {
        "residue_index": ar_t, "token_index": ar_t,
        "asym_id": torch.zeros(B, T, device=device, dtype=dt),
        "entity_id": torch.zeros(B, T, device=device, dtype=dt),
        "sym_id": torch.zeros(B, T, device=device, dtype=dt),
        "token_bonds": torch.zeros(B, T, T, device=device, dtype=dt),
        "atom_to_token_index": a2t.to(torch.int64),
        "ref_pos": torch.randn(B, A, 3, device=device, dtype=dt),
        "ref_charge": torch.randn(B, A, device=device, dtype=dt),
        "ref_mask": torch.ones(B, A, device=device, dtype=dt),
        "ref_element": torch.randn(B, A, c_elem, device=device, dtype=dt),
        "ref_atom_name_chars": torch.randn(B, A, 1, c_name, device=device, dtype=dt),
        "ref_space_uid": a2t.to(dt),
    }
    for k, v in defaults.items():
        batch.setdefault(k, v.contiguous() if isinstance(v, torch.Tensor) else v)
    # The diffusion ops embed the noise level as ``0.25 * log(t / sigma_data)``;
    # a random (possibly <= 0) ``t`` / ``noise_schedule`` -> NaN. Force positive,
    # and give the sampler a clean descending sigma schedule.
    t = kw.get("t")
    if isinstance(t, torch.Tensor) and t.is_floating_point():
        kw["t"] = t.abs() + 1.0
    ns = kw.get("noise_schedule")
    if isinstance(ns, torch.Tensor) and ns.numel() > 0:
        kw["noise_schedule"] = torch.linspace(
            160.0, 1e-2, ns.numel(), device=device, dtype=ns.dtype)
    return [], kw


def _build_oasis_rollout_inputs(fwd_args, device, init_args):
    """OasisRollout.forward takes two whole trained models (OasisDiT + the VAE)
    and runs a full DDIM sampling rollout + VAE encode/decode -- a model-level
    driver, not a kernel.  The captured ``model``/``vae`` args are opaque module
    references we cannot rebuild, so mark it a clean, intentional skip."""
    raise _UnsupportedInput(
        "whole-model rollout driver (needs full DiT + VAE); not a kernel")


# qualname -> builder(fwd_args, device, init_args) -> (args, kwargs).
_INPUT_BUILDERS = {
    "fastkernels.tasks.baseline.L1.fp8_linear:Fp8Linear": _build_fp8_linear_inputs,
    # KV-cache stores: bound slot_mapping by the paged cache's num_slots
    # (num_blocks * block_size), whose position differs by layout.
    "fastkernels.tasks.baseline.L1.store_kvcache:StoreKVCache":
        _build_store_kvcache_inputs("k_cache", lambda s: int(s[0]) * int(s[1])),
    "fastkernels.tasks.baseline.L1.store_kvcache:StoreKVCacheHND":
        _build_store_kvcache_inputs("k_cache", lambda s: int(s[0]) * int(s[2])),
    "fastkernels.tasks.baseline.L1.store_kvcache_fp8_mla:StoreKVCacheFP8MLA":
        _build_store_kvcache_inputs("kv_cache", lambda s: int(s[0]) * int(s[1])),
    # MLA chunked-context gather: synthesize consistent gather metadata.
    "fastkernels.tasks.baseline.L1.store_kvcache_fp8_mla:GatherAndDequantKVCacheMLA":
        _build_gather_dequant_mla_inputs,
    # RowParallelLinear: shard the captured full-width activation to input_size//tp.
    "fastkernels.tasks.baseline.L2.parallel_linear:RowParallelLinear":
        _build_row_parallel_inputs,
    # Kimi MLA full attention: drive the dense-prefill MLA path (empty cache).
    "fastkernels.tasks.baseline.L2.kimi_mla_attention:KimiMLAAttention":
        _build_kimi_mla_inputs,
    # Qwen3-Next full attention: synthesize a single-sequence paged-prefill state.
    "fastkernels.tasks.baseline.L2.qwen3_next_attention:Qwen3NextAttention":
        _build_qwen3_next_attention_inputs,
    # AlphaFold3: fill the missing batch feature-dict keys (ref_pos, ids, ...).
    "fastkernels.tasks.baseline.L2.alphafold3_atom_attention:AtomAttentionEncoder":
        _build_af3_inputs,
    "fastkernels.tasks.baseline.L2.alphafold3_atom_attention:AtomAttentionDecoder":
        _build_af3_inputs,
    "fastkernels.tasks.baseline.L2.alphafold3_input_embedder:InputEmbedder":
        _build_af3_inputs,
    "fastkernels.tasks.baseline.L2.alphafold3_diffusion_conditioning:DiffusionConditioning":
        _build_af3_inputs,
    "fastkernels.tasks.baseline.L3.alphafold3_diffusion_module:DiffusionModule":
        _build_af3_inputs,
    "fastkernels.tasks.baseline.L3.alphafold3_diffusion_module:SampleDiffusion":
        _build_af3_inputs,
    # OasisRollout: whole-model rollout driver, not a benchmarkable kernel.
    "fastkernels.tasks.baseline.L3.oasis_rollout:OasisRollout":
        _build_oasis_rollout_inputs,
    # Indexer K-cache gather: pack finite fp8 keys into referenced pages
    # (random uint8 scales decode to NaN; an empty cache is untestable).
    "fastkernels.tasks.baseline.L1.indexer_k_cache:IndexerKCacheGather":
        _build_indexer_k_cache_gather_inputs,
    # Fused MoE grouped GEMM: regenerate valid routing metadata.
    "fastkernels.tasks.baseline.L1.moe_grouped_gemm:MoeGroupedGemm":
        _build_moe_grouped_gemm_inputs,
    # MLA attention: drive the dense-prefill path (empty cache + prefill ctx),
    # wiring kv_b_proj and synthesizing cu_seqlens from the captured composition.
    "fastkernels.tasks.baseline.L2.mla_attention_impl:MLAAttention":
        _build_mla_attention_inputs,
    # DSA sparse indexer: drive the decode path (top-k over a synthesized paged
    # cache); pairs with _prep_sparse_attn_indexer, which fills that cache.
    "fastkernels.tasks.baseline.L2.sparse_attn_indexer:SparseAttnIndexer":
        _build_sparse_attn_indexer_inputs,
    # DeepSeek MLA composites: same prefill-Context trick as MLAAttention above,
    # but end to end (submodules already wired in __init__). One builder serves
    # the attention module and the decoder layer that wraps it.
    "fastkernels.tasks.baseline.L2.deepseek_mla_attention:DeepSeekMLAAttention":
        _build_deepseek_mla_composite_inputs,
    "fastkernels.tasks.baseline.L3.deepseek_decoder:DeepSeekDecoderLayer":
        _build_deepseek_mla_composite_inputs,
    # Paged decode: finite-fill the cache, bound block_table, cap cache_seqlens.
    "fastkernels.tasks.baseline.L1.flashinfer_decode:TRTLLMDecode":
        _build_trtllm_decode_inputs,
    "fastkernels.tasks.baseline.L1.flashinfer_mla_decode:FlashInferMLADecode":
        _build_flashinfer_mla_decode_inputs,
    # Chunked GLA: keep the log-space forget gate <= 0 so it doesn't overflow.
    "fastkernels.tasks.baseline.L1.chunk_gla:ChunkGLA":
        _build_chunk_gla_inputs,
    # GLA linear-attention: drop the opaque RecurrentCache -> cacheless prefill.
    "fastkernels.tasks.baseline.L2.gla_attention:GatedLinearAttention":
        _build_recurrent_prefill_inputs,
    "fastkernels.tasks.baseline.L3.gla_decoder:GLADecoderLayer":
        _build_recurrent_prefill_inputs,
    # Kimi-Linear KDA / Qwen3-Next GDN: attach a prep-built recurrent state +
    # single-sequence prefill Context (state manager comes off the Context).
    "fastkernels.tasks.baseline.L2.kimi_delta_attention:KimiDeltaAttention":
        _build_kimi_recurrent_inputs,
    "fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention:Qwen3NextGDNAttention":
        _build_kimi_recurrent_inputs,
    "fastkernels.tasks.baseline.L3.kimi_linear_decoder:KimiLinearDecoderLayer":
        _build_kimi_recurrent_inputs,
    "fastkernels.tasks.baseline.L3.qwen3_next_decoder:Qwen3NextDecoderLayer":
        _build_kimi_recurrent_inputs,
    # Attention family: publish a prefill Context so forward doesn't deref an
    # unset one, and drop any forward-arg rotary module (use the init rotary).
    "fastkernels.tasks.baseline.L2.attention_impl:Attention":
        _build_attention_prefill_inputs,
    "fastkernels.tasks.baseline.L2.attention:LlamaAttention":
        _build_attention_prefill_inputs,
    "fastkernels.tasks.baseline.L3.llama_decoder:LlamaDecoderLayer":
        _build_attention_prefill_inputs,
    "fastkernels.tasks.baseline.L3.qwen3_moe_decoder:Qwen3MoEDecoderLayer":
        _build_attention_prefill_inputs,
    "fastkernels.tasks.baseline.L3.gpt_oss_decoder:GptOssDecoderLayer":
        _build_attention_prefill_inputs,
    # Grouped-GEMM (Fp8GroupedGemmContiguous / fused_experts) and MLA fp8 ops
    # slot in here with the same reference-quantize approach; deferred until
    # there are captures for those models (DeepSeek / Kimi) to validate against.
}


def _canon_moe_align(out):
    """Canonicalize ``MoeAlign``'s (sorted_token_ids, expert_ids,
    num_tokens_post_padded) output. The token order *within* each expert block
    is assigned by atomics, so it is not deterministic run to run -- only the
    multiset of ids is. Sort the ids so a valid-but-reordered alignment compares
    equal; ``expert_ids`` (block->expert) and the padded length stay exact."""
    sorted_ids, *rest = out
    if isinstance(sorted_ids, torch.Tensor):
        sorted_ids = torch.sort(sorted_ids.flatten()).values
    return (sorted_ids, *rest)


# qualname -> fn(output) -> canonical output, applied to BOTH baseline and
# candidate before comparison (for outputs whose exact values aren't unique).
_OUTPUT_CANON = {
    "fastkernels.tasks.baseline.L1.moe_align:MoeAlign": _canon_moe_align,
}


def _make_call(op: "Operator", cls: type, fwd_args: dict, device: str, init_args: dict):
    """Materialize forward inputs: an operator-specific builder if one is
    registered for the op, else the generic shape/dtype materializer."""
    builder = _INPUT_BUILDERS.get(op.qualname)
    if builder is not None:
        return builder(fwd_args, device, init_args)
    return _build_call(cls, fwd_args, device, init_args)


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
    nonempty_refs = [r for r in ref_leaves if r.numel() > 0]
    if nonempty_refs and all(
            torch.linalg.vector_norm(r.detach().to(torch.float32)).item() == 0
            for r in nonempty_refs):
        return (SKIPPED, 0.0, 0.0, 0.0, "skip: reference output is all-zeros")
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
    "_check_integrity", "_make_call", "_build_call", "_materialize_value",
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

    Each *contiguous* source tensor gets one flat pool sized ``span +
    (iters-1)*step``; call ``i`` copies the (pristine) source into
    ``pool[i*step : i*step+span]`` and returns a view there, so consecutive
    iterations only shift the base address. Copying from a private clone means
    in-place kernels never corrupt later iterations. Adapted from
    SOL-ExecBench's ``ShiftingMemoryPoolAllocator``.

    Non-contiguous tensors (e.g. a column-major / TMA-aligned FP8 scale) are
    passed through unchanged -- copying them into a contiguous pool slot would
    destroy the layout the kernel requires. These are read-only weights/scales,
    so reusing the same tensor across iterations is correct.
    """

    def __init__(self, tensors: list[torch.Tensor], total: int):
        self.entries = []  # ("pool", pool, src, span, step, shape) | ("keep", tensor)
        self.total = total
        self.i = 0
        for t in tensors:
            if not t.is_contiguous():
                self.entries.append(("keep", t))
                continue
            step = max(1, 256 // t.element_size())
            span = t.numel()
            pool = torch.empty(span + (total - 1) * step, dtype=t.dtype, device=t.device)
            src = t.reshape(-1).clone()
            self.entries.append(("pool", pool, src, span, step, tuple(t.shape)))

    def next(self) -> list[torch.Tensor]:
        idx = min(self.i, self.total - 1)
        self.i += 1
        out = []
        for entry in self.entries:
            if entry[0] == "keep":
                out.append(entry[1])
                continue
            _, pool, src, span, step, shape = entry
            slot = pool.narrow(0, idx * step, span)
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
    dtype = _FORCE_DTYPE.get(op.qualname) or _case_dtype(fwd_args)
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
    # Give any FP8 linear / NVFP4 MoE submodules valid quantized weights, and
    # repair any uninitialized (torch.empty) high-precision weights, on both
    # modules (so param shapes match) before sharing weights baseline ->
    # candidate.
    _init_fp8_module_weights(baseline)
    _init_fp8_module_weights(candidate)
    _init_nvfp4_module_weights(baseline)
    _init_nvfp4_module_weights(candidate)
    _sanitize_float_params(baseline)
    _sanitize_float_params(candidate)
    try:
        candidate.load_state_dict(baseline.state_dict(), strict=False)
    except Exception:
        pass  # best-effort weight sharing (old runner does the same)

    # Op-specific module state (e.g. a valid paged cache) the forward reads from
    # ``self`` rather than its args. Seeded so both modules match; run after
    # weight sharing so it is not clobbered.
    prep = _MODULE_PREP.get(op.qualname)
    if prep is not None:
        try:
            prep(baseline, init_args, device)
            prep(candidate, init_args, device)
        except _UnsupportedInput as exc:
            res.detail = f"skip: {exc}"
            return res
        except Exception as exc:  # noqa: BLE001 - prep needs a kernel not present here
            res.detail = f"skip: module prep failed ({exc!r})"
            return res

    # 2) Correctness over N rounds of fresh inputs.
    canon = _OUTPUT_CANON.get(op.qualname)
    worst = None
    for r in range(rounds):
        torch.manual_seed(seed + r)
        try:
            base_call = _make_call(op, baseline_cls, fwd_args, device, init_args)
        except _UnsupportedInput as exc:
            res.detail = f"skip: {exc}"
            return res
        args, kwargs = base_call
        # Reseed identically before each forward so ops that draw from the
        # global RNG inside forward (e.g. stochastic diffusion samplers) see the
        # same noise on baseline and candidate; a no-op for deterministic ops.
        try:
            torch.manual_seed(seed + r)
            ref = _run_forward(baseline, *(_clone_tree((args, kwargs))))
        except Exception as exc:  # noqa: BLE001 - baseline rejects generated inputs
            res.detail = f"skip: baseline failed on generated inputs ({exc!r})"
            return res
        try:
            torch.manual_seed(seed + r)
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
        out_c, ref_c = (canon(out), canon(ref)) if canon else (out, ref)
        status, max_abs, max_rel, matched, detail = _compare(
            out_c, ref_c, check_dtype=(r == 0))
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
        base_call = _make_call(op, baseline_cls, fwd_args, device, init_args)
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
        # Reclaim between cases. A skipped/errored case leaves its two freshly
        # built modules (for an L4 model that is ~30 GB) alive inside a reference
        # cycle -- the caught exception's traceback pins the frame that holds
        # them -- which torch.cuda.empty_cache() alone cannot release. Without a
        # gc.collect() these cycles pile up case-over-case until the worker OOMs
        # (verified: allocated stayed at 30 GB after empty_cache, dropped to
        # ~0 GB after gc.collect). Collect first, then hand the freed blocks
        # back to the driver.
        del res
        gc.collect()
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
    cmd = [sys.executable, "-u", "-m", "fastkernels.bench", "--worker",
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
# Distributed (TP) ops: auto-spawn a world_size=TP NCCL group, no torchrun.
#
# A few ops only make sense across ranks: ``AllReduce`` issues a collective, and
# ``RowParallelLinear`` row-shards its input (captured activation width =
# input_size // TP) then all-reduces. A tp=1 rebuild mismatches the shard width
# and has no group to reduce over, so those cases skip. Instead we grab the whole
# GPU set and run the op in a real world_size=TP group. Every rank runs the same
# seeded cases in lockstep so the in-forward collectives stay matched; rank 0
# records the results. TP comes from the capture (its sharding is baked into the
# captured shapes), so replaying at the same world size is what makes it valid.
# ---------------------------------------------------------------------------
_DISTRIBUTED_OPS = {
    "fastkernels.tasks.baseline.L1.allreduce:AllReduce",
    "fastkernels.tasks.baseline.L2.parallel_linear:RowParallelLinear",
}


def _capture_tp(captures_dir: Path) -> int:
    """TP degree the capture ran at, parsed from report filenames (``..._tp8_``).
    RowParallelLinear's activation width is ``input_size // tp``, so a faithful
    replay must use the same world size. 0 if no filename encodes it."""
    best = 0
    for p in captures_dir.rglob("*.json"):
        m = re.search(r"_tp(\d+)", str(p))
        if m:
            best = max(best, int(m.group(1)))
    return best


def _free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _dist_worker_entry(rank, world_size, port, op_qualname, wargs, out_path, gpu_ids):
    """One rank of an auto-spawned NCCL group (run via ``mp.spawn``). Builds the
    op sharded at ``world_size`` (``_tp_size()`` reads the live group) and runs
    its cases; rank 0 appends result JSON to ``out_path``. All ranks do identical
    seeded work so the collectives inside each forward stay in lockstep."""
    import torch.distributed as dist
    from datetime import timedelta
    # Pin this rank to its GPU before any CUDA init (spawn children start clean).
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            world_size=world_size, rank=rank,
                            device_id=torch.device(device),
                            timeout=timedelta(minutes=5))
    try:
        op = _parse_operator(op_qualname)
        baseline_cls = _import_symbol(op_qualname)
        candidate_cls = (baseline_cls if wargs["self_test"]
                         else _load_candidate_class(op))
        if candidate_cls is None:
            return
        integrity = _snapshot_integrity()
        reports = _load_reports(Path(wargs["captures"]))
        cases = _collect_cases(op, reports, wargs["max_shapes"])
        sink = open(out_path, "a") if rank == 0 else None
        try:
            for case in cases:
                torch.manual_seed(DEFAULT_SEED)  # identical weights across ranks
                try:
                    res = _bench_one_case(op, baseline_cls, candidate_cls, case,
                                          device, wargs["warmup"], wargs["iters"],
                                          wargs["rounds"], DEFAULT_SEED, integrity)
                    d = res.to_dict()
                except Exception as exc:  # noqa: BLE001 - keep the group alive
                    traceback.print_exc()
                    d = ScenarioResult(
                        op=op.qualname, level=op.level,
                        shape=_shape_repr(case["fwd_args"]), dtype="-",
                        status=RUNTIME_ERROR, detail=f"{exc!r}").to_dict()
                if sink is not None:
                    sink.write(json.dumps(_json_safe(d)) + "\n")
                    sink.flush()
                dist.barrier()  # realign ranks before the next case
        finally:
            if sink is not None:
                sink.close()
    finally:
        dist.destroy_process_group()


def _run_distributed(ops: list[Operator], args, gpu_ids: list[str]) -> list[ScenarioResult]:
    """Run each distributed op in its own world_size=TP group (sequentially --
    each grabs every GPU). Falls back to a clear SKIP when fewer GPUs than TP are
    available (the shard width can't be reproduced)."""
    import torch.multiprocessing as mp

    tp = _capture_tp(Path(args.captures))
    world_size = tp or len(gpu_ids)
    log_dir = RESULTS_DIR / "kernel_bench_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out: list[ScenarioResult] = []
    for op in ops:
        if len(gpu_ids) < world_size:
            out.append(ScenarioResult(
                op=op.qualname, level=op.level, shape="-", dtype="-",
                status=SKIPPED,
                detail=f"needs {world_size} GPUs (capture TP), have {len(gpu_ids)}"))
            print(f"  -- {op.qualname} skipped (needs {world_size} GPUs)")
            continue
        ranks_gpus = gpu_ids[:world_size]
        jsonl = log_dir / f"{op.level}_{op.stem}_{op.class_name}.dist.jsonl"
        jsonl.write_text("")
        wargs = {"captures": str(args.captures), "max_shapes": args.max_shapes,
                 "warmup": args.warmup, "iters": args.iters,
                 "rounds": args.rounds, "self_test": bool(args.self_test)}
        print(f"  -> [GPUs {','.join(ranks_gpus)}] {op.qualname} "
              f"(distributed, world_size={world_size}) started")
        try:
            mp.spawn(_dist_worker_entry,
                     args=(world_size, _free_port(), op.qualname, wargs,
                           str(jsonl), ranks_gpus),
                     nprocs=world_size, join=True)
        except Exception as exc:  # noqa: BLE001 - a dead rank surfaces here
            out.append(ScenarioResult(
                op=op.qualname, level=op.level, shape="-", dtype="-",
                status=RUNTIME_ERROR, detail=f"distributed run failed: {exc!r}"))
            print(f"  <- {op.qualname} FAILED ({exc!r})")
            continue
        scenarios = _read_scenarios(jsonl, op)
        if not scenarios:
            scenarios = [ScenarioResult(
                op=op.qualname, level=op.level, shape="-", dtype="-",
                status=RUNTIME_ERROR, detail="no results from rank 0")]
        np = sum(1 for s in scenarios if s.status == PASSED)
        nf = sum(1 for s in scenarios if s.status in _FAIL_STATUSES)
        print(f"  <- {op.qualname} done ({np} passed, {nf} failed)")
        out.extend(scenarios)
    return out


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
                   help="Only operators at this level. L4 whole-model ops are "
                        "skipped by default; pass --level 4 or --target to run them.")
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
    # Restrict to list.py operator *targets*: never benchmark a non-target helper
    # (e.g. an inner model class excluded via ``__targets__``) that only ran as
    # part of a bigger op.
    try:
        from .list import discover_operator_targets
        targets = {(t.level, t.name, t.class_name) for t in discover_operator_targets()}
    except Exception:  # noqa: BLE001 - fall back to unfiltered if discovery fails
        targets = None
    if targets:
        ops = [o for o in ops if (o.level, o.stem, o.class_name) in targets]
    if args.level is not None:
        ops = [o for o in ops if o.level == args.level]
    elif args.target is None:
        # L4 ops are whole-model wrappers, not kernels: rebuilding a TP-sharded
        # model on one GPU OOMs. Skip by default; opt in via --level 4/--target.
        ops = [o for o in ops if o.level != 4]
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
    # Collective / TP-sharded ops need a real world_size=TP group (below); the
    # rest run one-per-GPU in parallel.
    dist_ops = [o for o in ops if o.qualname in _DISTRIBUTED_OPS]
    normal_ops = [o for o in ops if o.qualname not in _DISTRIBUTED_OPS]
    print(f"Benchmarking {len(ops)} operator(s) in {mode} mode on GPU(s) "
          f"{','.join(gpu_ids)} ({len(normal_ops)} one-per-GPU"
          + (f", {len(dist_ops)} distributed" if dist_ops else "") + ").")

    locked = _lock_clocks() if args.lock_clocks else False
    try:
        result = _run_parallel(normal_ops, args, gpu_ids, mode)
        if dist_ops:
            result.scenarios.extend(_run_distributed(dist_ops, args, gpu_ids))
    finally:
        if locked:
            _unlock_clocks()

    result.print_table()
    out_path = Path(args.output) if args.output else run_output_path("bench")
    result.save_json(out_path)
    print(f"\nResults written to {out_path}")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
