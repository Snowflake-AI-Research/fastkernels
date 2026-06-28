"""Isolated kernel-level benchmarking via direct forward() calls.

Instantiates baseline and candidate nn.Module instances, copies weights,
loads inputs from the InputRegistry (random or golden), compares outputs
and timing. No full model build required — per-kernel test time is seconds
rather than minutes.
"""

from __future__ import annotations

import time
import inspect
from typing import Any

import torch
import torch.nn as nn

from fastkernels.bench.kernels.scenario_registry import InputRegistry
from fastkernels.infra.kernel_swapper import (
    BenchTarget,
    discover_references,
    discover_targets,
    get,
    load_candidate,
    load_reference,
)

from .result import KernelBenchResult, OperatorResult, ScenarioResult

_DEFAULT_REGISTRY = None
_FP32_ATOL = 1e-5
_FP32_RTOL = 1e-3
_LOW_PRECISION_ATOL = 1e-2
_LOW_PRECISION_RTOL = 1e-2
_FP8_ATOL = 1.25e-1
_FP8_RTOL = 1.25e-1
_FP8_GROUP_SIZE = 128


def _short_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


def _get_registry() -> InputRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InputRegistry()
    return _DEFAULT_REGISTRY


def _find_candidate_path(target_name: str, level: int) -> str:
    """Return the relative path to the candidate file for display."""
    return f"tasks/candidate/L{level}/{target_name}.py"


def _find_reference_path(target_name: str, level: int) -> str:
    """Return the relative path to the semantic reference file for display."""
    return f"tasks/reference/L{level}/{target_name}.py"


def _instantiate_module(
    cls: type,
    init_args: dict[str, Any],
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Create an nn.Module instance with init_args, handling common patterns."""
    kwargs = dict(init_args)
    if "head_size" in kwargs and "head_dim" not in kwargs:
        kwargs["head_dim"] = kwargs.pop("head_size")
    if "base" in kwargs and "rope_theta" not in kwargs:
        kwargs["rope_theta"] = kwargs.pop("base")
    kwargs.pop("rotary_dim", None)
    kwargs.pop("is_neox_style", None)
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
        if not accepts_kwargs:
            kwargs = {
                k: v for k, v in kwargs.items()
                if k in params and k != "self"
            }
    except (TypeError, ValueError):
        pass

    try:
        module = cls(**kwargs)
    except TypeError:
        module = cls()

    module = module.to(device)
    if dtype is not None:
        # Cast learnable parameters to the scenario dtype without changing
        # precision-sensitive buffers such as RoPE/YARN cos/sin caches.
        with torch.no_grad():
            for param in module.parameters(recurse=True):
                if param.is_floating_point():
                    param.data = param.data.to(dtype=dtype)
    module.eval()
    return module


def _first_floating_dtype(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        if "float8" not in str(value.dtype):
            return value.dtype
        return None
    if isinstance(value, dict):
        for v in value.values():
            dtype = _first_floating_dtype(v)
            if dtype is not None:
                return dtype
    if isinstance(value, (tuple, list)):
        for v in value:
            dtype = _first_floating_dtype(v)
            if dtype is not None:
                return dtype
    return None


def _clone_input_value(value: Any) -> Any:
    """Clone tensors in an input tree so in-place kernels cannot cross-contaminate runs."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_input_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_input_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_input_value(v) for k, v in value.items()}
    return value


def _clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: _clone_input_value(v) for k, v in inputs.items()}


def _contains_cuda_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_cuda_tensor(v) for v in value)
    return False


def _synchronize_if_cuda(*values: Any) -> None:
    if any(_contains_cuda_tensor(v) for v in values):
        torch.cuda.synchronize()


def _time_forward(
    module: nn.Module,
    inputs: dict[str, Any],
    num_warmup: int,
    num_runs: int,
) -> tuple[Any, float]:
    """Warmup + time forward() calls. Returns (output, median_ms)."""
    tensor_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items()
        if not isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        for _ in range(num_warmup):
            module(**tensor_inputs, **scalar_inputs)

        _synchronize_if_cuda(tensor_inputs)
        times = []
        output = None
        for _ in range(num_runs):
            start = time.perf_counter()
            output = module(**tensor_inputs, **scalar_inputs)
            _synchronize_if_cuda(tensor_inputs, output)
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]
    if output is None:
        output = {k: v for k, v in tensor_inputs.items()}
    return output, median_ms


def _run_forward_once(module: nn.Module, inputs: dict[str, Any]) -> Any:
    tensor_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items()
        if not isinstance(v, torch.Tensor)
    }
    with torch.no_grad():
        output = module(**tensor_inputs, **scalar_inputs)
        _synchronize_if_cuda(tensor_inputs, output)
    return output


def _tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return (atol, rtol) for tolerance-normalized correctness."""
    dtype_name = str(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return _LOW_PRECISION_ATOL, _LOW_PRECISION_RTOL
    if "float8" in dtype_name:
        return _FP8_ATOL, _FP8_RTOL
    return _FP32_ATOL, _FP32_RTOL


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _is_fp8_tensor(tensor: Any) -> bool:
    return isinstance(tensor, torch.Tensor) and "float8" in str(tensor.dtype)


def _fp8_rtol(dtype: torch.dtype) -> float:
    if "e5m2" in str(dtype):
        return 0.25
    return 0.125


def _expand_fp8_scale(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int = _FP8_GROUP_SIZE,
) -> torch.Tensor | None:
    """Broadcast FP8 per-group/per-block scales to the FP8 tensor shape."""
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point():
        return None

    shape = tuple(fp8.shape)
    scale_shape = tuple(scale.shape)
    per_group_shape = (*shape[:-1], _ceil_div(shape[-1], group_size))
    if scale_shape == per_group_shape:
        return scale.float().repeat_interleave(group_size, dim=-1)[..., :shape[-1]]

    if fp8.ndim >= 2:
        per_block_shape = (
            *shape[:-2],
            _ceil_div(shape[-2], group_size),
            _ceil_div(shape[-1], group_size),
        )
        if scale_shape == per_block_shape:
            expanded = scale.float().repeat_interleave(group_size, dim=-2)
            expanded = expanded.repeat_interleave(group_size, dim=-1)
            return expanded[..., :shape[-2], :shape[-1]]

    return None


def _compare_fp8_scaled_outputs(
    baseline_fp8: torch.Tensor,
    baseline_scale: torch.Tensor,
    candidate_fp8: torch.Tensor,
    candidate_scale: torch.Tensor,
) -> tuple[bool, float, float]:
    """Compare FP8 tensors in dequantized value space using local scales."""
    if baseline_fp8.shape != candidate_fp8.shape:
        return False, float("inf"), float("inf")

    baseline_scale_expanded = _expand_fp8_scale(baseline_fp8, baseline_scale)
    candidate_scale_expanded = _expand_fp8_scale(candidate_fp8, candidate_scale)
    if baseline_scale_expanded is None or candidate_scale_expanded is None:
        return False, float("inf"), float("inf")

    baseline = baseline_fp8.float() * baseline_scale_expanded
    candidate = candidate_fp8.float() * candidate_scale_expanded
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        return False, float("inf"), float("inf")

    diff = (baseline - candidate).abs()
    mean_diff = diff.mean().item()
    atol = 0.5 * baseline_scale_expanded.abs().clamp_min(1e-12)
    tolerance = atol + _fp8_rtol(baseline_fp8.dtype) * baseline.abs()
    max_error_ratio = (diff / tolerance).max().item()
    passed = max_error_ratio <= 1.0
    return passed, max_error_ratio, mean_diff


def _compare_outputs(baseline_out: Any, candidate_out: Any) -> tuple[bool, float, float]:
    """Compare outputs: return (pass, max_error_ratio, mean_abs_diff)."""
    if isinstance(baseline_out, torch.Tensor) and isinstance(candidate_out, torch.Tensor):
        if baseline_out.shape != candidate_out.shape:
            return False, float("inf"), float("inf")

        baseline = baseline_out.float()
        candidate = candidate_out.float()
        if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
            return False, float("inf"), float("inf")

        diff = (baseline - candidate).abs()
        mean_diff = diff.mean().item()
        atol, rtol = _tolerances_for_dtype(baseline_out.dtype)
        tolerance = atol + rtol * baseline.abs()
        max_error_ratio = (diff / tolerance).max().item()
        passed = max_error_ratio <= 1.0
        return passed, max_error_ratio, mean_diff

    if isinstance(baseline_out, (tuple, list)) and isinstance(candidate_out, (tuple, list)):
        if len(baseline_out) != len(candidate_out):
            return False, float("inf"), float("inf")
        all_pass = True
        max_error_ratio = 0.0
        total_diff = 0.0
        count = 0
        i = 0
        while i < len(baseline_out):
            b = baseline_out[i]
            c = candidate_out[i]
            if (
                i + 1 < len(baseline_out)
                and _is_fp8_tensor(b)
                and _is_fp8_tensor(c)
                and isinstance(baseline_out[i + 1], torch.Tensor)
                and isinstance(candidate_out[i + 1], torch.Tensor)
            ):
                baseline_scale = baseline_out[i + 1]
                candidate_scale = candidate_out[i + 1]
                if (
                    _expand_fp8_scale(b, baseline_scale) is not None
                    and _expand_fp8_scale(c, candidate_scale) is not None
                ):
                    p, ratio, d = _compare_fp8_scaled_outputs(
                        b, baseline_scale, c, candidate_scale,
                    )
                    all_pass = all_pass and p
                    max_error_ratio = max(max_error_ratio, ratio)
                    total_diff += d
                    count += 1
                    i += 2
                    continue

            if isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                p, ratio, d = _compare_outputs(b, c)
                all_pass = all_pass and p
                max_error_ratio = max(max_error_ratio, ratio)
                total_diff += d
                count += 1
            i += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, max_error_ratio, mean_diff

    if isinstance(baseline_out, dict) and isinstance(candidate_out, dict):
        if set(baseline_out) != set(candidate_out):
            return False, float("inf"), float("inf")
        all_pass = True
        max_error_ratio = 0.0
        total_diff = 0.0
        count = 0
        for key in sorted(baseline_out):
            b = baseline_out[key]
            c = candidate_out[key]
            p, ratio, d = _compare_outputs(b, c)
            all_pass = all_pass and p
            max_error_ratio = max(max_error_ratio, ratio)
            total_diff += d
            count += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, max_error_ratio, mean_diff

    return True, 0.0, 0.0


def _merge_correctness(
    output_check: tuple[bool, float, float],
    input_check: tuple[bool, float, float],
) -> tuple[bool, float, float]:
    output_correct, output_ratio, output_diff = output_check
    input_correct, input_ratio, input_diff = input_check
    correct = output_correct and input_correct
    max_error_ratio = max(output_ratio, input_ratio)
    if output_diff == 0.0:
        mean_diff = input_diff
    elif input_diff == 0.0:
        mean_diff = output_diff
    else:
        mean_diff = 0.5 * (output_diff + input_diff)
    return correct, max_error_ratio, mean_diff


def run_kernel_benchmark(
    target_name: str,
    scenarios: list[str] | None = None,
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    pytorch_reference: bool = False,
    validation_mode: str = "candidate",
) -> OperatorResult:
    """Run isolated kernel benchmark for a single operator.

    For each matching scenario in the InputRegistry:
    1. Instantiate baseline and candidate with init_args
    2. Copy baseline weights to candidate (via load_state_dict)
    3. Prepare inputs (random or golden)
    4. Warmup both
    5. Time both (median of num_runs)
    6. Compare outputs: max error ratio pass/fail, mean abs diff

    The candidate implementation is auto-discovered from
    tasks/candidate/L{level}/{target_name}.py.

    Args:
        target_name: Operator name (e.g. 'rms_norm').
        scenarios: Filter by scenario name patterns.
        models: Filter by model key prefix.
        tp: Filter by TP degrees.
        category: Filter by category (not yet used).
        num_warmup: Warmup iterations.
        num_runs: Timed iterations for median.
        device: Device for tensors.

    Returns:
        OperatorResult with per-scenario correctness and speedup.
    """
    target = get(target_name)

    if pytorch_reference:
        validation_mode = "pytorch_reference"

    if validation_mode == "baseline_identity":
        user_impl = target.target_cls
    elif validation_mode == "pytorch_reference":
        user_impl = load_reference(target_name)
    else:
        user_impl = load_candidate(target_name)

    if user_impl is None:
        impl_kind = "PyTorch reference" if validation_mode == "pytorch_reference" else "candidate kernel"
        impl_dir = "reference" if validation_mode == "pytorch_reference" else "candidate"
        raise ValueError(
            f"No {impl_kind} found for {target_name!r}. "
            f"Place implementation in tasks/{impl_dir}/L{target.level}/{target_name}.py"
        )

    registry = _get_registry()
    all_scenarios = registry.scenarios(
        target_name, models=models, tp=tp, category=category,
    )

    if scenarios:
        all_scenarios = [
            s for s in all_scenarios
            if any(pat in s.name for pat in scenarios)
        ]

    if not all_scenarios:
        print(f"  WARNING: No scenarios found for {target_name} in InputRegistry.")
        return OperatorResult(
            target=target_name,
            level=target.level,
            candidate_path=(
                _find_reference_path(target_name, target.level)
                if pytorch_reference
                else _find_candidate_path(target_name, target.level)
            ),
        )

    if validation_mode == "baseline_identity":
        candidate_path = f"tasks/baseline/L{target.level}/{target_name}.py"
    elif validation_mode == "pytorch_reference":
        candidate_path = _find_reference_path(target_name, target.level)
    else:
        candidate_path = _find_candidate_path(target_name, target.level)
    scenario_results: list[ScenarioResult] = []

    for scenario in all_scenarios:
        try:
            inputs = registry.get_inputs(target_name, scenario.name, device=device)
            input_dtype = _first_floating_dtype(inputs)

            baseline_mod = _instantiate_module(
                target.target_cls, scenario.init_args, device, dtype=input_dtype,
            )
            candidate_mod = _instantiate_module(
                user_impl, scenario.init_args, device, dtype=input_dtype,
            )

            if hasattr(baseline_mod, "state_dict") and len(baseline_mod.state_dict()) > 0:
                try:
                    candidate_mod.load_state_dict(baseline_mod.state_dict(), strict=False)
                except Exception:
                    pass

            timing_warmup = 0 if validation_mode == "candidate_smoke" else num_warmup
            timing_runs = 1 if validation_mode == "candidate_smoke" else num_runs

            baseline_check_inputs = _clone_inputs(inputs)
            candidate_check_inputs = _clone_inputs(inputs)
            baseline_out = _run_forward_once(baseline_mod, baseline_check_inputs)
            candidate_out = _run_forward_once(candidate_mod, candidate_check_inputs)

            correct, max_error_ratio, mean_diff = _merge_correctness(
                _compare_outputs(baseline_out, candidate_out),
                _compare_outputs(baseline_check_inputs, candidate_check_inputs),
            )

            _, baseline_ms = _time_forward(
                baseline_mod, _clone_inputs(inputs), timing_warmup, timing_runs,
            )
            _, candidate_ms = _time_forward(
                candidate_mod, _clone_inputs(inputs), timing_warmup, timing_runs,
            )
            speedup = baseline_ms / candidate_ms if candidate_ms > 0 else float("inf")
            classification = (
                "harness_validation_passed"
                if validation_mode in ("baseline_identity", "pytorch_reference")
                and correct
                else "candidate_correct_and_timed"
                if correct
                else "candidate_correctness_failure"
            )

            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=correct,
                max_error_ratio=max_error_ratio,
                mean_abs_diff=mean_diff,
                baseline_ms=baseline_ms,
                candidate_ms=candidate_ms,
                speedup=speedup,
                failure_reason=None if correct else "output_mismatch",
                classification=classification,
            ))

        except Exception as e:
            failure_reason = _short_exception(e)
            print(f"  ERROR in scenario {scenario.name}: {failure_reason}")
            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=False,
                max_error_ratio=float("inf"),
                mean_abs_diff=float("inf"),
                baseline_ms=0.0,
                candidate_ms=0.0,
                speedup=0.0,
                failure_reason=failure_reason,
                classification="harness_or_candidate_exception",
            ))

        finally:
            for v in list(locals().values()):
                if isinstance(v, nn.Module):
                    del v

    op_result = OperatorResult(
        target=target_name,
        level=target.level,
        candidate_path=candidate_path,
        scenarios=scenario_results,
    )
    op_result.compute_aggregates()
    return op_result


def run_all_kernel_benchmarks(
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    pytorch_reference: bool = False,
    validation_mode: str = "candidate",
) -> KernelBenchResult:
    """Run kernel benchmarks for all operators that have candidate implementations.

    Discovers all candidate kernels and runs isolated benchmarks for each.
    """
    from fastkernels.infra.kernel_swapper import discover_candidates

    if pytorch_reference:
        validation_mode = "pytorch_reference"

    candidates = discover_references() if validation_mode == "pytorch_reference" else discover_candidates()
    if not candidates:
        if validation_mode == "pytorch_reference":
            print("No PyTorch references found in tasks/reference/.")
        else:
            print("No candidate kernels found in tasks/candidate/.")
        result = KernelBenchResult()
        result.compute_aggregates()
        return result

    operators: list[OperatorResult] = []
    for target, _ in candidates:
        label = (
            "baseline identity"
            if validation_mode == "baseline_identity"
            else "PyTorch reference"
            if validation_mode == "pytorch_reference"
            else "candidate smoke"
            if validation_mode == "candidate_smoke"
            else "candidate"
        )
        print(f"\n  Benchmarking {target.name} (L{target.level}, {label})...")
        op_result = run_kernel_benchmark(
            target.name,
            models=models,
            tp=tp,
            category=category,
            num_warmup=num_warmup,
            num_runs=num_runs,
            device=device,
            pytorch_reference=pytorch_reference,
            validation_mode=validation_mode,
        )
        operators.append(op_result)

    result = KernelBenchResult(operators=operators)
    result.compute_aggregates()
    return result
