"""Unified Input Registry for kernel-level benchmarking.

Loads YAML manifests that describe operator input specifications (shapes, dtypes,
init_args) and optionally references golden .pt files for data-dependent operators.
Provides tensors to the KernelRunner for isolated forward() testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file as load_safetensors

from kb_nano import GOLDEN_DIR, INPUTS_DIR

_INPUTS_DIR = INPUTS_DIR
_GOLDEN_DIR = GOLDEN_DIR

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_MAP["float8_e4m3fn"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _DTYPE_MAP["float8_e5m2"] = torch.float8_e5m2

_FP8_GROUP_SIZE = 128


def _parse_dtype(s: str) -> torch.dtype:
    dt = _DTYPE_MAP.get(s)
    if dt is None:
        raise ValueError(f"Unknown dtype {s!r}. Supported: {list(_DTYPE_MAP)}")
    return dt


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_scale_to_power_of_two(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(scale)))


def _input_shape(inputs: dict[str, Any], name: str) -> list[int] | None:
    spec = inputs.get(name)
    if isinstance(spec, dict) and "shape" in spec:
        return list(spec["shape"])
    return None


def _input_scalar(inputs: dict[str, Any], name: str) -> Any | None:
    value = inputs.get(name)
    if isinstance(value, dict):
        return None
    return value


def _infer_num_experts(scenario: "Scenario", result: dict[str, Any]) -> int | None:
    for source in (result, scenario.inputs, scenario.init_args):
        value = source.get("num_experts") if isinstance(source, dict) else None
        if isinstance(value, int) and value > 0:
            return value
    for name in ("B", "w13", "w2"):
        shape = _input_shape(scenario.inputs, name)
        if shape:
            return int(shape[0])
    return None


def _infer_cache_slots(scenario: "Scenario") -> int | None:
    shape = _input_shape(scenario.inputs, "k_cache") or _input_shape(scenario.inputs, "kv_cache")
    if not shape:
        return None
    page_size = scenario.init_args.get("page_size")
    if isinstance(page_size, int) and page_size > 0:
        return int(shape[0]) * page_size
    if len(shape) >= 2:
        return int(shape[0]) * int(shape[1])
    return int(shape[0])


def _infer_num_valid_tokens(scenario: "Scenario") -> int | None:
    for name in ("A", "hidden_states", "key", "router_logits", "gating_output"):
        shape = _input_shape(scenario.inputs, name)
        if shape:
            return int(shape[0])
    topk_shape = _input_shape(scenario.inputs, "topk_ids")
    if topk_shape:
        return int(topk_shape[0])
    return None


def _constrained_integer_tensor(
    *,
    operator: str,
    arg_name: str,
    shape: list[int],
    dtype: torch.dtype,
    scenario: "Scenario",
    result: dict[str, Any],
    device: str,
) -> torch.Tensor | None:
    if operator == "store_kvcache" and arg_name == "slot_mapping":
        num_slots = _infer_cache_slots(scenario)
        if num_slots is not None and num_slots > 0:
            return torch.randint(0, num_slots, shape, dtype=dtype, device=device)

    if arg_name == "topk_ids":
        num_experts = _infer_num_experts(scenario, result)
        if num_experts is not None and num_experts > 0:
            return torch.randint(0, num_experts, shape, dtype=dtype, device=device)

    if operator == "moe_grouped_gemm" and arg_name == "expert_ids":
        num_experts = _infer_num_experts(scenario, result)
        if num_experts is not None and num_experts > 0:
            return torch.randint(0, num_experts, shape, dtype=dtype, device=device)

    if operator == "moe_grouped_gemm" and arg_name == "sorted_token_ids":
        num_valid_tokens = _infer_num_valid_tokens(scenario)
        top_k = int(_input_scalar(scenario.inputs, "top_k") or scenario.init_args.get("top_k") or 1)
        upper = max(1, num_valid_tokens * max(1, top_k)) if num_valid_tokens else None
        if upper is not None:
            return torch.randint(0, upper, shape, dtype=dtype, device=device)

    if operator == "moe_grouped_gemm" and arg_name == "num_tokens_post_padded":
        sorted_shape = _input_shape(scenario.inputs, "sorted_token_ids")
        value = int(sorted_shape[0]) if sorted_shape else max(1, int(torch.tensor(shape).prod().item()))
        return torch.full(shape, value, dtype=dtype, device=device)

    return None


def _quantize_fp8_per_token_group(
    source: torch.Tensor,
    dtype: torch.dtype,
    *,
    group_size: int = _FP8_GROUP_SIZE,
    use_ue8m0: bool = True,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension and return coherent FP8 values + scales."""
    fp8_info = torch.finfo(dtype)
    orig_shape = source.shape
    hidden = orig_shape[-1]
    groups = _ceil_div(hidden, group_size)
    padded_hidden = groups * group_size
    flat = source.reshape(-1, hidden).float()

    if padded_hidden != hidden:
        padded = torch.zeros(
            flat.shape[0], padded_hidden, dtype=flat.dtype, device=flat.device,
        )
        padded[:, :hidden] = flat
        flat_for_scale = padded
    else:
        flat_for_scale = flat

    grouped = flat_for_scale.view(flat.shape[0], groups, group_size)
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / fp8_info.max
    if use_ue8m0:
        scale = _round_scale_to_power_of_two(scale)

    expanded = scale.repeat_interleave(group_size, dim=-1)[:, :hidden]
    quantized = torch.clamp(
        flat / expanded, fp8_info.min, fp8_info.max,
    ).to(dtype).reshape(orig_shape)
    scale = scale.reshape(*orig_shape[:-1], groups)
    return quantized, scale


def _quantize_fp8_per_block(
    source: torch.Tensor,
    dtype: torch.dtype,
    *,
    block_shape: list[int] | tuple[int, int] = (_FP8_GROUP_SIZE, _FP8_GROUP_SIZE),
    use_ue8m0: bool = True,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last two dimensions with block scales."""
    fp8_info = torch.finfo(dtype)
    block_m, block_n = int(block_shape[0]), int(block_shape[1])
    orig_shape = source.shape
    rows, cols = orig_shape[-2], orig_shape[-1]
    row_blocks = _ceil_div(rows, block_m)
    col_blocks = _ceil_div(cols, block_n)
    padded_rows = row_blocks * block_m
    padded_cols = col_blocks * block_n
    flat = source.reshape(-1, rows, cols).float()

    padded = torch.zeros(
        flat.shape[0], padded_rows, padded_cols, dtype=flat.dtype, device=flat.device,
    )
    padded[:, :rows, :cols] = flat
    grouped = padded.view(
        flat.shape[0], row_blocks, block_m, col_blocks, block_n,
    )
    scale = grouped.abs().amax(dim=(2, 4)).clamp_min(eps) / fp8_info.max
    if use_ue8m0:
        scale = _round_scale_to_power_of_two(scale)

    expanded = scale.repeat_interleave(block_m, dim=-2).repeat_interleave(block_n, dim=-1)
    expanded = expanded[:, :rows, :cols]
    quantized = torch.clamp(
        flat / expanded, fp8_info.min, fp8_info.max,
    ).to(dtype).reshape(orig_shape)
    scale = scale.reshape(*orig_shape[:-2], row_blocks, col_blocks)
    return quantized, scale


class Scenario:
    """A single test scenario for an operator."""

    __slots__ = (
        "name",
        "init_args",
        "inputs",
        "input_metadata",
        "golden_path",
        "golden_inputs",
    )

    def __init__(
        self,
        name: str,
        init_args: dict[str, Any],
        inputs: dict[str, Any],
        input_metadata: dict[str, Any] | None = None,
        golden_path: str | None = None,
        golden_inputs: list[str] | None = None,
    ):
        self.name = name
        self.init_args = init_args
        self.inputs = inputs
        self.input_metadata = input_metadata or {}
        self.golden_path = golden_path
        self.golden_inputs = golden_inputs or []

    def __repr__(self) -> str:
        return f"Scenario({self.name!r})"


class InputRegistry:
    """Loads and serves operator input specifications from YAML manifests."""

    def __init__(self, inputs_dir: str | Path | None = None, golden_dir: str | Path | None = None):
        self._inputs_dir = Path(inputs_dir) if inputs_dir else _INPUTS_DIR
        self._golden_dir = Path(golden_dir) if golden_dir else _GOLDEN_DIR
        self._operators: dict[str, list[Scenario]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._inputs_dir.is_dir():
            return
        for yaml_file in sorted(self._inputs_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            for op_name, op_spec in data.items():
                if not isinstance(op_spec, dict) or "scenarios" not in op_spec:
                    continue
                scenarios = []
                for s in op_spec.get("scenarios", []):
                    scenarios.append(Scenario(
                        name=s["name"],
                        init_args=s.get("init_args", {}),
                        inputs=s.get("inputs", {}),
                        input_metadata=s.get("input_metadata", {}),
                        golden_path=s.get("golden"),
                        golden_inputs=s.get("golden_inputs", []),
                    ))
                self._operators.setdefault(op_name, []).extend(scenarios)

    def operators(self) -> list[str]:
        """Return all operator names in the registry."""
        self._load()
        return sorted(self._operators.keys())

    def scenarios(
        self,
        operator: str,
        *,
        models: list[str] | None = None,
        tp: list[int] | None = None,
        category: str | None = None,
    ) -> list[Scenario]:
        """Return scenarios for an operator, optionally filtered.

        Filtering is based on scenario name conventions:
          - model filter: scenario name starts with model key (e.g. "llama31-8b/...")
          - tp filter: scenario name contains "/tpN" (e.g. ".../tp4")
        """
        self._load()
        all_scenarios = self._operators.get(operator, [])
        result = all_scenarios

        if models:
            result = [
                s for s in result
                if any(s.name.startswith(m) for m in models)
            ]

        if tp:
            tp_tags = {f"/tp{t}" for t in tp}
            result = [
                s for s in result
                if any(tag in s.name for tag in tp_tags)
            ]

        return result

    def get_inputs(
        self,
        operator: str,
        scenario_name: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Build input tensors for a specific scenario.

        For shape-only inputs, generates random tensors.
        For golden inputs, loads the .pt file.
        Scalar values (int, float) are passed through as-is.
        """
        self._load()
        scenario = None
        for s in self._operators.get(operator, []):
            if s.name == scenario_name:
                scenario = s
                break
        if scenario is None:
            raise KeyError(
                f"Scenario {scenario_name!r} not found for operator {operator!r}"
            )

        result = self._materialize_shape_inputs(operator, scenario, device)

        if scenario.golden_path:
            golden_file = self._golden_dir / scenario.golden_path
            if not golden_file.is_file():
                raise FileNotFoundError(
                    f"Golden data file not found: {golden_file}\n"
                    f"Run kb_nano capture-golden to generate it, "
                    f"or download from HuggingFace Hub."
                )
            if golden_file.suffix == ".safetensors":
                golden_data = {
                    k: v.to(device)
                    for k, v in load_safetensors(golden_file).items()
                }
                sidecar = golden_file.with_suffix(".json")
                if sidecar.is_file():
                    with open(sidecar) as f:
                        golden_data.update(yaml.safe_load(f) or {})
            else:
                golden_data = torch.load(golden_file, map_location=device, weights_only=True)
            result.update(golden_data)
            return result

        return result

    def _materialize_shape_inputs(self, operator: str, scenario: Scenario, device: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        generated_scale_args: set[str] = set()
        for arg_name, spec in scenario.inputs.items():
            if arg_name in generated_scale_args:
                continue
            if isinstance(spec, dict) and "shape" in spec:
                dtype = _parse_dtype(spec["dtype"])
                shape = spec["shape"]
                quantize = spec.get("quantize")
                if quantize == "fp8":
                    if not _is_fp8_dtype(dtype):
                        raise ValueError(
                            f"Input {arg_name!r} requests quantize=fp8 "
                            f"but dtype is {spec['dtype']!r}"
                        )
                    scale_arg = spec.get("scale_arg")
                    if not scale_arg:
                        raise ValueError(
                            f"Input {arg_name!r} with quantize=fp8 requires scale_arg"
                        )
                    source_dtype = _parse_dtype(spec.get("source_dtype", "bfloat16"))
                    source = torch.randn(shape, dtype=source_dtype, device=device)
                    scale_layout = spec.get("scale_layout", "per_token_group")
                    if scale_layout == "per_token_group":
                        tensor, scale = _quantize_fp8_per_token_group(
                            source,
                            dtype,
                            group_size=int(spec.get("group_size", _FP8_GROUP_SIZE)),
                            use_ue8m0=bool(spec.get("use_ue8m0", True)),
                            eps=float(spec.get("eps", 1e-10)),
                        )
                    elif scale_layout == "per_block":
                        if len(shape) < 2:
                            raise ValueError(
                                f"Input {arg_name!r} needs rank >= 2 for per_block FP8"
                            )
                        tensor, scale = _quantize_fp8_per_block(
                            source,
                            dtype,
                            block_shape=spec.get(
                                "block_shape",
                                [_FP8_GROUP_SIZE, _FP8_GROUP_SIZE],
                            ),
                            use_ue8m0=bool(spec.get("use_ue8m0", True)),
                            eps=float(spec.get("eps", 1e-10)),
                        )
                    else:
                        raise ValueError(
                            f"Unknown FP8 scale_layout {scale_layout!r} for {arg_name!r}"
                        )
                    result[arg_name] = tensor
                    result[scale_arg] = scale
                    generated_scale_args.add(scale_arg)
                elif dtype in (torch.int32, torch.int64):
                    constrained = _constrained_integer_tensor(
                        operator=operator,
                        arg_name=arg_name,
                        shape=shape,
                        dtype=dtype,
                        scenario=scenario,
                        result=result,
                        device=device,
                    )
                    if constrained is not None:
                        result[arg_name] = constrained
                    else:
                        result[arg_name] = torch.randint(0, 100, shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    result[arg_name] = torch.randint(0, 2, shape, dtype=torch.uint8, device=device).bool()
                elif _is_fp8_dtype(dtype):
                    source_dtype = _parse_dtype(spec.get("source_dtype", "bfloat16"))
                    result[arg_name] = torch.randn(
                        shape, dtype=source_dtype, device=device,
                    ).to(dtype)
                else:
                    result[arg_name] = torch.randn(shape, dtype=dtype, device=device)
            elif spec is None:
                result[arg_name] = None
            else:
                result[arg_name] = spec
        return result

    def get_init_args(self, operator: str, scenario_name: str) -> dict[str, Any]:
        """Return init_args for a specific scenario."""
        self._load()
        for s in self._operators.get(operator, []):
            if s.name == scenario_name:
                return dict(s.init_args)
        raise KeyError(
            f"Scenario {scenario_name!r} not found for operator {operator!r}"
        )
