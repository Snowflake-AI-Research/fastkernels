"""Schema helpers for workload-derived input registry traces.

The raw trace is fastkernels-specific and intentionally richer than the final
registry.  The exported workload trace mirrors the FlashInfer Trace shape:
``definition``, embedded ``workload``, and nullable ``solution``/``evaluation``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


DATA_DEPENDENT_OPS = {
    "fused_experts",
    "moe_align",
    "moe_grouped_gemm",
    "store_kvcache",
}

DATA_DEPENDENT_INPUTS = {
    # KV-cache stores branch/index by slot only; key/value/cache tensors can be
    # regenerated from shape metadata.
    "store_kvcache": {"slot_mapping"},
    # MoE alignment/routing kernels need real routing indices. Value tensors and
    # routed weights are not control-flow inputs for the benchmark harness.
    "moe_align": {"topk_ids"},
    "fused_experts": {"topk_ids"},
    # Grouped GEMM consumes the already-built routing layout.
    "moe_grouped_gemm": {
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
    },
}


@dataclass(frozen=True)
class TensorMeta:
    shape: list[int]
    dtype: str
    ndim: int
    numel: int
    stride: list[int]
    layout: str
    requires_grad: bool = False


@dataclass
class TraceEvent:
    op: str
    model_key: str
    model: str
    tp: int
    dtype: str
    workload: str
    module_path: str
    module_class: str
    occurrence: int
    first_occurrence: bool
    inputs: dict[str, Any]
    init_args: dict[str, Any] = field(default_factory=dict)
    outputs: Any | None = None
    golden_path: str | None = None

    def canonical_key(self) -> str:
        payload = {
            "op": self.op,
            "inputs": _canonical_for_key(self.inputs),
            "init_args": _canonical_for_key(self.init_args),
        }
        return stable_hash(payload)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["signature"] = self.canonical_key()
        return json.dumps(payload, sort_keys=True)


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def tensor_meta(tensor: torch.Tensor) -> TensorMeta:
    return TensorMeta(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        ndim=tensor.ndim,
        numel=tensor.numel(),
        stride=list(tensor.stride()),
        layout=str(tensor.layout).replace("torch.", ""),
        requires_grad=bool(tensor.requires_grad),
    )


def summarize_value(value: Any, *, scalar_limit: int = 16) -> Any:
    """Return JSON-serializable metadata for an input/output value."""
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", **asdict(tensor_meta(value))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, (list, tuple)):
        if len(value) <= scalar_limit and all(
            isinstance(v, (str, int, float, bool)) or v is None for v in value
        ):
            return {"kind": "sequence", "value": list(value)}
        return {
            "kind": "sequence",
            "length": len(value),
            "items": [summarize_value(v) for v in value[:scalar_limit]],
            "truncated": len(value) > scalar_limit,
        }
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": {str(k): summarize_value(v) for k, v in sorted(value.items())},
        }
    return {"kind": "object", "type": type(value).__name__, "repr": repr(value)[:200]}


def _canonical_for_key(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            return {
                "kind": "tensor",
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
                "stride": value.get("stride"),
                "layout": value.get("layout"),
            }
        if value.get("kind") in {"scalar", "sequence"}:
            return value
        return {str(k): _canonical_for_key(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_for_key(v) for v in value]
    return value


def flatten_named_values(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten tensors/scalars into names suitable for safetensors sidecars."""
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_named_values(item, name))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            name = f"{prefix}.{idx}" if prefix else str(idx)
            result.update(flatten_named_values(item, name))
    else:
        result[prefix] = value
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
