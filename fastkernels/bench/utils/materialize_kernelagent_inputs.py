#!/usr/bin/env python3
"""Materialize KernelAgent fairness inputs into golden .pt files.

This freezes the InputRegistry scenarios used for the KernelAgent fairness
benchmark so reruns are deterministic and auditable. The generated YAML can be
used by setting ``FASTKERNELS_INPUTS_DIR`` to the output directory and
``FASTKERNELS_GOLDEN_DIR`` to the generated golden directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from fastkernels import GOLDEN_DIR, INPUTS_DIR
from fastkernels.bench.utils.input_registry import InputRegistry


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def materialize(
    source_inputs_dir: Path,
    output_inputs_dir: Path,
    output_golden_dir: Path,
    *,
    seed: int,
    device: str,
    operators: set[str] | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    output_inputs_dir.mkdir(parents=True, exist_ok=True)
    output_golden_dir.mkdir(parents=True, exist_ok=True)

    registry = InputRegistry(inputs_dir=source_inputs_dir, golden_dir=GOLDEN_DIR)
    registry._load()  # Keep this utility close to the registry's manifest view.

    yaml_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    manifest: dict[str, Any] = {
        "source_inputs_dir": str(source_inputs_dir),
        "output_inputs_dir": str(output_inputs_dir),
        "output_golden_dir": str(output_golden_dir),
        "seed": seed,
        "device": device,
        "operators": {},
        "skipped": [],
    }

    for op_name in registry.operators():
        if operators is not None and op_name not in operators:
            continue
        scenarios = registry.scenarios(op_name)
        if not scenarios:
            continue
        yaml_scenarios = []
        manifest["operators"][op_name] = []
        for scenario in scenarios:
            rel = Path("kernelagent_materialized") / op_name / f"{_safe_name(scenario.name)}.pt"
            out_path = output_golden_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                inputs = registry.get_inputs(op_name, scenario.name, device=device)
            except FileNotFoundError as exc:
                manifest["skipped"].append({
                    "operator": op_name,
                    "scenario": scenario.name,
                    "reason": str(exc),
                })
                continue
            if op_name == "store_kvcache" and "slot_mapping" in inputs:
                n = int(inputs["slot_mapping"].numel())
                inputs["slot_mapping"] = torch.arange(
                    n,
                    dtype=inputs["slot_mapping"].dtype,
                    device=inputs["slot_mapping"].device,
                )
            torch.save(_to_cpu(inputs), out_path)
            yaml_scenarios.append({
                "name": scenario.name,
                "init_args": dict(scenario.init_args),
                "golden": str(rel),
            })
            manifest["operators"][op_name].append({
                "scenario": scenario.name,
                "golden": str(out_path),
                "source_kind": "materialized_registry_workload_shape",
            })
        if yaml_scenarios:
            yaml_data[op_name] = {"scenarios": yaml_scenarios}

    yaml_path = output_inputs_dir / "kernelagent_materialized.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False))
    manifest_path = output_inputs_dir / "kernelagent_materialized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "yaml": str(yaml_path),
        "manifest": str(manifest_path),
        "operators": len(yaml_data),
        "scenarios": sum(len(v["scenarios"]) for v in yaml_data.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inputs-dir", type=Path, default=INPUTS_DIR)
    parser.add_argument("--output-inputs-dir", type=Path, required=True)
    parser.add_argument("--output-golden-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--operators", nargs="*", default=None)
    args = parser.parse_args()

    result = materialize(
        args.source_inputs_dir,
        args.output_inputs_dir,
        args.output_golden_dir,
        seed=args.seed,
        device=args.device,
        operators=set(args.operators) if args.operators else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
