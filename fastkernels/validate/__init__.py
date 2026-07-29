"""Run reference-library validation harnesses through Ray.

The dispatcher accepts both the native ``workloads`` scenario schema and the
legacy ``legacy_workloads`` schema used by the paper validation table. Each
scenario becomes a named Ray task whose GPU requirement matches its TP degree.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from fastkernels import RESULTS_DIR

_VALIDATE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VALIDATE_DIR.parent.parent
_SCENARIO_DIR = _VALIDATE_DIR.parent / "scenarios"
_VALIDATE_ROOT = RESULTS_DIR / "validate"


@dataclass(frozen=True)
class ValidateScenario:
    """Validation-only scenario supporting legacy paper workload names."""

    hf_name: str
    tp: int
    dtype: str
    legacy_workloads: tuple[str, ...]
    enforce_eager: bool = False
    max_num_seqs: int | None = None
    draft_model: str | None = None
    variant: str | None = None
    scene: str | None = None
    reference_checkpoint: str | None = None


_MODULE_TO_HARNESS: dict[str, str] = {
    "llama": "bench_vllm",
    "deepseek": "bench_vllm",
    "mixtral": "bench_vllm",
    "gpt_oss": "bench_vllm",
    "gemma4": "bench_vllm",
    "mamba": "bench_vllm",
    "mamba2": "bench_vllm",
    "qwen3_next": "bench_vllm",
    "kimi_linear": "bench_vllm",
    "qwen2_vl": "bench_vllm",
    "qwen3_vl": "bench_vllm",
    "qwen2_5_omni": "bench_vllm",
    "whisper": "bench_vllm",
    "flux": "bench_vllm_omni",
    "hunyuan_video": "bench_vllm_omni",
    "cosyvoice3": "bench_vllm_omni",
    "sdxl": "bench_diffusers",
    "sam3": "bench_sam",
    "siglip2": "bench_timm",
    "dinov3": "bench_timm",
    "swinv2": "bench_timm",
    "mobilenetv4": "bench_timm",
    "convnextv2": "bench_image_cls",
    "efficientnetv2": "bench_image_cls",
    "yolov10": "bench_detection",
    "rtdetrv2": "bench_detection",
    "bge_m3": "bench_embedding",
    "colbertv2": "bench_embedding",
    "dlrmv2": "bench_recsys",
    "lightgcn": "bench_recsys",
    "gaussian_splatting": "bench_3dgs",
    "instant_ngp": "bench_instantngp",
    "pointtransformerv3": "bench_pointcloud",
    "openfold3": "bench_openfold3",
    "pi0": "bench_openpi",
    "dp3": "bench_dp3",
    "oasis": "bench_oasis",
    "vjepa2": "bench_vjepa2",
    "ttt_e2e": "bench_ttt_e2e",
    "llada": "bench_dllm",
}


def _harness_for(hf_name: str, draft_model: str | None = None) -> str | None:
    n = hf_name.lower()
    # A scenario with a draft model is speculative decoding regardless of what
    # the target is called; only the legacy composite names ("<target> + <draft>
    # (draft)") carry "eagle3" in hf_name itself.
    if draft_model or "eagle3" in n:
        return "bench_sglang"
    if n.startswith("fla-hub/"):
        return "bench_fla"
    if "jamba" in n:
        return "bench_jamba"
    if "bitnet" in n:
        return "bench_microsoft_bitnet"
    if "llada" in n:
        return "bench_dllm"
    if "stable-diffusion" in n or "sdxl" in n:
        return "bench_diffusers"
    from fastkernels.workloads import _module_from_name, module_for

    # Most validation rows are recognizable from the model name. Avoid a
    # transformers config download during planning/dry-run when that local
    # registry match is sufficient.
    module = _module_from_name(hf_name) or module_for(hf_name)
    return _MODULE_TO_HARNESS.get(module) if module else None


_REQUESTS_FLAG = {
    "bench_vllm": "--num-seqs",
    "bench_fla": "--num-seqs",
    "bench_jamba": "--num-seqs",
    "bench_sglang": "--num-seqs",
    "bench_microsoft_bitnet": "--num-prompts",
    "bench_detection": "--num-images",
    "bench_timm": "--num-images",
    "bench_image_cls": "--num-images",
    "bench_sam": "--num-items",
    "bench_dllm": "--max-samples",
    "bench_dp3": "--num-requests",
    "bench_openpi": "--num-requests",
    "bench_openfold3": "--num-seqs",
    "bench_pointcloud": "--max-samples",
    "bench_vjepa2": "--num-videos",
    "bench_ttt_e2e": "--n-sequences",
}
_TP_OK = {
    "bench_vllm",
    "bench_embedding",
    "bench_microsoft_bitnet",
    "bench_sam",
}
_MAXLAYERS_OK = {"bench_vllm"}
# Harnesses that take the optional scenario fields (see BenchmarkScenario):
# `variant` picks which net a multi-variant harness builds, `scene` which scene a
# renderer loads, `reference_checkpoint` where the reference implementation's
# weights live when they are not the same tree as `model`.
_VARIANT_OK = {"bench_dp3", "bench_ttt_e2e"}
_SCENE_OK = {"bench_3dgs", "bench_instantngp"}
_REFERENCE_CHECKPOINT_FLAG = {
    "bench_openpi": "--openpi-checkpoint",
    "bench_dp3": "--checkpoint",
}
_EAGER_OK = {
    "bench_vllm",
    "bench_sglang",
    "bench_diffusers",
    "bench_embedding",
    "bench_openpi",
}
_HF_MODEL_ARG = {
    "bench_vllm",
    "bench_sglang",
    "bench_fla",
    "bench_jamba",
    "bench_microsoft_bitnet",
    "bench_vllm_omni",
    "bench_diffusers",
    "bench_sam",
    "bench_timm",
    "bench_image_cls",
    "bench_detection",
    "bench_embedding",
    "bench_dllm",
    "bench_pointcloud",
    "bench_oasis",
    "bench_openpi",
    "bench_vjepa2",
}


def _safe_slug(text: str, *, max_len: int | None = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    slug = re.sub(r"_+", "_", slug) or "scenario"
    return slug[:max_len] if max_len is not None else slug


def _sanitize_run_id(run_id: str) -> str:
    safe = _safe_slug(run_id, max_len=None)
    if safe == "scenario" and not run_id.strip():
        raise SystemExit("--run-id must contain at least one path-safe character")
    return safe


def _scenario_workloads(scenario) -> tuple[str, ...]:
    legacy = getattr(scenario, "legacy_workloads", None)
    if legacy:
        return tuple(str(w) for w in legacy)
    return tuple(getattr(w, "value", str(w)) for w in scenario.workloads)


def _module_for_scenario(scenario) -> str | None:
    from fastkernels.workloads import module_for

    return module_for(scenario.hf_name)


def _openpi_datasets(scenario) -> list[str]:
    """Datasets named by a Pi0 scenario's workloads, in declaration order.

    Robotics workload values are ``<dataset>-<shape>`` (``aloha-3cam``,
    ``droid-single``, ...), so the set of datasets to benchmark follows from the
    workload list instead of needing its own field.
    """
    known = ("aloha", "droid", "libero")
    ordered: list[str] = []
    for workload in _scenario_workloads(scenario):
        dataset = workload.split("-", 1)[0]
        if dataset in known and dataset not in ordered:
            ordered.append(dataset)
    return ordered


def _model_args(scenario, harness: str) -> list[str]:
    if harness == "bench_sglang":
        draft = getattr(scenario, "draft_model", None)
        if draft:
            return ["--model", scenario.hf_name, "--draft-model", draft]
        # Legacy tables pack both checkpoints into one string:
        # "<target> + <draft> (draft)".
        raw = scenario.hf_name
        if " + " in raw:
            target, draft = raw.split(" + ", 1)
            return [
                "--model",
                target.strip(),
                "--draft-model",
                draft.replace("(draft)", "").strip(),
            ]
        return ["--model", raw]
    if harness == "bench_recsys":
        module = _module_for_scenario(scenario)
        return ["--model", module] if module else []
    if harness in _HF_MODEL_ARG:
        return ["--model", scenario.hf_name]
    return []


def _output_args(harness: str, output_dir: Path | None) -> list[str]:
    if output_dir is None:
        return []
    if harness == "bench_ttt_e2e":
        return ["--results-out", str(output_dir / "results.json")]
    return ["--output-dir", str(output_dir)]


def _build_cmd(
    scenario,
    harness: str,
    args,
    output_dir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        f"fastkernels.validate.{harness}",
        *_model_args(scenario, harness),
    ]
    if harness in _TP_OK and scenario.tp:
        cmd += ["--tp", str(scenario.tp)]
    if getattr(args, "max_layers", None) is not None and harness in _MAXLAYERS_OK:
        cmd += ["--max-layers", str(args.max_layers)]
    if getattr(args, "max_requests", None) is not None:
        flag = _REQUESTS_FLAG.get(harness)
        if flag:
            cmd += [flag, str(args.max_requests)]
    if scenario.enforce_eager and harness in _EAGER_OK:
        cmd.append("--enforce-eager")
    scenario_variant = getattr(scenario, "variant", None)
    if scenario_variant and harness in _VARIANT_OK:
        cmd += ["--variant", scenario_variant]
    scene = getattr(scenario, "scene", None)
    if scene and harness in _SCENE_OK:
        cmd += ["--scene", scene]
    reference_checkpoint = getattr(scenario, "reference_checkpoint", None)
    if reference_checkpoint and harness in _REFERENCE_CHECKPOINT_FLAG:
        cmd += [_REFERENCE_CHECKPOINT_FLAG[harness], reference_checkpoint]
    if harness == "bench_openpi":
        # One dataset per --datasets entry, each with its own checkpoint and
        # camera count; the Robotics workload names carry the dataset prefix.
        datasets = _openpi_datasets(scenario)
        if datasets:
            cmd += ["--datasets", *datasets]
    if harness == "bench_ttt_e2e":
        if not scenario_variant:
            workloads = _scenario_workloads(scenario)
            cmd += [
                "--variant",
                workloads[0]
                if len(workloads) == 1 and workloads[0].endswith("_e2e")
                else "125m_e2e",
            ]
        if output_dir is not None:
            cmd += ["--cache-dir", str(output_dir / "cache")]
    if harness == "bench_vllm" and getattr(args, "vllm_python", None):
        cmd += ["--vllm-python", args.vllm_python]
    cmd += _output_args(harness, output_dir)
    if harness == "bench_vllm" and getattr(args, "resume", False):
        cmd.append("--resume")
    return cmd


def _scenario_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    name = str(name_or_path)
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    packaged = _SCENARIO_DIR / name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(
        f"scenario table {name_or_path!r} not found as a path or in {_SCENARIO_DIR}"
    )


def _load_legacy_validate_scenarios(path: Path) -> list[ValidateScenario] | None:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    entries = data.get("scenarios")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: expected top-level 'scenarios' list")
    if not any(
        isinstance(entry, dict) and "legacy_workloads" in entry
        for entry in entries
    ):
        return None

    allowed_dtypes = {"bfloat16", "float16", "float32", "fp8", "mxfp4"}
    scenarios: list[ValidateScenario] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name}: scenario #{index} is not a mapping")
        try:
            model = str(entry["model"])
            tp = int(entry["tp"])
            dtype = str(entry["dtype"])
            workloads = entry["legacy_workloads"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: bad scenario #{index}: {exc}") from exc
        if tp < 1:
            raise ValueError(f"{path.name}: {model}: tp must be >= 1")
        if dtype not in allowed_dtypes:
            raise ValueError(
                f"{path.name}: {model}: dtype {dtype!r} not in "
                f"{sorted(allowed_dtypes)}"
            )
        if not isinstance(workloads, list) or not workloads:
            raise ValueError(
                f"{path.name}: {model}: legacy_workloads must be a non-empty list"
            )
        scenarios.append(
            ValidateScenario(
                hf_name=model,
                tp=tp,
                dtype=dtype,
                legacy_workloads=tuple(str(w) for w in workloads),
                enforce_eager=bool(entry.get("enforce_eager", False)),
                max_num_seqs=entry.get("max_num_seqs"),
                draft_model=entry.get("draft_model"),
                variant=entry.get("variant"),
                scene=entry.get("scene"),
                reference_checkpoint=entry.get("reference_checkpoint"),
            )
        )
    return scenarios


def _resolve_validate_scenarios(name_or_path: str | Path):
    path = _scenario_path(name_or_path)
    legacy = _load_legacy_validate_scenarios(path)
    if legacy is not None:
        return legacy
    from fastkernels.workloads import resolve_benchmark

    return resolve_benchmark(path)


def _detect_gpus(explicit: str | None) -> list[str]:
    if explicit:
        return [token.strip() for token in explicit.split(",") if token.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [token.strip() for token in visible.split(",") if token.strip()]
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            output = subprocess.run(
                [smi, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            ids = [line.strip() for line in output.splitlines() if line.strip()]
            if ids:
                return ids
        except Exception:
            pass
    return ["0"]


def _resolve_run_root(args) -> tuple[str, Path]:
    if args.output_dir:
        root = Path(args.output_dir)
        return root.name, root
    if args.run_id:
        run_id = _sanitize_run_id(args.run_id)
    elif args.resume:
        existing = (
            [path for path in _VALIDATE_ROOT.iterdir() if path.is_dir()]
            if _VALIDATE_ROOT.exists()
            else []
        )
        if not existing:
            raise ValueError(f"--resume given but no runs found under {_VALIDATE_ROOT}")
        run_id = max(existing, key=lambda path: path.stat().st_mtime).name
        print(f"  --resume: continuing most recent run {run_id!r}")
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    return run_id, _VALIDATE_ROOT / run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastkernels validate",
        description="Run each validation scenario as a named, GPU-aware Ray task.",
    )
    parser.add_argument(
        "scenarios",
        help="Scenario path or packaged name. Both workloads and "
        "legacy_workloads tables are supported.",
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=int(os.environ.get("FASTKERNELS_VALIDATE_STALL_SEC", "900")),
        help="Kill a task whose log has not changed for this many seconds.",
    )
    parser.add_argument(
        "--numactl-mode",
        choices=("off", "cpu", "strict"),
        default="cpu",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Existing Ray cluster address. Default starts a local Ray runtime.",
    )
    parser.add_argument(
        "--vllm-python",
        default=None,
        help="Optional interpreter for bench_vllm's reference worker.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        scenarios = _resolve_validate_scenarios(args.scenarios)
        run_id, run_root = _resolve_run_root(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"  run id : {run_id}")
    print(f"  outputs: {run_root}{'  (resume)' if args.resume else ''}")

    # Provision the reference libraries / task kernels the selected scenarios
    # need before dispatching (git clones, pip extras, source builds). This is
    # automatic and unconditional; provision() skips any component whose check()
    # already passes, so it is a no-op once the references are installed. Only
    # --dry-run bypasses it (below), since a dry run installs nothing.
    #
    # provision() never raises: it catches per component and returns a failure
    # count, so one unbuildable reference cannot kill the others. A non-zero count
    # is a warning here, not a run-ending error.
    if not args.dry_run:
        from .provision import components_for_harnesses, provision

        harnesses = {
            h
            for h in (
                _harness_for(s.hf_name, getattr(s, "draft_model", None))
                for s in scenarios
            )
            if h is not None
        }
        components = components_for_harnesses(harnesses)
        if components:
            print(f"  provision: checking {', '.join(components)}")
            failed = provision(components)
            if failed:
                # Do not abort the sweep: a reference library that will not build
                # (no `uv`, no network, a source build that needs a toolchain)
                # should not stop the scenarios that do not depend on it. The
                # scenarios that DO depend on it still get dispatched and fail on
                # their own missing dependency, which is recorded per scenario in
                # the run summary rather than losing the whole run.
                print(
                    f"WARNING: provisioning failed for {failed} component(s); "
                    f"scenarios needing them will run anyway and are expected to "
                    f"fail with their own missing-dependency error. See the "
                    f"[provision] lines above for what to fix.",
                    file=sys.stderr,
                )

    if args.dry_run:
        for index, scenario in enumerate(scenarios):
            harness = _harness_for(
                scenario.hf_name, getattr(scenario, "draft_model", None)
            )
            if harness is None:
                print(f"[{index}] {scenario.hf_name}: NO HARNESS MAPPED")
                continue
            output_dir = run_root / _safe_slug(
                f"{index:03d}_{harness}_{scenario.hf_name}"
            )
            cmd = _build_cmd(scenario, harness, args, output_dir)
            print(
                f"[{index}] {scenario.hf_name} "
                f"(tp={scenario.tp}, dtype={scenario.dtype}) -> {harness}"
            )
            print("      workloads: " + ", ".join(_scenario_workloads(scenario)))
            print("      " + " ".join(cmd))
        return 0

    from .ray_runner import run_validation

    return run_validation(scenarios, args, _detect_gpus(args.gpus), run_root)


if __name__ == "__main__":
    raise SystemExit(main())
