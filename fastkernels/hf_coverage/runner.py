"""Shared workload measurement and the isolated FastKernels worker.

Imports remain lightweight: the pinned HF worker imports Workload and measurement
helpers from here without importing FastKernels model or kernel implementations.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Any


def _noop() -> None:
    pass


def _identity(value):
    return value


@dataclass
class Workload:
    """Reset state, execute timed work, then collect outputs outside timing.

    collect may expose an existing cache in logical order. It must not perform
    model computation or repair outputs that should have been produced by run.
    """

    run: Callable[[], dict[str, Any]]
    prepare: Callable[[], None] = _noop
    collect: Callable[[dict], dict] = _identity


def seq2seq_cache_outputs(layers) -> dict:
    """Expose the self/cross K/V tensors produced by a full decoder forward."""
    outputs = {}
    for index, (self_attention, cross_attention) in enumerate(layers):
        for kind, (key, value) in (("self", self_attention), ("cross", cross_attention)):
            outputs[f"past_key_values.{index}.{kind}.key"] = key
            outputs[f"past_key_values.{index}.{kind}.value"] = value
    return outputs


def seq2seq_continuation_workloads(model, inputs, *, full_decoder_history=False,
                                 output_names=("logits", "encoder_last_hidden_state"),
                                 encoder_input_name="input_ids"):
    """Prefill and two teacher-forced tokens, retaining the complete cache."""
    decoder_ids = inputs["decoder_input_ids"]
    if decoder_ids.ndim != 2 or decoder_ids.shape[1] < 3:
        raise ValueError("Encoder-decoder continuation requires a decoder prefix and two new tokens")
    prefix_length = decoder_ids.shape[1] - 2
    state = {}

    def call(start, end, previous=None):
        return model(
            inputs[encoder_input_name], decoder_ids[:, 0 if full_decoder_history else start:end],
            encoder_hidden_states=None if previous is None else previous["encoder_last_hidden_state"],
            past_key_values=None if previous is None else previous["past_key_values"],
            attention_mask=inputs.get("attention_mask"),
            decoder_attention_mask=(None if "decoder_attention_mask" not in inputs
                                    else inputs["decoder_attention_mask"][:, :end]),
        )
    def retain(output):
        state["output"] = output
        return {"logits": output["logits"]}

    def collect(output):
        output = state.pop("output")
        result = {name: output[name] for name in output_names}
        result.update(seq2seq_cache_outputs(output["past_key_values"]))
        return result

    def prepare_first():
        state["previous"] = call(0, prefix_length)

    def prepare_second():
        prepare_first()
        state["previous"] = call(prefix_length, prefix_length + 1, state["previous"])

    return {
        "prefill": Workload(run=lambda: retain(call(0, prefix_length)), collect=collect),
        "decode_1": Workload(
            run=lambda: retain(call(prefix_length, prefix_length + 1, state["previous"])),
            prepare=prepare_first, collect=collect,
        ),
        "decode_2": Workload(
            run=lambda: retain(call(prefix_length + 1, prefix_length + 2, state["previous"])),
            prepare=prepare_second, collect=collect,
        ),
    }


class Config(dict):
    """Resolved HF configuration values with mapping and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self):
        return json.loads(json.dumps(self))


def config_values(value):
    if isinstance(value, dict):
        return Config({k: config_values(v) for k, v in value.items()})
    if isinstance(value, list):
        return [config_values(v) for v in value]
    return value


def symbol(path: str):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")


def git_value(directory: Path, *args) -> str | None:
    result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def versions() -> dict:
    result = {"python": sys.version, "executable": sys.executable}
    for package in ("torch", "transformers", "vllm", "flash-attn", "flashinfer-python", "triton",
                    "kernels", "accelerate", "xformers", "flash-linear-attention",
                    "causal-conv1d", "deep-gemm"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    result["transformers_distribution"] = result.pop("transformers")
    module = sys.modules.get("transformers")
    result["transformers"] = getattr(module, "__version__", None)
    result["transformers_path"] = getattr(module, "__file__", None)
    return result


def configure_torch(seed: int, dtype: str, *, gpu: bool, cudnn_deterministic: bool = False) -> None:
    import torch

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(getattr(torch, dtype))
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("this workload requires an available CUDA GPU")
        torch.cuda.set_device(0)


def to_device(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def measure(workloads: dict[str, Workload], *, warmup: int, iterations: int,
            generation_seed: int | None = None) -> tuple[dict, dict]:
    """Measure synchronized wall latency, including dispatch, without setup/copies.

    Both workers use this exact function. State preparation precedes the starting
    synchronization. Each retained output is copied only after its run completes.
    """
    import torch

    if not workloads:
        raise ValueError("model supplied no workloads")

    def prepare(workload):
        workload.prepare()
        if generation_seed is not None:
            # Constructors may consume different random draws. Reset only the
            # declared stochastic workload; sampling itself remains timed.
            torch.manual_seed(generation_seed)

    results, outputs = {}, {}
    with torch.inference_mode():
        for name, workload in workloads.items():
            prepare(workload)
            torch.cuda.synchronize()
            sample = workload.run()
            torch.cuda.synchronize()
            if not isinstance(sample, dict) or not sample:
                raise TypeError(f"{name}: run must return a nonempty mapping of GPU tensors")
            for key, tensor in sample.items():
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    raise TypeError(f"{name}/{key}: output must be a GPU tensor")
            del sample
            for _ in range(warmup):
                prepare(workload)
                workload.run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for iteration in range(iterations):
                prepare(workload)
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                output = workload.run()
                torch.cuda.synchronize()
                samples.append((time.perf_counter_ns() - start) / 1e6)
                if iteration + 1 < iterations:
                    del output
            # These are the outputs of the final measured execution, retaining
            # their native dtype; serialization occurs after the timing boundary.
            output = workload.collect(output)
            outputs[name] = {key: tensor.detach().cpu().clone() for key, tensor in output.items()}
            del output
            results[name] = {
                "latency_ms": {"median": statistics.median(samples), "min": min(samples),
                               "max": max(samples), "samples": samples},
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "memory_scope": "whole live process, including workload preparation",
                "outputs": {key: {"shape": list(t.shape), "dtype": str(t.dtype)}
                            for key, t in outputs[name].items()},
            }
    return results, outputs


def compare_outputs(actual, reference, *, allow_infinite_outputs=()) -> tuple[bool, dict, dict]:
    """Reuse the current benchmark's numerical comparator; enforce named outputs."""
    import torch
    from fastkernels.bench import _compare_tensor, _TOLERANCES, REQUIRED_MATCHED_RATIO

    from fastkernels import bench

    policy = {
        "name": "fastkernels.bench._compare_tensor", "provisional": True,
        "source_sha256": digest(Path(bench.__file__)),
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "tolerances": {str(k): {"atol": v[0], "rtol": v[1]} for k, v in _TOLERANCES.items()},
        "integer_rule": "exact equality", "dtype_rule": "matching output dtypes required",
        "allowed_infinity_outputs": list(allow_infinite_outputs),
    }
    if set(actual) != set(reference):
        return False, {"error": "workload names differ", "actual": list(actual), "reference": list(reference)}, policy
    results, all_ok = {}, True
    nondegenerate = False
    for workload, ref_outputs in reference.items():
        if any(ref.numel() and bool(torch.any(ref != 0)) for ref in ref_outputs.values()):
            nondegenerate = True
        candidate = actual[workload]
        if set(candidate) != set(ref_outputs):
            results[workload] = {"passed": False, "error": "output names differ"}
            all_ok = False
            continue
        rows = {}
        for name, ref in ref_outputs.items():
            out = candidate[name]
            row = {"actual_dtype": str(out.dtype), "reference_dtype": str(ref.dtype),
                   "actual_shape": list(out.shape), "reference_shape": list(ref.shape)}
            if out.shape != ref.shape:
                row.update(passed=False, error="shape mismatch")
            elif not ref.is_floating_point():
                row.update(passed=out.dtype == ref.dtype and torch.equal(out, ref), rule="exact")
            else:
                if name in allow_infinite_outputs:
                    # Some public outputs use signed infinities as explicit invalid
                    # markers. Compare their locations exactly, then use the
                    # unchanged benchmark comparator on the finite entries.
                    markers_match = (
                        not torch.isnan(out).any() and not torch.isnan(ref).any()
                        and torch.equal(torch.isposinf(out), torch.isposinf(ref))
                        and torch.equal(torch.isneginf(out), torch.isneginf(ref))
                    )
                    row["infinity_markers_match"] = bool(markers_match)
                    if not markers_match:
                        row.update(passed=False, error="nonfinite markers differ or NaN present")
                        rows[name] = row
                        all_ok = False
                        continue
                    finite = torch.isfinite(ref)
                    out, ref = out[finite], ref[finite]
                ok, absolute, relative, matched, detail = _compare_tensor(out, ref)
                row.update(passed=ok and out.dtype == ref.dtype, max_absolute_error=absolute,
                           max_relative_error=relative, matched_ratio=matched, detail=detail)
                if out.dtype != ref.dtype:
                    row["error"] = "dtype mismatch"
            rows[name] = row
            all_ok = all_ok and row["passed"]
        results[workload] = rows
    if not nondegenerate:
        results["error"] = "all reference outputs are empty or zero; no acceptance evidence"
        all_ok = False
    return all_ok, results, policy


def worker_metadata() -> dict:
    import torch

    return {"versions": versions(), "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda": torch.version.cuda,
            "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
            "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
            "sdpa_enabled_backends": {
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            },
            "kernel_internal_precision": "not controlled by the PyTorch TF32 flags",
            "outer_model_execution": "eager Python calls; runner does not apply torch.compile or CUDA graph capture",
            "operation_compilation": "each selected operation retains its own internal compilation/JIT behavior",
            "runtime_default_dtype": str(torch.get_default_dtype()),
            "timing": "synchronized wall-clock; preparation and output copies excluded"}


def timing_boundaries(case: dict) -> dict:
    result = {
        "included": "run callable, Python dispatch, its allocations and kernels, and synchronization to completion",
        "excluded": "model construction, weight loading, input transfer, workload preparation, and output copies/serialization",
        "memory": "whole live worker process including preparation; not incremental timed-call allocation",
    }
    if case["workload"] in ("causal_lm_continuation", "seq2seq_continuation", "memory_continuation"):
        result["continuation"] = {
            "steps": 2,
            "preparation": "rebuild prefix and any preceding continuation outside each timed continuation",
            "timed": "one initial forward or one new-token continuation, including its cache update",
            "collection": "logical cache export and CPU output copies excluded on both sides",
        }
    elif case["workload"] == "causal_lm":
        prompt_length = case["input"]["sequence_length"] - 1
        result["prefill"] = {
            "tokens_per_sequence": prompt_length,
            "reference_cache": "default HF cache creation/growth occurs inside the timed prefill call",
            "implementation_cache": "allocations/resets in construction or prepare are excluded; work inside run is included",
        }
        result["decode"] = {
            "tokens_per_sequence": 1,
            "prefix_tokens_per_sequence": prompt_length,
            "prefix_preparation": "both implementations build the same prompt state outside the timed decode call",
        }
        result["interpretation"] = "API execution comparison with these setup boundaries, not isolated identical kernel work"
    return result


def implementation_worker(job: dict, directory: Path) -> dict:
    import torch
    import torch.distributed as dist

    configure_torch(job["seed"], "float32", gpu=True,
                    cudnn_deterministic=job["case"].get("cudnn_deterministic", False))
    prepared = torch.load(directory / "prepared.pt", map_location="cpu", weights_only=True)
    config = config_values(prepared["config"])
    dist.init_process_group("nccl", init_method=(directory / "distributed.init").as_uri(),
                            rank=0, world_size=1, device_id=torch.device("cuda:0"))
    try:
        module = importlib.import_module(f"fastkernels.hf_coverage.models.{job['model']}")
        torch.set_default_dtype(getattr(torch, job["dtype"]))
        try:
            with torch.device("cuda:0"):
                model = module.build_from_config(
                    config, torch.device("cuda:0"), getattr(torch, job["dtype"]),
                    **job["case"].get("implementation_kwargs", {}),
                )
        finally:
            torch.set_default_dtype(torch.float32)
        # As in LlamaEngine, sequential attention calls can share temporary
        # TRTLLM storage. Release per-layer buffers before copying the weights.
        attention_layers = [layer for layer in model.modules()
                            if getattr(layer, "_use_trtllm", False)
                            and callable(getattr(layer, "set_trtllm_workspace", None))]
        if attention_layers:
            workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
            for layer in attention_layers:
                layer.set_trtllm_workspace(workspace)
            torch.cuda.empty_cache()
        module.load_state_dict_into(model, to_device(prepared["weights"], "cuda:0"), config)
        model.eval()
        inputs = to_device(prepared["inputs"], "cuda:0")
        if job["case"]["workload"] in ("generate", "causal_lm_continuation", "seq2seq_continuation", "memory_continuation"):
            workloads = module.make_workloads(model, inputs, config, case=job["case"])
        else:
            workloads = module.make_workloads(model, inputs, config)
        measurements, outputs = measure(workloads, warmup=job["warmup"], iterations=job["iterations"],
                                        generation_seed=job["case"].get("generation_seed"))
        torch.save(outputs, directory / "implementation_outputs.pt")
        return {"status": "completed", **worker_metadata(), "workloads": measurements}
    finally:
        dist.destroy_process_group()


def guarded_worker(job_path: Path, phase: str, function: Callable) -> int:
    directory = job_path.parent
    try:
        job = json.loads(job_path.read_text())
        result = function(job, directory)
        status = 0
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(), "versions": versions()}
        status = 1
    write_json(directory / f"{phase}.json", result)
    return status


def select_case(model: str, variant: str | None = None) -> dict:
    """Select a declared workload without counting it as another architecture."""
    from .cases import CASES

    case = CASES[model]
    if "variants" in case:
        if variant not in case["variants"]:
            raise ValueError(f"{model} requires --variant from {', '.join(case['variants'])}")
        case = case["variants"][variant]
    elif variant is not None:
        raise ValueError(f"{model} does not declare workload variants")
    return case


def run_case(args) -> int:
    case = json.loads(json.dumps(select_case(args.model, getattr(args, "variant", None))))
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    corpus = json.loads((package / "corpus.json").read_text())
    if args.model not in corpus["models"]:
        raise ValueError(f"{args.model!r} is absent from the pinned corpus")
    directory = Path(args.output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("output directory must be empty to prevent mixing or overwriting runs")
    if args.ref_attn is not None:
        case["reference_backend"] = args.ref_attn
    hf_source = None
    if args.hf_source:
        source = Path(args.hf_source).expanduser().resolve()
        hf_source = source / "src" if (source / "src" / "transformers").is_dir() else source
        if not (hf_source / "transformers" / "__init__.py").is_file():
            raise ValueError("--hf-source must contain transformers/ or src/transformers/")
    job = {"schema_version": 1, "model": args.model, "case": case,
           "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
           "transformers_revision": corpus["transformers_revision"], "dtype": args.dtype,
           "seed": args.seed, "warmup": args.warmup, "iterations": args.iterations,
           "reference_source": str(hf_source) if hf_source else None,
           "state_dict": str(Path(args.state_dict).expanduser().resolve()) if args.state_dict else None,
           "input_dict": str(Path(args.input_dict).expanduser().resolve()) if getattr(args, "input_dict", None) else None,
           "upcast_from": str(Path(args.upcast_from).expanduser().resolve()) if args.upcast_from else None,
           "reuse_from": str(Path(args.reuse_from).expanduser().resolve()) if getattr(args, "reuse_from", None) else None}
    if getattr(args, "variant", None) is not None:
        job["variant"] = args.variant
    write_json(directory / "job.json", job)
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "job": job, "fastkernels_commit": git_value(repository, "rev-parse", "HEAD"),
              "timing_boundaries": timing_boundaries(case),
              "working_tree_status": git_value(repository, "status", "--short"),
              "audit_sources": {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*"))
                                if p.is_file() and p.suffix in {".py", ".cu", ".cpp", ".json", ".md"}},
              "workers": {}}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FASTKERNELS_ROOT"] = str(package.parent)
    phases = [("prepare", args.hf_python, "fastkernels.hf_coverage.reference"),
              ("reference", args.hf_python, "fastkernels.hf_coverage.reference"),
              ("implementation", sys.executable, "fastkernels.hf_coverage.runner")]
    try:
        for phase, python, module in phases:
            command = [str(python), "-m", module, phase, str(directory / "job.json")]
            worker_env = env.copy()
            if phase != "implementation" and hf_source:
                worker_env["PYTHONPATH"] = str(hf_source) + os.pathsep + env["PYTHONPATH"]
            with (directory / f"{phase}.log").open("w") as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=worker_env, cwd=repository, timeout=args.worker_timeout)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    returncode = -1
            result_path = directory / f"{phase}.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {
                "status": "error", "error": "worker exited or timed out without a result"}
            result.update(command=command, returncode=returncode)
            if returncode != 0:
                result["status"] = "error"
            report["workers"][phase] = result
            write_json(directory / "result.json", report)
            if phase == "prepare" and result["status"] != "completed":
                raise RuntimeError("input/reference preparation failed; see prepare.log")
        if any(r["status"] != "completed" for r in report["workers"].values()):
            raise RuntimeError("one or more execution workers failed; see worker logs")
        import torch

        actual = torch.load(directory / "implementation_outputs.pt", map_location="cpu", weights_only=True)
        reference = torch.load(directory / "reference_outputs.pt", map_location="cpu", weights_only=True)
        passed, comparison, policy = compare_outputs(
            actual, reference, allow_infinite_outputs=case.get("allow_infinite_outputs", ()),
        )
        report.update(status="passed_provisional" if passed else "mismatch", comparison=comparison, policy=policy)
        report["speedups"] = {
            name: report["workers"]["reference"]["workloads"][name]["latency_ms"]["median"] /
                  measurements["latency_ms"]["median"]
            for name, measurements in report["workers"]["implementation"]["workloads"].items()
            if name in report["workers"]["reference"]["workloads"]}
        report["performance_interpretation"] = "speedups describe measured execution; only valid when correctness is accepted"
        report["prepared_sha256"] = digest(directory / "prepared.pt")
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "result.json", report)
    print(f"{args.model}: {report['status']} ({directory / 'result.json'})")
    return 0 if report["status"] == "passed_provisional" else 1


if __name__ == "__main__":
    phase, path = sys.argv[1:]
    if phase != "implementation":
        raise SystemExit("runner accepts only the implementation worker phase")
    raise SystemExit(guarded_worker(Path(path), phase, implementation_worker))
