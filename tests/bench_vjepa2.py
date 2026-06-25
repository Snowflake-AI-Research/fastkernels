#!/usr/bin/env python3
"""
Throughput, latency, and alignment benchmark: fastkernels V-JEPA 2 vs transformers.

The default path benchmarks the predictive world-model forward:
  - encoder output alignment
  - masked-context alignment
  - predictor output alignment

Usage:
    python tests/bench_vjepa2.py --model facebook/vjepa2-vitl-fpc64-256
    python tests/bench_vjepa2.py --skip-reference
    python tests/bench_vjepa2.py --skip-latency
    python tests/bench_vjepa2.py --task encoder
    python tests/bench_vjepa2.py --task classification --model facebook/vjepa2-vitl-fpc16-256-ssv2
    python tests/bench_vjepa2.py --dataset nateraw/kinetics-mini --dataset-split validation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent


def _bootstrap_local_package() -> None:
    existing = sys.modules.get("fastkernels")
    expected = str(_PACKAGE_DIR / "__init__.py")
    if existing is not None and getattr(existing, "__file__", None) == expected:
        return
    spec = importlib.util.spec_from_file_location(
        "fastkernels",
        _PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(_PACKAGE_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastkernels"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_bootstrap_local_package()

from fastkernels.bench.utils.worker import run_worker


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _default_dtype(device: str) -> str:
    return "bf16" if device.startswith("cuda") else "fp32"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).clamp(-1.0, 1.0).item()


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _load_tensor_bundle(path: str) -> dict[str, torch.Tensor]:
    bundle = torch.load(path, map_location="cpu")
    return {k: v for k, v in bundle.items() if isinstance(v, torch.Tensor)}


VJEPA2_WORKER = r'''
import importlib.util
import json
import os
import sys
import time
import warnings

import torch
from tqdm import tqdm


def _bootstrap_local_package(package_root: str) -> None:
    spec = importlib.util.spec_from_file_location(
        "fastkernels",
        os.path.join(package_root, "__init__.py"),
        submodule_search_locations=[package_root],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastkernels"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_masks(
    batch_size: int,
    num_patches: int,
    context_ratio: float,
    target_ratio: float,
    seed: int,
    device: str,
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    context_count = max(1, min(num_patches - 1, int(num_patches * context_ratio)))
    target_count = max(1, int(num_patches * target_ratio))
    target_count = min(target_count, num_patches - context_count)

    base_perm = torch.randperm(num_patches, generator=gen)
    context_ids = base_perm[:context_count].sort().values.unsqueeze(0).repeat(batch_size, 1).to(device)
    target_ids = base_perm[-target_count:].sort().values.unsqueeze(0).repeat(batch_size, 1).to(device)
    return [context_ids], [target_ids]


def _load_model(cfg):
    device = cfg["device"]
    dtype = _resolve_dtype(cfg["dtype"])
    task = cfg["task"]

    if cfg["backend"] == "reference":
        from transformers import VJEPA2ForVideoClassification, VJEPA2Model

        cls = VJEPA2ForVideoClassification if task == "classification" else VJEPA2Model
        model = cls.from_pretrained(cfg["model"], dtype=dtype).to(device).eval()
        return model, model.config, dtype

    _bootstrap_local_package(cfg["package_root"])
    if task == "classification":
        from fastkernels.tasks.baseline.L4.vjepa2 import VJEPA2ForVideoClassification as ModelCls
    else:
        from fastkernels.tasks.baseline.L4.vjepa2 import VJEPA2Model as ModelCls
    model = ModelCls.from_pretrained(cfg["model"]).to(device=device, dtype=dtype).eval()
    return model, model.config, dtype


def _forward(task, model, videos, context_mask, target_mask):
    with torch.inference_mode():
        if task == "classification":
            outputs = model(pixel_values_videos=videos)
            return {"logits": outputs.logits.detach().cpu().float()}

        if task == "encoder":
            outputs = model(pixel_values_videos=videos, skip_predictor=True)
            return {"last_hidden_state": outputs.last_hidden_state.detach().cpu().float()}

        outputs = model(
            pixel_values_videos=videos,
            context_mask=context_mask,
            target_mask=target_mask,
            skip_predictor=False,
        )
        return {
            "last_hidden_state": outputs.last_hidden_state.detach().cpu().float(),
            "masked_hidden_state": outputs.masked_hidden_state.detach().cpu().float(),
            "predictor_hidden_state": outputs.predictor_output.last_hidden_state.detach().cpu().float(),
            "target_hidden_state": outputs.predictor_output.target_hidden_state.detach().cpu().float(),
        }


def _make_synthetic_videos(total_videos, config, dtype, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (total_videos, config.frames_per_clip, 3, config.crop_size, config.crop_size),
        generator=gen,
        device=device,
        dtype=dtype,
    )


def _sample_frame_indices(num_frames, frames_per_clip):
    if num_frames <= 0:
        raise ValueError("decoded video has zero frames")
    if num_frames == 1:
        return torch.zeros(frames_per_clip, dtype=torch.long)
    return torch.linspace(0, num_frames - 1, steps=frames_per_clip).round().long()


def _load_video_paths(dataset_name, dataset_split, num_needed, seed):
    from datasets import Video, load_dataset

    print(
        f"Loading {num_needed} videos from {dataset_name} ({dataset_split})...",
        file=sys.stderr, flush=True,
    )
    dataset = load_dataset(dataset_name, split=dataset_split).cast_column("video", Video(decode=False))
    dataset = dataset.shuffle(seed=seed)

    paths = []
    for sample in dataset:
        video = sample.get("video")
        if isinstance(video, dict) and video.get("path"):
            paths.append(video["path"])
        if len(paths) >= num_needed:
            break

    if not paths:
        raise RuntimeError(
            f"No video paths found in dataset {dataset_name} ({dataset_split})."
        )

    if len(paths) < num_needed:
        repeats = (num_needed + len(paths) - 1) // len(paths)
        paths = (paths * repeats)[:num_needed]

    print(f"  Using {len(paths)} videos", file=sys.stderr, flush=True)
    return paths


def _preprocess_dataset_videos(cfg, config, dtype, device, total_videos):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The video decoding and encoding capabilities of torchvision are deprecated.*",
            )
            from torchvision.io import read_video
    except Exception as exc:
        raise RuntimeError(
            "Real video dataset benchmarking requires torchvision video decoding support. "
            "Install PyAV, e.g. `python -m pip install av`."
        ) from exc

    from transformers import AutoVideoProcessor

    dataset_name = cfg["dataset_name"]
    dataset_split = cfg["dataset_split"]
    video_paths = _load_video_paths(dataset_name, dataset_split, total_videos, cfg["seed"])
    processor = AutoVideoProcessor.from_pretrained(cfg["model"])

    clips = []
    for path in video_paths:
        video, _, _ = read_video(path, pts_unit="sec")
        indices = _sample_frame_indices(video.shape[0], config.frames_per_clip)
        clip = video.index_select(0, indices).numpy()
        clips.append(clip)

    pixel_values = processor(clips, return_tensors="pt")["pixel_values_videos"]
    return pixel_values.to(device=device, dtype=dtype)


def _make_videos(cfg, total_videos, config, dtype, device, seed):
    if cfg["input_source"] == "synthetic":
        return _make_synthetic_videos(total_videos, config, dtype, device, seed)
    return _preprocess_dataset_videos(cfg, config, dtype, device, total_videos)


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    model, model_config, dtype = _load_model(cfg)
    device = cfg["device"]
    task = cfg["task"]

    num_patches = (
        (model_config.frames_per_clip // model_config.tubelet_size)
        * (model_config.crop_size // model_config.patch_size)
        * (model_config.crop_size // model_config.patch_size)
    )

    mode = cfg["mode"]
    if mode == "throughput":
        total_videos = cfg["num_videos"]
        batch_size = cfg["batch_size"]
        videos = _make_videos(cfg, total_videos, model_config, dtype, device, cfg["seed"])
        context_mask, target_mask = _build_masks(
            batch_size=batch_size,
            num_patches=num_patches,
            context_ratio=cfg["context_ratio"],
            target_ratio=cfg["target_ratio"],
            seed=cfg["seed"],
            device=device,
        )

        warmup_batch = videos[:batch_size]
        _ = _forward(task, model, warmup_batch, context_mask, target_mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        total_elapsed = 0.0
        processed = 0
        for start in tqdm(range(0, total_videos, batch_size), desc=f"{cfg['backend']} throughput", file=sys.stderr):
            batch = videos[start:start + batch_size]
            if batch.shape[0] != batch_size:
                local_context_mask, local_target_mask = _build_masks(
                    batch_size=batch.shape[0],
                    num_patches=num_patches,
                    context_ratio=cfg["context_ratio"],
                    target_ratio=cfg["target_ratio"],
                    seed=cfg["seed"],
                    device=device,
                )
            else:
                local_context_mask, local_target_mask = context_mask, target_mask

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _forward(task, model, batch, local_context_mask, local_target_mask)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0
            processed += batch.shape[0]

        with open(cfg["output_file"], "w") as f:
            json.dump({
                "mode": mode,
                "videos": processed,
                "elapsed": total_elapsed,
                "videos_per_second": processed / total_elapsed,
                "patches_per_second": (processed * num_patches) / total_elapsed,
                "num_patches": num_patches,
            }, f)
        return

    if mode == "latency":
        batch_sizes = cfg["latency_batch_sizes"]
        results = []
        for batch_size in batch_sizes:
            videos = _make_videos(cfg, batch_size, model_config, dtype, device, cfg["seed"] + batch_size)
            context_mask, target_mask = _build_masks(
                batch_size=batch_size,
                num_patches=num_patches,
                context_ratio=cfg["context_ratio"],
                target_ratio=cfg["target_ratio"],
                seed=cfg["seed"],
                device=device,
            )

            for _ in range(cfg["latency_warmup"]):
                _ = _forward(task, model, videos, context_mask, target_mask)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

            latencies = []
            for _ in tqdm(range(cfg["latency_iters"]), desc=f"{cfg['backend']} latency bs={batch_size}", file=sys.stderr):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = _forward(task, model, videos, context_mask, target_mask)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)

            results.append({"batch_size": batch_size, "latencies": latencies})

        with open(cfg["output_file"], "w") as f:
            json.dump({"mode": mode, "results": results}, f)
        return

    if mode == "alignment":
        total_videos = cfg["alignment_videos"]
        videos = _make_videos(cfg, total_videos, model_config, dtype, device, cfg["seed"])
        context_mask, target_mask = _build_masks(
            batch_size=total_videos,
            num_patches=num_patches,
            context_ratio=cfg["context_ratio"],
            target_ratio=cfg["target_ratio"],
            seed=cfg["seed"],
            device=device,
        )
        outputs = _forward(task, model, videos, context_mask, target_mask)
        torch.save(outputs, cfg["tensor_file"])
        with open(cfg["output_file"], "w") as f:
            json.dump({
                "mode": mode,
                "tensor_file": cfg["tensor_file"],
                "keys": sorted(outputs.keys()),
                "num_patches": num_patches,
            }, f)
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
'''


def _run_phase(worker_cfg: dict, label: str) -> dict | None:
    return run_worker(VJEPA2_WORKER, worker_cfg, label=label, timeout=4 * 3600)


def _print_throughput_result(name: str, result: dict, ref_result: dict | None) -> None:
    print(f"\n{name} throughput")
    print(f"  videos/sec : {result['videos_per_second']:.3f}")
    print(f"  patches/sec: {result['patches_per_second']:.1f}")
    if ref_result is not None:
        ratio = result["videos_per_second"] / ref_result["videos_per_second"]
        print(f"  ratio      : {ratio:.2f}x vs transformers")


def _print_latency_result(name: str, result: dict, ref_result: dict | None) -> None:
    print(f"\n{name} latency")
    for ours in result["results"]:
        batch_size = ours["batch_size"]
        median = statistics.median(ours["latencies"])
        if ref_result is None:
            print(f"  bs={batch_size}: {median:.4f}s")
            continue
        ref = next(r for r in ref_result["results"] if r["batch_size"] == batch_size)
        ref_median = statistics.median(ref["latencies"])
        speedup = ref_median / median
        print(f"  bs={batch_size}: {median:.4f}s ({speedup:.2f}x vs transformers)")


def _alignment_metrics(task: str, ours_path: str, ref_path: str) -> dict[str, dict[str, float]]:
    ours = _load_tensor_bundle(ours_path)
    ref = _load_tensor_bundle(ref_path)
    metrics: dict[str, dict[str, float]] = {}

    keys = sorted(set(ours) & set(ref))
    for key in keys:
        metrics[key] = {
            "cosine": _cosine(ours[key], ref[key]),
            "mae": _mean_abs_diff(ours[key], ref[key]),
        }
    return metrics


def _print_alignment(metrics: dict[str, dict[str, float]]) -> None:
    print("\nalignment")
    for key, values in metrics.items():
        print(f"  {key}: cosine={values['cosine']:.6f} mae={values['mae']:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fastkernels V-JEPA 2 vs transformers")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--task", choices=["predictor", "encoder", "classification"], default="predictor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--input-source", choices=["dataset", "synthetic"], default="dataset")
    parser.add_argument("--dataset", default="nateraw/kinetics-mini")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--num-videos", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--alignment-videos", type=int, default=2)
    parser.add_argument("--context-ratio", type=float, default=0.75)
    parser.add_argument("--target-ratio", type=float, default=0.25)
    parser.add_argument("--latency-batch-sizes", default="1,2")
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    args.dtype = args.dtype or _default_dtype(args.device)
    latency_batch_sizes = [int(x) for x in args.latency_batch_sizes.split(",") if x]

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        tempfile.mkdtemp(prefix="vjepa2_bench_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_base = {
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "dtype": args.dtype,
        "input_source": args.input_source,
        "dataset_name": args.dataset,
        "dataset_split": args.dataset_split,
        "seed": args.seed,
        "package_root": str(_PACKAGE_DIR),
        "context_ratio": args.context_ratio,
        "target_ratio": args.target_ratio,
    }

    results: dict[str, object] = {
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "dtype": args.dtype,
        "gpu": _detect_gpu_name(),
        "input_source": args.input_source,
        "dataset_name": args.dataset,
        "dataset_split": args.dataset_split,
        "throughput": {},
        "latency": {},
        "alignment": {},
    }

    if not args.skip_throughput:
        ours_cfg = worker_base | {
            "backend": "local",
            "mode": "throughput",
            "num_videos": args.num_videos,
            "batch_size": args.batch_size,
        }
        ours = _run_phase(ours_cfg, "fastkernels V-JEPA 2 throughput")
        if ours is None:
            raise SystemExit(1)
        results["throughput"]["ours"] = ours

        ref = None
        if not args.skip_reference:
            ref_cfg = worker_base | {
                "backend": "reference",
                "mode": "throughput",
                "num_videos": args.num_videos,
                "batch_size": args.batch_size,
            }
            ref = _run_phase(ref_cfg, "transformers V-JEPA 2 throughput")
            if ref is None:
                raise SystemExit(1)
            results["throughput"]["reference"] = ref

        _print_throughput_result("V-JEPA 2", ours, ref)

    if not args.skip_latency:
        ours_cfg = worker_base | {
            "backend": "local",
            "mode": "latency",
            "latency_batch_sizes": latency_batch_sizes,
            "latency_warmup": args.latency_warmup,
            "latency_iters": args.latency_iters,
        }
        ours = _run_phase(ours_cfg, "fastkernels V-JEPA 2 latency")
        if ours is None:
            raise SystemExit(1)
        results["latency"]["ours"] = ours

        ref = None
        if not args.skip_reference:
            ref_cfg = worker_base | {
                "backend": "reference",
                "mode": "latency",
                "latency_batch_sizes": latency_batch_sizes,
                "latency_warmup": args.latency_warmup,
                "latency_iters": args.latency_iters,
            }
            ref = _run_phase(ref_cfg, "transformers V-JEPA 2 latency")
            if ref is None:
                raise SystemExit(1)
            results["latency"]["reference"] = ref

        _print_latency_result("V-JEPA 2", ours, ref)

    if not args.skip_alignment:
        ours_tensor = str(output_dir / "ours_alignment.pt")
        ours_cfg = worker_base | {
            "backend": "local",
            "mode": "alignment",
            "alignment_videos": args.alignment_videos,
            "tensor_file": ours_tensor,
        }
        ours = _run_phase(ours_cfg, "fastkernels V-JEPA 2 alignment")
        if ours is None:
            raise SystemExit(1)
        results["alignment"]["ours"] = ours

        if not args.skip_reference:
            ref_tensor = str(output_dir / "reference_alignment.pt")
            ref_cfg = worker_base | {
                "backend": "reference",
                "mode": "alignment",
                "alignment_videos": args.alignment_videos,
                "tensor_file": ref_tensor,
            }
            ref = _run_phase(ref_cfg, "transformers V-JEPA 2 alignment")
            if ref is None:
                raise SystemExit(1)
            results["alignment"]["reference"] = ref
            metrics = _alignment_metrics(args.task, ours_tensor, ref_tensor)
            results["alignment"]["metrics"] = metrics
            _print_alignment(metrics)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
