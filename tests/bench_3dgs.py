"""Benchmark 3D Gaussian Splatting render throughput and alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_KB_ROOT = Path(__file__).resolve().parents[1]
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))

import torch

from bench.utils.worker import run_worker

THREEDGS_WORKER = r'''
import json, math, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

from infra.graphics_loader import load_3dgs_scene, load_ours_3dgs, load_reference_3dgs

def render(model):
    with torch.inference_mode():
        out = model(return_meta=False)
    if isinstance(out, tuple) and len(out) == 3:
        return out
    raise RuntimeError(f"Unexpected 3DGS output format: {type(out)}")

def _measure_iters(model, iters):
    start = time.perf_counter()
    for _ in range(iters):
        render(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start

def measure(model, warmup_iters, measure_iters, target_measure_seconds, images_per_iter):
    for _ in range(warmup_iters):
        render(model)
    torch.cuda.synchronize()
    elapsed = _measure_iters(model, measure_iters)
    total_iters = measure_iters
    while elapsed < target_measure_seconds:
        per_iter = elapsed / max(total_iters, 1)
        extra_iters = max(measure_iters, math.ceil((target_measure_seconds - elapsed) / max(per_iter, 1e-9)))
        elapsed += _measure_iters(model, extra_iters)
        total_iters += extra_iters
    return {
        "images_per_second": (total_iters * images_per_iter) / elapsed,
        "elapsed": elapsed,
        "iterations": total_iters,
        "images": total_iters * images_per_iter,
    }

def measure_latency(model, latency_iters, images_per_iter):
    samples = []
    for _ in range(latency_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        render(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples = sorted(samples)
    median = samples[len(samples) // 2]
    mean = sum(samples) / len(samples)
    return {
        "iterations": latency_iters,
        "mean_seconds_per_iter": mean,
        "median_seconds_per_iter": median,
        "mean_seconds_per_image": mean / images_per_iter,
        "median_seconds_per_image": median / images_per_iter,
    }

def main():
    device = cfg["device"]
    dtype = torch.float32
    scene = load_3dgs_scene(
        scene_name=cfg["scene"],
        num_cameras=cfg["num_cameras"],
        max_points=cfg["max_points"],
        device=device,
        dtype=dtype,
        width=cfg["width"],
        height=cfg["height"],
    )
    ours = load_ours_3dgs(scene)
    if hasattr(ours, "prepare_graph"):
        try:
            ours.prepare_graph()
        except Exception:
            pass
    ref = None if cfg["skip_reference"] else load_reference_3dgs(scene)

    result = {
        "scene": scene.scene_name,
        "num_cameras": cfg["num_cameras"],
        "max_points": cfg["max_points"],
        "loaded_points": scene.loaded_points,
        "total_points": scene.total_points,
        "resolution": [scene.height, scene.width],
        "dtype": str(dtype).replace("torch.", ""),
        "model_source": scene.model_source,
        "model_path": scene.model_path,
        "iteration": scene.iteration,
    }

    ours_perf = measure(
        ours,
        cfg["warmup_iters"],
        cfg["measure_iters"],
        cfg["target_measure_seconds"],
        cfg["num_cameras"],
    )
    ours_latency = measure_latency(ours, cfg["latency_iters"], cfg["num_cameras"])
    result["ours"] = {
        "baseline_name": "fastkernels",
        **ours_perf,
        "latency": ours_latency,
    }

    if ref is not None:
        ref_perf = measure(
            ref,
            cfg["warmup_iters"],
            cfg["measure_iters"],
            cfg["target_measure_seconds"],
            cfg["num_cameras"],
        )
        ref_latency = measure_latency(ref, cfg["latency_iters"], cfg["num_cameras"])
        with torch.inference_mode():
            ours_rgb, ours_alpha, _ = render(ours)
            ref_rgb, ref_alpha, _ = render(ref)
            ours_rgb = ours_rgb.float()
            ref_rgb = ref_rgb.float()
            ours_alpha = ours_alpha.float()
            ref_alpha = ref_alpha.float()
        rgb_cos = torch.nn.functional.cosine_similarity(ours_rgb.reshape(1, -1), ref_rgb.reshape(1, -1)).item()
        rgb_mae = torch.mean(torch.abs(ours_rgb - ref_rgb)).item()
        alpha_cos = torch.nn.functional.cosine_similarity(ours_alpha.reshape(1, -1), ref_alpha.reshape(1, -1)).item()
        alpha_mae = torch.mean(torch.abs(ours_alpha - ref_alpha)).item()
        result["reference"] = {
            "baseline_name": "gsplat",
            **ref_perf,
            "latency": ref_latency,
        }
        result["comparison"] = {
            "throughput_ratio": ours_perf["images_per_second"] / ref_perf["images_per_second"] if ref_perf["images_per_second"] > 0 else float("inf"),
            "rgb_cosine": rgb_cos,
            "rgb_mae": rgb_mae,
            "alpha_cosine": alpha_cos,
            "alpha_mae": alpha_mae,
            "image_shape": list(ours_rgb.shape),
        }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 3D Gaussian Splatting")
    parser.add_argument("--scene", default="train")
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--target-measure-seconds", type=float, default=10.0)
    parser.add_argument("--latency-iters", type=int, default=20)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/3dgs_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "kb_root": str(_KB_ROOT),
        "device": device,
        "scene": args.scene,
        "num_cameras": args.num_cameras,
        "max_points": args.max_points,
        "width": args.width,
        "height": args.height,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "target_measure_seconds": args.target_measure_seconds,
        "latency_iters": args.latency_iters,
        "use_fp16": bool(args.use_fp16 and device == "cuda"),
        "skip_reference": args.skip_reference,
    }
    data = run_worker(THREEDGS_WORKER, cfg, "3DGS benchmark", timeout=7200)
    if data is None:
        raise SystemExit(1)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
