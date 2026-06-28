"""Benchmark InstantNGP render throughput and alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_KB_ROOT = Path(__file__).resolve().parents[1]
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))

from fastkernels.bench.utils.worker import run_worker


INSTANTNGP_WORKER = r'''
import json, math, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

from fastkernels.infra.nerf_loader import load_fox_scene, load_ours_instantngp, load_reference_instantngp

def render_views(model, view_indices):
    outputs = []
    with torch.inference_mode():
        for idx in view_indices:
            outputs.append(model(view_index=idx))
    return outputs

def _measure_iters(model, view_indices, iters):
    start = time.perf_counter()
    for _ in range(iters):
        render_views(model, view_indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start

def measure(model, view_indices, warmup_iters, measure_iters, target_measure_seconds):
    for _ in range(warmup_iters):
        render_views(model, view_indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = _measure_iters(model, view_indices, measure_iters)
    total_iters = measure_iters
    while elapsed < target_measure_seconds:
        per_iter = elapsed / max(total_iters, 1)
        extra_iters = max(measure_iters, math.ceil((target_measure_seconds - elapsed) / max(per_iter, 1e-9)))
        elapsed += _measure_iters(model, view_indices, extra_iters)
        total_iters += extra_iters
    total_images = total_iters * len(view_indices)
    return {
        "images_per_second": total_images / elapsed,
        "elapsed": elapsed,
        "iterations": total_iters,
        "images": total_images,
    }

def measure_latency(model, view_indices, latency_iters):
    samples = []
    for _ in range(latency_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        render_views(model, view_indices)
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
        "mean_seconds_per_image": mean / len(view_indices),
        "median_seconds_per_image": median / len(view_indices),
    }

def main():
    scene = load_fox_scene(cfg["scene"])
    width = cfg["width"] or scene.width
    height = cfg["height"] or scene.height
    view_indices = list(range(min(cfg["num_views"], scene.num_views)))
    if cfg["backend"] == "ours":
        model = load_ours_instantngp(
            scene_name=cfg["scene"],
            train_steps=cfg["train_steps"],
            width=width,
            height=height,
            spp=cfg["spp"],
        )
        baseline_name = "fastkernels"
    elif cfg["backend"] == "reference":
        model = load_reference_instantngp(
            scene_name=cfg["scene"],
            train_steps=cfg["train_steps"],
            width=width,
            height=height,
            spp=cfg["spp"],
        )
        baseline_name = "pyngp"
    else:
        raise ValueError(f"Unknown backend: {cfg['backend']}")

    result = {
        "scene": scene.scene_name,
        "train_steps": cfg["train_steps"],
        "num_views": len(view_indices),
        "resolution": [height, width],
        "spp": cfg["spp"],
        "backend": cfg["backend"],
    }

    perf = measure(
        model,
        view_indices,
        cfg["warmup_iters"],
        cfg["measure_iters"],
        cfg["target_measure_seconds"],
    )
    latency = measure_latency(model, view_indices, cfg["latency_iters"])
    with torch.inference_mode():
        imgs = render_views(model, view_indices)
    rgba = torch.stack([img.float() for img in imgs], dim=0)
    torch.save(rgba.cpu(), cfg["rgba_file"])
    result["result"] = {
        "baseline_name": baseline_name,
        **perf,
        "latency": latency,
        "rgba_file": cfg["rgba_file"],
        "image_shape": list(rgba.shape),
    }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark InstantNGP render throughput")
    parser.add_argument("--scene", default="fox")
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--target-measure-seconds", type=float, default=10.0)
    parser.add_argument("--latency-iters", type=int, default=20)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/instantngp_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = {
        "kb_root": str(_KB_ROOT),
        "scene": args.scene,
        "train_steps": args.train_steps,
        "num_views": args.num_views,
        "width": args.width,
        "height": args.height,
        "spp": args.spp,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "target_measure_seconds": args.target_measure_seconds,
        "latency_iters": args.latency_iters,
    }
    ours_cfg = dict(cfg, backend="ours", rgba_file=os.path.join(args.output_dir, "ours_rgba.pt"))
    ours_data = run_worker(INSTANTNGP_WORKER, ours_cfg, "InstantNGP ours benchmark", timeout=7200)
    if ours_data is None:
        raise SystemExit(1)

    result = {
        "scene": args.scene,
        "train_steps": args.train_steps,
        "num_views": args.num_views,
        "resolution": ours_data["resolution"],
        "spp": args.spp,
        "ours": {
            "baseline_name": ours_data["result"]["baseline_name"],
            "images_per_second": ours_data["result"]["images_per_second"],
            "elapsed": ours_data["result"]["elapsed"],
            "images": ours_data["result"]["images"],
            "latency": ours_data["result"]["latency"],
        },
    }

    if not args.skip_reference:
        ref_cfg = dict(
            cfg,
            backend="reference",
            rgba_file=os.path.join(args.output_dir, "reference_rgba.pt"),
        )
        ref_data = run_worker(INSTANTNGP_WORKER, ref_cfg, "InstantNGP reference benchmark", timeout=7200)
        if ref_data is None:
            raise SystemExit(1)
        ours_rgba = torch.load(ours_data["result"]["rgba_file"], map_location="cpu").to(torch.float32)
        ref_rgba = torch.load(ref_data["result"]["rgba_file"], map_location="cpu").to(torch.float32)
        rgba_cos = torch.nn.functional.cosine_similarity(
            ours_rgba.reshape(1, -1),
            ref_rgba.reshape(1, -1),
        ).item()
        rgba_cos = max(-1.0, min(1.0, rgba_cos))
        rgba_mae = torch.mean(torch.abs(ours_rgba - ref_rgba)).item()
        result["reference"] = {
            "baseline_name": ref_data["result"]["baseline_name"],
            "images_per_second": ref_data["result"]["images_per_second"],
            "elapsed": ref_data["result"]["elapsed"],
            "images": ref_data["result"]["images"],
            "latency": ref_data["result"]["latency"],
        }
        result["comparison"] = {
            "throughput_ratio": (
                ours_data["result"]["images_per_second"] / ref_data["result"]["images_per_second"]
                if ref_data["result"]["images_per_second"] > 0
                else float("inf")
            ),
            "rgba_cosine": rgba_cos,
            "rgba_mae": rgba_mae,
            "image_shape": ours_data["result"]["image_shape"],
        }

    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
