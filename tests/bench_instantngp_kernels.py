"""Benchmark kernel-level InstantNGP field primitives vs direct tinycudann."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_KB_ROOT = Path(__file__).resolve().parents[1]
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))

from fastkernels.bench.utils.worker import run_worker


INSTANTNGP_KERNEL_WORKER = r'''
import json, math, sys, time
import torch
import torch.nn.functional as F

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

import tinycudann as tcnn
from fastkernels.infra.nerf_loader import sample_real_fox_field_inputs
from fastkernels.tasks.baseline.L2.instantngp_field import InstantNGPField

class ReferenceField(torch.nn.Module):
    def __init__(self, field: InstantNGPField):
        super().__init__()
        self.position_encoding = tcnn.Encoding(
            3,
            field.position_encoding.config,
            seed=field.seed,
            dtype=field.dtype,
        )
        self.density_mlp = tcnn.Network(
            self.position_encoding.n_output_dims,
            1 + field.geo_feat_dims,
            field.density_mlp.config,
            seed=field.seed,
        )
        self.direction_encoding = tcnn.Encoding(
            3,
            field.direction_encoding.config,
            seed=field.seed,
            dtype=field.dtype,
        )
        self.rgb_mlp = tcnn.Network(
            self.direction_encoding.n_output_dims + 1 + field.geo_feat_dims,
            3,
            field.rgb_mlp.config,
            seed=field.seed,
        )

    def forward(self, positions: torch.Tensor, directions: torch.Tensor):
        density_features = self.density_mlp(self.position_encoding(positions))
        sigma = density_features[:, :1]
        geo_feat = density_features[:, 1:]
        rgb = self.rgb_mlp(torch.cat([density_features, self.direction_encoding(directions)], dim=-1))
        return sigma, geo_feat, rgb

def _measure_iters(fn, positions, directions, iters):
    start = time.perf_counter()
    for _ in range(iters):
        fn(positions, directions)
    torch.cuda.synchronize()
    return time.perf_counter() - start

def measure(fn, positions, directions, warmup_iters, measure_iters, target_measure_seconds):
    for _ in range(warmup_iters):
        fn(positions, directions)
    torch.cuda.synchronize()
    elapsed = _measure_iters(fn, positions, directions, measure_iters)
    total_iters = measure_iters
    while elapsed < target_measure_seconds:
        per_iter = elapsed / max(total_iters, 1)
        extra_iters = max(measure_iters, math.ceil((target_measure_seconds - elapsed) / max(per_iter, 1e-9)))
        elapsed += _measure_iters(fn, positions, directions, extra_iters)
        total_iters += extra_iters
    total_samples = total_iters * positions.shape[0]
    return {
        "samples_per_second": total_samples / elapsed,
        "elapsed": elapsed,
        "iterations": total_iters,
        "samples": total_samples,
    }

def measure_latency(fn, positions, directions, latency_iters):
    samples = []
    for _ in range(latency_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(positions, directions)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples = sorted(samples)
    median = samples[len(samples) // 2]
    mean = sum(samples) / len(samples)
    return {
        "iterations": latency_iters,
        "mean_seconds_per_iter": mean,
        "median_seconds_per_iter": median,
        "mean_seconds_per_sample": mean / positions.shape[0],
        "median_seconds_per_sample": median / positions.shape[0],
    }

def main():
    device = "cuda"
    positions, directions = sample_real_fox_field_inputs(
        num_samples=cfg["num_samples"],
        device=device,
        scene_name=cfg["scene"],
        train_steps=cfg["train_steps"],
        num_views=cfg["num_views"],
        seed=cfg["seed"],
    )

    ours = InstantNGPField(seed=cfg["seed"]).to(device).eval()
    ref = ReferenceField(ours).to(device).eval()

    ref.position_encoding.load_state_dict(ours.position_encoding.encoding.state_dict())
    ref.density_mlp.load_state_dict(ours.density_mlp.network.state_dict())
    ref.direction_encoding.load_state_dict(ours.direction_encoding.encoding.state_dict())
    ref.rgb_mlp.load_state_dict(ours.rgb_mlp.network.state_dict())

    with torch.inference_mode():
        ours_out = ours(positions, directions)
        ref_sigma, ref_geo, ref_rgb = ref(positions, directions)

    ours_perf = measure(
        lambda p, d: ours(p, d),
        positions,
        directions,
        cfg["warmup_iters"],
        cfg["measure_iters"],
        cfg["target_measure_seconds"],
    )
    ref_perf = measure(
        lambda p, d: ref(p, d),
        positions,
        directions,
        cfg["warmup_iters"],
        cfg["measure_iters"],
        cfg["target_measure_seconds"],
    )
    ours_latency = measure_latency(lambda p, d: ours(p, d), positions, directions, cfg["latency_iters"])
    ref_latency = measure_latency(lambda p, d: ref(p, d), positions, directions, cfg["latency_iters"])

    def cosine(a, b):
        return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()

    def mae(a, b):
        return torch.mean(torch.abs(a - b)).item()

    result = {
        "scene": cfg["scene"],
        "train_steps": cfg["train_steps"],
        "num_views": cfg["num_views"],
        "num_samples": cfg["num_samples"],
        "ours": {
            "baseline_name": "fastkernels-kernel",
            **ours_perf,
            "latency": ours_latency,
        },
        "reference": {
            "baseline_name": "tinycudann-direct",
            **ref_perf,
            "latency": ref_latency,
        },
        "comparison": {
            "throughput_ratio": ours_perf["samples_per_second"] / ref_perf["samples_per_second"] if ref_perf["samples_per_second"] > 0 else float("inf"),
            "sigma_cosine": cosine(ours_out.sigma.float(), ref_sigma.float()),
            "sigma_mae": mae(ours_out.sigma.float(), ref_sigma.float()),
            "geo_feat_cosine": cosine(ours_out.geo_feat.float(), ref_geo.float()),
            "geo_feat_mae": mae(ours_out.geo_feat.float(), ref_geo.float()),
            "rgb_cosine": cosine(ours_out.rgb.float(), ref_rgb.float()),
            "rgb_mae": mae(ours_out.rgb.float(), ref_rgb.float()),
        },
    }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark InstantNGP kernel-level field")
    parser.add_argument("--scene", default="fox")
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=50)
    parser.add_argument("--target-measure-seconds", type=float, default=10.0)
    parser.add_argument("--latency-iters", type=int, default=20)
    parser.add_argument("--output-dir", default="/tmp/instantngp_kernel_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = {
        "kb_root": str(_KB_ROOT),
        "scene": args.scene,
        "train_steps": args.train_steps,
        "num_views": args.num_views,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "target_measure_seconds": args.target_measure_seconds,
        "latency_iters": args.latency_iters,
    }
    data = run_worker(
        INSTANTNGP_KERNEL_WORKER,
        cfg,
        "InstantNGP kernel benchmark",
        timeout=7200,
    )
    if data is None:
        raise SystemExit(1)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
