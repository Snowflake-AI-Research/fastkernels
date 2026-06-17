#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs diffusers + torch.compile
for SDXL text-to-image.

Runs standardized diffusion workloads and compares:
  - Throughput: images/sec at various resolutions and step counts
  - Latency: per-image latency with percentile stats
  - Correctness: per-batch latent cosine similarity between outputs of both engines

Both engines run with output_type="latent" (no VAE decode) so the benchmark
measures the UNet backbone. Latents are saved per-batch and compared
numerically after both engines finish.

Prompts are drawn from nateraw/parti-prompts (P2) dataset (~1632 prompts).

Each engine runs in a subprocess to avoid import contamination.

Usage:
    python tests/bench_diffusers.py --model stabilityai/stable-diffusion-xl-base-1.0
    python tests/bench_diffusers.py --skip-diffusers  # kb-nano only (no correctness)
    python tests/bench_diffusers.py --enforce-eager    # disable torch.compile for correctness
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.worker import run_worker
from kb_nano.bench.utils.workloads import (
    DIFFUSION_LATENCY_WORKLOADS,
    DIFFUSION_THROUGHPUT_WORKLOADS,
    SDXL_CONFIG,
)


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


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_parti_prompts(seed: int = 42) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    prompts = [row["Prompt"] for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


_PARTI_PROMPTS: list[str] | None = None


def _get_bench_prompts(seed: int = 42) -> list[str]:
    global _PARTI_PROMPTS
    if _PARTI_PROMPTS is None:
        _PARTI_PROMPTS = _load_parti_prompts(seed)
    return _PARTI_PROMPTS


# ---------------------------------------------------------------------------
# Diffusers subprocess worker
# ---------------------------------------------------------------------------
DIFFUSERS_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from diffusers import StableDiffusionXLPipeline

    compile_mode = cfg.get("compile_mode", "max-autotune")
    enforce_eager = cfg.get("enforce_eager", False)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, variant="fp16",
    ).to("cuda")

    if not enforce_eager:
        pipe.unet = torch.compile(pipe.unet, mode=compile_mode)

    seed = cfg["seed"]

    # Warmup at each unique (height, width, batch_size) to trigger torch.compile
    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        bs = s.get("batch_size", len(s.get("prompts", ["w"])))
        key = (s["height"], s["width"], bs)
        if key not in seen:
            seen.add(key)
            warmup_prompts = [f"warmup {i}" for i in range(bs)]
            print(f"Warming up: {s['height']}x{s['width']} batch_size={bs}", file=sys.stderr, flush=True)
            warmup_gen = torch.Generator(device="cuda").manual_seed(seed)
            _ = pipe(
                warmup_prompts, height=s["height"], width=s["width"],
                num_inference_steps=2, generator=warmup_gen, output_type="latent",
            )
            torch.cuda.synchronize()

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        batches = scenario.get("batches", [scenario.get("prompts", [])])
        if not isinstance(batches[0], list):
            batches = [batches]

        total_elapsed = 0.0
        total_images = 0
        desc = f"diffusers {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = pipe(
                batch_prompts,
                height=scenario["height"],
                width=scenario["width"],
                num_inference_steps=scenario["num_inference_steps"],
                guidance_scale=scenario.get("guidance_scale", 5.0),
                generator=gen,
                output_type="latent",
            )
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)

            if latent_dir and output.images is not None:
                latent_tensor = output.images
                if isinstance(latent_tensor, torch.Tensor):
                    torch.save(
                        latent_tensor.cpu(),
                        os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                    )

            pbar.set_postfix(imgs=total_images, ips=f"{total_images / total_elapsed:.2f}")

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_images": total_images,
            "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        for _ in tqdm(range(num_warmup), desc=f"diffusers warmup {ls['name']}", file=sys.stderr):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            torch.cuda.synchronize()
            pipe(prompts, height=ls["height"], width=ls["width"],
                 num_inference_steps=ls["num_inference_steps"],
                 guidance_scale=ls.get("guidance_scale", 5.0),
                 generator=gen, output_type="latent")
            torch.cuda.synchronize()

        latencies = []
        for _ in tqdm(range(num_iters), desc=f"diffusers latency {ls['name']}", file=sys.stderr):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe(prompts, height=ls["height"], width=ls["width"],
                 num_inference_steps=ls["num_inference_steps"],
                 guidance_scale=ls.get("guidance_scale", 5.0),
                 generator=gen, output_type="latent")
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "height": ls["height"],
            "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del pipe
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano subprocess worker
# ---------------------------------------------------------------------------
KB_NANO_SDXL_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    if cfg.get("pytorch_reference", False):
        from kb_nano.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    eng_mod = __import__(
        f"{pkg}.infra.sdxl_engine",
        fromlist=["SDXLEngine"],
    )
    sdxl_mod = __import__(
        f"{pkg}.tasks.baseline.L4.sdxl",
        fromlist=["SDXLSamplingParams"],
    )
    SDXLEngine = eng_mod.SDXLEngine
    SDXLSamplingParams = sdxl_mod.SDXLSamplingParams

    engine = SDXLEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
    )

    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        bs = s.get("batch_size", len(s.get("prompts", ["w"])))
        key = (s["height"], s["width"], bs)
        if key not in seen:
            seen.add(key)
            wp = SDXLSamplingParams(
                height=s["height"], width=s["width"],
                num_inference_steps=2, seed=cfg["seed"],
                output_type="latent",
            )
            warmup_prompts = [f"warmup {i}" for i in range(bs)]
            print(f"Warming up: {s['height']}x{s['width']} batch_size={bs}", file=sys.stderr, flush=True)
            engine.generate(warmup_prompts, wp)
            torch.cuda.synchronize()

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        batches = scenario.get("batches", [scenario.get("prompts", [])])
        if not isinstance(batches[0], list):
            batches = [batches]
        params = SDXLSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 5.0),
            seed=cfg["seed"],
            output_type="latent",
        )

        total_elapsed = 0.0
        total_images = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)

            if latent_dir and output.latents is not None:
                torch.save(
                    output.latents.cpu(),
                    os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                )

            pbar.set_postfix(imgs=total_images, ips=f"{total_images / total_elapsed:.2f}")

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_images": total_images,
            "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        params = SDXLSamplingParams(
            height=ls["height"],
            width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 5.0),
            seed=cfg["seed"],
            output_type="latent",
        )

        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        for _ in tqdm(range(num_warmup), desc=f"kb-nano warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            engine.generate(prompts, params)
            torch.cuda.synchronize()

        latencies = []
        for _ in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "height": ls["height"],
            "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


def _build_throughput_scenarios(prompts: list[str]) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_THROUGHPUT_WORKLOADS:
        bs = w.batch_size
        num_requests = w.num_requests
        total_needed = bs * num_requests
        pool = (prompts * ((total_needed // len(prompts)) + 1))[:total_needed]
        batches = [pool[i * bs: (i + 1) * bs] for i in range(num_requests)]
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": SDXL_CONFIG.num_inference_steps,
            "guidance_scale": SDXL_CONFIG.guidance_scale,
            "batches": batches,
            "batch_size": bs,
            "num_requests": num_requests,
        })
    return scenarios


def _build_latency_scenarios(prompts: list[str]) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": SDXL_CONFIG.num_inference_steps,
            "guidance_scale": SDXL_CONFIG.guidance_scale,
            "prompts": prompts[:w.batch_size],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


def _print_throughput_comparison(kb_results: list[dict], ref_results: list[dict] | None):
    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (images/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Images':>7} {'kb-nano':>12}"
    if ref_results:
        header += f" {'diffusers':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb['num_images']:>7} {kb['images_per_second']:>12.2f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                speedup = kb["images_per_second"] / ref["images_per_second"]
                line += f" {ref['images_per_second']:>12.2f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(kb_results: list[dict], ref_results: list[dict] | None):
    print("\n" + "=" * 80)
    print("  LATENCY COMPARISON (seconds)")
    print("=" * 80)
    header = f"  {'Scenario':<25} {'kb-nano p50':>12}"
    if ref_results:
        header += f" {'diffusers p50':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 60)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<25} {kb_p50:>12.3f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                ref_lats = np.array(ref["latencies"])
                ref_p50 = np.percentile(ref_lats, 50)
                speedup = ref_p50 / kb_p50
                line += f" {ref_p50:>14.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _compare_latents(kb_latent_dir: str, ref_latent_dir: str) -> dict:
    import torch
    from collections import defaultdict

    kb_files = sorted(
        f for f in os.listdir(kb_latent_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_latent_dir) else []
    ref_files = sorted(
        f for f in os.listdir(ref_latent_dir) if f.endswith(".pt")
    ) if os.path.isdir(ref_latent_dir) else []

    common = sorted(set(kb_files) & set(ref_files))
    if not common:
        return {}

    scenario_stats: dict[str, list[dict]] = defaultdict(list)

    for fname in common:
        kb_lat = torch.load(
            os.path.join(kb_latent_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        ref_lat = torch.load(
            os.path.join(ref_latent_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()

        if len(kb_lat) != len(ref_lat):
            print(
                f"  WARNING: shape mismatch for {fname}: "
                f"kb-nano={kb_lat.shape} vs diffusers={ref_lat.shape}, skipping",
                file=sys.stderr,
            )
            continue

        kb_v = kb_lat.numpy()
        ref_v = ref_lat.numpy()

        mse = float(np.mean((kb_v - ref_v) ** 2))
        cos_sim = float(
            np.dot(kb_v, ref_v)
            / (np.linalg.norm(kb_v) * np.linalg.norm(ref_v) + 1e-12)
        )

        scenario_name = fname.rsplit("_batch", 1)[0]
        scenario_stats[scenario_name].append({
            "file": fname, "mse": mse, "cosine_similarity": cos_sim,
        })

    results = {}
    for scenario, batches in scenario_stats.items():
        mses = [b["mse"] for b in batches]
        cosines = [b["cosine_similarity"] for b in batches]
        results[scenario] = {
            "num_batches": len(batches),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
        }
    return results


def _print_correctness_comparison(correctness: dict):
    print("\n" + "=" * 90)
    print("  CORRECTNESS COMPARISON (latent space, per-batch)")
    print("=" * 90)
    print(f"  {'Scenario':<25} {'Batches':>8} {'Mean CosSim':>12} {'Min CosSim':>11} {'Mean MSE':>12} {'Max MSE':>12} {'Result':>8}")
    print("  " + "-" * 88)

    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        min_cos = stats["min_cosine_sim"]
        verdict = "PASS" if mean_cos > 0.95 else ("WARN" if mean_cos > 0.90 else "FAIL")
        print(
            f"  {scenario:<25} {stats['num_batches']:>8} "
            f"{stats['mean_cosine_sim']:>12.6f} {min_cos:>11.6f} "
            f"{stats['mean_mse']:>12.2e} {stats['max_mse']:>12.2e} "
            f"{verdict:>8}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="SDXL benchmark: kb-nano vs diffusers")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-diffusers", action="store_true",
                        help="Skip diffusers and only benchmark kb-nano")
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into kb-nano.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results (default: tests/results/<gpu>/<model>)",
    )
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu_name / short)

    print(f"\nBenchmark: SDXL on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Enforce eager: {args.enforce_eager}")
    print(f"Output dir: {args.output_dir}")

    bench_prompts = _get_bench_prompts(args.seed)
    print(f"Loaded {len(bench_prompts)} prompts from parti-prompts (P2)")

    run_diffusers = not args.skip_diffusers
    save_latents = run_diffusers

    os.makedirs(args.output_dir, exist_ok=True)
    if save_latents:
        kb_latent_dir = os.path.join(args.output_dir, "latents", "kb_nano")
        ref_latent_dir = os.path.join(args.output_dir, "latents", "diffusers")
    else:
        kb_latent_dir = None
        ref_latent_dir = None

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    scenarios = _build_throughput_scenarios(bench_prompts)
    latency_scenarios = _build_latency_scenarios(bench_prompts)

    # --- kb-nano benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "pytorch_reference": args.pytorch_reference,
    }
    if kb_latent_dir:
        kb_config["latent_dir"] = kb_latent_dir
    kb_data = run_worker(
        KB_NANO_SDXL_WORKER, kb_config,
        "kb-nano SDXL benchmark", timeout=36000,
    )

    # --- diffusers benchmark ---
    ref_data = None
    if run_diffusers:
        ref_config = {
            **base_config,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
        }
        if ref_latent_dir:
            ref_config["latent_dir"] = ref_latent_dir
        ref_data = run_worker(
            DIFFUSERS_WORKER, ref_config,
            "diffusers SDXL benchmark", timeout=36000,
        )

    # --- Print results ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        ref_tp = ref_data.get("throughput", []) if ref_data else None
        ref_lat = ref_data.get("latency", []) if ref_data else None

        _print_throughput_comparison(kb_tp, ref_tp)
        _print_latency_comparison(kb_lat, ref_lat)

        correctness = None
        if save_latents and kb_latent_dir and ref_latent_dir:
            correctness = _compare_latents(kb_latent_dir, ref_latent_dir)
            if correctness:
                _print_correctness_comparison(correctness)
            else:
                print("\n  WARNING: No matching latent files found for correctness comparison.")

        results_path = os.path.join(args.output_dir, "results.json")
        results = {
            "model": args.model,
            "gpu": gpu_name,
            "seed": args.seed,
            "kb_nano": kb_data,
        }
        if ref_data:
            results["diffusers"] = ref_data
        if correctness:
            results["correctness"] = correctness
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
