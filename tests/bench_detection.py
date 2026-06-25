#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: fastkernels detection models
against official baselines (ultralytics YOLOv10, transformers RTDetrV2).

Uses real images from COCO val2017 (5000 images) loaded via HuggingFace
datasets, matching the methodology of other fastkernels benchmarks.

Correctness is checked over ALL throughput images by collecting outputs
during the final measurement pass, not on a separate subset.

Each engine (fastkernels, reference) runs in a subprocess to avoid import
contamination.

Usage:
    python tests/bench_detection.py --model jameslahm/yolov10n
    python tests/bench_detection.py --model PekingU/rtdetr_v2_r18vd
    python tests/bench_detection.py --skip-reference  # fastkernels only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PACKAGE_DIR))

from bench.utils.worker import run_worker
from bench.utils.workloads import (
    DETECTION_LATENCY_WORKLOADS,
    DETECTION_THROUGHPUT_WORKLOADS,
)
from infra.detection_loader import infer_image_size


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
# COCO image loading and preprocessing
# ---------------------------------------------------------------------------

def _load_coco_images(
    num_images: int,
    image_size: int,
    seed: int = 42,
    cache_dir: str | None = None,
) -> str:
    """Download COCO val2017 images and save preprocessed tensors to disk.

    Returns path to the saved tensor file (N, 3, H, W) float16.
    """
    import random

    import torch
    from datasets import load_dataset
    from PIL import Image
    from torchvision import transforms

    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent.parent / "data" / "coco_cache")
    os.makedirs(cache_dir, exist_ok=True)

    tensor_path = os.path.join(
        cache_dir, f"coco_val_{num_images}_{image_size}_fp16.pt"
    )
    if os.path.exists(tensor_path):
        print(f"  Using cached preprocessed images: {tensor_path}")
        return tensor_path

    print(f"  Loading COCO val2017 images from HuggingFace ...", flush=True)
    ds = load_dataset(
        "detection-datasets/coco", split="val", streaming=True,
    )

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    all_tensors = []
    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)

    for item in items:
        if len(all_tensors) >= num_images:
            break
        try:
            img = item["image"]
            if not isinstance(img, Image.Image):
                continue
            img = img.convert("RGB")
            tensor = transform(img)
            all_tensors.append(tensor)
        except Exception:
            continue

    if not all_tensors:
        raise RuntimeError("Failed to load any COCO images")

    stacked = torch.stack(all_tensors).half()
    print(f"  Preprocessed {len(all_tensors)} images -> {stacked.shape}")
    torch.save(stacked, tensor_path)
    print(f"  Saved to {tensor_path}")
    return tensor_path


# ---------------------------------------------------------------------------
# Detection subprocess worker
# ---------------------------------------------------------------------------

DETECTION_WORKER = r'''
import json, os, sys, time
import importlib.util
from pathlib import Path
import torch
import numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])

    pkg_root = Path(cfg["project_root"])
    spec = importlib.util.spec_from_file_location(
        "fastkernels", pkg_root / "__init__.py",
        submodule_search_locations=[str(pkg_root)],
    )
    fastkernels = importlib.util.module_from_spec(spec)
    sys.modules["fastkernels"] = fastkernels
    spec.loader.exec_module(fastkernels)

    from fastkernels.infra.detection_loader import (
        load_ours_detector,
        load_reference_detector,
        run_ours_detector,
        run_reference_detector,
    )

    if cfg.get("pytorch_reference", False) and cfg.get("backend") == "ours":
        from fastkernels.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    def _run_model(model, backend, model_name, pixel_values, image_size, max_det):
        if backend == "ours":
            return run_ours_detector(
                model, model_name, pixel_values, image_size, max_detections=max_det,
            )
        return run_reference_detector(
            model, model_name, pixel_values, image_size, max_detections=max_det,
        )

    device = cfg.get("device", "cuda")
    dtype = torch.float16 if cfg.get("use_fp16", True) else torch.float32
    model_name = cfg["model"]
    backend = cfg["backend"]
    image_size = cfg["image_size"]
    max_detections = cfg["max_detections"]
    feats_dir = cfg.get("feats_dir")

    tensor_path = cfg["tensor_path"]
    print(f"  Loading preprocessed images from {tensor_path} ...", flush=True)
    all_images_cpu = torch.load(tensor_path, map_location="cpu", weights_only=True).to(dtype=dtype)
    print(f"  Loaded {all_images_cpu.shape[0]} images ({all_images_cpu.shape})", flush=True)

    if backend == "ours":
        model = load_ours_detector(model_name, device=device, dtype=dtype)
        baseline_name = "fastkernels"
    else:
        model, baseline_name = load_reference_detector(model_name, device=device, dtype=dtype)

    with torch.no_grad():
        warmup_batch = all_images_cpu[:1].to(device=device)
        _ = _run_model(model, backend, model_name, warmup_batch, image_size, max_detections)
        del warmup_batch
        torch.cuda.synchronize()

    # ---- Throughput scenarios ----
    throughput_results = []
    for scenario in cfg["throughput_scenarios"]:
        name = scenario["name"]
        num_images = min(scenario["num_images"], all_images_cpu.shape[0])
        batch_size = scenario["batch_size"]
        num_warmup = scenario.get("num_warmup", 3)
        num_measure = scenario.get("num_measure", 3)

        images_cpu = all_images_cpu[:num_images]

        with torch.no_grad():
            for _ in range(num_warmup):
                for start in range(0, num_images, batch_size):
                    batch = images_cpu[start:start + batch_size].to(device=device)
                    _run_model(model, backend, model_name, batch, image_size, max_detections)
            torch.cuda.synchronize()

            elapsed_runs = []
            for run_idx in range(num_measure):
                is_last = (run_idx == num_measure - 1)
                all_boxes = []
                all_scores = []
                all_labels = []

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for start in range(0, num_images, batch_size):
                    batch = images_cpu[start:start + batch_size].to(device=device)
                    det = _run_model(model, backend, model_name, batch, image_size, max_detections)
                    if is_last:
                        all_boxes.append(det["boxes"].float().cpu())
                        all_scores.append(det["scores"].float().cpu())
                        all_labels.append(det["labels"].cpu())
                torch.cuda.synchronize()
                elapsed_runs.append(time.perf_counter() - t0)

        # Save outputs from last run for correctness comparison
        if feats_dir:
            os.makedirs(feats_dir, exist_ok=True)
            prefix = "kb" if backend == "ours" else "ref"
            torch.save(torch.cat(all_boxes, dim=0), os.path.join(feats_dir, f"{prefix}_boxes.pt"))
            torch.save(torch.cat(all_scores, dim=0), os.path.join(feats_dir, f"{prefix}_scores.pt"))
            torch.save(torch.cat(all_labels, dim=0), os.path.join(feats_dir, f"{prefix}_labels.pt"))

        median_elapsed = sorted(elapsed_runs)[len(elapsed_runs) // 2]
        throughput_results.append({
            "name": name,
            "num_images": num_images,
            "batch_size": batch_size,
            "elapsed": median_elapsed,
            "images_per_second": num_images / median_elapsed,
            "elapsed_runs": elapsed_runs,
        })
        print(
            f"  [{backend}] {name}: {num_images / median_elapsed:.1f} img/s "
            f"(median {median_elapsed:.3f}s over {num_measure} runs)",
            flush=True,
        )

    # ---- Latency scenarios ----
    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        name = ls["name"]
        batch_size = ls["batch_size"]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 20)

        batch = all_images_cpu[:batch_size].to(device=device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _run_model(model, backend, model_name, batch, image_size, max_detections)
            torch.cuda.synchronize()

            latencies = []
            for _ in range(num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _run_model(model, backend, model_name, batch, image_size, max_detections)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": name,
            "batch_size": batch_size,
            "num_iters": num_iters,
            "latencies": latencies,
            "median": sorted(latencies)[len(latencies) // 2],
            "mean": sum(latencies) / len(latencies),
        })
        med = sorted(latencies)[len(latencies) // 2]
        print(f"  [{backend}] latency {name}: median {med*1000:.2f}ms", flush=True)

    result = {
        "backend": backend,
        "baseline_name": baseline_name,
        "throughput": throughput_results,
        "latency": latency_results,
    }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    os._exit(0)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Correctness comparison (loads saved tensors from feats_dir)
# ---------------------------------------------------------------------------

def _compare_saved_outputs(feats_dir: str) -> dict | None:
    """Compare detection outputs saved by both workers during throughput."""
    import torch

    kb_boxes_path = os.path.join(feats_dir, "kb_boxes.pt")
    ref_boxes_path = os.path.join(feats_dir, "ref_boxes.pt")
    if not os.path.exists(kb_boxes_path) or not os.path.exists(ref_boxes_path):
        return None

    kb_boxes = torch.load(kb_boxes_path, map_location="cpu", weights_only=True).detach().float().numpy()
    ref_boxes = torch.load(ref_boxes_path, map_location="cpu", weights_only=True).detach().float().numpy()
    kb_scores = torch.load(os.path.join(feats_dir, "kb_scores.pt"), map_location="cpu", weights_only=True).detach().float().numpy()
    ref_scores = torch.load(os.path.join(feats_dir, "ref_scores.pt"), map_location="cpu", weights_only=True).detach().float().numpy()
    kb_labels = torch.load(os.path.join(feats_dir, "kb_labels.pt"), map_location="cpu", weights_only=True).detach().numpy()
    ref_labels = torch.load(os.path.join(feats_dir, "ref_labels.pt"), map_location="cpu", weights_only=True).detach().numpy()

    num_images = kb_boxes.shape[0]

    kb_b = kb_boxes.reshape(-1).astype(np.float64)
    ref_b = ref_boxes.reshape(-1).astype(np.float64)
    nb_kb, nb_ref = np.linalg.norm(kb_b), np.linalg.norm(ref_b)
    boxes_cos = float(kb_b @ ref_b / max(nb_kb * nb_ref, 1e-12)) if nb_kb > 0 or nb_ref > 0 else 1.0
    boxes_mae = float(np.mean(np.abs(kb_b - ref_b)))

    kb_s = kb_scores.reshape(-1).astype(np.float64)
    ref_s = ref_scores.reshape(-1).astype(np.float64)
    ns_kb, ns_ref = np.linalg.norm(kb_s), np.linalg.norm(ref_s)
    scores_cos = float(kb_s @ ref_s / max(ns_kb * ns_ref, 1e-12)) if ns_kb > 0 or ns_ref > 0 else 1.0
    scores_mae = float(np.mean(np.abs(kb_s - ref_s)))

    total = kb_labels.size
    matched = int(np.sum(kb_labels.reshape(-1) == ref_labels.reshape(-1)))
    labels_match = matched / max(total, 1)

    return {
        "num_images": num_images,
        "boxes_cosine": boxes_cos,
        "boxes_mae": boxes_mae,
        "scores_cosine": scores_cos,
        "scores_mae": scores_mae,
        "labels_match_rate": labels_match,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_throughput_comparison(kb_results, ref_results):
    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (images/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<20} {'Images':>7} {'BS':>4} {'fastkernels':>12}"
    if ref_results:
        header += f" {'reference':>12} {'Ratio':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = (
            f"  {kb['name']:<20} {kb['num_images']:>7} "
            f"{kb['batch_size']:>4} {kb['images_per_second']:>12.1f}"
        )
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                ratio = kb["images_per_second"] / ref["images_per_second"]
                line += f" {ref['images_per_second']:>12.1f} {ratio:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(kb_results, ref_results):
    print("=" * 90)
    print("  LATENCY COMPARISON (milliseconds)")
    print("=" * 90)
    header = f"  {'Scenario':<20} {'BS':>4} {'Iters':>6} {'fastkernels p50':>14}"
    if ref_results:
        header += f" {'reference p50':>14} {'Ratio':>10}"
    print(header)
    print("  " + "-" * 75)

    for kb in kb_results:
        kb_med = kb["median"] * 1000
        line = (
            f"  {kb['name']:<20} {kb['batch_size']:>4} "
            f"{kb['num_iters']:>6} {kb_med:>13.2f}ms"
        )
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                ref_med = ref["median"] * 1000
                ratio = ref["median"] / kb["median"]
                line += f" {ref_med:>13.2f}ms {ratio:>9.2f}x"
        print(line)
    print()


def _print_correctness(correctness: dict):
    print("=" * 90)
    print("  CORRECTNESS COMPARISON (fastkernels vs reference, all throughput images)")
    print("=" * 90)
    n = correctness["num_images"]
    print(f"  Images compared  : {n}")
    print(
        f"  Boxes cosine     : {correctness['boxes_cosine']:.6f}  "
        f"(MAE: {correctness['boxes_mae']:.6f})"
    )
    print(
        f"  Scores cosine    : {correctness['scores_cosine']:.6f}  "
        f"(MAE: {correctness['scores_mae']:.6f})"
    )
    print(f"  Labels match     : {correctness['labels_match_rate']:.4f}")

    boxes_pass = correctness["boxes_cosine"] >= 0.99
    scores_pass = correctness["scores_cosine"] >= 0.99
    labels_pass = correctness["labels_match_rate"] >= 0.99
    overall = boxes_pass and scores_pass and labels_pass
    verdict = "PASS" if overall else "FAIL"
    print(f"  Overall          : {verdict}")
    print()
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Detection benchmark: fastkernels vs official baselines (COCO val2017)",
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=0,
                    help="Input resolution (0 = infer from model)")
    ap.add_argument("--num-images", type=int, default=5000,
                    help="Number of COCO images for throughput + correctness (default: 5000)")
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--use-fp16", action="store_true", default=True)
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into fastkernels.",
    )
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--data-cache-dir", type=str, default=None,
                    help="Directory to cache preprocessed COCO images")
    args = ap.parse_args()

    image_size = args.image_size or infer_image_size(args.model)
    gpu = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu / short)
    os.makedirs(args.output_dir, exist_ok=True)

    feats_dir = os.path.join(args.output_dir, "feats")

    print("=" * 70)
    print("  Preparing COCO val2017 images")
    print("=" * 70)

    tensor_path = _load_coco_images(
        args.num_images, image_size,
        seed=args.seed, cache_dir=args.data_cache_dir,
    )

    throughput_scenarios = []
    if not args.skip_throughput:
        for w in DETECTION_THROUGHPUT_WORKLOADS:
            num = min(w.num_images, args.num_images)
            throughput_scenarios.append({
                "name": w.name,
                "num_images": num,
                "batch_size": w.batch_size,
                "num_warmup": 3,
                "num_measure": 3,
            })

    latency_scenarios = []
    if not args.skip_latency:
        for w in DETECTION_LATENCY_WORKLOADS:
            latency_scenarios.append({
                "name": w.name,
                "batch_size": w.batch_size,
                "num_warmup": w.num_warmup,
                "num_iters": w.num_iters,
            })

    print("\n" + "=" * 70)
    print("  Detection Benchmark: fastkernels vs Reference")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu}")
    print(f"  Image size     : {image_size}")
    print(f"  Dataset        : COCO val2017 ({args.num_images} images)")
    print(f"  Seed           : {args.seed}")
    if throughput_scenarios:
        print(f"  Throughput     : {', '.join(s['name'] for s in throughput_scenarios)}")
    if latency_scenarios:
        print(f"  Latency        : {', '.join(s['name'] for s in latency_scenarios)}")
    print("=" * 70)

    common = {
        "project_root": str(_PACKAGE_DIR),
        "model": args.model,
        "image_size": image_size,
        "max_detections": args.max_detections,
        "use_fp16": args.use_fp16,
        "device": "cuda",
        "seed": args.seed,
        "tensor_path": tensor_path,
        "throughput_scenarios": throughput_scenarios,
        "latency_scenarios": latency_scenarios,
        "feats_dir": feats_dir,
    }

    # -- Run reference first --
    ref_raw = None
    if not args.skip_reference:
        ref_raw = run_worker(
            DETECTION_WORKER,
            {**common, "backend": "reference"},
            f"Reference [{args.model.split('/')[-1]}]",
            timeout=3600,
        )
        if ref_raw is None:
            print("  WARNING: Reference subprocess failed.")

    # -- Run fastkernels --
    kb_raw = run_worker(
        DETECTION_WORKER,
        {**common, "backend": "ours", "pytorch_reference": args.pytorch_reference},
        f"fastkernels [{args.model.split('/')[-1]}]",
        timeout=3600,
    )
    if kb_raw is None:
        print("  ERROR: fastkernels subprocess failed.")
        sys.exit(1)

    # -- Print results --
    kb_tp = kb_raw.get("throughput", [])
    kb_lat = kb_raw.get("latency", [])
    ref_tp = ref_raw.get("throughput", []) if ref_raw else None
    ref_lat = ref_raw.get("latency", []) if ref_raw else None

    if kb_tp:
        _print_throughput_comparison(kb_tp, ref_tp)
    if kb_lat:
        _print_latency_comparison(kb_lat, ref_lat)

    correctness = None
    overall_pass = True
    if ref_raw and not args.skip_throughput:
        correctness = _compare_saved_outputs(feats_dir)
        if correctness:
            overall_pass = _print_correctness(correctness)
        else:
            print("  WARNING: No saved outputs found for correctness comparison.")

    # -- Save results --
    results = {
        "model": args.model,
        "gpu": gpu,
        "image_size": image_size,
        "dataset": "COCO val2017",
        "num_images": args.num_images,
        "seed": args.seed,
        "fastkernels": {
            "baseline_name": kb_raw["baseline_name"],
            "throughput": kb_tp,
            "latency": kb_lat,
        },
    }
    if ref_raw:
        results["reference"] = {
            "baseline_name": ref_raw["baseline_name"],
            "throughput": ref_tp,
            "latency": ref_lat,
        }
    if correctness:
        results["correctness"] = correctness
        results["overall_pass"] = overall_pass

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
