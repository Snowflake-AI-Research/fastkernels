#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: fastkernels vs timm
for vision encoder models (SigLIP-2, DINOv3, SwinV2, MobileNetV4).

Runs standardized vision encoder workloads and compares:
  - Throughput: images/sec at default and high resolution
  - Latency: per-image latency with percentile stats
  - Correctness: per-batch embedding cosine similarity between both engines

Both engines process identical real images from ImageNet-1K validation
(ILSVRC/imagenet-1k) loaded via HuggingFace datasets streaming. Requires
accepting gated access terms at
https://huggingface.co/datasets/ILSVRC/imagenet-1k and running
`huggingface-cli login`.

Images are resized and center-cropped to the target resolution, and
normalized with model-specific mean/std. Embeddings are saved per-batch
during the throughput run and compared numerically afterward (inline
correctness).

Each engine runs in a subprocess to avoid import contamination.

Usage:
    python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex
    python tests/bench_timm.py --model facebook/dinov3-vit7b16-pretrain-lvd1689m
    python tests/bench_timm.py --model timm/swinv2_large_window12_192.ms_in22k
    python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex --skip-timm
    python tests/bench_timm.py --model timm/mobilenetv4_conv_medium.e500_r256_in1k
    python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex --dataset food101 --dataset-split validation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from fastkernels.validate.worker import run_worker
from fastkernels.workloads import (
    VISION_ENCODER_LATENCY_WORKLOADS,
    VISION_ENCODER_THROUGHPUT_WORKLOADS,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "google/siglip2-so400m-patch16-naflex": {
        "timm_name": "naflexvit_so400m_patch16_siglip.v2_webli",
        "kb_module": "siglip2",
        "kb_class": "SigLIP2Model",
        "default_resolution": 384,
        "short_name": "siglip2-so400m",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "default_num_images": 2000,
    },
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {
        "timm_name": "vit_7b_patch16_dinov3.lvd1689m",
        "kb_module": "dinov3",
        "kb_class": "DINOv3Model",
        "default_resolution": 256,
        "short_name": "dinov3-7b",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "default_num_images": 1500,
    },
    "timm/swinv2_large_window12_192.ms_in22k": {
        "timm_name": "swinv2_large_window12_192.ms_in22k",
        "kb_module": "swinv2",
        "kb_class": "SwinV2Model",
        "default_resolution": 192,
        "short_name": "swinv2-large",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "default_num_images": 1000,
        "strict_img_size": False,
    },
    "timm/mobilenetv4_conv_medium.e500_r256_in1k": {
        "timm_name": "mobilenetv4_conv_medium.e500_r256_in1k",
        "kb_module": "mobilenetv4",
        "kb_class": "MobileNetV4Model",
        "default_resolution": 256,
        "short_name": "mobilenetv4-conv-medium",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "default_num_images": 1000,
    },
}


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
# timm subprocess worker
# ---------------------------------------------------------------------------
_WORKER_COMMON = r'''
import hashlib, json, os, sys, time
import numpy as np
import torch
from tqdm import tqdm

EMBED_SAVE_CAP = 500

# A streaming shuffle must fill its buffer before yielding the first example, so
# buffer_size is paid in full JPEG decodes even when only a handful of images are
# wanted. At 5000 that dominated the whole task: bench_timm spent 61 min of the
# 20260804-035022 run to produce 2.8 min of measured work. 64 keeps the part that
# actually matters -- ``IterableDataset.shuffle`` also permutes SHARD order from
# the seed, which is what gives class diversity on a class-ordered split -- while
# costing 64 decodes instead of 5000.
_SHUFFLE_BUFFER = 64

# Warm up by DURATION, not iteration count. Once the image cache removed the
# untimed CPU preprocessing, the throughput loop went from ~12% GPU duty cycle to
# ~93%, and the measured rate stopped being flat: siglip2's running images/sec
# fell 1734 -> 1565 across a 63-batch run as the GPU left boost clocks. The two
# engines run in SEQUENTIAL subprocesses, so without a common thermal starting
# point that drift leaks straight into the ratio -- it moved siglip2's high-res
# from 1.0019x to 0.9715x and single-image from 0.9981x to 1.1787x purely from
# measurement order. A few seconds of continuous work first puts both sides in
# the same steady state; 3 iterations did not.
_WARMUP_SECONDS = float(os.environ.get("FASTKERNELS_BENCH_WARMUP_S", "10"))


def _image_cache_dir():
    root = os.environ.get("FASTKERNELS_IMAGE_CACHE") or os.path.expanduser(
        "~/.fastkernels/cache/bench_images")
    os.makedirs(root, exist_ok=True)
    return root


def _image_cache_path(dataset_name, dataset_split, seed, num_images, resolution):
    key = f"{dataset_name}|{dataset_split}|{seed}|{num_images}|{resolution}|{_SHUFFLE_BUFFER}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(_image_cache_dir(), f"imgs_{digest}.npy")


def _load_pil_images(dataset_name, dataset_split, num_needed, seed):
    """Load raw PIL images from a HuggingFace dataset via streaming."""
    from datasets import load_dataset
    from PIL import Image

    print(
        f"Loading {num_needed} unique images from {dataset_name} ({dataset_split})...",
        file=sys.stderr, flush=True,
    )
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=_SHUFFLE_BUFFER)

    images = []
    for sample in ds:
        img = sample.get("image") or sample.get("img")
        if img is None:
            for v in sample.values():
                if hasattr(v, "convert"):
                    img = v
                    break
        if img is None:
            continue
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO
            img = Image.open(BytesIO(img["bytes"]))
        images.append(img.convert("RGB"))
        if len(images) >= num_needed:
            break
    print(f"  Loaded {len(images)} unique images", file=sys.stderr, flush=True)
    return images


def _get_cached_uint8(dataset_name, dataset_split, seed, num_images, resolution):
    """uint8 [N,3,R,R] of resize+center-cropped images, memmapped from disk.

    Cached because the decode/resize is pure setup that the harness does not
    time, yet it ran twice per model (once in each side's subprocess) and again
    on every rerun.

    The split point is exact: Resize/CenterCrop operate on the PIL image in
    uint8 space, and caching their output was verified bit-identical to
    recomputing it. Everything after them (ToTensor's ``/255``, Normalize's
    ``(x-m)/s``) is elementwise float and moves to the GPU per batch.

    That last move is not bit-exact, and deliberately so: doing the normalize on
    device instead of on the host shifts float32 results by up to 1 ULP
    (1.2e-07), which occasionally tips the round-to-bfloat16. It is not worth
    the ~15 ms/batch of single-threaded host work to avoid, because both sides
    read this same cache through this same path -- so the fastkernels-vs-timm
    comparison sees byte-identical inputs, which is if anything stronger than
    before, when each subprocess preprocessed independently.
    """
    from torchvision import transforms

    path = _image_cache_path(dataset_name, dataset_split, seed, num_images, resolution)
    want = (num_images, 3, resolution, resolution)
    if os.path.exists(path):
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.shape == want and arr.dtype == np.uint8:
                print(f"  image cache HIT {os.path.basename(path)} {want}",
                      file=sys.stderr, flush=True)
                return arr
        except Exception:
            pass

    print(f"  image cache MISS {os.path.basename(path)} {want} -- building",
          file=sys.stderr, flush=True)
    geom = transforms.Compose([
        transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
    ])
    pil_images = _load_pil_images(dataset_name, dataset_split, num_images, seed)
    if len(pil_images) < num_images:
        raise RuntimeError(
            f"requested {num_images} images from {dataset_name}:{dataset_split}, "
            f"got {len(pil_images)}"
        )

    out = np.empty(want, dtype=np.uint8)
    for i, img in enumerate(tqdm(pil_images, desc="  resize+crop", unit="img",
                                 file=sys.stderr)):
        out[i] = np.asarray(geom(img), dtype=np.uint8).transpose(2, 0, 1)

    # Atomic publish so a crash mid-build cannot leave a short/torn cache that a
    # later run would happily mmap. The temp name already ends in .npy so np.save
    # does not append a second suffix.
    tmp = os.path.join(_image_cache_dir(), f".tmp_{os.getpid()}_{os.path.basename(path)}")
    try:
        np.save(tmp, out, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return np.load(path, mmap_mode="r")


class _BatchSource:
    """Turns the cached uint8 array into normalized GPU batches."""

    def __init__(self, arr, image_mean, image_std, dtype, device="cuda"):
        self.arr = arr
        self.dtype = dtype
        self.device = device
        self.mean = torch.tensor(image_mean, dtype=torch.float32,
                                 device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(image_std, dtype=torch.float32,
                                device=device).view(1, 3, 1, 1)

    def __len__(self):
        return self.arr.shape[0]

    def batch(self, start, end):
        chunk = np.ascontiguousarray(self.arr[start:end])
        t = torch.from_numpy(chunk).to(device=self.device, non_blocking=True)
        t = t.float().div_(255.0)
        t = t.sub_(self.mean).div_(self.std)
        return t.to(self.dtype)


def run_benchmark(model, cfg, label):
    seed = cfg["seed"]
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    image_mean = cfg["image_mean"]
    image_std = cfg["image_std"]
    dataset_name = cfg["dataset_name"]
    dataset_split = cfg["dataset_split"]

    embed_dir = cfg.get("embed_dir")
    if embed_dir:
        os.makedirs(embed_dir, exist_ok=True)

    scenarios = cfg.get("scenarios", [])
    latency_scenarios = cfg.get("latency_scenarios", [])

    # One cache per distinct resolution, sized to the largest consumer of it.
    need = {}
    for s in scenarios:
        need[s["resolution"]] = max(need.get(s["resolution"], 0), s["num_images"])
    for ls in latency_scenarios:
        need[ls["resolution"]] = max(need.get(ls["resolution"], 0), ls["batch_size"])
    sources = {
        res: _BatchSource(
            _get_cached_uint8(dataset_name, dataset_split, seed, n, res),
            image_mean, image_std, dtype,
        )
        for res, n in sorted(need.items())
    }

    # Per-(resolution, batch) warmup
    seen_shapes = set()
    for s in scenarios + latency_scenarios:
        key = (s["resolution"], s.get("batch_size", 1))
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        res, bs = key
        print(f"Warmup at {res}x{res} bs={bs} for >={_WARMUP_SECONDS:.0f}s",
              file=sys.stderr, flush=True)
        warm = sources[res].batch(0, bs)
        with torch.no_grad():
            t_end = time.perf_counter() + _WARMUP_SECONDS
            iters = 0
            while True:
                _ = model(warm)
                iters += 1
                if iters >= 3 and time.perf_counter() >= t_end:
                    break
            torch.cuda.synchronize()
        print(f"  warmup iters={iters}", file=sys.stderr, flush=True)
        del warm

    all_results = []
    for scenario in scenarios:
        res = scenario["resolution"]
        batch_size = scenario["batch_size"]
        src = sources[res]
        num_images = min(scenario["num_images"], len(src))
        num_batches = (num_images + batch_size - 1) // batch_size
        embed_max_batch = (EMBED_SAVE_CAP + batch_size - 1) // batch_size

        total_elapsed = 0.0
        total_images = 0
        desc = f"{label} {scenario['name']} ({res}x{res} bs={batch_size})"
        pbar = tqdm(range(num_batches), desc=desc, unit="batch", file=sys.stderr)
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_images)
            actual_bs = end - start

            x = src.batch(start, end)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - t0
            total_elapsed += batch_elapsed
            total_images += actual_bs

            if embed_dir and batch_idx < embed_max_batch:
                torch.save(
                    out.detach().cpu(),
                    os.path.join(
                        embed_dir,
                        f"{scenario['name']}_batch{batch_idx:04d}.pt",
                    ),
                )

            del x
            pbar.set_postfix(
                imgs=total_images,
                ips=f"{total_images / total_elapsed:.2f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_images": total_images,
            "images_per_second": (
                total_images / total_elapsed if total_elapsed > 0 else 0
            ),
        })

    latency_results = []
    for ls in latency_scenarios:
        res = ls["resolution"]
        batch_size = ls["batch_size"]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 10)

        x = sources[res].batch(0, batch_size)

        # At least num_warmup iterations AND at least _WARMUP_SECONDS of
        # continuous work, so the timed iterations start from the same steady
        # state on both sides (see _WARMUP_SECONDS).
        with torch.no_grad():
            t_end = time.perf_counter() + _WARMUP_SECONDS
            it = 0
            while True:
                torch.cuda.synchronize()
                _ = model(x)
                torch.cuda.synchronize()
                it += 1
                if it >= num_warmup and time.perf_counter() >= t_end:
                    break
        print(f"{label} warmup {ls['name']}: {it} iters",
              file=sys.stderr, flush=True)

        latencies = []
        for _ in tqdm(
            range(num_iters), desc=f"{label} latency {ls['name']}",
            file=sys.stderr,
        ):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        del x
        latency_results.append({
            "name": ls["name"],
            "resolution": res,
            "batch_size": batch_size,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del model
    torch.cuda.empty_cache()
    return {"throughput": all_results, "latency": latency_results}
'''

TIMM_WORKER = _WORKER_COMMON + r'''

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    import timm

    timm_name = cfg["timm_name"]
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    print(f"Loading timm model: {timm_name}", file=sys.stderr, flush=True)

    # Hierarchical models (SwinV2) need strict_img_size=False for variable
    # resolution benchmarks; flat ViTs ignore this kwarg.
    create_kwargs = {}
    if cfg.get("strict_img_size") is False:
        create_kwargs["strict_img_size"] = False

    model = timm.create_model(
        timm_name, pretrained=True, **create_kwargs,
    ).to(device="cuda", dtype=dtype).eval()

    results = run_benchmark(model, cfg, "timm")

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# fastkernels subprocess worker
# ---------------------------------------------------------------------------
FASTKERNELS_WORKER = _WORKER_COMMON + r'''

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    kb_module = cfg["kb_module"]
    kb_class = cfg["kb_class"]
    timm_name = cfg["timm_name"]
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    mod = __import__(
        f"{pkg}.tasks.baseline.L4.{kb_module}",
        fromlist=[kb_class],
    )
    ModelClass = getattr(mod, kb_class)

    print(
        f"Loading fastkernels model: {kb_class} from timm={timm_name}",
        file=sys.stderr, flush=True,
    )
    model = ModelClass.from_timm(timm_name).to(device="cuda", dtype=dtype).eval()

    results = run_benchmark(model, cfg, "fastkernels")

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _build_throughput_scenarios(default_res: int) -> list[dict]:
    scenarios = []
    for w in VISION_ENCODER_THROUGHPUT_WORKLOADS:
        res = w.resolution if w.resolution > 0 else default_res
        scenarios.append({
            "name": w.name,
            "resolution": res,
            "num_images": w.num_images,
            "batch_size": w.batch_size,
        })
    return scenarios


def _build_latency_scenarios(default_res: int) -> list[dict]:
    scenarios = []
    for w in VISION_ENCODER_LATENCY_WORKLOADS:
        res = w.resolution if w.resolution > 0 else default_res
        scenarios.append({
            "name": w.name,
            "resolution": res,
            "batch_size": w.batch_size,
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def _print_throughput_comparison(
    kb_results: list[dict],
    ref_results: list[dict] | None,
):
    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (images/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Images':>7} {'fastkernels':>12}"
    if ref_results:
        header += f" {'timm':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb['num_images']:>7} {kb['images_per_second']:>12.2f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                speedup = kb["images_per_second"] / ref["images_per_second"] if ref["images_per_second"] > 0 else 0
                line += f" {ref['images_per_second']:>12.2f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(
    kb_results: list[dict],
    ref_results: list[dict] | None,
):
    print("\n" + "=" * 90)
    print("  LATENCY COMPARISON (seconds)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'fastkernels p50':>12} {'fastkernels p99':>12}"
    if ref_results:
        header += f" {'timm p50':>12} {'timm p99':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 80)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        kb_p99 = np.percentile(kb_lats, 99)
        line = f"  {kb['name']:<25} {kb_p50:>12.4f} {kb_p99:>12.4f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                ref_lats = np.array(ref["latencies"])
                ref_p50 = np.percentile(ref_lats, 50)
                ref_p99 = np.percentile(ref_lats, 99)
                speedup = ref_p50 / kb_p50 if kb_p50 > 0 else 0
                line += f" {ref_p50:>12.4f} {ref_p99:>12.4f} {speedup:>9.2f}x"
        print(line)
    print()


def _compare_embeddings(kb_embed_dir: str, ref_embed_dir: str) -> dict:
    import torch

    kb_files = sorted(
        f for f in os.listdir(kb_embed_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_embed_dir) else []
    ref_files = sorted(
        f for f in os.listdir(ref_embed_dir) if f.endswith(".pt")
    ) if os.path.isdir(ref_embed_dir) else []

    common = sorted(set(kb_files) & set(ref_files))
    if not common:
        return {}

    scenario_stats: dict[str, list[dict]] = defaultdict(list)

    for fname in common:
        kb_emb = torch.load(
            os.path.join(kb_embed_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        ref_emb = torch.load(
            os.path.join(ref_embed_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()

        if len(kb_emb) != len(ref_emb):
            print(
                f"  WARNING: shape mismatch for {fname}: "
                f"fastkernels={kb_emb.shape} vs timm={ref_emb.shape}, skipping",
                file=sys.stderr,
            )
            continue

        kb_v = kb_emb.numpy()
        ref_v = ref_emb.numpy()

        cos_sim = float(
            np.dot(kb_v, ref_v)
            / (np.linalg.norm(kb_v) * np.linalg.norm(ref_v) + 1e-12)
        )
        mse = float(np.mean((kb_v - ref_v) ** 2))

        scenario_name = fname.rsplit("_batch", 1)[0]
        scenario_stats[scenario_name].append({
            "file": fname, "mse": mse, "cosine_similarity": cos_sim,
        })

    results = {}
    for scenario, batches in scenario_stats.items():
        cosines = [b["cosine_similarity"] for b in batches]
        mses = [b["mse"] for b in batches]
        results[scenario] = {
            "num_batches": len(batches),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
        }
    return results


def _print_correctness_comparison(correctness: dict):
    print("\n" + "=" * 100)
    print("  CORRECTNESS COMPARISON (embedding cosine similarity)")
    print("=" * 100)
    print(
        f"  {'Scenario':<25} {'Batches':>8} {'Mean CosSim':>12} "
        f"{'Min CosSim':>11} {'Mean MSE':>12} {'Max MSE':>12} {'Result':>8}"
    )
    print("  " + "-" * 94)

    all_pass = True
    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        min_cos = stats["min_cosine_sim"]
        if mean_cos >= 0.99 and min_cos >= 0.95:
            verdict = "PASS"
        elif mean_cos >= 0.95:
            verdict = "WARN"
        else:
            verdict = "FAIL"
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_batches']:>8} "
            f"{mean_cos:>12.6f} {min_cos:>11.6f} "
            f"{stats['mean_mse']:>12.2e} {stats['max_mse']:>12.2e} "
            f"{verdict:>8}"
        )
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vision encoder benchmark: fastkernels vs timm",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. google/siglip2-so400m-patch16-naflex)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-timm", action="store_true",
        help="Skip timm reference; only benchmark fastkernels (no correctness)",
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--num-images", type=int, default=None,
        help="Override num_images for throughput workloads",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch_size for throughput workloads",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Override resolution for all workloads",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results (default: tests/results/<gpu>/<model_short>)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override HF dataset name (default: ILSVRC/imagenet-1k)",
    )
    parser.add_argument(
        "--dataset-split", type=str, default=None,
        help="Override dataset split (default: validation)",
    )
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model {args.model!r}")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_info = MODEL_REGISTRY[args.model]
    gpu_name = _detect_gpu_name()

    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu_name / model_info["short_name"])

    default_res = args.resolution or model_info["default_resolution"]

    dataset_name = args.dataset or VISION_ENCODER_THROUGHPUT_WORKLOADS[0].dataset_name
    dataset_split = args.dataset_split or VISION_ENCODER_THROUGHPUT_WORKLOADS[0].dataset_split

    print(f"\nBenchmark: Vision Encoder on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"  timm name: {model_info['timm_name']}")
    print(f"  kb module: {model_info['kb_module']}.{model_info['kb_class']}")
    print(f"  default resolution: {default_res}")
    print(f"  dtype: {args.dtype}")
    print(f"Dataset: {dataset_name} ({dataset_split})")
    print(f"  image_mean: {model_info['image_mean']}")
    print(f"  image_std: {model_info['image_std']}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {args.output_dir}")

    run_timm = not args.skip_timm
    save_embeddings = run_timm and not args.skip_throughput

    os.makedirs(args.output_dir, exist_ok=True)
    if save_embeddings:
        kb_embed_dir = os.path.join(args.output_dir, "embeddings", "fastkernels")
        ref_embed_dir = os.path.join(args.output_dir, "embeddings", "timm")
    else:
        kb_embed_dir = None
        ref_embed_dir = None

    scenarios = _build_throughput_scenarios(default_res) if not args.skip_throughput else []
    latency_scenarios = _build_latency_scenarios(default_res) if not args.skip_latency else []

    # Apply model-specific default num_images, then CLI overrides
    default_num_images = model_info.get("default_num_images")
    if args.num_images is not None:
        for s in scenarios:
            s["num_images"] = args.num_images
    elif default_num_images is not None:
        for s in scenarios:
            s["num_images"] = default_num_images
    if args.batch_size is not None:
        for s in scenarios:
            s["batch_size"] = args.batch_size
    if args.resolution is not None:
        for s in scenarios:
            s["resolution"] = args.resolution
        for s in latency_scenarios:
            s["resolution"] = args.resolution

    base_config = {
        "timm_name": model_info["timm_name"],
        "kb_module": model_info["kb_module"],
        "kb_class": model_info["kb_class"],
        "seed": args.seed,
        "dtype": args.dtype,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "fastkernels",
        "warmup_resolution": default_res,
        "image_mean": model_info["image_mean"],
        "image_std": model_info["image_std"],
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
    }
    if "strict_img_size" in model_info:
        base_config["strict_img_size"] = model_info["strict_img_size"]

    # --- fastkernels benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
    }
    if kb_embed_dir:
        kb_config["embed_dir"] = kb_embed_dir
    kb_data = run_worker(
        FASTKERNELS_WORKER, kb_config,
        "fastkernels vision encoder benchmark", timeout=36000,
    )

    # --- timm benchmark ---
    ref_data = None
    if run_timm:
        ref_config = {
            **base_config,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
        }
        if ref_embed_dir:
            ref_config["embed_dir"] = ref_embed_dir
        ref_data = run_worker(
            TIMM_WORKER, ref_config,
            "timm vision encoder benchmark", timeout=36000,
        )

    # --- Print results ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        ref_tp = ref_data.get("throughput", []) if ref_data else None
        ref_lat = ref_data.get("latency", []) if ref_data else None

        if kb_tp:
            _print_throughput_comparison(kb_tp, ref_tp)
        if kb_lat:
            _print_latency_comparison(kb_lat, ref_lat)

        correctness = None
        if save_embeddings and kb_embed_dir and ref_embed_dir:
            correctness = _compare_embeddings(kb_embed_dir, ref_embed_dir)
            if correctness:
                _print_correctness_comparison(correctness)
            else:
                print(
                    "\n  WARNING: No matching embedding files found "
                    "for correctness comparison."
                )

        results_path = os.path.join(args.output_dir, "results.json")
        results = {
            "model": args.model,
            "timm_name": model_info["timm_name"],
            "gpu": gpu_name,
            "seed": args.seed,
            "dtype": args.dtype,
            "dataset": dataset_name,
            "dataset_split": dataset_split,
            "fastkernels": kb_data,
        }
        if ref_data:
            results["timm"] = ref_data
        if correctness:
            results["correctness"] = correctness
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")
    else:
        print("ERROR: fastkernels benchmark failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
