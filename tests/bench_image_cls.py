#!/usr/bin/env python3
"""Benchmark repo-native image classification models against official baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PACKAGE_DIR))

from bench.utils.worker import run_worker
from infra.image_cls_loader import infer_image_mean_std, infer_image_size


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


IMAGE_CLS_WORKER = r'''
import json, os, sys, time
import importlib.util
from pathlib import Path
import torch

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

from fastkernels.infra.image_cls_loader import load_ours_model, load_reference_model


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output


def _load_pil_images(dataset_name, dataset_split, num_needed, seed):
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError
    from PIL import Image

    try:
        ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    except DatasetNotFoundError as exc:
        if "gated dataset" in str(exc).lower():
            raise RuntimeError(
                f"{dataset_name}:{dataset_split} requires gated Hub access. "
                f"Either request access/login for that dataset, or use an ungated dataset such as "
                f"'food101' with '--dataset-split validation'.",
            ) from exc
        raise
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    images = []
    for sample in ds:
        img = sample.get("image") or sample.get("img")
        if img is None:
            for value in sample.values():
                if hasattr(value, "convert"):
                    img = value
                    break
        if img is None:
            continue
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO

            img = Image.open(BytesIO(img["bytes"]))
        images.append(img.convert("RGB"))
        if len(images) >= num_needed:
            break
    return images


def _preprocess_batch(pil_images, resolution, image_mean, image_std, dtype):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
    tensors = [transform(img) for img in pil_images]
    return torch.stack(tensors).to(device="cuda", dtype=dtype)


def _preprocess_batches(raw_images, resolution, image_mean, image_std, dtype, batch_size, num_images):
    return [
        _preprocess_batch(raw_images[start:start + batch_size], resolution, image_mean, image_std, dtype)
        for start in range(0, num_images, batch_size)
    ]


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    device = cfg.get("device", "cuda")
    dtype = torch.float16 if cfg.get("use_fp16", True) else torch.float32
    model_name = cfg["model"]
    backend = cfg["backend"]
    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    num_images = cfg["num_images"]
    dataset_name = cfg["dataset_name"]
    dataset_split = cfg["dataset_split"]
    image_mean = cfg["image_mean"]
    image_std = cfg["image_std"]

    max_needed = max(
        num_images,
        batch_size,
        cfg["alignment_images"],
        *(cfg.get("latency_batch_sizes", []) or [0]),
    )
    raw_images = _load_pil_images(dataset_name, dataset_split, max_needed, cfg["seed"])
    if len(raw_images) < max_needed:
        raise RuntimeError(
            f"Requested {max_needed} images from {dataset_name}:{dataset_split}, got {len(raw_images)}",
        )

    if backend == "ours":
        model = load_ours_model(model_name, device=device, dtype=dtype)
        baseline_name = "fastkernels"
    else:
        model, baseline_name = load_reference_model(model_name, device=device, dtype=dtype)

    throughput_batches = _preprocess_batches(
        raw_images, image_size, image_mean, image_std, dtype, batch_size, num_images,
    )
    warm_batch = throughput_batches[0][:1]
    with torch.inference_mode():
        logits = _extract_logits(model(warm_batch))

    for _ in range(cfg["warmup_iters"]):
        for x in throughput_batches:
            with torch.inference_mode():
                _ = _extract_logits(model(x))
    torch.cuda.synchronize()

    total_elapsed = 0.0
    total_images = 0
    for _ in range(cfg["measure_iters"]):
        for x in throughput_batches:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = _extract_logits(model(x))
            torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0
            total_images += x.shape[0]

    avg_elapsed = total_elapsed / max(cfg["measure_iters"], 1)
    avg_images = total_images / max(cfg["measure_iters"], 1)

    result = {
        "backend": backend,
        "baseline_name": baseline_name,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "elapsed": avg_elapsed,
        "images_per_second": (avg_images / avg_elapsed) if avg_elapsed > 0 else 0.0,
        "logits_shape": list(logits.shape),
        "logits_sample": logits[:, :64].float().cpu().tolist(),
    }

    latencies = {}
    latency_batches = {
        bs: _preprocess_batch(raw_images[:bs], image_size, image_mean, image_std, dtype)
        for bs in cfg.get("latency_batch_sizes", [])
    }
    for bs in cfg.get("latency_batch_sizes", []):
        batch = latency_batches[bs]
        samples = []
        for _ in range(cfg["latency_iters"]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = _extract_logits(model(batch))
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        latencies[str(bs)] = {
            "median": sorted(samples)[len(samples) // 2],
            "mean": sum(samples) / len(samples),
        }
    result["latency"] = latencies

    align_batch = _preprocess_batch(raw_images[:cfg["alignment_images"]], image_size, image_mean, image_std, dtype)
    with torch.inference_mode():
        align_logits = _extract_logits(model(align_batch))
    result["alignment_logits"] = align_logits.float().cpu().tolist()
    result["alignment_top1"] = align_logits.argmax(dim=-1).cpu().tolist()

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    os._exit(0)


if __name__ == "__main__":
    main()
'''


def _cosine(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


def _mae(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def _compare_outputs(ours, ref):
    top1_match = sum(int(a == b) for a, b in zip(ours["alignment_top1"], ref["alignment_top1"])) / max(len(ours["alignment_top1"]), 1)
    latency_ratio = {}
    for bs, stats in ours.get("latency", {}).items():
        if bs in ref.get("latency", {}):
            latency_ratio[bs] = ref["latency"][bs]["median"] / stats["median"]
    return {
        "throughput_ratio": ours["images_per_second"] / ref["images_per_second"],
        "logits_cosine": _cosine(ours["alignment_logits"], ref["alignment_logits"]),
        "logits_mae": _mae(ours["alignment_logits"], ref["alignment_logits"]),
        "top1_match_rate": top1_match,
        "latency_ratio": latency_ratio,
    }


def _summarize_backend_result(result):
    if not result:
        return None
    return {
        "baseline_name": result.get("baseline_name"),
        "images_per_second": result.get("images_per_second"),
        "latency": result.get("latency", {}),
        "logits_shape": result.get("logits_shape"),
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark repo-native CNN models vs official baselines")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=0)
    ap.add_argument("--dataset", type=str, default="food101")
    ap.add_argument("--dataset-split", type=str, default="validation")
    ap.add_argument("--num-images", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--alignment-images", type=int, default=8)
    ap.add_argument("--latency-batch-sizes", type=str, default="1,8")
    ap.add_argument("--latency-iters", type=int, default=5)
    ap.add_argument("--warmup-iters", type=int, default=1)
    ap.add_argument("--measure-iters", type=int, default=3)
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    image_size = args.image_size or infer_image_size(args.model)
    image_mean, image_std = infer_image_mean_std(args.model)
    gpu = _detect_gpu_name()
    out_dir = Path(args.output_dir or f"tests/results/{gpu}/{Path(args.model).name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "project_root": str(_PACKAGE_DIR),
        "model": args.model,
        "image_size": image_size,
        "dataset_name": args.dataset,
        "dataset_split": args.dataset_split,
        "image_mean": image_mean,
        "image_std": image_std,
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "alignment_images": args.alignment_images,
        "latency_batch_sizes": [int(x) for x in args.latency_batch_sizes.split(",") if x],
        "latency_iters": args.latency_iters,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "use_fp16": args.use_fp16,
        "device": "cuda",
        "seed": 1234,
    }

    ours = run_worker(IMAGE_CLS_WORKER, {**common, "backend": "ours"}, "fastkernels image classification")
    references = {}
    comparisons = {}
    if not args.skip_reference:
        ref = run_worker(IMAGE_CLS_WORKER, {**common, "backend": "reference"}, "official image classification baseline")
        references["reference"] = ref
        if ours and ref:
            comparisons["reference"] = _compare_outputs(ours, ref)

    results = {
        "model": args.model,
        "dataset_name": args.dataset,
        "dataset_split": args.dataset_split,
        "image_size": image_size,
        "ours": ours,
        "references": references,
        "comparisons": comparisons,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(results, indent=2))
    summary = {
        "model": args.model,
        "dataset_name": args.dataset,
        "dataset_split": args.dataset_split,
        "image_size": image_size,
        "ours": _summarize_backend_result(ours),
        "references": {name: _summarize_backend_result(result) for name, result in references.items()},
        "comparisons": comparisons,
    }
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
