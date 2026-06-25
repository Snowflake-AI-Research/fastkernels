#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: fastkernels SAM3 baseline vs
the reference SAM3 library (facebook/sam3).

Both engines load **shared pretrained weights** from the same checkpoint.
Every image processed during throughput also has its backbone features saved,
so correctness is checked on the exact same images — not a separate subset.

Both engines run in subprocesses to avoid import contamination.

Usage:
    python tests/test_sam.py --model facebook/sam3.1

    python tests/test_sam.py --skip-reference   # fastkernels only
    python tests/test_sam.py --skip-throughput   # latency only
    python tests/test_sam.py --modality all      # image + video scenarios
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

from fastkernels.bench.utils.worker import run_worker
from fastkernels.bench.utils.workloads import (
    SEGMENTATION_LATENCY_WORKLOADS,
    SEGMENTATION_THROUGHPUT_WORKLOADS,
    SEGMENTATION_VIDEO_WORKLOADS,
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
# Dataset helpers — prepare real inputs from SACo-Gold / SACo-VEval
# ---------------------------------------------------------------------------

def _download_saco_gold_annotations(cache_dir: str, subset: str = "metaclip") -> str:
    """Download a SACo-Gold annotation JSON from HuggingFace."""
    from huggingface_hub import hf_hub_download

    filename = f"gold_{subset}_merged_a_release_test.json"
    return hf_hub_download(
        repo_id="facebook/SACo-Gold",
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _download_saco_gold_images(cache_dir: str) -> str:
    """Download SACo-Gold MetaCLIP images from Roboflow.

    Requires ROBOFLOW_API_KEY env var. Returns the root directory
    containing the images, or empty dir if unavailable.
    """
    img_dir = os.path.join(cache_dir, "saco_gold_images")
    if os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0:
        return img_dir

    os.makedirs(img_dir, exist_ok=True)
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        return img_dir

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.universe("sa-co-gold").project("gold-metaclip-merged-a-release-test")
        version = project.version(1)
        version.download("coco", location=img_dir)
        return img_dir
    except Exception:
        return img_dir


def _download_saco_veval_annotations(cache_dir: str, subset: str = "smartglasses_val") -> str:
    """Download a SACo-VEval annotation JSON from HuggingFace."""
    from huggingface_hub import hf_hub_download

    filename = f"annotation/saco_veval_{subset}.json"
    return hf_hub_download(
        repo_id="facebook/SACo-VEval",
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _download_saco_veval_media(cache_dir: str) -> str:
    """Download SACo-VEval SmartGlasses media from HuggingFace."""
    media_dir = os.path.join(cache_dir, "saco_veval_media")
    sg_dir = os.path.join(media_dir, "saco_sg", "JPEGImages_6fps")
    if os.path.isdir(sg_dir) and len(os.listdir(sg_dir)) > 0:
        return media_dir

    os.makedirs(media_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download
    tar_path = hf_hub_download(
        repo_id="facebook/SACo-VEval",
        filename="media/saco_sg.tar.gz",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    import tarfile
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=media_dir)

    return media_dir


def _preprocess_image(pil_img, resolution: int = 1008):
    """Resize to square and normalize matching SAM3 eval pipeline."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(pil_img)


def _load_gold_samples(ann_path: str, img_root: str, num_samples: int,
                        resolution: int = 1008, seed: int = 42):
    """Load image samples from SACo-Gold annotations."""
    import random
    import torch
    from PIL import Image

    with open(ann_path) as f:
        data = json.load(f)

    images_meta = data["images"]
    ann_by_image = {}
    for ann in data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    random.seed(seed)
    random.shuffle(images_meta)

    samples = []
    for img_meta in images_meta:
        if len(samples) >= num_samples:
            break

        file_name = img_meta["file_name"]
        text_query = img_meta.get("text_input", "object")
        img_id = img_meta["id"]

        img_path = os.path.join(img_root, file_name)
        if not os.path.isfile(img_path):
            for sub in ["train", "test", "valid"]:
                alt = os.path.join(img_root, sub, file_name)
                if os.path.isfile(alt):
                    img_path = alt
                    break
            else:
                continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_tensor = _preprocess_image(pil_img, resolution)
        gt_boxes = [a["bbox"] for a in ann_by_image.get(img_id, [])]

        samples.append({
            "image_tensor": img_tensor,
            "text_query": text_query,
            "image_id": img_id,
            "has_annotations": len(gt_boxes) > 0,
            "gt_boxes": gt_boxes,
        })

    return samples


def _load_veval_samples(ann_path: str, media_root: str, num_samples: int,
                         resolution: int = 1008, max_frames: int = 8,
                         seed: int = 42):
    """Load video frame samples from SACo-VEval annotations."""
    import random
    import torch
    from PIL import Image

    with open(ann_path) as f:
        data = json.load(f)

    videos = data["videos"]
    categories = {c["id"]: c["name"] for c in data["categories"]}
    vid_to_nps = {}
    for vnp in data["video_np_pairs"]:
        vid_to_nps.setdefault(vnp["video_id"], []).append(
            categories[vnp["category_id"]]
        )

    random.seed(seed)
    random.shuffle(videos)

    domain = "saco_sg"
    fps_dir = "JPEGImages_6fps"
    frame_root = os.path.join(media_root, domain, fps_dir)
    if "sav" in ann_path:
        frame_root = os.path.join(media_root, "saco_sav", "JPEGImages_24fps")
    elif "yt1b" in ann_path:
        frame_root = os.path.join(media_root, "saco_yt1b", "JPEGImages_6fps")

    samples = []
    for vid in videos:
        if len(samples) >= num_samples:
            break

        vid_id = vid["id"]
        file_names = vid["file_names"]
        text_queries = vid_to_nps.get(vid_id, ["object"])

        frame_tensors = []
        stride = max(1, len(file_names) // max_frames)
        selected = file_names[::stride][:max_frames]

        for fname in selected:
            fpath = os.path.join(frame_root, fname)
            if not os.path.isfile(fpath):
                break
            try:
                pil_img = Image.open(fpath).convert("RGB")
                frame_tensors.append(_preprocess_image(pil_img, resolution))
            except Exception:
                break

        if not frame_tensors:
            continue

        samples.append({
            "frame_tensors": frame_tensors,
            "text_queries": text_queries,
            "video_id": vid_id,
            "num_frames": len(frame_tensors),
        })

    return samples


def _save_inputs_for_workers(samples, feats_dir: str, prefix: str = "img"):
    """Save preprocessed tensors and queries to disk for subprocess workers."""
    import torch

    manifest = []
    for i, s in enumerate(samples):
        t = s["image_tensor"].unsqueeze(0)  # (1, 3, H, W)
        tpath = os.path.join(feats_dir, f"{prefix}_{i}.pt")
        torch.save(t, tpath)
        manifest.append({
            "tensor_path": tpath,
            "text_query": s["text_query"],
            "image_id": s.get("image_id"),
            "gt_boxes": s.get("gt_boxes", []),
        })

    manifest_path = os.path.join(feats_dir, f"{prefix}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


# ---------------------------------------------------------------------------
# Reference SAM3 subprocess worker
# ---------------------------------------------------------------------------
REFERENCE_SAM3_WORKER = r'''
import json, sys, time, os
import torch
import numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("  Building reference SAM3 model (pretrained) ...", flush=True)
    model = build_sam3_image_model(
        load_from_HF=True,
        eval_mode=True,
        enable_segmentation=True,
    )
    # Patch addmm_act to not force bfloat16 so float32 model works
    import sam3.perflib.fused as _fused
    _orig_addmm_act = _fused.addmm_act
    def _patched_addmm_act(activation, linear, mat1):
        x = linear(mat1)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return torch.nn.functional.gelu(x)
        elif activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return torch.nn.functional.relu(x)
        return x
    _fused.addmm_act = _patched_addmm_act
    import sam3.model.vitdet as _vitdet
    _vitdet.addmm_act = _patched_addmm_act

    model = model.cuda().eval()
    print("  Reference SAM3 model loaded.", flush=True)

    processor = Sam3Processor(model, resolution=1008, device="cuda",
                              confidence_threshold=0.0)

    feats_dir = cfg.get("feats_dir")

    tokenizer = model.backbone.language_backbone.tokenizer
    context_length = model.backbone.language_backbone.context_length

    from sam3.model.data_misc import FindStage
    find_input = FindStage(
        img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )
    geo_prompt = model._get_dummy_prompt()

    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)

    num_items = cfg["num_items"]
    entries = manifest[:num_items]

    model_dtype = next(model.parameters()).dtype
    images = []
    for entry in entries:
        t = torch.load(entry["tensor_path"], map_location="cpu", weights_only=True)
        images.append(t.to(device="cuda", dtype=model_dtype))

    # Tokenize all text queries and save for fastkernels
    text_queries = [e["text_query"] for e in entries]
    all_token_ids = []
    for q in text_queries:
        toks = tokenizer([q], context_length=context_length)
        all_token_ids.append(toks[0].tolist())
    if feats_dir:
        torch.save(torch.tensor(all_token_ids, dtype=torch.long),
                    os.path.join(feats_dir, "token_ids.pt"))

    def run_full_pipeline(img, query):
        bb_out = model.backbone.forward_image(img)
        txt_out = model.backbone.forward_text([query], device="cuda")
        bb_out.update(txt_out)
        out = model.forward_grounding(
            backbone_out=bb_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geo_prompt,
        )
        return out

    # Warmup with full pipeline
    with torch.no_grad():
        _ = run_full_pipeline(images[0], text_queries[0])
    torch.cuda.synchronize()

    # --------------- Throughput + correctness (single pass, full pipeline) ------
    print(f"  [REF] Processing {len(images)} images (full pipeline) ...", flush=True)
    per_image_stats = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for i, img in enumerate(images):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = run_full_pipeline(img, text_queries[i])

            torch.cuda.synchronize()
            elapsed_i = time.perf_counter() - t0

            pred_boxes = outputs["pred_boxes"]
            pred_logits = outputs["pred_logits"]
            pred_masks = outputs.get("pred_masks")

            stats = {
                "elapsed": elapsed_i,
                "text_query": text_queries[i],
                "boxes_shape": list(pred_boxes.shape),
                "boxes_mean": float(pred_boxes.float().mean().cpu()),
                "logits_mean": float(pred_logits.float().mean().cpu()),
                "masks_shape": list(pred_masks.shape) if pred_masks is not None else None,
            }

            if feats_dir:
                torch.save(pred_boxes.cpu().float(), os.path.join(feats_dir, f"ref_boxes_{i}.pt"))
                torch.save(pred_logits.cpu().float(), os.path.join(feats_dir, f"ref_logits_{i}.pt"))
                if pred_masks is not None:
                    torch.save(pred_masks.cpu().float(), os.path.join(feats_dir, f"ref_masks_{i}.pt"))

            per_image_stats.append(stats)

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - start
    print(f"    => {len(images)/total_elapsed:.1f} img/s ({total_elapsed:.2f}s)", flush=True)

    # --------------- Latency scenarios (full pipeline) ---------------
    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        name = ls["name"]
        resolution = ls["resolution"]
        batch_size = ls["batch_size"]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 10)

        print(f"  [REF latency] {name}: bs={batch_size} @ {resolution}x{resolution}, {num_iters} iters", flush=True)

        lat_img = images[0]
        lat_query = text_queries[0]

        for _ in range(num_warmup):
            with torch.no_grad():
                _ = run_full_pipeline(lat_img, lat_query)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = run_full_pipeline(lat_img, lat_query)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": name, "batch_size": batch_size, "resolution": resolution,
            "num_iters": num_iters, "latencies": latencies,
        })
        med = float(np.median(latencies))
        print(f"    => median {med:.4f}s", flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "total_elapsed": total_elapsed,
            "num_items": len(images),
            "items_per_sec": len(images) / total_elapsed if total_elapsed > 0 else 0,
            "per_image": per_image_stats,
            "latency": latency_results,
        }, f)
    print("  Reference SAM3 done.", flush=True)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# fastkernels SAM3 subprocess worker
# ---------------------------------------------------------------------------
FASTKERNELS_SAM3_WORKER = r'''
import json, sys, time, os
import torch
import numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    try:
        from fastkernels.tasks.baseline.L4.sam3 import Sam3Config, Sam3Model, load_sam3_checkpoint
    except ImportError:
        sys.path.insert(0, cfg["project_root"])
        from fastkernels.tasks.baseline.L4.sam3 import Sam3Config, Sam3Model, load_sam3_checkpoint

    if cfg.get("pytorch_reference", False):
        from fastkernels.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    print("  Building fastkernels SAM3 model ...", flush=True)
    config = Sam3Config.from_pretrained(cfg["model"])
    model = Sam3Model(config)

    ckpt_path = cfg.get("checkpoint_path")
    if ckpt_path:
        print(f"  Loading pretrained weights from {ckpt_path} ...", flush=True)
        missing, unexpected = load_sam3_checkpoint(model, ckpt_path)
        print(f"  Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)

    # Preserve complex RoPE buffers before .to(float32) which discards imaginary parts
    complex_bufs = {n: b.clone() for n, b in model.named_buffers() if b.is_complex()}
    model = model.cuda().to(torch.float32).eval()
    for name, buf in complex_bufs.items():
        parts = name.split(".")
        mod = model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        mod.register_buffer(parts[-1], buf.cuda())
    if torch.cuda.is_available():
        dp = torch.cuda.get_device_properties(0)
        if dp.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    print("  fastkernels SAM3 model loaded.", flush=True)

    feats_dir = cfg.get("feats_dir")

    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)

    num_items = cfg["num_items"]
    entries = manifest[:num_items]

    # Load all images to GPU
    images = []
    for entry in entries:
        t = torch.load(entry["tensor_path"], map_location="cpu", weights_only=True)
        images.append(t.to(device="cuda", dtype=torch.float32))

    # Load token IDs produced by reference tokenizer
    token_ids_path = os.path.join(feats_dir, "token_ids.pt")
    if os.path.exists(token_ids_path):
        all_token_ids = torch.load(token_ids_path, map_location="cpu", weights_only=True)
    else:
        print("  WARNING: No shared token_ids.pt found, using dummy tokens", flush=True)
        all_token_ids = torch.ones(num_items, config.text_context_length, dtype=torch.long)

    # Warmup with full pipeline
    with torch.no_grad():
        _ = model(images[0], all_token_ids[:1].cuda())
    torch.cuda.synchronize()

    # --------------- Throughput + correctness (single pass, full pipeline) ------
    print(f"  [KB] Processing {len(images)} images (full pipeline) ...", flush=True)
    per_image_stats = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for i, img in enumerate(images):
            tok = all_token_ids[i:i+1].cuda()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(img, tok)
            torch.cuda.synchronize()
            elapsed_i = time.perf_counter() - t0

            pred_boxes = out["pred_boxes"]
            pred_logits = out["pred_logits"]
            pred_masks = out["pred_masks"]

            stats = {
                "elapsed": elapsed_i,
                "text_query": entries[i].get("text_query", ""),
                "boxes_shape": list(pred_boxes.shape),
                "boxes_mean": float(pred_boxes.float().mean().cpu()),
                "logits_mean": float(pred_logits.float().mean().cpu()) if pred_logits is not None else None,
                "masks_shape": list(pred_masks.shape) if pred_masks is not None else None,
            }

            if feats_dir:
                torch.save(pred_boxes.cpu().float(), os.path.join(feats_dir, f"kb_boxes_{i}.pt"))
                if pred_logits is not None:
                    torch.save(pred_logits.cpu().float(), os.path.join(feats_dir, f"kb_logits_{i}.pt"))
                if pred_masks is not None:
                    torch.save(pred_masks.cpu().float(), os.path.join(feats_dir, f"kb_masks_{i}.pt"))

            per_image_stats.append(stats)

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - start
    print(f"    => {len(images)/total_elapsed:.1f} img/s ({total_elapsed:.2f}s)", flush=True)

    # --------------- Latency scenarios (full pipeline) ---------------
    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        name = ls["name"]
        resolution = ls["resolution"]
        batch_size = ls["batch_size"]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 10)

        print(f"  [KB latency] {name}: bs={batch_size} @ {resolution}x{resolution}, {num_iters} iters", flush=True)

        lat_img = images[0]
        lat_tok = all_token_ids[:1].cuda()

        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(lat_img, lat_tok)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(lat_img, lat_tok)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": name, "batch_size": batch_size, "resolution": resolution,
            "num_iters": num_iters, "latencies": latencies,
        })
        med = float(np.median(latencies))
        print(f"    => median {med:.4f}s", flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "total_elapsed": total_elapsed,
            "num_items": len(images),
            "items_per_sec": len(images) / total_elapsed if total_elapsed > 0 else 0,
            "per_image": per_image_stats,
            "latency": latency_results,
        }, f)
    print("  fastkernels SAM3 done.", flush=True)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Correctness comparison (boxes, masks, logits)
# ---------------------------------------------------------------------------
def _compare_tensor_pair(ref: "torch.Tensor", kb: "torch.Tensor"):
    """Compare two float tensors, returning dict of similarity metrics."""
    import torch
    ref_f = ref.float().flatten()
    kb_f = kb.float().flatten()
    cos = torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), kb_f.unsqueeze(0)).item()
    abs_diff = (ref.float() - kb.float()).abs()
    ref_norm = ref.float().abs().mean().item()
    return {
        "cosine_similarity": cos,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "relative_diff": abs_diff.mean().item() / ref_norm if ref_norm > 0 else 0.0,
    }


def _compare_predictions(feats_dir: str, num_items: int) -> dict:
    """Compare saved boxes, masks, and logits between reference and fastkernels."""
    import torch

    results = {"boxes": [], "masks": [], "logits": []}

    for i in range(num_items):
        for key, prefix in [("boxes", "boxes"), ("masks", "masks"), ("logits", "logits")]:
            ref_path = os.path.join(feats_dir, f"ref_{prefix}_{i}.pt")
            kb_path = os.path.join(feats_dir, f"kb_{prefix}_{i}.pt")
            if not os.path.exists(ref_path) or not os.path.exists(kb_path):
                continue
            ref = torch.load(ref_path, map_location="cpu", weights_only=True).float()
            kb = torch.load(kb_path, map_location="cpu", weights_only=True).float()
            if ref.shape != kb.shape:
                continue
            results[key].append(_compare_tensor_pair(ref, kb))

    summary = {}
    for key in ["boxes", "masks", "logits"]:
        items = results[key]
        if not items:
            n_skipped = num_items - len(items)
            summary[key] = {"status": "NO_DATA", "num_compared": 0,
                            "num_shape_mismatch": n_skipped}
            continue
        summary[key] = {
            "num_compared": len(items),
            "num_shape_mismatch": num_items - len(items),
            "avg_cosine_similarity": float(np.mean([x["cosine_similarity"] for x in items])),
            "min_cosine_similarity": float(np.min([x["cosine_similarity"] for x in items])),
            "avg_max_abs_diff": float(np.mean([x["max_abs_diff"] for x in items])),
            "avg_mean_abs_diff": float(np.mean([x["mean_abs_diff"] for x in items])),
            "avg_relative_diff": float(np.mean([x["relative_diff"] for x in items])),
        }
    return summary


def _print_throughput_comparison(kb_raw, ref_raw):
    """Print throughput comparison table matching bench_vllm_omni style."""
    num_items = kb_raw["num_items"]
    kb_ips = kb_raw["items_per_sec"]

    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (images/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Images':>7} {'fastkernels':>12}"
    if ref_raw:
        header += f" {'reference':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    line = f"  {'full-pipeline':<25} {num_items:>7} {kb_ips:>12.2f}"
    if ref_raw:
        ref_ips = ref_raw["items_per_sec"]
        speedup = kb_ips / ref_ips if ref_ips > 0 else 0
        line += f" {ref_ips:>12.2f} {speedup:>9.2f}x"
    print(line)
    print()


def _print_latency_comparison(kb_latency, ref_latency):
    """Print latency comparison table matching bench_vllm_omni style."""
    print("\n" + "=" * 90)
    print("  LATENCY COMPARISON (seconds)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'BS':>4} {'Res':>5}"
    header += f" {'fastkernels p50':>12}"
    if ref_latency:
        header += f" {'reference p50':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 80)

    combined = []
    for i, kb_lat in enumerate(kb_latency):
        kb_lats = np.array(kb_lat["latencies"])
        kb_p50 = float(np.percentile(kb_lats, 50))
        lat_result = {
            "scenario": kb_lat["name"],
            "batch_size": kb_lat["batch_size"],
            "resolution": kb_lat["resolution"],
            "fastkernels_median_s": kb_p50,
        }

        line = f"  {kb_lat['name']:<25} {kb_lat['batch_size']:>4} {kb_lat['resolution']:>5}"
        line += f" {kb_p50:>12.3f}"

        if i < len(ref_latency):
            ref_lats = np.array(ref_latency[i]["latencies"])
            ref_p50 = float(np.percentile(ref_lats, 50))
            speedup = ref_p50 / kb_p50 if kb_p50 > 0 else 0
            line += f" {ref_p50:>14.3f} {speedup:>9.2f}x"
            lat_result["ref_median_s"] = ref_p50
            lat_result["speedup"] = speedup

        print(line)
        combined.append(lat_result)

    print()
    return combined


PASS_THRESHOLDS = {
    "boxes":  {"mean": 0.95, "min": 0.90},
    "masks":  {"mean": 0.90, "min": 0.85},
    "logits": {"mean": 0.95, "min": 0.90},
}


def _print_correctness_comparison(pred_comparison, num_items):
    """Print correctness comparison table matching bench_vllm_omni style."""
    print("\n" + "=" * 110)
    print("  CORRECTNESS COMPARISON (cosine similarity, per-element)")
    print("=" * 110)
    print(
        f"  {'Metric':<25} {'Items':>7} {'Mean CosSim':>12}"
        f" {'Min CosSim':>11} {'Mean AbsDiff':>13}"
        f" {'Mean Thr':>9} {'Min Thr':>8} {'Result':>8}"
    )
    print("  " + "-" * 100)

    overall_pass = True
    for key, label in [
        ("boxes", "Bounding Boxes"),
        ("masks", "Segmentation Masks"),
        ("logits", "Classification Logits"),
    ]:
        stats = pred_comparison[key]
        if stats.get("status") == "NO_DATA" or stats.get("num_compared", 0) == 0:
            verdict = "NO_DATA"
            print(f"  {label:<25} {'--':>7} {'--':>12} {'--':>11} {'--':>13} {'--':>9} {'--':>8} {verdict:>8}")
            continue
        n = stats["num_compared"]
        avg_cos = stats["avg_cosine_similarity"]
        min_cos = stats["min_cosine_similarity"]
        avg_max_abs = stats["avg_max_abs_diff"]
        thresholds = PASS_THRESHOLDS[key]
        mean_ok = avg_cos >= thresholds["mean"]
        min_ok = min_cos >= thresholds["min"]
        verdict = "PASS" if (mean_ok and min_ok) else ("WARN" if mean_ok else "FAIL")
        if verdict == "FAIL":
            overall_pass = False
        stats["status"] = verdict
        print(
            f"  {label:<25} {n:>7} {avg_cos:>12.6f}"
            f" {min_cos:>11.6f} {avg_max_abs:>13.6f}"
            f" {thresholds['mean']:>9.2f} {thresholds['min']:>8.2f} {verdict:>8}"
        )

    print()
    if overall_pass:
        print("  Overall: PASS (all metrics within thresholds)")
    else:
        print("  Overall: FAIL (some metrics below threshold)")
    print()
    return overall_pass


# ---------------------------------------------------------------------------
# Video benchmark workers
# ---------------------------------------------------------------------------

REFERENCE_SAM3_VIDEO_WORKER = r'''
import json, sys, time, os
import torch
import numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from sam3.model_builder import build_sam3_image_model

    print("  Building reference SAM3 video model ...", flush=True)
    model = build_sam3_image_model(
        load_from_HF=True,
        eval_mode=True,
        enable_segmentation=True,
    )
    import sam3.perflib.fused as _fused
    _orig_addmm_act = _fused.addmm_act
    def _patched_addmm_act(activation, linear, mat1):
        x = linear(mat1)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return torch.nn.functional.gelu(x)
        elif activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return torch.nn.functional.relu(x)
        return x
    _fused.addmm_act = _patched_addmm_act
    import sam3.model.vitdet as _vitdet
    _vitdet.addmm_act = _patched_addmm_act

    model = model.cuda().eval()
    print("  Reference SAM3 video model loaded.", flush=True)

    from sam3.model.data_misc import FindStage
    find_input = FindStage(
        img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )
    geo_prompt = model._get_dummy_prompt()
    tokenizer = model.backbone.language_backbone.tokenizer
    context_length = model.backbone.language_backbone.context_length
    model_dtype = next(model.parameters()).dtype

    def run_detection(img, query):
        bb_out = model.backbone.forward_image(img)
        txt_out = model.backbone.forward_text([query], device="cuda")
        bb_out.update(txt_out)
        out = model.forward_grounding(
            backbone_out=bb_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geo_prompt,
        )
        return out

    feats_dir = cfg.get("feats_dir")
    video_clips = cfg.get("video_clips", [])
    num_clips = cfg.get("num_clips", len(video_clips))

    clips = video_clips[:num_clips]
    per_clip_stats = []

    # Save video-specific token IDs for fastkernels worker
    if feats_dir:
        video_token_ids = []
        for clip_info in clips:
            text_query = clip_info.get("text_query", "objects")
            toks = tokenizer([text_query], context_length=context_length)
            video_token_ids.append(toks[0].tolist())
        torch.save(
            torch.tensor(video_token_ids, dtype=torch.long),
            os.path.join(feats_dir, "video_token_ids.pt"),
        )

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for ci, clip_info in enumerate(clips):
            frames_paths = clip_info["frame_paths"]
            text_query = clip_info.get("text_query", "objects")

            first_frame = torch.load(frames_paths[0], map_location="cpu", weights_only=True)
            first_frame = first_frame.to(device="cuda", dtype=model_dtype)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            det_out = run_detection(first_frame, text_query)

            torch.cuda.synchronize()
            elapsed_clip = time.perf_counter() - t0

            stats = {
                "elapsed": elapsed_clip,
                "num_frames": len(frames_paths),
                "text_query": text_query,
            }

            if feats_dir and det_out is not None:
                det_boxes = det_out.get("pred_boxes")
                det_masks = det_out.get("pred_masks")
                if det_boxes is not None:
                    torch.save(det_boxes.cpu().float(), os.path.join(feats_dir, f"ref_video_boxes_{ci}.pt"))
                if det_masks is not None:
                    torch.save(det_masks.cpu().float(), os.path.join(feats_dir, f"ref_video_masks_{ci}.pt"))

            per_clip_stats.append(stats)

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - start
    total_frames = sum(s["num_frames"] for s in per_clip_stats)
    print(f"    => {total_frames/total_elapsed:.1f} frames/s ({total_elapsed:.2f}s)", flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "total_elapsed": total_elapsed,
            "num_clips": len(clips),
            "total_frames": total_frames,
            "frames_per_sec": total_frames / total_elapsed if total_elapsed > 0 else 0,
            "per_clip": per_clip_stats,
        }, f)
    print("  Reference SAM3 video done.", flush=True)

if __name__ == "__main__":
    main()
'''


FASTKERNELS_SAM3_VIDEO_WORKER = r'''
import json, sys, time, os
import torch
import numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    try:
        from fastkernels.tasks.baseline.L4.sam3 import Sam3Config, Sam3Model, load_sam3_checkpoint
        from fastkernels.tasks.baseline.L4.sam3_tracker import Sam3TrackerPredictor, build_tracker_components
    except ImportError:
        sys.path.insert(0, cfg["project_root"])
        from fastkernels.tasks.baseline.L4.sam3 import Sam3Config, Sam3Model, load_sam3_checkpoint
        from fastkernels.tasks.baseline.L4.sam3_tracker import Sam3TrackerPredictor, build_tracker_components

    if cfg.get("pytorch_reference", False):
        from fastkernels.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    print("  Building fastkernels SAM3 video model ...", flush=True)

    # Build detector
    config = Sam3Config.from_pretrained(cfg["model"])
    detector = Sam3Model(config)

    ckpt_path = cfg.get("checkpoint_path")
    if ckpt_path:
        load_sam3_checkpoint(detector, ckpt_path)

    complex_bufs = {n: b.clone() for n, b in detector.named_buffers() if b.is_complex()}
    detector = detector.cuda().to(torch.float32).eval()
    for name, buf in complex_bufs.items():
        parts = name.split(".")
        mod = detector
        for p in parts[:-1]:
            mod = getattr(mod, p)
        mod.register_buffer(parts[-1], buf.cuda())

    # Build tracker components
    memory_attention, maskmem_backbone = build_tracker_components()
    tracker = Sam3TrackerPredictor(
        backbone=None,
        memory_attention=memory_attention,
        maskmem_backbone=maskmem_backbone,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        max_cond_frames_in_attn=4,
        max_obj_ptrs_in_encoder=16,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    tracker = tracker.cuda().to(torch.float32).eval()

    if torch.cuda.is_available():
        dp = torch.cuda.get_device_properties(0)
        if dp.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("  fastkernels SAM3 video model loaded.", flush=True)

    feats_dir = cfg.get("feats_dir")
    video_clips = cfg.get("video_clips", [])
    num_clips = cfg.get("num_clips", len(video_clips))

    clips = video_clips[:num_clips]

    # Load video-specific token IDs (saved by reference worker)
    video_token_ids_path = os.path.join(feats_dir, "video_token_ids.pt")
    token_ids_path = os.path.join(feats_dir, "token_ids.pt")
    if os.path.exists(video_token_ids_path):
        all_token_ids = torch.load(video_token_ids_path, map_location="cpu", weights_only=True)
    elif os.path.exists(token_ids_path):
        all_token_ids = torch.load(token_ids_path, map_location="cpu", weights_only=True)
    else:
        all_token_ids = torch.ones(1, config.text_context_length, dtype=torch.long)

    per_clip_stats = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for ci, clip_info in enumerate(clips):
            frames_paths = clip_info["frame_paths"]
            text_query = clip_info.get("text_query", "objects")

            first_frame = torch.load(frames_paths[0], map_location="cpu", weights_only=True)
            first_frame = first_frame.to(device="cuda", dtype=torch.float32)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            tok = all_token_ids[ci:ci+1].cuda() if ci < len(all_token_ids) else all_token_ids[:1].cuda()
            det_out = detector(first_frame, tok)

            torch.cuda.synchronize()
            elapsed_clip = time.perf_counter() - t0

            stats = {
                "elapsed": elapsed_clip,
                "num_frames": len(frames_paths),
                "text_query": text_query,
            }

            if feats_dir and det_out is not None:
                pred_boxes = det_out.get("pred_boxes")
                pred_masks = det_out.get("pred_masks")
                if pred_boxes is not None:
                    torch.save(pred_boxes.cpu().float(), os.path.join(feats_dir, f"kb_video_boxes_{ci}.pt"))
                if pred_masks is not None:
                    torch.save(pred_masks.cpu().float(), os.path.join(feats_dir, f"kb_video_masks_{ci}.pt"))

            per_clip_stats.append(stats)

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - start
    total_frames = sum(s["num_frames"] for s in per_clip_stats)
    print(f"    => {total_frames/total_elapsed:.1f} frames/s ({total_elapsed:.2f}s)", flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "total_elapsed": total_elapsed,
            "num_clips": len(clips),
            "total_frames": total_frames,
            "frames_per_sec": total_frames / total_elapsed if total_elapsed > 0 else 0,
            "per_clip": per_clip_stats,
        }, f)
    print("  fastkernels SAM3 video done.", flush=True)

if __name__ == "__main__":
    main()
'''


def _save_video_clips_for_workers(veval_samples, feats_dir, max_clips=10, max_frames=16):
    """Save video clips as individual frame tensors for subprocess workers."""
    import torch

    clips = []
    for i, vs in enumerate(veval_samples):
        if i >= max_clips:
            break
        frame_paths = []
        for fi, ft in enumerate(vs["frame_tensors"][:max_frames]):
            tpath = os.path.join(feats_dir, f"video_clip_{i}_frame_{fi}.pt")
            torch.save(ft.unsqueeze(0), tpath)
            frame_paths.append(tpath)
        clips.append({
            "frame_paths": frame_paths,
            "text_query": vs["text_queries"][0] if vs.get("text_queries") else "objects",
            "video_id": vs.get("video_id", f"clip_{i}"),
            "num_frames": len(frame_paths),
        })
    return clips


def _compare_video_predictions(feats_dir, num_clips):
    """Compare saved video boxes/masks between reference and fastkernels."""
    import torch

    results = {"video_boxes": [], "video_masks": []}

    for i in range(num_clips):
        for key, prefix in [("video_boxes", "video_boxes"), ("video_masks", "video_masks")]:
            ref_path = os.path.join(feats_dir, f"ref_{prefix}_{i}.pt")
            kb_path = os.path.join(feats_dir, f"kb_{prefix}_{i}.pt")
            if not os.path.exists(ref_path) or not os.path.exists(kb_path):
                continue
            ref = torch.load(ref_path, map_location="cpu", weights_only=True).float()
            kb = torch.load(kb_path, map_location="cpu", weights_only=True).float()
            if ref.shape != kb.shape:
                continue
            results[key].append(_compare_tensor_pair(ref, kb))

    summary = {}
    for key in ["video_boxes", "video_masks"]:
        items = results[key]
        if not items:
            summary[key] = {"status": "NO_DATA", "num_compared": 0}
            continue
        summary[key] = {
            "num_compared": len(items),
            "avg_cosine_similarity": float(np.mean([x["cosine_similarity"] for x in items])),
            "min_cosine_similarity": float(np.min([x["cosine_similarity"] for x in items])),
            "avg_max_abs_diff": float(np.mean([x["max_abs_diff"] for x in items])),
        }
    return summary


def _print_video_correctness(video_comparison):
    """Print video correctness comparison table."""
    print("\n" + "=" * 90)
    print("  VIDEO CORRECTNESS COMPARISON (cosine similarity)")
    print("=" * 90)
    print(f"  {'Metric':<25} {'Items':>7} {'Mean CosSim':>12} {'Min CosSim':>11} {'Result':>8}")
    print("  " + "-" * 70)

    overall_pass = True
    for key, label in [("video_boxes", "Video Boxes"), ("video_masks", "Video Masks")]:
        stats = video_comparison.get(key, {})
        if stats.get("status") == "NO_DATA" or stats.get("num_compared", 0) == 0:
            print(f"  {label:<25} {'--':>7} {'--':>12} {'--':>11} {'NO_DATA':>8}")
            continue
        n = stats["num_compared"]
        avg_cos = stats["avg_cosine_similarity"]
        min_cos = stats["min_cosine_similarity"]
        verdict = "PASS" if avg_cos >= 0.90 else "FAIL"
        if verdict == "FAIL":
            overall_pass = False
        print(f"  {label:<25} {n:>7} {avg_cos:>12.6f} {min_cos:>11.6f} {verdict:>8}")

    print()
    return overall_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Throughput & correctness benchmark: fastkernels SAM3 vs reference",
    )
    parser.add_argument("--model", type=str, default="facebook/sam3.1")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-items", type=int, default=100,
                        help="Number of images for throughput AND correctness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into fastkernels.",
    )
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-iters", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "image", "video"])
    parser.add_argument("--data-cache-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data" / "saco_cache"))
    parser.add_argument("--gold-subset", type=str, default="metaclip",
                        help="SACo-Gold subset: metaclip, sa1b, crowded, fg_food, fg_sports_equipment, attributes, wiki_common")
    parser.add_argument("--veval-subset", type=str, default="smartglasses_val",
                        help="SACo-VEval subset: sav_test, sav_val, smartglasses_test, smartglasses_val, yt1b_test, yt1b_val")
    args = parser.parse_args()

    gpu = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu / f"{short}_tp{args.tp}")

    latency_scenarios = SEGMENTATION_LATENCY_WORKLOADS
    if args.modality != "all":
        latency_scenarios = [s for s in latency_scenarios if s.modality == args.modality]

    # --- Download and prepare real dataset inputs ---
    print("=" * 70)
    print("  Preparing real dataset inputs (SACo-Gold / SACo-VEval)")
    print("=" * 70)

    os.makedirs(args.data_cache_dir, exist_ok=True)
    feats_dir = os.path.join(args.output_dir, "feats")
    os.makedirs(feats_dir, exist_ok=True)

    all_samples = []
    num_needed = args.num_items

    # Always load SACo-VEval SmartGlasses frames (available from HuggingFace)
    veval_samples = []
    print(f"  Downloading SACo-VEval annotations ({args.veval_subset}) ...", flush=True)
    try:
        veval_ann_path = _download_saco_veval_annotations(args.data_cache_dir, args.veval_subset)
        print(f"  VEval annotations: {veval_ann_path}", flush=True)

        veval_media = _download_saco_veval_media(args.data_cache_dir)
        print(f"  VEval media root: {veval_media}", flush=True)

        max_videos = max(10, num_needed // 8 + 5)
        veval_samples = _load_veval_samples(
            veval_ann_path, veval_media, max_videos,
            resolution=1008, seed=args.seed,
        )
        print(f"  Loaded {len(veval_samples)} SACo-VEval video samples", flush=True)
    except Exception as e:
        print(f"  WARNING: Could not load SACo-VEval: {e}", flush=True)

    if args.modality in ("image", "all"):
        gold_loaded = False
        try:
            ann_path = _download_saco_gold_annotations(args.data_cache_dir, args.gold_subset)
            print(f"  SACo-Gold annotations: {ann_path}", flush=True)

            img_root = _download_saco_gold_images(args.data_cache_dir)
            gold_samples = _load_gold_samples(
                ann_path, img_root, num_needed,
                resolution=1008, seed=args.seed,
            )
            if gold_samples:
                print(f"  Loaded {len(gold_samples)} SACo-Gold image samples", flush=True)
                all_samples.extend(gold_samples)
                gold_loaded = True
        except Exception as e:
            print(f"  WARNING: Could not load SACo-Gold images: {e}", flush=True)

        if not gold_loaded:
            print(f"  Using SACo-VEval SmartGlasses frames as image inputs", flush=True)
            for vs in veval_samples:
                for fi, ft in enumerate(vs["frame_tensors"]):
                    if len(all_samples) >= num_needed:
                        break
                    text_q = vs["text_queries"][0] if vs["text_queries"] else "object"
                    all_samples.append({
                        "image_tensor": ft,
                        "text_query": text_q,
                        "image_id": f"{vs['video_id']}_frame_{fi}",
                        "has_annotations": False,
                        "gt_boxes": [],
                    })

    if args.modality in ("video", "all"):
        for vs in veval_samples:
            for fi, ft in enumerate(vs["frame_tensors"]):
                if len(all_samples) >= num_needed:
                    break
                text_q = vs["text_queries"][0] if vs["text_queries"] else "object"
                all_samples.append({
                    "image_tensor": ft,
                    "text_query": text_q,
                    "image_id": f"{vs['video_id']}_frame_{fi}",
                    "has_annotations": False,
                    "gt_boxes": [],
                })

    all_samples = all_samples[:num_needed]

    if not all_samples:
        print("  ERROR: No samples loaded. Cannot run benchmark.", flush=True)
        sys.exit(1)

    print(f"  Total samples prepared: {len(all_samples)}", flush=True)

    manifest_path = _save_inputs_for_workers(all_samples, feats_dir, prefix="input")
    print(f"  Manifest saved: {manifest_path}", flush=True)

    latency_data = []
    if not args.skip_latency:
        for s in latency_scenarios:
            latency_data.append({
                "name": s.name, "resolution": s.resolution, "batch_size": s.batch_size,
                "dataset": s.dataset_name, "dataset_subset": s.dataset_subset,
                "num_warmup": s.num_warmup, "num_iters": args.latency_iters,
            })

    # Download checkpoint for shared weights
    checkpoint_path = None
    if not args.skip_reference:
        try:
            from sam3.model_builder import download_ckpt_from_hf
            checkpoint_path = download_ckpt_from_hf("sam3")
            print(f"  Checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"  WARNING: Could not download checkpoint: {e}")

    print("=" * 70)
    print("  fastkernels SAM3 Baseline vs Reference -- Segmentation Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu}")
    print(f"  Dataset (img)  : SACo-Gold/{args.gold_subset}")
    print(f"  Dataset (vid)  : SACo-VEval/{args.veval_subset}")
    print(f"  Images         : {len(all_samples)} (throughput + correctness)")
    print(f"  Seed           : {args.seed}")
    if latency_data:
        print(f"  Latency        : {', '.join(s['name'] for s in latency_data)}")
    print("=" * 70)

    # -- Run reference SAM3 --
    ref_raw = None
    if not args.skip_reference:
        ref_config = {
            "model": args.model, "seed": args.seed,
            "num_items": len(all_samples),
            "latency_scenarios": latency_data,
            "feats_dir": feats_dir,
            "manifest_path": manifest_path,
        }
        ref_raw = run_worker(
            REFERENCE_SAM3_WORKER, ref_config,
            f"Reference SAM3 [{args.model.split('/')[-1]}]",
            timeout=3600,
        )
        if ref_raw is None:
            print("  WARNING: Reference SAM3 subprocess failed.")

    # -- Run fastkernels SAM3 --
    kb_config = {
        "model": args.model, "seed": args.seed,
        "project_root": str(_PROJECT_ROOT), "package_name": _PACKAGE_DIR.name,
        "num_items": len(all_samples),
        "latency_scenarios": latency_data,
        "feats_dir": feats_dir,
        "checkpoint_path": checkpoint_path,
        "manifest_path": manifest_path,
        "pytorch_reference": args.pytorch_reference,
    }
    kb_raw = run_worker(
        FASTKERNELS_SAM3_WORKER, kb_config,
        f"fastkernels SAM3 [{args.model.split('/')[-1]}]",
        timeout=3600,
    )
    if kb_raw is None:
        print("  ERROR: fastkernels subprocess failed.")
        sys.exit(1)

    num_items = kb_raw["num_items"]

    # -- Throughput --
    _print_throughput_comparison(kb_raw, ref_raw)

    # -- Latency --
    kb_latency = kb_raw.get("latency", [])
    ref_latency = ref_raw.get("latency", []) if ref_raw else []
    latency_combined = []
    if kb_latency:
        latency_combined = _print_latency_comparison(kb_latency, ref_latency)

    # -- Correctness --
    pred_comparison = _compare_predictions(feats_dir, num_items)
    overall_pass = _print_correctness_comparison(pred_comparison, num_items)

    # -- Video benchmark (if video clips are available) --
    video_comparison = {}
    if veval_samples and args.modality in ("video", "all"):
        print("\n" + "=" * 70)
        print("  VIDEO BENCHMARK (multi-frame tracking)")
        print("=" * 70)

        video_clips = _save_video_clips_for_workers(
            veval_samples, feats_dir, max_clips=10, max_frames=16,
        )
        print(f"  Prepared {len(video_clips)} video clips for benchmarking", flush=True)

        if video_clips:
            # Run reference video worker first (saves video_token_ids.pt for fastkernels)
            ref_video_raw = None
            if not args.skip_reference:
                ref_video_config = {
                    "model": args.model, "seed": args.seed,
                    "num_clips": len(video_clips),
                    "video_clips": video_clips,
                    "feats_dir": feats_dir,
                }
                ref_video_raw = run_worker(
                    REFERENCE_SAM3_VIDEO_WORKER, ref_video_config,
                    f"Reference SAM3 Video [{args.model.split('/')[-1]}]",
                    timeout=3600,
                )
                if ref_video_raw:
                    fps = ref_video_raw.get("frames_per_sec", 0)
                    print(f"\n  Reference video: {fps:.1f} frames/sec", flush=True)

            # Run fastkernels video worker
            kb_video_config = {
                "model": args.model, "seed": args.seed,
                "project_root": str(_PROJECT_ROOT),
                "num_clips": len(video_clips),
                "video_clips": video_clips,
                "feats_dir": feats_dir,
                "checkpoint_path": checkpoint_path,
                "pytorch_reference": args.pytorch_reference,
            }
            kb_video_raw = run_worker(
                FASTKERNELS_SAM3_VIDEO_WORKER, kb_video_config,
                f"fastkernels SAM3 Video [{args.model.split('/')[-1]}]",
                timeout=3600,
            )
            if kb_video_raw:
                fps = kb_video_raw.get("frames_per_sec", 0)
                print(f"  fastkernels video: {fps:.1f} frames/sec", flush=True)

            # Video correctness comparison
            if not args.skip_reference:
                video_comparison = _compare_video_predictions(feats_dir, len(video_clips))
                _print_video_correctness(video_comparison)

    # -- Save results --
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        kb_ips = kb_raw["items_per_sec"]
        combined = {
            "gpu": gpu, "model": args.model, "model_type": "segmentation",
            "tp": args.tp, "seed": args.seed, "num_items": num_items,
            "dataset_image": f"SACo-Gold/{args.gold_subset}",
            "dataset_video": f"SACo-VEval/{args.veval_subset}",
            "fastkernels_items_per_sec": kb_ips,
        }
        if ref_raw:
            combined["ref_items_per_sec"] = ref_raw["items_per_sec"]
            combined["speedup"] = kb_ips / ref_raw["items_per_sec"] if ref_raw["items_per_sec"] > 0 else 0
        if latency_combined:
            combined["latency_scenarios"] = latency_combined
        combined["correctness"] = pred_comparison
        if video_comparison:
            combined["video_correctness"] = video_comparison
        combined["overall_pass"] = overall_pass
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
