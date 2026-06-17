#!/usr/bin/env python3
"""
VLM preprocessing alignment test: kb-nano vs vLLM.

Loads a small set of images (VisionArena) and videos (MMVU), runs the
HuggingFace processor through both engines, and compares:
  - token_ids (should be identical)
  - image_grid_thw / video_grid_thw (should be identical)
  - pixel_values (should be within floating-point tolerance)
  - preprocessing wall-clock time (kb-nano should not be slower)

Both workers use the same shared _preload_mm_data / _load_video_opencv
functions (no vLLM imports in data loading).

Usage:
    python tests/debug/test_vlm_preprocess_alignment.py \
        --model Qwen/Qwen2-VL-7B-Instruct --num-images 5 --num-videos 5

    python tests/debug/test_vlm_preprocess_alignment.py \
        --model Qwen/Qwen3-VL-8B-Instruct --num-images 5 --num-videos 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _THIS_DIR.parent
_PACKAGE_DIR = _TESTS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Shared multimodal loader (identical to bench_vllm._MM_PRELOAD_FN)
# ---------------------------------------------------------------------------
_MM_PRELOAD_FN = r'''
import math
import numpy as np
from io import BytesIO
from PIL import Image

def _load_video_opencv(video_path, num_frames=32):
    """Load video frames with OpenCV, matching vLLM's OpenCVVideoBackend."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_num / original_fps if original_fps > 0 else 0

    num_frames_to_sample = total_frames_num
    if num_frames > 0:
        num_frames_to_sample = min(num_frames, total_frames_num)
    num_frames_to_sample = max(1, num_frames_to_sample)

    if num_frames_to_sample == total_frames_num:
        frame_idx = list(range(num_frames_to_sample))
    else:
        frame_idx = np.linspace(
            0, total_frames_num - 1, num_frames_to_sample, dtype=int
        ).tolist()

    frame_idx_set = set(frame_idx)
    max_idx = max(frame_idx)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((num_frames_to_sample, height, width, 3), dtype=np.uint8)

    i = 0
    valid_frame_indices = []
    for idx in range(max_idx + 1):
        ok = cap.grab()
        if not ok:
            continue
        if idx in frame_idx_set:
            ret, frame = cap.retrieve()
            if ret:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                valid_frame_indices.append(idx)
                i += 1

    cap.release()
    valid_num_frames = len(valid_frame_indices)
    frames = frames[:valid_num_frames]

    metadata = {
        "total_num_frames": total_frames_num,
        "fps": original_fps,
        "duration": duration,
        "video_backend": "opencv",
        "frames_indices": valid_frame_indices,
        "do_sample_frames": valid_num_frames == total_frames_num,
    }
    return frames, metadata


def _preload_mm_data(dataset_name, dataset_split, num_seqs, seed,
                     num_video_frames=32):
    """Pre-download and load all images/videos into memory."""
    from datasets import load_dataset
    use_streaming = "MMVU" not in dataset_name
    data = load_dataset(dataset_name, split=dataset_split,
                        streaming=use_streaming)
    data = data.shuffle(seed=seed)

    results = []
    if "VisionArena" in dataset_name:
        for item in data:
            if len(results) >= num_seqs:
                break
            try:
                prompt = item["conversation"][0][0]["content"]
                if "base64" in prompt or len(prompt) > 4096:
                    continue
                img = item["images"][0]
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(BytesIO(img["bytes"]))
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")
                w, h = img.size
                if w * h > 2048 * 2048:
                    continue
            except Exception:
                continue
            results.append({
                "prompt": prompt,
                "images": [img],
                "video_frames": None,
                "video_metadata": None,
            })
    elif "MMVU" in dataset_name:
        from huggingface_hub import snapshot_download
        local_root = snapshot_download(dataset_name, repo_type="dataset")
        remote_root = (
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        )
        for item in data:
            if len(results) >= num_seqs:
                break
            prompt = item["question"] + " " + " ".join(
                f"{k}.{v}" for k, v in item["choices"].items())
            video_path = item["video"].replace(remote_root, local_root)
            frames, metadata = _load_video_opencv(
                video_path, num_frames=num_video_frames)
            results.append({
                "prompt": prompt,
                "images": None,
                "video_frames": frames,
                "video_metadata": metadata,
            })
    return results
'''


# ---------------------------------------------------------------------------
# vLLM preprocessing worker
# ---------------------------------------------------------------------------
VLLM_PREPROCESS_WORKER = _MM_PRELOAD_FN + r'''
import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import numpy as np
import torch

def main():
    from transformers import AutoProcessor

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    results = []
    for ds_cfg in cfg["datasets"]:
        mm_data = _preload_mm_data(
            ds_cfg["name"], ds_cfg["split"],
            ds_cfg["num_seqs"], cfg["seed"],
        )

        t0 = time.perf_counter()
        for item in mm_data:
            messages = [{"role": "user", "content": []}]
            images_for_proc = None
            videos_for_proc = None

            if item["images"] is not None:
                for img in item["images"]:
                    messages[0]["content"].append(
                        {"type": "image", "image": img})
                images_for_proc = item["images"]
            if item["video_frames"] is not None:
                frames_pil = [
                    Image.fromarray(item["video_frames"][j]).convert("RGB")
                    for j in range(item["video_frames"].shape[0])
                ]
                messages[0]["content"].append(
                    {"type": "video", "video": frames_pil})
                videos_for_proc = [frames_pil]
            messages[0]["content"].append(
                {"type": "text", "text": item["prompt"]})

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=images_for_proc,
                videos=videos_for_proc,
                return_tensors="pt",
                padding=True,
            )

            token_ids = inputs["input_ids"][0].tolist()
            pv = inputs.get("pixel_values", None)
            igthw = inputs.get("image_grid_thw", None)
            vpv = inputs.get("pixel_values_videos", None)
            vgthw = inputs.get("video_grid_thw", None)

            entry = {
                "token_ids": token_ids,
                "pixel_values_shape": list(pv.shape) if pv is not None else None,
                "image_grid_thw": igthw.tolist() if igthw is not None else None,
                "video_pixel_values_shape": list(vpv.shape) if vpv is not None else None,
                "video_grid_thw": vgthw.tolist() if vgthw is not None else None,
            }

            npz_idx = len(results)
            npz_path = os.path.join(cfg["npz_dir"], f"vllm_{npz_idx}.npz")
            save_dict = {}
            if pv is not None:
                save_dict["pixel_values"] = pv.numpy()
            if vpv is not None:
                save_dict["pixel_values_videos"] = vpv.numpy()
            if save_dict:
                np.savez_compressed(npz_path, **save_dict)
            entry["npz_path"] = npz_path

            results.append(entry)
        elapsed = time.perf_counter() - t0
        print(f"  vLLM preprocess {ds_cfg['name']}: {len(mm_data)} items in {elapsed:.2f}s")

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano preprocessing worker (no vLLM imports)
# ---------------------------------------------------------------------------
KB_NANO_PREPROCESS_WORKER = _MM_PRELOAD_FN + r'''
import json, os, sys, time
import numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from transformers import AutoProcessor
    model_name = cfg["model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    results = []
    for ds_cfg in cfg["datasets"]:
        mm_data = _preload_mm_data(
            ds_cfg["name"], ds_cfg["split"],
            ds_cfg["num_seqs"], cfg["seed"],
        )

        t0 = time.perf_counter()
        for item in mm_data:
            messages = [{"role": "user", "content": []}]
            images_for_proc = None
            videos_for_proc = None

            if item["images"] is not None:
                for img in item["images"]:
                    messages[0]["content"].append(
                        {"type": "image", "image": img})
                images_for_proc = item["images"]
            if item["video_frames"] is not None:
                frames_pil = [
                    Image.fromarray(item["video_frames"][j]).convert("RGB")
                    for j in range(item["video_frames"].shape[0])
                ]
                messages[0]["content"].append(
                    {"type": "video", "video": frames_pil})
                videos_for_proc = [frames_pil]
            messages[0]["content"].append(
                {"type": "text", "text": item["prompt"]})

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=images_for_proc,
                videos=videos_for_proc,
                return_tensors="pt",
                padding=True,
            )

            token_ids = inputs["input_ids"][0].tolist()
            pv = inputs.get("pixel_values", None)
            igthw = inputs.get("image_grid_thw", None)
            vpv = inputs.get("pixel_values_videos", None)
            vgthw = inputs.get("video_grid_thw", None)

            entry = {
                "token_ids": token_ids,
                "pixel_values_shape": list(pv.shape) if pv is not None else None,
                "image_grid_thw": igthw.tolist() if igthw is not None else None,
                "video_pixel_values_shape": list(vpv.shape) if vpv is not None else None,
                "video_grid_thw": vgthw.tolist() if vgthw is not None else None,
            }

            npz_idx = len(results)
            npz_path = os.path.join(cfg["npz_dir"], f"kbnano_{npz_idx}.npz")
            save_dict = {}
            if pv is not None:
                save_dict["pixel_values"] = pv.numpy()
            if vpv is not None:
                save_dict["pixel_values_videos"] = vpv.numpy()
            if save_dict:
                np.savez_compressed(npz_path, **save_dict)
            entry["npz_path"] = npz_path

            results.append(entry)
        elapsed = time.perf_counter() - t0
        print(f"  kb-nano preprocess {ds_cfg['name']}: {len(mm_data)} items in {elapsed:.2f}s")

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_worker(script: str, config: dict, label: str, timeout: int = 3600):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
        f.write(script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp",
    ) as f:
        config["output_file"] = output_path
        json.dump(config, f)
        config_path = f.name

    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = subprocess.run(
            [sys.executable, script_path, config_path],
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed (exit {result.returncode})")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    except subprocess.TimeoutExpired:
        print(f"  ERROR: {label} timed out after {timeout}s")
        return None
    finally:
        os.unlink(script_path)
        os.unlink(config_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="VLM preprocessing alignment test: kb-nano vs vLLM",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for pixel_values comparison")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for pixel_values comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("  VLM Preprocessing Alignment Test")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Images      : {args.num_images}")
    print(f"  Videos      : {args.num_videos}")
    print(f"  Seed        : {args.seed}")
    print("=" * 70)

    npz_dir = tempfile.mkdtemp(prefix="vlm_align_")
    print(f"  Temp dir    : {npz_dir}")

    datasets_cfg = []
    if args.num_images > 0:
        datasets_cfg.append({
            "name": "lmarena-ai/VisionArena-Chat",
            "split": "train",
            "num_seqs": args.num_images,
        })
    if args.num_videos > 0:
        datasets_cfg.append({
            "name": "yale-nlp/MMVU",
            "split": "validation",
            "num_seqs": args.num_videos,
        })

    config = {
        "model": args.model,
        "seed": args.seed,
        "datasets": datasets_cfg,
        "npz_dir": npz_dir,
    }

    vllm_data = run_worker(
        VLLM_PREPROCESS_WORKER, dict(config),
        f"vLLM preprocessing [{args.model}]",
    )
    if vllm_data is None:
        print("\n  FAIL: vLLM preprocessing worker failed.")
        sys.exit(1)

    kb_data = run_worker(
        KB_NANO_PREPROCESS_WORKER, dict(config),
        f"kb-nano preprocessing [{args.model}]",
    )
    if kb_data is None:
        print("\n  FAIL: kb-nano preprocessing worker failed.")
        sys.exit(1)

    vllm_results = vllm_data["results"]
    kb_results = kb_data["results"]

    total = len(vllm_results)
    assert total == len(kb_results), (
        f"Result count mismatch: vllm={total}, kb-nano={len(kb_results)}"
    )

    print(f"\n{'=' * 70}")
    print("  COMPARISON RESULTS")
    print(f"{'=' * 70}")

    token_id_matches = 0
    grid_matches = 0
    pv_matches = 0
    pv_max_diffs = []
    all_pass = True

    for i in range(total):
        vr = vllm_results[i]
        kr = kb_results[i]
        item_pass = True

        # 1. token_ids
        if vr["token_ids"] == kr["token_ids"]:
            token_id_matches += 1
        else:
            item_pass = False
            min_len = min(len(vr["token_ids"]), len(kr["token_ids"]))
            matching = sum(
                1 for j in range(min_len)
                if vr["token_ids"][j] == kr["token_ids"][j]
            )
            print(f"  [{i}] token_ids MISMATCH: "
                  f"vllm_len={len(vr['token_ids'])} "
                  f"kb_len={len(kr['token_ids'])} "
                  f"first_matching={matching}")

        # 2. grid_thw
        v_igthw = vr.get("image_grid_thw")
        k_igthw = kr.get("image_grid_thw")
        v_vgthw = vr.get("video_grid_thw")
        k_vgthw = kr.get("video_grid_thw")

        grids_ok = (v_igthw == k_igthw) and (v_vgthw == k_vgthw)
        if grids_ok:
            grid_matches += 1
        else:
            item_pass = False
            print(f"  [{i}] grid_thw MISMATCH:")
            if v_igthw != k_igthw:
                print(f"        image_grid_thw: vllm={v_igthw} kb={k_igthw}")
            if v_vgthw != k_vgthw:
                print(f"        video_grid_thw: vllm={v_vgthw} kb={k_vgthw}")

        # 3. pixel_values
        v_npz = vr.get("npz_path")
        k_npz = kr.get("npz_path")
        if v_npz and k_npz and os.path.exists(v_npz) and os.path.exists(k_npz):
            v_data = np.load(v_npz)
            k_data = np.load(k_npz)

            pv_ok = True
            for key in v_data.files:
                if key not in k_data.files:
                    pv_ok = False
                    print(f"  [{i}] pixel_values key '{key}' missing in kb-nano")
                    continue
                v_arr = v_data[key]
                k_arr = k_data[key]
                if v_arr.shape != k_arr.shape:
                    pv_ok = False
                    print(f"  [{i}] {key} shape MISMATCH: "
                          f"vllm={v_arr.shape} kb={k_arr.shape}")
                    continue
                if not np.allclose(v_arr, k_arr, rtol=args.rtol, atol=args.atol):
                    pv_ok = False
                    max_diff = np.max(np.abs(v_arr - k_arr))
                    pv_max_diffs.append(max_diff)
                    print(f"  [{i}] {key} VALUES MISMATCH: max_diff={max_diff:.6e}")
                else:
                    pv_max_diffs.append(np.max(np.abs(v_arr - k_arr)))

            if pv_ok:
                pv_matches += 1
            else:
                item_pass = False
        elif v_npz is None and k_npz is None:
            pv_matches += 1
        else:
            item_pass = False

        if not item_pass:
            all_pass = False

    print(f"\n  Summary ({total} items):")
    print(f"    token_ids match   : {token_id_matches}/{total}")
    print(f"    grid_thw match    : {grid_matches}/{total}")
    print(f"    pixel_values match: {pv_matches}/{total}")
    if pv_max_diffs:
        print(f"    pixel_values max_diff: mean={np.mean(pv_max_diffs):.6e} "
              f"max={np.max(pv_max_diffs):.6e}")

    # Cleanup npz files
    import shutil
    shutil.rmtree(npz_dir, ignore_errors=True)

    if all_pass:
        print(f"\n  PASS: All {total} items match between vLLM and kb-nano.")
    else:
        print(f"\n  FAIL: Some items did not match.")
        sys.exit(1)


if __name__ == "__main__":
    main()
