#!/usr/bin/env python3
"""Profile multimodal preprocessing throughput for the Qwen-VL benchmarks.

``LlamaEngine.generate`` preprocesses every multimodal prompt through the HF
processor before the first GPU step. On the bench_vllm image scenario that
barrier is ~34s of the 66.5s total, against vLLM's ~6.9s -- so it dominates the
speedup gap. This script isolates it and compares pool shapes:

  serial                    1 worker, torch intra-op threads as inherited
  pool-N                    N threads, torch intra-op threads as inherited
  pool-N-1thread            N threads, torch.set_num_threads(1)

Usage:
    python tests/debug/profile_vlm_preprocess.py \
        --model Qwen/Qwen2-VL-7B-Instruct --num-images 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_images(num_images, seed):
    from io import BytesIO

    from datasets import load_dataset
    from PIL import Image

    data = load_dataset("lmarena-ai/VisionArena-Chat", split="train",
                        streaming=True).shuffle(seed=seed)
    out = []
    for item in data:
        if len(out) >= num_images:
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
        out.append({"prompt": prompt, "images": [img]})
    return out


def _make_process_fn(processor):
    def process(item):
        messages = [{"role": "user", "content": []}]
        for img in item["images"]:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": item["prompt"]})
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=item["images"],
                           return_tensors="pt", padding=True)
        return inputs["input_ids"].shape[1]
    return process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pools", default="8,14,28")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor

    inherited = torch.get_num_threads()
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"torch.get_num_threads()={inherited} os.cpu_count()={os.cpu_count()}")

    print(f"  loading {args.num_images} VisionArena images...", flush=True)
    items = _load_images(args.num_images, args.seed)
    print(f"  loaded {len(items)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    process = _make_process_fn(processor)

    # Warm the processor (first call pays import/compile costs).
    process(items[0])

    n = len(items)
    results = []

    def run(label, fn):
        torch.set_num_threads(inherited)
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        results.append((label, dt, n / dt))
        print(f"  {label:22s} {dt:7.2f}s  {n / dt:8.1f} items/s  "
              f"({dt / n * 1e3:6.2f} ms/item)", flush=True)

    run("serial", lambda: [process(it) for it in items])

    for w in [int(x) for x in args.pools.split(",")]:
        def pooled(w=w):
            with ThreadPoolExecutor(max_workers=w) as pool:
                list(pool.map(process, items))
        run(f"pool-{w}", pooled)

        def pooled_1t(w=w):
            torch.set_num_threads(1)
            try:
                with ThreadPoolExecutor(max_workers=w) as pool:
                    list(pool.map(process, items))
            finally:
                torch.set_num_threads(inherited)
        run(f"pool-{w}-1thread", pooled_1t)

    best = max(results, key=lambda r: r[2])
    serial = results[0]
    print()
    print(f"  best: {best[0]} at {best[2]:.1f} items/s "
          f"({best[2] / serial[2]:.2f}x serial)")
    print(f"  projected for 1000 images: {1000 / best[2]:.1f}s "
          f"(serial {1000 / serial[2]:.1f}s)")


if __name__ == "__main__":
    main()
