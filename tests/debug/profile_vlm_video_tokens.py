#!/usr/bin/env python3
"""Compare video preprocessing variants against vLLM's real input path.

The bench_vllm harness hands vLLM ``(frames_ndarray, metadata)`` for video --
vLLM's native video item form -- but hands fastkernels a list of PIL frames.
For Qwen3-VL that is not equivalent: vLLM derives per-frame timestamps from
``metadata["frames_indices"]``/``["fps"]`` and keeps every supplied frame, while
the metadata-less path lets the HF video processor re-sample at its default fps.
Measured over the MMVU throughput scenario that is 3.52M vs 0.50M input tokens.

This script processes the same clips three ways and reports the token count and
video grid so the divergence (and the fix) is visible without a GPU:

  pil-frames        list[PIL.Image]              (what fastkernels received)
  ndarray           frames ndarray, no metadata
  ndarray+metadata  frames ndarray + metadata, do_sample_frames from metadata

Usage:
    python tests/debug/profile_vlm_video_tokens.py \
        --model Qwen/Qwen3-VL-8B-Instruct --num-videos 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the harness's own loader so frames and metadata match the benchmark.
_HARNESS = _PROJECT_ROOT / "fastkernels" / "validate" / "bench_vllm.py"
_ns: dict = {}
exec(  # noqa: S102 - the harness inlines this same source into its workers
    __import__("re")
    .search(r"_MM_PRELOAD_FN = r'''(.*?)'''", _HARNESS.read_text(), 16)
    .group(1),
    _ns,
)
_preload_mm_data = _ns["_preload_mm_data"]


def _run(processor, prompt, frames, metadata, mode):
    from PIL import Image

    messages = [{"role": "user", "content": []}]
    kwargs = dict(return_tensors="pt", padding=True)

    if mode == "pil-frames":
        vid = [Image.fromarray(frames[j]).convert("RGB")
               for j in range(frames.shape[0])]
        messages[0]["content"].append({"type": "video", "video": vid})
        kwargs["videos"] = [vid]
    elif mode == "ndarray":
        messages[0]["content"].append({"type": "video", "video": frames})
        kwargs["videos"] = [frames]
    elif mode == "ndarray+metadata":
        messages[0]["content"].append({"type": "video", "video": frames})
        kwargs["videos"] = [frames]
        # HF's VideoMetadata has no do_sample_frames field; it travels as a
        # processor kwarg instead (this is what vLLM does).
        kwargs["video_metadata"] = [
            {k: v for k, v in metadata.items() if k != "do_sample_frames"}
        ]
        kwargs["do_sample_frames"] = bool(metadata.get("do_sample_frames", False))
    else:
        raise ValueError(mode)

    messages[0]["content"].append({"type": "text", "text": prompt})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    kwargs["text"] = [text]
    inputs = processor(**kwargs)
    vgthw = inputs.get("video_grid_thw")
    vpv = inputs.get("pixel_values_videos")
    return {
        "tokens": int(inputs["input_ids"].shape[1]),
        "grid": vgthw.tolist() if vgthw is not None else None,
        "pv_shape": list(vpv.shape) if vpv is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--num-videos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoProcessor

    print(f"  loading {args.num_videos} MMVU clips...", flush=True)
    data = _preload_mm_data("yale-nlp/MMVU", "validation",
                            args.num_videos, args.seed)
    print(f"  loaded {len(data)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    modes = ["pil-frames", "ndarray", "ndarray+metadata"]
    totals = {m: 0 for m in modes}
    for i, item in enumerate(data):
        frames = item["video_frames"]
        meta = item["video_metadata"]
        print(f"\n  clip {i}: {frames.shape[0]} frames "
              f"{frames.shape[2]}x{frames.shape[1]}, "
              f"fps={meta['fps']:.2f} total={meta['total_num_frames']} "
              f"do_sample_frames={meta['do_sample_frames']}")
        for mode in modes:
            try:
                r = _run(processor, item["prompt"], frames, meta, mode)
                totals[mode] += r["tokens"]
                print(f"    {mode:18s} tokens={r['tokens']:6d} "
                      f"grid={r['grid']} pv={r['pv_shape']}")
            except Exception as exc:
                print(f"    {mode:18s} FAILED: {type(exc).__name__}: {exc}")

    n = len(data)
    print()
    for mode in modes:
        if totals[mode]:
            print(f"  {mode:18s} avg tokens/clip = {totals[mode] / n:8.1f} "
                  f"-> {totals[mode] / n * 1000 / 1e6:.2f}M over 1000 clips")
    print("\n  vLLM's measured Qwen3-VL MMVU total was 3.52M input tokens "
          "(~3520/clip).")


if __name__ == "__main__":
    main()
