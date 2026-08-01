#!/usr/bin/env python3
"""Sample vLLM's running-request count through a bench_vllm VLM scenario.

fastkernels' decode-batch histogram on the Qwen3-VL image scenario is bimodal:
493 of 1033 decode steps run with fewer than 128 sequences resident while 484 run
at ~950, so half the steps carry a tenth of the tokens. That is over-admission
followed by preemption, with the preempted sequences forming a second, serialized
wave. This script asks whether vLLM's occupancy has the same shape, which decides
whether the remaining gap is the tail or per-step cost.

Runs the same model/data/config the harness uses, with stats logging enabled at a
1s interval, and prints the Running/Waiting/KV-usage series.

Usage:
    python tests/debug/profile_vllm_occupancy.py \
        --model Qwen/Qwen3-VL-8B-Instruct --modality image
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WORKER = r'''
import json, os, sys, time
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOG_STATS_INTERVAL"] = "1.0"

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bv", os.path.join(cfg["project_root"],
                           "fastkernels", "validate", "bench_vllm.py"))
    bv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bv)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(cfg["model"], trust_remote_code=True)
    ns = {}
    exec(bv._MM_PRELOAD_FN, ns)
    mm = ns["_preload_mm_data"](cfg["dataset"], cfg["split"],
                                cfg["num_seqs"], cfg["seed"])
    mm = ns["_filter_and_prepare"](mm, processor,
                                   cfg["max_model_len"] - cfg["output_len"])
    print(f"  {len(mm)} items after filter", flush=True)

    llm = LLM(
        model=cfg["model"], trust_remote_code=True, seed=cfg["seed"],
        max_model_len=cfg["max_model_len"], enforce_eager=False,
        tensor_parallel_size=1, enable_prefix_caching=False,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        disable_log_stats=False, mm_processor_cache_gb=0,
        load_format="fastsafetensors",
    )
    prompts, sps = [], []
    for it in mm:
        d = {}
        if it["images"] is not None:
            d["image"] = it["images"]
        if it["video_frames"] is not None:
            d["video"] = [(it["video_frames"], it["video_metadata"])]
        prompts.append(dict(prompt=it["chat_text"], multi_modal_data=d))
        sps.append(SamplingParams(temperature=0.0, ignore_eos=True,
                                  max_tokens=cfg["output_len"]))
    llm.generate(prompts, SamplingParams(temperature=0.0, ignore_eos=True,
                                         max_tokens=1), use_tqdm=False)
    t0 = time.perf_counter()
    llm.generate(prompts, sps, use_tqdm=False)
    print(f"  TIMED_ELAPSED {time.perf_counter() - t0:.2f}", flush=True)
    os._exit(0)


# Guard required: vLLM spawns its EngineCore as a child that re-imports this
# module, so a bare main() at module scope would run the whole benchmark again
# inside the child and fail engine startup.
if __name__ == "__main__":
    main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--modality", default="image", choices=["image", "video"])
    ap.add_argument("--num-seqs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = ("lmarena-ai/VisionArena-Chat", "train") if args.modality == "image" \
        else ("yale-nlp/MMVU", "validation")
    gmu = 0.80 if "qwen2-vl" in args.model.lower() else 0.90
    cfg = {
        "project_root": str(_PROJECT_ROOT), "model": args.model,
        "dataset": ds[0], "split": ds[1], "num_seqs": args.num_seqs,
        "seed": args.seed, "output_len": 512, "max_model_len": 16896,
        "gpu_memory_utilization": gmu,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        import json
        json.dump(cfg, f)
        cfg_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(WORKER)
        script = f.name

    proc = subprocess.run(
        [sys.executable, "-u", script, cfg_path],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    samples = []
    for line in out.splitlines():
        m = re.search(r"Running: (\d+) reqs, Waiting: (\d+) reqs.*?"
                      r"KV cache usage: ([\d.]+)%", line)
        if m:
            samples.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    for line in out.splitlines():
        if "GPU KV cache size" in line or "items after filter" in line \
                or "TIMED_ELAPSED" in line or "Available KV cache" in line:
            print("  " + line.strip()[-120:])
    print(f"\n  {len(samples)} stat samples (1s interval): "
          f"running / waiting / kv%")
    for i, (r, w, k) in enumerate(samples):
        print(f"   t={i:3d}s  running={r:5d}  waiting={w:5d}  kv={k:5.1f}%")
    if samples:
        run = [s[0] for s in samples if s[0] > 0]
        low = sum(1 for r in run if r < 128)
        print(f"\n  samples with running>0: {len(run)}, of which <128 reqs: "
              f"{low} ({low / max(1, len(run)):.0%})")
    if proc.returncode != 0:
        print(f"\n  worker exited {proc.returncode}; tail:")
        print("\n".join(out.splitlines()[-15:]))


if __name__ == "__main__":
    main()
