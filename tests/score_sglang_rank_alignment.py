#!/usr/bin/env python3
"""Rank alignment scorer for EAGLE-3 outputs.

This complements exact prefix matching in ``bench_sglang.py``.  It feeds
``prompt + generated_tokens`` back into a target-only sglang engine and reports
where each generated token ranks under the reference target model.  This is the
right metric when greedy outputs diverge after close-logit argmax flips.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from fastkernels.bench.utils.real_prompts import (  # noqa: E402
    DEFAULT_WORKLOAD_DATASETS,
    load_real_prompt_workload,
)
from fastkernels.bench.utils.worker import run_worker  # noqa: E402
from fastkernels.tests.bench_sglang import PROMPTS  # noqa: E402


RANK_WORKER = r'''
import json, os, statistics, sys

def _rank_stats(ranks, top_k):
    actual = [r for r in ranks if r is not None]
    n = len(actual)
    if n == 0:
        return {
            "n": 0, "missing": len(ranks), "top1": 0.0, "top5": 0.0,
            "top10": 0.0, f"top{top_k}": 0.0, "mrr": 0.0,
            "median_rank": 0, "avg_rank": 0.0, "worst_rank": 0,
        }
    return {
        "n": n,
        "missing": len(ranks) - n,
        "top1": sum(r == 1 for r in actual) / n,
        "top5": sum(r <= 5 for r in actual) / n,
        "top10": sum(r <= 10 for r in actual) / n,
        f"top{top_k}": sum(r <= top_k for r in actual) / n,
        "mrr": sum(1.0 / r for r in actual) / n,
        "median_rank": statistics.median(actual),
        "avg_rank": sum(actual) / n,
        "worst_rank": max(actual),
    }


def _score_outputs(engine, prompts, generated, top_k):
    concat = [p + g for p, g in zip(prompts, generated)]
    sp = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    outputs = engine.generate(
        input_ids=concat,
        sampling_params=sp,
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=top_k,
    )

    ranks = []
    per_prompt = []
    for i, (out, gen) in enumerate(zip(outputs, generated)):
        prompt_len = len(prompts[i])
        top_logprobs = (out.get("meta_info", {}) or {}).get("input_top_logprobs") or []
        prompt_ranks = []
        for j, token_id in enumerate(gen):
            pos = prompt_len + j
            entry = top_logprobs[pos] if pos < len(top_logprobs) else None
            rank = None
            if entry is not None:
                for r, item in enumerate(entry, start=1):
                    if int(item[1]) == int(token_id):
                        rank = r
                        break
            ranks.append(rank)
            prompt_ranks.append(rank)
        actual = [r for r in prompt_ranks if r is not None]
        per_prompt.append({
            "idx": i,
            "n": len(prompt_ranks),
            "missing": len(prompt_ranks) - len(actual),
            "top1": sum(r == 1 for r in actual) / max(1, len(actual)),
            "top5": sum(r <= 5 for r in actual) / max(1, len(actual)),
            "top10": sum(r <= 10 for r in actual) / max(1, len(actual)),
            f"top{top_k}": sum(r <= top_k for r in actual) / max(1, len(actual)),
        })
    stats = _rank_stats(ranks, top_k)
    stats["per_prompt"] = per_prompt
    return stats


def _fmt(label, stats, top_k):
    return (
        f"{label}: n={stats['n']} top1={100 * stats['top1']:.2f}% "
        f"top5={100 * stats['top5']:.2f}% "
        f"top10={100 * stats['top10']:.2f}% "
        f"top{top_k}={100 * stats[f'top{top_k}']:.2f}% "
        f"mrr={stats['mrr']:.4f} med={stats['median_rank']} "
        f"avg={stats['avg_rank']:.2f} worst={stats['worst_rank']} "
        f"missing={stats['missing']}"
    )


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    import sglang as sgl

    engine = sgl.Engine(
        model_path=cfg["model"],
        mem_fraction_static=cfg.get("gpu_memory_utilization", 0.7),
        max_running_requests=cfg.get("max_running_requests", 8),
        random_seed=cfg["seed"],
        log_level="error",
        cuda_graph_max_bs=cfg.get("cuda_graph_max_bs", 8),
        attention_backend=cfg.get("attention_backend", "fa3"),
        disable_radix_cache=True,
        context_length=cfg.get("max_model_len", 2048),
        dtype="bfloat16",
    )

    summary = {}
    prompts = cfg["prompt_token_ids"]
    for label, outputs in cfg["outputs"].items():
        generated = [o["token_ids"] for o in outputs]
        summary[label] = _score_outputs(engine, prompts, generated, cfg["top_k"])
        print(_fmt(label, summary[label], cfg["top_k"]), flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump(summary, f, indent=2)
    engine.shutdown()


if __name__ == "__main__":
    main()
'''


def _build_fixed_prompts(model: str, num_seqs: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    pool = list(PROMPTS)
    rng.shuffle(pool)
    prompts = []
    while len(prompts) < num_seqs:
        prompts.extend(pool)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return [
        tokenizer.encode(prompt, add_special_tokens=True)
        for prompt in prompts[:num_seqs]
    ]


def _build_wildchat_prompts(
    model: str,
    scenario: str,
    num_seqs: int,
    seed: int,
) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    samples = load_real_prompt_workload(
        scenario,
        tokenizer,
        num_requests=num_seqs,
        dataset_name=DEFAULT_WORKLOAD_DATASETS[scenario],
        seed=seed,
    )
    return [sample.prompt_token_ids for sample in samples]


def _scenario_name(args: argparse.Namespace) -> str:
    if args.workload == "fixed":
        return f"eagle3-{args.num_seqs}seqs-out{args.output_len}"
    return f"wildchat-{args.scenario}-{args.num_seqs}seqs"


def _load_prompts(args: argparse.Namespace) -> list[list[int]]:
    if args.workload == "fixed":
        return _build_fixed_prompts(args.model, args.num_seqs, args.seed)
    return _build_wildchat_prompts(
        args.model,
        args.scenario,
        args.num_seqs,
        args.seed,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score EAGLE-3 generated tokens by target-model rank.",
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--workload", choices=["fixed", "wildchat"], default="wildchat")
    p.add_argument(
        "--scenario",
        choices=["prefill-heavy", "balanced", "decode-heavy"],
        default="balanced",
    )
    p.add_argument("--num-seqs", type=int, default=1000)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-running-requests", type=int, default=8)
    p.add_argument("--cuda-graph-max-bs", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument(
        "--sglang-python",
        default="/home/yak/miniconda3/envs/sglang-bench/bin/python",
        help="Python interpreter for the isolated sglang subprocess.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="bench_sglang.py output directory containing the scenario outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenario = _scenario_name(args)
    scenario_dir = Path(args.output_dir) / scenario
    kb_path = scenario_dir / "fastkernels_outputs.json"
    sglang_path = scenario_dir / "sglang_outputs.json"
    if not kb_path.exists() or not sglang_path.exists():
        raise SystemExit(
            f"Expected {kb_path} and {sglang_path}; run bench_sglang.py first."
        )

    prompts = _load_prompts(args)
    outputs = {
        "sglang": json.loads(sglang_path.read_text())["outputs"],
        "fastkernels": json.loads(kb_path.read_text())["outputs"],
    }
    max_total = max(
        len(prompt) + len(output["token_ids"])
        for label_outputs in outputs.values()
        for prompt, output in zip(prompts, label_outputs)
    )
    if max_total > args.max_model_len:
        raise SystemExit(
            f"prompt+generated length {max_total} exceeds --max-model-len "
            f"{args.max_model_len}"
        )

    cfg = {
        "model": args.model,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_model_len": args.max_model_len,
        "max_running_requests": args.max_running_requests,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prompt_token_ids": prompts,
        "outputs": outputs,
    }
    summary = run_worker(
        RANK_WORKER,
        cfg,
        f"sglang target rank scorer [{scenario}]",
        python_executable=args.sglang_python,
    )
    if summary is None:
        raise SystemExit("sglang rank scorer failed")

    out_path = scenario_dir / "rank_alignment.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
