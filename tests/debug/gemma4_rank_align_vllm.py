#!/usr/bin/env python3
"""Gemma4 rank alignment using vLLM as a prefill scorer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")


def _repo_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    package_parent = repo_root.parent
    sys.path.insert(0, str(package_parent))
    sys.path.insert(0, str(repo_root))
    return repo_root, package_parent


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except AttributeError as exc:
        msg = str(exc)
        if "extra_special_tokens" not in msg and "keys" not in msg:
            raise
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_name, "tokenizer_config.json")
        with open(cfg_path) as f:
            tok_cfg = json.load(f)
        extra = tok_cfg.get("extra_special_tokens")
        if not isinstance(extra, list):
            raise
        extra_map = {
            f"extra_special_token_{i}": token
            for i, token in enumerate(extra)
        }
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            extra_special_tokens=extra_map,
        )


def _rank_stats(ranks: list[int | None]) -> dict[str, float | int]:
    actual = [r for r in ranks if r is not None]
    n = len(actual)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "top20": 0.0,
            "mrr": 0.0,
            "median_rank": 0,
            "avg_rank": 0.0,
            "worst_rank": 0,
            "missing": len(ranks),
        }
    return {
        "n": n,
        "top1": sum(r == 1 for r in actual) / n,
        "top5": sum(r <= 5 for r in actual) / n,
        "top10": sum(r <= 10 for r in actual) / n,
        "top20": sum(r <= 20 for r in actual) / n,
        "mrr": sum(1.0 / r for r in actual) / n,
        "median_rank": statistics.median(actual),
        "avg_rank": sum(actual) / n,
        "worst_rank": max(actual),
        "missing": len(ranks) - n,
    }


def _fmt(stats: dict[str, float | int]) -> str:
    return (
        f"n={stats['n']} top1={100 * stats['top1']:.2f}% "
        f"top5={100 * stats['top5']:.2f}% "
        f"top10={100 * stats['top10']:.2f}% "
        f"top20={100 * stats['top20']:.2f}% "
        f"mrr={stats['mrr']:.4f} med={stats['median_rank']} "
        f"avg={stats['avg_rank']:.2f} worst={stats['worst_rank']} "
        f"missing={stats['missing']}"
    )


def _score_generated(
    llm: Any,
    prompts: list[list[int]],
    generated: list[list[int]],
    *,
    top_k: int,
) -> tuple[list[int | None], list[dict[str, float | int | None]]]:
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=top_k,
        ignore_eos=True,
    )
    concat = [p + g for p, g in zip(prompts, generated)]
    out = llm.generate(
        [dict(prompt_token_ids=x) for x in concat],
        sp,
        use_tqdm=False,
    )

    ranks: list[int | None] = []
    per_request: list[dict[str, float | int | None]] = []
    for i, (o, prompt, gen) in enumerate(zip(out, prompts, generated)):
        prompt_logprobs = o.prompt_logprobs or []
        request_ranks: list[int | None] = []
        first_non_top1 = None
        for j, token_id in enumerate(gen):
            pos = len(prompt) + j
            entry = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            rank = None
            if entry is not None and token_id in entry:
                rank = entry[token_id].rank
                if rank is not None:
                    rank = int(rank)
            if rank != 1 and first_non_top1 is None:
                first_non_top1 = j
            request_ranks.append(rank)
            ranks.append(rank)

        actual = [r for r in request_ranks if r is not None]
        per_request.append({
            "idx": i,
            "n": len(request_ranks),
            "top1": sum(r == 1 for r in actual) / max(1, len(actual)),
            "top20": sum(r <= 20 for r in actual) / max(1, len(actual)),
            "first_non_top1": first_non_top1,
            "missing": len(request_ranks) - len(actual),
        })

    return ranks, per_request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-requests", type=int, default=1000)
    p.add_argument("--sample", type=int, default=32)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=2393)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--summary", default="tmp/gemma4_rank_score_1000_sample32.json")
    return p.parse_args()


def main() -> None:
    repo_root, _ = _repo_paths()
    from kb_nano.bench.utils.real_prompts import (
        DEFAULT_WORKLOAD_DATASETS,
        load_real_prompt_workload,
    )
    from vllm import LLM, SamplingParams

    args = parse_args()
    result_dir = Path(args.result_dir)
    scenarios = ["prefill-heavy", "balanced", "decode-heavy"]
    indices = [
        round(i * (args.num_requests - 1) / (args.sample - 1))
        for i in range(args.sample)
    ]

    print("Loading tokenizer/workloads...", flush=True)
    tokenizer = _load_tokenizer(args.model)
    scenario_prompts: dict[str, list[list[int]]] = {}
    for i, name in enumerate(scenarios):
        samples = load_real_prompt_workload(
            name,
            tokenizer,
            num_requests=args.num_requests,
            dataset_name=DEFAULT_WORKLOAD_DATASETS[name],
            seed=args.seed + i,
        )
        scenario_prompts[name] = [samples[j].prompt_token_ids for j in indices]

    print(f"sample indices: {indices}", flush=True)
    print("Loading vLLM scorer...", flush=True)
    llm = LLM(
        model=args.model,
        seed=args.seed,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=2),
        use_tqdm=False,
    )

    summary: dict[str, Any] = {
        "model": args.model,
        "result_dir": str(result_dir),
        "sample_indices": indices,
        "top_k": args.top_k,
        "scenarios": {},
    }
    for name in scenarios:
        print(f"\nScoring {name}...", flush=True)
        prompts = scenario_prompts[name]
        vllm_out = json.loads(
            (result_dir / name / "vllm_outputs.json").read_text()
        )["outputs"]
        kb_out = json.loads(
            (result_dir / name / "kb_nano_outputs.json").read_text()
        )["outputs"]
        v_tokens = [vllm_out[i]["token_ids"] for i in indices]
        kb_tokens = [kb_out[i]["token_ids"] for i in indices]

        v_ranks, v_per = _score_generated(
            llm, prompts, v_tokens, top_k=args.top_k,
        )
        kb_ranks, kb_per = _score_generated(
            llm, prompts, kb_tokens, top_k=args.top_k,
        )
        v_stats = _rank_stats(v_ranks)
        kb_stats = _rank_stats(kb_ranks)
        summary["scenarios"][name] = {
            "vllm_self": v_stats,
            "kb_nano": kb_stats,
            "vllm_per_request": v_per,
            "kb_per_request": kb_per,
        }
        print(f"  vLLM self: {_fmt(v_stats)}", flush=True)
        print(f"  kb-nano  : {_fmt(kb_stats)}", flush=True)

    summary_path = repo_root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {summary_path}", flush=True)
    del llm
    gc.collect()


if __name__ == "__main__":
    main()
