#!/usr/bin/env python3
"""
Analyze generated-token ranks with vLLM prompt prefill.

For ``tests/bench_vllm.py`` outputs, prefer ``--bench-output-dir``. That mode
reconstructs the original random prompts from the benchmark seed, feeds
``prompt + generated_tokens`` back into vLLM, and scores only the generated
continuation tokens. This avoids treating the generated continuation as a
standalone prompt.

Examples:
    # Score both vLLM and fastkernels outputs from a bench_vllm.py run.
    python tests/analyze_prefill_token_ranks.py \
        --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
        --bench-output-dir tmp/readme_kimi_linear_full_vs_vllm_rerun \
        --tp 2 \
        --limit 64 \
        --output-json tmp/kimi_rank_alignment.json

    # Generic mode: score saved token IDs as one standalone prompt.
    python tests/analyze_prefill_token_ranks.py \
        --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
        --input-json tmp/run/balanced/fastkernels_outputs.json \
        --input-mode token_ids \
        --tp 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


def _needs_trust_remote_code(model_name: str) -> bool:
    lower = model_name.lower()
    return "kimi" in lower or "qwen3-next" in lower


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved vLLM/fastkernels outputs with vLLM prompt prefill.",
    )
    parser.add_argument("--model", required=True, help="Model name or path for vLLM.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--bench-output-dir",
        default=None,
        help="Output directory produced by tests/bench_vllm.py.",
    )
    input_group.add_argument(
        "--input-json",
        default=None,
        help="Generic JSON file containing an 'outputs' list.",
    )

    parser.add_argument(
        "--scenario",
        default=None,
        help="Only analyze one bench_vllm.py scenario, for example balanced.",
    )
    parser.add_argument(
        "--engine",
        choices=("both", "vllm", "fastkernels"),
        default="both",
        help="Which engine outputs to score in --bench-output-dir mode.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("text", "token_ids"),
        default="token_ids",
        help="Generic --input-json mode only: score generated_text or token_ids.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max_model_len. Defaults to max prompt+generated length.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--prompt-logprobs",
        type=int,
        default=1,
        help=(
            "vLLM prompt_logprobs setting. vLLM still includes the realized "
            "token rank, so 1 is enough for top-k rank statistics."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only analyze the first N outputs per scenario/engine.",
    )
    parser.add_argument(
        "--batch-mode",
        choices=("together", "serial"),
        default="together",
        help="Score prompts together in one batch, or one-by-one with bsz=1.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run vLLM in eager mode.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Force trust_remote_code=True.",
    )
    parser.add_argument(
        "--load-format",
        default=None,
        help="Optional vLLM load_format, for example fastsafetensors.",
    )
    parser.add_argument(
        "--show-worst",
        type=int,
        default=0,
        help="Store this many worst-ranked tokens per request in --output-json.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save summary JSON.",
    )
    return parser.parse_args()


def _load_outputs(path: Path, limit: int | None) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    outputs = obj["outputs"] if isinstance(obj, dict) and "outputs" in obj else obj
    if not isinstance(outputs, list):
        raise ValueError(f"{path} does not contain a valid outputs list")
    return outputs[:limit] if limit is not None else outputs


def _make_random_prompts(input_len: int, seed: int, count: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, 10000) for _ in range(input_len)] for _ in range(count)]


def _load_bench_records(
    bench_output_dir: Path,
    scenario_filter: str | None,
    engine_filter: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    results_path = bench_output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"missing {results_path}")

    results = json.loads(results_path.read_text())
    seed = int(results.get("seed", 42))
    scenarios = results.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{results_path} has no throughput scenarios")

    engines = ("vllm", "fastkernels") if engine_filter == "both" else (engine_filter,)
    records: list[dict[str, Any]] = []
    for scenario_idx, scenario in enumerate(scenarios):
        name = scenario.get("scenario")
        if scenario_filter is not None and name != scenario_filter:
            continue
        if not isinstance(name, str):
            raise ValueError(f"invalid scenario entry in {results_path}: {scenario!r}")
        if "input_len" not in scenario:
            raise ValueError(f"scenario {name!r} has no input_len")

        input_len = int(scenario["input_len"])
        scenario_seed = seed + scenario_idx
        scenario_dir = bench_output_dir / name

        for engine in engines:
            output_path = scenario_dir / f"{engine}_outputs.json"
            outputs = _load_outputs(output_path, limit)
            prompts = _make_random_prompts(input_len, scenario_seed, len(outputs))

            for request_idx, (prompt_ids, output) in enumerate(zip(prompts, outputs)):
                generated_ids = output.get("token_ids")
                if not isinstance(generated_ids, list):
                    raise ValueError(f"{output_path} output {request_idx} has no token_ids")

                combined_ids = prompt_ids + list(generated_ids)
                records.append(
                    {
                        "source": "bench",
                        "scenario": name,
                        "engine": engine,
                        "index": request_idx,
                        "prompt": {"prompt_token_ids": combined_ids},
                        "score_start": len(prompt_ids),
                        "score_end": len(combined_ids),
                        "prompt_len": len(prompt_ids),
                        "generated_len": len(generated_ids),
                        "saved_text": output.get("generated_text"),
                    },
                )

    if not records:
        raise ValueError("no records selected")
    return records


def _load_generic_records(
    input_path: Path,
    input_mode: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    outputs = _load_outputs(input_path, limit)
    records: list[dict[str, Any]] = []
    for idx, output in enumerate(outputs):
        if input_mode == "text":
            text = output.get("generated_text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Output {idx} has no generated_text; use --input-mode token_ids.",
                )
            prompt: str | dict[str, list[int]] = text
            token_count = None
        else:
            token_ids = output.get("token_ids")
            if not isinstance(token_ids, list):
                raise ValueError(f"Output {idx} has no token_ids")
            prompt = {"prompt_token_ids": token_ids}
            token_count = len(token_ids)

        records.append(
            {
                "source": "generic",
                "scenario": "generic",
                "engine": input_path.stem,
                "index": idx,
                "prompt": prompt,
                "score_start": 1,
                "score_end": None,
                "prompt_len": 0,
                "generated_len": token_count,
                "saved_text": output.get("generated_text"),
            },
        )
    return records


def _preview_text(text: str | None, limit: int = 120) -> str:
    if not text:
        return ""
    text = text.replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def _rank_stats(ranks: list[int], missing: int = 0) -> dict[str, float | int]:
    if not ranks:
        return {
            "num_scored_tokens": 0,
            "missing_tokens": missing,
            "avg_rank": math.nan,
            "median_rank": math.nan,
            "best_rank": math.nan,
            "worst_rank": math.nan,
            "top1": math.nan,
            "top5": math.nan,
            "top10": math.nan,
            "top20": math.nan,
            "top100": math.nan,
        }
    n = len(ranks)
    ordered = sorted(ranks)
    return {
        "num_scored_tokens": n,
        "missing_tokens": missing,
        "avg_rank": sum(ranks) / n,
        "median_rank": statistics.median(ranks),
        "best_rank": min(ranks),
        "worst_rank": max(ranks),
        "p95_rank": ordered[int(0.95 * (n - 1))],
        "top1": sum(r == 1 for r in ranks) / n,
        "top5": sum(r <= 5 for r in ranks) / n,
        "top10": sum(r <= 10 for r in ranks) / n,
        "top20": sum(r <= 20 for r in ranks) / n,
        "top100": sum(r <= 100 for r in ranks) / n,
    }


def _format_stats(stats: dict[str, float | int]) -> str:
    return (
        f"n={stats['num_scored_tokens']} missing={stats['missing_tokens']} "
        f"top1={stats['top1']:.4f} top5={stats['top5']:.4f} "
        f"top10={stats['top10']:.4f} top20={stats['top20']:.4f} "
        f"top100={stats['top100']:.4f} median={stats['median_rank']} "
        f"avg={stats['avg_rank']:.2f} p95={stats.get('p95_rank', math.nan)} "
        f"worst={stats['worst_rank']}"
    )


def _max_prompt_len(records: list[dict[str, Any]]) -> int:
    max_len = 0
    for record in records:
        prompt = record["prompt"]
        if isinstance(prompt, dict):
            max_len = max(max_len, len(prompt["prompt_token_ids"]))
    return max_len


def main() -> None:
    args = _parse_args()

    if args.bench_output_dir is not None:
        records = _load_bench_records(
            Path(args.bench_output_dir),
            args.scenario,
            args.engine,
            args.limit,
        )
    else:
        records = _load_generic_records(
            Path(args.input_json),
            args.input_mode,
            args.limit,
        )

    from vllm import LLM, SamplingParams

    max_model_len = args.max_model_len or max(4096, _max_prompt_len(records))
    llm_kwargs: dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
    )
    if args.trust_remote_code or _needs_trust_remote_code(args.model):
        llm_kwargs["trust_remote_code"] = True
    if args.load_format:
        llm_kwargs["load_format"] = args.load_format

    print(
        f"Loading vLLM scorer: model={args.model} tp={args.tp} "
        f"max_model_len={max_model_len} enforce_eager={args.enforce_eager}",
        flush=True,
    )
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=args.prompt_logprobs,
        detokenize=False,
        ignore_eos=True,
    )
    prompts = [record["prompt"] for record in records]
    if args.batch_mode == "together":
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    else:
        outputs = [
            llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            for prompt in prompts
        ]

    all_ranks: list[int] = []
    all_missing = 0
    group_ranks: dict[tuple[str, str], list[int]] = defaultdict(list)
    group_missing: dict[tuple[str, str], int] = defaultdict(int)
    request_summaries: list[dict[str, Any]] = []

    for record, output in zip(records, outputs):
        prompt_ids = list(output.prompt_token_ids or [])
        prompt_logprobs = output.prompt_logprobs or []
        score_start = int(record["score_start"])
        score_end = record["score_end"]
        if score_end is None:
            score_end = len(prompt_ids)
        score_end = min(int(score_end), len(prompt_ids))

        ranks: list[int] = []
        missing = 0
        worst_tokens: list[dict[str, Any]] = []
        for pos in range(score_start, score_end):
            lp_dict = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            if lp_dict is None:
                missing += 1
                continue

            token_id = prompt_ids[pos]
            token_info = lp_dict.get(token_id)
            if token_info is None or token_info.rank is None:
                missing += 1
                continue

            rank = int(token_info.rank)
            ranks.append(rank)
            if args.show_worst:
                worst_tokens.append(
                    {
                        "pos": pos,
                        "generated_pos": pos - score_start,
                        "token_id": token_id,
                        "rank": rank,
                        "logprob": float(token_info.logprob),
                    },
                )

        key = (record["scenario"], record["engine"])
        group_ranks[key].extend(ranks)
        group_missing[key] += missing
        all_ranks.extend(ranks)
        all_missing += missing

        request_stats = _rank_stats(ranks, missing)
        request_summary = {
            "scenario": record["scenario"],
            "engine": record["engine"],
            "index": record["index"],
            "prompt_len": record["prompt_len"],
            "generated_len": record["generated_len"],
            "saved_text_preview": _preview_text(record.get("saved_text")),
            "stats": request_stats,
        }
        if args.show_worst:
            request_summary["worst_tokens"] = sorted(
                worst_tokens,
                key=lambda row: row["rank"],
                reverse=True,
            )[: args.show_worst]
        request_summaries.append(request_summary)

    print("\n=== group stats ===")
    group_summary: dict[str, dict[str, float | int]] = {}
    for key in sorted(group_ranks):
        scenario, engine = key
        stats = _rank_stats(group_ranks[key], group_missing[key])
        group_summary[f"{scenario}/{engine}"] = stats
        print(f"{scenario:>14} {engine:>7}: {_format_stats(stats)}")

    global_stats = _rank_stats(all_ranks, all_missing)
    print("\n=== global ===")
    print(_format_stats(global_stats))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "bench_output_dir": args.bench_output_dir,
                    "input_json": args.input_json,
                    "limit": args.limit,
                    "prompt_logprobs": args.prompt_logprobs,
                    "group_stats": group_summary,
                    "global_stats": global_stats,
                    "requests": request_summaries,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        print(f"\nsaved analysis to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
