"""Offline throughput benchmark for kb-nano.

Modeled after ``vllm bench throughput``. Runs ``LlamaEngine`` in offline mode
with batched generation and measures requests/s, total tokens/s, and output
tokens/s.

Usage (standalone):
    python -m kb_nano.bench.e2e throughput \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --dataset-name kb_nano \\
        --num-prompts 100

Usage (with subprocess isolation):
    python -m kb_nano.bench.e2e throughput \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --dataset-name kb_nano \\
        --subprocess
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from kb_nano.bench.utils.datasets import (
    SampleRequest,
    add_dataset_parser,
    get_samples,
)
from kb_nano.bench.utils.real_prompts import load_real_prompt_workload
from kb_nano.bench.utils.worker import KB_NANO_WORKER, run_worker
from kb_nano.infra.kernel_swapper import (
    apply_candidates,
    discover_candidates,
    print_candidate_summary,
)


def run_kb_nano(
    requests: list[SampleRequest],
    model: str,
    tp: int,
    seed: int,
    temperature: float,
    top_p: float,
    enforce_eager: bool,
    save_outputs: bool,
) -> dict:
    """Run kb-nano offline throughput benchmark (in-process)."""
    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    engine = LlamaEngine(
        model_name=model,
        seed=seed,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
    )

    prompts = []
    sp_list = []
    for req in requests:
        prompts.append(req.prompt)
        sp_list.append(SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=req.expected_output_len,
            ignore_eos=True,
            seed=seed,
        ))

    print("  Warmup run...")
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    print(f"  Running {len(requests)} requests...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, sp_list)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_input_tokens = sum(
        len(engine.tokenizer.encode(p)) if isinstance(p, str) else len(p)
        for p in prompts
    )
    total_output_tokens = sum(len(o.token_ids) for o in outputs)

    result = {
        "elapsed": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    if save_outputs:
        result["outputs"] = [
            {
                "prompt": o.prompt,
                "generated_text": o.generated_text,
                "token_ids": o.token_ids,
            }
            for o in outputs
        ]

    engine._cleanup()
    del engine
    return result


def run_kb_nano_subprocess(
    requests: list[SampleRequest],
    model: str,
    tp: int,
    seed: int,
    temperature: float,
    top_p: float,
    enforce_eager: bool,
    save_outputs: bool,
    no_candidate_kernels: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> dict | None:
    """Run kb-nano offline throughput benchmark in a subprocess."""
    from kb_nano import KB_ROOT, PROJECT_ROOT
    kb_root = str(PROJECT_ROOT)
    package_name = KB_ROOT.name

    prompts = []
    output_lens = []
    for req in requests:
        prompts.append(req.prompt)
        output_lens.append(req.expected_output_len)

    config = {
        "model": model,
        "tp": tp,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "enforce_eager": enforce_eager,
        "prompts": prompts,
        "output_lens": output_lens,
        "ignore_eos": True,
        "save_outputs": save_outputs,
        "no_candidate_kernels": no_candidate_kernels,
        "project_root": kb_root,
        "package_name": package_name,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        config["max_model_len"] = max_model_len

    short_name = model.split("/")[-1]
    return run_worker(
        KB_NANO_WORKER, config,
        f"kb-nano [{short_name}] (TP={tp})",
    )


def validate_args(args: argparse.Namespace):
    """Validate CLI arguments for the throughput benchmark."""
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_p is not None and not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    dataset_name = getattr(args, "dataset_name", "random")
    dataset_path = getattr(args, "dataset_path", None)

    if dataset_name == "kb_nano":
        return

    if dataset_name in ("random", "random-mm", "random-rerank"):
        random_input_len = getattr(args, "random_input_len", None)
        input_len = getattr(args, "input_len", None)
        if random_input_len is None and input_len is None:
            raise ValueError(
                "Either --input-len or --random-input-len must be provided "
                "for a random dataset"
            )
        if input_len is not None and random_input_len is not None:
            warnings.warn(
                "Both --input-len and --random-input-len are specified. "
                "The random version (--random-input-len) will be preferred.",
                stacklevel=2,
            )
        random_output_len = getattr(args, "random_output_len", None)
        output_len = getattr(args, "output_len", None)
        if output_len is not None and random_output_len is not None:
            warnings.warn(
                "Both --output-len and --random-output-len are specified. "
                "The random version (--random-output-len) will be preferred.",
                stacklevel=2,
            )

    if (
        dataset_name not in ("random", "random-mm", "random-rerank", None)
        and dataset_path is None
        and dataset_name not in ("prefix_repetition",)
    ):
        print(
            f"WARNING: --dataset-name={dataset_name} typically requires "
            "--dataset-path to be set."
        )


def add_cli_args(parser: argparse.ArgumentParser):
    """Add throughput-specific CLI arguments."""
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer name or path (defaults to --model)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0, matching vLLM. "
             "Use 0.0 for greedy/deterministic)",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=False,
        help="Disable CUDA graphs / torch.compile (default: False = full speed)",
    )
    parser.add_argument(
        "--subprocess", action="store_true", default=False,
        help="Run the engine in a clean subprocess for isolation",
    )
    parser.add_argument(
        "--input-len", type=int, default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len", type=int, default=None,
        help="Output length for each request. Overrides the "
             "output length from the dataset.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save performance results in JSON format",
    )
    parser.add_argument(
        "--save-outputs", type=str, default=None,
        help="Path to save generated outputs alongside performance data",
    )
    parser.add_argument(
        "--no-candidate-kernels", action="store_true", default=False,
        help="Disable candidate kernel auto-detection; use only baseline kernels",
    )
    parser.add_argument(
        "--kb-nano-scenario", type=str, default="balanced",
        choices=["prefill-heavy", "balanced", "decode-heavy"],
        help="kb-nano WildChat-derived scenario to use with "
             "--dataset-name=kb_nano.",
    )

    add_dataset_parser(parser)
    parser.set_defaults(seed=42)


def main(args: argparse.Namespace):
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    undo_info = None
    if not args.no_candidate_kernels:
        candidates = discover_candidates()
        if candidates:
            print_candidate_summary(candidates)
            undo_info = apply_candidates(candidates)

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    if args.input_len is not None:
        args.random_input_len = args.input_len
        args.sonnet_input_len = args.input_len
    if args.output_len is not None:
        args.random_output_len = args.output_len
        args.sonnet_output_len = args.output_len
        args.sharegpt_output_len = args.output_len
        args.custom_output_len = args.output_len
        args.hf_output_len = args.output_len
        args.spec_bench_output_len = args.output_len
        args.prefix_repetition_output_len = args.output_len

    if not hasattr(args, "backend"):
        args.backend = "vllm"
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    dataset_name = getattr(args, "dataset_name", "random")
    use_real_workload = dataset_name == "kb_nano"

    if use_real_workload:
        scenario_name = args.kb_nano_scenario
        samples = load_real_prompt_workload(
            scenario_name,
            tokenizer,
            num_requests=args.num_prompts,
            decode_cap=args.output_len,
            seed=args.seed,
        )
        requests = [
            SimpleNamespace(
                prompt=s.prompt_token_ids,
                prompt_len=len(s.prompt_token_ids),
                expected_output_len=s.output_len,
            )
            for s in samples
        ]
    else:
        requests = get_samples(args, tokenizer)
    total_input_tokens = sum(r.prompt_len for r in requests)
    total_expected_output = sum(r.expected_output_len for r in requests)

    print("=" * 70)
    print("  kb-nano Throughput Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  TP             : {args.tp}")
    print(f"  Requests       : {len(requests)}")
    print(f"  Total input    : {total_input_tokens:,} tokens")
    print(f"  Total output   : {total_expected_output:,} tokens (expected)")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Top-p          : {args.top_p}")
    print(f"  Enforce eager  : {args.enforce_eager}")
    print(f"  Seed           : {args.seed}")
    print("=" * 70)

    save_outputs = args.save_outputs is not None

    if args.subprocess:
        data = run_kb_nano_subprocess(
            requests, args.model, args.tp, args.seed,
            args.temperature, args.top_p, args.enforce_eager, save_outputs,
            no_candidate_kernels=args.no_candidate_kernels,
        )
        if data is None:
            print("  ERROR: Subprocess benchmark failed.")
            return
    else:
        data = run_kb_nano(
            requests, args.model, args.tp, args.seed,
            args.temperature, args.top_p, args.enforce_eager, save_outputs,
        )

    elapsed = data["elapsed"]
    total_input = data.get("total_input_tokens", total_input_tokens)
    total_output = data["total_output_tokens"]
    total_tokens = total_input + total_output

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Elapsed time    : {elapsed:.2f}s")
    print(f"  Input tokens    : {total_input:,}")
    print(f"  Output tokens   : {total_output:,}")
    print(f"  Requests/s      : {len(requests) / elapsed:.2f}")
    print(f"  Total tokens/s  : {total_tokens / elapsed:,.1f}")
    print(f"  Output tokens/s : {total_output / elapsed:,.1f}")
    print(f"{'=' * 70}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "tp": args.tp,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enforce_eager": args.enforce_eager,
        "num_requests": len(requests),
        "elapsed_time": elapsed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_num_tokens": total_tokens,
        "requests_per_second": len(requests) / elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "output_tokens_per_second": total_output / elapsed,
    }

    # Log to MLflow
    from kb_nano.bench.tracking import tracker

    tracker.log_e2e(results, bench_type="throughput")

    if args.output_json:
        output_json = args.output_json
    else:
        from kb_nano import run_output_path
        output_json = str(run_output_path("throughput"))

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_json}")

    if args.save_outputs and "outputs" in data:
        output_data = {**results, "outputs": data["outputs"]}
        os.makedirs(os.path.dirname(args.save_outputs) or ".", exist_ok=True)
        with open(args.save_outputs, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Outputs saved to: {args.save_outputs}")

    if undo_info is not None:
        from kb_nano.infra.kernel_swapper import restore
        restore(undo_info)
