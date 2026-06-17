#!/usr/bin/env python3
"""
Throughput, latency, and alignment benchmark for EAGLE-3 speculative decoding:
``kb-nano`` (LlamaEagle3Engine) vs the reference SOTA ``sglang`` implementation.

Both engines are loaded in long-lived subprocesses to avoid CUDA/import
contamination and to make sure each backend gets a fresh, isolated process.

Default model pair is the same EAGLE-3 head sglang ships in their own
benchmark/CI:

  target : ``meta-llama/Llama-3.1-8B-Instruct``
  draft  : ``jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B``
           (the "hot-token-id" EAGLE-3 head trained for sglang)

We use fixed chat-style prompts (random token ids would defeat the purpose of a
spec-decoding benchmark since the draft head has nothing meaningful to predict).

Usage:
    python tests/bench_sglang.py
    python tests/bench_sglang.py --skip-sglang   # kb-nano only
    python tests/bench_sglang.py --num-seqs 64 --output-len 256
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Paths / package wiring
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.real_prompts import (  # noqa: E402
    DEFAULT_WORKLOAD_DATASETS,
    load_real_prompt_workload,
)
from kb_nano.bench.utils.worker import run_worker  # noqa: E402


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
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
# EAGLE-3 prompts: fixed chat-style prompts so spec decoding has signal.
# Roughly mirrors sglang's `scripts/playground/bench_speculative.py`.
# ---------------------------------------------------------------------------
PROMPTS = [
    "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
    "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
    "Human: Solve x^2 = -1. Think step-by-step. Give me a long detailed explanation.\n\nAssistant:",
    "Human: Hello? Who are you? Write code, math, and poem to explain yourself.\n\nAssistant:",
    "Human: Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger.\n\nAssistant:",
    "Human: Act as a storyteller. Write an engaging story about a brave knight who saves a kingdom from a dragon.\n\nAssistant:",
    "Human: Explain the theory of relativity in simple terms with concrete everyday examples.\n\nAssistant:",
    "Human: Write a long Python function that implements quicksort with detailed inline comments explaining each step.\n\nAssistant:",
    "Human: Summarize the plot of Hamlet act by act. Be thorough.\n\nAssistant:",
    "Human: List ten ways to improve memory and study habits, with explanations for each.\n\nAssistant:",
    "Human: Write a haiku about every season of the year, then explain each one.\n\nAssistant:",
    "Human: Compare and contrast supervised, unsupervised, and reinforcement learning.\n\nAssistant:",
    "Human: Walk me through how a CPU executes a simple `int main() { return 0; }` C program.\n\nAssistant:",
    "Human: Describe the rise and fall of the Roman Empire in detail.\n\nAssistant:",
    "Human: Write a Bash one-liner to find the 10 largest files under the current directory and explain it.\n\nAssistant:",
]

WILDCHAT_SCENARIOS = ("prefill-heavy", "balanced", "decode-heavy")


def _build_prompt_set(num_seqs: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    pool = list(PROMPTS)
    rng.shuffle(pool)
    if num_seqs <= len(pool):
        return pool[:num_seqs]
    # Tile out to num_seqs prompts (ordering randomized by ``seed``).
    out = []
    while len(out) < num_seqs:
        out.extend(pool)
    return out[:num_seqs]


def _build_wildchat_scenarios(
    model: str,
    scenario_names: list[str],
    num_seqs: int,
    seed: int,
    max_model_len: int,
) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    scenarios = []
    for i, name in enumerate(scenario_names):
        samples = load_real_prompt_workload(
            name,
            tokenizer,
            num_requests=num_seqs,
            dataset_name=DEFAULT_WORKLOAD_DATASETS[name],
            seed=seed + i,
        )
        prompt_token_ids = [s.prompt_token_ids for s in samples]
        output_lens = [s.output_len for s in samples]
        max_total_len = max(
            len(p) + ol for p, ol in zip(prompt_token_ids, output_lens)
        )
        if max_total_len > max_model_len:
            raise SystemExit(
                f"WildChat scenario {name!r} exceeds --max-model-len: "
                f"{max_total_len} > {max_model_len}"
            )
        input_lens = [len(p) for p in prompt_token_ids]
        scenarios.append({
            "name": f"wildchat-{name}-{num_seqs}seqs",
            "dataset": DEFAULT_WORKLOAD_DATASETS[name],
            "prompt_token_ids": prompt_token_ids,
            "output_lens": output_lens,
            "output_len": max(output_lens),
            "avg_input_len": float(np.mean(input_lens)),
            "max_input_len": max(input_lens),
            "avg_output_len": float(np.mean(output_lens)),
            "max_output_len": max(output_lens),
        })
    return scenarios


# ---------------------------------------------------------------------------
# sglang subprocess worker
# ---------------------------------------------------------------------------
SGLANG_WORKER = r'''
import json, os, sys, time

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    # The EAGLE-3 draft head's max_position_embeddings is 2048 which sglang
    # uses as the *derived* context length cap; allow a larger
    # context_length on the target.
    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

    import sglang as sgl

    engine_kwargs = dict(
        model_path=cfg["model"],
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=cfg["draft_model"],
        speculative_num_steps=cfg["spec_steps"],
        speculative_eagle_topk=cfg["spec_topk"],
        speculative_num_draft_tokens=cfg["spec_num_draft_tokens"],
        mem_fraction_static=cfg.get("gpu_memory_utilization", 0.85),
        max_running_requests=cfg.get("max_running_requests", 64),
        random_seed=cfg["seed"],
        log_level="error",
        cuda_graph_max_bs=cfg.get("cuda_graph_max_bs", 16),
        attention_backend=cfg.get("attention_backend", "fa3"),
        disable_radix_cache=True,
        context_length=cfg.get("max_model_len", 2048),
        dtype="bfloat16",
    )
    if cfg.get("disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True
    engine = sgl.Engine(**engine_kwargs)

    # Warmup
    warmup_prompts = ["Hello, who are you?"]
    warmup_sp = {"temperature": 0.0, "max_new_tokens": 16}
    engine.generate(warmup_prompts, warmup_sp)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])

    def _tokenize(prompts):
        out = []
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=True)
            out.append(ids)
        return out

    def _record_outputs(outputs, prompts=None, prompt_token_ids=None):
        recs = []
        n_out = 0
        n_in = 0
        for i, o in enumerate(outputs):
            meta = o.get("meta_info", {}) or {}
            out_ids = list(meta.get("output_token_logprobs") or [])
            if out_ids and isinstance(out_ids[0], (list, tuple)):
                # logprob entries are (logprob, token_id, ...).
                out_ids = [int(e[1]) for e in out_ids]
            else:
                out_ids = list(meta.get("output_ids") or [])
            if not out_ids:
                # Fallback to retokenizing the text.
                out_ids = tok.encode(o.get("text", ""), add_special_tokens=False)
            in_ids = list(meta.get("prompt_tokens_ids") or [])
            if not in_ids and prompt_token_ids is not None:
                in_ids = list(prompt_token_ids[i])
            if not in_ids and prompts is not None:
                in_ids = tok.encode(prompts[i], add_special_tokens=False)
            n_out += len(out_ids)
            n_in += len(in_ids)
            recs.append({"text": o.get("text", ""), "token_ids": out_ids})
        return recs, n_in, n_out

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompts = scenario.get("prompts")
        if "prompt_token_ids" in scenario:
            prompt_token_ids = [list(p) for p in scenario["prompt_token_ids"]]
        else:
            prompt_token_ids = _tokenize(prompts)
        output_lens = scenario.get("output_lens")
        if output_lens is None:
            output_lens = [scenario["output_len"]] * len(prompt_token_ids)
        sp = [
            {
                "temperature": cfg.get("temperature", 0.0),
                "max_new_tokens": int(output_len),
                "ignore_eos": True,
            }
            for output_len in output_lens
        ]

        start = time.perf_counter()
        outputs = engine.generate(
            input_ids=prompt_token_ids,
            sampling_params=sp,
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=0,
        )
        elapsed = time.perf_counter() - start

        out_records, n_in, n_out = _record_outputs(
            outputs,
            prompts=prompts,
            prompt_token_ids=prompt_token_ids,
        )
        all_results.append({
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": n_in,
            "total_output_tokens": n_out,
            "outputs": out_records,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        prompt_token_ids = _tokenize(prompts)
        sp = {
            "temperature": 0.0,
            "max_new_tokens": ls["output_len"],
            "ignore_eos": True,
        }
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.generate(input_ids=prompt_token_ids, sampling_params=sp)
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            engine.generate(input_ids=prompt_token_ids, sampling_params=sp)
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    out = {
        "throughput": all_results,
        "latency": latency_results,
    }
    try:
        info = engine.get_server_info()
        out["server_info"] = info
    except Exception:
        pass

    with open(cfg["output_file"], "w") as f:
        json.dump(out, f)

    engine.shutdown()

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano subprocess worker
# ---------------------------------------------------------------------------
KB_NANO_EAGLE3_WORKER = r'''
import json, sys, time

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(
        f"{pkg}.infra.eagle3_engine",
        fromlist=["LlamaEagle3Engine", "Eagle3SamplingParams"],
    )
    LlamaEagle3Engine = mod.LlamaEagle3Engine
    Eagle3SamplingParams = mod.Eagle3SamplingParams

    engine = LlamaEagle3Engine(
        model_name=cfg["model"],
        draft_repo=cfg["draft_model"],
        seed=cfg["seed"],
        max_model_len=cfg["max_model_len"],
        max_num_seqs=cfg.get("max_running_requests", 64),
        spec_steps=cfg["spec_steps"],
        spec_topk=cfg.get("spec_topk", 1),
        num_draft_tokens=cfg.get("spec_num_draft_tokens", None),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        enforce_eager=cfg.get("enforce_eager", False),
        cuda_graph_max_bs=cfg.get("cuda_graph_max_bs", 8),
    )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])

    def _tokenize(prompts):
        return [tok.encode(p, add_special_tokens=True) for p in prompts]

    # Warmup
    engine.generate(
        _tokenize(["Hello, who are you?"]),
        Eagle3SamplingParams(max_tokens=16),
    )

    import torch

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompts = scenario.get("prompts")
        if "prompt_token_ids" in scenario:
            prompt_token_ids = [list(p) for p in scenario["prompt_token_ids"]]
        else:
            prompt_token_ids = _tokenize(prompts)
        output_lens = scenario.get("output_lens")
        if output_lens is None:
            output_lens = [scenario["output_len"]] * len(prompt_token_ids)
        sp = [
            Eagle3SamplingParams(max_tokens=int(output_len), ignore_eos=True)
            for output_len in output_lens
        ]

        engine.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(
            prompt_token_ids, sp, use_tqdm=False, decode_text=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_output_tokens = sum(len(o.token_ids) for o in outputs)
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        out_records = [
            {
                "text": o.generated_text,
                "token_ids": list(o.token_ids),
            }
            for o in outputs
        ]

        all_results.append({
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": out_records,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        prompt_token_ids = _tokenize(prompts)
        sp = Eagle3SamplingParams(max_tokens=ls["output_len"], ignore_eos=True)
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.reset()
            torch.cuda.synchronize()
            engine.generate(prompt_token_ids, sp)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            engine.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompt_token_ids, sp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Alignment metric (matches bench_vllm.py).
# ---------------------------------------------------------------------------
def compute_alignment(a_outputs: list[dict], b_outputs: list[dict]) -> dict:
    total_seqs = len(a_outputs)
    exact_matches = 0
    total_matching_tokens = 0
    total_output_tokens = 0

    for a, b in zip(a_outputs, b_outputs):
        a_ids = a["token_ids"]
        b_ids = b["token_ids"]
        out_len = max(len(a_ids), len(b_ids))
        total_output_tokens += out_len

        if a_ids == b_ids:
            exact_matches += 1
            total_matching_tokens += len(a_ids)
        else:
            min_len = min(len(a_ids), len(b_ids))
            matching = sum(1 for j in range(min_len) if a_ids[j] == b_ids[j])
            total_matching_tokens += matching

    avg_matching = total_matching_tokens / total_seqs if total_seqs else 0
    avg_output_len = total_output_tokens / total_seqs if total_seqs else 0
    return {
        "exact_matches": exact_matches,
        "total_seqs": total_seqs,
        "total_matching_tokens": total_matching_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_matching_tokens_per_request": avg_matching,
        "avg_output_len": avg_output_len,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EAGLE-3 spec-decoding benchmark: kb-nano vs sglang",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--draft-model", type=str,
        default="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
        help="EAGLE-3 head used by both engines.",
    )
    parser.add_argument("--num-seqs", type=int, default=16,
                        help="Total prompts per throughput scenario.")
    parser.add_argument("--output-len", type=int, default=256,
                        help="Max new tokens per request.")
    parser.add_argument(
        "--workload",
        choices=["fixed", "wildchat"],
        default="fixed",
        help="Throughput workload: fixed built-in prompts or WildChat-derived "
             "HF datasets.",
    )
    parser.add_argument(
        "--scenario",
        choices=[*WILDCHAT_SCENARIOS, "all"],
        default="balanced",
        help="WildChat throughput scenario to run. Ignored for --workload=fixed.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy for alignment).")
    parser.add_argument("--latency-iters", type=int, default=5)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    # The following EAGLE-3 knobs follow sglang's reference defaults. Runtime
    # batch/graph limits are exposed below so larger workloads can use a larger
    # identical batch cap for both engines.
    spec_steps = 3
    spec_topk = 4
    parser.add_argument("--spec-num-draft-tokens", type=int, default=16)
    gpu_memory_utilization = 0.7
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--skip-sglang", action="store_true")
    parser.add_argument("--skip-kb-nano", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Run both engines in pure eager mode (no CUDA graphs, no torch.compile). "
             "On sglang this maps to disable_cuda_graph=True; kb-nano is already "
             "eager so this is a no-op on its side. Use for apples-to-apples "
             "comparisons isolating raw kernel/dispatch perf.",
    )
    parser.add_argument(
        "--sglang-python", type=str,
        default="/home/yak/miniconda3/envs/sglang-bench/bin/python",
        help="Python interpreter to use for the sglang subprocess. We launch "
             "sglang in an isolated conda env so its torch / CUDA versions do "
             "not conflict with kb-nano's (e.g. sglang ships torch 2.9 + cu128, "
             "kb-nano runs on torch 2.10 + cu130). Default points at the "
             "`sglang-bench` env created by tests/setup_sglang_env.sh.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to dump per-scenario outputs / results.json. "
             "Default: tests/results/<gpu>/<model>_eagle3",
    )
    args = parser.parse_args()
    args.spec_steps = spec_steps
    args.spec_topk = spec_topk
    args.gpu_memory_utilization = gpu_memory_utilization

    gpu = _detect_gpu_name()
    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(
            _PACKAGE_DIR / "tests" / "results" / gpu / f"{short}_eagle3"
        )

    # Build scenarios.
    scenarios = []
    if not args.skip_throughput:
        if args.workload == "fixed":
            prompts = _build_prompt_set(args.num_seqs, args.seed)
            scenarios.append({
                "name": f"eagle3-{args.num_seqs}seqs-out{args.output_len}",
                "prompts": prompts,
                "output_len": args.output_len,
            })
        else:
            scenario_names = (
                list(WILDCHAT_SCENARIOS)
                if args.scenario == "all"
                else [args.scenario]
            )
            scenarios.extend(_build_wildchat_scenarios(
                args.model,
                scenario_names,
                args.num_seqs,
                args.seed,
                args.max_model_len,
            ))

    latency_scenarios = []
    if not args.skip_latency:
        bs = args.latency_batch_size
        prompts = _build_prompt_set(bs, args.seed + 100)
        latency_scenarios.append({
            "name": f"latency-bs{bs}-out{args.output_len}",
            "prompts": prompts,
            "output_len": args.output_len,
            "batch_size": bs,
            "num_warmup": 2,
            "num_iters": args.latency_iters,
        })

    print("=" * 70)
    print("  EAGLE-3 Speculative Decoding Benchmark: kb-nano vs sglang")
    print("=" * 70)
    print(f"  Target model           : {args.model}")
    print(f"  Draft  model           : {args.draft_model}")
    print(f"  GPU                    : {gpu}")
    print(f"  workload               : {args.workload}")
    if args.workload == "wildchat":
        print(f"  wildchat scenario      : {args.scenario}")
    print(f"  num_seqs / scenario    : {args.num_seqs}")
    print(f"  output_len             : {args.output_len}")
    print(f"  spec_steps             : {args.spec_steps}")
    print(f"  spec_topk (sglang)     : {args.spec_topk}")
    print(f"  spec_num_draft_tokens  : {args.spec_num_draft_tokens}")
    print(f"  max_model_len          : {args.max_model_len}")
    print(f"  max_running_requests   : {args.max_running_requests}")
    print(f"  temperature            : {args.temperature}")
    print(f"  seed                   : {args.seed}")
    print(f"  output_dir             : {args.output_dir}")
    print("=" * 70)

    # ------------------ sglang ------------------
    sgl_raw = None
    if not args.skip_sglang:
        sgl_cfg = {
            "model": args.model,
            "draft_model": args.draft_model,
            "seed": args.seed,
            "temperature": args.temperature,
            "spec_steps": args.spec_steps,
            "spec_topk": args.spec_topk,
            "spec_num_draft_tokens": args.spec_num_draft_tokens,
            "max_running_requests": args.max_running_requests,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "max_model_len": args.max_model_len,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
            "disable_cuda_graph": args.enforce_eager,
        }
        sgl_raw = run_worker(
            SGLANG_WORKER, sgl_cfg,
            f"sglang [EAGLE-3] {args.model.split('/')[-1]}",
            python_executable=args.sglang_python,
        )
        if sgl_raw is None:
            print("  WARNING: sglang subprocess failed -- continuing with kb-nano only")

    # ------------------ kb-nano ------------------
    kb_raw = None
    if not args.skip_kb_nano:
        kb_cfg = {
            "model": args.model,
            "draft_model": args.draft_model,
            "seed": args.seed,
            "temperature": args.temperature,
            "spec_steps": args.spec_steps,
            "spec_topk": args.spec_topk,
            "spec_num_draft_tokens": args.spec_num_draft_tokens,
            "max_model_len": args.max_model_len,
            "max_running_requests": args.max_running_requests,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "enforce_eager": args.enforce_eager,
            "project_root": str(_PROJECT_ROOT),
            "package_name": _PACKAGE_DIR.name,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
        }
        kb_raw = run_worker(
            KB_NANO_EAGLE3_WORKER, kb_cfg,
            f"kb-nano [EAGLE-3] {args.model.split('/')[-1]}",
        )
        if kb_raw is None:
            print("  ERROR: kb-nano subprocess failed.")
            sys.exit(1)

    # ------------------ Throughput summary + alignment ------------------
    throughput_summary = []
    if scenarios and (kb_raw or sgl_raw):
        kb_thr = kb_raw.get("throughput", []) if kb_raw else []
        sgl_thr = sgl_raw.get("throughput", []) if sgl_raw else []

        print(f"\n{'=' * 100}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 100}")
        print(
            f"  {'SCENARIO':<32} {'KB-NANO tok/s':>14} {'SGLANG tok/s':>14}"
            f" {'SPEEDUP':>8} {'AVG MATCH TOKS':>18}"
        )
        print(f"  {'-' * 96}")

        for i, sc in enumerate(scenarios):
            kb_data = kb_thr[i] if i < len(kb_thr) else None
            sg_data = sgl_thr[i] if i < len(sgl_thr) else None

            entry = {"scenario": sc["name"], "output_len": sc["output_len"]}
            for key in (
                "dataset",
                "avg_input_len",
                "max_input_len",
                "avg_output_len",
                "max_output_len",
            ):
                if key in sc:
                    entry[key] = sc[key]
            kb_tps_str = "N/A"
            sg_tps_str = "N/A"
            speedup_str = "N/A"
            match_str = "N/A"

            if kb_data:
                kb_tps = kb_data["total_output_tokens"] / max(1e-9, kb_data["elapsed"])
                kb_tps_str = f"{kb_tps:,.1f}"
                entry["kb_nano_elapsed"] = kb_data["elapsed"]
                entry["kb_nano_output_tokens"] = kb_data["total_output_tokens"]
                entry["kb_nano_tok_per_s"] = kb_tps
            if sg_data:
                sg_tps = sg_data["total_output_tokens"] / max(1e-9, sg_data["elapsed"])
                sg_tps_str = f"{sg_tps:,.1f}"
                entry["sglang_elapsed"] = sg_data["elapsed"]
                entry["sglang_output_tokens"] = sg_data["total_output_tokens"]
                entry["sglang_tok_per_s"] = sg_tps
            if kb_data and sg_data:
                entry["speedup_vs_sglang"] = entry["kb_nano_tok_per_s"] / entry["sglang_tok_per_s"]
                speedup_str = f"{entry['speedup_vs_sglang']:.2f}x"

                if args.temperature == 0.0:
                    align = compute_alignment(kb_data["outputs"], sg_data["outputs"])
                    entry["alignment"] = align
                    avg_match = align["avg_matching_tokens_per_request"]
                    avg_out = align["avg_output_len"]
                    match_str = f"{avg_match:.1f}/{avg_out:.0f}"

            print(
                f"  {sc['name']:<32} {kb_tps_str:>14} {sg_tps_str:>14}"
                f" {speedup_str:>8} {match_str:>18}"
            )
            throughput_summary.append(entry)

        print(f"{'=' * 100}")

        if throughput_summary and "alignment" in throughput_summary[0]:
            avg_match = throughput_summary[0]["alignment"]["avg_matching_tokens_per_request"]
            print(f"\n  Alignment vs sglang (greedy): "
                  f"avg matching tokens = {avg_match:.1f} / request")
            if avg_match >= 100:
                print(f"  PASS: avg_matching_tokens_per_request = {avg_match:.1f} >= 100")
            else:
                print(f"  WARN: avg_matching_tokens_per_request = {avg_match:.1f} < 100")

    # ------------------ Latency summary ------------------
    latency_summary = []
    if latency_scenarios and (kb_raw or sgl_raw):
        kb_lat = kb_raw.get("latency", []) if kb_raw else []
        sg_lat = sgl_raw.get("latency", []) if sgl_raw else []

        print(f"\n{'=' * 110}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 110}")
        print(
            f"  {'SCENARIO':<32} {'BS':>4} {'OUT':>5} {'ITERS':>6}"
            f"  {'KB-NANO med':>13} {'SGLANG med':>13}"
            f"  {'KB ms/tok':>11} {'SG ms/tok':>11} {'SPEEDUP':>9}"
        )
        print(f"  {'-' * 106}")

        for i, ls in enumerate(latency_scenarios):
            kb_lat_i = kb_lat[i] if i < len(kb_lat) else None
            sg_lat_i = sg_lat[i] if i < len(sg_lat) else None
            entry = {
                "scenario": ls["name"],
                "batch_size": ls["batch_size"],
                "output_len": ls["output_len"],
                "num_iters": ls["num_iters"],
            }

            kb_med_str = "N/A"
            sg_med_str = "N/A"
            kb_ms_str = "N/A"
            sg_ms_str = "N/A"
            speedup_str = "N/A"

            total_out = ls["batch_size"] * ls["output_len"]

            if kb_lat_i:
                kb_lats = np.array(kb_lat_i["latencies"])
                kb_med = float(np.median(kb_lats))
                entry["kb_nano_median_s"] = kb_med
                entry["kb_nano_p99_s"] = float(np.percentile(kb_lats, 99))
                entry["kb_nano_ms_per_tok"] = (kb_med / total_out) * 1000
                entry["kb_nano_latencies"] = kb_lat_i["latencies"]
                kb_med_str = f"{kb_med:.4f}s"
                kb_ms_str = f"{entry['kb_nano_ms_per_tok']:.2f}"
            if sg_lat_i:
                sg_lats = np.array(sg_lat_i["latencies"])
                sg_med = float(np.median(sg_lats))
                entry["sglang_median_s"] = sg_med
                entry["sglang_p99_s"] = float(np.percentile(sg_lats, 99))
                entry["sglang_ms_per_tok"] = (sg_med / total_out) * 1000
                entry["sglang_latencies"] = sg_lat_i["latencies"]
                sg_med_str = f"{sg_med:.4f}s"
                sg_ms_str = f"{entry['sglang_ms_per_tok']:.2f}"
            if kb_lat_i and sg_lat_i:
                entry["speedup_vs_sglang"] = entry["sglang_median_s"] / entry["kb_nano_median_s"]
                speedup_str = f"{entry['speedup_vs_sglang']:.2f}x"

            print(
                f"  {ls['name']:<32} {ls['batch_size']:>4} {ls['output_len']:>5}"
                f" {ls['num_iters']:>6}"
                f"  {kb_med_str:>13} {sg_med_str:>13}"
                f"  {kb_ms_str:>11} {sg_ms_str:>11} {speedup_str:>9}"
            )
            latency_summary.append(entry)

        print(f"{'=' * 110}")

    # ------------------ Save outputs ------------------
    if args.output_dir and (throughput_summary or latency_summary):
        os.makedirs(args.output_dir, exist_ok=True)

        if throughput_summary and (kb_raw or sgl_raw):
            for i, sc in enumerate(scenarios):
                sc_dir = os.path.join(args.output_dir, sc["name"])
                os.makedirs(sc_dir, exist_ok=True)
                if kb_raw and i < len(kb_raw.get("throughput", [])):
                    with open(os.path.join(sc_dir, "kb_nano_outputs.json"), "w") as f:
                        json.dump(kb_raw["throughput"][i], f, indent=2)
                if sgl_raw and i < len(sgl_raw.get("throughput", [])):
                    with open(os.path.join(sc_dir, "sglang_outputs.json"), "w") as f:
                        json.dump(sgl_raw["throughput"][i], f, indent=2)

        combined = {
            "gpu": gpu,
            "model": args.model,
            "draft_model": args.draft_model,
            "seed": args.seed,
            "temperature": args.temperature,
            "workload": args.workload,
            "wildchat_scenario": args.scenario if args.workload == "wildchat" else None,
            "num_seqs": args.num_seqs,
            "output_len": args.output_len,
            "spec_steps": args.spec_steps,
            "spec_topk_sglang": args.spec_topk,
            "spec_num_draft_tokens": args.spec_num_draft_tokens,
        }
        if throughput_summary:
            combined["scenarios"] = throughput_summary
        if latency_summary:
            combined["latency_scenarios"] = latency_summary

        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
