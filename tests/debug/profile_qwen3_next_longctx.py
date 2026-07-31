#!/usr/bin/env python3
"""Where does the long-context scenario's wall time actually go?

Splits the real LongBench-v2 workload into its prefill and decode phases by
timing ``max_tokens=1`` against the full run, for either engine. Also reports,
for fastkernels, the KV block budget and worst-case reservation per sequence,
which is what bounds how many sequences are admitted at once.

The synthetic probes in ``profile_qwen3_next_gap.py`` all use uniform-length
batches, so they cannot see a kernel that pads a mixed-length batch to its
longest sequence. This runs the real length distribution (8K..128K) through both
engines so the 2 s gap can be attributed to a phase before anything is changed.

Usage:
    python tests/debug/profile_qwen3_next_longctx.py --engine fastkernels --tp 2
    python tests/debug/profile_qwen3_next_longctx.py --engine vllm --tp 2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def load_workload(args):
    from transformers import AutoTokenizer

    from fastkernels.workloads import (
        DEFAULT_WORKLOAD_DATASETS, load_real_prompt_workload,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    samples = load_real_prompt_workload(
        "long-context", tok, num_requests=args.num_seqs,
        decode_cap=args.output_len,
        dataset_name=DEFAULT_WORKLOAD_DATASETS["long-context"],
        seed=args.seed,
    )
    prompts = [s.prompt_token_ids for s in samples]
    out_lens = [s.output_len for s in samples]
    lens = sorted(len(p) for p in prompts)
    total_in = sum(lens)
    print(f"\n  {len(prompts)} prompts  total_in={total_in:,}  "
          f"min={lens[0]:,} p50={lens[len(lens) // 2]:,} max={lens[-1]:,}")
    print(f"  output_lens: min={min(out_lens)} max={max(out_lens)} "
          f"total={sum(out_lens):,}")
    return prompts, out_lens, total_in


def report(pf, full, total_in, steps):
    print(f"\n  prefill {pf:7.3f}s ({total_in / pf:>9,.0f} tok/s)   "
          f"decode {full - pf:6.3f}s ({(full - pf) / steps * 1e3:5.2f} ms/step)"
          f"   total {full:7.3f}s", flush=True)


def run_vllm(args, prompts, out_lens, total_in):
    import torch
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    max_model_len = max(len(p) + ol for p, ol in zip(prompts, out_lens)) + 8
    kwargs = dict(
        model=MODEL, seed=args.seed, trust_remote_code=True, enforce_eager=False,
        tensor_parallel_size=args.tp, gpu_memory_utilization=0.9,
        max_model_len=max_model_len, enable_prefix_caching=False,
    )
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    llm = LLM(**kwargs)
    vp = [dict(prompt_token_ids=p) for p in prompts]

    def timed(one, label):
        sp = [
            SamplingParams(temperature=0.0, ignore_eos=True, detokenize=False,
                           max_tokens=(1 if one else ol))
            for ol in out_lens
        ]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(vp, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {label:<22} {dt:7.3f}s", flush=True)
        return dt

    timed(True, "prefill warmup")
    pf = timed(True, "prefill only")
    full = timed(False, "prefill + decode")
    report(pf, full, total_in, max(out_lens))


def run_fastkernels(args, prompts, out_lens, total_in):
    import torch

    from fastkernels.infra.engine import LlamaEngine, SamplingParams

    max_model_len = max(len(p) + ol for p, ol in zip(prompts, out_lens)) + 8
    engine = LlamaEngine(
        model_name=MODEL, seed=args.seed, enforce_eager=False,
        tensor_parallel_size=args.tp, max_model_len=max_model_len,
    )
    engine.generate([[0] * 16],
                    SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))

    sm = engine.model_runner.mamba_state_manager
    block_size = getattr(sm, "block_size", 0)
    total_blocks = getattr(sm, "num_mla_blocks", 0)
    worst = [
        (len(p) + ol + block_size - 1) // block_size
        for p, ol in zip(prompts, out_lens)
    ]
    print(f"\n  block_size={block_size} total_kv_blocks={total_blocks:,} "
          f"state_slots={getattr(sm, 'num_slots', 0)}")
    print(f"  worst-case blocks/seq: min={min(worst):,} max={max(worst):,} "
          f"sum={sum(worst):,}")
    budget = max(1, total_blocks - 2) if total_blocks else 0
    rounds, i, per_round = 0, 0, []
    while i < len(worst):
        rounds += 1
        used, n = 0, 0
        while i < len(worst) and (n == 0 or used + worst[i] <= budget):
            used += worst[i]
            n += 1
            i += 1
        per_round.append(n)
    print(f"  => admission rounds={rounds} sizes={per_round}")

    def timed(one, label):
        sp = [
            SamplingParams(temperature=0.0, ignore_eos=True,
                           max_tokens=(1 if one else ol))
            for ol in out_lens
        ]
        engine.block_manager.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate(prompts, sp, use_tqdm=False, decode_text=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {label:<22} {dt:7.3f}s", flush=True)
        return dt

    timed(True, "prefill warmup")
    pf = timed(True, "prefill only")
    full = timed(False, "prefill + decode")
    report(pf, full, total_in, max(out_lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fastkernels", "vllm"],
                    default="fastkernels")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--num-seqs", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompts, out_lens, total_in = load_workload(args)
    if args.engine == "vllm":
        run_vllm(args, prompts, out_lens, total_in)
    else:
        run_fastkernels(args, prompts, out_lens, total_in)


if __name__ == "__main__":
    main()
