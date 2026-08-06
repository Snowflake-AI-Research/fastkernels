#!/usr/bin/env python3
"""Where does the Gemma4 throughput gap against vLLM actually go?

Gemma4-26B-A4B loses on both throughput scenarios (mixed 0.86x, long-context
0.93x) while *winning* both latency scenarios (1.07x), so the gap is not a
per-kernel deficit -- it only appears once many sequences are in flight. This
splits the real workload into its prefill and decode phases by timing
``max_tokens=1`` against the full run, for either engine, so the phase can be
attributed before anything is changed.

For fastkernels it also reports the KV block budget and the worst-case
reservation per sequence, which is what bounds how many sequences are admitted
at once. Gemma4 is a hybrid model (25 sliding_attention layers with a 1024
window + 5 full_attention layers), so an engine that reserves full sequence
length in *every* layer holds far fewer sequences than one that bounds the
sliding layers at the window.

Usage:
    python tests/debug/profile_gemma4_phases.py --engine fastkernels --scenario mixed
    python tests/debug/profile_gemma4_phases.py --engine vllm --scenario mixed
    python tests/debug/profile_gemma4_phases.py --engine fastkernels --scenario long-context
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

MODEL = "google/gemma-4-26B-A4B-it"


def load_workload(args):
    from transformers import AutoTokenizer

    from fastkernels.workloads import (
        DEFAULT_DECODE_CAPS, DEFAULT_WORKLOAD_DATASETS,
        load_real_prompt_workload,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cap = args.output_len or DEFAULT_DECODE_CAPS[args.scenario]
    samples = load_real_prompt_workload(
        args.scenario, tok, num_requests=args.num_seqs,
        decode_cap=cap,
        dataset_name=DEFAULT_WORKLOAD_DATASETS[args.scenario],
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


def report(pf, full, total_in, total_out, steps):
    dec = full - pf
    print(f"\n  prefill {pf:7.3f}s ({total_in / pf:>9,.0f} tok/s)   "
          f"decode {dec:6.3f}s ({dec / steps * 1e3:5.2f} ms/step)"
          f"   total {full:7.3f}s ({total_out / full:>8,.0f} out tok/s)",
          flush=True)


def run_vllm(args, prompts, out_lens, total_in):
    import torch
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    max_model_len = max(len(p) + ol for p, ol in zip(prompts, out_lens)) + 8
    kwargs = dict(
        model=MODEL, seed=args.seed, trust_remote_code=True, enforce_eager=False,
        tensor_parallel_size=args.tp, gpu_memory_utilization=0.9,
        max_model_len=max_model_len, enable_prefix_caching=False,
        disable_log_stats=True,
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
    report(pf, full, total_in, sum(out_lens), max(out_lens))


def _report_kv_budget(engine, prompts, out_lens):
    """Print the block budget and what it implies for concurrency.

    ``worst`` is each request's *final* length in blocks. The scheduler no
    longer reserves that (it reserves the request's current length, as vLLM's
    ``full_sequence_must_fit`` does), so ``sum(worst) > total_blocks`` is not by
    itself a limit -- it only says the pool could not hold every request at its
    final length simultaneously, which is fine because requests retire as
    others grow. It is reported because it *was* the binding constraint before,
    and because it still bounds the worst case.
    """
    from fastkernels.infra.engine import BLOCK_SIZE

    total_blocks = engine.block_manager._num_blocks
    worst = [
        (len(p) + ol + BLOCK_SIZE - 1) // BLOCK_SIZE
        for p, ol in zip(prompts, out_lens)
    ]
    cur = [(len(p) + BLOCK_SIZE - 1) // BLOCK_SIZE for p in prompts]
    print(f"\n  block_size={BLOCK_SIZE} total_kv_blocks={total_blocks:,} "
          f"(= {total_blocks * BLOCK_SIZE:,} token slots)")
    print(f"  blocks/seq at admission (prompt only): min={min(cur):,} "
          f"max={max(cur):,} sum={sum(cur):,}")
    print(f"  blocks/seq at final length:            min={min(worst):,} "
          f"max={max(worst):,} sum={sum(worst):,}")
    print(f"  => largest single request needs {max(cur) / total_blocks:.1%} of "
          f"the pool at admission; all final lengths would need "
          f"{sum(worst) / total_blocks:.1f}x the pool")


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

    _report_kv_budget(engine, prompts, out_lens)

    def timed(one, label):
        sp = [
            SamplingParams(temperature=0.0, ignore_eos=True,
                           max_tokens=(1 if one else ol))
            for ol in out_lens
        ]
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
    report(pf, full, total_in, sum(out_lens), max(out_lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fastkernels", "vllm"],
                    default="fastkernels")
    ap.add_argument("--scenario", choices=["mixed", "long-context"],
                    default="mixed")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-seqs", type=int, default=None)
    ap.add_argument("--output-len", type=int, default=None,
                    help="Override the scenario's decode cap.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.num_seqs is None:
        args.num_seqs = 1000 if args.scenario == "mixed" else 64

    prompts, out_lens, total_in = load_workload(args)
    if args.engine == "vllm":
        run_vllm(args, prompts, out_lens, total_in)
    else:
        run_fastkernels(args, prompts, out_lens, total_in)


if __name__ == "__main__":
    main()
