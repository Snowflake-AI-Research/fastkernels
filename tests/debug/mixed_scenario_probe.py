#!/usr/bin/env python3
"""Single-engine probe of one throughput scenario, at a chosen context length.

``bench_vllm`` sizes ``max_model_len`` from the *largest* scenario in the run, so
gemma-4's mixed row is measured at 167,799 tokens in a full validate pass but at
9,982 when ``--scenario mixed`` runs alone. This probe takes ``--max-model-len``
explicitly so that variable can be isolated, and drives one engine at a time so a
run costs ~2 minutes instead of ~12 -- which is what makes it usable for A/B'ing
scheduler settings (``FASTKERNELS_MM_ADMIT_SCOPE``,
``FASTKERNELS_MM_ADMIT_STRIDE``, ...) across models.

The warmup sequence is copied from ``bench_vllm``'s workers verbatim (engine
warmup, then a ``max_tokens=1`` pass at the scenario's real shapes, then
``block_manager.reset()`` on the fastkernels side) so the timed region here is
the same one the benchmark reports. Pair it with ``FASTKERNELS_STEP_PROFILE=1``
for the step/occupancy/admission breakdown.

Usage:
    python tests/debug/mixed_scenario_probe.py --engine fastkernels --max-model-len 167799
    python tests/debug/mixed_scenario_probe.py --engine vllm --max-model-len 9982
    FASTKERNELS_STEP_PROFILE=1 python tests/debug/mixed_scenario_probe.py \
        --model meta-llama/Llama-3.1-8B-Instruct --max-model-len 9982
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))



def load_workload(args):
    from transformers import AutoTokenizer

    from fastkernels.workloads import (
        DEFAULT_DECODE_CAPS, DEFAULT_WORKLOAD_DATASETS,
        load_real_prompt_workload,
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples = load_real_prompt_workload(
        args.scenario, tok, num_requests=args.num_seqs,
        decode_cap=DEFAULT_DECODE_CAPS[args.scenario],
        dataset_name=DEFAULT_WORKLOAD_DATASETS[args.scenario],
        seed=args.seed,
    )
    prompts = [s.prompt_token_ids for s in samples]
    out_lens = [s.output_len for s in samples]
    lens = sorted(len(p) for p in prompts)
    print(f"\n  {len(prompts)} prompts  total_in={sum(lens):,}  "
          f"min={lens[0]:,} p50={lens[len(lens) // 2]:,} max={lens[-1]:,}")
    print(f"  out_lens: min={min(out_lens)} max={max(out_lens)} "
          f"total={sum(out_lens):,}")
    return prompts, out_lens, sum(lens)


def run_fastkernels(args, prompts, out_lens):
    import torch

    from fastkernels.infra.engine import LlamaEngine, SamplingParams

    engine = LlamaEngine(
        model_name=args.model, seed=args.seed, enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
        **({"kv_cache_dtype": args.kv_cache_dtype}
           if args.kv_cache_dtype else {}),
    )
    engine.generate([[0] * 16], SamplingParams(temperature=0.0, max_tokens=16,
                                               ignore_eos=True))
    sp_list = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in out_lens
    ]
    engine.generate(prompts,
                    SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                    use_tqdm=False, decode_text=False)
    engine.block_manager.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sp_list, use_tqdm=False, decode_text=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, sum(len(o.token_ids) for o in outs)


def run_vllm(args, prompts, out_lens):
    import torch
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    kwargs = dict(
        model=args.model, seed=args.seed, trust_remote_code=True,
        enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9, max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        disable_log_stats=not args.log_stats,
        **({"kv_cache_dtype": args.kv_cache_dtype}
           if args.kv_cache_dtype else {}),
    )
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    llm = LLM(**kwargs)
    llm.generate([dict(prompt_token_ids=[0] * 16)],
                 SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
    vp = [dict(prompt_token_ids=p) for p in prompts]
    sp_list = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True,
                       detokenize=False)
        for ol in out_lens
    ]
    llm.generate(vp, SamplingParams(temperature=0.0, max_tokens=1,
                                    ignore_eos=True, detokenize=False),
                 use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(vp, sp_list, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, sum(len(c.token_ids) for o in outs for c in o.outputs if c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fastkernels", "vllm"],
                    default="fastkernels")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--scenario", default="mixed")
    ap.add_argument("--max-model-len", type=int, default=167799,
                    help="167799 reproduces a full validate pass; 9982 is what "
                         "--scenario mixed alone resolves to.")
    ap.add_argument("--num-seqs", type=int, default=1000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--kv-cache-dtype", default="",
                    help="e.g. fp8_e4m3, for rows whose sweep entry sets it "
                         "(it changes the KV pool size, hence admission).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--log-stats", action="store_true",
                    help="vLLM only: keep its periodic Running/Waiting/KV-usage "
                         "log, which is the only view of its batch occupancy.")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    prompts, out_lens, _ = load_workload(args)
    runner = run_vllm if args.engine == "vllm" else run_fastkernels
    dt, out_toks = runner(args, prompts, out_lens)
    print(f"\n  RESULT {args.label or args.engine} "
          f"max_model_len={args.max_model_len} "
          f"elapsed={dt:.3f}s out_tokens={out_toks:,} "
          f"tok/s={out_toks / dt:,.0f}", flush=True)


if __name__ == "__main__":
    main()
