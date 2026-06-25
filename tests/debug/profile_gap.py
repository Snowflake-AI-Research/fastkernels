#!/usr/bin/env python3
"""
Profile the performance gap between our engine and vLLM.

Measures prefill vs decode time, per-step latency breakdown (scheduling,
GPU call, tolist sync, post-processing), and scaling across batch sizes.

Usage:
    python -m fastkernels.tests.profile_gap \
        --model meta-llama/Llama-3.1-8B-Instruct --tp 4

    # Multiple batch sizes
    python -m fastkernels.tests.profile_gap \
        --model meta-llama/Llama-3.1-8B-Instruct --tp 4 \
        --batch-sizes 32 64 128 256

    # Also run 70B to check model-size scaling
    python -m fastkernels.tests.profile_gap \
        --model meta-llama/Llama-3.1-70B-Instruct --tp 4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from random import randint, seed as set_seed

# ---------------------------------------------------------------------------
# vLLM worker script (subprocess)
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, os, sys, time

def main():
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from vllm import LLM, SamplingParams

    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    llm = LLM(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=False,
        tensor_parallel_size=cfg["tp"],
    )

    prompt_token_ids = cfg["prompt_token_ids"]
    output_lens = cfg["output_lens"]

    sp_list = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=ol)
        for ol in output_lens
    ]

    # Warmup
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    vllm_prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    start = time.perf_counter()
    outputs = llm.generate(vllm_prompts, sp_list, use_tqdm=False)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_input_tokens = sum(len(p) for p in prompt_token_ids)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
            "total_input_tokens": total_input_tokens,
        }, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Our engine worker script (subprocess, with FASTKERNELS_PROFILE=1)
# ---------------------------------------------------------------------------
OURS_WORKER = r'''
import json, os, sys, time
os.environ["FASTKERNELS_PROFILE"] = "1"

with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    pkg = cfg["package_name"]
    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine = LlamaEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=False,
        tensor_parallel_size=cfg["tp"],
    )

    prompt_token_ids = cfg["prompt_token_ids"]
    output_lens = cfg["output_lens"]

    sp_list = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in output_lens
    ]

    # Warmup
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompt_token_ids, sp_list)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(len(o.token_ids) for o in outputs)

    profile = getattr(engine, '_profile_data', {})
    profile["elapsed"] = elapsed
    profile["total_output_tokens"] = total_output_tokens
    profile["total_input_tokens"] = sum(len(p) for p in prompt_token_ids)

    with open(cfg["output_file"], "w") as f:
        json.dump(profile, f)

    del engine

if __name__ == "__main__":
    main()
'''


def run_worker(script: str, config: dict, label: str) -> dict | None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
        f.write(script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp",
    ) as f:
        config["output_file"] = output_path
        json.dump(config, f)
        config_path = f.name

    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = subprocess.run(
            [sys.executable, script_path, config_path],
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed (exit {result.returncode})")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        os.unlink(config_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def generate_workload(num_seqs, input_len, output_len, seed_val=0):
    set_seed(seed_val)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, input_len))]
        for _ in range(num_seqs)
    ]
    output_lens = [
        randint(100, output_len) for _ in range(num_seqs)
    ]
    return prompt_token_ids, output_lens


def print_ours_breakdown(data):
    """Print detailed timing breakdown from our profiled engine."""
    elapsed = data["elapsed"]
    pf = data.get("prefill_time", 0)
    dc = data.get("decode_time", 0)
    dc_steps = data.get("decode_steps", 0)
    dc_tokens = data.get("decode_tokens", 0)
    pf_steps = data.get("prefill_steps", 0)
    pf_tokens = data.get("prefill_tokens", 0)

    dc_sched = data.get("decode_sched_time", 0)
    dc_call = data.get("decode_call_time", 0)
    dc_tolist = data.get("decode_tolist_time", 0)
    dc_post = data.get("decode_post_time", 0)

    bs_counts = data.get("decode_bs_counts", [])
    avg_bs = sum(bs_counts) / len(bs_counts) if bs_counts else 0

    total_out = data["total_output_tokens"]
    tps = total_out / elapsed if elapsed > 0 else 0

    print(f"    Total time        : {elapsed:.3f}s")
    print(f"    Throughput        : {tps:,.1f} tok/s")
    print(f"    Output tokens     : {total_out:,}")
    print()
    print(f"    Prefill:")
    print(f"      Time            : {pf:.3f}s  ({pf/elapsed*100:.1f}%)")
    print(f"      Steps           : {pf_steps}")
    print(f"      Tokens          : {pf_tokens:,}")
    if pf_steps > 0:
        print(f"      Avg step        : {pf/pf_steps*1000:.2f}ms")
    print()
    print(f"    Decode:")
    print(f"      Time            : {dc:.3f}s  ({dc/elapsed*100:.1f}%)")
    print(f"      Steps           : {dc_steps}")
    print(f"      Tokens          : {dc_tokens:,}")
    print(f"      Avg batch size  : {avg_bs:.1f}")
    if dc_steps > 0:
        avg_step = dc / dc_steps * 1000
        print(f"      Avg step        : {avg_step:.3f}ms")
        print(f"      Breakdown per step:")
        print(f"        Scheduling    : {dc_sched/dc_steps*1000:.3f}ms ({dc_sched/dc*100:.1f}%)")
        print(f"        GPU call      : {dc_call/dc_steps*1000:.3f}ms ({dc_call/dc*100:.1f}%)")
        print(f"        .tolist() sync: {dc_tolist/dc_steps*1000:.3f}ms ({dc_tolist/dc*100:.1f}%)")
        print(f"        Post-process  : {dc_post/dc_steps*1000:.3f}ms ({dc_post/dc*100:.1f}%)")
    print()
    unaccounted = elapsed - pf - dc
    print(f"    Unaccounted       : {unaccounted:.3f}s  ({unaccounted/elapsed*100:.1f}%)")

    cd = data.get("call_detail")
    if cd:
        print()
        print(f"    call_decode_greedy detail (avg over {cd['n_calls']} calls):")
        print(f"      Prepare arrays  : {cd['prepare_ms']:.3f}ms")
        print(f"      Signal workers  : {cd['signal_ms']:.3f}ms")
        print(f"      GPU exec (sync) : {cd['gpu_exec_ms']:.3f}ms")


def run_one_batch_size(model, tp, num_seqs, input_len, output_len, seed_val,
                       project_root, package_name, skip_vllm=False):
    """Run both engines for a single batch size and return results dict."""
    prompt_token_ids, output_lens = generate_workload(
        num_seqs, input_len, output_len, seed_val
    )
    total_input = sum(len(p) for p in prompt_token_ids)
    total_output_expected = sum(output_lens)

    config = {
        "model": model,
        "tp": tp,
        "seed": seed_val,
        "prompt_token_ids": prompt_token_ids,
        "output_lens": output_lens,
        "project_root": project_root,
        "package_name": package_name,
    }

    short_name = model.split("/")[-1]

    vllm_data = None
    if not skip_vllm:
        vllm_data = run_worker(
            VLLM_WORKER, dict(config),
            f"vLLM  [{short_name}] TP={tp} seqs={num_seqs}",
        )

    ours_data = run_worker(
        OURS_WORKER, dict(config),
        f"Ours  [{short_name}] TP={tp} seqs={num_seqs} (profiled)",
    )

    return {
        "num_seqs": num_seqs,
        "total_input": total_input,
        "total_output_expected": total_output_expected,
        "vllm": vllm_data,
        "ours": ours_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile performance gap")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[256],
                        help="Batch sizes to test (default: [256])")
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM run (only profile our engine)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write raw results to JSON file")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(package_dir)
    package_name = os.path.basename(package_dir)

    short_name = args.model.split("/")[-1]

    print("=" * 70)
    print("  Performance Gap Profiler")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  TP             : {args.tp}")
    print(f"  Batch sizes    : {args.batch_sizes}")
    print(f"  Input lengths  : 100-{args.max_input_len}")
    print(f"  Output lengths : 100-{args.max_output_len}")
    print("=" * 70)

    all_results = []

    for bs in args.batch_sizes:
        print(f"\n{'#' * 70}")
        print(f"  BATCH SIZE = {bs}")
        print(f"{'#' * 70}")

        result = run_one_batch_size(
            args.model, args.tp, bs,
            args.max_input_len, args.max_output_len,
            args.seed, project_root, package_name,
            skip_vllm=args.skip_vllm,
        )
        all_results.append(result)

        print(f"\n{'=' * 70}")
        print(f"  RESULTS: {short_name} TP={args.tp} seqs={bs}")
        print(f"{'=' * 70}")

        if result["vllm"]:
            vd = result["vllm"]
            v_tps = vd["total_output_tokens"] / vd["elapsed"]
            print(f"\n  vLLM:")
            print(f"    Total time        : {vd['elapsed']:.3f}s")
            print(f"    Throughput        : {v_tps:,.1f} tok/s")
            print(f"    Output tokens     : {vd['total_output_tokens']:,}")
        else:
            v_tps = None
            print(f"\n  vLLM: SKIPPED or FAILED")

        if result["ours"]:
            od = result["ours"]
            o_tps = od["total_output_tokens"] / od["elapsed"]
            print(f"\n  Ours (detailed breakdown):")
            print_ours_breakdown(od)
        else:
            o_tps = None
            print(f"\n  Ours: FAILED")

        if v_tps and o_tps:
            ratio = o_tps / v_tps
            print(f"\n  Ratio (Ours / vLLM) : {ratio:.3f}x")

    # Summary table across batch sizes
    if len(all_results) > 1:
        print(f"\n\n{'=' * 70}")
        print(f"  SUMMARY: Scaling across batch sizes")
        print(f"{'=' * 70}")
        print(f"  {'Seqs':>6}  {'vLLM tok/s':>12}  {'Ours tok/s':>12}  {'Ratio':>8}  {'Decode ms/step':>16}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*16}")
        for r in all_results:
            seqs = r["num_seqs"]
            v = r["vllm"]
            o = r["ours"]
            v_tps = v["total_output_tokens"] / v["elapsed"] if v else float("nan")
            o_tps = o["total_output_tokens"] / o["elapsed"] if o else float("nan")
            ratio = o_tps / v_tps if v and o else float("nan")
            dc_steps = o.get("decode_steps", 0) if o else 0
            dc_time = o.get("decode_time", 0) if o else 0
            ms_step = dc_time / dc_steps * 1000 if dc_steps > 0 else float("nan")
            print(f"  {seqs:>6}  {v_tps:>12,.1f}  {o_tps:>12,.1f}  {ratio:>8.3f}  {ms_step:>16.3f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Raw results written to: {args.output_json}")


if __name__ == "__main__":
    main()
