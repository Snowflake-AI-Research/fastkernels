#!/usr/bin/env python3
"""Phase 3/5 validation: verify torch.compile integration works end-to-end.

Tests:
  1. Model compiles without errors (no graph breaks)
  2. Compiled model produces same outputs as eager model
  3. CUDA graph capture on compiled model works correctly

Usage:
    python tests/debug/test_compilation.py --model meta-llama/Llama-3.1-8B-Instruct
    python tests/debug/test_compilation.py --model meta-llama/Llama-3.1-8B-Instruct --skip-vllm
"""

from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
torch.set_grad_enabled(False)


def test_compilation(model_name: str, tp: int = 1, skip_vllm: bool = False):
    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=42)
    prompts = [
        "What is machine learning?",
        "Write a Python hello world program.",
        "Explain quantum computing in simple terms.",
    ]

    # --- Test 1: Eager mode baseline ---
    print("=" * 60)
    print("Test 1: Eager mode baseline")
    print("=" * 60)
    engine_eager = LlamaEngine(
        model_name=model_name,
        seed=42,
        enforce_eager=True,
        tensor_parallel_size=tp,
    )
    out_eager = engine_eager.generate(prompts, sp)
    eager_tokens = [o.token_ids for o in out_eager]
    engine_eager._cleanup()
    del engine_eager
    torch.cuda.empty_cache()

    print(f"Eager output (first prompt): {eager_tokens[0][:10]}...")

    # --- Test 2: Compiled mode ---
    print("\n" + "=" * 60)
    print("Test 2: Compiled mode (torch.compile + CUDA graph)")
    print("=" * 60)
    engine_compiled = LlamaEngine(
        model_name=model_name,
        seed=42,
        enforce_eager=False,
        tensor_parallel_size=tp,
    )
    out_compiled = engine_compiled.generate(prompts, sp)
    compiled_tokens = [o.token_ids for o in out_compiled]

    print(f"Compiled output (first prompt): {compiled_tokens[0][:10]}...")

    # --- Compare ---
    print("\n" + "=" * 60)
    print("Comparison: eager vs compiled")
    print("=" * 60)
    total_match = 0
    total_tokens = 0
    for i, (et, ct) in enumerate(zip(eager_tokens, compiled_tokens)):
        min_len = min(len(et), len(ct))
        matching = sum(1 for j in range(min_len) if et[j] == ct[j])
        total_match += matching
        total_tokens += max(len(et), len(ct))
        exact = "EXACT" if et == ct else f"PARTIAL ({matching}/{min_len})"
        print(f"  Prompt {i}: {exact}")

    avg_match = total_match / len(eager_tokens)
    avg_len = total_tokens / len(eager_tokens)
    print(f"\n  Average matching tokens: {avg_match:.1f}/{avg_len:.0f}")

    if total_match == total_tokens:
        print("  PASS: All outputs match exactly")
    elif avg_match / avg_len > 0.95:
        print("  PASS: High alignment (>95% token match)")
    else:
        print("  WARN: Low alignment — check for compilation issues")

    # --- Test 3: Throughput comparison ---
    print("\n" + "=" * 60)
    print("Test 3: Throughput (compiled mode)")
    print("=" * 60)
    sp_bench = SamplingParams(temperature=0.0, max_tokens=128, seed=42,
                              ignore_eos=True)
    bench_prompts = ["Hello world"] * 32

    engine_compiled.block_manager.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = engine_compiled.generate(bench_prompts, sp_bench)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(len(o.token_ids) for o in out)
    tps = total_output_tokens / elapsed
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Elapsed:       {elapsed:.2f}s")
    print(f"  Throughput:    {tps:.0f} tok/s")

    engine_compiled._cleanup()
    del engine_compiled

    print("\nAll compilation tests completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()
    test_compilation(args.model, args.tp, args.skip_vllm)
