#!/usr/bin/env python3
"""Jamba MoE kernel A/B: our Triton grouped GEMM vs the kernel vLLM picks.

vLLM 0.26 logs ``Using FlashInfer CUTLASS Unquantized MoE backend`` /
``FlashInferExperts`` for AI21-Jamba-Mini-1.7 on B200 -- not the trtllm-gen
monolithic path it uses for Mixtral, because Jamba routes softmax-then-topk
with NO renormalization (``RoutingMethodType.Default``), which
``TrtLlmBf16ExpertsMonolithic._supports_routing_method`` does not accept. Our
``JambaMoE`` instead runs the Triton ``_fused_moe_kernel``, so this measures
what that choice costs at the token counts the four bench rows actually hit:
1 and 32 for the latency rows, ~200-256 for decode at concurrency, and 16384
for a full chunked-prefill step.

CAUTION: a MoE microbenchmark has already misled this project once -- the
Mixtral trtllm/Triton crossover could not be reproduced outside the engine
because the engine replays the call inside a CUDA graph. Treat a win here as a
reason to wire the kernel up and re-measure end-to-end, not as a result.

Usage:
    python tests/debug/bench_jamba_moe.py
    python tests/debug/bench_jamba_moe.py --tokens 1 --tokens 16384 --iters 50
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Jamba-Mini-1.7 MoE shape (config.json): 16 experts, top-2, no renormalize.
NUM_EXPERTS = 16
TOP_K = 2
HIDDEN = 4096
INTERMEDIATE = 14336


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, action="append", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    token_counts = args.tokens or [1, 32, 64, 256, 1024, 16384]

    torch.manual_seed(0)
    dev = "cuda"

    from fastkernels.tasks.baseline.L1.topk_softmax import TopKSoftmax
    from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts

    w13 = torch.empty(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=dev
    ).normal_(std=0.02)
    w2 = torch.empty(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=dev
    ).normal_(std=0.02)
    router = torch.empty(
        NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=dev
    ).normal_()

    topk_softmax = TopKSoftmax()
    fused = FusedExperts()

    try:
        from flashinfer.fused_moe import cutlass_fused_moe

        has_fi = True
    except Exception as exc:  # pragma: no cover
        print(f"  flashinfer cutlass_fused_moe unavailable: {exc}")
        has_fi = False

    print(f"  E={NUM_EXPERTS} top_k={TOP_K} H={HIDDEN} I={INTERMEDIATE} bf16")
    print(f"  {'TOKENS':>7} {'TRITON us':>12} {'FI-CUTLASS us':>15} {'SPEEDUP':>9} "
          f"{'MAX ABS DIFF':>13}")

    for m in token_counts:
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        logits = x @ router.t()
        weights, ids = topk_softmax(logits, TOP_K, renormalize=False)
        weights = weights.to(x.dtype)

        def run_triton():
            return fused(x, w13, w2, weights, ids, NUM_EXPERTS)

        triton_us = _time(run_triton, args.warmup, args.iters)
        ref_out = run_triton()

        fi_us = float("nan")
        diff = float("nan")
        if has_fi:
            ids_i32 = ids.to(torch.int32)
            scales = weights.to(torch.float32)
            out_buf = torch.empty_like(x)

            def run_fi():
                return cutlass_fused_moe(
                    input=x,
                    token_selected_experts=ids_i32,
                    token_final_scales=scales,
                    fc1_expert_weights=w13,
                    fc2_expert_weights=w2,
                    output_dtype=torch.bfloat16,
                    quant_scales=[],
                    output=out_buf,
                )

            try:
                run_fi()
                fi_us = _time(run_fi, args.warmup, args.iters)
                diff = (out_buf.float() - ref_out.float()).abs().max().item()
            except Exception as exc:
                print(f"  {m:>7} FI failed: {type(exc).__name__}: {exc}")
                continue

        spd = triton_us / fi_us if fi_us == fi_us and fi_us else float("nan")
        print(f"  {m:>7} {triton_us:>12.1f} {fi_us:>15.1f} {spd:>8.2f}x "
              f"{diff:>13.4g}")


if __name__ == "__main__":
    main()
