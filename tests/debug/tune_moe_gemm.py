#!/usr/bin/env python3
"""Tune Triton MoE GEMM configs for Mixtral TP=4 decode batch sizes."""

import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastkernels.tasks.baseline.L1.moe_grouped_gemm import _fused_moe_kernel
from fastkernels.tasks.baseline.L1.moe_align import MoeAlign
import triton
import triton.language as tl


def bench_config(M, N, K, E, top_k, config, num_iters=200):
    """Benchmark a single config."""
    align = MoeAlign().cuda()
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M * top_k, N, device="cuda", dtype=torch.bfloat16)
    topk_weights = torch.randn(M, top_k, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, E, (M, top_k), device="cuda", dtype=torch.int32)

    sorted_ids, expert_ids, num_tokens_pp = align(topk_ids, config["BLOCK_SIZE_M"], E)
    EM = sorted_ids.size(0)

    grid = (
        triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )

    launch_kw = {}
    if "num_warps" in config:
        launch_kw["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        launch_kw["num_stages"] = config["num_stages"]

    # Warmup
    for _ in range(10):
        _fused_moe_kernel[grid](
            A, B, C, topk_weights, sorted_ids, expert_ids, num_tokens_pp,
            N, K, EM, M * top_k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            MUL_ROUTED_WEIGHT=False, top_k=top_k,
            compute_type=tl.bfloat16,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            **launch_kw,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _fused_moe_kernel[grid](
            A, B, C, topk_weights, sorted_ids, expert_ids, num_tokens_pp,
            N, K, EM, M * top_k,
            A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            MUL_ROUTED_WEIGHT=False, top_k=top_k,
            compute_type=tl.bfloat16,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            **launch_kw,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / num_iters
    return elapsed * 1e6  # us


def main():
    E = 8
    top_k = 2

    configs = [
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
    ]

    for M in [32, 64, 128, 256]:
        for gemm_name, N, K in [("w13", 7168, 4096), ("w2", 4096, 3584)]:
            print(f"\n=== M={M}, {gemm_name}: ({M}x{K}) @ ({E}x{N}x{K}) ===")
            best_time = float("inf")
            best_cfg = None
            for cfg in configs:
                try:
                    t = bench_config(M, N, K, E, top_k, cfg)
                    tag = f"  M={cfg['BLOCK_SIZE_M']:3d} N={cfg['BLOCK_SIZE_N']:3d} K={cfg['BLOCK_SIZE_K']:3d} G={cfg['GROUP_SIZE_M']:2d} w={cfg['num_warps']} s={cfg['num_stages']}"
                    print(f"{tag}: {t:8.1f} us {'*' if t < best_time else ''}")
                    if t < best_time:
                        best_time = t
                        best_cfg = cfg
                except Exception as e:
                    pass
            print(f"  BEST: {best_time:.1f} us -> {best_cfg}")


if __name__ == "__main__":
    main()
