#!/usr/bin/env python3
"""Head-to-head timing of the Qwen-VL vision encoder: fastkernels vs vLLM.

The vision encoder is 45-50% of multimodal prefill time in every bench_vllm
scenario (e.g. Qwen2-VL video: 62.5s of the 125.6s spent in mixed prefill+decode
steps), so it is the remaining lever on the image/video speedups. Both engines
run it eagerly with varlen FlashAttention, so any difference is implementation
efficiency rather than backend choice.

Random weights: only shapes and the module graph matter for timing. Batches are
built to mirror what the scheduler actually admits -- N images filling the
encoder token budget, or a 32-frame clip.

Usage:
    python tests/debug/profile_vision_encoder_vs_vllm.py \
        --model Qwen/Qwen2-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _peak(fn):
    """Peak GPU memory the forward allocates above its own baseline.

    This is the number the KV cache is sized around: fastkernels reserves it as
    multimodal runtime headroom, and vLLM reports its equivalent as
    "peak activation" in the gpu_worker startup log (1.51 GiB for Qwen3-VL-8B).
    """
    fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def _make_inputs(vcfg, grids, dtype, device):
    """pixel_values + grid_thw for a list of (t, h, w) grids."""
    patch = vcfg.patch_size
    tps = getattr(vcfg, "temporal_patch_size", 2)
    in_ch = vcfg.in_channels
    feat = in_ch * tps * patch * patch
    n = sum(t * h * w for (t, h, w) in grids)
    pv = torch.randn(n, feat, dtype=dtype, device=device)
    thw = torch.tensor(grids, dtype=torch.long)
    return pv, thw


def _build_ours(model_name, vcfg, dtype, device, is_qwen3):
    if is_qwen3:
        from fastkernels.tasks.baseline.L4.qwen3_vl import (
            Qwen3VLVisionConfig, Qwen3VisionTransformer,
        )
        cfg = Qwen3VLVisionConfig(
            depth=vcfg.depth, hidden_size=vcfg.hidden_size,
            in_channels=vcfg.in_channels,
            intermediate_size=vcfg.intermediate_size,
            num_heads=vcfg.num_heads, out_hidden_size=vcfg.out_hidden_size,
            patch_size=vcfg.patch_size,
            spatial_merge_size=vcfg.spatial_merge_size,
            temporal_patch_size=vcfg.temporal_patch_size,
            num_position_embeddings=vcfg.num_position_embeddings,
            deepstack_visual_indexes=list(vcfg.deepstack_visual_indexes),
        )
        m = Qwen3VisionTransformer(cfg)
    else:
        from fastkernels.tasks.baseline.L4.qwen2_vl import (
            Qwen2VLVisionConfig, Qwen2VisionTransformer,
        )
        cfg = Qwen2VLVisionConfig(
            depth=vcfg.depth, embed_dim=vcfg.embed_dim,
            hidden_size=vcfg.hidden_size, in_channels=vcfg.in_channels,
            num_heads=vcfg.num_heads, mlp_ratio=vcfg.mlp_ratio,
            patch_size=vcfg.patch_size,
            spatial_merge_size=vcfg.spatial_merge_size,
            temporal_patch_size=vcfg.temporal_patch_size,
        )
        m = Qwen2VisionTransformer(cfg)
    return m.to(device=device, dtype=dtype).eval()


def _build_vllm(vcfg, dtype, device, is_qwen3, norm_eps):
    import vllm.envs  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment, initialize_model_parallel,
    )

    # vLLM's parallel-state and CustomOp machinery both read the ambient
    # VllmConfig, so the whole build has to happen inside that context.
    cfg_ctx = set_current_vllm_config(VllmConfig())
    cfg_ctx.__enter__()
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method="env://",
            local_rank=0, backend="nccl",
        )
        initialize_model_parallel(1, 1)

    torch.set_default_dtype(dtype)
    if is_qwen3:
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
        m = Qwen3_VisionTransformer(vcfg, norm_eps=norm_eps)
    else:
        from vllm.model_executor.models.qwen2_vl import Qwen2VisionTransformer
        m = Qwen2VisionTransformer(vcfg, norm_eps=norm_eps)
    torch.set_default_dtype(torch.float32)
    return m.to(device=device, dtype=dtype).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vcfg = hf.vision_config
    is_qwen3 = "qwen3" in args.model.lower()
    norm_eps = getattr(getattr(hf, "text_config", hf), "rms_norm_eps", 1e-6)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    merge = vcfg.spatial_merge_size
    # Scheduler-realistic batches. VisionArena images average ~2912 patches
    # (728 merged tokens); MMVU clips are 32 frames -> t=16 temporal patches.
    batches = {
        "1 image (728 merged tok)": [(1, 52, 56)],
        "22 images (~16k merged tok, full encoder budget)": [(1, 52, 56)] * 22,
        "1 clip 32f (t=16)": [(16, 26, 46)],
        "3 clips 32f (~13k merged tok)": [(16, 26, 46)] * 3,
    }

    print(f"  model={args.model} merge_size={merge} depth={vcfg.depth} "
          f"heads={vcfg.num_heads}")

    ours = _build_ours(args.model, vcfg, dtype, device, is_qwen3)
    theirs = _build_vllm(vcfg, dtype, device, is_qwen3, norm_eps)

    print(f"  {'batch':44s} {'fk time':>8s} {'vLLM':>8s} {'spd':>5s} | "
          f"{'fk peak':>10s} {'vLLM':>5s} {'ratio':>5s}")
    for label, grids in batches.items():
        pv, thw = _make_inputs(vcfg, grids, dtype, device)
        merged = int((thw.prod(-1) // (merge ** 2)).sum())
        try:
            with torch.inference_mode():
                t_ours = _bench(lambda: ours(pv, thw), args.iters)
                t_theirs = _bench(lambda: theirs(pv, thw), args.iters)
                m_ours = _peak(lambda: ours(pv, thw))
                m_theirs = _peak(lambda: theirs(pv, thw))
        except Exception as exc:
            print(f"  {label:50s} FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"  {label:44s} {t_ours*1e3:8.1f}ms {t_theirs*1e3:8.1f}ms "
              f"{t_theirs/t_ours:5.2f}x | peak {m_ours/2**30:5.2f}G "
              f"{m_theirs/2**30:5.2f}G {m_ours/max(1,m_theirs):5.2f}x")
        del pv, thw
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
