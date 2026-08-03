---
name: fastkernels-bench-ops
description: Benchmark each new/modified fastkernels operator against its SOTA-reference counterpart for correctness (same init/inputs -> same output) and performance (>= reference); fix and iterate. Invoke with /fastkernels-bench-ops <model/arch> [reference lib].
disable-model-invocation: false
---

# Benchmark fastkernels ops vs the SOTA reference

For each operator the model added or modified (L1–L3). References: `/home/yak/reference_code/{vllm,sglang}/`.

1. **Harness**: instantiate the fastkernels task AND the reference library's counterpart with the
   SAME init args and SAME weights, feed the SAME inputs (use real shapes/dtypes — `fastkernels
   capture` records them), and compare outputs + wall-clock (warm up, median over N iters). Keep
   dtypes exactly as the reference — no bf16 shortcut.
2. **Pass bar**: outputs match within the dtype's tolerance, AND fastkernels is ≥ the reference
   (on par or faster).
3. **On a mismatch or perf gap**:
   - First diff our op against the reference (interface, library, dtype, algorithm) — usually the
     cause. Fix to match.
   - If the implementations look identical but we're slower, profile with `ncu` / nsight and
     optimize the kernel (occupancy, tiling, memory). Do not change the math or dtype to go faster.
4. Iterate until every op passes correctness + performance.

## Debugging a divergence (do this FIRST — it pinpoints the op in minutes)
- **Clean-room injection, not tolerance.** Capture the reference op's EXACT input tensors, feed
  them into the fk op, require **max|Δ| = 0** (bit-identical). Bisect per-layer/per-op to the FIRST
  op that diverges on identical input; everything after is just propagation.
- **Capture cleanly:** single-sequence, ONE hook at a time. Many hooks / kernel-arg wrappers force
  mid-forward `.cpu()` syncs that perturb async paths (e.g. a separate-stream shared expert) and
  fabricate divergences; batched runs add reference nondeterminism. A 1-ULP error scales with
  activation magnitude — it can read 0.0 at low-magnitude layers, so judge at high-magnitude
  layers / clean-room.
- **fp8 prime suspects (each was a real GLM-5.2 bug):** (1) a fk custom kernel vs the reference's
  exact op — RoPE: `fastkernels_rope` ≠ `torch.ops._C.rotary_embedding`; (2) weight dequant/prep
  method — direct block-dequant ≠ vLLM's `use_deep_gemm` fp8-GEMM-on-identity for MLA absorbed
  W_UK/W_UV; (3) fused-kernel variant — routed MoE kernel truncates combine weights to bf16 ≠
  monolithic `trtllm_fp8_block_scale_moe` in-kernel routing. Match the reference's EXACT kernel
  AND weight-prep AND invocation, not just "same math."

No regressions: after each change, confirm the op's other users (`fastkernels list --map`) still pass.
