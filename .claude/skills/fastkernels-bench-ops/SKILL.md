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

No regressions: after each change, confirm the op's other users (`fastkernels list --map`) still pass.
