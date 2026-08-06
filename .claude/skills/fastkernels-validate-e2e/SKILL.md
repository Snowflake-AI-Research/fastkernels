---
name: fastkernels-validate-e2e
description: Validate a fastkernels model end-to-end against its SOTA reference with tests/bench_vllm.py — token alignment >= 100 and speedup >= 1.0x, eager and compiled, small then full workload, TP if multi-GPU. Invoke with /fastkernels-validate-e2e <HF model id>.
disable-model-invocation: false
---

# End-to-end validation

Harness: `tests/bench_vllm.py` (extend it, or add `tests/bench_<lib>.py` if the reference is not
vLLM — see `.claude/skills/adding-arch-instructions.md`). Always run with
`VLLM_ENABLE_V1_MULTIPROCESSING=0 TORCH_CUDA_ARCH_LIST=10.0`.

Targets (regardless of dtype): **avg matching tokens ≥ 100** per request AND **speedup ≥ 1.0x**
vs the reference (both shown in the run's summary table).

## Ladder — advance only when the current rung passes
1. **Small + eager** — fastest signal: `--max-layers 12` (default for big/slow models), a low
   `--num-seqs`, `--enforce-eager`.
2. **Small + compiled** — same, without `--enforce-eager`.
3. **Full model** — drop `--max-layers` (only if it fits the GPUs), keep low `--num-seqs`, both eager and compiled.
4. **Full workload** — drop `--num-seqs` (all requests).
5. **Tensor parallel** — if the machine has >1 GPU, add `--tp <N>` and repeat 1–4.

(Add `--trust-remote-code` when the reference worker needs it.)

## Fix loop
- Alignment < 100 → correctness bug: use `/fastkernels-check-parity` and `/fastkernels-bench-ops` to localize the
  diverging op/layer (compare per-layer outputs vs the reference), fix, re-run.
- **Low token-match is almost always a real op bug — do NOT dismiss it as "truncated proxy /
  near-tie / hypersensitivity."** On MoE+sparse models one flipped expert/index, or a
  magnitude-scaled 1-ULP fp8 error, makes greedy-match look like irreducible noise, but each traces
  to a concrete op. Don't burn time on staging deeper proxies, depth sweeps, tp variations, or the
  reference's own batch-nondeterminism — they don't localize op bugs. Check divergence-relevant env
  flags up front (e.g. `VLLM_BATCH_INVARIANT`).
- Speedup < 1.0x → profile and optimize the hot op (see `/fastkernels-bench-ops`).
Iterate each rung until both targets are met before moving on.

## Debugging a divergence (do this FIRST — it pinpoints the op in minutes)
- **Clean-room injection, not tolerance.** Capture the reference op's EXACT input tensors, feed
  them into the fk op, require **max|Δ| = 0** (bit-identical). Bisect per-layer/per-op to the FIRST
  op that diverges on identical input; everything after is just propagation. When injection points 
  at an op, diff its full numeric pipeline against the reference side-by-side - dtypes, exact
  kernel fn, weight/scale prep, and kernel args (not just the math). The mismatch can often be a kernel
  variant or weight-prep method, e.g. routed-vs-monolithic MoE, direct-dequant-vs-eye-GEMM.
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

No regressions: any shared-code change must leave other models' bench numbers unchanged.
