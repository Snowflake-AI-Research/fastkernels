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
- Speedup < 1.0x → profile and optimize the hot op (see `/fastkernels-bench-ops`).
Iterate each rung until both targets are met before moving on.

No regressions: any shared-code change must leave other models' bench numbers unchanged.
