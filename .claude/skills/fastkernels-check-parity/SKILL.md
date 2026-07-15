---
name: fastkernels-check-parity
description: Adversarially check a newly added/modified fastkernels model against its SOTA reference library, hunting for interface/implementation mismatches and shortcuts; fix and iterate to zero differences. Invoke with /fastkernels-check-parity <model/arch> [reference lib].
disable-model-invocation: true
---

# Adversarially check fastkernels vs the SOTA reference

The implementing agent is often lazy — assume shortcuts exist and hunt them.
References: `/home/yak/reference_code/{vllm,sglang}/`.

## Fan out
Enumerate the model's fastkernels tasks (`fastkernels list --map`, plus the files it added/edited
under `tasks/baseline/L*/`) and the engine areas it touched (batching, chunked prefill, kv-cache,
compilation, weight loading). Spawn ONE subagent per task / engine-area **in parallel**, each given
the fastkernels file(s) and the exact reference file(s)/functions to compare against.

Each subagent diffs the two line-by-line and reports concrete mismatches against this checklist:
- Reimplemented from memory/scratch instead of calling the SAME library/kernel as the reference.
- An operator simplified or dropped that the user never asked to change.
- Same math via a different library or language (e.g. a Triton rewrite of the reference's CUDA kernel).
- A correct-but-slower kernel (different algorithm/tiling than the reference).
- A different dtype anywhere (especially fp8/mxfp4 → bf16 round-trips); mismatched scales/quant layout.
- Missing `torch.compile` support.
- Batching / chunked-prefill / kv-cache incomplete or not matching the reference.
- One kernel for both prefill and decode where the reference uses two separate ones.
- `__init__`/`forward` signature or argument semantics drifting from the reference.

## Fix + iterate
Collect findings; fix each by aligning to the reference (not by patching over the symptom).
Re-run the affected subagents. Repeat until no differences remain. Gate every fix so shared
ops/engine paths stay bit-identical for their other users (check `fastkernels list --map`).
