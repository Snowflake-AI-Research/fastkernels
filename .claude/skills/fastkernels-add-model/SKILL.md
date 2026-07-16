---
name: fastkernels-add-model
description: Implement a new model architecture in fastkernels from a SOTA reference library (vLLM or SGLang), matching its interface, kernels, libraries, and dtypes exactly — no shortcuts. Invoke with /fastkernels-add-model <HF model id or arch> [reference lib].
disable-model-invocation: true
---

# Add a model architecture to fastkernels

Follow the hierarchy + parity rules in `.claude/skills/adding-arch-instructions.md`.
References: `/home/yak/reference_code/vllm/`, `/home/yak/reference_code/sglang/`.

Inputs: the exact HF model id (e.g. `zai-org/GLM-4.6`) and which reference library is SOTA
for it. If either is missing, ask before starting.

## 1. Familiarize — get the RIGHT variant (do not skip)
- Read the model's HF `config.json`: use its exact `model_type` / `architectures` to find the
  reference class. Do NOT model it on a similarly-named one — variants differ radically (e.g.
  `DeepSeek-V3.2-Exp` uses DSA/MLA and must not be based on `DeepSeek-V3`).
- In the reference lib, locate and read: the model file, its attention/MoE/norm layers, the
  attention **backend** selected on this GPU, every external lib (flashinfer, deep_gemm,
  causal-conv1d, …) and custom/compiled kernel it calls, the kv-cache layout, and the **dtype
  at each step** (especially fp8/mxfp4 paths). Note whether prefill and decode use different kernels.
- Record this as the contract to match: op list, exact `forward` signatures, libraries, dtypes.

## 2. Implement — parity, no shortcuts
- Bottom-up per the hierarchy: L1 primitives (single kernels — call the SAME library/kernel the
  reference does), L2 composites (only L1 ops; no `torch.nn`/`F`/external libs), L3 decoder layer,
  L4 wiring only. Condense into existing shared classes (e.g. the unified `Attention`) unless that
  forces the `forward` interface to diverge from the reference.
- Match exactly: same libraries, same custom/compiled kernels, same dtypes (never
  fp8→bf16→fp8 round-trips as a shortcut), separate prefill vs decode paths where the reference
  has them, same kv-cache and weight layouts/loader.
- Wire the engine: register in `workloads.py` (`FASTKERNELS_ARCHITECTURES`) and
  `infra/kernel_swapper.py` (`_L4_MODEL_KEYS`); add a workload/dataset if needed. In
  `infra/engine.py` add the model-family detection + dispatch (kv-cache alloc, `prepare_prefill`/
  `prepare_decode`/`prepare_mixed_batch`, weight loading). If the architecture is radically
  different (novel cache/state), add a new engine instead of contorting the existing one.
- Support BOTH eager and compiled (`torch.compile`) modes, and full continuous batching /
  chunked prefill / paged kv-cache — matching the reference.

## 3. No regressions
Ops and engine branches are shared. Before reusing or editing one, find its other users
(`fastkernels list --map`, grep) and gate changes on the new `model_type` so other models stay
bit-identical.

## Done when
The model builds and runs a forward in eager and compiled mode. Then verify with
`/fastkernels-check-parity`, `/fastkernels-bench-ops`, and `/fastkernels-validate-e2e`.
