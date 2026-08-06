---
name: fastkernels-check-parity
description: Adversarially check a newly added/modified fastkernels model against its SOTA reference library, hunting for interface/implementation mismatches and shortcuts; fix and iterate to zero differences. Invoke with /fastkernels-check-parity <model/arch> [reference lib].
disable-model-invocation: false
---

# Adversarially check fastkernels vs the SOTA reference

The implementing agent is often lazy — assume shortcuts exist and hunt them.
References: `/home/yak/reference_code/{vllm,sglang}/`.

## 0. No stubs — the real reference library must be installed and used (hard gate)

Parity against a faked library is worthless. Before comparing or benchmarking anything, prove
the reference kernels/libraries actually import and execute for real:

- For every external lib the model/reference uses (deep_gemm, flashinfer, flash_mla,
  `vllm._custom_ops` / vLLM compiled ops, causal-conv1d, DeepGEMM, cutlass wrappers, …): import
  it for real and assert it is NOT a stub — it must have a real `__file__`, the specific ops you
  rely on must exist as attributes, and a one-shot call on real tensors must run and return
  finite/correct output. A bare `types.ModuleType` or an object whose every attribute resolves is
  a stub.
- **Never** make the code (or your harness) paper over a missing library: no
  `sys.modules["deep_gemm"] = types.ModuleType(...)`, no monkeypatched op impls, no accepting a
  silently-active `_HAS_X = False` / `except ImportError:` fallback. Grep the model's files AND
  your own test harness for these signals (`types.ModuleType`, `sys.modules[...] =`,
  `_HAS_*= *False`, `except (ImportError|ModuleNotFoundError)`, `monkeypatch`, `.impl(` overrides).
  If any op you're comparing is stubbed, faked, or silently falls back to a different path, the
  check is an **automatic FAIL** — results obtained that way are invalid.
- If a required lib genuinely won't import, FIX the real import — do not stub around it. Common
  causes: an env/loader issue (e.g. `deep_gemm_cpp` needs `libc10.so`, which importing `torch`
  first provides) or a reference-lib API/version change (e.g. vLLM dropped `vllm._C` for the
  stable-ABI `vllm._custom_ops`). Resolve it so the genuine kernel runs, then continue.

## 1. Enumerate — what to compare AND what the reference does that we might not

Enumerate the model's fastkernels tasks (`fastkernels list --map`, plus the files it added/edited
under `tasks/baseline/L*/`) and the engine areas it touched (batching, chunked prefill, kv-cache,
compilation, weight loading).

Separately, read the reference's **config class** and the reference **model / decoder-layer /
attention `__init__` and `forward`** end-to-end and list, verbatim, (a) every `getattr(config, …)`
/ config field it consumes and (b) every construction or control-flow branch (per-layer gating,
conditionals, dtype switches, MTP/nextn handling, sliding-window/full-attn patterns, etc.). This
list is the coverage contract — hand the FULL list to the subagents; do not drop fields you think
are irrelevant.

## 2. Fan out — per-op faithfulness

Spawn ONE subagent per task / engine-area **in parallel**, each given the fastkernels file(s) and
the exact reference file(s)/functions to compare against. Each subagent diffs the two line-by-line
and reports concrete mismatches against this checklist:
- Reimplemented from memory/scratch instead of calling the SAME library/kernel as the reference.
- An operator simplified or dropped that the user never asked to change.
- Same math via a different library or language (e.g. a Triton rewrite of the reference's CUDA kernel).
- A correct-but-slower kernel (different algorithm/tiling than the reference).
- A different dtype anywhere (especially fp8/mxfp4 → bf16 round-trips); mismatched scales/quant layout.
- Missing `torch.compile` support.
- Batching / chunked-prefill / kv-cache incomplete or not matching the reference.
- One kernel for both prefill and decode where the reference uses two separate ones.
- `__init__`/`forward` signature or argument semantics drifting from the reference.

## 3. Coverage pass — catch MISSING features, not just divergent ops (mandatory)

Per-op diffs (§2) find divergent implementations of code present on both sides. They do NOT find
features the reference has and fastkernels lacks **entirely** — those have no fastkernels line to
diff against, so a file-vs-file comparison silently reads as "parity." IndexShare is the canonical
miss: vLLM's DSA skips/reuses the top-k index on most layers (`index_topk_freq` /
`index_topk_pattern` / `index_skip_topk_offset`, `deepseek_v2.py` `_skip_topk`), while fastkernels
recomputed it every layer — a whole config-driven control-flow feature that was absent, not wrong.

Dedicate a subagent whose ONLY job is: *"what does the reference do that fastkernels does not do at
all?"* Working from the §1 coverage contract, it must:
- Verify every reference config field is parsed AND actually consumed with the same effect in
  fastkernels. A field the reference reads that fastkernels never parses (or parses but ignores)
  is a finding — especially per-layer/behavioral gates (skip/share patterns, MTP/nextn layers,
  layer-type patterns, per-layer dtype/window, expert-parallel/redundant experts, scaling guards).
- Walk the reference model/decoder/attention construction and confirm each conditional branch has
  a fastkernels counterpart; flag any branch with no equivalent.
- Report missing features as first-class findings (severity = correctness if they change numerics,
  else perf), even though nothing in the per-op diff pointed at them.

## 4. Fix + iterate
Collect findings from §0/§2/§3; fix each by aligning to the reference (not by patching over the
symptom, and never by stubbing a library). Re-run the affected subagents AND the §0 no-stub gate
AND the §3 coverage pass. Repeat until no differences remain. Gate every fix so shared ops/engine
paths stay bit-identical for their other users (check `fastkernels list --map`).
