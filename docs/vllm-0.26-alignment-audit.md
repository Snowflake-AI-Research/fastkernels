# vLLM 0.26 alignment audit

Audit of `fastkernels/infra/{engine,compilation,weight_loader}.py` and the
`tasks/baseline` operators against vLLM **v0.26.0** (upgrade from v0.18),
using `reference_code/vllm` @ `v0.26.0` and the installed `vllm==0.26.0`.

Hardware this was audited and validated on: **8x NVIDIA B200 (SM100,
compute capability 10.0)**, torch 2.11.0+cu130, flashinfer 0.6.14.

---

## 1. Attention backend selection

### What vLLM 0.26 does on this host

Read from the reference run logs of `20260728-075104` (vLLM's own
`cuda.py:482` / `flash_attn.py:776` / `selector.py:202` lines):

| model | vLLM backend | KV layout | notes |
|---|---|---|---|
| Llama-3.1-8B, Mixtral, Qwen2-VL, Qwen3-VL, gpt-oss-120b, Qwen3-Next | `FLASHINFER` | HND | candidates `['FLASHINFER','FLASH_ATTN','TRITON_ATTN','FLEX_ATTENTION']` |
| Whisper (decoder self/cross) | `FLASH_ATTN`, **FlashAttention version 4** | NHD | |
| Whisper (encoder / ViT) | `FLASHINFER` | HND | |
| Gemma-4-26B | `TRITON_ATTN` | — | candidates only `['TRITON_ATTN','FLEX_ATTENTION']` |
| Kimi-Linear | `FLASHINFER_MLA` (+ `FLASH_ATTN` MLA prefill) | HND | crashes, see §5 |

`vllm/v1/attention/backends/fa_utils.py::get_flash_attn_version()` on SM100
prefers **FA4** (CuTeDSL), falling back to FA2. FA3 is Hopper-only:
`is_fa_version_supported(3)` returns
`"FA3 is only supported on devices with compute capability 9.x"`.

### What fastkernels did, and the bug this hid

Every L1 flash-attn wrapper gated its fast path on

```python
if is_fa_version_supported(3) and cc[0] >= 9:   # False on SM100
    ...use vllm.vllm_flash_attn...
else:
    from flash_attn import ...                  # PyPI FA2 2.8.3
```

On B200 that condition is `False and True` -> **False**, so all paged
attention fell through to the PyPI `flash_attn` package, whose paged-KV path
hard-requires `page_block_size % 256 == 0`:

```
RuntimeError: Paged KV cache block size must be divisible by 256
```

This is the single root cause of the Whisper, Qwen3-Next and EAGLE-3
failures. vLLM's own bundled build advertises `MultipleOf(16)`
(`FlashAttentionBackend.get_supported_kernel_block_sizes`), which is what the
engine's block size (16) needs.

### Which kernel, not just which backend name

"FLASHINFER" is a *backend*, and it dispatches to several different kernels.
Resolved from source and confirmed by running vLLM's own selector
(`CudaPlatform.get_valid_backends`) at `DeviceCapability(10, 0)`:

**Prefill** (`flashinfer.py:1157`): `prefill_use_trtllm = causal and
use_trtllm_attention(..., is_prefill=True, force_use_trtllm=None)`, whose
auto-detection is `use_trtllm = kv_cache_dtype == "auto"` — true for every
bf16 model here, so vLLM uses `trtllm_batch_context_with_kv_cache`.

**Decode**: chosen **statically at builder init**, not per batch —
`use_trtllm_decode_attention = can_use_trtllm_attention(num_qo_heads,
num_kv_heads, is_prefill=False)`, because "the decode kernel must be selected
statically for FULL cudagraph capture". The `num_tokens <= 256` heuristic in
`use_trtllm_attention`'s decode branch is **never reached** by this backend
(its only call site passes `is_prefill=True`), so a large decode batch does
*not* switch vLLM off trtllm-gen. vLLM uses
`trtllm_batch_decode_with_kv_cache`.

That matches fastkernels' `TRTLLMPrefill` / `TRTLLMDecode`, which call the same
two FlashInfer entry points unconditionally.

Selector output, reproduced (matches the candidate lists in the reference logs
exactly):

| attention config | vLLM picks | why the faster backends were excluded |
|---|---|---|
| head_size 128/256, DECODER | `FLASHINFER` | — |
| ENCODER_ONLY (Whisper encoder) | `FLASH_ATTN` | FlashInfer: *attention type ENCODER_ONLY not supported* |
| ENCODER_DECODER (Whisper cross) | `FLASH_ATTN` | FlashInfer: *attention type ENCODER_DECODER not supported* |
| Gemma4, `use_mm_prefix=True` (both 256 and 512) | `TRITON_ATTN` | FlashInfer: *partial multimodal token full attention not supported*; FlashAttention: *mm_prefix (PrefixLM bidirectional attention) requires FlashAttention v4, which does not resolve for this head_size* |
| Gemma4 with `use_mm_prefix=False` | `FLASHINFER` | — |

Two consequences, both now implemented:

- **Whisper** splits across two backends, and fastkernels now matches both:
  decoder self-attention on trtllm-gen/HND, encoder self-attention and
  cross-attention on FlashAttention/NHD. The cross-attention NHD pin in §2 is
  therefore the vLLM-aligned choice, not just a crash fix.
- **Gemma-4** uses `TRITON_ATTN` for *every* layer, because it is a PrefixLM
  with bidirectional image attention — the exclusion is about `mm_prefix`, not
  head size. Routing only the 512-wide layers to Triton (and leaving the
  256-wide sliding layers on trtllm-gen) would have compared a different
  kernel than vLLM runs, so `Gemma4Attention` now passes
  `prefer_triton=True` and all its layers use the Triton unified kernel on an
  NHD cache.

---

### Fix (FA version selection)

New `tasks/baseline/L1/fa_utils.py` mirrors vLLM's version selection and
exports vLLM's `flash_attn_varlen_func`. It delegates to vLLM's own
`get_flash_attn_version()` when importable so the two cannot drift.
All call sites now route through it:

- `flash_attn_prefill.py`, `flash_attn_decode.py`, `flash_attn_varlen.py`
- `flashinfer_prefill.py` (dense fallback)
- `tree_attn_prefill.py` (EAGLE-3 verify cascade)

fastkernels no longer calls the PyPI `flash_attn` package for paged
attention on any supported device.

---

## 2. Per-layer backend selection and KV cache layout

vLLM 0.26 picks a backend **per KV-cache group**, not once per model. It also
refactored the cache layout in this release
(#44455 "Pack K/V into the content dim", #42095 "num-blocks first").

fastkernels chose one global backend from `AttnBackendConfig.auto_detect()`
and allocated every cache in that one layout, while several modules
unconditionally used FlashAttention ops. Three latent layout mismatches
existed; all were masked by the §1 crash happening first.

| module | wrote | read with | result |
|---|---|---|---|
| Gemma-4 512-wide full-attn layers | HND | SDPA fallback (`_cache_seq`, NHD) + `.item()` | `CUDA error: operation not permitted when stream is capturing` |
| Whisper cross-attention | HND | FlashAttention (NHD) | `num_head must be divisible by num_head_kv` (FA read `block_size` as the KV head count) |
| Qwen3-Next MHA layers | NHD | FlashAttention (NHD) — consistent, but head_dim=256 | `SM100 forward with head_dim=256 does not support seqused_q/seqused_k` |

### Fix

`kv_layout` is now a **per-layer property** that the engine reads back when
sizing the cache, mirroring vLLM's per-group choice:

- `attention_impl.Attention`: layers with `head_size > 256` (the largest the
  paged trtllm-gen/FlashAttention kernels advertise) drop to the Triton
  unified kernel on an NHD cache — exactly the `TRITON_ATTN` choice vLLM logs
  for Gemma-4. `_can_use_triton_unified` now gates on `kv_layout == "NHD"`
  (the kernel's real requirement) instead of `not self._use_trtllm`.
- `whisper_attention.WhisperCrossAttention`: pins `kv_layout = "NHD"` and the
  NHD store, because it reads through FlashAttention.
- `qwen3_next_attention.Qwen3NextAttention`: now follows the same
  device-based selection as `Attention`, so on Blackwell its head_dim=256 MHA
  layers use trtllm-gen/FlashInfer on an HND cache — matching vLLM's
  `FLASHINFER` + HND. FlashAttention is not a substitute: FA4's SM100
  head_dim=256 forward rejects `seqused_k`.
- `engine.allocate_kv_cache` / `_allocate_variable_kv_cache` and
  `mamba_state` allocate from each layer's declared layout; the startup line
  now reports e.g. `kv_layout=HND/NHD` for mixed models.
- `_share_trtllm_workspace` also reaches layers that keep their KV cache in
  the engine's state manager (Qwen3-Next, Kimi-Linear) and so carry no
  `k_cache` attribute; without this each of Qwen3-Next's 12 MHA layers kept
  its own 512 MiB workspace.

---

## 3. TP shared-memory decode payload

`ModelRunner._write_decode_shm` wrote the decode batch into a **hardcoded
1 MiB** `SharedMemory` segment whose control fields sat at fixed `2**20`
offsets. The block-table field alone needs
`max_num_seqs * ceil(max_model_len / block_size) * 4` bytes — 8 MiB for
Mixtral at `max_num_seqs=1024`, `max_model_len=32800`, `block_size=16`.

Past the end of the buffer, `buf[off:off+nb] = arr.tobytes()` silently
truncated the *lvalue* slice, surfacing as:

```
ValueError: memoryview assignment: lvalue and rvalue have different structures
```

This is the root cause of the Mixtral, gpt-oss-120b and Qwen3-VL-235B
failures — all TP>1; every TP=1 model passed.

### Fix

`_init_shm_layout()` sizes the segment from the final scheduler/cache config
(covering the attention, hybrid and mamba decode payloads, MRoPE's 3x
positions and MLA's int64 slot mapping), keeps a 1 MiB floor for `call()`'s
pickled payloads, page-aligns the total, and makes the control offsets
instance attributes. Every rank derives the same layout from the same config,
so rank 0's `create=True` size and the workers' offsets agree without an
extra handshake. `_write_decode_shm` now bounds-checks and reports the actual
numbers; `_init_shm_layout` rejects configs whose counts overflow the 2-byte
header fields at startup rather than mid-run.

---

## 4. Compilation stack

Compared against the `compilation_config` vLLM 0.26 logs at startup.

**Aligned already:** `CompilationMode.VLLM_COMPILE` equivalent (single Dynamo
trace, guards dropped, FX split at attention custom ops, one `compile_fx` per
unique subgraph with symbolic shapes, `autograd_cache_key` dedup);
`cudagraph_mode=FULL_AND_PIECEWISE` (mode-matched `CUDAGraphWrapper`);
`cudagraph_num_of_warmups=1`; `custom_ops=['none']` /
`ir_op_priority(rms_norm=['native'])` via the dual `forward_native` /
`forward_cuda` dispatch; MoE deliberately **not** a splitting op.

**Divergence fixed — `inductor_compile_config`.** fastkernels passed none of
the settings vLLM passes to `compile_fx`. `vllm_aligned_inductor_config()`
now reproduces them exactly:

| key | value | why vLLM sets it |
|---|---|---|
| `enable_auto_functionalized_v2` | `False` | custom post-grad fusion passes are written against V1 |
| `size_asserts`, `alignment_asserts`, `scalar_asserts` | `False` (torch < 2.12) | vLLM measured **~2 ms per forward pass** of `assert_size_stride`/`assert_alignment` calls on large models |
| `combo_kernels`, `benchmark_combo_kernel` | `True` (torch >= 2.9) | horizontal fusion, specifically to fuse qk-norm and qk-rope where q and k differ in shape |

**Divergence fixed — missing splitting op.** `kimi_delta_attention.py` called
`torch.ops.fastkernels.kda_attention`, which was **never registered** — dead
code behind `_use_custom_op` (KDA is absent from `context._TARGET_NAMES`) that
would `AttributeError` the moment it went live. The op is now registered and
added to `SPLITTING_OPS`, matching vLLM's `vllm::kda_attention`.

**Remaining known divergence (not changed).** vLLM 0.26 split the KV-cache
write out of attention into its own splitting op
(`vllm::unified_kv_cache_update`, returning a dummy tensor purely to give
torch.compile an ordering dependency) and moved attention to the
output-argument form `unified_attention_with_output`. fastkernels keeps the
store inside `forward_impl` (so it is opaque to Inductor either way) and
returns the output tensor. The compiled-graph shape is equivalent; adopting
`with_output` would mainly enable vLLM's `fuse_attn_quant` pass. This is an
invasive refactor of every L2 attention module and is left as follow-up.

---

## 5. Scheduler / batching

The `generate()` loop matches vLLM 0.26's `Scheduler.schedule()` shape:
`token_budget = max_num_batched_tokens`; running/decode scheduled first (one
token each), then in-flight chunked prefills, then admission from the waiting
queue; preemption on block exhaustion; watermark reserve.

`long_prefill_token_threshold` is **not** implemented — and does not need to
be: vLLM 0.26 defaults it to `0` (disabled) and only auto-derives
`int(max_model_len * 0.04)` when `max_num_partial_prefills > 1`, whose default
is `1` (`vllm/config/scheduler.py`). Encoder-decoder models disable chunked
prefill entirely in both.

### Reference-side vLLM bug: Kimi-Linear

Not a fastkernels bug, and **not worked around**. On Blackwell vLLM selects
`FLASHINFER_MLA` with `block_size=64`, whose trtllm-gen decode kernel requires
`block_num % (128 // block_size) == 0`. vLLM aligns the per-request
block-table width to that multiple (`v1/worker/block_table.py:275`, upstream
#39324) but **not** the block count of the KV cache used by
`profile_cudagraph_memory()` -> `_warmup_and_capture(..., force_attention=True)`
-> `_dummy_run`, so an odd count aborts start-up:

```
ValueError: Expected block_num % (128 / block_size) == 0,
            got block_num=525 and block_size=64
```

Reproduced directly against vLLM (no fastkernels involved) with a plain
`LLM(model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
tensor_parallel_size=2, trust_remote_code=True)`:

| `gpu_memory_utilization` | `max_model_len` | `block_num` | result |
|---|---|---|---|
| default | **default (none passed)** | 16395 (odd) | **crash** |
| 0.9 | harness default | 2055 (odd) | crash |
| 0.9 | 32768 | 525 (odd) | crash |
| 0.85 | 32768 | 525 (odd) | crash |
| 0.9 | 16384 | even | **runs** — generated `' Paris. The capital of Italy is Rome'` |

`block_num` is identical at `gpu_memory_utilization` 0.9 and 0.85, so it is
**not** a function of free memory — it is deterministic in `max_model_len`
(and `block_size`). vLLM therefore fails or starts *reproducibly* for a given
config: `max_model_len=16384` works, `32768` does not.

Notably the **default** config fails: plain
`LLM(model="moonshotai/Kimi-Linear-48B-A3B-Instruct", tensor_parallel_size=2,
trust_remote_code=True)` — i.e. the equivalent of `vllm serve
moonshotai/Kimi-Linear-48B-A3B-Instruct` on this host — aborts at
`block_num=16395`. So this is a real upstream defect at the model's own
default context length, not an artifact of the harness's settings.

No page size avoids the check: `FLASHINFER_MLA` advertises only `[32, 64]` and
`CUTLASS_MLA` only `[128]`, and both route decode through the same trtllm-gen
shape validation. `num_gpu_blocks_override` does not help either, because the
crash happens inside `determine_available_memory` — before that override is
applied.

**Decision: no harness workaround.** Pinning the reference to `TRITON_MLA`
(the one candidate without the assertion) would run vLLM on a slower
third-choice MLA decode kernel, so any reported fastkernels speedup would be
measured against a handicapped baseline instead of like-for-like. The
reference is left to fail and the upstream bug reported. The one-line upstream
fix is to round the profiling block count up to `128 // block_size`, exactly as
`block_table.py` already does for the block-table width.

The only workaround that preserves backend/page-size/kernel parity is to nudge
the reference's `max_model_len` to a value whose block count is even. That is
a live option, but it makes the two engines' context envelopes differ, so it
is left as a decision for the benchmark owner rather than applied silently.

---

## 5.5 Page size parity

Verified against the installed vLLM rather than inferred:

```python
EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct").create_engine_config()
#   cache_config.block_size == 16
```

| | vLLM 0.26 | fastkernels |
|---|---|---|
| dense / MoE attention | 16 | 16 (`AttnBackendConfig.auto_detect` -> trtllm, `block_size=16`) |
| MLA (Kimi-Linear, DeepSeek) | 64 | 64 (`_MLA_BLOCK_SIZE`) |

Note that `FlashInferBackend.get_supported_kernel_block_sizes()` *advertises*
`[16, 32, 64, 128, 256, 512, 1024]` on Blackwell GQA (the trtllm-gen dynamic
kernel), but vLLM still resolves `cache_config.block_size` to **16** by
default, so 16 is the aligned value.

The one place fastkernels still carries a non-vLLM page size is the
`flash_attn` branch of `AttnBackendConfig` (`block_size=256`), which is only
reachable on pre-Blackwell GPUs. That 256 is a leftover from routing paged
attention through the PyPI `flash_attn` package; vLLM's own
`FlashAttentionBackend` advertises `MultipleOf(16)`. It is documented as
tech-debt in `context.py` rather than changed, because it alters block-manager
and watermark behaviour on hardware not available on this host.

---

## 6. Weight loader

No v0.18 -> v0.26 API breakage found. `load_weights` reads HF safetensors
directly and drives each parameter's own `weight_loader`; it does not depend
on `vllm.model_executor.model_loader`. The `fastsafetensors` iterator still
mirrors vLLM's `fastsafetensors_weights_iterator` (one file per rank per
round, GDS disabled for TP>1 to avoid `cuFileDriverOpen` creating a CUDA
context on every visible GPU). The only vLLM import in the load path is
`kimi_linear`'s `set_current_vllm_config(VllmConfig())`, which is still valid
in 0.26.

---

## 7. Validation

`--skip-vllm --skip-latency` runs on B200, previously-failing scenarios:

| model | tp | before | after |
|---|---|---|---|
| Whisper-large-v3 | 1 | FA2 paged block-size 256 | **PASS** 7,666 tok/s |
| Gemma-4-26B-A4B | 1 | cudagraph capture abort | **PASS** 10,190 tok/s |
| Mixtral-8x7B | 4 | SHM memoryview | **PASS** 17,758 tok/s |
| gpt-oss-120b | 2 | SHM memoryview | **PASS** 10,290 tok/s |
| Qwen3-VL-235B-A22B-FP8 | 4 | SHM memoryview (MRoPE 3x) | **PASS** 4,244 tok/s |
| Qwen3-Next-80B-A3B | 2 | FA2 paged, then FA4 hd256 `seqused_k` | **PASS** 11,510 tok/s |

`tests/test_validate.py` + `tests/test_bench.py`: 17 passed.

**Not fixed — EAGLE-3** (reference: SGLang, not vLLM). Its attention blocker
is resolved: the engine now pins flash_attn/NHD, because the tree-verify
cascade merges softmax LSEs and trtllm-gen returns none, and it captures all
56 CUDA graphs (40 target-verify + 8 draft-chain + 8 draft-extend) — which it
had never reached before, having previously bailed out on the unavailable-FA3
check. A separate, pre-existing bug in `eagle3_engine.generate` then surfaces:
some sequences are admitted but neither finalized nor kept active, tripping
`assert all(o is not None for o in all_outputs)`.

**Not fixed — Kimi-Linear** reference: upstream vLLM bug, see §5.

---

## 8. Findings from the first full sweep (20260729-070206)

Running against the real vLLM reference (rather than `--skip-vllm`) surfaced
three things the earlier smoke runs could not.

### 8a. The §1-§4 fixes hold up against the reference

| model | tp | fastkernels | vLLM | ratio | match toks |
|---|---|---|---|---|---|
| Llama-3.1-8B | 1 | 21,350 | 22,481 | 0.95x | 151.2 / 390 |
| Mixtral-8x7B | 2 | 14,721 | 15,595 | 0.94x | 96.8 / 447 |
| gpt-oss-120b | 2 | 9,520 | 24,576 | 0.39x | 0.8 / 385 |
| Gemma-4-26B-A4B | 1 | 9,067 | 10,637 | 0.85x | 72.4 / 401 |

No regression: Llama was 21,244 @ 0.95x / 146.8 match on 20260728-075104 and
is 21,350 @ 0.95x / 151.2 match now.

### 8b. `disable_mm_preprocessor_cache` removed in vLLM 0.26

Five reference-side failures (Whisper, Qwen2-VL, Qwen3-VL-8B,
Qwen3-VL-235B-FP8, Qwen2.5-Omni), all one cause:

```
TypeError: EngineArgs.__init__() got an unexpected keyword argument
           'disable_mm_preprocessor_cache'
```

Introduced by `7bf8787` (after the 07-28 run, which is why those models
passed then) and a genuine v0.18 -> v0.26 API removal missed in §1-§6.
vLLM 0.26 replaces it with `MultiModalConfig.mm_processor_cache_gb`
(default 4 GiB), documented as *"Set to `0` to disable this cache completely"* —
which preserves the original intent of making the timed region pay for
multi-modal preprocessing on both sides. Both worker templates now pass
`mm_processor_cache_gb=0`.

### 8c. Attention sinks and sliding window silently dropped on trtllm-gen

`gpt-oss-120b` matched **0.8 of 385** tokens against vLLM — a numerical bug,
not a crash. Root cause: `TRTLLMPrefill.forward` / `TRTLLMDecode.forward`
accepted `**kwargs` and never forwarded it. `Attention._forward_pure` and
`_forward_mixed` pass sinks and the window under FlashAttention's names
(`s_aux`, `window_size`), while the trtllm-gen kernels call them `sinks` and
`window_left`. Both therefore fell into `**kwargs` and vanished, so on
Blackwell gpt-oss ran with **no sink bias and no sliding window** — every
layer full-causal. vLLM passes both explicitly
(`flashinfer.py:2036-2037` prefill, `2225-2226` decode).

Only gpt-oss is affected: it is the sole model with sinks, and the only other
`sliding_window` user (Gemma-4) now routes through the Triton unified kernel,
which forwards `window_size` correctly. The `flash_attn_*` wrappers were never
affected — they pass `**kwargs` straight to `flash_attn_varlen_func`, which
accepts `s_aux` and `window_size` natively.

Verified after the fix that each argument now changes the output (before, all
variants were bit-identical):

| path | sinks maxdiff | window maxdiff |
|---|---|---|
| `TRTLLMDecode` | 0.0020 | 0.3357 |
| `TRTLLMPrefill` | 2.1445 | 0.6116 |

Part of gpt-oss's 0.39x throughput is likely the same bug: with the window
dropped, its alternating sliding layers attended the full context instead of
128 tokens, i.e. strictly more work than vLLM did. Re-measurement needed.

### 8d. Still open

- **Throughput below parity** for Llama (0.95x), Mixtral (0.94x) and Gemma-4
  (0.85x). The project guideline is on-par-or-better, so these need work.
- **Match tokens below the >=100 guideline** for Mixtral (96.8) and Gemma-4
  (72.4). Separate from the crashes above and not yet diagnosed.
- **EAGLE-3** (index 4): the `all_outputs` assert in §7.
- **Kimi-Linear** (index 12): upstream vLLM bug in §5, `block_num=2055` again.

---

## 9. Second sweep round (20260729-070206 + --resume)

43/46 pass after the §8 fixes. The `mm_processor_cache_gb` change cleared all
five multimodal reference failures (Whisper, Qwen2-VL, Qwen3-VL-8B,
Qwen3-VL-235B-FP8, Qwen2.5-Omni), which now report real comparisons:

| model | text | image | video | match toks |
|---|---|---|---|---|
| Whisper-large-v3 | 0.94x | | | 400.2 / 444 |
| Qwen2-VL-7B | 0.98x | 0.51x | 0.52x | 204-394 |
| Qwen3-VL-8B | 0.97x | 0.61x | **3.13x** | 85-149 |
| Qwen3-VL-235B-FP8 | 0.98x | 0.96x | **3.03x** | 49.8 (text) |
| Qwen2.5-Omni-7B | 0.98x | 1.10x | 1.18x (audio 1.87x) | 155-238 |

### 9a. EAGLE-3: two separate defects

**Scheduler drain loop.** The bare `assert all(o is not None ...)` was replaced
with a diagnostic, which immediately localised the fault:

```
RuntimeError: EAGLE-3 scheduler stalled with 8 request(s) queued and none
admissible (max_num_seqs=8, target_kv_free=1087, draft_kv_free=1087)
```

KV was 1087/1088 free, so capacity was never the issue. `bench_sglang` runs a
**`max_tokens=1`** pass to pre-absorb Triton JIT before timing. With
`max_tokens=1`, `_target_prefill` appends the first sampled token, so
`len(generated_ids) == 1 >= max_tokens` and *every* request finishes during
prefill: `live_for_extend` is empty, `active` never becomes non-empty, and the
old `while active:` loop exited with half the queue unprocessed, leaving those
outputs `None`.

Checked against SGLang rather than assumed: `Req.update_finish_state`
(`srt/managers/schedule_batch.py:1486`) finishes on
`len(self.output_ids) >= sampling_params.max_new_tokens`, so finishing at
prefill for `max_new_tokens=1` is correct SGLang behaviour — the defect was
purely our loop. Our EOS-beats-length truncation also matches SGLang's
`finished_len` handling. Fixed with `while idx_pool or active:`, treating "the
admitted batch all finished at prefill" as forward progress.

**Missing reference (silent pass).** With the loop fixed the job exited 0 but
printed `SGLANG N/A / SPEEDUP N/A`: the harness looks for SGLang in a dedicated
`sglang-bench` conda env (torch-version isolation), and on a miss printed
`WARNING: sglang subprocess failed -- continuing with fastkernels only` and
carried on. The runner then recorded PASS for a scenario that never compared
against its SOTA baseline. `bench_sglang` now exits non-zero in that case, and
`--skip-sglang` remains the explicit opt-out. The env was created via
`tests/setup_sglang_env.sh` (sglang 0.5.16, torch 2.11.0+cu130).

### 9b. Kimi-Linear: upstream bug, fixed without touching the backend

Two wrong diagnoses preceded the right one, both worth recording because the
error message is misleading.

`block_num` in

```
ValueError: Expected block_num % (128 / block_size) == 0,
            got block_num=135 and block_size=64
```

is **not** a KV-cache block count. flashinfer reads it as
`block_num = page_table.shape[-1]` with `block_size = page_size`
(`flashinfer/mla/_core.py:686-696`) -- i.e. the **block-table width**, validated
against the *kernel* page size. Attempts to align the cache block count
(`num_gpu_blocks_override` search, then flooring `may_override_num_blocks`) were
therefore aimed at the wrong quantity; a patch-install marker confirmed the code
was live and the failure unchanged, which is what forced the re-read.

vLLM does align that width -- against the wrong quantity:

```python
max_num_blocks = [cdiv(n, 128 // bs) * (128 // bs) if bs <= 128 else n
                  for n, bs in zip(max_num_blocks, block_sizes)]
```

(`v1/worker/block_table.py`, upstream #39324). `block_sizes` is the **spec**
block size. For Kimi-Linear the hybrid allocator pads attention up to the mamba
page -- *"Setting attention block size to 960 tokens to ensure that attention
page size is >= mamba page size"* -- so `bs = 960 > 128`, the `else n` branch
skips alignment entirely, and `cdiv(max_len, 960)` lands on an odd 135 while the
kernel still demands `width % 2 == 0` against its 64-token page.

The fix is the one-line upstream correction: align against
`kernel_block_sizes`, which is what the kernel actually reads. Applied as a
`MultiGroupBlockTable.__init__` wrapper in the harness sitecustomize (the
failure occurs inside spawned TP worker processes). Verified in isolation:
spec bs=960 / kernel bs=64 / width 135 -> 136.

Rounding the width *up* only adds a couple of unused (-1 padded) block-table
columns, so it changes no kernel, no page size, no KV capacity and no scheduler
setting -- `FLASHINFER_MLA` is retained. Substituting `TRITON_MLA` would have
made every reported Kimi speedup a comparison against a handicapped reference.

### 9b-2. Kimi-Linear, fastkernels side: FlashMLA dense decode is SM90a-only

With the reference fixed, our own engine then failed:

```
RuntimeError: dense_attn_decode_interface,
flashmla-src/csrc/api/dense_decode.h:29,
Dense decode MLA is only supported on SM90a architecture
```

`mla_attention_impl` hardcoded `FlashMLADecode` for the BF16 dense decode path,
and FlashMLA's dense decode kernel is Hopper-only -- so on Blackwell it cannot
run at all. This is the same class of defect as section 1: a Hopper-only kernel
selected unconditionally.

vLLM selects `FLASHINFER_MLA` on this device and dispatches decode through
`trtllm_batch_decode_with_kv_cache_mla`
(`FlashInferMLAImpl.forward_mqa`). New `L1/flashinfer_mla_decode.py` wraps that
same entry point with the argument shape our decode path already produces, and
`mla_attention_impl` prefers it when
`torch.cuda.get_device_capability()[0] >= 10`, leaving the FlashMLA path
untouched on Hopper and below.

Note this shares the MLA impl with DeepSeek-V3.2, which is commented out of
`scenarios/full.yaml`, so that path is fixed but not exercised by the sweep.

### 9b-3. Kimi-Linear now runs end to end, but is numerically wrong

With both sides fixed, the scenario completes for the first time:

```
mixed  1000  var  389   fastkernels 16,175 tok/s   vLLM 13,566   1.19x   AVG MATCH TOKS 5.0/389
```

**That 1.19x is not a valid result.** 5 of 389 matching tokens means the outputs
disagree almost immediately, so the throughput number is measured on wrong
output. Comparing raw token ids:

```
fastkernels: 198, 9335, 48956, 269, ... 276, 5173, 318, 276, 5173, 318, ...  (looping)
vLLM       : 3251, 15962, 198, 1700, 4037, 16184, 1127, ...                  (varied)
```

They diverge at position 0 and ours degenerates into a repeating cycle.

**This is pre-existing, not a regression from the MLA decode change.** Token 0
is produced by prefill, and the new `FlashInferMLADecode` only replaces the BF16
*decode* branch. The op itself was validated numerically in isolation against a
torch absorbed-MLA reference:

| block-table width | `max_seq_len` | rel. error |
|---|---|---|
| 6 (even, batch-sized) | 300 (true batch max) | 0.0020 |
| 6 | 131072 (`max_model_len`) | 0.0033 |
| 2048 (`max_model_len/64`) | 300 | 0.0034 |
| 2048 | 131072 | 0.0034 |

i.e. bf16-level agreement, and neither the inflated `max_seq_len` the engine
actually passes nor the full-width block table perturbs it.

Kimi's alignment had never been measured before: the reference aborted during
startup in every earlier run (20260728-075104, the sweep, and the resume), so
the harness never reached the comparison. The fixes above made it observable and
it is failing. Root-causing that divergence -- most likely the KDA linear
attention recurrence, the prefill path, or MLA weight absorption -- is separate
work and is **not** done.

The same isolated test also showed the trtllm-gen MLA kernel enforces the
even-block-table-width rule on *our* side too (`block_num=5` was rejected).
fastkernels currently satisfies it only incidentally, because
`ceil(max_model_len / 64)` happens to be even; a `max_model_len` that lands odd
would fail the same way. Worth aligning explicitly.

### 9b-4. Kimi divergence: what has been ruled out

The token-0 divergence is pre-existing and not yet root-caused. Eliminated so
far, so this is not re-done:

| hypothesis | test | result |
|---|---|---|
| weights silently unloaded (`load_weights` has several `except AttributeError: continue` paths) | loaded-shard count vs checkpoint `weight_map` | 20493 = 20493, **complete** |
| KDA/MLA layer types shifted (config lists `full_attn_layers=[4,8,...,27]` and `kda_layers` are **1-indexed**) | compare our `is_kda_layer` to vLLM's | both use `(layer_idx + 1) in kda_layers`, **identical** |
| new `FlashInferMLADecode` op wrong | vs torch absorbed-MLA reference, 4 shape/length combos | rel 0.0020-0.0034 (bf16 level), **correct** |
| wrong KDA entry point -- vLLM 0.26 calls `chunk_kda_with_fused_gate`, fastkernels calls `chunk_kda` + a separate `fused_kda_gate` | run both on identical inputs | max abs diff **0.000000**, bit-identical -- an API difference with identical math, **not a divergence** |
| KDA `beta` / gate formula differs | line-diff vs vLLM | identical (`b_proj(...).float().sigmoid()`, `f_b_proj(f_a_proj(...))`) |
| `A_log` / `dt_bias` shapes or TP shard dims differ | compare declarations + checkpoint shapes | ours `A_log [1,1,H,1]` shard dim 2, `dt_bias [proj/tp]` shard dim 0; checkpoint is `(1,1,32,1)` / `(4096,)` -- **matches vLLM** |
| conv1d bias dropped -- we pass `None` where vLLM passes `self.q_conv1d.bias` | grep checkpoint for conv1d bias tensors | checkpoint has **weights only, no conv bias** (60 conv tensors, none `bias`), so `None` is correct |
| MLA prefill backend differs | vLLM logs `Using FLASH_ATTN MLA prefill backend`; ours uses `FlashAttnVarlen` | **matches** |
| TP sharding bug | run the same scenario at **TP=1** | 4.6/428 matching vs 5.0/389 at TP=2 -- **sharding exonerated**, the defect reproduces on one GPU |
| `routed_scaling_factor` (2.446, non-default; every layer from 1 is MoE since `first_k_dense_replace=1`) applied wrongly -- vLLM passes it *into* the routing, we pass `1.0` there and scale the output | read the ordering | ours scales the routed output *then* adds the shared expert, which is algebraically the same as scaling the routing weights -- **equivalent** |
| MoE routing config from fallback defaults (the keys are named `num_experts_per_token`, `moe_router_activation_func`, `moe_renormalize`, `num_shared_experts`, `use_grouped_topk` -- easy to miss) | dump both configs | ours: 256 experts, top_k 8, `scoring_func="sigmoid"`, `renormalize=True`, 1 shared expert, `num_expert_group=1`, `topk_group=1` -- **all match vLLM** |

Because the defect reproduces at **TP=1**, the search space is now single-GPU
and much smaller. Remaining suspects, in rough order: KDA prefill *state and
metadata* wiring (`cu_seqlens`, `has_initial_state`, `state_indices` -- the
kernel itself is proven identical, so its *inputs* are the open question), MLA
prefill numerics, expert routing selection (`grouped_topk` / `e_score_correction_bias`
/ `n_group` / `topk_group` defaults, several of which are absent from
`config.json` and therefore come from our fallbacks), and the output
gating/normalisation.

The efficient next step is a hidden-state diff rather than more end-to-end runs:
capture layer-0..3 outputs from both engines for one fixed prompt and find the
first layer that disagrees. `--max-layers 4` is the smallest viable config
(layer 0 is KDA + dense MLP, layer 3 is the first MLA layer).

Note `--max-layers 1` is not a usable bisect point for this model: layer 0 is
KDA, so zero MLA layers exist and the engine raises
`ZeroDivisionError`. The first MLA layer is index 3 (`full_attn_layers[0] == 4`,
1-indexed), so `--max-layers 4` is the smallest viable configuration.

### 9c. Correction to a §8 claim

The Qwen3-Next (index 11) OOM was attributed in §8 to GPU contention from a
local verification run. That was wrong: both the original and the resume run
report a foreign process holding *exactly* 824 MiB (different PIDs, same size),
i.e. the sweep co-scheduling a second process onto GPU 0 while Qwen3-Next tp=2
requests ~177 GiB. It is a genuine, reproducible scheduling/capacity failure of
the sweep, not test contamination.

### 9d. Still open

* Qwen3-Next (index 11): the co-scheduling OOM above.
* rtdetr_v2 (index 32): `overall_pass=False` (labels 0.734, box MAE 70 px) while
  the harness still exits 0 — `bench_detection` computes the verdict but never
  propagates it. The comparison is also purely positional (flatten and compare
  index-by-index, no detection matching), which is consistent with a permuted
  set of near-tied detections; needs testing against the saved tensors before
  deciding whether it is a model defect or a measurement artifact.
* Throughput below parity broadly (0.84-0.98x), and match tokens under the
  >=100 guideline for Mixtral (96.8), Gemma-4 (72.4) and Qwen3-VL-235B text
  (49.8).
* Embedding indices 41/42 need re-running: they were measured before the
  `torch._dynamo.disable` fix, so their numbers still include the FA4 re-JIT
  overhead (bge-m3 was 63.5 s/batch vs vLLM's 21 s for 1000 requests).

## 10. Result-reporting standardization

The sweep was unanalysable in aggregate: rows recorded a speedup in nine
different shapes, some recorded none, and one recorded a PASS with no result
file at all. This section covers the reporting fixes, which are independent of
the kernel-alignment work above.

### 10a. The silent failure

`002_bench_microsoft_bitnet` in 20260729-070206 is recorded
`status=PASS returncode=0` with **no `results.json` in its run directory**. Two
independent defects combined:

1. `bench_microsoft_bitnet.main` did `kb_data = run_worker(...)` then
   `if kb_data:` — a dead engine returned `None`, the comparison was skipped,
   and the process exited 0.
2. `ray_runner` derived status from the return code alone, so a harness that
   exits 0 without writing anything was indistinguishable from a real pass.

Fixes: the harness now raises `SystemExit` when the worker returns `None`
(matching `bench_image_cls`/`bench_detection`), and `ray_runner` downgrades a
PASS to `FAIL(no-results)` when the expected artifact is absent.

Verified against the recorded sweep: of the six rows lacking `results.json`,
four (`004` sglang-llama, `011` Qwen3-Next, `012` Kimi-Linear, plus `002`) were
genuine failures and two (`037` openpi, `043` dllm) wrote a differently-named
artifact — now fixed by 10c below. So the new gate flags exactly the rows that
should be flagged.

The underlying bitnet crash was `AttributeError: rope_theta` from
`BitNetConfig` no longer exposing that attribute directly in current
transformers; `L4/bitnet.py` now resolves it from `rope_parameters` /
`rope_scaling` / `to_dict()` with the class default as the last resort
(`rope_theta=500000.0` for `bitnet-b1.58-2B-4T`).

### 10b. `fastkernels/validate/comparison.py`

One shape for every harness:

    scenarios:         [{scenario, fastkernels_<metric>, reference_<metric>,
                         speedup = ours/ref, alignment?}]
    latency_scenarios: [{scenario, fastkernels_<metric>, reference_<metric>,
                         speedup = ref/ours}]

`speedup` is always oriented so >1 means fastkernels is faster.
`alignment_from_token_ids` reproduces the prefix-match block the generative
harnesses already emit; `alignment_from_similarity` covers cosine / match-rate
metrics and takes `higher_is_better=False` for error metrics (MAE, NLL
difference), where `score` becomes `1 - value` and the threshold is an upper
bound. Without that flag a TTT NLL *difference* of 0.02 would have scored 0.02
instead of 0.978.

`ray_runner._throughput_rows_for_result` / `_latency_rows_for_result` gained a
generic fallback that reads this shape when no harness-specific branch produced
rows, so newly standardized harnesses need no per-harness branch.

### 10c. Per-harness gaps closed

| Harness | Before | After |
| --- | --- | --- |
| bench_image_cls (convnextv2, efficientnetv2) | worker died on a bad `pkg_root`, `ours: None`, exit 0 | pkg_root guard, hard-fail, `ethz/food101`; 1.001x tp, top1 1.0, batch-1/8 latency |
| bench_ttt_e2e | no speedup, no alignment | 3.66x / 1.04x tp + latency vs JAX, NLL-diff alignment |
| bench_microsoft_bitnet | no results file at all | tp + per-request latency vs microsoft-bitnet-gpu |
| bench_openpi | only `summary.json`, keyed `aloha.*` | standard `results.json`; 3.59x/2.99x tp, 3.57x/4.22x latency, cosine 0.999998 |
| bench_dllm (LLaDA) | only a scenario-named 4.8 MB json | standard `results.json` (6.8 KB, points at the detailed file); 1.10x tp, 1.10x latency, 165.8 match tokens |
| bench_oasis | 5 tp speedups, no latency speedup | `latency_ratio_p50` promoted to a latency scenario (1.151x); raw workload list renamed to `workloads` |
| bench_pointcloud | `comparison.throughput_ratio` only | all three declared workloads: scanobjectnn 0.997x, batch-8 0.995x, single-cloud latency 0.994x |
| bench_3dgs | `comparison.throughput_ratio` only | render 0.995x + single-render latency 0.996x (both sides were already timed; nothing compared them) |
| bench_instantngp | `comparison.throughput_ratio` only | render 1.009x + single-render latency 1.004x |

whisper-large-v3 needed no change: `bench_vllm` already emits standard
throughput and latency entries for all three ASR workloads. See 10d for its
actual problem.

### 10d. Whisper decode is ~24 ms/step (new finding)

whisper-large-v3 latency is 0.10x vs vLLM at bs=1 (10.67 s vs 1.07 s for 448
tokens) and 0.23x at bs=32 (10.72 s vs 2.51 s). The wall clock is *independent
of batch size*, and the step profile shows 443 fast-path decode steps in
10.6 s = 23.9 ms/step for both batch sizes.

So this is a fixed per-step host cost, not a per-token compute cost: the
sequence does enter the cudagraph fast path (`fast_decode=443`, `scheduled=1`),
but each replay costs ~24 ms where a 1.5B decoder should cost ~1-2 ms.
Cross-attention is the obvious suspect (whisper is the only encoder-decoder row,
and `_set_cross_attn_context_decode` is the only whisper-specific per-step
work). Not yet root-caused.

### 10e. BitNet sweep cost

`--num-prompts` defaults to 1000 and `--kb-bsz` to 1 — the latter is required,
because the official BitNet GPU int2 decode kernels only dispatch for `M == 1`,
so the reference cannot batch and fastkernels matches it to keep the comparison
like-for-like. At 1000 serialized prompts x 512-1024 output tokens x 3
scenarios x 2 engines that is several hours for this one row. Left as-is (it is
the honest like-for-like configuration), but worth a `--num-prompts` override in
the sweep config if wall clock matters.

### 10f. BitNet's *second* silent failure: the reference was never provisioned

Fixing the `rope_theta` crash exposed a second one behind it. With the
fastkernels side finally running, the harness printed

    [skip-sota] Microsoft BitNet GPU artifacts missing:
      ['/home/yak/vllm_repo/BitNet/gpu/bitnet_kernels/libbitnet.so', ...]

then wrote a `results.json` with `scenarios: []`, `latency_scenarios: []` and
exited 0. So even with the crash fixed, this row could still pass while
recording no speedup and no alignment. Three defects:

1. `--bitnet-repo` defaulted to `/home/yak/vllm_repo/BitNet`, a hardcoded
   personal path that does not exist on this host, while every other reference
   lives under `THIRD_PARTY_DIR`. Now defaults to `THIRD_PARTY_DIR/"BitNet"`.
2. There was no `provision.py` component for it, so the reference was
   documentation-only — unlike the nine other references.
3. A missing reference warned and continued. It now raises `SystemExit` with the
   provisioning command; `--skip-sota` remains for deliberately running
   fastkernels alone.

The new `bitnet` component clones microsoft/BitNet, builds the kernel, installs
xformers, downloads the bf16 checkpoint and runs both conversion steps. Two
deliberate deviations from upstream's README:

* **Kernel arch.** `bitnet_kernels/compile.sh` hardcodes
  `-gencode=arch=compute_80,code=compute_80` — PTX only, so on Blackwell the
  driver JITs sm_80 PTX and the reference is not measured with native codegen.
  Provisioning builds for the local CC instead (`compute_100/sm_100` here),
  which is what `_prov_instant_ngp` already does via `TCNN_CUDA_ARCHITECTURES`.
  It compiles clean for sm_100 with no source changes.
* **xformers.** The reference's `gpu/model.py` imports `RMSNorm`, `fmha` and
  `rope_padded` from xformers, so both the conversion script and the SOTA worker
  need it. xformers 0.0.35 requires only `torch>=2.10`, so it installs against
  the env's torch 2.11.0+cu130 without perturbing it (verified with
  `pip install --dry-run`: "Would install xformers-0.0.35", nothing else).

Incidental confirmation of the §10a fix: `convert_checkpoint.py` prints the
official model config, including `rope_theta: 500000.0` — the same value
`L4/bitnet.py` now resolves from `rope_parameters`/class default.

The intermediate `model_state.pt` (4.8 GB) is deleted after conversion, per
upstream; the retained int2 + fp16 splits are 1.8 GB + 5.5 GB.

### 10g. The missing kernel also silently downgraded the *fastkernels* side

`libbitnet.so` gates both engines, not just the reference. The KB worker does

    if cfg.get("bitnet_kernel_so"):
        os.environ.setdefault("KB_BITNET_KERNEL_LIB", cfg["bitnet_kernel_so"])

and `main` passes `kernel_so if os.path.isfile(kernel_so) else ""`, with the
summary label chosen the same way: `"official ladder decode"` if the .so exists,
`"Triton fallback"` otherwise. So before provisioning, this row would have
benchmarked *our Triton fallback* against nothing at all, and reported it as a
pass.

Concretely, in the pre-provisioning run the fastkernels side measured 96.9 tok/s
on `prefill-heavy` (8 prompts, 31 ms/step) while the provisioned reference does
the same scenario at 1158 tok/s (2.6 ms/step). Those two numbers are **not** a
valid speedup: they come from different runs, and the first one was the fallback
path. The real comparison requires both sides with the .so present.

A useful cross-check while investigating: at bs=1 in 20260729-070206,
Llama-3.1-8B is 3.56 ms/tok (1.053x), Gemma-4-26B 3.81 (1.067x), Qwen2.5-Omni
3.51 (0.996x) — i.e. there is *no* general small-batch decode overhead in the
engine. The two real outliers at bs=1 are whisper-large-v3 (24.51 ms/tok, 0.087x,
see 10d) and gpt-oss-120b (6.25 vs 2.48, 0.397x), plus Mixtral-8x7B (5.11 vs
3.62, 0.707x). Those are model-specific, not a shared cause.

### 10h. BitNet's throughput comparison was eager-vs-cudagraph

With both sides holding `libbitnet.so`, the comparison as first recorded was:

| scenario | reference (tok/s) | fastkernels (tok/s) | ratio |
| --- | --- | --- | --- |
| prefill-heavy | 1158.1 (10.61 s) | 103.0 (119.31 s) | 0.089x |
| balanced | 784.3 (10.45 s) | 68.9 (118.97 s) | 0.088x |

Not like-for-like. The reference times its CUDA-graph path
(`generate_all_fixed(..., use_cuda_graphs=True)`), while the fastkernels side was
built with `enforce_eager = not args.use_kb_cudagraph` -- and
`--use-kb-cudagraph` was opt-in, documented as "for debugging". Per decode step
(output tokens only, 8 sequences serially at `kb_bsz=1`): ours 29.1 ms,
reference 2.59 ms.

Two observations pointing at launch overhead rather than the ternary GEMM:

* Enabling the official ladder kernel barely moved our number: 96.9 tok/s
  (Triton fallback, pre-provisioning) -> 103.0 tok/s, i.e. 6%.
* The four `m == 1` shapes bitnet-2B needs -- (3840, 2560), (2560, 2560),
  (13824, 2560), (2560, 6912) -- are all in `_official_shape_supported`, so the
  kernel *is* eligible at decode.

Control run with graphs enabled on our side confirmed it:

| scenario | reference | ours (cudagraph) | speedup | ours ms/step | ref ms/step |
| --- | --- | --- | --- | --- | --- |
| prefill-heavy | 1158.1 | 1356.4 | 1.171x | 2.21 | 2.59 |
| balanced | 784.3 | 1019.7 | 1.300x | 1.96 | 2.55 |
| decode-heavy | 647.1 | 774.7 | 1.197x | 1.94 | 2.32 |

So the eager default turned a genuine 1.17-1.30x win into an apparent 11x loss.
Resolved by the restructure in 10k.

### 10i. Eager and CUDA-graph decode disagree on bitnet-b1.58-2B-4T

Comparing the two fastkernels paths directly, identical prompts, greedy,
`ignore_eos`, 8 prompts per scenario:

| scenario | identical seqs | avg common prefix | earliest divergence |
| --- | --- | --- | --- |
| prefill-heavy | 2/8 | 190.6 | token 7 |
| balanced | 1/8 | 121.2 | token 4 |
| decode-heavy | 1/8 | 185.2 | token 4 |

They do **not** agree. Divergence at token 4 under greedy decoding is a
behavioural difference, not accumulated fp noise (though near-tied logits on a
ternary-weight model make early flips cheap). **Open bug.** The harness now runs
non-eager by default, so this is the path being measured; `--enforce-eager`
selects the other one for bisection, as does `KB_BITNET_FORCE_BF16=1` for the
quantization question specifically.

Separately, alignment against the reference is weak on its own terms: avg
matching prefix 11.75 / 144.1 / 208.1 tokens with 0-1 of 8 exact matches on the
old prefill-heavy / balanced / decode-heavy regimes. prefill-heavy being by far
the worst (11.75 of 512) pointed at the long left-padded prompt path rather than
at decode -- the two 512-input scenarios did 12-18x better. Not yet
investigated; the regimes have since changed (10j), so this needs re-measuring.

### 10j. Workloads, dtype and W2A8: what this row actually runs

`full.yaml` declares
`workloads: [LLM.mixed, LLM.long_context, LLM.single_request, LLM.fixed_batch_32]`
and `dtype: bfloat16`. Before the restructure, **none of it reached the
harness**: `_build_cmd` passed only `--model --tp --output-dir`, there is no
`--dtype` argument at all, and the harness ran three hardcoded regimes
(prefill-heavy 1024/512, balanced 512/512, decode-heavy 512/1024) drawn solely
from `DEFAULT_WORKLOAD_DATASETS["mixed"]`. So `longbench-longctx` was never
loaded, the two latency probes had no counterpart, and the run summary's
`workloads` column advertised coverage that never happened.

On dtype, the answer is that the row was already correct, by coincidence rather
than by plumbing. Both sides run W2A8 in the same hybrid split:

| phase | reference (`gpu/generate.py`) | fastkernels (`bitnet_linear.py:211-223`) |
| --- | --- | --- |
| prefill | `ModelArgs(use_kernel=False)` + `model_state_fp16.pt` -> `BitLinear`, fake-quant bf16 | `_bitnet_use_bf16_path()` -> `F.linear(_fake_quant_act_bf16(x), bf16_weight)` |
| decode (M==1) | `ModelArgs(use_kernel=True)` + `model_state_int2.pt` -> `BitLinearKernel` | `bitnet_int8xint2_linear_official()` -- the same `bitlinear_int8xint2` symbol from the same `libbitnet.so` |

`_bitnet_use_bf16_path` reads the engine's prefill flag, falling back to an
M>=1024 heuristic outside an engine context. `dtype: bfloat16` correctly
describes the surrounding compute and matches the reference's
`torch.set_default_dtype(torch.bfloat16)`; the `model_state_fp16.pt` filename is
misleading (it holds the *unpacked* weights, loaded under a bf16 default). So the
alignment gap in 10i is not a dtype mismatch.

### 10k. Restructured into throughput + latency phases

* **One run per phase.** The separate eager alignment run is gone; throughput and
  correctness come from the same run in the same execution mode.
* **Non-eager by default**, `--enforce-eager` to opt out, matching
  `bench_vllm.py`. `--kb-timing` and `--use-kb-cudagraph` removed.
* **Workloads come from the scenario** via a new `_WORKLOADS_FLAG` in
  `validate/__init__.py`; `_resolve_workloads` splits them by declared `Purpose`
  and reads each one's dataset and counts. `bench_microsoft_bitnet` also joined
  `_EAGER_OK` so the scenario's `enforce_eager` reaches it.
* **Real latency phase**: warmup x3 then 5 timed iterations of one fixed batch,
  median/mean/p99 on both sides -- not `elapsed / num_prompts`.
* **`fixed-batch-32` has no reference.** The official int2 decode kernel
  dispatches only for `M == 1` (`bitnet_kernels.cu` gates every branch on it, and
  the kernel body is a GEMV: `A` is indexed with no row offset, the output has no
  row stride, and the activation scale is read at `s[0]`). Those rows record
  `speedup: null` with `reference_unsupported_reason` rather than comparing our
  batched run against a serial reference loop.

Two normalization decisions, both wrong on the first attempt:

1. **Median -> mean.** The reference needs one fixed shape per workload. Median
   minimizes distortion of the length *distribution* but destroys the aggregate
   prefill work: wildchat-mixed is mostly short prompts, so a median-length
   `mixed` came out at 60 input tokens against 1024 output -- a pure-decode test
   labelled "mixed". The mean preserves the trace's total prefill token count
   (171 for the sampled set).
2. **Clamp to the model context.** The mean for `long-context` is 24556 tokens
   (min 8185, p50 8190, max 130969), but bitnet-b1.58-2B-4T has
   `max_position_embeddings=4096`. The first run normalized to 8190 and would have
   asked the model to represent positions it was never trained for, also blowing
   past `--max-model-len`'s 2048 default. Input length is now clamped to
   `model_ctx - output_len`, so `long-context` runs at 3968 in / 128 out, and
   `--max-model-len` auto-raises to fit the longest shape (the sweep never passes
   it).

Resulting regimes at `--num-prompts 8`:

| workload | dataset | raw min/p50/max/mean | normalized | out |
| --- | --- | --- | --- | --- |
| mixed | wildchat-mixed-1k | 11/60/601/171 | 171 | 1024 |
| long-context | longbench-longctx | 8185/8190/130969/24556 | 3968 (clamped) | 128 |
| single-request | wildchat-mixed-1k | 572/572/572/572 | 572 | 128 |
| fixed-batch-32 | wildchat-mixed-1k | 10/35/2241/197 | 197 | 128 |

Caveat worth keeping in view: with a 4096-position model, every longbench prompt
is suffix-kept to under half its length, so `long-context` measures "as long as
this model goes", not longbench's actual context lengths.

Still open: `long-context`'s output length is a local default
(`DEFAULT_LONG_CONTEXT_OUTPUT_LEN = 128`) because `LLM.long_context` declares
`decode_cap=None`; and whether to patch the reference kernel for batching (a
~4-line grid-y change, but it would make the baseline a fork we authored rather
than upstream).

---

## 11. Third sweep round: the four failures in `20260729-193844`

`summary.json` reported 42/46 PASS. Two of the four failures had the same root
cause, and two were newly *reachable* bugs that an earlier crash had been
hiding.

| index | model | reported status | actual cause |
|---|---|---|---|
| 3 | openai/gpt-oss-120b | `FAIL(rc=1)` | sinks dtype, then an MXFP4 MoE over-read (two defects, §11a/§11b) |
| 11 | Qwen/Qwen3-Next-80B | `FAIL(rc=-6)` (OOM -> NCCL timeout) | unbounded prefill (§11c) |
| 12 | moonshotai/Kimi-Linear-48B | `FAIL(rc=-6)` (OOM -> NCCL timeout) | unbounded prefill (§11c) |
| 2 | microsoft/bitnet-b1.58-2B-4T | `FAIL(no log output for 900s)` | not in this round's scope |

§9c attributed index 11's OOM to the sweep co-scheduling a second process onto
its GPU. That was wrong, or at least incomplete: the 824 MiB foreign process is
real but irrelevant next to the actual allocation. See §11c.

### 11a. Attention sinks: the two kernels want *different* dtypes

The previous round wired sinks and the sliding window through to trtllm-gen
(§8c), which is correct, and immediately hit

```
RuntimeError: Check failed: attention_sinks.value().dtype() == dl_float32
              (bfloat16 vs. float32) : attention_sinks must be a float tensor
```

`trtllm_batch_{decode,context}_with_kv_cache` require **float32** sinks. vLLM
handles this in `FlashInferImpl.process_weights_after_loading`, which keeps a
float32 copy of the parameter. Converting the layer's sinks once at load time is
*not* sufficient here, because the same layer also reaches FlashAttention:

```
AssertionError: learnable_sink must be bfloat16
  vllm/vllm_flash_attn/cute/interface.py:502
```

`TRTLLMPrefill` falls back to `flash_attn_varlen_func` whenever the prefill has
no block table (a fresh prompt), and the FA4 CuTeDSL kernel asserts the sinks
match the model dtype. So the conversion has to sit where the kernel choice is
known, not on the layer:

* `Attention` keeps the checkpoint parameter (`_fa3_sinks`, now a plain
  attribute rather than a second registration of the same `nn.Parameter`).
* `TRTLLMPrefill` / `TRTLLMDecode` hold their own float32 copy, primed from
  `Attention.process_weights_after_loading` so the cast never lands inside a
  forward or a graph capture.

`load_model` now calls `process_weights_after_loading()` on every `Attention`,
alongside the existing GptOssMoE / BitLinear passes.

### 11b. MXFP4 MoE: `matmul_ogs` over-reads exactly-sized ragged operands

With the sinks fixed, gpt-oss reached real decoding and died with `an illegal
memory access` — in the MoE, not in attention:

```
mxfp4_moe.py:234 in _fused_experts -> matmul_ogs        (the w2 / scatter matmul)
triton_kernels/matmul_ogs.py:467
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

Localizing it took some care because none of the obvious suspects held up:

| hypothesis | test | result |
|---|---|---|
| the MoE is wrong at small M | standalone `Mxfp4MoE` at M=1,2,4,8,16,33,1024 with gpt-oss's real shapes (E=128, top-4, H=2880, I_pad=1472) | all fine |
| garbage/NaN router logits produce out-of-range expert ids | feed NaN / Inf / 1e4 logits | no fault |
| custom all-reduce | `FASTKERNELS_DISABLE_CUSTOM_AR=1` | still faults |
| TP sharding | tp=1, single process | still faults |
| it writes past `y` | run with `y` at the head of a padded buffer and check the tail | no overrun |

What did discriminate was `max_model_len`: the harness derives it from the
dataset, and `--num-seqs 4` gives 946, which faults, while 2048 does not. That
is an allocator-layout dependence, i.e. a read past the end of a *small*
allocation that is normally absorbed by slack.

Instrumenting `_fused_experts` pinned the call: 36 layers x prefill (M=16) all
pass, and the **first single-token decode** (M=1, so a 4-row / 11 KiB
intermediate cache) faults on the second matmul. `matmul_ogs` reads its ragged
operands a full `block_m` tile at a time, and `opt_flags` picks
`block_m = max(16, min(next_power_of_2(tokens_per_expt), 128))` — up to 128 rows
against a 4-row buffer.

vLLM never trips this because its MoE buffers come from a workspace sized for
`max_num_batched_tokens` and are narrowed with `_resize_cache`; the over-read
stays inside the workspace. That is what `_resize_cache` is *for*, and this file
was calling it on exactly-sized allocations, making it a no-op. Both the
intermediate cache and the output are now allocated with their row count rounded
up to 128 and narrowed back, which is the same guarantee vLLM gets.

### 11c. Qwen3-Next and Kimi-Linear: prefill was never chunked

Both models OOM'd in the *second* workload (`long-context`, LongBench-v2, 8K-128K
prompts) after `mixed` completed:

```
Qwen3-Next: flashinfer/gdn_prefill.py:293  torch.empty  -> OOM (514 MiB)
Kimi:       flash_linear_attention/ops/chunk_delta_h.py:352
            h = k.new_empty(B, NT, H, V, K) -> OOM (988 MiB)
```

then a 600 s NCCL all-reduce timeout on the other rank and `SIGABRT`. The
all-reduce shape names the real problem: `NumelIn=269391872` at hidden_size 2048
is **131539 tokens in one prefill batch**, against
`max_num_batched_tokens=16384`. `_generate_kimi_linear`'s admission loop only
enforced the token budget from the *second* sequence on

```python
if prefill_seqs and prefill_tokens + seq_len > max_batched_tokens:
    break
```

so a single 128K-token prompt was always admitted whole. vLLM chunks it into
8 x 16384 (`Chunked prefill is enabled with max_num_batched_tokens=16384`), and
`_generate_mamba` already did the same for Mamba/Mamba2 — the hybrid path was
the one that never got it.

The scheduler now matches `_generate_mamba`'s shape: a `prefilling` queue that
admission fills (gated on recurrent-state slots and the worst-case KV block
reservation, not on prompt length), then a per-step chunk planner that fills
`max_num_batched_tokens` in FIFO order. State continuity across chunks was
already expressible — both `run_qwen3_next_mixed` and `_run_kimi_linear_batch`
build `has_initial_state` from `num_computed_tokens` — so the runners only needed
to take a per-sequence chunk length instead of "the whole remainder", and to
advance `num_computed_tokens` by the chunk rather than to `len(token_ids)`.

Three things had to follow:

1. **Kimi's MLA layers need the chunked-context path.** A resuming chunk attends
   to its own new tokens with the dense causal kernel and to the cached prefix
   through gather + non-causal attention + `merge_attn_states`
   (`MLACommonImpl._compute_prefill_context`). `_build_chunked_context` already
   existed for DeepSeek; `_run_kimi_linear_batch` now builds it whenever any
   scheduled sequence has `num_computed_tokens > 0`. Qwen3-Next needs nothing
   here: its full-attention layers already derive `cu_seqlens_k` from
   `md.seq_lens` (total) and `cu_seqlens_q` from `query_start_loc` (chunk).
2. **Empty context chunks must be masked.** A step can mix a resuming sequence
   with fresh ones, and a query whose context chunk is empty attended to zero
   keys, so the backend leaves its output rows undefined even though the LSE is
   `-inf`. vLLM calls `mask_empty_context` before merging;
   `ChunkedContextMetadata` now carries `has_empty_context` per chunk and
   `_compute_prefill_context` calls vLLM's own kernel.
3. **`GatherAndDequantKVCacheMLA` hardcoded `"fp8_ds_mla"`.** vLLM forwards its
   own `self.kv_cache_dtype`. With `kv_cache_dtype="auto"` (the default, and what
   Kimi runs) that reinterprets a BF16 cache as the 656-byte FP8 layout. It only
   became reachable now, because the gather is only used when a prefill has prior
   context. The dtype is a constructor argument, threaded from `MLAAttention`.

Also removed: the `qwen3_next`-only bump of `max_num_batched_tokens` to 32768.
vLLM logs 16384 for this model, and with chunked prefill there is nothing left
for the larger budget to buy.

### 11d. trtllm-gen MLA rejects an odd page-table width

Once Kimi's long-context prefill stopped OOMing, engine startup failed instead:

```
flashinfer/mla/_core.py:694
ValueError: Expected block_num % (128 / block_size) == 0,
            got block_num=2029 and block_size=64
```

`_init_kimi_decode_buffers` sizes the decode page table at
`ceil(max_model_len / block_size)` columns, and the trtllm-gen MLA decode kernel
walks it in 128-token strides. Whether the count comes out even depends on the
dataset (§9b-3 flagged this as satisfied "only incidentally"); 20260729-193844
got lucky and this smaller prompt set did not. Page-table widths on the
MLA/DeepSeek path are now rounded up to `ceil(128 / block_size)` — the extra
columns hold the pad block id and are never read, since the walk is bounded by
`seq_lens` — with a backstop pad inside `FlashInferMLADecode` for any other
caller.

### 11e. Decode graph capture used a zero context length

Unrelated to the crashes, found while chasing them: `capture_cudagraph` filled
`context_lens` with zeros, so every decode graph was captured against a state no
real step reaches. vLLM's uniform-decode dummy run uses
`seq_lens = max_query_len` (== 1) and zeroes only the padded tail
(`gpu_model_runner._dummy_run`). Now matched.

This is the only change in this round that touches models outside the three, so
it was checked against a control: `meta-llama/Llama-3.1-8B-Instruct` (tp=1,
`mixed`, 64 requests) comes out at **0.99x with 136.6/384 match tokens**, i.e.
unchanged and comfortably over the >=100 guideline. That also calibrates the
numbers in §11f: 136 leading tokens is what an aligned model looks like here, so
Qwen3-Next's 17 and Kimi's 5 are genuine outliers rather than the noise floor of
greedy decoding at this scale.

### 11f. Verdict

`fastkernels validate` over the three rows, workloads copied verbatim from
`full.yaml`, run `20260729-231524`:

```
RUN: PASS 3 / 3
  openai/gpt-oss-120b                     -> PASS
  Qwen/Qwen3-Next-80B-A3B-Instruct        -> PASS
  moonshotai/Kimi-Linear-48B-A3B-Instruct -> PASS
```

All four workloads (`mixed`, `long-context`, `single-request`,
`fixed-batch-32`) complete on both engines for all three, and both sides agree
on `max_num_batched_tokens=16384`.

### 11f-1. Measured alignment, now that all four workloads run

First end-to-end comparison for these rows -- all three used to abort before the
harness reached a verdict.

| model | scenario | speedup | avg match toks | exact matches |
|---|---|---|---|---|
| gpt-oss-120b | mixed | 0.35x | 61.5/385 | 106/1000 |
| gpt-oss-120b | long-context | 0.68x | 75.4/256 | 1/64 |
| Qwen3-Next-80B | mixed | 1.07x | 17.0/392 | 27/1000 |
| Qwen3-Next-80B | long-context | 0.66x | 18.8/256 | 0/64 |
| Kimi-Linear-48B | mixed | 1.24x | 5.0/389 | 5/1000 |
| Kimi-Linear-48B | long-context | 0.87x | 5.0/256 | 0/64 |

Two things to read off this:

* **Both hybrid linear-attention models diverge early** (17 and 5 leading
  tokens), which points at something shared in the GDN/KDA path rather than at
  either checkpoint. §9b-4's elimination list was built for Kimi alone; the
  Qwen3-Next number is new and says the same search should cover both.
* **The long-context regressions are the price of chunked prefill as
  implemented.** A prompt longer than the budget is prefilled strictly serially
  (one sequence's chunk per step, FIFO), whereas vLLM mixes decode tokens into
  the same step. That is correct but leaves parallelism on the table; the
  previous behaviour was a crash, so there is no earlier number to regress
  against.

### 11g. Negative result: the FP32 demotion was real but numerically inert

`load_model`'s `model.to(device, dtype)` was casting *every* parameter, which
demoted the tensors vLLM keeps in FP32:

| param | vLLM | fastkernels (before) |
|---|---|---|
| Kimi KDA `A_log`, `dt_bias` | `dtype=torch.float32` | declared FP32, cast to BF16 |
| Kimi KDA `{q,k,v}_conv1d.weight` | `params_dtype=torch.float32` | declared FP32, cast to BF16 |
| Kimi MoE `e_score_correction_bias` | model dtype | declared FP32, cast to BF16 |
| Qwen3-Next GDN `A_log` | `dtype=torch.float32` | declared BF16 |
| Qwen3-Next GDN `dt_bias`, `conv1d.weight` | model dtype | model dtype |

`_restore_mamba_ssm_params` exists precisely to undo this for mamba/mamba2, and
was never extended. `load_model` now uses a dtype-preserving move for the two
hybrid model types (the same rule the `quant_config` branch already applied),
Qwen3-Next's `A_log` is declared FP32, and Kimi's router bias is declared in the
model dtype. Verified after load: `A_log float32`, `dt_bias bfloat16`,
`conv1d.weight bfloat16` — matching vLLM element for element.

**It changed nothing.** Kimi went 5.1 -> 5.0 match tokens and Qwen3-Next was
bit-identical (16.950 match, 27 exact, both runs). The reason is that the
checkpoints store these tensors in BF16, so an FP32 parameter only ever holds
BF16-exact values in either engine — the container is wider, the numbers are the
same. Worth keeping (it is what vLLM does, and it matters the moment a
checkpoint ships FP32 values or a kernel accumulates in place) but it is not the
divergence, and this rules the hypothesis out rather than leaving it open.

### 11h. Still open

* **Kimi-Linear / Qwen3-Next numerics.** Unresolved, and §11g removes the most
  promising cheap explanation. Remaining untested lead: vLLM's `KimiMoE.gate` is
  a plain `ReplicatedLinear`, so its router logits reach
  `fused_grouped_topk` -> `ops.grouped_topk` as **bfloat16**, while fastkernels
  routes Kimi through `GateLinear(out_dtype=torch.float32)`. Unlike a stored
  weight, a matmul *output* dtype genuinely changes the values, and with 256
  experts, top-8 and sigmoid scoring plus a correction bias, that is exactly
  where near-tied expert selection flips. (Qwen3-Next does *not* have this
  problem: `use_grouped_topk=False` there, so `SharedExpertMoE` already uses the
  plain BF16 gate, and both engines then cast to FP32 for the softmax.) The next
  step remains the layer-by-layer hidden-state diff from §9b-4, now to be run for
  both models.
* **Page size parity for hybrid models** (extends §5.5): vLLM raises
  `cache_config.block_size` so the attention page is at least as large as the
  mamba page — 544 tokens for Qwen3-Next, 960 for Kimi-Linear, with the mamba
  page padded by 1.49% / 1.89% to match exactly
  (`CudaPlatform` -> `interface.py:905/929`). fastkernels keeps 16 (dense) and 64
  (MLA). This does not change attention math, but it does change how the two
  engines divide their KV budget: for Qwen3-Next, vLLM ends up with ~1.77M
  attention token slots and 3248 GDN state blocks out of 80.88 GiB, while
  fastkernels takes 3.85M token slots and 1024 state slots out of ~67.5 GiB.
* **gpt-oss-120b throughput is 0.35x vLLM** on `mixed` (8.5k vs 24.4k tok/s).
  Newly measurable, since the row never completed before. Not investigated.
* Qwen3-Next's GDN conv and recurrent *state cache* dtypes were checked against
  `MambaStateDtypeCalculator.gated_delta_net_state_dtype`: with
  `mamba_cache_dtype`/`mamba_ssm_cache_dtype` both `"auto"` they follow the model
  dtype (BF16), which is what `KimiLinearStateManager._allocate_qwen3_next`
  allocates. Kimi's KDA recurrent state is FP32 in both
  (`kda_state_dtype` returns `(state_dtype, torch.float32)`). No action.

---

## 12. Fourth round: throughput parity and the hybrid corruption

### 12a. All three models used the wrong MoE kernel

The single largest performance divergence. vLLM's oracle on B200 picks trtllm-gen
for every one of these rows; fastkernels used Triton for all three:

| model | vLLM backend / kernel | fastkernels (before) |
|---|---|---|
| gpt-oss-120b | `FLASHINFER_TRTLLM_MXFP4_BF16` -> `TrtLlmMxfp4ExpertsMonolithic` -> `flashinfer::trtllm_fp4_block_scale_moe` | OAI Triton `matmul_ogs` |
| Qwen3-Next-80B | `FlashInfer TRTLLM` -> `TrtLlmBf16ExpertsMonolithic` -> `flashinfer::trtllm_bf16_moe` | Triton `_fused_moe_kernel` |
| Kimi-Linear-48B | same as Qwen3-Next | Triton `_fused_moe_kernel` |

A decode profile of gpt-oss put 48% of all CUDA time in the two
`_p_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128` launches; a 32768-token
Qwen3-Next prefill put 22.5% in `_fused_moe_kernel` plus another 7% in
PyTorch's `mbtopk` (vLLM uses a fused `topk_softmax`).

**gpt-oss is done** (`L1/trtllm_mxfp4_moe.py` + `GptOssMoE` selection). Three
things that path needs and the Triton one does not:

1. 256-element rounding of hidden *and* intermediate
   (`mxfp4_round_up_hidden_size_and_intermediate_size`): 2880 -> 3072 and, at
   tp=2, 1440 -> 1536. The activation is zero-padded in; the kernel writes an
   unpadded output (`has_unpadded_output`).
2. A gate/up row swap (trtllm-gen orders the SwiGLU halves the other way) plus
   the shuffled/interleaved weight+scale layout for the transposed MMA epilogue.
3. `tune_max_num_tokens` = the 1024 max-capture size, *and* FlashInfer's startup
   autotuner. Both matter a lot: 0.71x with `tune_max_num_tokens=1` and no
   autotune, 0.87x with vLLM's values.

| gpt-oss | start | +trtllm MoE | +autotune | vLLM | target |
|---|---|---|---|---|---|
| mixed | 0.35x (8.5k) | 0.71x (17.3k) | **0.87x (21.4k)** | 24.5k | 0.95x |
| long-context | 0.68x | 0.78x | **0.90x** | — | 0.95x |
| single-request | 0.38x | 0.67x | **0.66x** | — | — |
| fixed-batch-32 | 0.37x | 0.81x | **0.86x** | — | — |

Match tokens held throughout (61.8 mixed, 74.9 long-context), so both of
gpt-oss's alignment targets are met. **Qwen3-Next and Kimi still need the
`trtllm_bf16_moe` port** — not started.

### 12b. `causal_conv1d_fn` returns zeros: the hybrid corruption

§11f-1 read the 17.0 and 5.0 match tokens as "diverges early". That was too
generous. The models are producing *garbage*, and the calibration is
unambiguous:

* Llama-3.1-8B, same prompt alone vs inside a batch of 32: **256/256 identical**.
* Kimi-Linear, same comparison: **0/256**, and the batched copy is token id 0
  repeated. Token 0 is `'!'`, not a pad or EOS, so this is not an
  `ignore_eos` artifact.
* Batch position changes which sequences are corrupt, and single-sequence runs
  are corrupt too. Qwen3-Next behaves the same way.

Bisected inward on Kimi (tp=1, 6 layers, 6-token prefill):

| level | finding |
|---|---|
| residual stream | clean through layer 3, **all NaN from layer 4** (36864/36864) |
| KDA layer output | **exactly 0** at layers 0-1, NaN at 2+ — the KDA branch contributes nothing, the residual only survived on the MLP branch |
| KDA params | every `A_log`, `dt_bias`, `{q,k,v}_conv1d.weight` finite and in range; nothing unloaded |
| state/metadata | `_get_state()` finds both; `num_prefills=1`, `num_prefill_tokens=6` — the real `chunk_kda` path runs |
| `chunk_kda` in isolation | NaN-free at T=6/64/300 with random inputs |
| **`causal_conv1d_fn` in-engine** | **returns 100% zeros** from `in amax=7.3`, `w amax=0.88` at layers 0-1; 50%/75% zeros at layer 2 |

So the varlen causal conv never writes its output. Its zeros become the
recurrence's q/k/v, which zeroes the KDA output; the partially-written garbage
at later layers then turns into NaN through the l2norm/recurrence. Both hybrids
call this op in prefill, which is why both are corrupt.

Not yet root-caused *inside* that call. The metadata it reads was compared
against vLLM's `compute_causal_conv1d_metadata` and matches (same `BLOCK_M=8`,
same `nums`/`mlist`/`offsetlist`, `PAD_SLOT_ID = -1`); the one structural
difference is that ours derives `seqlens` from a GPU `query_start_loc` (a DtoH
sync) where vLLM passes a CPU tensor. Next step is to call
`causal_conv1d_fn` standalone with the engine's exact arguments and bisect
`conv_states` layout / `cache_indices` / `has_initial_state` / `metadata`
against a direct `F.conv1d` reference.

### 12c. Fixed on the way: `cu_seqlens` integer width

The FLA chunk kernels index with the width they are handed, and vLLM's GDN
metadata builds `non_spec_query_start_loc` as **int32**. fastkernels passed
int64 (Kimi via `non_spec_query_start_loc`, Qwen3-Next via an explicit
`.to(torch.long)`). Verified in isolation on `chunk_kda` with otherwise
identical inputs: `amax=0.042` with int32 vs `amax=0.0005` with int64 — an
85x difference, i.e. silently wrong results. Now int32 in both paths. This is a
real bug but *not* the source of the zeros/NaN, which persist after the fix.

### 12d. Root cause of the conv partial write: conv-state stride

Captured the real argument set by hooking `causal_conv1d_fn` inside a *working*
vLLM Kimi-Linear worker (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, tp=1, patch
`kimi_gdn_linear_attn.causal_conv1d_fn`). Ground truth for one 89-token prefill:

| arg | vLLM | fastkernels | match |
|---|---|---|---|
| `x` | `[4096, 89]` stride `[1, 4096]` bf16, non-contig | same | yes |
| `weight` | `[4096, 4]` stride `[4, 1]` **fp32** contig | same | yes |
| `query_start_loc` | `[2]` **int32** | was int64, now int32 | fixed (§12c) |
| `has_initial_state` | `[1]` bool | same | yes |
| `cache_indices` | `[1]` int32 | same | yes |
| **`conv_states`** | `[blocks, 4096, 3]` stride **`[1087488, 1, 12288]`** | `[slots, 4096, 3]` stride **`[12288, 1, 4096]`** | **NO** |
| `OUT` | `[4096, 89]` stride `[1, 4096]`, **zero_pct 0.0** | zero_pct 50-75% | **NO** |

The KDA conv state is the divergence. vLLM allocates **one packed conv state per
layer** of width `3 * proj_size` (`kda_state_shape`: `conv_dim = proj_size +
2 * proj_k_size`) and passes the kernel
`conv_state.transpose(-1, -2).chunk(3, dim=-2)` — three views whose
state-length stride is the *packed* width, 12288.
`KimiLinearStateManager` instead allocates three separate
`[num_slots, kernel-1, proj]` tensors, so each transposed view carries
state-length stride 4096. Same shape, same `contig=False`, different stride,
and the kernel writes only part of its output.

Reproduced by construction, independent of the capture:

```python
packed = torch.zeros(blocks, kern-1, 3*proj)          # vLLM
packed.transpose(-1,-2).chunk(3, dim=-2)[0].stride()  # (36864, 1, 12288)
sep = torch.zeros(blocks, kern-1, proj)               # fastkernels
sep.transpose(-1,-2).stride()                         # (12288, 1,  4096)
```

**Tried, and it is NOT the cause.** Allocating one packed conv state per layer
and taking `transpose(-1,-2).chunk(3, dim=-2)` views reproduces vLLM's stride
exactly and left the conv output *byte-identical* (0% / 50% / 75% zeros
unchanged). In hindsight that is the expected result: on a fresh prefill
`has_initial_state` is all False, so the conv state is only ever *written*, and
its stride cannot affect that step's output. The stride is a genuine interface
divergence worth aligning for the decode path (where the state *is* read), but
it is not this bug. The change was reverted rather than left unvalidated,
because `conv_q/k/v` would become views into a packed tensor that
`mark_static_address` is not applied to, which the CUDA-graph decode path may
care about, and there was no budget to test that end to end.

**Where to look next.** The zero fraction depends only on *which of q/k/v* is
being computed -- 1st call 0%, 2nd 50%, 3rd 75%, with all three inputs and
weights healthy and independent. Since conv_states is write-only here and the
metadata, index width, layout orientation and weights are all eliminated, the
remaining candidates are the *output* buffer the kernel allocates internally and
the `activation="silu"` epilogue. Measured: the launch **is** fully covered and **identical** for all three
calls. For a 256-token prefill each of q/k/v sees `tot=32`, `mlist_len=32`,
`nums=[32]`, `batch_ptr` entries used `=32` (= `cdiv(256, 8)`), and
`x_shape=(256, 4096)` -- yet the outputs come back 0%, 50% and 75% zeros
respectively. So identical shapes, identical metadata and identical grid produce
different zero fractions, which rules out launch coverage and points the
remaining search at the kernel's own output allocation or its per-call state.

**Localized.** The zeros are per-*channel*, and the written channel count
**halves on each successive call within a layer**:

| call | channels written | zero channels | zero tokens |
|---|---|---|---|
| q (1st) | 4096 / 4096 | 0 | 0 / 256 |
| k (2nd) | 2048 / 4096 | 2048 | 0 / 256 |
| v (3rd) | 1024 / 4096 | 3072 | 0 / 256 |

Every token is touched; it is the channel extent that shrinks 4096 -> 2048 ->
1024. All three calls have identical input shape, dtype, strides, weights shape
and metadata (`tot=32`, `nums=[32]`, `batch_ptr` used 32), and the launch grid is
identical, so this is neither data-dependent nor a coverage problem. Something
carried *between* the three calls is halving a channel bound.

Note this survives both fixes already tried: rebuilding the metadata per call,
and switching to vLLM's packed `conv_states` stride. So the carrier is not the
metadata object and not the state tensor.

**Not a cached Triton config**: rerunning with `TRITON_ALWAYS_COMPILE=1` gives
byte-identical results (0 / 2048 / 3072 zero channels), so autotune caching is
ruled out too. The halving is deterministic and reproduces on every run.

Superseded hypothesis, kept for the record: a cached kernel launch configuration. `causal_conv1d_fn` caches
its Triton config, and the three calls are indistinguishable in their cache key,
so a *correct* cache would hand back the same config -- a shrinking one implies
either the key is colliding with something mutable or the cached config holds a
channel/`BLOCK_N` bound that is being consumed. Next step: log the config
`causal_conv1d_fn` selects on each of the three calls (and whether it hits or
misses its cache), and try `TRITON_ALWAYS_COMPILE=1` / clearing the cache between
calls to see whether the halving disappears. If it does, the fix is to stop
whatever is mutating the cached entry; the three convs may need distinct cache
identities the way vLLM's packed-chunk views naturally give them.

Note the capture run also tripped `Expected block_num % (128 / block_size) == 0,
got block_num=59 and block_size=32` inside vLLM's own MLA decode at
`max_model_len=1024` — the same constraint §11d fixed on our side, hit here by
the reference itself.

### 12e. Disproved, do not retry

* Conv metadata builder: ours vs vLLM's `compute_causal_conv1d_metadata` both
  give garbage in a standalone harness, and **vLLM's own builder produced NaN**
  there — the harness had a wrong argument, so its error column was never
  meaningful. Only the zeros/NaN signal was.
* `metadata` as a consumed work queue: rebuilding it per conv call changed
  nothing (0%/50%/75% zeros unchanged). Reverted; it cost three DtoH syncs per
  layer for no gain.
* Conv state layout orientation: `is_conv_state_dim_first()` is False -> "SD",
  so our `[slots, kernel-1, dim]` storage plus transpose matches vLLM's
  orientation. The *stride* is the problem, not the orientation.
* `query_start_loc` int width: a real bug (§12c) but not this one — the engine
  still returned zeros after the fix.
* KDA parameters: every `A_log`, `dt_bias`, `{q,k,v}_conv1d.weight` finite and
  in range; nothing unloaded.
* `chunk_kda` itself: NaN-free standalone at T=6/64/300.

### 12f. Conv bug: state of the search

Confirmed behaviour: within one KDA layer the three `causal_conv1d_fn` calls
write 4096, then 2048, then 1024 of 4096 channels. Deterministic, every run,
every layer. All tokens are touched. Inputs, weights, strides, metadata and
launch grid are identical across the three calls.

Eliminated (do not retry): metadata builder choice; per-call metadata rebuild;
`query_start_loc` int width; `conv_states` stride / packed-vs-separate layout;
conv state layout orientation; parameter health; `chunk_kda` itself; launch
coverage; Triton config caching.

The halving 4096 -> 2048 -> 1024 is the whole signal. It is a factor of two per
call, which no argument that is *identical* across the calls can explain -- so
the next thing to establish is what the kernel sees that we have not inspected:
dump the resolved kernel arguments (not our Python-side ones) inside
`causal_conv1d_fn` for each of the three calls and diff them. The one input we
have never compared between the calls is `conv_weight`, which comes from three
distinct `_Conv1DWeights` modules via `.view(size(0), size(2))` on a
`[proj, 1, kernel]` parameter -- worth confirming all three views really are
`[4096, 4]` with stride `(4, 1)` at the point of the call, since a stale or
mis-viewed weight is the remaining way three "identical" calls could differ.

## 13. The hybrid corruption: `causal_conv1d_fn` skipped state slot 0

§12d-f chased this as a "partial write" -- the three conv calls in one KDA layer
appearing to cover 4096, then 2048, then 1024 of 4096 output channels. That
framing was wrong, and it is worth recording why, because it cost several
rounds.

**The measurement that broke it open.** Dumping the conv input's amax next to the
output's amax showed the output of call N carrying the *input* amax of call N-1,
exactly:

```
L0 q: in amax=7.344 -> out amax=0.2656
L0 k: in amax=5.156 -> out amax=7.344   <- L0 q's input
L0 v: in amax=8.438 -> out amax=5.156   <- L0 k's input
```

A conv of k's input cannot produce q's input's amax. The buffer was never
written: `causal_conv1d_fn` does `out = torch.empty_like(x)`, so an unwritten
output returns whatever the caching allocator last left there. The
"channel halving" was an artifact of which freed block got recycled, not a
property of the kernel -- which is why nine hypotheses about grid coverage,
metadata, strides and int widths all came back clean. Comparing against an
`F.conv1d` reference per 256-channel tile confirmed it: **0 of 16 tiles matched,
on every call**, not some.

**The cause.** `causal_conv1d_fn(..., null_block_id=NULL_BLOCK_ID)` defaults to
vLLM's `NULL_BLOCK_ID`, which is `0`. The kernel's first act after loading a
sequence's cache index is

```python
if HAS_NULL_BLOCK:
    if conv_states_input_coord == null_block_id:
        return          # treats this sequence as padding
```

vLLM never trips this because its block allocator reserves block 0 as the null
block, so real state indices start at 1 -- its own unit test builds
`state_indices` as `randperm(total_entries - 1) + 1` with the comment "+1 to
exclude index 0 (null block)". Our state manager hands out slot 0 as an ordinary
slot, so **every batch containing the sequence in slot 0 had that sequence's
entire conv output left uninitialized**, and with one sequence in flight that is
the whole batch.

**The fix.** Pass `null_block_id=-1` on both `causal_conv1d_fn` calls
(`kimi_delta_attention._run_conv_prefill`, `qwen3_next_gdn_attention.forward`).
-1 is already this allocator's pad/skip slot and is already what we pass to
`causal_conv1d_update`, so prefill and decode now share one convention: -1 means
skip, every non-negative slot is real. After the fix, 16/16 tiles match
`F.conv1d` on all nine probed calls.

**Why it hid for so long.** The decode call had `null_block_id=-1` already, so
only prefill was affected; and kb_nano's B200 reproduction records
Kimi-Linear at 4.8 matched tokens against vLLM 0.18, i.e. this predates the
0.26 port rather than being introduced by it.

**Generalization worth checking elsewhere.** Any vLLM kernel taking a
`null_block_id`/`pad_slot_id` sentinel assumes vLLM's allocator conventions. Our
allocators differ, so each such call site needs the sentinel set explicitly
rather than defaulted. `causal_conv1d_update` and `causal_conv1d_fn` are now
consistent; other mamba/conv entry points should be audited the same way.

## 14. BF16 MoE: vLLM runs trtllm-gen, we ran Triton

vLLM 0.26 picks the `FLASHINFER_TRTLLM` unquantized backend ->
`TrtLlmBf16ExpertsMonolithic` for both hybrids' MoE on SM100; a reference run
logs `[AutoTuner]: Tuning flashinfer::trtllm_bf16_moe`. We ran Triton
`_fused_moe_kernel` plus a separate gate/top-k, which a Qwen3-Next 32768-token
prefill profile put at 22.5% (+ ~7% in PyTorch `mbtopk`).

`L1/trtllm_bf16_moe.py` ports the kernel call and the 4D BlockMajorK weight
shuffle (`convert_moe_weights_to_flashinfer_trtllm_block_layout`). Two things
the port got wrong first time, both silent rather than loud:

* `activation_type` must be `ActivationType.Swiglu == 3`. vLLM maps
  `MoEActivation.SILU` to the *gated* Swiglu, and the launcher derives
  `intermediate_size_factor` from it -- a non-gated id makes
  `check_weights_shape` reject w13's `2*I` rows (`256 vs. 512`).
* `routing_method_type` must come from vLLM's `get_routing_method_type`:
  Qwen3-Next (softmax + renormalize, no bias) -> `RenormalizeNaive` (4); Kimi
  (sigmoid + router bias + expert groups) -> `DeepSeekV3` (2). The kernel then
  applies gating, top-k, `routed_scaling_factor` and the weighted reduction
  itself, so those steps come out of the Python path.

Measured against the Triton path on tp=2 shapes: cos 0.99998 for both models at
8 and 512 tokens.

**Speed, and why the eager numbers are misleading.** Eagerly, trtllm-gen is
*slower* below ~2048 tokens -- a flat ~0.72 ms regardless of token count against
Triton's 0.13-0.31 ms -- and faster above (2.15x at 8192 for Qwen3-Next, 1.41x
for Kimi). That floor is host-side launch/dispatch, not device time: captured
into a CUDA graph and replayed, trtllm-gen is faster at *every* size measured.

| tokens | Triton graph ms | trtllm graph ms |
|---|---|---|
| 8 | 0.061 | 0.051 |
| 64 | 0.202 | 0.185 |
| 256 | 0.279 | 0.252 |
| 512 | 0.297 | 0.268 |

So enabling it unconditionally (as vLLM does) is right, and it depends on decode
being graph-captured -- which `capture_kimi_cudagraph` does for both hybrids.

## 15. CUDA-graph bucket lists: the hybrid path is coarser than vLLM's

vLLM's `cudagraph_capture_sizes` steps by 8 from 16 to 256 and by 16 from 256 to
`max_cudagraph_capture_size` (512):
`[1, 2, 4, 8, 16, 24, 32, 40, ..., 248, 256, 272, ..., 512]`.

Our main path (`engine.py`, `graph_bs_list`) already has that shape. But
`capture_kimi_cudagraph` -- used by **both** Kimi-Linear and Qwen3-Next -- uses
`[1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256]` plus by-16 above
256, which is coarser below 256. A decode batch of 100 pads to 128 (28% wasted
work) where vLLM pads to 104 (4%). `capture_mamba_cudagraph` has the same coarse
list.

## 16. Qwen3-Next's full-attention KV cache was allocated NHD for an HND consumer

Qwen3-Next's full-attention layers select trtllm-gen on Blackwell and say so
explicitly -- `qwen3_next_attention` builds `StoreKVCacheHND(page_size=...)` and
`flashinfer_decode` passes `kv_layout="HND"`. But
`MambaStateManager._allocate_qwen3_next` allocated the cache in the *other*
order:

```python
self.k_cache[i] = torch.zeros(num_mla_blocks, block_size, local_kv_heads, head_dim)
```

trtllm's launcher reads `num_kv_heads` off `kv_cache.shape[1]`, so it saw the
page size (16) where the head count belonged and refused to launch:

```
num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: 16 and num_qo_heads: 8
```

during `capture_kimi_cudagraph`. The generic path already branches correctly
(`HND -> [blocks, num_kv_heads, BLOCK_SIZE, head_dim]`, else
`[blocks, BLOCK_SIZE, num_kv_heads, head_dim]`); `_allocate_qwen3_next` now does
the same.

Worth noting why this was easy to miss: at **tp=2** `local_kv_heads` is 1, so the
two orders describe the *same* bytes -- only the reported shape differed, so
writes through `StoreKVCacheHND` were fine and nothing was numerically wrong
until a consumer inspected `shape[1]`. At tp=1 (2 kv heads) the orders genuinely
differ, so this was also a latent correctness bug there.

Ruled out first, by A/B with `FASTKERNELS_TRTLLM_BF16_MOE=0`: the failure is
independent of the new MoE backend (both arms fail identically), so the MoE port
was not implicated.

## 17. Both hybrids generate incoherent text, and it is not the MoE or TP

A fast repro exists now: `/tmp/fkdev/hyb_smoke.py` (init + 48 greedy tokens at a
chosen TP), ~8 minutes against ~50 for a validate row. `SMOKE_MODEL` picks the
model, `SMOKE_TP` the parallelism.

```
Kimi tp=2 : '  \nevaData昔 questions Films后 Osw人体amera取材),执法同步 Tart/music月)...'
Kimi tp=1 : '  \n阿布面试～害怕照护履转换为~-  CBD面试 AR出息聊 Limb转换为?...'
Qwen tp=2 : ' 崀ing趸·edefトン\n\nдон爐...\n Sir hammer就必须 鐵 Strait and.fed團隊...'
```

This is incoherent output, not merely divergent-from-vLLM output, and it explains
the matched-token figures directly: Kimi 5.1, and Qwen3-Next's earlier 17.0 is
the only number inconsistent with garbage, so Qwen3-Next may have regressed at
some point after run `20260729-231524`.

Eliminated, each by experiment rather than by argument:

* **The trtllm BF16 MoE port.** `FASTKERNELS_TRTLLM_BF16_MOE=0` leaves both
  models equally incoherent (Kimi tp=1, Qwen3-Next tp=2). The port is also
  cos 0.99998 against the Triton path in isolation.
* **TP weight sharding.** Kimi is incoherent at tp=1 as well as tp=2, so no
  sharding split is implicated.
* **The varlen conv.** §13's fix is verified against `F.conv1d` in-engine, 16/16
  channel tiles on all nine probed calls.
* **The KDA kernel pair.** Kimi was already at ~4.9 matched tokens before the
  port to `chunk_kda_with_fused_gate` / `fused_recurrent_kda`, and kb_nano
  records 4.8 against vLLM 0.18, so the corruption predates it.

The first generated token is already wrong, which places the fault in **prefill**,
not decode. That matters for triage: it rules out decode-only suspects, including
the MLA decode kernel discussed below.

Open leads, in the order they look worth testing:

1. **Kimi's MLA prefill.** Kimi runs 7 full MLA layers alongside its KDA layers.
   The engine logs `MLA attention backend: FlashMLA` on B200, and FlashMLA's
   *dense decode* is Hopper-only (kb_nano defect 7, whose fix was to select a
   FlashInfer `trtllm_batch_decode_with_kv_cache_mla` path on sm100+). Note
   `tasks/baseline/L1/flashinfer_mla_decode.py` exists in this tree but is
   imported nowhere -- dead code. Wiring it is required alignment regardless, but
   it is a *decode* path and so cannot explain a wrong first token; do not expect
   it to fix this symptom on its own.
2. **Qwen3-Next's full-attention layers on trtllm-gen.** head_dim is 256, which
   is exactly `_TRTLLM_MAX_HEAD_SIZE`, so the generic per-layer gate admits it and
   vLLM likewise logs `Using FLASHINFER attention backend` / `HND KV cache
   layout`. But vLLM's FLASHINFER backend uses its *wrapper* for large batches and
   trtllm-gen only when `num_tokens <= 256` (`vllm/utils/flashinfer.py`), whereas
   `qwen3_next_attention` uses trtllm-gen unconditionally. trtllm-gen also has no
   context kernel for some head_dim-256 configurations -- a smaller config raised
   `Missing TRTLLM-GEN kernel (context): ... headDimQk=256`.
3. **A layer-wise magnitude walk** (`/tmp/fkdev/kimi_layer_probe.py`, unfinished)
   to see whether the corruption enters at a KDA or an MLA layer.

Caveat on reading matched-token deltas: two identical gpt-oss builds gave 63.2
and 56.1 on `mixed`. Treat single-run differences under ~10 tokens as noise --
kb_nano's §"batch-occupancy dependent" analysis explains why.

### 17a. Kimi prefill: six candidates eliminated by measurement

Each of these was checked directly rather than argued about, so they should not be
re-litigated:

| candidate | how it was ruled out |
|---|---|
| trtllm BF16 MoE port | `FASTKERNELS_TRTLLM_BF16_MOE=0` -> still incoherent (Kimi tp=1, Qwen tp=2) |
| TP weight sharding | incoherent at tp=1 as well as tp=2 |
| varlen conv | 16/16 channel tiles match `F.conv1d` in-engine (§13) |
| KDA kernel choice | ~4.9 matched tokens *before* the port; kb_nano saw 4.8 on vLLM 0.18 |
| layer-type assignment | `linear_attn_config.full_attn_layers` is **1-indexed** (`[4,8,12,16,20,24,27]`) -> 0-indexed 3,7,11,15,19,23,26, exactly the pattern a layer walk observes. 27 layers, 20 KDA + 7 MLA. |
| uninitialized KDA parameters | every parameter of the first three KDA layers audited: all loaded, all finite, magnitudes O(0.01-5). `A_log` `(1,1,32,1)` fp32, `dt_bias` `(4096,)` fp32 -- both matching vLLM's shapes/dtypes. |

The uninitialized-parameter check was worth doing explicitly: every KDA parameter
is `nn.Parameter(torch.empty(...))`, so a `weight_loader` that never fires (name
mismatch, shape mismatch) leaves garbage silently, with no error and no NaN. It
is not what is happening here, but it is the failure mode this layer's design
invites.

Also checked and *not* applicable: kb_nano's defect 9 (`_pad_v` for MLA prefill
where qk is 192 and v is 128). Kimi-Linear sets `mla_use_nope=True`, so qk and v
are both 128 and no padding is required -- our `FlashAttnVarlen` has no `_pad_v`,
which is fine for this model but would bite a 192/128 MLA config.

**Remaining suspects, narrowest first:** Kimi's 7 MLA layers (`_forward_mha`
prefill, the absorbed `kv_b_proj` weights, and `FlashMLA` being selected on B200
where its dense decode is Hopper-only), and the KDA output gating
`o_norm(core_attn_out, g2)` (note vLLM constructs `FusedRMSNormGated(head_dim,
activation="sigmoid")` without passing `eps`, where we pass
`eps=config.rms_norm_eps`).

**The experiment that would settle it** is the one still missing: a reference.
Run vLLM in-process on the same prompt with a per-decoder-layer hook, and diff its
hidden states against ours layer by layer. Every check above narrows the space but
none can localize without ground truth, and a layer-magnitude walk on our side
alone is not conclusive -- the residual/normed-output convention makes the
trajectory legitimately non-monotonic (observed rms: 0.014, 0.034, 0.29, 0.27,
0.069, ... 6.30 at L22, 5.12, 1.02, 1.79, 2.82).

### 17b. Correction: prefill is correct; the fault is in decode

§17 claimed "the first generated token is already wrong, which places the fault
in prefill". **That was wrong**, and it was an inference from the look of the
decoded string rather than a measurement. Comparing against vLLM op by op
settles it.

Method (per reviewer guidance): dump per-op activations from both engines on the
same prompt token ids, walk them in execution order, and find the *first* op whose
cosine drops. Everything after the first divergence is noise -- once one expert
index or one recurrent state differs, the error grows catastrophically, so later
numbers carry no information. Harness: `/tmp/fkdev/dump_vllm.py` (vLLM in-process
via `VLLM_ENABLE_V1_MULTIPROCESSING=0`), `/tmp/fkdev/dump_ours.py` (hooks
installed by patching classes, since `LlamaEngine` exposes no model attribute),
`/tmp/fkdev/cos_compare.py`.

Both engines load the **full** 27-layer checkpoint and capture only layers 0-3
(which for Kimi covers 3 KDA layers plus the first MLA layer, since
`full_attn_layers` is 0-indexed 3, 7, 11, ...). Truncating the model instead is
not an option: vLLM's loader raises
`KeyError: layers.4.block_sparse_moe.experts.routed_experts.w13_weight` on the
leftover checkpoint keys.

Result, prefill step, against vLLM:

| op | cos |
|---|---|
| L00.layer | 1.000002 |
| L01.layer | 0.999609 |
| L02.layer | 0.999996 |
| L03.layer | 0.999949 |

and **both engines emit the same first token (31082)**. So prefill -- embeddings,
the KDA sublayer, the MoE, and the first MLA layer -- is right to bf16 rounding.

(The `L00.mlp` cos of 0.07 in the first run was an artifact of this harness, not a
finding: our MoE hook tagged captures with a call *counter* rather than the layer
index, so it compared different layers. `ref_amax` 0.377 is L00.layer's value and
`ours_amax` 0.3242 is L01.layer's, which is the tell. Fixed by taking the counter
modulo the layer count.)

Also ruled out by this: **the compiled/CUDA-graph path**. Kimi degenerates
identically with `enforce_eager=True` over 48 tokens, so graph capture, buffer
reuse and metadata staleness are not involved.

So the corruption enters at a **decode** step, with prefill's output verified
good. That relocates the search to the prefill -> decode handoff and the decode
kernels: the recurrent state written by `chunk_kda_with_fused_gate` and read by
`fused_recurrent_kda`, the conv state written by `causal_conv1d_fn` and read by
`causal_conv1d_update`, and the decode-side `cu_seqlens` / state-index slicing.

Note the state-layout question from §12 becomes live again here: the custom decode
kernel that was removed existed precisely to bridge a claimed `[V, K]` vs `[K, V]`
disagreement between `chunk_kda` and `fused_recurrent_kda`. Prefill-good /
decode-bad is exactly the signature such a mismatch would produce. But Kimi scored
~4.9 matched tokens *with* that custom kernel too, so if the mismatch is real,
that kernel did not correctly bridge it either.

The next measurement is per-decode-step: capture the same ops at steps 0, 1, 2 and
find the first divergent op at step 1.

### 17c. Localized: the *first* decode step is wrong

A prefill/decode self-consistency test settles this without needing vLLM at all
(`/tmp/fkdev/prefill_decode_consistency.py`). Generate N tokens normally (one
prefill, N-1 decode steps), then re-run each prefix as a pure prefill and check it
predicts the same next token. Prefill is the verified-good path (§17b), so any
disagreement is decode.

```
decode path  : [31082, 153974, 22629, 50569, 4248, 4248]
prefill-only : [31082,     13,   378,   382,   88, 2189]
FIRST DISAGREEMENT at generated position 1
```

Position 0 agrees (and matches vLLM). Position 1 -- the very first decode step --
diverges. So the corruption is not cumulative drift; the first decode step is
already wrong.

This reproduces on a **4-layer** truncation (3 KDA layers + the first MLA layer),
which makes the repro ~2 minutes. Note the consistency test stays valid on a
truncated model even though its text is gibberish: it asks only whether decode
reproduces prefill's own prediction, not whether that prediction is good.

Corrected along the way: the engine prints `MLA attention backend: FlashMLA` on
B200, but that string is stale. `MLAAttention` already constructs
`FlashInferMLADecode` whenever `flashinfer_mla_decode_supported()` (capability
>= 10) and `_forward_dense_decode` branches to it, so Blackwell already runs
trtllm-gen MLA decode rather than the Hopper-only FlashMLA dense kernel. That also
explains the absence of the `Dense decode MLA is only supported on SM90a` error
kb_nano saw. `flashinfer_mla_decode.py` is therefore *not* dead code -- an earlier
note in this document said it was imported nowhere, which was wrong; it is imported
lazily inside `MLAAttention.__init__`.

**What is left to split:** the first decode step runs both KDA decode
(`causal_conv1d_update` + `fused_kda_gate` + `fused_recurrent_kda`, reading the
conv and recurrent state that prefill wrote) and MLA decode
(`trtllm_batch_decode_with_kv_cache_mla`). Two ways to separate them:

1. A KDA-only truncation (`max_layers=3`, since `full_attn_layers` starts at
   0-indexed 3) currently dies with `ZeroDivisionError` in the MLA KV-cache
   sizing, which divides by the number of MLA layers. Making that tolerate zero
   MLA layers would give a one-run answer.
2. Extend the op dump to decode steps and diff step 1 op by op. Our side already
   produces this (33 tensors over 3 steps, tagged `S{step}.L{layer}.{sublayer}`);
   only the vLLM side needs its page table sized so
   `block_num % (128 / block_size) == 0` -- it raised that at both
   `max_model_len=512` (block_num=59) and 2048 (block_num=118). Note vLLM trips
   its own constraint here, which is the same one `_pad_page_table_cols` works
   around on our side.

Given prefill writes state that only decode reads, the recurrent-state handoff
between `chunk_kda_with_fused_gate` (writes `pf_last_state` into
`recurrent_state[pf_state_indices]`) and `fused_recurrent_kda` (reads
`recurrent_state` indexed by `ssm_state_indices`) is the single most likely
culprit, and the §12 `[V, K]` vs `[K, V]` layout question is directly in scope.

### 17d. The decode recurrence never advances its state

Probing the hand-off directly (`/tmp/fkdev/state_probe.py`, KDA layer 0, 4-layer
truncation) shows what the earlier indirection was hiding:

```
PREFILL  recurrent before: amax=0      rms=0         zero%=100.0
PREFILL  recurrent after : amax=1.892  rms=0.01214   zero%=0.0     <- written
DECODE 1 recurrent before: amax=1.892  rms=0.01214
DECODE 1 recurrent after : amax=1.892  rms=0.01214                 <- UNCHANGED
DECODE 2 recurrent after : amax=1.892  rms=0.01214                 <- UNCHANGED
q_conv   prefill 6.406 -> decode 5.594 -> 5.594 (rms 0.5626/0.5281/0.5194)
```

So `causal_conv1d_update` advances the conv state correctly, but
`fused_recurrent_kda` leaves the recurrent state byte-identical. Every decode step
re-runs from prefill's state, so the sequence cannot progress -- which is exactly
"the first decode step is already wrong, and it degenerates from there".

This also explains a null result that was otherwise confusing: transposing
`pf_last_state` at the store site (§17c's `[V, K]` vs `[K, V]` hypothesis) produced
**byte-identical** tokens. It was not evidence against a layout mismatch; the
stored state simply is not being read-modified-written by decode at all.

Two things tried:

* **`ssm_state_indices` as int64.** vLLM annotates it `torch.LongTensor` and we
  passed int32. This *does* change the decode output ([31082, 153974, ...] ->
  [149143, 32600, ...] on the 4-layer repro), so the index width was being
  misread -- the fix is kept. It does not make the state advance.
* **`inplace_final_state=False` plus an explicit scatter** of the returned final
  state into the bank, mirroring what the prefill branch does. This changed
  *nothing*: same tokens, same unchanged state. Since scattering uninitialized
  memory would have visibly corrupted the bank, the returned final state must
  equal the initial state -- i.e. **the kernel itself is not advancing the
  recurrence**, rather than the write-back being lost. Reverted, since a no-op
  change with a fix-shaped comment is worse than none.

Where that points: the kernel produces an output (the int64 change moved it) but
no state delta. A state update of the form `h <- h * exp(g) + beta * (v - h k) k`
degenerates to `h <- h` when `g == 0` and `beta == 0`, so the decode-side `g` and
`beta` are the next things to inspect -- specifically `dec_g`, built as
`fused_kda_gate(rearrange(raw_g[:, dec_start:], "1 n h d -> n (h d)"), A_log,
head_dim, g_bias=dt_bias)`, and `dec_beta = beta[:, dec_start:]`. Print their
magnitudes at a decode step; if either is ~0 the cause is upstream of the kernel,
in the raw-gate plumbing added when the layer was switched to vLLM's fused-gate
prefill (§8).

### 17e. The decode recurrence returns NaN

Instrumenting the decode call's inputs and output (`/tmp/fkdev/decode_inputs.py`)
gives the actual failure, and reframes §17d: the frozen state is a *consequence*,
not the cause.

```
fused_kda_gate in  (1, 4096) amax=8.438  mean=-2.171
fused_kda_gate out (1,32,128) amax=91.02 min=-91.02   exp(g): min=2.95e-40 max=1
rec.q    (1,1,32,128) bf16 amax=0.3398  rms=0.03425
rec.k    (1,1,32,128) bf16 amax=0.9883  rms=0.04413
rec.v    (1,1,32,128) bf16 amax=2.375   rms=0.08747
rec.g    (1,1,32,128) fp32 amax=91.02 min=-91.02
rec.beta (1,1,32)     fp32 amax=0.9999 mean=0.9465 min=0.4026
rec.initial_state (1024,32,128,128) fp32  idx=[0] int64
rec.out  (1,1,32,128) amax=nan mean=nan min=nan   final_state is initial_state: True
tokens: [149143, 0, 0]
```

`fused_recurrent_kda` returns **NaN**. It therefore writes nothing back (hence the
byte-identical state of §17d), and NaN logits make argmax collapse -- the trailing
zeros in the token ids, and the reason token ids differed run-to-run earlier (NaN
argmax is not well-defined).

Every input is finite and reasonable: q/k/v small, `beta` in [0.10, 1.00] as
sigmoid requires, `g` in [-91, 0] as `-exp(A_log) * softplus(...)` requires (A_log
amax ~5.3 so `exp(A_log)` ~200, times softplus up to ~8, lands in the tens).
Shapes and dtypes all match vLLM's call: `g` `[1, n, H, D]` fp32, `beta`
`[1, n, H]` fp32, `initial_state` the full `[slots, H, D, D]` fp32 bank indexed by
`ssm_state_indices`.

The one numerically suspicious quantity is `exp(g)`, whose minimum is **2.95e-40**
-- a denormal float32. If the kernel divides by the cumulative decay anywhere, and
denormals flush to zero under Triton's fast-math, that is inf/NaN. Note prefill
computes the same gate from the same `A_log`/`dt_bias` *inside*
`chunk_kda_with_fused_gate` and is verified correct against vLLM, so the gate
values themselves are not wrong -- the chunk kernel tolerates them and the
recurrent one does not.

Next: reproduce standalone by feeding `fused_recurrent_kda` these magnitudes
(g down to -91, beta ~0.9, one token, one sequence, a non-zero initial state) and
bisect on `g`'s range to find where NaN appears. Then check what vLLM does
differently -- whether its Kimi decode ever sees `g` this extreme in practice, and
whether `chunk_kda_with_fused_gate`'s internal gate applies a clamp that
`fused_kda_gate` does not.

### 17f. Root cause: the recurrent kernels skip state slot 0

`fused_recurrent_kda` treats state index 0 as a null/skip slot -- the same
`NULL_BLOCK_ID = 0` convention as `causal_conv1d` (§13). Given slot 0 it returns
NaN and writes no state; slots >= 1 are correct. Standalone, no model involved:

```
slot=0  idx int64/int32:  out nan=5/8  amax=nan     state delta=0
slot=1  idx int64/int32:  out nan=0    amax=0.0039  state delta=0.0876
slot=3  idx int64/int32:  out nan=0    amax=0.0040  state delta=0.0890
```

`chunk_kda_with_fused_gate` on the same inputs is clean, which is why prefill was
correct and only decode failed. NaN logits then make argmax collapse, so the
generated ids were partly meaningless -- and non-deterministic run to run, which
had been quietly confusing several earlier A/Bs.

This is the **same root cause as §13**. Fixing that one at the call site with
`null_block_id=-1` left it latent for the recurrent kernels, which hardcode the
convention and expose no such parameter. The right fix is the allocator: reserve
slot 0, as vLLM's block allocator does.

That took four sites, because the free-slot pool is rebuilt three times after
construction:

| site | when it runs |
|---|---|
| `KimiLinearStateManager.__init__` | construction |
| `engine.py` `use_decode_graph` branch (x2) | when decode graphs are captured |
| `engine.py` per-generation reset (`sm._free_slots = deque(...)`) | **every generate** -- the one that actually mattered |

The last one rebuilt `range(sm.num_slots)` excluding only the pad slot, so the
first three edits appeared to do nothing.

Verified after the fix -- the recurrence advances, and both hybrids produce
coherent text at tp=2 in compiled mode:

```
PREFILL  slot=[1]  recurrent 0 -> amax 1.892 rms 0.01214
DECODE 1 slot=[1]  1.892/0.01214 -> 1.489/0.01114
DECODE 2 slot=[1]  1.489/0.01114 -> 1.184/0.009776

Kimi : '1. Mercury - the smallest planet and closest to the Sun. 2. Venus - the
        hottest planet due to its thick, toxic atmosphere. 3. Earth - ...'
Qwen : '1. **Mercury** - Closest to the Sun, Mercury has the shortest year of any
        planet, orbiting the Sun in just 88 Earth days. 2. **Venus** - ...'
```

**Two dead ends recorded so they are not retried.** The
`ssm_state_indices` int32-vs-int64 width: irrelevant, the two are bit-identical
above; it only looked significant because NaN argmax is non-deterministic. And the
`[V, K]` vs `[K, V]` state-layout theory from §12: dead -- transposing the stored
state was a no-op because the kernel was not reading or writing it at all.

**Generalization.** Any vLLM/FLA kernel taking a state or block index may treat 0
as null. Our allocators must reserve index 0 globally rather than relying on
per-call-site sentinels. `MambaStateManager` (mamba1/2, line ~148) still hands out
slot 0 and should get the same treatment before those models are trusted.

### 17g. Results after the slot-0 fix

`validate` run `20260730-032546`, all three scenarios PASS:

| model | mixed speed | mixed match | long-ctx speed | long-ctx match |
|---|---|---|---|---|
| gpt-oss-120b | 0.88x | 64.5 / 385 | 0.92x | 84.9 / 256 |
| Qwen3-Next-80B | 1.19x | 17.3 / 392 | 0.75x | 20.2 / 256 |
| Kimi-Linear-48B | 1.31x | 59.1 / 389 | 0.88x | 104.6 / 256 |

Kimi's matched tokens went 5.1 -> 59.1 (mixed) and 5.2 -> 104.6 (long-context);
Qwen3-Next went from incoherent to 17.3 / 20.2. For reference, kb_nano's B200
reproduction recorded Kimi at 4.8 and Qwen3-Next at 26.1 against vLLM 0.18, so
Kimi is now an order of magnitude past its historical figure.

Against the current targets (>= 0.95x speed both scenarios; >= 50 matched tokens
mixed, >= 20 long-context) four gaps remain:

1. **gpt-oss speed** 0.88x / 0.92x. Match is comfortable. Post-MoE-port profile
   was ~37% MoE, ~15% in three small per-layer cuBLAS GEMMs (qkv / o / router),
   ~6% routing. Its CUDA-graph buckets already match vLLM's, and it already runs
   vLLM's trtllm MXFP4 MoE kernel.
2. **Qwen3-Next mixed match** 17.3 vs 50. Note kb_nano's analysis of exactly this
   number: it is batch-occupancy-dependent greedy divergence ("not garbage
   output" -- fluent, semantically equivalent completions that differ in wording),
   and it falls monotonically with position in the request queue. So this is a
   numerical-tightness problem at concurrency, not a broken kernel, and it will
   not move much without addressing the attention path's batch-composition
   sensitivity.
3. **Long-context speed across the board** -- 0.92x / 0.75x / 0.88x. This is the
   prefill-heavy scenario (64 sequences, long prompts), and it is the one place
   the trtllm BF16 MoE's large-token advantage (2.15x at 8192 tokens for
   Qwen3-Next) is *not* showing up end to end. Worth profiling before changing
   anything: the win may be masked by the GDN/KDA prefill chunk kernels or by
   chunked-prefill scheduling rather than the MoE.
4. **Qwen3-Next long-context 0.75x** specifically is the largest single speed gap
   and regressed relative to its mixed 1.19x, which points at its full-attention
   layers under long context (trtllm-gen used unconditionally, where vLLM's
   FLASHINFER backend switches to its wrapper above 256 tokens -- see §17).

### 17h. Long-context prefill profile (Qwen3-Next, the 0.75x case)

One 32768-token prefill, tp=2, 355.1 ms total CUDA self time:

| kernel | ms | % | calls |
|---|---|---|---|
| FlashInfer GDN prefill (Blackwell CUTLASS `gated_delta_net`) | 58.76 | 16.5 | 72 |
| trtllm MoE `bmm` gemm1 + gemm2 | 66.32 | 18.6 | 192 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 45.63 | 12.9 | 290 |
| trtllm full-attn `fmhaSm100fKernel ... H256PagedKvCausal` | 37.36 | 10.5 | 24 |
| `at::native::elementwise_kernel` | 33.41 | 9.4 | **996** |
| `nvjet_sm100_tst_128x256` (dense GEMMs) | 30.54 | 8.6 | 288 |
| moe finalize | 14.95 | 4.2 | 96 |
| RMSNorm (inductor `triton_red_fused_...rsqrt`) | 10.04 | 2.8 | 363 |
| `_causal_conv1d_fwd_kernel` | 7.46 | 2.1 | 72 |

Reading it: the MoE is now trtllm-gen as intended (18.6% in the two bmms, 4.2% in
finalize), and the GDN prefill does use a Blackwell CUTLASS kernel -- so kb_nano's
defect 8 (FlashInfer's GDN prefill being SM90-only) no longer applies to this
FlashInfer version.

Closing 0.75x -> 0.95x needs roughly 21%. The two candidates that could supply it:

* **996 elementwise launches, 9.4%.** These are temporaries in the GDN prefill
  plumbing: `_flashinfer_gdn_prefill` alone does `q/k/v .squeeze(0).contiguous()`,
  `torch.exp(g.contiguous().to(float32))`, `beta.contiguous().to(float32)` and
  `initial_state.to(torch.float32)`, per GDN layer per chunk (36 layers x 3
  chunks). The `initial_state` cast is unconditional: our Qwen3-Next recurrent
  state is allocated in model dtype (bf16), matching vLLM
  (`gated_delta_net_state_dtype` -> `_mamba_state_dtype` -> `conv_state_dtype` ->
  model dtype when `mamba_cache_dtype="auto"`), so we cast bf16 -> fp32 on every
  layer. Check what vLLM's own FlashInfer GDN call passes before assuming the cast
  is required.
* **290 NCCL all-reduces, 12.9%.** ~2 per layer per chunk (48 layers x 2 x 3 =
  288), 0.157 ms each. vLLM at tp=2 issues the same number, so this is probably
  *not* a gap by itself -- but vLLM sets `fuse_allreduce_rms: True` in its inductor
  `pass_config`, folding the following RMSNorm into the all-reduce. Our RMSNorm is
  a separate 10.04 ms / 363 launches. That fusion is the alignment item, worth
  ~2.8% plus launch overhead, not the 12.9%.

Note also `nvjet_sm100_tst_*` at 8.6% over 288 calls -- the small per-layer dense
GEMMs (in_proj/out_proj/router), the same class of overhead identified for gpt-oss.

Caveat on interpretation: none of the above is yet established as a *gap versus
vLLM*. The percentages are of our own runtime; a fair comparison needs the same
profile from the vLLM side before any of these is treated as recoverable.

### 17i. Three long-context suspects eliminated; scheduling is what is left

Checking each candidate from §17h against vLLM rather than assuming:

* **The 996 elementwise launches (9.4%) are not a gap.** vLLM's
  `fi_chunk_gated_delta_rule` does the *identical* work per GDN layer:
  `l2norm_fwd(q)`, `l2norm_fwd(k)`, `q/k/v.squeeze(0).contiguous()`,
  `g.squeeze(0).contiguous()`, `beta.squeeze(0).contiguous()`,
  `initial_state.to(torch.float32)`, `g.to(torch.float32)`,
  `beta.to(torch.float32)`, `torch.exp(fi_g)`. Same temporaries, same casts.
* **The GDN prefill kernel (16.5%) is not a gap.** vLLM's
  `_resolve_gdn_prefill_backend` picks FlashInfer on Blackwell when
  `head_k_dim == 128` and CUDA runtime >= 13, which holds here, and the reference
  run's log confirms it: `Using FlashInfer GDN prefill kernel (requested=auto,
  head_k_dim=128)`. Same kernel as ours.
* **The full-attention backend is not a gap, and §17's claim about it was wrong.**
  §17 said vLLM's FLASHINFER backend "uses its wrapper for large batches and
  trtllm-gen only when `num_tokens <= 256`". That threshold is **decode-only**.
  `use_trtllm_attention`'s prefill branch is simply
  `use_trtllm = kv_cache_dtype == "auto"` -- always trtllm-gen for prefill. And
  long-context runs 64 sequences, so its decode batch is under 256 and takes
  trtllm-gen on both sides too. Our unconditional use of trtllm-gen matches vLLM
  for both phases of this scenario.

So the long-context deficit is not in the kernels the profile ranks highest -- we
run the same kernels vLLM does for the GDN core, the MoE, and attention. What is
left, in order of how much of the goal they'd address:

1. **Chunked-prefill scheduling.** Both engines agree on
   `max_num_batched_tokens=16384`, but agreeing on the budget is not the same as
   agreeing on the policy: how prefill chunks are batched against decodes, whether
   a step mixes them, and how many chunks a long prompt is split into. The
   profile's 3x multiplier on per-layer call counts (72 GDN calls for 36 layers,
   288 all-reduces for 48 layers x 2) says we ran 3 chunks for 32768 tokens, i.e.
   ~10922 tokens per chunk rather than the full 16384 budget. Worth confirming
   against vLLM's chunk count for the same prompt.
2. **`fuse_allreduce_rms`.** vLLM sets this in its inductor `pass_config`; our
   RMSNorm is a separate 10.04 ms / 363 launches on top of the all-reduces.
3. **The small per-layer dense GEMMs** (`nvjet_sm100_*`, 8.6%, 288 calls) -- the
   same overhead class as gpt-oss's residual gap, and the one place a
   `combo_kernels`-style fusion (which vLLM enables in its inductor config) would
   apply.

### 17j. Correction: chunked-prefill scheduling matches too

§17i's lever 1 was based on a miscount and is wrong. Re-deriving the pass count
from the profile: 72 GDN calls / 36 GDN layers = **2** forward passes, and 24
full-attention fmha calls / 12 full-attn layers = **2**, and the MoE's 192 bmm
calls / 48 layers = 4 = 2 kernels x 2 passes. So a 32768-token prompt was prefilled
in **2 chunks of 16384**, exactly filling `max_num_batched_tokens`. The "3 chunks
of ~10922" reading came from dividing the 290 all-reduces by 48x2, which does not
work because the profile also covers the two decode steps and because a subset of
all-reduces take the custom kernel (`cross_device_reduce_1stage`, 145 calls)
rather than NCCL.

vLLM's side agrees: `enable_chunked_prefill=True`,
`max_num_batched_tokens=16384`, `max_num_seqs=1024` -- identical to ours. And its
concurrent-partial-prefill path is not active: `long_prefill_token_threshold`
becomes `int(max_model_len * 0.04)` only when `max_num_partial_prefills > 1`,
which defaults to 1, and the reference log never prints "Concurrent partial
prefills enabled".

So chunked prefill, the token budget, `max_num_seqs`, the GDN prefill kernel, the
MoE kernel and the attention backend all match vLLM for this scenario. That is
most of what the profile's mass sits in, and it means the long-context deficit is
**not yet localized**. The honest state: we know what it is *not*.

Still-unexamined candidates, and what each would take:

* **`fuse_allreduce_rms`** (vLLM's inductor `pass_config`) and `combo_kernels` --
  compilation-level fusions we do not enable. Together these target the 10.04 ms
  of separate RMSNorm launches and some of the 8.6% in small per-layer dense
  GEMMs. This is the only *confirmed* configuration difference remaining.
* **A vLLM-side profile of the same workload.** Everything above compares kernel
  *choice*; none of it compares kernel *time*. If vLLM runs the same kernels and is
  still 1.33x faster on this scenario, the difference is in overlap/occupancy or in
  per-step host overhead, and only a side-by-side timeline will show it.

Do the vLLM-side profile before writing any more kernel code for this row.

### 17k. Compilation config: only `fuse_allreduce_rms` differs

Checked our inductor config against vLLM's `inductor_compile_config` /
`pass_config`:

* `combo_kernels` and `benchmark_combo_kernel` -- **already enabled** here for
  torch >= 2.9 (`infra/compilation.py`), matching vLLM. So §17i's lever 3 is
  already aligned; the small per-layer GEMMs are already eligible for combo
  fusion.
* `size_asserts` / `alignment_asserts` / `scalar_asserts` -- matched (off unless
  debug logging).
* **`fuse_allreduce_rms` -- not implemented here.** vLLM sets it in its
  `pass_config` and supplies a custom FX pass that folds the RMSNorm following an
  all-reduce into it. We have no equivalent pass.

Sizing that gap honestly: RMSNorm is 10.04 ms of 355.1 ms (2.8%) in the
Qwen3-Next long-context prefill, plus whatever launch overhead the 363 launches
cost. **That cannot close Qwen3-Next's 0.75x -> 0.95x gap, which needs ~21%.** It
is roughly the right size for gpt-oss's 0.92x -> 0.95x long-context gap, and is
worth doing on alignment grounds regardless, but it should not be sold as the fix
for the hybrids.

Which leaves the side-by-side vLLM profile as the only outstanding way to localize
the hybrid long-context deficit. Every kernel choice matches; the remaining
possibilities are per-kernel time, overlap/occupancy, or per-step host overhead --
none of which our own profile can distinguish.

### 17l. Phase timing, because the vLLM profile is not obtainable that way

`torch.profiler` cannot see vLLM V1's kernels: its model runs in a separate engine
thread, so a main-thread profiler reports `total CUDA self 0.0 ms` even with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. vLLM's own profiler writes a trace to
`VLLM_TORCH_PROFILER_DIR`, which would need trace parsing to line up against our
`key_averages()` table.

Wall clock answers the question that actually matters next -- *which phase* the
long-context deficit is in -- without either. `/tmp/fkdev/phase_timing.py`,
same prompt ids both sides, best of 3:

  prefill+1tok    -> prefill cost
  prefill+65tok   -> decode cost by subtraction over 64 steps

Ours (Qwen3-Next, 32768 tokens, tp=2):

| phase | time |
|---|---|
| prefill + 1 token | 379.2 ms |
| prefill + 65 tokens | 757.6 ms |
| 64 decode steps (by subtraction) | 378.4 ms (**5.91 ms/step**) |

Note the split is almost exactly 50/50 at this prompt length, which invalidates an
assumption carried through §17h-17k: long-context was treated as
prefill-dominated, and at 32768 tokens with 256 output tokens it is not. The
`long_context` scenario generates 256 tokens per request, so decode is ~4x the 64
steps measured here -- i.e. **decode dominates that scenario**, and the prefill
profile in §17h was ranking the wrong half.

That reframes the remaining candidates: per-step decode cost (5.91 ms/step for one
sequence at tp=2 is high) and the decode-side kernels, not the prefill kernels
whose choice §17i verified.

### 17m. The long-context gap is in DECODE, not prefill

Same prompt ids, best of 3, Qwen3-Next 32768 tokens tp=2:

| phase | ours | vLLM | ratio |
|---|---|---|---|
| prefill (+1 token) | 379.2 ms | 320.1 ms | 0.84x |
| 64 decode steps | 378.4 ms | 237.5 ms | **0.63x** |
| per decode step | **5.91 ms** | **3.71 ms** | **0.63x** |

Decode is 0.63x; prefill is 0.84x. The `long_context` scenario emits 256 tokens
per request, so decode carries most of its wall clock -- which makes 0.63x on
decode the dominant term behind the 0.75x scenario number, and means §17h's prefill
profile was ranking the wrong half of the workload.

Per layer: 5.91 ms / 48 layers = 0.123 ms/layer for us, 0.077 ms/layer for vLLM.
2.2 ms/step to find.

Where to look, and one measurement already in hand that constrains it: from
`/tmp/fkdev/bf16_moe_graph.py`, the trtllm BF16 MoE at 8 tokens costs 0.051 ms
under CUDA-graph replay but **0.716 ms eagerly** -- a 14x host-side launch
penalty. 48 layers x 0.7 ms would be ~34 ms/step, far more than the 5.91 ms
observed, so the decode path is certainly graph-captured overall. But the same
measurement says the MoE alone is ~48 x 0.051 = 2.4 ms of our 5.91 ms step, and
vLLM's *entire* step is 3.71 ms. So either vLLM's MoE is materially cheaper at
batch 1, or part of our step is not in the graph.

Two concrete checks, in order:

1. **Confirm the decode graph is actually replayed at batch 1** for Qwen3-Next
   rather than falling back to eager -- and that the whole step (model forward +
   lm_head + greedy argmax) is inside it, not just the backbone.
   `capture_kimi_cudagraph` captures bs=1, but §17f showed the free-slot pool is
   rebuilt in three places, so verifying the replay path is used is cheap and worth
   doing before optimizing anything.
2. **Time the MoE at batch 1 on both sides.** If vLLM's per-layer MoE at bs=1 is
   well under 0.051 ms, its `tune_max_num_tokens` / autotuned config differs from
   ours (we pass `DEFAULT_TUNE_MAX_NUM_TOKENS = 8192`, vLLM computes
   `max(max_num_tokens * dp_size, 8192)`), and the decode-shape tactic it selects
   is the thing to copy.

Superseded by this: §17k's framing of `fuse_allreduce_rms` as the main remaining
lever. At 2.8% of a *prefill* profile it was already too small, and prefill is not
where the deficit is.

### 17n. The decode gap is inside the CUDA graph, not host-side

Timing the two halves of `ModelRunner._run_kimi_decode_graph_greedy` separately
(`/tmp/fkdev/decode_step_split.py`; note both methods live on `ModelRunner`, not
`LlamaEngine`):

| part | per step |
|---|---|
| `_replay_kimi_decode_graph_greedy` | **5.812 ms** |
| `_stage_kimi_decode_graph_inputs` | 0.004 ms |

Replay is the whole step. Per-step host work -- metadata construction, H2D
staging -- is 4 microseconds and irrelevant. So the 5.91 vs 3.71 ms/step deficit
(§17m) is **kernel time inside the captured graph**, and the "move host work into
the graph or off the critical path" option is dead.

What that leaves, with the arithmetic that makes it uncomfortable: our MoE should
be ~48 x 0.051 = 2.4 ms of the 5.81 ms step (per-layer figure from
`bf16_moe_graph.py` under graph replay at 8 tokens), and vLLM's *entire* step is
3.71 ms. Either

* vLLM's MoE is materially cheaper than 0.051 ms/layer at batch 1 -- in which case
  the autotuned tactic differs and `tune_max_num_tokens` is the lever (we hardcode
  `DEFAULT_TUNE_MAX_NUM_TOKENS = 8192`; vLLM computes
  `max(max_num_tokens * dp_size, 8192)`, which for this config is 16384, so it
  tunes over a different bucket set); or
* the 0.051 ms/layer figure does not transfer from the synthetic isolated call to
  the real graph, and the cost is elsewhere in the step (36 GDN decode layers, 12
  trtllm full-attn decode layers, the per-layer all-reduces, the norms).

The measurement that separates those is a kernel-level profile of the *decode*
phase specifically -- `torch.profiler` does observe kernels inside CUDA-graph
replays, so ranking the decode step's kernels the way §17h ranked prefill's is
straightforward and is the next step. Do that before touching
`tune_max_num_tokens` or any kernel: §17h ranked the wrong phase and §17i-17k
chased three suspects that all turned out to match vLLM exactly.

### 17o. Decode kernel ranking, and one real find: a redundant all-reduce per layer

Qwen3-Next, plen=512, 32 decode steps, 226.6 ms total CUDA self:

| kernel | ms | % | calls | ~per step |
|---|---|---|---|---|
| `cross_device_reduce_1stage` | 42.69 | **18.8** | 4785 | **~150** |
| `elementwise_kernel` | 17.90 | 7.9 | 8436 | ~264 |
| `nvjet_sm100_tst_16x64...` | 16.51 | 7.3 | 3072 | |
| MoE `bmm` (t128x8x128, 2 variants) | 25.00 | 11.0 | 3120 | |
| `nvjet_sm100_tst_32x64...splitK` | 12.17 | 5.4 | 3120 | |
| `cublasLt::splitKreduce_kernel` | 11.87 | 5.2 | 4308 | |
| `nvjet_sm100_tst_64x8...` | 10.46 | 4.6 | 1536 | |
| RMSNorm (inductor) | 8.27 | 3.7 | 3993 | |
| moe routing / finalize | 13.81 | 6.1 | 3072 | |
| trtllm full-attn decode | 6.35 | 2.8 | 384 | |
| `fused_sigmoid_gating_delta_rule_update` | 4.17 | 1.8 | 1152 | |
| `_causal_conv1d_update` | 3.07 | 1.4 | 1152 | |

**All-reduce is the largest single item, and the count is wrong.** ~150 per step
over 48 layers is ~3 per layer, where 2 is the minimum (attention/GDN `o_proj`,
plus the MoE). The third came from `SharedExpertMoE`: the shared expert's
`down_proj` is a `RowParallelLinear` that all-reduced its own output, and the
routed-expert output was all-reduced separately, and only then were the two added.

vLLM adds the per-rank partials first and all-reduces the sum once. Fixed the same
way: `_TPSwiGLUMLP` takes `reduce_results`, the shared expert defers its reduce at
tp>1, and `forward` folds both partials before a single `AllReduce`. The transform
is exact -- both terms are per-rank partial sums over the same output space -- and
the `shared_expert_gate` multiply stays exact too, since the gate is computed from
replicated `hidden_states` so the same scalar scales every rank's partial.

Verified: Qwen3-Next's 48-token greedy continuation is **byte-identical** before
and after, which is what an exact transform should produce. Expected saving is one
of three all-reduces per layer, ~6% of decode -- real, but not on its own enough
for 0.75x -> 0.95x.

Not yet acted on, from the same table: 8436 elementwise launches (7.9%) and three
`nvjet_*` variants totalling 17.3% over 7728 calls plus `splitKreduce` at 5.2% --
the small per-layer GEMMs at batch 1, which is the same overhead class as gpt-oss's
residual gap. `splitK` appearing at batch 1 suggests cuBLAS is picking split-K
tactics for very thin GEMMs, where vLLM's `linear_backend='auto'` may resolve
differently.

### 17p. All-reduce fold: measured +3.7% on decode, not the projected 6%

Same harness, same prompt, after the §17o change:

| | before | after | vLLM |
|---|---|---|---|
| prefill (32768 tok) | 379.2 ms | 367.9 ms | 320.1 ms |
| per decode step | 5.91 ms | **5.69 ms** | 3.71 ms |

3.7% on decode, 3.0% on prefill -- **below the ~6% projected** from "one of three
all-reduces per layer". The projection over-counted: not every layer's all-reduce
sits on the critical path the same way, and the shared-expert reduce is one of the
cheaper ones (its output is `hidden_size`-wide like the others, but it overlapped
partially with the routed path's work). Recording the miss because the estimate was
stated as if it followed arithmetically, and it did not.

Where that leaves the row: 0.63x -> 0.65x on decode. 2.0 ms/step still to find
against vLLM's 3.71 ms.

The next block is the small per-layer GEMMs, which the §17o table puts at 22.5% of
decode across `nvjet_sm100_tst_16x64` (7.3%), `nvjet_sm100_tst_32x64...splitK`
(5.4%), `nvjet_sm100_tst_64x8` (4.6%) and `cublasLt::splitKreduce_kernel` (5.2%).
Two things stand out:

* **`splitK` tactics at batch 1.** Split-K is for GEMMs with a large reduction
  dimension and too few tiles to fill the GPU; at batch 1 it adds a separate
  reduction kernel (`splitKreduce`, 4308 calls) for very thin work. That the
  autotuner picks it suggests the shapes are being presented in a form cuBLAS
  reads as under-occupied.
* **vLLM's `linear_backend='auto'`** (in its `KernelConfig`) may resolve these to a
  different implementation entirely. Worth checking what it selects for
  `[1, hidden] x [hidden, N]` decode shapes before assuming cuBLAS is the
  comparison.

### 17q. `linear_backend` does not apply; the GEMM block is not a backend mismatch

`KernelConfig.linear_backend` is documented as "Backend for **quantized** linear
layer GEMM kernels", and its consumers are the quantization paths
(`model_executor/kernels/linear/`,
`quantization/compressed_tensors/compressed_tensors.py`). Qwen3-Next is
unquantized bf16, so vLLM's dense linears go through `F.linear`/cuBLAS exactly as
ours do. §17p's lead (b) is dead.

Nor is the *count* obviously wrong. Per decode step: `nvjet_16x64` 96,
`nvjet_32x64...splitK` ~98, `nvjet_64x8` 48, `gemv2N` 48, `splitKreduce` ~135.
Roughly 6 GEMM-class launches per layer over 48 layers, which is what
Qwen3-Next's per-layer linear set implies (in_proj_qkvz, in_proj_ba, out_proj or
qkv/o, router, shared-expert gate_up and down). vLLM issues a comparable set.

So the GEMM block is per-kernel *time* at batch 1, on the same library, at
comparable call counts -- which does not point at anything we choose. The
`splitKreduce` count (~135/step, more than one per GEMM) is still the odd feature
and the only concrete thread left here: split-K on batch-1 GEMMs is a poor tactic,
and if vLLM's identical shapes avoid it, the difference is in how the operands are
laid out (contiguity/transpose flags) rather than in which library is called.

Running total of eliminated explanations for the long-context gap: chunked-prefill
policy, token budget, `max_num_seqs`, CUDA-graph bucket shape, GDN prefill kernel,
prefill attention backend, decode attention backend, MoE kernel choice, GDN
elementwise/cast overhead, host-side per-step work, `combo_kernels`, and
`linear_backend`. Landed: the redundant per-layer all-reduce (+3.7% decode).

### 17r. A productive search direction: find where vLLM already optimized something

Both speed wins so far came from the same place -- vLLM having already optimized a
construct that we ported in its *pre-optimization* form. That is a better search
strategy than profiling for hot kernels, because §17i-17q established the kernels
themselves match; the gap is in the plumbing around them, and vLLM's source
comments mark where it has been tuned.

Find 2: `Qwen3NextGatedDeltaNet.rearrange_mixed_qkv`, whose docstring says it
outright:

> The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)`` followed by
> ``.contiguous()`` on each tensor. This version flattens all three splits into a
> single buffer via ``torch.cat`` so that torch.compile emits one Triton copy
> kernel instead of three separate contiguous() calls.

We were doing exactly what it moved away from: three `.contiguous()` calls per GDN
layer, i.e. 108 copy launches per decode step at 36 GDN layers rather than 36. The
decode profile's 8436 elementwise launches over 32 steps (7.9%) is consistent with
that.

Ported as `_rearrange_mixed_qkv`: split, `torch.cat` the three flattened pieces
into one buffer, slice contiguous views back out. Operands are contiguous by
construction afterwards, so the six now-redundant `.contiguous()` calls at the
prefill and decode call sites were dropped too. Qwen3-Next's 48-token greedy
continuation is byte-identical, as an exact reshaping should be.

Worth sweeping the rest of vLLM's hybrid path for the same pattern -- docstrings
and comments that describe a prior slower spelling are a direct index of places we
may still hold the slower one.

### 17s. The qkv-rearrange port produced no measurable win

| | before | after | vLLM |
|---|---|---|---|
| prefill | 367.9 ms | 381.2 ms | 320.1 ms |
| per decode step | 5.69 ms | **5.68 ms** | 3.71 ms |

0.2% on decode -- indistinguishable from noise, and prefill moved the wrong way by
about the same margin, which is the run-to-run spread of this harness. So §17r's
reasoning was sound and its conclusion was wrong: vLLM's own comment says the
three-`contiguous()` spelling costs extra copy kernels, and it presumably does in
vLLM's setting, but in ours it does not show up. The likely reason is that our
decode path is inside a CUDA graph and compiled by inductor, which already fuses or
amortizes those copies -- the profile's elementwise launches are real but cheap.

The change is kept: it is exactly what vLLM does, it is provably exact (byte-identical
output), and it removes six redundant `.contiguous()` calls. But it is not a
speed fix, and §17r's framing of "find where vLLM already optimized something" as a
*productive* search direction is not supported -- one such find gave 3.7% (the
all-reduce, §17p) and the second gave nothing.

Two remaining candidates from that sweep, with expected value now marked down:

* `qwen_gdn_linear_attn.py:688` -- the same cat-then-slice treatment applied to the
  `mixed_qkv/z/b/a` unpack, where we use `_unpack_qkvz` with per-tensor
  `reshape`. Same class as the change that just failed to move anything.
* `maybe_disable_tp` (line 541) -- **not applicable**: it replicates `ba_proj` only
  for Marlin/AWQ/GPTQ/INC quantized configs, to keep
  `output_size_per_partition >= 64`. Qwen3-Next here is unquantized bf16 and uses
  the interleaved layout, which vLLM leaves TP-sharded, same as us.

Honest read of the row: decode is 5.68 vs 3.71 ms/step (0.65x) after two exact
alignment fixes worth 3.9% combined. Twelve structural explanations have been
eliminated (§17q) and the two remaining plumbing candidates are of the class that
just produced nothing. The gap is most likely diffuse per-kernel efficiency at
batch 1 across ~290 launches per step, which is not reachable by porting
individual constructs.

### 17t. Correction: the decode time is concentrated, not diffuse

§17s concluded the residual gap is "diffuse per-kernel efficiency across ~290
launches per step" and therefore not reachable by targeted work. The profile
contradicts that:

```
top 20 kernels = 201.8 of 226.6 ms = 89.1% of decode
top  6 kernels = 114.3 ms          = 50.4%
```

Six kernels are half the decode time. That is concentrated, and it means the row is
tractable after all -- the conclusion was pessimism, not measurement, and it was
based on the launch *count* rather than the time distribution.

Half of decode, by block, after the all-reduce fold:

| block | % of decode |
|---|---|
| small per-layer GEMMs (`nvjet_*` x3 + `cublasLt::splitKreduce`) | 22.5 |
| MoE (`bmm` x2 + routing + finalize) | 17.1 |
| all-reduce | ~12.5 (was 18.8) |
| elementwise | 7.9 |

The GEMM block is the largest and the least explained. One hypothesis not yet
tested, and the reason to test it first: `nvjet_sm100_tst_16x64` runs 96 times per
step for a **single-sequence** decode. If the CUDA-graph replay were padding to a
bucket larger than 1, every GEMM, all-reduce and MoE call in the step would be
doing more work than the batch requires -- which would explain a ~2x gap directly
and would not show up in any *count*-based comparison, only in the shapes. The
bucket list starts `[1, 2, 4]` so bs=1 exists, and `_kimi_graph_bs_for_n[1]` should
select it, but that has never been verified; the tile geometry in the kernel name is
weak evidence either way since cuBLAS pads tiles for thin GEMMs regardless.

Next: print the bucket `_kimi_graph_bs_for_n` selects for n=1 and the actual M
dimension reaching the per-layer GEMMs, and compare against vLLM's decode shapes.
Cheap, and it either finds a 2x or definitively rules out the one explanation that
could account for the whole gap at once.

### 17u. Graph bucket padding ruled out

```
DECODE n=1 -> bucket 1 (bs_list head [1, 2, 4, 8, 16, 24])
```

A single-sequence decode replays the bs=1 graph. No padding, so every GEMM,
all-reduce and MoE call in the step is already sized to the batch. (The
`RowParallelLinear input shape (1024, 2048)` lines in that probe are graph
*capture* at the largest bucket, emitted before any decode replay -- not decode
shapes.)

That eliminates the last single explanation that could have accounted for the whole
2 ms/step at once. Where the row stands, stated plainly:

* decode 5.68 ms/step vs vLLM's 3.71 (0.65x); prefill ~370 vs 320 ms (0.86x)
* time is **concentrated** (top 6 kernels = 50%, top 20 = 89%), so it is targetable
* the same kernels, counts, shapes, scheduling policy, token budget, graph buckets
  and compilation config as vLLM -- twelve structural explanations eliminated (§17q)
  plus bucket padding here
* two exact fixes landed, worth 3.9% of decode combined (§17p, §17s)

So we run what vLLM runs, at the sizes vLLM runs it, and are still 1.5x slower per
decode step. The remaining difference has to be in per-kernel execution rather than
in anything selected at the Python level: kernel-launch dependency structure inside
the graph, occupancy, or the specific cuBLAS/trtllm tactics chosen for these shapes.

The three highest-value things left, in order:

1. **Compare per-kernel times directly against vLLM for the same shapes.** Not
   kernel choice -- time. vLLM's profiler writes to `VLLM_TORCH_PROFILER_DIR`;
   parsing that trace and joining on kernel name against our `key_averages()` table
   would say whether, e.g., our `nvjet_16x64` at 96 calls/step is slower per call
   than vLLM's, or whether vLLM simply issues fewer of them. Every comparison so far
   has stopped short of this and it is the only way to distinguish "same kernel,
   slower" from "fewer kernels".
2. **The MoE block at 17.1%** with `tune_max_num_tokens` still unaligned (we
   hardcode 8192; vLLM computes `max(max_num_tokens * dp_size, 8192)` = 16384 here,
   a different tuning bucket set, so a different tactic may be selected for decode
   shapes).
3. **`fuse_allreduce_rms`** (§17k) -- small, but now that all-reduce is ~12.5% of
   decode rather than 2.8% of prefill, folding the following RMSNorm in is worth
   more than it was when first sized.

### 17v. ncu: the decode GEMMs are latency-bound, not tactic-bound

Profiling the batch-1 decode GEMMs *standalone* (`/tmp/fkdev/gemm_shapes.py` --
the shapes are fully determined by the config, so no 80B load is needed; runs in
seconds instead of minutes). Biggest kernel, `nvjet_sm100_tst_16x64_64x16_4x1`,
shape `[1, 2048] x [2048, 6144]` (`in_proj_qkvz` at tp=2):

```
grid = 128 blocks    duration = 8.6 us
SM throughput   =  4.4 %
warps active    = 10.0 %
DRAM throughput =  0.02 %   <- artifact, see below
```

128 blocks against 148 SMs, 4.4% SM throughput, 10% occupancy. These GEMMs are
**latency-bound**: at batch 1 there is almost no arithmetic, and the ~8.6 us is
essentially fixed per-kernel cost. At ~290 launches per decode step that is ~2.5 ms,
the order of the entire 5.68 ms step.

That retires the split-K theory from §17q. cuBLAS is not picking a bad tactic;
there is nothing for any tactic to do. It also means the remedy is **fewer, larger
kernels**, not a different GEMM implementation -- which is a different kind of
change from everything attempted so far, and the first one the evidence actually
points at.

**A caveat of my own making:** `--cache-control none` was passed to make the run
fast, so weights stayed resident in L2 across replays. A 25 MB weight matrix read
in 8.6 us would be ~36% of HBM peak, so the 0.02% DRAM figure is an artifact, and
the duration is optimistic. Occupancy and SM throughput are unaffected. Re-run with
default cache control before trusting anything bandwidth-related.

**The concrete fusion opportunity**, and why it is not done here: each GDN layer
issues `in_proj_qkvz` (`[2048, 6144]`) and `in_proj_ba` (`[2048, 32]`) as two
separate GEMMs over the *same* input. The 32-wide one is pure overhead -- a full
kernel launch for 32 output values. Concatenating the two weights into one
`[2048, 6176]` linear and splitting the output would remove 36 launches per step.
It requires merging two different sharding schemes though: `in_proj_qkvz` is a
plain `ColumnParallelLinear`, while `in_proj_ba` uses a custom loader that splits
`[2*num_v_heads]` at the midpoint and TP-shards each half independently
(deliberately, to match vLLM's `MergedColumnParallelLinear(output_sizes=[nv]*2)`
for token-level alignment). Getting that wrong silently corrupts routing, which is
the same class of failure as the slot-0 bug this session spent most of its length
finding. It should be done with the parity harness available to verify, not blind.

Note also this would be a *deviation* from vLLM, which keeps the two projections
separate. Justified on speed grounds if verified, but it is no longer "match vLLM".

### 17w. Fusing in_proj_qkvz + in_proj_ba, without merging the sharding schemes

§17v flagged this fusion as risky because `in_proj_ba` needs its own loader (it
splits `[2*num_v_heads]` at the midpoint and TP-shards each half independently, to
match vLLM's `MergedColumnParallelLinear(output_sizes=[nv]*2)`), and merging that
with `in_proj_qkvz`'s plain column-parallel sharding at declaration time could
silently mis-route tokens.

The risk was in the *merge*, not the *fusion*. Both loaders are left untouched;
`Qwen3NextGDNAttention.process_weights_after_loading` then concatenates the
**already-correctly-loaded** weights into one buffer, and `forward` does a single
`F.linear` and slices the output. Concatenating correct weights cannot produce
incorrect sharding. The two originals are re-pointed at slices of the fused buffer,
so nothing is duplicated and either module still works if called directly.

Why this shape specifically: `in_proj_ba` is `[2048, 32]` -- a full kernel launch
for **32 output values**, at the ~8.6 us fixed cost §17v measured. 36 GDN layers is
36 such launches per decode step, each doing essentially nothing. It is the
clearest instance of the latency-bound pattern ncu exposed.

Verified byte-identical output (third exact transform in a row, after §17o and
§17r). Hooked into the post-load pass in `weight_loader.py` alongside the MoE
classes.

**This is a deliberate deviation from vLLM**, which keeps the two projections
separate. Defensible on speed grounds given the ncu evidence, but it is no longer
strict parity and should be recorded as such rather than presented as alignment.

### 17x. Decode speed: cumulative result of the four changes

| change | decode ms/step | delta |
|---|---|---|
| baseline (§17m) | 5.91 | -- |
| all-reduce fold (§17o) | 5.69 | -3.7% |
| qkv rearrange port (§17r) | 5.68 | -0.2% |
| `tune_max_num_tokens` 8192 -> 16384 + in_proj fusion (§17w) | **5.57** | -1.9% |

Cumulative **-5.8%**, against vLLM's 3.71 ms/step. The row moves 0.63x -> 0.67x on
decode. All four changes verified byte-identical on output.

Scale of what remains: 5.57 -> 3.71 needs another **33%**. The ncu evidence (§17v)
says the decode step is launch-latency-bound -- ~290 launches at ~8.6 us of largely
fixed cost each -- so the remaining 33% is not in any one kernel. It would come from
substantially reducing the launch count, and the per-layer linears that are left
(`out_proj`, the router gate, the shared-expert pair, the full-attention qkv/o) are
already single GEMMs each; fusing *across* layers is not possible, and fusing the
remaining within-layer pairs would need them to share an input, which they do not.

That bounds what this line of work can deliver. The honest read is that closing
0.67x -> 0.95x on decode requires either (a) a fundamentally lower-launch-count
decode path -- e.g. a single fused per-layer kernel of the kind vLLM's
`splitting_ops` / custom-op design enables and our per-op composition does not, or
(b) accepting that our decode step issues more kernels than vLLM's for structural
reasons and finding where that structure differs, which none of the fourteen
comparisons in §17i-17u located.

Worth noting for whoever continues: vLLM's `compilation_config.splitting_ops` lists
15 custom ops it deliberately keeps *out* of the compiled graph
(`vllm::unified_attention_with_output`, `vllm::mamba_mixer2`,
`vllm::qwen_gdn_attention_core`, `vllm::kda_attention`, ...). Everything else gets
inductor-fused into piecewise graphs. If vLLM's per-layer linears are being fused by
inductor into larger kernels where ours are separate `F.linear` calls, that would
explain a launch-count gap directly, and it is checkable by counting kernels per
decode step on each side -- which no measurement this session has done.

### 17y. Reverted: the changes that did not measure

| change | measured | disposition |
|---|---|---|
| all-reduce fold (§17o) | **-3.7%** decode | **kept** -- and it matches vLLM |
| `tune_max_num_tokens` 8192 -> 16384 | alignment | **kept** -- vLLM computes 16384 here |
| qkv-rearrange port (§17r) | +0.2% (noise) | **reverted** |
| `in_proj_qkvz`+`in_proj_ba` fusion (§17w) | part of a 1.9% pair | **reverted** |

The fusion is reverted despite a nonzero pair measurement, for two reasons: its
contribution was never separated from the `tune_max_num_tokens` change measured
alongside it, and it is a deliberate deviation from vLLM (which keeps the two
projections separate). If the kernel evidence later justifies it, it can come back
with its own measurement.

Also removed: the `FASTKERNELS_KDA_STATE_T` diagnostic switch (its hypothesis is
dead -- §17f) and a stale comment claiming `fused_recurrent_kda` performs no state
update, which was the slot-0 bug rather than a property of the kernel.

### 17z. Kernel-level A/B at realistic shapes

`/tmp/fkdev/kernel_ab.py`. Every layer of a given type is identical, so each
distinct kernel is profiled **once** at the shape both engines actually see
(Qwen3-Next tp=2: hidden 2048, 8 k-heads / 16 v-heads of 128, 512 experts of 256,
full-attn 8 q-heads of 256; decode batch 1, prefill 16384-token chunk). No model
load, no whole-model replay.

Eager, batch 1:

| op | shape | us/call |
|---|---|---|
| `lin_gdn_qkvz` | `[1,2048]x[2048,6144]` | 10.55 |
| `lin_gdn_ba` | `[1,2048]x[2048,32]` | **13.43** |
| `lin_gdn_out` | `[1,2048]x[2048,2048]` | 10.40 |
| `lin_attn_qkv` | `[1,2048]x[2048,2560]` | 10.45 |
| `lin_attn_o` | `[1,2048]x[2048,2048]` | 13.32 |
| `lin_router` | `[1,2048]x[2048,512]` | 13.08 |
| `lin_shared_up` | `[1,2048]x[2048,512]` | 13.11 |
| `lin_shared_dn` | `[1,256]x[256,2048]` | 13.18 |
| `gdn_decode` | 1 token, 16 v-heads | 41.85 |
| `moe_decode_ours` | 1 token, 512 experts | 719.86 |

**A 32-output GEMM costs more than a 6144-output one** (13.43 vs 10.55 us). The
cost is entirely fixed per-launch and independent of the arithmetic -- the strongest
possible confirmation of §17v's latency-bound reading, and it means no choice of
GEMM tactic or library can help.

**And it settles the fastkernels-vs-vLLM question at this level:** for every op in
the table we and vLLM invoke the *same* kernel --
`fused_sigmoid_gating_delta_rule_update` for GDN decode (vLLM's own op),
`trtllm_bf16_moe` for the MoE (§14), and `F.linear`/cuBLAS for every dense
projection (§17q showed `linear_backend` applies only to quantized paths). Per-kernel
time is therefore identical by construction; there is nothing to win by swapping
implementations.

So the entire remaining decode gap must be in **how many kernels are issued per
step, and how they are scheduled** -- not in any kernel's speed. The 719.86 us eager
MoE against 51 us under graph replay (§17n) shows how large that effect is: a ~14x
swing from launch handling alone, on an identical kernel.

That makes one never-run measurement the whole question: **count kernels per decode
step on each side.** vLLM's `splitting_ops` keeps 15 custom ops out of the compiled
graph and lets inductor fuse everything else into piecewise graphs; if its per-layer
linears are fused where ours are separate `F.linear` calls, the launch counts differ
and that is the gap. Counting is cheap on both sides and no measurement this session
has done it.

### 18. Validate after the decode fixes, and the launch-count measurement

Run `20260730-054258` (code state *before* the §17y reverts, i.e. with the qkv port
and in_proj fusion still in):

| model | mixed | long-context |
|---|---|---|
| gpt-oss-120b | 0.88x, 56.2 | 0.92x, 72.0 |
| Qwen3-Next-80B | 1.20x, 17.3 | **0.78x**, 20.0 |
| Kimi-Linear-48B | 1.31x, 55.2 | **0.89x**, 94.5 |

Still 10 of 12. Against `20260730-032546`: Qwen3-Next long-context 0.75 -> 0.78,
Kimi 0.88 -> 0.89 -- real but small, consistent with -5.8% on decode diluted by
prefill. Match moved 64.5 -> 56.2 (gpt-oss) and 59.1 -> 55.2 (Kimi) with no
correctness change in between, i.e. inside the +/-10-token noise band.

**Qwen3-Next long-context match is 20.0 against a >= 20 threshold.** That criterion
is passing on the boundary and could flip on noise; it should not be counted as
securely met.

#### The launch-count measurement, and two errors in my first attempt

The question §17z left as decisive -- do we issue more kernels per decode step than
vLLM -- is still unanswered. Method (`/tmp/fkdev/decode_count.py`): nsys rather than
torch profiler (which cannot see vLLM V1's kernels, §17m) or ncu (replay far too slow
for a whole step); and a *delta* over two runs differing only in output-token count,
so `(launches_long - launches_short) / n_steps` cancels prefill, warmup and capture.

Two mistakes to avoid on the retry:

1. **Traced with `-t cuda`, which excludes NVTX**, so the SHORT/LONG ranges were not
   recorded and the delta could not be taken. Needs `-t cuda,nvtx`.
2. **Falling back to comparing whole-trace totals is invalid.** Ours reports 694,510
   launches across 197 distinct kernels, but that is dominated by CUDA-graph
   *capture* -- we capture 83 buckets x 48 layers, vLLM captures 51 buckets -- not by
   the 19 decode steps. Capture counts differ by construction between the engines, so
   totals are not comparable. The range delta is required, not optional.

Also, GPU bookkeeping: the first attempt OOM'd because a validate run launched two
turns earlier was still occupying GPUs 2-5 and had not been tracked. Check
`nvidia-smi` before scheduling GPU work.

## 19. gpt-oss-120b factorial: the gap needs BOTH tp>1 and compiled

A 2x2x2 over {ours, vLLM} x {tp=1, tp=2} x {compiled, eager} on one model
(`/tmp/fkdev/gptoss_factorial.py`, 2048-token prompt, decode ms/step by
subtraction) localized the gap in a single run, after many turns of kernel-level
work had not. Decode ms/step:

| cell | ours | vLLM | ratio |
|---|---|---|---|
| tp=1, compiled | 2.527 | 2.789 | **1.10x (we are faster)** |
| tp=2, compiled | 3.026 | 2.291 | **0.76x** |
| tp=2, eager | 45.198 | 43.339 | 0.96x |

The gap requires **both** factors: we win at tp=1, and we are at parity in eager.
And our TP scaling is *inverted* -- tp=1 -> tp=2 costs us +20% while vLLM gains 18%
from the second GPU.

**Ignore the prefill column of that run.** vLLM showed 8.3/7.4 ms against our
46.5/55.9, which is not a real 6x: `enable_prefix_caching` was left at its default
and the timing loop repeats an identical prompt, so vLLM served the later
repetitions from cache. The validate harness sets `enable_prefix_caching=False`,
which is why validate reports 0.88x rather than 6x. Decode is unaffected (not
cached), so the decode conclusion stands.

### 19a. Cause: the compiled path bypassed the custom all-reduce

`AllReduce.forward` took `dist.all_reduce` unconditionally under
`torch.compiler.is_compiling()`:

```python
if torch.compiler.is_compiling():
    dist.all_reduce(tensor)      # NCCL, always
    return tensor
```

So every collective in the compiled decode graph went through NCCL, at one-token
message sizes (~5.7 KB for gpt-oss's 2880 hidden) where NCCL is far off its
efficient range and the custom one-shot IPC kernel exists precisely to win.
`should_custom_ar` would have accepted these buffers -- `max_size` is 8 MB, world
size 2, contiguous, 16-byte aligned -- so the guard was skipping a qualifying path.

This also explains a profile reading I got wrong earlier: the Qwen3-Next trace showed
290 NCCL all-reduces *and* 145 custom ones, and §17h took that as "large messages
correctly use NCCL". It was the compiled decode graph taking NCCL while eager regions
took the custom path.

Fix: register `fastkernels::custom_all_reduce` as a custom op with a meta impl, so
inductor treats it as opaque and the IPC pointer work happens at runtime rather than
being traced -- the same mechanism `compilation.py` already uses for
`kda_attention`, and what vLLM does. NCCL fallback retained inside the op for buffers
that do not qualify.

### 19b. Measured: -5.2%, and the anomaly is not closed

| cell | before | after |
|---|---|---|
| tp=2, compiled | 3.026 | **2.868** (0.76x -> 0.80x) |
| tp=1, compiled | 2.527 | 2.560 (noise, no regression) |
| tp=2, eager | 45.198 | 44.948 (unchanged, as designed) |

The TP penalty shrinks from +20% to +12% (2.560 -> 2.868) but does not invert:
vLLM still *gains* 18% from the second GPU where we lose 12%. So NCCL-under-compile
was a real contributor, not the whole cause.

What is left in the tp>1 path, given the collective kernel is now the right one:

1. **Collective count per layer.** gpt-oss issues attention-`o_proj` + MoE
   all-reduces = 2 per layer x 36 layers = 72 per decode step. Worth counting
   vLLM's for the same model -- §17z's method now works with
   `nsys --cuda-graph-trace=node` (ours 3121 vs vLLM 2590 launches/step on
   Qwen3-Next, a 20% excess, measured but never broken down by kind).
2. **Placement relative to compute.** vLLM sets `fuse_allreduce_rms` in its inductor
   `pass_config`, folding the following RMSNorm into the collective. Ours is a
   separate kernel after it.
3. **Overlap.** Whether either engine overlaps the collective with compute at all in
   the decode graph; an nsys timeline would show it directly and cheaply.

### 19c. Verified: the custom all-reduce is live in the compiled decode graph

gpt-oss-120b, tp=2, 32 decode steps, 99.9 ms total CUDA self:

```
cross_device_reduce_1stage   16.54 ms   16.6%   2409 calls
(no ncclDevKernel_AllReduce present)
```

NCCL is gone from the compiled decode path; every collective now takes the custom
IPC kernel. 2409 calls / 32 steps = ~75 per step, matching 36 layers x 2
(attention `o_proj` + MoE) plus a few. So §19a's fix is confirmed live, and the
-5.2% it bought is the whole of what switching kernels was worth.

The collective is still **16.6% of decode at 6.9 us per call**. That is the honest
size of the remaining tp>1 cost: the kernel is now the right one, so what is left is
count and placement, not implementation. vLLM issues a comparable number of
collectives at tp=2, which is why the next question is whether it overlaps them with
compute or folds them into the following norm (`fuse_allreduce_rms`), rather than
whether its collective kernel is faster.

## 20. The compiled tp>1 difference: vLLM fuses all-reduce with RMSNorm

Three findings from the reference run's own logs and source, in order of how much
they matter.

**1. Plain all-reduce backend IS aligned (now).** vLLM logs, for the tp group:

```
Using ['CUSTOM', 'SYMM_MEM', 'PYNCCL'] all-reduce backends (in dispatch order)
  for group 'tp:0' out of potential: ['NCCL_SYMM_MEM', 'QUICK_REDUCE',
  'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']
```

`CUSTOM` first -- the same `cross_device_reduce` IPC kernel §19a switched us to.
Its `max_size` is 8 MB like ours, so decode-size messages take the same path on both
sides. (`ep:0` uses PYNCCL only, but gpt-oss here is TP not EP.)

**2. But most of vLLM's collectives never take that path -- they get fused away.**
The same run logs:

```
Enabled custom fusions: allreduce_rms
Initialized FlashInfer All Reduce workspace: backend=mnnvl
```

`fuse_allreduce_rms` is an inductor pass that rewrites an all-reduce followed by a
residual-add + RMSNorm into **one** FlashInfer kernel,
`flashinfer_comm.allreduce_fusion(..., pattern=kARResidualRMSNorm)`, on the **mnnvl**
(NVLink multicast) backend. In a transformer decode layer essentially every
all-reduce *is* followed by a norm, so most sites are served by the fused kernel
rather than by `CUSTOM`.

**3. And that fused call uses PDL.** `allreduce_fusion(..., launch_with_pdl=True,
trigger_completion_at_end=num_tokens > PDL_ADVANCE_LAUNCH_TOKENS)` -- Programmatic
Dependent Launch, which lets the kernel begin while its predecessor drains. That is
the compute/communication overlap we do not have.

So at every collective site in a compiled tp>1 decode step:

| | vLLM | ours |
|---|---|---|
| kernels | 1 (fused AR+residual+RMSNorm) | 2 (IPC AR, then RMSNorm) |
| HBM round-trips | 1 | 2 |
| launch overlap | PDL | none |
| backend for that site | FlashInfer mnnvl multicast | custom IPC one-shot |

gpt-oss-120b at tp=2 has 36 layers x 2 collective sites = 72 per decode step. Our
profile (§19c) puts `cross_device_reduce_1stage` at 16.6% of decode over 2409 calls
(~75/step, matching), with RMSNorm a separate cost on top. Collapsing 144 kernels to
72, halving the memory traffic, and gaining PDL overlap is the right order of
magnitude for the residual +12% TP penalty (ours 2.560 -> 2.868 ms/step from tp=1 to
tp=2, where vLLM goes 2.789 -> 2.291).

This is a single coherent explanation for the compiled-tp>1-only signature from
§19: at tp=1 there are no collectives, so no fusion opportunity and we win; in eager
neither engine fuses, so we are at parity; only compiled-and-tp>1 exposes it.

### What implementing it requires

1. An inductor post-grad pattern-matching pass recognising
   `all_reduce -> add(residual) -> rms_norm` and replacing it with the fused op.
   `compilation.py` already installs `post_grad_custom_post_pass`, so the hook
   exists.
2. FlashInfer AR workspace management (`flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion`
   equivalent), created once per (world_size, hidden_dim, dtype) and destroyed on
   shutdown -- vLLM's `get_fi_ar_workspace` / `destroy_fi_ar_workspace`.
3. A size guard: the fused path has a `max_num_tokens` limit, above which vLLM falls
   back to the unfused pair. Prefill chunks (16384 tokens) will exceed it.
4. **Our RMSNorm must be pattern-matchable.** vLLM sets `custom_ops: ['none']` and
   `ir_op_priority rms_norm=['native']` precisely so the norm appears as traceable
   aten ops the pass can match. If ours is an opaque custom op at these sites, the
   pattern will never fire and step 1 is wasted -- check this first.

## 21. Closing it: the fused all-reduce, measured

§20 named the fused AR+RMSNorm kernel as the compiled-tp>1 difference. This
section is what happened when it was measured and built.

### 21a. The prize, measured on vLLM itself

Reading vLLM's source only says the fusion exists; it does not say what it is
worth, and the pattern-replacement count alone was ambiguous (`Replaced 1
patterns` / `Replaced 2 patterns` per compiled piece, not 72 -- because
`splitting_ops` splits the graph at attention and the 36 structurally identical
pieces are cache-deduplicated, so 2 replacements in the steady-state piece *is*
both sites of every layer).

The unambiguous measurement is vLLM's own knob, `pass_config.fuse_allreduce_rms`
(`/tmp/fkdev/vllm_fuse_ab.py`), on gpt-oss-120b tp=2, prompt 2048, 32 decode
steps, `enable_prefix_caching=False`:

| vLLM tp=2 | decode ms/step |
| --- | --- |
| `fuse_allreduce_rms=1` (default) | 2.292 |
| `fuse_allreduce_rms=0` | 2.684 |

**0.392 ms/step, 14.6% of vLLM's decode.** Our baseline was 2.868, so the fusion
accounted for 68% of the 0.576 ms gap; against *unfused* vLLM we were already at
0.94x. That is what made the port worth doing rather than merely plausible.

`allreduce_rms` is also the **only** custom fusion vLLM enables for this model
(`Enabled custom fusions: allreduce_rms`); the other passes that run are IR,
functionalization and cleanup, not kernel fusions. So this one fusion is the
whole fusion-set difference.

### 21b. Three preconditions, all checked before writing the pass

1. **mnnvl workspace exists on this topology.** It needs NVSwitch multicast;
   `create_allreduce_fusion_workspace(backend="mnnvl", ...)` succeeds on this
   8xB200 box (`/tmp/fkdev/fi_ar_probe.py`).
2. **Numerics.** Against the kernel it replaces -- all-reduce then
   `torch.ops._C.fused_add_rms_norm` -- the residual output is **bitwise
   identical** and the norm differs by about one bf16 ULP at the observed
   magnitude (1.56e-2 vs 1 ULP ~ 1.40e-2), cos ~ 0.9999999
   (`/tmp/fkdev/fi_ar_probe2.py`).
3. **CUDA graph capture and replay.** Decode runs entirely inside a replayed
   graph. Capture succeeds and five replays match eager with no NaN.

A fourth check made the port simpler: the kernel accepts output buffers distinct
from its inputs, producing bitwise-identical results and leaving `input`
untouched (`/tmp/fkdev/fi_ar_probe3.py`). vLLM calls it in place and relies on
auto-functionalization; we can register a plain functional op instead.

### 21c. Why `register_replacement` cannot express this pattern

The first implementation mirrored vLLM's mechanism -- an inductor
`register_replacement` pattern traced from `RMSNorm.forward_native`. Registration
succeeded and the post-grad graph provably contained the chain, yet it matched
nothing. Driving `MultiOutputPattern.match()` by hand over every candidate root
gave the reason:

```
mul_1: FailedMatch | no anchor found
```

vLLM's pattern has two outputs that are both `getitem`s off a *single*
`vllm.ir.ops.fused_add_rms_norm` node, so the matcher can always locate the
second output from the first. Ours are two independent traced chains that merely
share the residual add, and `MultiOutputPattern` cannot bridge those -- it has no
anchor to search from. This is a consequence of vLLM having an IR layer where
RMSNorm is one op that a later lowering pass expands; we decompose at trace time.

`AllReduceFusedAddRMSNormPass` therefore walks the chain structurally
(`infra/compilation.py`). Any deviation aborts that site, so a failed match costs
performance and never correctness -- and each rejection logs its reason, because
a silent zero is otherwise indistinguishable from the pass not running.

### 21d. Two findings the pass surfaced

**Inductor keeps our residual stream in fp32; vLLM's is bf16.** The first
structural match failed with "add's other operand is not convert(residual, f32)".
`forward_native` returns the residual as `x.to(bf16)` and the next layer does
`residual.float()`, so the graph should contain a bf16 round-trip. Inductor's
`pointless_convert` (an AMP cleanup) collapses `convert(convert(x, bf16), f32)`
and the round-trip disappears -- the summed activation flows between layers as
raw fp32:

```
add   = add(convert(ar,  f32), convert(arg4_1, f32))   # first site: clean
add_2 = add(convert(ar2, f32), add)                    # residual arrives as fp32
```

vLLM never sees this because its `fused_add_rms_norm` writes a bf16 residual as a
kernel output. So the compiled paths carried *different residual precision*.
Feeding the fused kernel a bf16 residual restores vLLM's semantics; the pass
visits sites in graph order and reinstates a `convert(residual, f32)` for any
consumer that still wants fp32, which is what lets the next site down the stream
match.

**A stale inductor cache silently disables post-grad passes.**
`PostGradPassManager.uuid()` returned `None`, which tells Inductor to cache
compiled graphs *without* accounting for these passes. The fusion fired 4/4 on a
toy model and 0/72 on gpt-oss purely because the 120B graph came back from a
cache written before the pass existed -- a false negative that looks exactly like
a broken pattern. `uuid()` now hashes `compilation.py`'s source plus the active
pass names, so editing a pass invalidates the artifacts. Any earlier measurement
of a post-grad pass in this repo should be treated as suspect unless it ran
against a fresh `TORCHINDUCTOR_CACHE_DIR`.

### 21e. The MoE all-reduce was invisible to Inductor

With the pass working, only **36 of 72** sites fused: one per compiled piece. The
other 36 are the MoE reductions, and under compile our MoE dispatches through the
opaque `torch.ops.fastkernels.moe_forward`, which swallowed the collective. vLLM
keeps its MoE reduction in traced Python
(`moe_runner._maybe_reduce_final_output`) for exactly this reason. Hoisting the
all-reduce out of the custom op -- `forward()` calls the opaque op and then
reduces -- brought it to 72/72. Applied to every MoE class
(`gpt_oss_moe`, `qwen3_moe`, `deepseek_moe`, `mixtral_moe`, `gemma4_moe`,
`kimi_moe`); `shared_expert_moe` was already traced.

### 21f. Result

gpt-oss-120b, prompt 2048, 32 decode steps (`/tmp/fkdev/gptoss_factorial.py`):

| cell | before | after | vLLM |
| --- | --- | --- | --- |
| tp=2 compiled | 2.868 | **2.462** | 2.292 |
| tp=1 compiled | 2.560 | 2.561 | 2.789 |
| tp=2 eager | 45.198 | 45.526 | 43.339 |

tp=2 compiled: **0.799x -> 0.931x**. The gain, 0.406 ms/step, matches the 0.392
vLLM measured for the same fusion on itself -- i.e. essentially all of the
fusion-attributable difference is now captured. tp=1 and eager are unchanged, as
expected: no collectives at tp=1, no inductor passes in eager.

Output is unaffected: with `FASTKERNELS_FI_ALLREDUCE_FUSION=0` the engine emits
**the same text and the same 8-token greedy match prefix** against vLLM, so the
fused kernel changed nothing observable and that divergence is pre-existing.

### 21g. What remains

A residual ~0.17 ms/step (2.462 vs 2.292) that is *not* fusion-attributable: it
was 0.184 ms between the two unfused configurations too. It is the same size as
the previously measured launch-count excess (ours 3121 vs vLLM 2590 launches per
decode step, before this work removed 72), at roughly 0.3 us per launch --
consistent with being launch-bound rather than kernel-bound, and the next thing
to attack.

## 22. Acceptance run, and why long-context does not move

`fastkernels validate` on gpt-oss-120b tp=2, run `fused-ar-01`, against the
`20260730-054258` baseline:

| workload | before | after | target |
| --- | --- | --- | --- |
| mixed | 0.88x, 56.2 | **0.95x, 61.9** | >=0.95x, >=50 -- met |
| long-context | 0.92x, 72.0 | 0.91x, **79.0** | >=0.95x, >=20 -- speed short |

Match tokens improved in both (56.2 -> 61.9, 72.0 -> 79.0), consistent with the
residual stream now carrying bf16 like vLLM's instead of fp32 (§21d).

**Long-context is prefill-bound, so the all-reduce fusion cannot reach it.** The
scenario is 64 sequences of ~24k prompt tokens and 256 output tokens: 1.57M
prefill tokens against 16k decode tokens, so essentially all the work is prefill.
And at prefill the fusion is inactive on *both* engines -- the FlashInfer
workspace tops out at 11650 tokens for this (hidden=2880, tp=2, SM100)
configuration while both engines chunk prefill at `max_num_batched_tokens=16384`,
so vLLM skips its own pass for compile range (11651, 16384) and our runtime guard
takes the unfused fallback. The factorial confirms the gap is in prefill and not
decode: prefill 54.0 ms (ours) vs 46.4 ms (vLLM) at 2048 tokens, barely changed
by the fusion (55.7 -> 54.0 ours, 48.9 -> 46.4 vLLM).

### 22a. Communication differences ruled out for prefill

Two candidates were checked and rejected on magnitude before implementing
anything:

- **Symmetric-memory all-reduce.** vLLM's tp dispatch order includes `SYMM_MEM`,
  which we lack. But `SYMM_MEM_ALL_REDUCE_MAX_SIZES["10.0"][2]` is **8 MB** --
  the same bound as the custom IPC path that is tried first -- and multimem only
  engages at world sizes 6 and 8 on SM100. So symm_mem is unreachable at tp=2 on
  B200 for both engines. Not a difference.
- **Our all-reduce clones; vLLM's does not.** `_custom_all_reduce_impl` does
  `t.clone()` then an in-place NCCL reduce, where vLLM's pynccl passes a fresh
  `empty_like` as `ncclAllReduce`'s recvbuff and copies nothing. Real, but 72
  sites x 188 MB of extra traffic is ~1.9 ms against a ~285 ms chunk: 0.7%, not a
  9% gap. (Worth fixing for its own sake; not the long-context cause. Note that
  `torch.ops._c10d_functional.all_reduce` clones too, so fixing it needs a
  pynccl-style out-of-place NCCL call, not a torch functional collective.)

### 22b. Status against the axes this investigation covered

| axis | finding |
| --- | --- |
| all-reduce backend | aligned: custom IPC <= 8 MB, NCCL above; symm_mem unreachable at tp=2/SM100 for both |
| all-reduce under compilation | **was different, now fixed**: fused AR+RMSNorm, 72/72 sites (§21) |
| compute/communication overlap | **was different, now fixed**: PDL via `launch_with_pdl=True` |
| data types | **was different, now fixed**: residual stream was fp32, now bf16 as vLLM (§21d) |
| fusions | `allreduce_rms` is vLLM's only custom fusion for this model; now matched |
| CUDA graph settings | already aligned: `[1,2,4] + range(8,256,8) + range(256,max+1,16)`, max capture 1024 |
| prefill chunk size | already aligned: `max_num_batched_tokens=16384` both |

The remaining gaps are outside these axes: ~0.17 ms/step of decode
(launch-count-shaped, §21g) and the prefill throughput gap that governs
long-context, which is a kernel-efficiency question rather than a
communication or compilation one.

### 22c. Prefill profile, and the MoE hidden padding lead

One 16384-token prefill chunk on our engine, tp=2 (`/tmp/fkdev/prefill_profile.py`),
139.1 ms of CUDA time:

| kernel | us | % | calls | us/call |
| --- | --- | --- | --- | --- |
| MoE expert GEMM 1 (mxfp4 bmm) | 40253 | 28.9% | 36 | 1118 |
| attention `fmhaSm100f...PagedKvCausal` | 27375 | 19.7% | 18 | 1521 |
| MoE expert GEMM 2 | 19768 | 14.2% | 36 | 549 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 16171 | 11.6% | 73 | 221.5 |
| RMSNorm (`vllm::_typeConvert`) | 5467 | 3.9% | 72 | 75.9 |
| Memcpy DtoD | 5221 | 3.8% | 183 | 28.5 |
| nvjet GEMMs (qkv, o_proj) | 9796 | 7.1% | 72 | ~136 |
| `elementwise_kernel<128,4,...>` | 3978 | 2.9% | 36 | 110.5 |

**NCCL protocol is not the lever.** `RING_LL` on 94 MB messages looks wrong (94.4
MB / 221.5 us = 426 GB/s, about half of NVLink5), but forcing `NCCL_PROTO=Simple`
changes nothing (138.6 vs 139.1 ms total) and `LL128` is worse (NCCL 21.5 vs 16.2
ms). vLLM reaches the same NCCL for these sizes, so this is shared cost.

The three dominant kernels (87.5 ms, 63%) are the same kernels vLLM runs at the
same shapes, so the ~12 ms gap has to be in the rest. The strongest candidate:

**We pad the MoE hidden dim every layer; vLLM may not.** `gpt_oss_moe.__init__`
rounds hidden up to `TRTLLM_MXFP4_ALIGN` (256), so 2880 -> 3072, and
`forward_impl` zero-pads the activation to that width on every call. That is the
36 x 110 us `elementwise_kernel` line (4.0 ms, a 94 MB read / 100 MB write), and
more importantly it runs both MoE GEMMs at K=3072 instead of 2880 -- **+6.7% on
60.0 ms of GEMM, another ~4 ms**. Together ~8 ms of 139 ms = 5.8%, which is the
size of the long-context gap (0.91x -> ~0.96x).

The vLLM side is genuinely ambiguous and must be settled before changing this:
`mxfp4_round_up_hidden_size_and_intermediate_size` *does* apply
`round_up(hidden_size, 256)` for `TRTLLM_BACKENDS`, yet the runtime
`FusedMoEConfig` for this run logs `hidden_dim=2880, hidden_dim_unpadded=2880,
intermediate_pad=None, intermediate_size_per_partition=1440` -- all unrounded --
and `moe_runner._maybe_pad_hidden_states` pads only when `moe_config.hidden_dim`
differs from the incoming width, so it would pad nothing. Resolve by confirming
which `Mxfp4MoeBackend` was selected and what K the expert GEMM actually
contracts over; if vLLM contracts 2880, our padding is pure overhead.

**Correction to the padding lead.** vLLM selected `TRTLLM_MXFP4_BF16`
(`TrtLlmMxfp4ExpertsMonolithic`, `flashinfer::trtllm_fp4_block_scale_moe`), which
is in `TRTLLM_BACKENDS`, so `round_up(hidden_size, 256)` applies on its side too
-- 2880 -> 3072 for weights, exactly as ours. The unrounded `hidden_dim=2880` in
the logged `FusedMoEConfig` is the config value, not the allocated weight width.
So the padding is most likely **shared cost rather than a difference**, and the
5.8% estimate above does not describe a gap. What remains genuinely
ours-only from this profile is the ~2.1 ms of all-reduce clones (73 x 28.5 us,
1.5%) plus smaller elementwise/triton residue. Localising the rest needs a
vLLM-side kernel profile; torch.profiler cannot see vLLM V1's kernels (they run
on a separate engine thread), so this requires nsys with
`--cuda-graph-trace=node`.

## 23. Correction: long-context is a decode gap, not a prefill gap

Section 22 concluded long-context was prefill-bound. That was inferred from a
throughput ratio plus a 2048-token prefill measurement, and it is wrong. The
factorial at the shape the scenario actually runs (24576-token prompts,
`/tmp/fkdev/longctx_factorial.py`) says:

| 24576 tok | ours prefill | vLLM prefill | ratio | ours decode | vLLM decode | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| tp=2 compiled | 238.3 | 237.3 | **1.00x** | 2.556 | 2.365 | **0.925x** |
| tp=2 eager | 254.2 | 488.4 | 1.92x | 45.863 | 42.870 | 0.935x |
| tp=1 compiled | segfault (§24) | 395.3 | -- | -- | 2.784 | -- |
| tp=1 eager | 420.6 | 754.3 | 1.79x | 41.799 | 37.907 | 0.907x |

**Prefill is at parity compiled and 1.8-1.9x faster than vLLM in eager.** The
0.86x prefill ratio in §22 was a 2048-token artefact that does not survive at
24576: vLLM's prefill gains 2.06x from compilation (488.4 -> 237.3) where ours
gains 1.07x (254.2 -> 238.3), so its compiled prefill catches up to where we
already were. The long-context deficit is therefore in **decode**, the same place
as the mixed-scenario gap, and §22a-22c's prefill hunt (NCCL protocol, symm_mem,
MoE padding, the all-reduce clone) was aimed at the wrong phase.

The decode deficit is ~7% in both compiled and eager, and vLLM's decode scales
2.784 -> 2.365 (1.18x) from tp=1 to tp=2 where ours scaled 2.561 -> 2.462 (1.04x)
at 2048 tokens against vLLM's 1.22x -- and at tp=1 we are *faster* than vLLM
(2.561 vs 2.789). So it remains a TP-scaling problem in decode.

**Measurement bug found in the process.** `gptoss_factorial.py` never set
`enable_prefix_caching=False`, and `timeit()` replays the same prompt, so vLLM
served runs 2 and 3 from its prefix cache while our engine (which has no prefix
cache at all) recomputed every time. Every vLLM *prefill* number from that script
is invalid. The 46.4 ms figure quoted in §22 happens to survive because it came
from `vllm_fuse_ab.py`, which does disable it.

## 24. gpt-oss tp=1 compiled segfaults in the trtllm MXFP4 MoE at long prefill

`SIGSEGV` inside `flashinfer::FP4BlockScaleLauncher::run` ->
`TrtllmGenBatchedGemmRunner::run` -> `BatchedGemmInterface::run`, on the **first**
autotune profile (0/12), for gpt-oss-120b at tp=1 with a 24576-token prompt
(16384-token chunk). Four things it is not:

- **Not an OOM.** Reproduces identically at `gpu_memory_utilization` 0.90 and
  0.75, and no allocator or out-of-memory message appears anywhere in the log.
- **Not a concurrency race.** An earlier instance was blamed on two 120B
  processes sharing FlashInfer's autotune cache; it reproduces in a strictly
  sequential run.
- **Not shape-independent.** tp=1 compiled is fine at 2048 tokens (48.7 ms
  prefill) and tp=1 *eager* is fine at 24576 (420.6 ms); tp=2 completes all 12
  autotune profiles at 24576.
- **Not vLLM's problem.** vLLM at tp=1 compiled, 24576 tokens, runs fine (prefill
  395.3 ms, decode 2.784 ms/step) through the same
  `flashinfer::trtllm_fp4_block_scale_moe`.

Isolated to our call into that kernel: with `FASTKERNELS_TRTLLM_MXFP4_MOE=0`
(added for this, mirroring the bf16 switch) the Triton path runs the same cell
successfully -- prefill 568.4 ms, decode 5.684 ms/step, the decode figure showing
why Triton is not a real alternative.

Since vLLM calls the same kernel with the same `tune_max_num_tokens`
(`max_capture_size`, 1024) and the same 256-alignment, the difference is in our
argument set for the tp=1 shape specifically. The one shape that distinguishes
tp=1 from the working tp=2 case is the unsharded intermediate:
`I_pad = round_up(2880, 256) = 3072` versus 1536. That is where to look next --
diff our `trtllm_fp4_block_scale_moe` kwargs against
`TrtLlmMxfp4ExpertsMonolithic.apply` at tp=1, not at tp=2 where both agree.

## 25. The decode gap is all-reduce *kernel time*, not launch count

`nsys --cuda-graph-trace=node` over 200 batch-1 decode steps, gpt-oss-120b tp=2,
profiling only the decode region via `cudaProfilerApi`
(`/tmp/fkdev/decode_launch_count.py`). Counts and times are summed across both
ranks.

**Launch count is not the problem.** 206,934 kernel instances for us against
208,828 for vLLM -- vLLM launches slightly *more* kernels (1044 vs 1035 per step
across two ranks). The "ours 3121 vs vLLM 2590 launches/step" figure that
motivated §21g does not reproduce, and the launch-bound reading built on it was
wrong. What differs is total kernel time: 1202.3 ms vs 1045.9 ms, +156 ms.

| bucket | ours | vLLM | delta | ours/step | vLLM/step |
| --- | --- | --- | --- | --- | --- |
| allreduce (fused) | 280.2 ms | 161.5 ms | **+118.7** | 144 | 146 |
| reduce (incl. custom AR) | 93.2 | 37.3 | **+55.9** | 78 | 74 |
| nvjet dense GEMMs | 330.2 | 312.7 | +17.5 | 218 | 218 |
| bmm (MoE mxfp4) | 250.6 | 250.1 | +0.5 | 144 | 144 |
| fmha | 82.0 | 86.5 | -4.5 | 72 | 72 |
| routing | 43.2 | 49.7 | -6.5 | 72 | 72 |
| kv-cache store | 19.8 | 36.2 | -16.4 | 72 | 72 |

The all-reduce bucket alone is 76% of the gap, at the same call count. Naming the
kernels shows why that is surprising:

| kernel | ours | vLLM |
| --- | --- | --- |
| `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 144/step, **avg 9.73 us** | 146/step, **avg 5.53 us** |
| `cross_device_reduce_1stage` | 2/step, **avg 119.73 us**, 47.9 ms | **absent** |

It is the *same* kernel -- same backend, same one-shot path -- running 1.76x
slower per call for us. Every documented argument matches vLLM's
(`use_oneshot=None` for mnnvl, `fp32_acc=True`, `launch_with_pdl=True`,
`trigger_completion_at_end = num_tokens > 16`, `max_token_num=11650`).

### 25a. The extra collective, and why it likely causes both rows

We run one `cross_device_reduce_1stage` per rank per step that vLLM does not run
at all. Its source is `VocabParallelEmbedding.forward`
(`L2/parallel_embedding.py:50`): our engine computes embeddings *outside* the
compiled graph, so that collective stays a standalone eager custom all-reduce.
vLLM puts `@support_torch_compile` on `GptOssModel` **including** `embed_tokens`,
so its embedding all-reduce is a graph node that the `AllReduceRMSNormPattern`
(the no-residual variant, which we skipped as "one site, negligible") fuses into
layer 0's `input_layernorm`. That is the `Replaced 1 patterns` piece observed in
§21a -- it is the embedding, not a layer.

119.73 us for a 5.7 KB message (batch 1 x 2880 bf16) is not transfer time; the
IPC one-shot should be ~5 us. It is a barrier absorbing rank skew. And because
FlashInfer's one-shot Lamport all-reduce busy-waits on peer flags, *residual*
skew is charged to every subsequent fused AR -- which is a coherent single
explanation for both rows: a slow per-step synchronisation point inflates the
standalone AR to 119.73 us and each of the 72 fused ARs from 5.53 to 9.73 us.

So the two candidate root causes, in order:

1. **Per-step rank desynchronisation.** Something our engine does once per step
   that is not symmetric across ranks (sampling, scheduler bookkeeping, or a
   host<->device copy) leaves rank 1 waiting. Confirm by timing each rank's step
   independently, or by inserting a `dist.barrier()` before the embedding and
   watching whether the 9.73 us drops toward 5.53 us -- if the barrier absorbs
   the cost and the fused ARs speed up, skew is proven.
2. **Embedding outside the compiled graph.** Moving `embed_tokens` inside the
   compiled boundary and adding the no-residual `AllReduceRMSNormPattern` would
   remove the standalone collective entirely, matching vLLM. This is worth doing
   regardless of (1), and is the narrower change.

Note the two non-AR items that go the other way and are worth keeping in mind as
credit already banked: our kv-cache store is 16.4 ms cheaper and routing 6.5 ms
cheaper than vLLM's.

### 25b. Correction: the kernel is at parity in the median; the gap is a tail

"1.76x slower per call" above is a mean, and the mean is misleading here. The
distribution of `oneshotAllreduceFusionKernel` over 28,800 / 29,200 instances:

| | avg | **median** | min | **max** | stddev |
| --- | --- | --- | --- | --- | --- |
| ours | 9.73 | **5.86** | 4.58 | **2707.64** | **78.69** |
| vLLM | 5.53 | **5.31** | 4.06 | 571.90 | 4.97 |

The medians are within 10%, and standalone the kernel costs 4.43-4.48 us at this
shape with ranks in lockstep. So the kernel, the backend and the arguments are all
fine -- `(9.73 - 5.86) x 28800 = 111 ms` of the 118.7 ms all-reduce delta lives in
a tail whose worst case is 2.7 ms, about one entire decode step. A rank
occasionally falls a full step behind and the one-shot Lamport busy-wait charges
that to whichever all-reduce is executing.

Buffer aliasing is ruled out as a cause: timed inside a CUDA graph at the decode
shape, in-place (vLLM's form) is 4.48 us/call and distinct output buffers (ours)
4.43 us/call (`/tmp/fkdev/fi_ar_inplace_timing.py`).

This reframes the remaining work. It is not a kernel, fusion, dtype or argument
difference -- those are now aligned. It is **per-step pacing**: our two ranks
drift and periodically resynchronise, and the stall is charged to the collectives.
The 119.73 us standalone embedding all-reduce (§25a) is the same phenomenon at the
step boundary. Both point at host-side per-step work -- our decode is ~2.5 ms of
GPU time per step, so any comparable Python/scheduler cost per rank starves the
GPU and any jitter between the two ranks' host loops becomes collective wait time.

Next measurements, in order:
1. Histogram the per-step wall time per rank to see whether the drift is periodic
   (e.g. every Nth step, suggesting a host sync, allocator growth, or a logging or
   bookkeeping path) or continuous.
2. Trace the host side (`--trace=cuda,osrt,nvtx`) and compare our per-step CPU
   work against vLLM's; look for a `cudaStreamSynchronize`, a `.item()`, or a
   D2H copy on one rank only.
3. Only then consider moving `embed_tokens` inside the compiled boundary, which
   removes one collective but does not by itself fix pacing.

### 25c. Not host starvation: the GPU is saturated, the skew is in step launch

Same 200-step batch-1 decode at tp=2, wall-clock: 539.4 ms, 2.697 ms/step, against
601 ms/rank of GPU kernel time from the nsys run. The GPU is saturated, so the
host loop is not starving it and "reduce per-step Python work" is not the answer.
(The two figures are close enough that the per-rank split is approximate; the
point is only that GPU busy time is not far below wall time.)

Counting the tail instead of averaging it locates the stall precisely. Our excess
is 111 ms over 200 steps with a 2707 us worst case: at a few hundred microseconds
per slow call that is **about one slow all-reduce per step**, not a per-call cost.
vLLM shows the same shape -- 6.4 ms of excess, 571.9 us worst case, also roughly
one per step -- but roughly 15x smaller. That is the signature of the *first*
collective in each step waiting on the peer rank's graph launch: within a replayed
graph the remaining 71 all-reduces execute back-to-back and cannot drift.

So the open question is narrow: **why do our two ranks begin a step up to ~500 us
apart when vLLM's begin within ~30 us?** The leading suspect is how per-step work
reaches the ranks. There is no `dist.broadcast` of decode metadata in our engine
(the only broadcast sites are the multimodal visual/audio paths); work is handed
to rank workers through shared memory, so rank 0 writes and the other rank polls,
and polling latency and jitter become launch skew. vLLM V1 broadcasts scheduler
output to workers each step, which bounds skew by a collective's latency.

Confirming and fixing, in order:
1. Timestamp the start of each rank's graph launch (a cheap `torch.cuda.Event`
   recorded before replay on each rank, gathered afterwards) and histogram the
   delta. This directly measures the skew and shows whether it is periodic.
2. If the shared-memory handoff is the source, replace the per-step wakeup with a
   collective-based one (broadcast the step descriptor, or a device-side flag)
   so both ranks leave the barrier together.
3. Independently worth doing and already understood: move `embed_tokens` inside
   the compiled boundary and add the no-residual `AllReduceRMSNormPattern`, which
   removes the standalone 119.73 us embedding all-reduce and matches vLLM's
   compilation boundary. This removes one eager collective from the inter-step
   critical path, though on its own it relocates the skew rather than removing it.

**Status: the long-context target is not met.** mixed 0.95x (met), long-context
0.91x. Everything on the axes this investigation covered -- all-reduce backend,
all-reduce under compilation, compute/communication overlap, dtypes, fusions,
CUDA-graph settings, prefill chunk size -- is now aligned with vLLM, and the fused
all-reduce is at parity in the median. The residual is inter-rank step-launch
skew, which is an engine scheduling property rather than a kernel or compilation
difference, and it is not closed.

---

## 26. gemma-4-26B-A4B: the throughput gap is admission control, not kernels

`20260731-000103/005` reported mixed **0.859x** and long-context **0.925x**
while *both* latency scenarios won (single-request 1.070x, fixed-batch-32
1.074x). Winning at batch 1 and 32 but losing at 1000 sequences points away
from kernels before any profiling: a per-kernel deficit would show up in the
latency probes too.

### Phase split

`tests/debug/profile_gemma4_phases.py --engine {fastkernels,vllm} --scenario ...`
times `max_tokens=1` against the full run to separate prefill from decode:

| mixed | fastkernels | vLLM |
| --- | --- | --- |
| prefill | 5.971s | 4.129s (fk 1.45x slower) |
| decode | 36.623s | 32.569s (fk 1.12x slower) |
| total | 42.595s | 36.698s |

Both phases were behind, decode carrying 69% of the absolute gap. Note the KV
pools were the same size in this run (526,560 vs 509,272 token slots, because
`max_model_len` is only 9,990 for the mixed set), so capacity was *not* the
mixed-scenario cause.

### Kernel census: our decode is faster

`tests/debug/profile_gemma4_kernels.py` profiles two decode lengths and diffs
the censuses, so the prefill phase -- and the decode steps chunked prefill
interleaves into it -- cancels exactly. A 1-vs-N diff does not cancel them and
inflates the decode launch counts ~3x; that artifact initially read as
"fastkernels launches attention 3.2x per layer", which is false.

At batch 256, 64 decode steps:

| | fastkernels | vLLM |
| --- | --- | --- |
| device time | **17.31 ms/step** | 18.11 ms/step |
| `kernel_unified_attention` | 10.67 | 10.02 |
| MoE | 2.88 (Triton) | 4.40 (cutlass) |
| launches | 640/step | 709/step |
| host idle | ~0% | 0.3% |

Decode is fully GPU-bound on both sides and our kernels are *faster* in
aggregate. So the decode phase being 12% slower end-to-end had to be fewer or
smaller batches, not slower steps.

### Cause 1: admission reserved each request's final length

The waiting-queue gate summed `ceil((num_prompt_tokens + max_tokens)/block_size)`
over every in-flight sequence and refused admission once that exceeded the whole
pool. vLLM's equivalent (`full_sequence_must_fit`, on by default via
`scheduler_reserve_full_isl`) reserves `min(request.num_tokens, max_model_len)`
-- prompt + tokens generated *so far*, i.e. just the prompt at admission. Its
purpose is only to stop chunked prefill admitting a 100k prompt on the strength
of its first chunk; it never reserves ungenerated tokens, and growth past that
point is handled by preemption.

The reservation bound gemma-4 specifically because its pool is small (32,910
blocks -- see Cause 3), so the 46,549-block worst case exceeded it and capped
admission at 692 of 1000 requests. Replaying both policies through
`capture.py::_simulate_continuous_batching` (analytic, no GPU):

| mixed | old policy | new policy |
| --- | --- | --- |
| gemma-4 (pool 32,910 < 46,549 worst case) | 1432 steps, mean batch 279.3 | **1044 steps, mean 383.1** |
| Llama-3.1-8B (pool 71,175 > 45,277) | 1044 steps, mean 372.4 | **identical** |

1044 is the floor (`max(output_len)` = 1024 decode steps + ~20 prefill-only
steps), and the simulated 1432 matches the measured 1468. Preemption fires
**zero** times either way on both throughput scenarios, so the reservation was
pure loss. Llama's schedule is bit-identical, so the change is provably a no-op
wherever the pool was already large enough.

Preemption was also made faithful while it became reachable: vLLM evicts
`self.running.pop()` -- the *newest* request -- and loops until the allocation
fits. fastkernels evicted whichever sequence it was iterating (the oldest),
which discards the most work and can starve the head of the queue.

`FASTKERNELS_STEP_PROFILE=1` now prints a preemption count, because
`Sequence.preempt()` resets to the prompt and drops `generated_ids` rather than
keeping them like vLLM's `_preempt_request`. Greedy decoding makes the replayed
tokens identical so output is unaffected, but each preemption costs a re-prefill
plus a replay of every decode step already taken. It is worth fixing only if
that counter is ever nonzero.

### Cause 2: `max_num_batched_tokens` was pinned to 2048

`LlamaEngine.__init__` special-cased gemma-4 to 2048 (no comment, present since
the initial commit) where vLLM logs `Chunked prefill is enabled with
max_num_batched_tokens=16384`. At 2048 a 256x512 prefill needed 64 chunks
instead of 8:

| prefill, bs=256 x len 512 | fk @2048 | fk @16384 | vLLM |
| --- | --- | --- | --- |
| device time | 1411 ms | 1242 ms | 1081 ms |
| wall time | 1952 ms | 1287 ms | 1093 ms |
| host idle | **27.7%** | 3.5% | 1.1% |
| launches | 41,280 | 5,160 | 6,651 |

An `FASTKERNELS_MAX_NUM_BATCHED_TOKENS` override is kept for A/B.

The residual 161 ms of device time is three kernels, not the MoE or attention
(both level or faster): `triton_poi_fused_4` (+84 ms), an unfused `aten::sum`
(+54 ms), and `_C::gelu_tanh_and_mul` vs vLLM's Inductor-fused
`triton_poi_fused_gelu_mul_slice_1` (+28 ms). The `aten::sum` was
`moe_sum.cu`'s `default:` branch -- it specialised only `topk` 2/3/4 and
gemma-4 has `top_k_experts=8`, so it fell back to `at::sum_out`. A
float-accumulating `case 8` (matching `at::sum_out`'s `acc_type` and k-order, so
no accuracy change) removes it.

### Cause 3 (documented, not fixed): sliding-window KV capacity

25 of gemma-4's 30 layers are `sliding_attention` with a 1024 window.
`_allocate_variable_kv_cache` gives every layer the same block count indexed by
one shared `seq.block_table`, so a sequence holds its full length even in the
sliding layers. vLLM puts them in a `SlidingWindowSpec` group whose
`SlidingWindowManager.remove_skipped_blocks` frees out-of-window blocks on every
`allocate_slots`, capping a request at
`cdiv(sliding_window - 1 + max_in_flight_tokens, block_size) + 1` blocks there.
Hence 498,272 token slots against vLLM's 1,862,998.

This costs *capacity only*. The per-layer mask is right (`sliding_window` is
plumbed to `window_size=(1023, 0)`), so the resident extra KV is never attended
to, and vLLM's Triton kernel bounds its tile loop by `SLIDING_WINDOW`
(`loop_lo`/`loop_hi`) so it is not read either. Closing it needs per-layer-group
block tables threaded through CUDA-graph capture, which assumes a single
`block_tables` tensor. It buys nothing on these workloads (mixed averages 737
tokens, inside the window; long-context is 89% prefill) but it is what made
gemma-4 uniquely exposed to Cause 1.

**Attention parity was verified otherwise complete for all 30 layers:**
head_dim 256/512, kv heads 8/2 (`attention_k_eq_v`), `scale=1.0` (gemma-4 does
*not* use `query_pre_attn_scalar`), no attention softcap
(`attn_logit_softcapping` absent; `final_logit_softcapping=30.0` is the LM-head
one), per-layer-type RoPE theta and `partial_rotary_factor`, and
`num_kv_shared_layers=0`. vLLM logs an intent to use FA4 for all layers, then
`FA4 on Blackwell does not support head_size=256/512 due to TMEM capacity
limits` and resolves to `TRITON_ATTN` for every layer -- which is what
`prefer_triton=True` already forces. The only compute-side divergence is the MoE
backend (vLLM: `FlashInferExperts` CUTLASS, chosen over trtllm-gen because
gemma-4's activation is `gelu_pytorch_tanh` not gated swiglu; ours: Triton
grouped GEMM), and ours is faster at decode shapes.

### Method note

This host drifts 3-8% between runs of identical code (Llama-3.1-8B mixed came
out 21,504 / 19,662 / 20,849 tok/s on three full runs; vLLM's own
single-request median moved 4%). A suspected 8% Llama regression from the
scheduler change was noise: fastkernels' *batch-1* median moved 0.4557 ->
0.4564s, and an admission-control change cannot affect a single-sequence run at
all. Confirm scheduler changes with the analytic simulator, and A/B only under
identical invocation.

### Acceptance (tp=1, compiled, B200)

| scenario | original | scheduler + mnbt | + `moe_sum` case 8 |
| --- | --- | --- | --- |
| mixed | 0.859x | 0.996x | **1.016x** |
| long-context | 0.925x | 1.002x | **1.005x** |
| single-request | 1.070x | 1.123x | **1.156x** |
| fixed-batch-32 | 1.074x | 1.146x | **1.132x** |

Output token counts are identical throughout (400,938 mixed / 16,384
long-context). Mixed alignment is unchanged (72.53 -> 72.86 avg prefix match
over 1000 requests). Long-context alignment reads 83.5 -> 73.3, but that set is
64 requests with a per-request prefix-match sd of ~63, i.e. stderr ~8 per run
and ~11 on the difference -- the change is inside one standard error and is not
evidence of a numerics regression. Any perturbation reshuffles this metric,
because greedy decoding on a 26B MoE diverges chaotically: fastkernels-vs-vLLM
prefix match is only ~18% of output length even at baseline, and two
fastkernels runs differing only in the `moe_sum` reduction order agreed on
19.7% of tokens with 158/1000 sequences bit-identical.

### tp=2: the gap was never TP-specific

Every measurement above is tp=1, where the original gap reproduced in full, so
`tp>1` was ruled out as the trigger early. Re-running with both fixes at tp=2
confirms there is more headroom there, not less:

| scenario | tp=1 | tp=2 |
| --- | --- | --- |
| mixed | 1.016x | **1.127x** |
| long-context | 1.005x | **1.105x** |
| single-request | 1.156x | 1.099x |
| fixed-batch-32 | 1.132x | 1.076x |

tp=2 mixed is 26.97s against vLLM's 30.39s; long-context 87.15s against 96.34s.
Long-context alignment reads 83.28 avg prefix match at tp=2 -- back at the
pre-change 83.5 -- which supports reading the tp=1 value of 73.3 as sampling
noise on a 64-request set rather than a numerics regression. Backend selection
is unchanged at tp=2: the same `FA4 on Blackwell does not support head_size=...`
fallback to `TRITON_ATTN` for every layer. vLLM turns on `fuse_allreduce_rms` at
tp>1, which fastkernels already matches (sections 20-21).

---

## 27. nvidia/GLM-5.2-NVFP4 with an fp8_e4m3 KV cache

The NVFP4 checkpoint is not the FP8 one with different weights. It changes which
modules are quantized, which MoE kernel runs, and — because the KV cache dtype
drives vLLM's backend choice — which attention and MLA-prefill backends the
reference uses. All of the following is read out of a real vLLM run on the model,
not inferred from the selector code:

```
Using FLASHINFER_MLA_SPARSE attention backend out of potential backends:
  ['FLASHINFER_MLA_SPARSE'].
Using HND KV cache layout for FLASHINFER_MLA_SPARSE backend.
Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla KV cache
  format, please set `--attention-backend FLASHMLA_SPARSE`
Using TRTLLM_RAGGED MLA prefill backend.
Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential backends: [...]
Using MoEPrepareAndFinalizeNoDPEPMonolithic
kv_cache_dtype=fp8_e4m3, block_size=64, quantization=modelopt_fp4
Checkpoint does not provide a q scaling factor ... Using KV cache scaling
  factor 1.0 for fp8_e4m3.
```

### 27a. Three things to match

**Quantization scope.** Only `mlp.experts.*` is NVFP4. The checkpoint's own
`ignore` list covers every `self_attn*`, every `mlp.shared_experts*`, the dense
layers 0-2, the MTP layer 78, `lm_head` and `embed_tokens`; and vLLM builds MoE
gates with no `quant_config` at all (`deepseek_v2.py`), so the router is BF16
too. So `quant_config is not None` can no longer mean "every linear is
quantized": `infra/quant_scheme.linear_quant_config` hands every `Linear` a
`None` config under NVFP4, and only `DeepSeekMoE` sees the real one.

**KV layout.** `fp8_e4m3` here is the plain per-tensor layout —
`[num_blocks, block_size, 576]` `float8_e4m3fn`, one byte per element — **not**
DeepSeek's 656-byte block-scaled `fp8_ds_mla`, which FLASHINFER_MLA_SPARSE
rejects outright (`supports_combination`: "SM10 does not support fp8_ds_mla
kv-cache dtype"). The two are separate `MLAAttention.kv_cache_dtype` values, and
`FASTKERNELS_KV_CACHE_DTYPE=fp8` still aliases to `fp8_ds_mla`.

**The prefill gate.** Sparse MLA only runs the trtllm-gen MQA kernel for tokens
it has to. When every prefill sequence is at most `index_topk` (2048) long the
indexer's top-k selects the whole context, sparse degenerates to dense, and vLLM
routes prefill through the dense `trtllm_ragged_attention_deepseek` instead
(`mla_attention.py`'s `use_mha`). Skipping that gate would run 2048-slot gathers
for a 512-token prompt.

### 27b. vLLM 0.26 forces FP32 MoE routing for glm_moe_dsa

`_get_moe_router_dtype` returns FP32 unconditionally when
`model_type == "glm_moe_dsa"` — "Older GLM-5/5.2 configs require fp32 routing but
do not expose moe_router_dtype yet" — and the gate is built with that
`out_dtype`. fastkernels passed `None`, which is right for DeepSeek but for GLM
gives FP32 only in decode: the `F.linear` fallback prefill lands on does not
cast, so every prefill token reached grouped-topk in BF16 where vLLM had FP32.
Near-tie expert selection flips on exactly that bit pattern. This is drift from
vLLM 0.24, which had no such special case.

### 27c. deep_gemm must be vLLM 0.26's vendored commit

vLLM 0.26's sparse indexer calls the unified `fp8_fp4_mqa_logits`, which
`v2.1.1.post3` does not export — every DSA model dies on its first prefill with
"DeepGEMM backend is unavailable in the current vLLM environment". The pin is now
`a6b593d` (reports itself as 2.5.0), which is a strict superset and is what the
DSA ops in this tree were already written against.

### 27d. Decode CUDA graphs: three defects, ~20x at stake

GLM-5.2 — FP8 and NVFP4 alike — died on the **first decode CUDA-graph replay**,
so it was only ever runnable with `--enforce-eager`. That is not a small loss:
vLLM's own compiled run does the mixed workload in 11.5 s against **227.6 s
eager**, i.e. eager gives up roughly 20x on a 78-layer MoE at bs=32. Any
"eager-vs-eager" ratio for these models measures Python launch overhead, not
kernels.

1. `get_paged_mqa_logits_metadata` was called from **inside** the captured
   region. Capture records fine; the first replay dies inside
   `deep_gemm.fp8_fp4_paged_mqa_logits` — `cudaErrorInvalidAddressSpace` on
   NVFP4, `cudaErrorIllegalAddress` on FP8, and **nothing at all under
   compute-sanitizer**, which is what a pool/launch-timing fault looks like. The
   kernel is not the problem: fed a schedule built beforehand it captures and
   replays fine standalone. The schedule is `(num_sms+1, 2)` for every batch size
   and context length, so the engine keeps one persistent buffer and refreshes it
   per step at each decode site, including immediately before `replay()`. vLLM
   builds the same tensor in its metadata builder — also outside the graph.
2. That buffer, allocated during KV-cache setup (which runs under
   `inference_mode`), was an *inference tensor*, and the compiled model takes it
   as an input. Allocate it under `inference_mode(False)`.
3. `capture_cudagraph` traced the compiled model with grad enabled, so
   AOTAutograd built a training graph behind an `autograd.Function`; the first
   mixed prefill+decode batch afterwards rejected its own inference-mode inputs
   ("Inference tensors cannot be saved for backward"). Decorate the capture with
   `inference_mode`.

Localised by bisection: stub the indexer's decode top-k → replay OK; keep its
metadata build but stub only the paged-logits kernel → replay OK.

Fixed on the way: the DSA radix top-k and both trtllm-gen MLA workspaces were
allocated on first use, which — once capture worked — would have put them in one
graph's private memory pool while every later batch size replayed atomics into
it. They are now materialised from KV-cache allocation via `ensure_workspaces`.
