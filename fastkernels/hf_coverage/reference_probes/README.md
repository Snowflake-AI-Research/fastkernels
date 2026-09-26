# HF Transformers reference probes

Standalone reproductions of the known Hugging Face Transformers reference bugs
for 12 models. They need no prepared artifacts or scratch files: each probe builds a
small seeded random-init model from the HF config classes (or drives the exact
native method under test) and writes one JSON per model.

Two Transformers revisions are tracked:

| name | commit | transformers version |
|---|---|---|
| OLD (fastkernels pin) | `da6c53e431f7c9ef0691239d4ce89b0f711ecad7` | 5.8.0.dev0 |
| NEW | `89b6b17574892ec0770551537a3fe69d6886703e` | 5.18.0.dev0 |

## Setup

Use a Python 3.12 environment with `torch` 2.11 (CUDA build) and `ninja` on `PATH`.
The OLD revision works with `huggingface_hub` 1.30, `tokenizers` 0.22.2,
`safetensors` 0.8.0 and `kernels` 0.12.3 installed in that environment. The NEW
revision needs newer versions, installed to a separate directory that is passed
with `--extra-path`:

```bash
HF_OLD=$PWD/transformers-old HF_NEW=$PWD/transformers-new DEPS_NEW=$PWD/deps-new
git clone https://github.com/huggingface/transformers.git $HF_OLD
git -C $HF_OLD checkout da6c53e431f7c9ef0691239d4ce89b0f711ecad7
git clone https://github.com/huggingface/transformers.git $HF_NEW
git -C $HF_NEW checkout 89b6b17574892ec0770551537a3fe69d6886703e
pip install --no-deps --target $DEPS_NEW huggingface_hub==1.33.0 tokenizers==0.23.1 safetensors==0.8.0 kernels==0.17.0
```

The MRA probe and the DeepSeek-V4 FP8 forward load `kernels-community/mra` and
`kernels-community/finegrained-fp8` from the Hub through `kernels`. The first run
needs network access or a pre-populated `HF_HOME`. Do not set `HF_HUB_OFFLINE=1`:
NEW `kernels` 0.17 then refuses its cached versioned kernels, and OLD `kernels`
0.12.3 cannot resolve the FP8 kernel. No model weights are downloaded.

## Running

```bash
cd fastkernels/hf_coverage/reference_probes
export CUDA_VISIBLE_DEVICES=<idle gpu>
python run_all.py --hf-source $HF_OLD --out $RUNS/probes-old
python run_all.py --hf-source $HF_NEW --extra-path $DEPS_NEW --out $RUNS/probes-new
python probes.py sam_hq --hf-source $HF_NEW --extra-path $DEPS_NEW --out $RUNS/probes-new/sam_hq.json
```

Run `probes.py` as a script (not with `python -m`). `--hf-source` and then the
`--extra-path` entries are put at the front of `sys.path` (and exported in
`PYTHONPATH`) before Transformers is imported. `run_all.py` runs each model in its own
process and writes `<out>/<model>.json`, `<out>/<model>.log` and `<out>/summary.json`.
The summary compares each verdict with the expected verdict for the detected git
revision. Every JSON records `transformers_file`, `transformers_version`,
`transformers_git_revision` and the dependency versions.
`gemma4_assistant`, `sam3_video` and `granite4_vision` always run on CPU; the
other probes use `--device` (default `cuda`).

`verdict` is one of:

- `execution_error`: the HF reference raises in the probed setting.
- `wrong_computation`: it runs but produces a value that is verifiably wrong.
- `reference_selection_failure`: no runnable HF reference matches the published model.
- `ok`: the known bug did not reproduce.
- `probe_error`: the probe failed before reaching the trigger (exit code 1).

A reproduced bug is the expected outcome and exits 0. Error entries include
`hf_site`, the innermost Transformers model-file frame that raised.

## Results

Verified on 2026-09-25 on one NVIDIA B200 with torch 2.11.0+cu130. All 12
verdicts matched the expected verdicts on both revisions.

| model | OLD verdict | OLD evidence | NEW verdict | NEW evidence |
|---|---|---|---|---|
| deepseek_v4 | wrong_computation | leak 0.551 first / 0.733 prefix; no cache | execution_error | leak 0.0, cache returned; FP32 norm into BF16 router `:1115` |
| mra | execution_error | FP32 attention into BF16 dense `:624`; FP32 ok | execution_error | same, `:623`; FP32 ok |
| reformer | execution_error | FP32 axial positions into BF16 LayerNorm `:1373` | execution_error | same, `:1383` |
| grounding_dino | execution_error | FP32 text positions into BF16 query `:1154` | wrong_computation | int64 position embedding, max abs 1.0 |
| mm_grounding_dino | execution_error | same, `:761` | wrong_computation | int64 position embedding, max abs 1.0 |
| clvp | wrong_computation | positions `[[0]]`, then 2/3/4; prefill mask 1 key/query | wrong_computation | positions `[[0]]`, then 2/3/4; mask 1..12 keys |
| nllb_moe | wrong_computation | routes 2/3, calls 0/1 | wrong_computation | routes 2/3, calls 0/1 |
| sam_hq | wrong_computation | upscaler input is pre-transformer; diff 3.069 | wrong_computation | same |
| phi4_multimodal | wrong_computation | 64 wrong positions in BF16, 0 in FP32 | wrong_computation | same |
| gemma4_assistant | wrong_computation | first-token counts `[100, 0]` | wrong_computation | `[100, 0]` |
| sam3_video | wrong_computation | object 20 gets object 10 mask, score 0.731 | wrong_computation | same |
| granite4_vision | reference_selection_failure | IBM import fails; built-in needs `mlp_bias` `:443` | reference_selection_failure | same, `:437` |

Each suite takes about 90 s of wall time; DeepSeek-V4 takes about 17 s, and each
other probe takes 4-9 s, including process startup.

## Models

Line numbers are `OLD / NEW` in `src/transformers/`.

### deepseek_v4 (OLD: wrong computation; NEW: crash)

- OLD: future tokens change earlier logits, and no cache is returned. The
  FP32 eager model with 513 tokens and tokens 256+ replaced has a first-logit delta
  of 0.551 and a prefix delta of 0.733. Sliding-window layers show delta 0; the
  first compressed layer (CSA/HCA) shows a nonzero delta. Cause:
  `models/deepseek_v4/modeling_deepseek_v4.py:777-778` right-pads the attention mask
  with `0.0` (visible) for the compressed KV entries, so every query sees compressed
  windows that summarize future tokens. At `:1203`,
  `return_cache = past_key_values if use_cache else None` returns no cache unless
  the caller passes one.
- NEW: causality is fixed (delta 0.0) and a cache is returned. With BF16 loading,
  `_keep_in_fp32_modules_strict` (`:1235`) now keeps `post_attention_layernorm`, `input_layernorm`,
  `q_a_norm`, `kv_norm` and `norm` in FP32. Their FP32 outputs feed BF16 linears.
  The selected-workload config (hidden 512, 256 experts, 3 hash layers,
  CSA 4 / HCA 128, 2053 tokens) is FP8-quantized on load. It raises
  `expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`
  in the hash router `F.linear` (`:1115`), called from `:1188`. The
  `bf16_router_boundary` check reproduces the same error with no FP8 kernel,
  directly on layer 0 (post-attention norm, then hash router). Without FP8, the
  first failure is earlier, at the BF16 `q_a_proj` (`:819`); it is the same defect.
- The probe runs three checks: `causality_fp32`, `bf16_router_boundary` and
  `bf16_fp8_workload_forward`. The top-level verdict is the most severe of them.
  On OLD the norms stay BF16 and the router boundary passes. The OLD FP8 forward
  does not crash, but it reports `inconclusive` with non-finite logits: OLD
  on-the-fly quantization leaves two indexer `weight_scale_inv` tensors unloaded.
  The FP8 forward reports `not_run` if the FP8 kernel cannot be obtained; OLD
  `kernels` 0.12.3 must reach the Hub, and NEW `kernels` 0.17 fails under
  `HF_HUB_OFFLINE=1`. The FP8 weights are quantized on load from seeded BF16
  values; the original workload used pre-quantized weights.

### mra (crash, both)

- BF16 raises `mat1 and mat2 must have the same dtype, but got Float and BFloat16`
  at `MraSelfOutput.dense` (`models/mra/modeling_mra.py:624 / :623`).
  Attention casts q/k/v/mask to FP32 before calling the native kernel
  (`:591-594 / :590-593`), and its FP32 output is never cast back.
- The native CUDA kernel is required. The probe records its path and gives
  `probe_error` if the kernel is missing, rather than accepting the zero-output
  fallback. Attention output is finite and nonzero. FP32 control: the full forward
  runs and is finite.

### reformer (crash, both)

- BF16 raises `expected scalar type Float but found BFloat16` at the first attention
  LayerNorm (`models/reformer/modeling_reformer.py:1373 / :1383`).
  `AxialPositionEmbeddings` creates its weights with explicit
  `dtype=torch.float32` (`:219 / :220`). BF16 loading therefore leaves them FP32,
  and adding them promotes the hidden states to FP32 before the BF16 LayerNorm.
- Loaded through native `from_pretrained(..., dtype=bfloat16)` from in-memory
  weights. FP32 control runs and is finite.

### grounding_dino and mm_grounding_dino (OLD: crash; NEW: wrong computation)

- OLD: `get_sine_pos_embed` returns FP32 for the text position IDs regardless of
  the model dtype (`models/grounding_dino/modeling_grounding_dino.py:1005`, FP32
  `dim_t` at `:1025`, called at `:1068`; MM: `:967`/`:987`, called at `:1030`). Adding it to BF16 text features gives FP32 queries.
  The BF16 text-enhancer query projection then raises
  `mat1 and mat2 must have the same dtype, but got Float and BFloat16` (`:1154`;
  MM `:761`).
- NEW: `encode_sinusoidal_position_embedding` ends with `.to(pos_tensor.dtype)`
  (`:72`; MM `:981`). The integer text position IDs (called at `:1062`; MM `:1011`) therefore
  produce an int64 sine/cosine embedding: `[0, 1, 0, 0, ...]`. Max abs error versus
  float position IDs is 1.0. The forward completes, and the encoder consumes that int64 tensor.

### clvp (wrong computation, both; masks fixed in NEW)

- `models/clvp/modeling_clvp.py:1143-1151 / :1140-1148` subtracts position embeddings
  0..11 from the 12 conditioning embeddings, expecting the decoder to add them back.
  But generation passes `position_ids=[[0]]` for all 12 on prefill. Decode steps then get
  positions 2, 3, 4 (`:1184 / :1181`, `input_ids` length) instead of 12, 13, 14.
- OLD additionally builds a prefill causal mask that admits 1 key per query
  (decode steps: 2, 3, 4 keys). NEW admits 1..12 keys on prefill and 13, 14, 15 on
  decode. The decoder code is identical in both revisions, so that fix lives outside
  `modeling_clvp.py`.

### nllb_moe (wrong computation, both)

- `NllbMoeSparseMLP.forward` passes the router's one-hot `top_1_mask` as
  `router_mask` (`models/nllb_moe/modeling_nllb_moe.py:382`). `NllbMoeExperts` then
  one-hot encodes it again as if it held expert indices (`:351`). The router selects
  experts 2 and 3, but experts 0 and 1 are called.

### sam_hq (wrong computation, both)

- In the mask decoder (`models/sam_hq/modeling_sam_hq.py:989-1003`, same lines in
  both), the transformer's updated image features are bound to `iou_token_out` and
  then overwritten (`:996`). The upscaler (`:1003`) therefore receives the
  pre-transformer image embeddings. Probe: upscaler input equals the pre-transformer
  features exactly and does not equal the transformer output; max abs difference 3.07.
  The author's code (SysCV/sam-hq `mask_decoder_hq.py`) upscales the transformer output.

### phi4_multimodal (wrong computation in BF16, both)

- `Phi4MultimodalVisionEmbeddings` casts the fractional patch coordinates to the
  pixel dtype before `bucketize` (`models/phi4_multimodal/modeling_phi4_multimodal.py:352-356 / :350-354`).
  In BF16 this rounding puts patches in the wrong buckets. The probe uses 7 crops
  of 32x32 patches, two with 31 valid columns: 64 valid positions are wrong in BF16
  and 0 in FP32, measured against the integer crop geometry.

### gemma4_assistant (wrong computation, both; CPU)

- `SinglePositionMultiTokenCandidateGenerator` drafts with argmax
  (`generation/candidate_generator.py:1390 / :1356`) and returns the raw draft
  logits. Sampled assisted generation (`generation/utils.py:3617-3618 / :3916-3917`)
  then runs `_speculative_sampling`. That function treats `softmax(draft logits)`
  as the proposal distribution q (`:3860 / :4195`) and accepts with p/q
  (`:3864 / :4204`). With a mocked assistant that prescribes q = p = (0.6, 0.4),
  the first token over 100 seeds is `[100, 0]` instead of about 60/40. Gemma4 assistants
  are routed to this generator at `generation/utils.py:991-999 / :1179-1188`.
  Only the assistant forward is mocked; the candidate generation and sampling
  methods are native.

### sam3_video (wrong computation, both; CPU)

- `models/sam3_video/modeling_sam3_video.py:1498` removes an object during hotstart
  before `build_outputs` zips the surviving object IDs with the pre-removal mask
  tensor (`:1521`); scores are zipped the same way (`:1683`). The lines are the same in
  both revisions. The probe runs the native forward over frames 1-8 with default
  hotstart (delay 15, unmatch threshold 8). Only the detector and tracker neural
  producers are stubbed, with distinct per-object masks. Object 10 is removed.
  The survivor, object 20, receives object 10's mask and score (0.731 instead of 0.953).

### granite4_vision (reference selection failure, both; CPU, meta device)

- IBM publishes `ibm-granite/granite-4.0-3b-vision` (revision
  `bf108f36960fb4df79bf035e506c592f4ee3c2d3`) as `trust_remote_code` for
  transformers 4.57.6. Its `modeling.py:19-21` imports
  `HybridMambaAttentionDynamicCache` from `granitemoehybrid`, which neither revision
  defines. The probe executes that exact import: `ImportError`.
- The built-in `Granite4VisionForConditionalGeneration` expects a `GraniteConfig`
  text model: `mlp_bias` at `models/granite4_vision/modeling_granite4_vision.py:443 / :437`,
  and the text model is hard-coded at `:768 / :764`. IBM's config gives
  `GraniteMoeHybridConfig`, and construction raises `AttributeError: ... 'mlp_bias'`.
  Control: the same dimensions with a `granite` text config construct.
  The class's own default config (`LlamaConfig` text) fails with
  `attention_multiplier` (`:358 / :352`).
- The probe also tries to load IBM's remote code from the local HF cache only. It
  reports `OSError` when that code is not cached; it never downloads it. The probe
  shows the two HF-side routes are unusable. It does not show that IBM's model
  is broken under its documented transformers 4.57.6.
