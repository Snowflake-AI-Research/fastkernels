# HF reference issues

Twelve entries are blocked because the HF Transformers reference itself looks
wrong or unusable at the pinned revision (`da6c53e4`). The task is a judgment
call for each: **is the problem real, and how should the entry be classified?**

Every claim below has been reproduced on small seeded models and checked against
a newer revision (`89b6b175`); the reproductions are in
[`reference_probes/`](reference_probes/README.md). Line numbers refer to
`src/transformers/` at the pinned revision.

## Classifications

| Classification | When it applies | Table entry |
|---|---|---|
| **FP32 only** | Reference math is right but the BF16 path crashes | Evaluate in FP32; label the dtype |
| **Newer revision** | Fixed upstream | Evaluate against that revision; label it |
| **Patched reference** | Clear bug with a small, documented fix | Evaluate against the patched reference; label it |
| **Out of scope** | Bug confined to a path the table does not evaluate | Evaluate the rest; state the exclusion |
| **Upstream bug** | Real, and no acceptable workaround | Leave blank; cite the bug |
| **Reference selection** | Not an HF bug; the right reference is unsettled | Choose a reference or exclude |

Six other entries (CTRL, DBRX, Doge, Emu3, DeepSeekV3, MiniMaxM2) were already
resolved as **newer revision** and pass against `89b6b175`.

## Summary

| Model | Problem | Newer revision | Our side | Tentative call |
|---|---|---|---|---|
| mra | BF16 crash | same | implementation, no case | FP32 only |
| reformer | BF16 crash | same | case | FP32 only |
| grounding_dino, mm_grounding_dino | BF16 crash | runs, but wrong | FP32 case | FP32 only |
| deepseek_v4 | attends to future tokens | fixed; BF16 crash | none | newer revision, FP32 |
| nllb_moe | calls the wrong experts | same | none | patched reference |
| sam_hq | upscales stale features | same | none | patched reference |
| clvp | wrong positions and masks | masks fixed | implementation, no case | patched reference |
| phi4_multimodal (vision) | BF16 rounding misplaces patches | same | case; speech passes | match HF, note it |
| gemma4_assistant | biased sampled assisted generation | same | implementation, no case | out of scope |
| sam3_video | survivor gets removed object's mask | same | none | out of scope |
| granite4_vision | no loadable reference | same | none | reference selection |

The tentative calls are starting points, not decisions.

## Details

### Crash in BF16 only

**mra.** The native attention kernel works in FP32 (`models/mra/modeling_mra.py:591-594`),
and its output is never cast back, so the BF16 dense layer raises (`:624`). FP32
runs, and an earlier paired FP32 comparison with our implementation matched
(0.00002% L2). *False positive?* Not a wrong-answer bug; BF16 is simply untested.
*Check:* whether an FP32 row is acceptable, given the kernel itself is FP32-only.

**reformer.** Axial position weights are created explicitly in FP32
(`models/reformer/modeling_reformer.py:219`), which promotes hidden states to FP32
before a BF16 LayerNorm (`:1373`). FP32 runs and our FP32 comparison passed.
*False positive?* A deliberate HF dtype choice, not wrong math. *Check:* same
question as MRA.

**grounding_dino, mm_grounding_dino.** Text position embeddings are always FP32
(`models/grounding_dino/modeling_grounding_dino.py:1005`, `:1025`), so the BF16
query projection raises (`:1154`; MM `:761`). The newer revision avoids the crash
by casting the embedding to the integer position dtype, which truncates it to
`[0, 1, 0, ...]`: silently wrong, so the newer revision is not an option. Our
cases already run in FP32 and the finite outputs match (at most 0.0014% L2).
*Check:* the runner reports a mismatch only because both sides emit intentional
`-inf` masks; add `allow_infinite_outputs` for those outputs, rerun, and confirm the
masks match exactly.

**deepseek_v4.** At the pinned revision, the compressed attention entries are
padded as visible (`models/deepseek_v4/modeling_deepseek_v4.py:777-778`), so every
query sees summaries of future tokens: changing later tokens changes earlier
logits. No cache is returned either (`:1203`). The newer revision fixes both, but
in BF16 it keeps several norms in FP32 whose outputs then hit BF16 linear layers
and raise (router at `:1115` in that revision). In FP32 it runs, with no leak and a
returned cache.
*False positive?* The pinned leak is real. The newer-revision crash came from our
reduced config quantized on load; the published checkpoint is pre-quantized, so
confirm it also occurs through the normal `from_pretrained` path. *Check:* whether
"newer revision, FP32" is acceptable.

### Wrong results

**nllb_moe.** The router's one-hot mask is passed where expert indices are
expected and one-hot encoded again (`models/nllb_moe/modeling_nllb_moe.py:382`,
`:351`). Tokens routed to experts 2 and 3 are processed by experts 0 and 1. The
path is unconditional, so every sparse layer is affected. *False positive?* Unlikely;
the code contradicts itself. *Check:* whether a one-line reference fix is
acceptable, or whether this should be recorded as an upstream bug.

**sam_hq.** In the HQ mask decoder the transformer's output is assigned and then
overwritten (`models/sam_hq/modeling_sam_hq.py:989-1003`), so the upscaler receives
the pre-transformer embeddings. The authors' code (SysCV/sam-hq,
`mask_decoder_hq.py`) upscales the transformer output. *False positive?* Unlikely.
*Check:* whether a patched reference is acceptable.

**clvp.** Generation subtracts position embeddings 0-11 from the 12 conditioning
embeddings, expecting the decoder to add them back
(`models/clvp/modeling_clvp.py:1143-1151`), but passes position 0 for all 12 and
then positions 2, 3, 4 for the next tokens instead of 12, 13, 14 (`:1184`). The
pinned revision's causal mask also lets each prefill query see only one key; the
newer revision fixes the mask but not the positions. *False positive?* The
subtract-then-re-add design makes the intended positions unambiguous. *Check:*
compare against the original Tortoise TTS implementation before patching.

### Narrow or shared issues

**phi4_multimodal (vision).** Fractional patch coordinates are cast to BF16 before
bucketing (`models/phi4_multimodal/modeling_phi4_multimodal.py:352-356`), which
misplaces 64 patches in our test (none in FP32). Our implementation uses the same
formula, so it matches HF; this does not block a parity comparison. *Check:*
decide whether matching HF is acceptable or the entry needs FP32 coordinates. The
speech variant already passes.

**gemma4_assistant.** The assistant drafts tokens by argmax
(`generation/candidate_generator.py:1390`), but sampled speculative decoding
treats the draft's softmax as its proposal distribution (`generation/utils.py:3860`,
`:3864`). With equal draft and target distributions (0.6, 0.4), the first token was
the argmax 100 times out of 100. Only sampled assisted generation is affected;
the checkpoint's default generation config samples. The assistant's own forward
pass is unaffected. *Check:* whether evaluating the forward pass and greedy
assisted generation, and excluding sampled mode, is acceptable.

**sam3_video.** When an object is removed during hotstart
(`models/sam3_video/modeling_sam3_video.py:1498`), output assembly pairs the
surviving object IDs with the pre-removal masks and scores (`:1521`, `:1683`). The
survivor receives the removed object's mask. This was shown with the detector and
tracker outputs stubbed, not in a full model run. *Check:* whether realistic inputs
reach this path; if rarely, exclude it; otherwise patch the reference.

### Not an HF bug

**granite4_vision.** IBM publishes this model as remote code for transformers
4.57.6. That code imports `HybridMambaAttentionDynamicCache`, which neither revision
defines. The built-in class expects a Granite text config (`mlp_bias`,
`models/granite4_vision/modeling_granite4_vision.py:443`) but the published config
has a GraniteMoeHybrid text model, so construction fails. *Check:* choose between
IBM's code under transformers 4.57.6 (a different pin), the built-in class with a
compatible config (not the published model), or excluding the entry.
