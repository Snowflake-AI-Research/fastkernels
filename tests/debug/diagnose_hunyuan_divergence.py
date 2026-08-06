#!/usr/bin/env python3
"""Component-by-component divergence diagnostic: fastkernels vs vllm-omni for
HunyuanVideo-1.5 (T2V).

Modeled on ``diagnose_flux_divergence.py``.  Loads the fastkernels pipeline and
the vllm-omni reference components in a single process, feeds identical inputs,
and reports cosine similarity at every intermediate stage:

  Stage 1  initial noise (prepare_latents)
  Stage 2  MLLM text encoder (ours Qwen25VLTextEncoder vs HF Qwen2_5_VLTextModel)
  Stage 2b ByT5 text encoder (glyph path; zeros in plain T2V)
  Stage 3  transformer, per-submodule + per-block (ours vs vllm-omni reference)
  Stage 4  full multi-step denoise, identical text (ours transformer vs reference)
  Stage 5  A/B/C decomposition + VAE-decoded frame cosine:
             A = ours transformer + ours text
             B = ours transformer + HF  text
             C = reference transformer + HF text
           A-vs-B isolates the text encoder; B-vs-C isolates the transformer;
           A-vs-C is the end-to-end (what the bench measures).
  Stage 6  root-cause drill: MLLM valid-vs-padding split + weight-loading audit.

FINDING: initial noise is bit-identical; the MLLM text encoder is correct on
valid tokens (the Stage-2 cos~0.35 is padding-only, which the refiner drops);
the divergence is the transformer's ``context_embedder`` (token refiner). Its
``to_q/to_k/to_v`` are LEFT AT RANDOM INIT because ``load_weights`` uses
``break`` instead of ``continue`` when the stacked ``.to_qkv`` remap target is
absent (the refiner uses separate q/k/v, not packed). Fix: ``break`` -> ``continue``.

Run:  CUDA_VISIBLE_DEVICES=2 python tests/debug/diagnose_hunyuan_divergence.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
from copy import deepcopy
from glob import glob

import numpy as np
import torch

torch.set_grad_enabled(False)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
SEED = 42
# Small config to keep it fast (bench shape is 480x832 / 25f / more steps).
HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 5
NUM_STEPS = 4
GUIDANCE = 6.0
PROMPT = "A red fox trotting across a snowy field at dawn, cinematic lighting"
NEG_PROMPT = ""


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.detach().float() - b.detach().float()) ** 2).mean())


def report(name: str, a: torch.Tensor, b: torch.Tensor, indent: int = 0) -> float:
    prefix = "  " * indent
    if a.shape != b.shape:
        print(f"{prefix}{name:<52s}  SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)} <<<")
        # still try flattened cos if numel matches
        if a.numel() != b.numel():
            return 0.0
    c = cos_sim(a, b)
    m = mse(a, b)
    d = max_abs_diff(a, b)
    marker = "" if c > 0.999 else (" <<<" if c < 0.99 else " <")
    print(f"{prefix}{name:<52s}  cos={c:.6f}  mse={m:.2e}  max_diff={d:.4f}{marker}")
    return c


# ============================================================================
# Load fastkernels pipeline
# ============================================================================
print("=" * 90)
print("Loading fastkernels HunyuanVideo pipeline...")
print("=" * 90)

from fastkernels.infra.diffusion_engine import (
    _download_diffusion_model,
    _load_hunyuan_video_weights,
)
from fastkernels.tasks.baseline.L4.hunyuan_video import (
    HunyuanVideoConfig,
    HunyuanVideoPipeline,
)

model_path = _download_diffusion_model(MODEL)
kb_config = HunyuanVideoConfig.from_pretrained(model_path)
kb_pipe = HunyuanVideoPipeline(kb_config, model_path)
_load_hunyuan_video_weights(kb_pipe, model_path)
kb_pipe.transformer.to(device=DEVICE, dtype=DTYPE)
kb_pipe.text_encoder.to(device=DEVICE, dtype=DTYPE)
kb_pipe.text_encoder_2.to(device=DEVICE, dtype=DTYPE)
kb_pipe.vae.to(device=DEVICE)
kb_pipe.eval()

with open(os.path.join(model_path, "transformer", "config.json")) as f:
    TF_CFG = json.load(f)

print("  fastkernels pipeline ready.")


def _load_safetensors(directory):
    weights = []
    from safetensors import safe_open
    for sf_file in sorted(glob(os.path.join(directory, "*.safetensors"))):
        with safe_open(sf_file, "pt", "cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))
    return weights


# ============================================================================
# Load HF Qwen2.5-VL text encoder (this is EXACTLY what vllm-omni uses)
# ============================================================================
print("\n" + "=" * 90)
print("Loading HF Qwen2_5_VLTextModel (vllm-omni reference text encoder)...")
print("=" * 90)

from transformers import Qwen2_5_VLTextModel

hf_qwen = Qwen2_5_VLTextModel.from_pretrained(
    model_path, subfolder="text_encoder", local_files_only=True, torch_dtype=DTYPE,
).to(DEVICE)
hf_qwen.eval()
print("  HF Qwen2.5-VL text encoder ready.")


# ============================================================================
# STAGE 1: Initial noise (prepare_latents)
# ============================================================================
print("\n" + "=" * 90)
print("STAGE 1: INITIAL NOISE (prepare_latents, seed=%d)" % SEED)
print("=" * 90)

gen_a = torch.Generator(device=DEVICE).manual_seed(SEED)
latents_kb = kb_pipe.prepare_latents(1, HEIGHT, WIDTH, NUM_FRAMES, DTYPE, DEVICE, gen_a)
print(f"  ours latents shape = {tuple(latents_kb.shape)}  dtype={latents_kb.dtype}")

# Reference prepare_latents (pipeline_hunyuan_video_1_5.py) is byte-identical:
# randn_tensor(shape, generator, device, dtype) with the SAME shape formula.
from diffusers.utils.torch_utils import randn_tensor

shape = (
    1,
    kb_pipe.num_channels_latents,
    (NUM_FRAMES - 1) // kb_pipe.vae_scale_factor_temporal + 1,
    int(HEIGHT) // kb_pipe.vae_scale_factor_spatial,
    int(WIDTH) // kb_pipe.vae_scale_factor_spatial,
)
gen_b = torch.Generator(device=DEVICE).manual_seed(SEED)
latents_ref = randn_tensor(shape, generator=gen_b, device=torch.device(DEVICE), dtype=DTYPE)
print(f"  reference formula shape = {shape}")
report("initial noise (ours vs reference randn_tensor)", latents_kb, latents_ref)


# ============================================================================
# STAGE 2: MLLM text encoder (ours Qwen25VLTextEncoder vs HF Qwen2_5_VLTextModel)
# ============================================================================
print("\n" + "=" * 90)
print("STAGE 2: MLLM TEXT ENCODER (Qwen2.5-VL): ours vs HF reference")
print("=" * 90)

NUM_SKIP = 2  # num_hidden_layers_to_skip
CROP = kb_pipe.prompt_template_encode_start_idx  # 108


def _mllm_tokenize(prompt_list):
    formatted = [
        [{"role": "system", "content": kb_pipe.system_message},
         {"role": "user", "content": p if p else " "}]
        for p in prompt_list
    ]
    ti = kb_pipe.tokenizer.apply_chat_template(
        formatted, add_generation_prompt=True, tokenize=True, return_dict=True,
        padding="max_length",
        max_length=kb_pipe.tokenizer_max_length + kb_pipe.prompt_template_encode_start_idx,
        truncation=True, return_tensors="pt",
    )
    return ti.input_ids.to(DEVICE), ti.attention_mask.to(DEVICE)


def _mllm_embed_ours(prompt_list):
    ids, mask = _mllm_tokenize(prompt_list)
    out = kb_pipe.text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    emb = out.hidden_states[-(NUM_SKIP + 1)]
    emb = emb[:, CROP:]
    m = mask[:, CROP:]
    return emb.to(DTYPE), m


def _mllm_embed_hf(prompt_list):
    ids, mask = _mllm_tokenize(prompt_list)
    out = hf_qwen(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    emb = out.hidden_states[-(NUM_SKIP + 1)]
    emb = emb[:, CROP:]
    m = mask[:, CROP:]
    return emb.to(DTYPE), m


ids0, mask0 = _mllm_tokenize([PROMPT])
print(f"  tokenized ids shape = {tuple(ids0.shape)}, valid tokens = {int(mask0.sum())}")

kb_mllm_emb, kb_mllm_mask = _mllm_embed_ours([PROMPT])
hf_mllm_emb, hf_mllm_mask = _mllm_embed_hf([PROMPT])
print(f"  ours MLLM embed shape = {tuple(kb_mllm_emb.shape)}")
report("MLLM prompt_embeds hidden_states[-3] (ours vs HF)", kb_mllm_emb, hf_mllm_emb)
report("  MLLM attention_mask (ours vs HF)", kb_mllm_mask.float(), hf_mllm_mask.float())

# Compare the FULL last_hidden_state and a few individual hidden layers too.
ids0, mask0 = _mllm_tokenize([PROMPT])
kb_full = kb_pipe.text_encoder(input_ids=ids0, attention_mask=mask0, output_hidden_states=True)
hf_full = hf_qwen(input_ids=ids0, attention_mask=mask0, output_hidden_states=True)
print(f"  #hidden_states: ours={len(kb_full.hidden_states)} hf={len(hf_full.hidden_states)}")
for li in [1, len(hf_full.hidden_states) // 2, len(hf_full.hidden_states) - 3]:
    report(f"  hidden_states[{li}] full-seq (ours vs HF)",
           kb_full.hidden_states[li].to(DTYPE), hf_full.hidden_states[li].to(DTYPE), indent=1)


# ============================================================================
# STAGE 2b: ByT5 glyph encoder (zeros for plain T2V — no quoted glyph text)
# ============================================================================
print("\n" + "=" * 90)
print("STAGE 2b: ByT5 glyph encoder (text_encoder_2)")
print("=" * 90)
kb_byt5, kb_byt5_mask = kb_pipe._get_byte5_prompt_embeds([PROMPT], torch.device(DEVICE), DTYPE)
print(f"  ByT5 embed shape={tuple(kb_byt5.shape)}  nonzero={int((kb_byt5 != 0).sum())}  "
      f"mask_sum={int(kb_byt5_mask.sum())}  (0 => pure-T2V zero stream, no effect)")


# ============================================================================
# Build the vllm-omni reference transformer (guarded)
# ============================================================================
print("\n" + "=" * 90)
print("Building vllm-omni reference HunyuanVideo15Transformer3DModel...")
print("=" * 90)

vo_transformer = None
vo_ctx_mgrs = []
try:
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

    _mc = ModelConfig(model="Qwen/Qwen3-0.6B")
    vllm_config = VllmConfig(model_config=_mc)
    vllm_config.compilation_config.level = 0
    _c = set_current_vllm_config(vllm_config)
    _c.__enter__()
    vo_ctx_mgrs.append(_c)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    from vllm_omni.diffusion.distributed import parallel_state as vo_ps

    if not vo_ps.model_parallel_is_initialized():
        vo_ps.init_distributed_environment(world_size=1, rank=0, local_rank=0)
        vo_ps.initialize_model_parallel(sequence_parallel_size=1, tensor_parallel_size=1)

    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Transformer3DModel as VOTransformer,
    )

    class _StubParallel:
        sequence_parallel_size = 1
        ulysses_degree = 1
        ring_degree = 1
        allgather_degree = 1
        world_size = 1
        use_hsdp = False

    class _StubTF:
        # no num_layers attr -> transformer falls back to kwarg num_layers
        use_meanflow = False

    class _StubOD:
        tf_model_config = _StubTF()
        parallel_config = _StubParallel()
        quantization_config = None
        diffusion_attention_config = None
        model_class_name = "HunyuanVideo15Pipeline"
        enable_diffusion_pipeline_profiler = False

    _stub_od = _StubOD()

    vo_transformer = VOTransformer(
        od_config=_stub_od,
        in_channels=TF_CFG["in_channels"],
        out_channels=TF_CFG["out_channels"],
        num_attention_heads=TF_CFG["num_attention_heads"],
        attention_head_dim=TF_CFG["attention_head_dim"],
        num_layers=TF_CFG["num_layers"],
        num_refiner_layers=TF_CFG["num_refiner_layers"],
        mlp_ratio=TF_CFG["mlp_ratio"],
        patch_size=TF_CFG["patch_size"],
        patch_size_t=TF_CFG["patch_size_t"],
        qk_norm=TF_CFG["qk_norm"],
        text_embed_dim=TF_CFG["text_embed_dim"],
        text_embed_2_dim=TF_CFG["text_embed_2_dim"],
        image_embed_dim=TF_CFG["image_embed_dim"],
        rope_theta=TF_CFG["rope_theta"],
        rope_axes_dim=tuple(TF_CFG["rope_axes_dim"]),
        use_meanflow=TF_CFG["use_meanflow"],
    )
    tf_weights = _load_safetensors(os.path.join(model_path, "transformer"))
    vo_transformer.load_weights(tf_weights)
    vo_transformer.to(device=DEVICE, dtype=DTYPE)
    vo_transformer.eval()

    def _fwd_ctx():
        return set_forward_context(omni_diffusion_config=_stub_od)

    print("  reference transformer ready.")
except Exception as e:
    import traceback
    print(f"  !! Could not build reference transformer: {e}")
    traceback.print_exc()
    vo_transformer = None

    def _fwd_ctx():
        import contextlib
        return contextlib.nullcontext()


# ============================================================================
# Shared transformer inputs (from ours pipeline)
# ============================================================================
device = torch.device(DEVICE)
(
    prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
    neg_embeds, neg_mask, neg_embeds_2, neg_mask_2,
) = kb_pipe.encode_prompt(prompt=[PROMPT], device=device, dtype=DTYPE,
                          negative_prompt=NEG_PROMPT, do_classifier_free_guidance=True)

cond_latents, cmask = kb_pipe.prepare_cond_latents_and_mask(latents_kb, DTYPE, device)
image_embeds = torch.zeros(1, kb_pipe.vision_num_semantic_tokens, kb_pipe.vision_states_dim,
                           dtype=DTYPE, device=device)

sched = deepcopy(kb_pipe.scheduler)
sigmas = np.linspace(1.0, 0.0, NUM_STEPS + 1)[:-1]
sched.set_timesteps(sigmas=sigmas, device=device)
timesteps = sched.timesteps
print(f"\n  timesteps = {[round(float(t), 2) for t in timesteps]}")

lmi = torch.cat([latents_kb, cond_latents, cmask], dim=1)
t0 = timesteps[0]
ts0 = t0.expand(lmi.shape[0]).to(lmi.dtype)


# ============================================================================
# STAGE 3: Transformer, per-submodule + per-block (ours vs reference)
# ============================================================================
if vo_transformer is not None:
    print("\n" + "=" * 90)
    print("STAGE 3: TRANSFORMER SUBMODULES & BLOCKS (ours vs reference, identical inputs)")
    print("=" * 90)

    kbT = kb_pipe.transformer

    # --- 3a submodules ---
    kb_rope = kbT.rope(lmi)
    vo_rope = vo_transformer.rope(lmi)
    report("rope cos", kb_rope[0], vo_rope[0])
    report("rope sin", kb_rope[1], vo_rope[1])

    kb_temb = kbT.time_embed(ts0)
    vo_temb = vo_transformer.time_embed(ts0)
    report("time_embed (temb)", kb_temb, vo_temb)

    kb_hs = kbT.x_embedder(lmi)
    vo_hs = vo_transformer.x_embedder(lmi)
    report("x_embedder", kb_hs, vo_hs)

    kb_ctx = kbT.context_embedder(prompt_embeds, ts0, prompt_embeds_mask)
    vo_ctx = vo_transformer.context_embedder(prompt_embeds, ts0, prompt_embeds_mask)
    report("context_embedder (token refiner)", kb_ctx, vo_ctx)

    kb_ctx2 = kbT.context_embedder_2(prompt_embeds_2)
    vo_ctx2 = vo_transformer.context_embedder_2(prompt_embeds_2)
    report("context_embedder_2 (ByT5 proj)", kb_ctx2, vo_ctx2)

    kb_img = kbT.image_embedder(image_embeds)
    vo_img = vo_transformer.image_embedder(image_embeds)
    report("image_embedder", kb_img, vo_img)

    # --- 3b conditioning merge (ours module vs reference inline reorder) ---
    # feed IDENTICAL context_embedder outputs (ours) into both merge impls.
    print("\n  -- conditioning merge (fed identical ctx outputs) --")
    # ours path (zero image stream for T2V, as ours forward does):
    is_t2v = bool(torch.all(image_embeds == 0))
    kb_img_stream = kb_img * 0.0 if is_t2v else kb_img
    kb_img_mask = torch.zeros((1, kb_img.shape[1]), dtype=prompt_embeds_mask.dtype, device=device)
    merged_kb, merged_mask_kb = kbT.conditioning_merge(
        kb_ctx.clone(), prompt_embeds_mask, kb_ctx2.clone(), prompt_embeds_mask_2,
        kb_img_stream.clone(), kb_img_mask,
    )

    # reference inline reorder (copied from hunyuan_video_15_transformer.py:725-813),
    # driven with the SAME ctx outputs + reference cond_type_embed.
    ehs = kb_ctx.clone() + vo_transformer.cond_type_embed(
        torch.zeros_like(kb_ctx[:, :, 0], dtype=torch.long))
    ehs2 = kb_ctx2.clone() + vo_transformer.cond_type_embed(
        torch.ones_like(kb_ctx2[:, :, 0], dtype=torch.long))
    ehs3 = (kb_img * 0.0 if is_t2v else kb_img) + vo_transformer.cond_type_embed(
        2 * torch.ones_like(kb_img[:, :, 0], dtype=torch.long))
    m1 = prompt_embeds_mask.bool()
    m2 = prompt_embeds_mask_2.bool()
    m3 = kb_img_mask.bool()
    new_h, new_m = [], []
    for text, tm, text2, tm2, image, im in zip(ehs, m1, ehs2, m2, ehs3, m3):
        new_h.append(torch.cat([image[im], text2[tm2], text[tm],
                                image[~im], torch.zeros_like(text2[~tm2]),
                                torch.zeros_like(text[~tm])], dim=0))
        new_m.append(torch.cat([im[im], tm2[tm2], tm[tm], im[~im], tm2[~tm2], tm[~tm]], dim=0))
    merged_vo = torch.stack(new_h)
    merged_mask_vo = torch.stack(new_m)
    max_valid = int(merged_mask_vo.sum(dim=1).max().item())
    if max_valid < merged_mask_vo.shape[1]:
        merged_vo = merged_vo[:, :max_valid]
        merged_mask_vo = merged_mask_vo[:, :max_valid]
    ref_mask_is_none = bool(merged_mask_vo.all())

    print(f"  ours merged shape={tuple(merged_kb.shape)} mask={'None' if merged_mask_kb is None else tuple(merged_mask_kb.shape)}")
    print(f"  ref  merged shape={tuple(merged_vo.shape)} mask={'None(all-valid)' if ref_mask_is_none else tuple(merged_mask_vo.shape)}")
    report("merged encoder_hidden_states (ours-merge vs ref-merge)", merged_kb, merged_vo)

    # --- 3c per-block (feed identical hs/enc/temb/rope to both stacks) ---
    print("\n  -- per-block (identical start; each stack evolves) --")
    a_hs, a_enc = kb_hs.clone(), merged_kb.clone()
    b_hs, b_enc = kb_hs.clone(), merged_kb.clone()
    a_mask = merged_mask_kb
    n_blocks = len(kbT.transformer_blocks)
    with _fwd_ctx():
        for i in range(n_blocks):
            a_hs, a_enc = kbT.transformer_blocks[i](a_hs, a_enc, kb_temb, a_mask, kb_rope)
            b_hs, b_enc = vo_transformer.transformer_blocks[i](
                b_hs, b_enc, kb_temb, a_mask, kb_rope)
            if i < 5 or i >= n_blocks - 2:
                report(f"block[{i:2d}] hidden ", a_hs, b_hs, indent=1)
                report(f"block[{i:2d}] encoder", a_enc, b_enc, indent=1)
            elif i == 5:
                print(f"    ... (blocks 5-{n_blocks - 3} omitted) ...")

    # --- 3d output ---
    kb_norm = kbT.norm_out(a_hs, kb_temb)
    vo_norm = vo_transformer.norm_out(b_hs, kb_temb)
    report("norm_out", kb_norm, vo_norm)
    report("proj_out", kbT.proj_out(kb_norm), vo_transformer.proj_out(vo_norm))

    # --- 3e full single-step forward on identical inputs ---
    print("\n  -- full single forward (identical inputs) --")
    kb_noise = kbT(hidden_states=lmi, timestep=ts0, encoder_hidden_states=prompt_embeds,
                   encoder_attention_mask=prompt_embeds_mask,
                   encoder_hidden_states_2=prompt_embeds_2,
                   encoder_attention_mask_2=prompt_embeds_mask_2,
                   image_embeds=image_embeds, return_dict=False)[0]
    with _fwd_ctx():
        vo_noise = vo_transformer(hidden_states=lmi, timestep=ts0,
                                  encoder_hidden_states=prompt_embeds,
                                  encoder_attention_mask=prompt_embeds_mask,
                                  encoder_hidden_states_2=prompt_embeds_2,
                                  encoder_attention_mask_2=prompt_embeds_mask_2,
                                  image_embeds=image_embeds, return_dict=False)[0]
    report("noise_pred (full forward, ours vs reference)", kb_noise, vo_noise)


# ============================================================================
# Denoise helper (mirrors HunyuanVideoPipeline.diffuse, with CFG)
# ============================================================================
def run_denoise(transformer, pe, pm, pe2, pm2, ne, nm, ne2, nm2, latents0, use_fwd_ctx=False):
    latents = latents0.clone()
    cl, mk = kb_pipe.prepare_cond_latents_and_mask(latents, DTYPE, device)
    imemb = torch.zeros(1, kb_pipe.vision_num_semantic_tokens, kb_pipe.vision_states_dim,
                        dtype=DTYPE, device=device)
    s = deepcopy(kb_pipe.scheduler)
    s.set_timesteps(sigmas=sigmas, device=device)
    import contextlib
    ctx = _fwd_ctx() if use_fwd_ctx else contextlib.nullcontext()
    with ctx:
        for t in s.timesteps:
            li = torch.cat([latents, cl, mk], dim=1)
            tt = t.expand(li.shape[0]).to(li.dtype)
            np_ = transformer(hidden_states=li, timestep=tt, encoder_hidden_states=pe,
                              encoder_attention_mask=pm, encoder_hidden_states_2=pe2,
                              encoder_attention_mask_2=pm2, image_embeds=imemb,
                              return_dict=False)[0]
            nu_ = transformer(hidden_states=li, timestep=tt, encoder_hidden_states=ne,
                              encoder_attention_mask=nm, encoder_hidden_states_2=ne2,
                              encoder_attention_mask_2=nm2, image_embeds=imemb,
                              return_dict=False)[0]
            noise = nu_ + GUIDANCE * (np_ - nu_)
            latents = s.step(noise, t, latents, return_dict=False)[0]
    return latents


def decode(latents):
    lat = latents.to(kb_pipe.vae.dtype) / kb_pipe.vae.config.scaling_factor
    return kb_pipe.vae.decode(lat, return_dict=False)[0]


# ============================================================================
# STAGE 4: full multi-step denoise, IDENTICAL text (ours transformer vs reference)
# ============================================================================
if vo_transformer is not None:
    print("\n" + "=" * 90)
    print("STAGE 4: FULL DENOISE, identical (ours) text — ours transformer vs reference")
    print("=" * 90)
    lat_kb = run_denoise(kb_pipe.transformer,
                         prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
                         neg_embeds, neg_mask, neg_embeds_2, neg_mask_2, latents_kb)
    lat_vo = run_denoise(vo_transformer,
                         prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
                         neg_embeds, neg_mask, neg_embeds_2, neg_mask_2, latents_kb,
                         use_fwd_ctx=True)
    report("final latents (ours-tf vs ref-tf, same text)", lat_kb, lat_vo)
    report("decoded frames (ours-tf vs ref-tf, same text)", decode(lat_kb), decode(lat_vo))


# ============================================================================
# STAGE 5: A/B/C decomposition (text encoder vs transformer, end-to-end + decode)
# ============================================================================
print("\n" + "=" * 90)
print("STAGE 5: A/B/C DECOMPOSITION (end-to-end + VAE decode)")
print("=" * 90)
print("  A = ours transformer + ours text | B = ours transformer + HF text | "
      "C = ref transformer + HF text")

# HF-text embeddings (positive + negative), ByT5/mask taken from ours (identical for T2V).
hf_pe, hf_pm = _mllm_embed_hf([PROMPT])
hf_ne, hf_nm = _mllm_embed_hf([NEG_PROMPT])
hf_pm = hf_pm.to(DTYPE)
hf_nm = hf_nm.to(DTYPE)

# A: ours transformer, ours text
lat_A = run_denoise(kb_pipe.transformer,
                    prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
                    neg_embeds, neg_mask, neg_embeds_2, neg_mask_2, latents_kb)
# B: ours transformer, HF text
lat_B = run_denoise(kb_pipe.transformer,
                    hf_pe, hf_pm, prompt_embeds_2, prompt_embeds_mask_2,
                    hf_ne, hf_nm, neg_embeds_2, neg_mask_2, latents_kb)

frames_A = decode(lat_A)
frames_B = decode(lat_B)
print()
report("A vs B  latents  (TEXT-ENCODER effect)", lat_A, lat_B)
report("A vs B  frames   (TEXT-ENCODER effect)", frames_A, frames_B)

if vo_transformer is not None:
    # C: reference transformer, HF text
    lat_C = run_denoise(vo_transformer,
                        hf_pe, hf_pm, prompt_embeds_2, prompt_embeds_mask_2,
                        hf_ne, hf_nm, neg_embeds_2, neg_mask_2, latents_kb,
                        use_fwd_ctx=True)
    frames_C = decode(lat_C)
    report("B vs C  latents  (TRANSFORMER effect, same HF text)", lat_B, lat_C)
    report("B vs C  frames   (TRANSFORMER effect, same HF text)", frames_B, frames_C)
    report("A vs C  latents  (END-TO-END: ours vs reference)", lat_A, lat_C)
    report("A vs C  frames   (END-TO-END: ours vs reference)", frames_A, frames_C)


# ============================================================================
# STAGE 6: ROOT-CAUSE DRILL — MLLM valid/padding split + weight-loading audit
# ============================================================================
print("\n" + "=" * 90)
print("STAGE 6: ROOT-CAUSE DRILL")
print("=" * 90)

# 6a. The Stage-2 cos~0.35 is padding-only; VALID MLLM tokens actually agree.
valid = kb_mllm_mask[0].bool()
print(f"  MLLM valid tokens = {int(valid.sum())} / {kb_mllm_mask.shape[1]}")
report("MLLM VALID tokens only (ours vs HF)", kb_mllm_emb[:, valid], hf_mllm_emb[:, valid], indent=1)
report("MLLM PADDING tokens only (ours vs HF)", kb_mllm_emb[:, ~valid], hf_mllm_emb[:, ~valid], indent=1)
print("  => text encoder is CORRECT on valid tokens; Stage-2 low cos is padding "
      "(dropped by the refiner truncation).")

# 6b. Weight-loading audit: which transformer params ours FAILED to load
# (checkpoint has the exact key+shape but ours differs from it => left at init).
print("\n  -- weight-loading audit (ours transformer vs checkpoint) --")
import re
from collections import Counter
_ckpt = {}
from safetensors import safe_open
for _fp in sorted(glob(os.path.join(model_path, "transformer", "*.safetensors"))):
    with safe_open(_fp, "pt", "cpu") as _f:
        for _k in _f.keys():
            _ckpt[_k] = _f.get_tensor(_k)
_sd = dict(kb_pipe.transformer.named_parameters())
_unloaded = []
for _k, _p in _sd.items():
    if _k in _ckpt and tuple(_p.shape) == tuple(_ckpt[_k].shape):
        # Compare in the PARAM's dtype (cast ckpt the same way loading does),
        # else bf16-rounding of fp32-checkpoint weights yields false positives.
        _ck = _ckpt[_k].to(_p.dtype).float().cpu()
        if (_p.detach().float().cpu() - _ck).abs().max().item() > 1e-2:
            _unloaded.append(_k)
print(f"  params with matching ckpt key+shape but NOT loaded: {len(_unloaded)}")
for _name, _n in sorted(Counter(re.sub(r'\.\d+\.', '.N.', k) for k in _unloaded).items()):
    print(f"    {_n:3d}x  {_name}  <<< left at random init")
print("\n  ROOT CAUSE: HunyuanVideo15Transformer3DModel.load_weights (hunyuan_video.py)")
print("  uses `break` (not `continue`) when the stacked .to_q/.to_k/.to_v -> .to_qkv")
print("  remap target is absent, so the token refiner's SEPARATE to_q/to_k/to_v")
print("  (context_embedder.token_refiner.*.attn.*) never reach the fallback loader.")
print("  Reference (hunyuan_video_15_transformer.py:883-884) uses `continue`.")

print("\n" + "=" * 90)
print("DONE.")
print("=" * 90)

for _c in reversed(vo_ctx_mgrs):
    try:
        _c.__exit__(None, None, None)
    except Exception:
        pass
