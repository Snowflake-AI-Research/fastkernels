#!/usr/bin/env python3
"""Component-by-component divergence diagnostic between kb-nano and vllm-omni.

Loads both pipelines in a single process, feeds identical inputs, and
reports cosine similarity at every intermediate stage of the forward pass
for a single image at 512x512.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from contextlib import contextmanager
from collections import OrderedDict

torch.set_grad_enabled(False)
DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL = "black-forest-labs/FLUX.1-dev"
SEED = 42
HEIGHT, WIDTH = 512, 512
NUM_STEPS = 28
GUIDANCE = 3.5
PROMPT = "A majestic eagle soaring over snow-capped mountains at sunset"


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.detach().float() - b.detach().float()) ** 2).mean())


def report(name: str, a: torch.Tensor, b: torch.Tensor, indent: int = 0):
    prefix = "  " * indent
    c = cos_sim(a, b)
    m = mse(a, b)
    d = max_abs_diff(a, b)
    marker = "" if c > 0.999 else (" <<<" if c < 0.99 else " <")
    print(f"{prefix}{name:<55s}  cos={c:.6f}  mse={m:.2e}  max_diff={d:.4f}{marker}")
    return c


# ============================================================================
# Load kb-nano pipeline
# ============================================================================
print("=" * 80)
print("Loading kb-nano pipeline...")
print("=" * 80)

from kb_nano.infra.diffusion_engine import DiffusionEngine, _download_flux_model, _load_flux_weights
from kb_nano.tasks.baseline.L4.flux import FluxConfig, FluxPipeline, DiffusionSamplingParams, _calculate_shift

model_path = _download_flux_model(MODEL)
config = FluxConfig.from_pretrained(model_path)
kb_pipe = FluxPipeline(config, model_path)
_load_flux_weights(kb_pipe, model_path)
kb_pipe.to(device=DEVICE, dtype=DTYPE)
kb_pipe.text_encoder.to(device=DEVICE)
kb_pipe.text_encoder_2.to(device=DEVICE)
kb_pipe.vae.to(device=DEVICE)
kb_pipe.eval()

# ============================================================================
# Load vllm-omni pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Loading vllm-omni pipeline...")
print("=" * 80)

from vllm.config import VllmConfig, set_current_vllm_config, ModelConfig

_mc = ModelConfig(model="Qwen/Qwen3-0.6B")
vllm_config = VllmConfig(model_config=_mc)
vllm_config.compilation_config.level = 0

_vllm_ctx = set_current_vllm_config(vllm_config)
_vllm_ctx.__enter__()

from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1)

from vllm_omni.diffusion.models.flux.flux_transformer import (
    FluxTransformer2DModel as VOFluxTransformer2DModel,
    FluxPosEmbed as VOFluxPosEmbed,
    FluxTransformerBlock as VOFluxTransformerBlock,
    FluxSingleTransformerBlock as VOFluxSingleTransformerBlock,
)
from vllm_omni.diffusion.models.t5_encoder.t5_encoder import T5EncoderModel as VOT5EncoderModel

from transformers import AutoConfig, CLIPTextModel as HFCLIPTextModel, CLIPTokenizer
import json

tf_config_path = os.path.join(model_path, "transformer", "config.json")
with open(tf_config_path) as f:
    tf_data = json.load(f)

vo_transformer = VOFluxTransformer2DModel(
    od_config=None,
    in_channels=tf_data.get("in_channels", 64),
    out_channels=tf_data.get("out_channels", None),
    num_layers=tf_data.get("num_layers", 19),
    num_single_layers=tf_data.get("num_single_layers", 38),
    attention_head_dim=tf_data.get("attention_head_dim", 128),
    num_attention_heads=tf_data.get("num_attention_heads", 24),
    joint_attention_dim=tf_data.get("joint_attention_dim", 4096),
    pooled_projection_dim=tf_data.get("pooled_projection_dim", 768),
    guidance_embeds=tf_data.get("guidance_embeds", True),
    axes_dims_rope=tuple(tf_data.get("axes_dims_rope", [16, 56, 56])),
)

# Load transformer weights
from glob import glob
from safetensors import safe_open

def load_safetensors(directory):
    weights = []
    for sf_file in sorted(glob(os.path.join(directory, "*.safetensors"))):
        with safe_open(sf_file, "pt", "cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))
    return weights

transformer_weights = load_safetensors(os.path.join(model_path, "transformer"))
vo_transformer.load_weights(transformer_weights)
vo_transformer.to(device=DEVICE, dtype=DTYPE)
vo_transformer.eval()

# Patch vllm-omni T5 load_weights (same fix as bench_vllm_omni.py)
import inspect
_vo_src = inspect.getsource(VOT5EncoderModel.load_weights)
if 'name.replace(f".{weight_name}."' not in _vo_src:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    _orig_load = VOT5EncoderModel.load_weights
    def _patched_load_weights(self, weights):
        stacked_params_mapping = [
            ("qkv_proj", "q", "q"), ("qkv_proj", "k", "k"), ("qkv_proj", "v", "v"),
            ("wi", "wi_0", 0), ("wi", "wi_1", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                lookup_name = name.replace(f".{weight_name}.", f".{param_name}.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params
    VOT5EncoderModel.load_weights = _patched_load_weights
    print("  [diag] Patched vllm-omni T5EncoderModel.load_weights")

# Build and load vllm-omni T5
t5_config = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder_2")
vo_t5 = VOT5EncoderModel(t5_config)
t5_weights = load_safetensors(os.path.join(model_path, "text_encoder_2"))
vo_t5.load_weights(t5_weights)
vo_t5.to(device=DEVICE, dtype=DTYPE)
vo_t5.eval()

# Load HuggingFace CLIPTextModel (what vllm-omni uses)
hf_clip = HFCLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder", local_files_only=True,
)
hf_clip.to(device=DEVICE)
hf_clip.eval()

print("\nAll pipelines loaded successfully.")

# ============================================================================
# STAGE 1: Text encoders — CLIP
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1a: CLIP TEXT ENCODER")
print("=" * 80)

clip_inputs = kb_pipe.tokenizer(
    [PROMPT], padding="max_length", max_length=kb_pipe.tokenizer_max_length,
    truncation=True, return_tensors="pt",
)
clip_ids = clip_inputs.input_ids.to(DEVICE)

kb_clip_output = kb_pipe.text_encoder(clip_ids, output_hidden_states=False)
hf_clip_output = hf_clip(clip_ids, output_hidden_states=False)

kb_pooled = kb_clip_output.pooler_output.to(dtype=DTYPE, device=DEVICE)
hf_pooled = hf_clip_output.pooler_output.to(dtype=DTYPE, device=DEVICE)

kb_clip_hidden = kb_clip_output.last_hidden_state.to(dtype=DTYPE, device=DEVICE)
hf_clip_hidden = hf_clip_output.last_hidden_state.to(dtype=DTYPE, device=DEVICE)

report("CLIP last_hidden_state (kb vs hf)", kb_clip_hidden, hf_clip_hidden)
report("CLIP pooler_output (kb vs hf)", kb_pooled, hf_pooled)

# ============================================================================
# STAGE 1b: T5 Text encoder
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1b: T5 TEXT ENCODER")
print("=" * 80)

t5_inputs = kb_pipe.tokenizer_2(
    [PROMPT], padding="max_length", max_length=512,
    truncation=True, return_tensors="pt",
)
t5_ids = t5_inputs.input_ids.to(DEVICE)

kb_t5_out = kb_pipe.text_encoder_2(t5_ids)[0].to(dtype=DTYPE, device=DEVICE)
vo_t5_out = vo_t5(t5_ids)[0].to(dtype=DTYPE, device=DEVICE)

report("T5 encoder output", kb_t5_out, vo_t5_out)

# Use kb-nano's text embeddings as canonical input for the rest
prompt_embeds = kb_t5_out.clone()
pooled_prompt_embeds = kb_pooled.clone()

# Also run with hf CLIP pooled to see full end-to-end impact
pooled_from_hf = hf_pooled.clone()

# ============================================================================
# STAGE 1c: Full encode_prompt comparison (end-to-end)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1c: FULL ENCODE_PROMPT (end-to-end)")
print("=" * 80)

# What kb-nano's encode_prompt produces
kb_prompt_embeds, kb_pooled_final, kb_text_ids = kb_pipe.encode_prompt(
    prompt=[PROMPT], num_images_per_prompt=1, max_sequence_length=512,
)

# What vllm-omni would produce (using HF CLIP + vllm T5)
hf_pooled_final = hf_pooled.to(dtype=DTYPE, device=DEVICE)
# vllm-omni uses the same T5 encoder, so vo_t5_out should match
vo_prompt_embeds = vo_t5_out.clone()

report("encode_prompt: prompt_embeds (T5)", kb_prompt_embeds, vo_prompt_embeds)
report("encode_prompt: pooled (CLIP)", kb_pooled_final, hf_pooled_final)

# ============================================================================
# STAGE 2: Latent preparation & timesteps
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 2: LATENT PREPARATION & TIMESTEPS")
print("=" * 80)

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
num_channels_latents = kb_pipe.transformer.in_channels // 4
latents, latent_image_ids = kb_pipe.prepare_latents(
    1, num_channels_latents, HEIGHT, WIDTH, DTYPE, DEVICE, generator,
)
print(f"Latents: shape={latents.shape}")
print(f"Latent image IDs: shape={latent_image_ids.shape}")

timesteps, _ = kb_pipe.prepare_timesteps(NUM_STEPS, None, latents.shape[1])
print(f"Timesteps: {timesteps.shape}, first={timesteps[0]:.4f}, last={timesteps[-1]:.4f}")

# Text IDs
text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=DEVICE, dtype=DTYPE)

# Guidance
guidance = torch.full([1], GUIDANCE, dtype=DTYPE, device=DEVICE).expand(latents.shape[0])

# vllm-omni creates guidance in float32, then the forward casts it
vo_guidance = torch.full([1], GUIDANCE, dtype=torch.float32, device=DEVICE).expand(latents.shape[0])

# ============================================================================
# STAGE 3: Transformer forward — first denoising step only
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 3: TRANSFORMER FORWARD (first timestep)")
print("=" * 80)

# Prepare identical inputs for both transformers
t = timesteps[0]
timestep_input = t.expand(latents.shape[0]).to(device=DEVICE, dtype=DTYPE)

# ---- Pre-transformer embeddings ----
# x_embedder
kb_hidden = kb_pipe.transformer.x_embedder(latents)
vo_hidden = vo_transformer.x_embedder(latents)
report("x_embedder output", kb_hidden, vo_hidden)

# Let's compute temb directly
kb_ts_scaled = timestep_input.to(dtype=kb_hidden.dtype) * 1000
vo_ts_scaled = timestep_input.to(dtype=vo_hidden.dtype) * 1000

kb_guidance_scaled = guidance.to(dtype=kb_hidden.dtype) * 1000
vo_guidance_scaled = vo_guidance.to(dtype=vo_hidden.dtype) * 1000

# time_text_embed
kb_temb = kb_pipe.transformer.time_text_embed(kb_ts_scaled, kb_guidance_scaled, pooled_prompt_embeds)
vo_temb = vo_transformer.time_text_embed(vo_ts_scaled, vo_guidance_scaled, pooled_prompt_embeds)
report("time_text_embed (temb)", kb_temb, vo_temb)

# Check guidance dtype difference impact
report("  guidance_scaled kb vs vo", kb_guidance_scaled, vo_guidance_scaled.to(DTYPE))

# context_embedder
kb_ctx = kb_pipe.transformer.context_embedder(prompt_embeds)
vo_ctx = vo_transformer.context_embedder(prompt_embeds)
report("context_embedder output", kb_ctx, vo_ctx)

# Position embeddings
txt_ids = text_ids
img_ids = latent_image_ids
if txt_ids.ndim == 3:
    txt_ids = txt_ids[0]
if img_ids.ndim == 3:
    img_ids = img_ids[0]
ids = torch.cat((txt_ids, img_ids), dim=0)

kb_rope = kb_pipe.transformer.pos_embed(ids)
vo_rope = vo_transformer.pos_embed(ids)
report("pos_embed cos", kb_rope[0], vo_rope[0])
report("pos_embed sin", kb_rope[1], vo_rope[1])

# ============================================================================
# STAGE 4: Dual-stream transformer blocks
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 4: DUAL-STREAM TRANSFORMER BLOCKS")
print("=" * 80)

kb_hs = kb_hidden.clone()
vo_hs = vo_hidden.clone()
kb_enc = kb_ctx.clone()
vo_enc = vo_ctx.clone()

for i, (kb_block, vo_block) in enumerate(
    zip(kb_pipe.transformer.transformer_blocks, vo_transformer.transformer_blocks)
):
    kb_enc_out, kb_hs_out = kb_block(
        hidden_states=kb_hs, encoder_hidden_states=kb_enc,
        temb=kb_temb, image_rotary_emb=kb_rope, joint_attention_kwargs={},
    )
    vo_enc_out, vo_hs_out = vo_block(
        hidden_states=vo_hs, encoder_hidden_states=vo_enc,
        temb=vo_temb, image_rotary_emb=vo_rope, joint_attention_kwargs={},
    )

    if i < 5 or i >= len(kb_pipe.transformer.transformer_blocks) - 2:
        c_hs = report(f"dual_block[{i:2d}] hidden_states", kb_hs_out, vo_hs_out, indent=1)
        c_enc = report(f"dual_block[{i:2d}] encoder_hidden_states", kb_enc_out, vo_enc_out, indent=1)
    elif i == 5:
        print("    ... (blocks 5-16 omitted, printing first/last) ...")

    kb_hs = kb_hs_out
    vo_hs = vo_hs_out
    kb_enc = kb_enc_out
    vo_enc = vo_enc_out

# ============================================================================
# STAGE 5: Single-stream transformer blocks
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 5: SINGLE-STREAM TRANSFORMER BLOCKS")
print("=" * 80)

for i, (kb_block, vo_block) in enumerate(
    zip(kb_pipe.transformer.single_transformer_blocks, vo_transformer.single_transformer_blocks)
):
    kb_enc, kb_hs = kb_block(
        hidden_states=kb_hs, encoder_hidden_states=kb_enc,
        temb=kb_temb, image_rotary_emb=kb_rope, joint_attention_kwargs={},
    )
    vo_enc, vo_hs = vo_block(
        hidden_states=vo_hs, encoder_hidden_states=vo_enc,
        temb=vo_temb, image_rotary_emb=vo_rope, joint_attention_kwargs={},
    )

    if i < 3 or i >= len(kb_pipe.transformer.single_transformer_blocks) - 2:
        c_hs = report(f"single_block[{i:2d}] hidden_states", kb_hs, vo_hs, indent=1)
        c_enc = report(f"single_block[{i:2d}] encoder_hidden_states", kb_enc, vo_enc, indent=1)
    elif i == 3:
        print("    ... (blocks 3-35 omitted, printing first/last) ...")

# ============================================================================
# STAGE 6: Output norm & projection
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 6: OUTPUT NORM & PROJECTION")
print("=" * 80)

kb_normed = kb_pipe.transformer.norm_out(kb_hs, kb_temb)
vo_normed = vo_transformer.norm_out(vo_hs, vo_temb)
report("norm_out", kb_normed, vo_normed)

kb_out = kb_pipe.transformer.proj_out(kb_normed)
vo_out = vo_transformer.proj_out(vo_normed)
report("proj_out (noise prediction)", kb_out, vo_out)

# ============================================================================
# STAGE 7: Full denoising loop (all 28 steps) — with identical inputs
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 7: FULL DENOISING LOOP (identical inputs, all steps)")
print("=" * 80)

from copy import deepcopy
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

# Reset latents with same seed
generator = torch.Generator(device=DEVICE).manual_seed(SEED)
latents_kb, _ = kb_pipe.prepare_latents(
    1, num_channels_latents, HEIGHT, WIDTH, DTYPE, DEVICE, generator,
)
latents_vo = latents_kb.clone()

# Fresh schedulers
sched_kb = deepcopy(kb_pipe.scheduler)
sched_vo = deepcopy(kb_pipe.scheduler)

sigmas = np.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
image_seq_len = latents_kb.shape[1]
mu = _calculate_shift(
    image_seq_len,
    sched_kb.config.get("base_image_seq_len", 256),
    sched_kb.config.get("max_image_seq_len", 4096),
    sched_kb.config.get("base_shift", 0.5),
    sched_kb.config.get("max_shift", 1.15),
)
sched_kb.set_timesteps(sigmas=sigmas, mu=mu)
sched_vo.set_timesteps(sigmas=sigmas, mu=mu)
timesteps = sched_kb.timesteps

sched_kb.set_begin_index(0)
sched_vo.set_begin_index(0)

for step_idx, t in enumerate(timesteps):
    ts = t.expand(latents_kb.shape[0]).to(device=DEVICE, dtype=DTYPE)

    # kb-nano forward
    kb_noise = kb_pipe.transformer(
        hidden_states=latents_kb, timestep=ts / 1000, guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=latent_image_ids,
        joint_attention_kwargs={}, return_dict=False,
    )
    if isinstance(kb_noise, tuple):
        kb_noise = kb_noise[0]

    # vllm-omni forward
    vo_noise = vo_transformer(
        hidden_states=latents_vo, timestep=ts / 1000, guidance=vo_guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=latent_image_ids,
        joint_attention_kwargs={}, return_dict=False,
    )
    if isinstance(vo_noise, tuple):
        vo_noise = vo_noise[0]

    c = report(f"step {step_idx:2d} (t={t:.4f}) noise_pred", kb_noise, vo_noise, indent=1)

    # Scheduler step
    latents_kb = sched_kb.step(kb_noise, t, latents_kb, return_dict=False)[0]
    latents_vo = sched_vo.step(vo_noise, t, latents_vo, return_dict=False)[0]

    c_lat = report(f"step {step_idx:2d} (t={t:.4f}) latents   ", latents_kb, latents_vo, indent=1)

print("\n" + "=" * 80)
print("STAGE 7 result: Denoised latents comparison (same text embs)")
print("=" * 80)
report("Final denoised latents", latents_kb, latents_vo)

# ============================================================================
# STAGE 8: Full end-to-end with each pipeline's OWN text encoders
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 8: FULL END-TO-END (each pipeline uses its own text encoders)")
print("=" * 80)
print("  This replicates what the benchmark does: each engine runs independently.")

# kb-nano full pipeline
generator_kb = torch.Generator(device=DEVICE).manual_seed(SEED)
kb_full_prompt_embeds, kb_full_pooled, kb_full_text_ids = kb_pipe.encode_prompt(
    prompt=[PROMPT], num_images_per_prompt=1, max_sequence_length=512,
)
latents_kb_full, latent_image_ids_kb = kb_pipe.prepare_latents(
    1, num_channels_latents, HEIGHT, WIDTH, DTYPE, DEVICE, generator_kb,
)

# vllm-omni text encoding (HF CLIP + vllm T5)
clip_ids_vo = clip_ids.clone()
hf_clip_out_vo = hf_clip(clip_ids_vo, output_hidden_states=False)
vo_full_pooled = hf_clip_out_vo.pooler_output.to(dtype=DTYPE, device=DEVICE)
vo_full_prompt_embeds = vo_t5(t5_ids)[0].to(dtype=DTYPE, device=DEVICE)
latents_vo_full = latents_kb_full.clone()  # same latents

report("end-to-end: prompt_embeds (T5)", kb_full_prompt_embeds, vo_full_prompt_embeds)
report("end-to-end: pooled (CLIP)", kb_full_pooled, vo_full_pooled)

# Full denoising with each pipeline's own text embeddings
vo_full_text_ids = torch.zeros(vo_full_prompt_embeds.shape[1], 3).to(device=DEVICE, dtype=DTYPE)

sched_kb2 = deepcopy(kb_pipe.scheduler)
sched_vo2 = deepcopy(kb_pipe.scheduler)
sched_kb2.set_timesteps(sigmas=sigmas, mu=mu)
sched_vo2.set_timesteps(sigmas=sigmas, mu=mu)
sched_kb2.set_begin_index(0)
sched_vo2.set_begin_index(0)
timesteps2 = sched_kb2.timesteps

kb_guidance_e2e = torch.full([1], GUIDANCE, dtype=DTYPE, device=DEVICE).expand(latents_kb_full.shape[0])
vo_guidance_e2e = torch.full([1], GUIDANCE, dtype=torch.float32, device=DEVICE).expand(latents_vo_full.shape[0])

for step_idx, t in enumerate(timesteps2):
    ts = t.expand(latents_kb_full.shape[0]).to(device=DEVICE, dtype=DTYPE)

    kb_n = kb_pipe.transformer(
        hidden_states=latents_kb_full, timestep=ts / 1000, guidance=kb_guidance_e2e,
        pooled_projections=kb_full_pooled,
        encoder_hidden_states=kb_full_prompt_embeds,
        txt_ids=kb_full_text_ids, img_ids=latent_image_ids_kb,
        joint_attention_kwargs={}, return_dict=False,
    )
    if isinstance(kb_n, tuple):
        kb_n = kb_n[0]

    vo_n = vo_transformer(
        hidden_states=latents_vo_full, timestep=ts / 1000, guidance=vo_guidance_e2e,
        pooled_projections=vo_full_pooled,
        encoder_hidden_states=vo_full_prompt_embeds,
        txt_ids=vo_full_text_ids, img_ids=latent_image_ids_kb,
        joint_attention_kwargs={}, return_dict=False,
    )
    if isinstance(vo_n, tuple):
        vo_n = vo_n[0]

    latents_kb_full = sched_kb2.step(kb_n, t, latents_kb_full, return_dict=False)[0]
    latents_vo_full = sched_vo2.step(vo_n, t, latents_vo_full, return_dict=False)[0]

    if step_idx < 3 or step_idx >= NUM_STEPS - 2:
        report(f"e2e step {step_idx:2d} noise_pred", kb_n, vo_n, indent=1)
        report(f"e2e step {step_idx:2d} latents   ", latents_kb_full, latents_vo_full, indent=1)
    elif step_idx == 3:
        print("    ... (steps 3-25 omitted) ...")

print("\n" + "=" * 80)
print("STAGE 8 result: End-to-end denoised latents")
print("=" * 80)
report("End-to-end final denoised latents", latents_kb_full, latents_vo_full)

# Also decode through VAE to compare final images
latents_kb_unpacked = kb_pipe._unpack_latents(latents_kb_full, HEIGHT, WIDTH, kb_pipe.vae_scale_factor)
latents_vo_unpacked = kb_pipe._unpack_latents(latents_vo_full, HEIGHT, WIDTH, kb_pipe.vae_scale_factor)
latents_kb_dec = (latents_kb_unpacked / kb_pipe.vae.config.scaling_factor) + kb_pipe.vae.config.shift_factor
latents_vo_dec = (latents_vo_unpacked / kb_pipe.vae.config.scaling_factor) + kb_pipe.vae.config.shift_factor
img_kb = kb_pipe.vae.decode(latents_kb_dec, return_dict=False)[0]
img_vo = kb_pipe.vae.decode(latents_vo_dec, return_dict=False)[0]
report("End-to-end final decoded images", img_kb, img_vo)
