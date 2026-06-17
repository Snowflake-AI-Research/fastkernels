#!/usr/bin/env python3
"""Definitive diagnosis of kb-nano vs vllm-omni misalignment.

Tests each hypothesis with a single prompt/batch, identical seed,
comparing at every stage:
  H1: Initial noise identity
  H2: VAE decode path (bf16 VAE vs fp32 VAE)
  H3: Text encoding differences
  H4: torch.compile impact
  H5: Scheduler/timestep differences
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
torch.set_grad_enabled(False)

LOG_PATH = "/home/yak/.cursor/debug-ba5280.log"

def log(hypothesis_id, location, message, data):
    import json, time
    entry = {
        "sessionId": "ba5280",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": {k: (v.tolist() if hasattr(v, 'tolist') else str(v) if isinstance(v, (torch.dtype, torch.Size)) else v) for k, v in data.items()},
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def cos_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.detach().float().flatten().unsqueeze(0),
        b.detach().float().flatten().unsqueeze(0)))

def maxd(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL = "black-forest-labs/FLUX.1-dev"
SEED = 42
PROMPT = ["A cat on a mat", "A sunset over the ocean", "A mountain landscape", "A bird in flight"]
HEIGHT, WIDTH = 1024, 1024

# ============= LOAD KB-NANO =============
print("Loading kb-nano...")
from kb_nano.infra.diffusion_engine import _download_flux_model, _load_flux_weights
from kb_nano.tasks.baseline.L4.flux import FluxConfig, FluxPipeline, DiffusionSamplingParams

model_path = _download_flux_model(MODEL)
config = FluxConfig.from_pretrained(model_path)
kb_pipe = FluxPipeline(config, model_path)
_load_flux_weights(kb_pipe, model_path)
kb_pipe.to(device=DEVICE, dtype=DTYPE)
kb_pipe.text_encoder.to(device=DEVICE)
kb_pipe.text_encoder_2.to(device=DEVICE)
kb_pipe.vae.to(device=DEVICE)
kb_pipe.eval()

log("setup", "diag.py:58", "kb-nano loaded", {
    "vae_dtype": kb_pipe.vae.dtype,
    "transformer_dtype": next(kb_pipe.transformer.parameters()).dtype,
    "clip_dtype": next(kb_pipe.text_encoder.parameters()).dtype,
    "t5_dtype": next(kb_pipe.text_encoder_2.parameters()).dtype,
})

# ============= LOAD VLLM-OMNI =============
print("Loading vllm-omni...")
from vllm.config import VllmConfig, set_current_vllm_config, ModelConfig
_mc = ModelConfig(model="Qwen/Qwen3-0.6B")
vllm_config = VllmConfig(model_config=_mc)
vllm_config.compilation_config.level = 0
_ctx = set_current_vllm_config(vllm_config)
_ctx.__enter__()
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

from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel as VOFluxTransformer2D
from transformers import CLIPTextModel as HFCLIPTextModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

tf_config_path = os.path.join(model_path, "transformer", "config.json")
with open(tf_config_path) as f:
    tf_data = json.load(f)
vo_transformer = VOFluxTransformer2D(
    od_config=None, in_channels=tf_data.get("in_channels", 64),
    out_channels=tf_data.get("out_channels"), num_layers=tf_data.get("num_layers", 19),
    num_single_layers=tf_data.get("num_single_layers", 38),
    attention_head_dim=tf_data.get("attention_head_dim", 128),
    num_attention_heads=tf_data.get("num_attention_heads", 24),
    joint_attention_dim=tf_data.get("joint_attention_dim", 4096),
    pooled_projection_dim=tf_data.get("pooled_projection_dim", 768),
    guidance_embeds=tf_data.get("guidance_embeds", True),
    axes_dims_rope=tuple(tf_data.get("axes_dims_rope", [16, 56, 56])),
)
from glob import glob
from safetensors import safe_open
def load_sf(d):
    w = []
    for f in sorted(glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt", "cpu") as sf:
            for k in sf.keys(): w.append((k, sf.get_tensor(k)))
    return w
vo_transformer.load_weights(load_sf(os.path.join(model_path, "transformer")))
vo_transformer.to(device=DEVICE, dtype=DTYPE)
vo_transformer.eval()

# Load TWO VAEs: fp32 and bf16 (to match what vllm-omni actually uses)
vae_fp32 = AutoencoderKL.from_pretrained(model_path, subfolder="vae", local_files_only=True).to(DEVICE)
vae_bf16 = AutoencoderKL.from_pretrained(model_path, subfolder="vae", local_files_only=True).to(device=DEVICE, dtype=DTYPE)

log("setup", "diag.py:115", "vllm-omni loaded", {
    "vo_transformer_dtype": next(vo_transformer.parameters()).dtype,
    "vae_fp32_dtype": vae_fp32.dtype,
    "vae_bf16_dtype": vae_bf16.dtype,
})

print("Both loaded.\n")

# ============= H1: Initial noise =============
print("=" * 80)
print("H1: INITIAL NOISE COMPARISON")
print("=" * 80)

with torch.inference_mode():
    gen_kb = torch.Generator(device=DEVICE).manual_seed(SEED)
    kb_latents, kb_lat_ids = kb_pipe.prepare_latents(
        4, kb_pipe.transformer.in_channels // 4, HEIGHT, WIDTH, DTYPE, DEVICE, gen_kb)

    gen_vo = torch.Generator(device=DEVICE).manual_seed(SEED)
    from diffusers.utils.torch_utils import randn_tensor
    num_ch = kb_pipe.transformer.in_channels // 4
    h = 2 * (HEIGHT // (kb_pipe.vae_scale_factor * 2))
    w = 2 * (WIDTH // (kb_pipe.vae_scale_factor * 2))
    vo_noise = randn_tensor((4, num_ch, h, w), generator=gen_vo, device=DEVICE, dtype=DTYPE)
    vo_latents = kb_pipe._pack_latents(vo_noise, 4, num_ch, h, w)
    
    noise_cos = cos_sim(kb_latents, vo_latents)
    noise_exact = torch.equal(kb_latents, vo_latents)
    log("H1", "diag.py:140", "noise comparison", {
        "cos": noise_cos, "exact_match": noise_exact,
        "kb_shape": kb_latents.shape, "vo_shape": vo_latents.shape,
    })
    print(f"  Noise cos={noise_cos:.10f} exact={noise_exact}")

# ============= H3: Text encoding =============
print("\n" + "=" * 80)
print("H3: TEXT ENCODING COMPARISON")
print("=" * 80)

with torch.inference_mode():
    kb_prompt_embeds, kb_pooled, kb_text_ids = kb_pipe.encode_prompt(
        PROMPT, num_images_per_prompt=1, max_sequence_length=512)

    hf_clip = HFCLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", local_files_only=True)
    hf_clip.to(device=DEVICE).eval()
    clip_inputs = kb_pipe.tokenizer(PROMPT, padding="max_length", max_length=kb_pipe.tokenizer_max_length,
                                     truncation=True, return_tensors="pt")
    clip_ids = clip_inputs.input_ids.to(DEVICE)
    hf_clip_out = hf_clip(clip_ids, output_hidden_states=False)
    hf_pooled = hf_clip_out.pooler_output.to(dtype=DTYPE, device=DEVICE)

    clip_cos = cos_sim(kb_pooled, hf_pooled)
    log("H3", "diag.py:165", "CLIP pooled comparison", {
        "cos": clip_cos, "max_diff": maxd(kb_pooled, hf_pooled),
        "kb_dtype": kb_pooled.dtype, "hf_dtype": hf_pooled.dtype,
    })
    print(f"  CLIP pooled cos={clip_cos:.10f}")
    
    # T5 — confirmed exact in diagnostic, skip
    print(f"  T5 — confirmed cos=1.0 from diagnostic")

# ============= TRANSFORMER FORWARD =============
print("\n" + "=" * 80)
print("TRANSFORMER FORWARD (1 step, batch=4, 1024x1024)")
print("=" * 80)

with torch.inference_mode():
    text_ids = kb_text_ids
    if text_ids.ndim == 3: text_ids = text_ids[0]
    lat_ids = kb_lat_ids
    if lat_ids.ndim == 3: lat_ids = lat_ids[0]
    
    timesteps, _ = kb_pipe.prepare_timesteps(28, None, kb_latents.shape[1])
    guidance = torch.full([1], 3.5, dtype=DTYPE, device=DEVICE).expand(4)
    t = timesteps[0].expand(4).to(DEVICE, DTYPE) / 1000

    kb_noise = kb_pipe.transformer(
        hidden_states=kb_latents, timestep=t, guidance=guidance,
        pooled_projections=kb_pooled, encoder_hidden_states=kb_prompt_embeds,
        txt_ids=text_ids, img_ids=lat_ids,
        joint_attention_kwargs={}, return_dict=False,
    )
    if isinstance(kb_noise, tuple): kb_noise = kb_noise[0]
    
    vo_noise = vo_transformer(
        hidden_states=kb_latents, timestep=t, guidance=guidance,
        pooled_projections=kb_pooled, encoder_hidden_states=kb_prompt_embeds,
        txt_ids=text_ids, img_ids=lat_ids,
        joint_attention_kwargs={}, return_dict=False,
    )
    if isinstance(vo_noise, tuple): vo_noise = vo_noise[0]
    
    tf_cos = cos_sim(kb_noise, vo_noise)
    log("transformer", "diag.py:200", "transformer forward match", {
        "cos": tf_cos, "max_diff": maxd(kb_noise, vo_noise),
    })
    print(f"  Transformer forward cos={tf_cos:.10f}")

# ============= FULL DENOISING LOOP (same inputs) =============
print("\n" + "=" * 80)
print("FULL DENOISING LOOP (28 steps, same inputs)")
print("=" * 80)

with torch.inference_mode():
    from copy import deepcopy
    import numpy as np
    
    sched_kb = deepcopy(kb_pipe.scheduler)
    sched_vo = deepcopy(kb_pipe.scheduler)
    sigmas = np.linspace(1.0, 1.0 / 28, 28)
    from kb_nano.tasks.baseline.L4.flux import _calculate_shift
    mu = _calculate_shift(kb_latents.shape[1],
        sched_kb.config.get("base_image_seq_len", 256),
        sched_kb.config.get("max_image_seq_len", 4096),
        sched_kb.config.get("base_shift", 0.5),
        sched_kb.config.get("max_shift", 1.15))
    sched_kb.set_timesteps(sigmas=sigmas, mu=mu)
    sched_vo.set_timesteps(sigmas=sigmas, mu=mu)
    sched_kb.set_begin_index(0)
    sched_vo.set_begin_index(0)
    
    lat_kb = kb_latents.clone()
    lat_vo = kb_latents.clone()
    
    for step_idx, t_step in enumerate(sched_kb.timesteps):
        ts = t_step.expand(4).to(DEVICE, DTYPE)
        
        kb_n = kb_pipe.transformer(
            hidden_states=lat_kb, timestep=ts / 1000, guidance=guidance,
            pooled_projections=kb_pooled, encoder_hidden_states=kb_prompt_embeds,
            txt_ids=text_ids, img_ids=lat_ids,
            joint_attention_kwargs={}, return_dict=False,
        )
        if isinstance(kb_n, tuple): kb_n = kb_n[0]
        
        vo_n = vo_transformer(
            hidden_states=lat_vo, timestep=ts / 1000, guidance=guidance,
            pooled_projections=kb_pooled, encoder_hidden_states=kb_prompt_embeds,
            txt_ids=text_ids, img_ids=lat_ids,
            joint_attention_kwargs={}, return_dict=False,
        )
        if isinstance(vo_n, tuple): vo_n = vo_n[0]
        
        lat_kb = sched_kb.step(kb_n, t_step, lat_kb, return_dict=False)[0]
        lat_vo = sched_vo.step(vo_n, t_step, lat_vo, return_dict=False)[0]
    
    latent_cos = cos_sim(lat_kb, lat_vo)
    log("denoising", "diag.py:255", "final denoised latents (same inputs)", {
        "cos": latent_cos, "max_diff": maxd(lat_kb, lat_vo),
    })
    print(f"  Final denoised latents (same inputs) cos={latent_cos:.10f}")

# ============= H2: VAE DECODE COMPARISON =============
print("\n" + "=" * 80)
print("H2: VAE DECODE COMPARISON")
print("=" * 80)

with torch.inference_mode():
    unpacked = kb_pipe._unpack_latents(lat_kb, HEIGHT, WIDTH, kb_pipe.vae_scale_factor)
    pre_vae = (unpacked / kb_pipe.vae.config.scaling_factor) + kb_pipe.vae.config.shift_factor
    
    # Path A: fp32 VAE (what kb-nano benchmark does)
    img_fp32 = vae_fp32.decode(pre_vae.to(vae_fp32.dtype), return_dict=False)[0]
    
    # Path B: bf16 VAE (what vllm-omni does via set_default_torch_dtype(bf16))
    img_bf16 = vae_bf16.decode(pre_vae.to(vae_bf16.dtype), return_dict=False)[0]
    
    vae_cos = cos_sim(img_fp32, img_bf16)
    vae_maxd = maxd(img_fp32, img_bf16)
    log("H2", "diag.py:275", "VAE fp32 vs bf16 decode", {
        "cos": vae_cos, "max_diff": vae_maxd,
        "fp32_shape": img_fp32.shape, "fp32_dtype": img_fp32.dtype,
        "bf16_shape": img_bf16.shape, "bf16_dtype": img_bf16.dtype,
        "fp32_range": [float(img_fp32.min()), float(img_fp32.max())],
        "bf16_range": [float(img_bf16.float().min()), float(img_bf16.float().max())],
    })
    print(f"  VAE fp32 vs bf16 decode: cos={vae_cos:.10f} max_diff={vae_maxd:.6f}")

    # Per-sample
    for i in range(4):
        per_cos = cos_sim(img_fp32[i], img_bf16[i])
        print(f"    sample[{i}]: cos={per_cos:.10f}")
        log("H2", f"diag.py:285_{i}", f"VAE per-sample[{i}]", {"cos": per_cos})

# ============= FULL E2E: each pipeline's own CLIP =============
print("\n" + "=" * 80)
print("FULL E2E: kb-nano CLIP + T5 vs HF CLIP + same T5, same noise")
print("=" * 80)

with torch.inference_mode():
    # vllm-omni uses HF CLIP pooled (already computed above)
    # Denoise with kb-nano's CLIP pooled
    sched_a = deepcopy(kb_pipe.scheduler)
    sched_a.set_timesteps(sigmas=sigmas, mu=mu)
    sched_a.set_begin_index(0)
    lat_a = kb_latents.clone()
    
    # Denoise with HF CLIP pooled (what vllm-omni uses)
    sched_b = deepcopy(kb_pipe.scheduler)
    sched_b.set_timesteps(sigmas=sigmas, mu=mu)
    sched_b.set_begin_index(0)
    lat_b = kb_latents.clone()
    
    for step_idx, t_step in enumerate(sched_a.timesteps):
        ts = t_step.expand(4).to(DEVICE, DTYPE)
        
        n_a = kb_pipe.transformer(
            hidden_states=lat_a, timestep=ts / 1000, guidance=guidance,
            pooled_projections=kb_pooled, encoder_hidden_states=kb_prompt_embeds,
            txt_ids=text_ids, img_ids=lat_ids,
            joint_attention_kwargs={}, return_dict=False,
        )
        if isinstance(n_a, tuple): n_a = n_a[0]
        
        n_b = kb_pipe.transformer(
            hidden_states=lat_b, timestep=ts / 1000, guidance=guidance,
            pooled_projections=hf_pooled, encoder_hidden_states=kb_prompt_embeds,
            txt_ids=text_ids, img_ids=lat_ids,
            joint_attention_kwargs={}, return_dict=False,
        )
        if isinstance(n_b, tuple): n_b = n_b[0]
        
        lat_a = sched_a.step(n_a, t_step, lat_a, return_dict=False)[0]
        lat_b = sched_b.step(n_b, t_step, lat_b, return_dict=False)[0]
    
    clip_latent_cos = cos_sim(lat_a, lat_b)
    log("H3_e2e", "diag.py:330", "CLIP pooled impact on final latents", {
        "cos": clip_latent_cos, "max_diff": maxd(lat_a, lat_b),
    })
    print(f"  kb-CLIP vs hf-CLIP final latents: cos={clip_latent_cos:.10f}")
    
    # Now decode both through fp32 VAE
    up_a = kb_pipe._unpack_latents(lat_a, HEIGHT, WIDTH, kb_pipe.vae_scale_factor)
    pv_a = (up_a / vae_fp32.config.scaling_factor) + vae_fp32.config.shift_factor
    img_a_fp32 = vae_fp32.decode(pv_a.float(), return_dict=False)[0]
    
    up_b = kb_pipe._unpack_latents(lat_b, HEIGHT, WIDTH, kb_pipe.vae_scale_factor)
    pv_b = (up_b / vae_fp32.config.scaling_factor) + vae_fp32.config.shift_factor
    img_b_fp32 = vae_fp32.decode(pv_b.float(), return_dict=False)[0]
    
    clip_img_cos = cos_sim(img_a_fp32, img_b_fp32)
    log("H3_e2e", "diag.py:345", "CLIP pooled impact on decoded images (same VAE)", {
        "cos": clip_img_cos,
    })
    print(f"  kb-CLIP vs hf-CLIP decoded images (same fp32 VAE): cos={clip_img_cos:.10f}")
    
    # Now the full benchmark path:
    # kb-nano: kb_pooled, fp32 VAE decode
    # vllm-omni: hf_pooled, bf16 VAE decode
    img_b_bf16 = vae_bf16.decode(pv_b.to(DTYPE), return_dict=False)[0]
    
    benchmark_cos = cos_sim(img_a_fp32, img_b_bf16)
    log("FINAL", "diag.py:355", "EXACT BENCHMARK PATH: kb(kb_clip, fp32_vae) vs vo(hf_clip, bf16_vae)", {
        "cos": benchmark_cos, "max_diff": maxd(img_a_fp32, img_b_bf16),
    })
    print(f"\n  *** EXACT BENCHMARK PATH ***")
    print(f"  kb-nano (kb_clip + fp32 VAE) vs vllm-omni (hf_clip + bf16 VAE): cos={benchmark_cos:.10f}")
    
    for i in range(4):
        pc = cos_sim(img_a_fp32[i], img_b_bf16[i])
        log("FINAL", f"diag.py:362_{i}", f"benchmark per-sample[{i}]", {"cos": pc})
        print(f"    sample[{i}]: cos={pc:.10f}")

    # Attribution: how much comes from CLIP vs VAE?
    # Same CLIP, different VAE
    img_a_bf16 = vae_bf16.decode(pv_a.to(DTYPE), return_dict=False)[0]
    vae_only_cos = cos_sim(img_a_fp32, img_a_bf16)
    
    # Different CLIP, same VAE (fp32)
    clip_only_cos = cos_sim(img_a_fp32, img_b_fp32)
    
    log("ATTRIBUTION", "diag.py:375", "attribution", {
        "vae_only_cos": vae_only_cos,
        "clip_only_cos": clip_only_cos,
        "combined_cos": benchmark_cos,
    })
    print(f"\n  === ATTRIBUTION ===")
    print(f"  VAE dtype only (same CLIP): cos={vae_only_cos:.10f}")
    print(f"  CLIP only (same VAE):       cos={clip_only_cos:.10f}")
    print(f"  Combined (different CLIP + different VAE): cos={benchmark_cos:.10f}")

print("\nDone.")
