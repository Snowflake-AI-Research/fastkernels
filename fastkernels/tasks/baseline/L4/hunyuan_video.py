"""HunyuanVideo-1.5 diffusion pipeline (L4 pipeline).

Contains:
- HunyuanVideoConfig: configuration dataclass
- HunyuanVideo15Transformer3DModel: the DiT backbone (54 dual-stream blocks)
- HunyuanVideoPipeline: full text-to-video pipeline (encode, diffuse, decode)

L4 wiring/configuration; computation lives in L1-L3 tasks.

Mirrors vllm-omni's implementation in
``vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py``
and ``pipeline_hunyuan_video_1_5.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoConfig, ByT5Tokenizer, Qwen2Tokenizer

from .qwen25_vl_encoder import Qwen25VLTextEncoder
from ..L1.video_processor import VideoProcessor

from ..L1.linear import Linear
from ..L1.hunyuan_video_rope import HunyuanVideo15RotaryPosEmbed
from ..L2.ada_layer_norm_continuous import AdaLayerNormContinuous
from ..L2.hunyuan_video_embeddings import (
    HunyuanVideo15ByT5TextProjection,
    HunyuanVideo15ImageProjection,
    HunyuanVideo15PatchEmbed,
    HunyuanVideo15TimeEmbedding,
)
from ..L2.hunyuan_video_conditioning import HunyuanVideo15ConditioningMerge
from ..L3.hunyuan_video_token_refiner_block import HunyuanVideo15TokenRefiner
from ..L3.hunyuan_video_transformer_block import HunyuanVideo15TransformerBlock
from .t5_encoder import T5EncoderModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class HunyuanVideoConfig:
    """Configuration for the HunyuanVideo-1.5 transformer model."""
    in_channels: int = 65
    out_channels: int = 32
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 54
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 1
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    text_embed_dim: int = 3584
    text_embed_2_dim: int = 1472
    image_embed_dim: int = 1152
    rope_theta: float = 256.0
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    target_size: int = 640
    task_type: str = "i2v"
    use_meanflow: bool = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> "HunyuanVideoConfig":
        """Load config from transformer/config.json in a HunyuanVideo model repo."""
        from huggingface_hub import hf_hub_download
        if os.path.isdir(model_name):
            config_path = os.path.join(model_name, "transformer", "config.json")
        else:
            config_path = hf_hub_download(model_name, "transformer/config.json")
        with open(config_path) as f:
            data = json.load(f)
        return cls(
            in_channels=data.get("in_channels", 65),
            out_channels=data.get("out_channels", 32),
            num_attention_heads=data.get("num_attention_heads", 16),
            attention_head_dim=data.get("attention_head_dim", 128),
            num_layers=data.get("num_layers", 54),
            num_refiner_layers=data.get("num_refiner_layers", 2),
            mlp_ratio=data.get("mlp_ratio", 4.0),
            patch_size=data.get("patch_size", 1),
            patch_size_t=data.get("patch_size_t", 1),
            qk_norm=data.get("qk_norm", "rms_norm"),
            text_embed_dim=data.get("text_embed_dim", 3584),
            text_embed_2_dim=data.get("text_embed_2_dim", 1472),
            image_embed_dim=data.get("image_embed_dim", 1152),
            rope_theta=data.get("rope_theta", 256.0),
            rope_axes_dim=tuple(data.get("rope_axes_dim", [16, 56, 56])),
            target_size=data.get("target_size", 640),
            task_type=data.get("task_type", "i2v"),
            use_meanflow=data.get("use_meanflow", False),
        )


@dataclass
class HunyuanVideoDiffusionSamplingParams:
    """Sampling parameters for HunyuanVideo generation."""
    height: int = 480
    width: int = 832
    num_frames: int = 121
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: int | None = None
    output_type: str = "pil"


@dataclass
class HunyuanVideoDiffusionOutput:
    """Output of the HunyuanVideo pipeline."""
    video: list[Any] | None = None
    latents: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)


def _format_text_input(prompt: list[str], system_message: str) -> list[list[dict[str, str]]]:
    return [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]


def _extract_glyph_texts(prompt: str) -> str | None:
    pattern = r'"(.*?)"|' + r"\u201c(.*?)\u201d"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    if result:
        return ". ".join([f'Text "{text}"' for text in result]) + ". "
    return None


# ---------------------------------------------------------------------------
# Transformer backbone
# ---------------------------------------------------------------------------

class HunyuanVideo15Transformer3DModel(nn.Module):
    """HunyuanVideo-1.5 DiT backbone: 54 dual-stream transformer blocks.

    Takes 3D video latent patches + multi-modal text/image embeddings +
    timestep conditioning and produces noise predictions.
    """

    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(self, config: HunyuanVideoConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels or config.in_channels
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = inner_dim
        self.patch_size = config.patch_size
        self.patch_size_t = config.patch_size_t

        self.x_embedder = HunyuanVideo15PatchEmbed(
            (config.patch_size_t, config.patch_size, config.patch_size),
            config.in_channels, inner_dim,
        )
        self.image_embedder = HunyuanVideo15ImageProjection(config.image_embed_dim, inner_dim)

        self.context_embedder = HunyuanVideo15TokenRefiner(
            config.text_embed_dim, config.num_attention_heads, config.attention_head_dim,
            num_layers=config.num_refiner_layers,
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(
            config.text_embed_2_dim, 2048, inner_dim,
        )

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=config.use_meanflow)
        self.conditioning_merge = HunyuanVideo15ConditioningMerge(inner_dim)

        self.rope = HunyuanVideo15RotaryPosEmbed(
            config.patch_size, config.patch_size_t, list(config.rope_axes_dim), config.rope_theta,
        )

        self.transformer_blocks = nn.ModuleList([
            HunyuanVideo15TransformerBlock(
                config.num_attention_heads, config.attention_head_dim,
                mlp_ratio=config.mlp_ratio, qk_norm=config.qk_norm,
            )
            for _ in range(config.num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = Linear(inner_dim, config.patch_size_t * config.patch_size * config.patch_size * self.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: torch.LongTensor | None = None,
        encoder_hidden_states_2: torch.Tensor | None = None,
        encoder_attention_mask_2: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        image_embeds_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size_t, self.patch_size, self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        image_rotary_emb = self.rope(hidden_states)

        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        image_hidden_states = self.image_embedder(image_embeds)
        if image_embeds_mask is not None:
            image_attention_mask = image_embeds_mask
        else:
            is_t2v = torch.all(image_embeds == 0)
            if is_t2v:
                image_hidden_states = image_hidden_states * 0.0
                image_attention_mask = torch.zeros(
                    (batch_size, image_hidden_states.shape[1]),
                    dtype=encoder_attention_mask.dtype,
                    device=encoder_attention_mask.device,
                )
            else:
                image_attention_mask = torch.ones(
                    (batch_size, image_hidden_states.shape[1]),
                    dtype=encoder_attention_mask.dtype,
                    device=encoder_attention_mask.device,
                )

        encoder_hidden_states, encoder_attention_mask = self.conditioning_merge(
            encoder_hidden_states, encoder_attention_mask,
            encoder_hidden_states_2, encoder_attention_mask_2,
            image_hidden_states, image_attention_mask,
        )

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            if lookup_name == "cond_type_embed.weight":
                lookup_name = "conditioning_merge.cond_type_embed.emb.weight"

            if lookup_name not in params_dict and ".proj." in lookup_name:
                conv_name = lookup_name.replace(".proj.", ".proj.conv.")
                if conv_name in params_dict:
                    lookup_name = conv_name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    break
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class HunyuanVideoPipeline(nn.Module):
    """Full HunyuanVideo-1.5 text-to-video pipeline.

    Composes: Qwen2.5-VL + ByT5 text encoding -> latent preparation ->
    denoising loop (flow-match Euler) -> VAE decode.
    """

    def __init__(self, config: HunyuanVideoConfig, model_name: str):
        super().__init__()
        self.config = config
        self.model_name = model_name

        local_files_only = os.path.isdir(model_name)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler", local_files_only=local_files_only,
        )

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model_name, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.text_encoder = Qwen25VLTextEncoder.from_pretrained(
            model_name, subfolder="text_encoder", local_files_only=local_files_only,
        )

        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            model_name, subfolder="tokenizer_2", local_files_only=local_files_only,
        )
        t5_config = AutoConfig.from_pretrained(
            model_name, subfolder="text_encoder_2", local_files_only=local_files_only,
        )
        self.text_encoder_2 = T5EncoderModel(t5_config)

        self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )

        self.transformer = HunyuanVideo15Transformer3DModel(config)

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 16
        )
        self.num_channels_latents = self.vae.config.latent_channels if hasattr(self.vae, "config") else 32

        self.system_message = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729
        self.vision_states_dim = 1152

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _get_mllm_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        num_hidden_layers_to_skip: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_formatted = _format_text_input(prompt, self.system_message)

        text_inputs = self.tokenizer.apply_chat_template(
            prompt_formatted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        crop_start = self.prompt_template_encode_start_idx
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        return prompt_embeds.to(dtype=dtype), prompt_attention_mask

    def _get_byte5_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds_list = []
        prompt_embeds_mask_list = []

        for p in prompt:
            glyph_text = _extract_glyph_texts(p)

            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, self.tokenizer_2_max_length, self.text_encoder_2.config.d_model),
                    device=device, dtype=dtype,
                )
                glyph_text_embeds_mask = torch.zeros(
                    (1, self.tokenizer_2_max_length), device=device, dtype=torch.int64,
                )
            else:
                txt_tokens = self.tokenizer_2(
                    glyph_text,
                    padding="max_length",
                    max_length=self.tokenizer_2_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = self.text_encoder_2(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                glyph_text_embeds = glyph_text_embeds.to(device=device, dtype=dtype)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        return torch.cat(prompt_embeds_list, dim=0), torch.cat(prompt_embeds_mask_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
    ]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt, device, dtype)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_byte5_prompt_embeds(prompt, device, dtype)

        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype)

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        negative_prompt_embeds_2 = None
        negative_prompt_embeds_mask_2 = None

        if do_classifier_free_guidance:
            if negative_prompt is not None:
                neg = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            else:
                neg = [""] * len(prompt)
            negative_prompt_embeds, negative_prompt_embeds_mask = self._get_mllm_prompt_embeds(neg, device, dtype)
            negative_prompt_embeds_2, negative_prompt_embeds_mask_2 = self._get_byte5_prompt_embeds(neg, device, dtype)
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(dtype=dtype)
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(dtype=dtype)

        return (
            prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
            negative_prompt_embeds, negative_prompt_embeds_mask,
            negative_prompt_embeds_2, negative_prompt_embeds_mask_2,
        )

    # -----------------------------------------------------------------------
    # Latent preparation
    # -----------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_cond_latents_and_mask(
        self, latents: torch.Tensor, dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, frames, height, width = latents.shape
        cond_latents = torch.zeros(batch, channels, frames, height, width, dtype=dtype, device=device)
        mask = torch.zeros(batch, 1, frames, height, width, dtype=dtype, device=device)
        return cond_latents, mask

    # -----------------------------------------------------------------------
    # Denoising
    # -----------------------------------------------------------------------

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_embeds_mask_2: torch.Tensor,
        latents: torch.Tensor,
        cond_latents: torch.Tensor,
        mask: torch.Tensor,
        image_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds_2: torch.Tensor | None = None,
        negative_prompt_embeds_mask_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        for t in timesteps:
            latent_model_input = torch.cat([latents, cond_latents, mask], dim=1)
            timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_embeds_mask,
                encoder_hidden_states_2=prompt_embeds_2,
                encoder_attention_mask_2=prompt_embeds_mask_2,
                image_embeds=image_embeds,
                return_dict=False,
            )
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]

            if do_cfg:
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_attention_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states_2=negative_prompt_embeds_2,
                    encoder_attention_mask_2=negative_prompt_embeds_mask_2,
                    image_embeds=image_embeds,
                    return_dict=False,
                )
                if isinstance(noise_pred_uncond, tuple):
                    noise_pred_uncond = noise_pred_uncond[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # -----------------------------------------------------------------------
    # Forward (full pipeline)
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        prompts: str | list[str],
        params: HunyuanVideoDiffusionSamplingParams | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> HunyuanVideoDiffusionOutput:
        params = params or HunyuanVideoDiffusionSamplingParams()
        if isinstance(prompts, str):
            prompts = [prompts]

        height = params.height
        width = params.width
        num_frames = params.num_frames
        batch_size = len(prompts)
        device = self.vae.device
        dtype = self.transformer.transformer_blocks[0].norm1.linear.weight.dtype

        guidance_scale = params.guidance_scale
        do_cfg = guidance_scale > 1.0

        (
            prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
            negative_prompt_embeds, negative_prompt_embeds_mask,
            negative_prompt_embeds_2, negative_prompt_embeds_mask_2,
        ) = self.encode_prompt(
            prompt=prompts, device=device, dtype=dtype,
            do_classifier_free_guidance=do_cfg,
        )

        latents = self.prepare_latents(
            batch_size, height, width, num_frames, dtype, device, generator,
        )
        cond_latents, mask = self.prepare_cond_latents_and_mask(latents, dtype, device)

        image_embeds = torch.zeros(
            batch_size, self.vision_num_semantic_tokens, self.vision_states_dim,
            dtype=dtype, device=device,
        )

        sigmas = np.linspace(1.0, 0.0, params.num_inference_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.diffuse(
            prompt_embeds, prompt_embeds_mask,
            prompt_embeds_2, prompt_embeds_mask_2,
            latents, cond_latents, mask, image_embeds,
            timesteps,
            guidance_scale=guidance_scale,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
        )

        if params.output_type == "latent":
            return HunyuanVideoDiffusionOutput(latents=latents)

        latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
        video = self.vae.decode(latents, return_dict=False)[0]

        video_processor = VideoProcessor(vae_scale_factor=16)
        pil_video = video_processor.postprocess_video(video, output_type=params.output_type)
        if isinstance(pil_video, list) and pil_video and isinstance(pil_video[0], list):
            pil_video = pil_video[0]

        return HunyuanVideoDiffusionOutput(video=pil_video, latents=latents)
