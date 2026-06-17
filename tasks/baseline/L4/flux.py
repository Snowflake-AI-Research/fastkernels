"""FLUX.1-dev diffusion pipeline (L4 pipeline).

Contains:
- FluxTransformer2DModel: the DiT backbone (dual + single stream blocks).
- FluxPipeline: full text-to-image pipeline (encode, diffuse, decode).

L4 wiring/configuration; computation lives in L1-L3 tasks.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from ..L2.ada_layer_norm_continuous import AdaLayerNormContinuous
from ..L2.timestep_embedding import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from ..L1.video_processor import VideoProcessor as VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from torch import nn
from transformers import AutoConfig, CLIPTokenizer, T5TokenizerFast

from .clip_text_model import CLIPTextModel
from .t5_encoder import T5EncoderModel

from ..L1.flux_pos_embed import FluxPosEmbed
from ..L1.linear import Linear
from ..L3.flux_transformer_block import FluxTransformerBlock, FluxSingleTransformerBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class FluxConfig:
    """Configuration for the FLUX transformer model."""
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    in_channels: int = 64
    out_channels: int | None = None
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = True
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    patch_size: int = 1

    @classmethod
    def from_pretrained(cls, model_name: str) -> "FluxConfig":
        """Load config from transformer/config.json in a FLUX model repo."""
        from huggingface_hub import hf_hub_download
        if os.path.isdir(model_name):
            config_path = os.path.join(model_name, "transformer", "config.json")
        else:
            config_path = hf_hub_download(model_name, "transformer/config.json")
        with open(config_path) as f:
            data = json.load(f)
        return cls(
            num_layers=data.get("num_layers", 19),
            num_single_layers=data.get("num_single_layers", 38),
            attention_head_dim=data.get("attention_head_dim", 128),
            num_attention_heads=data.get("num_attention_heads", 24),
            in_channels=data.get("in_channels", 64),
            out_channels=data.get("out_channels", None),
            joint_attention_dim=data.get("joint_attention_dim", 4096),
            pooled_projection_dim=data.get("pooled_projection_dim", 768),
            guidance_embeds=data.get("guidance_embeds", True),
            axes_dims_rope=tuple(data.get("axes_dims_rope", [16, 56, 56])),
            patch_size=data.get("patch_size", 1),
        )


@dataclass
class DiffusionSamplingParams:
    """Sampling parameters for diffusion generation."""
    height: int | None = None
    width: int | None = None
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    num_outputs_per_prompt: int = 1
    seed: int | None = None
    sigmas: list[float] | None = None
    output_type: str = "pil"
    max_sequence_length: int = 512


@dataclass
class DiffusionOutput:
    """Output of the diffusion pipeline."""
    images: list[Any] | None = None
    latents: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Transformer backbone
# ---------------------------------------------------------------------------

class FluxTransformer2DModel(nn.Module):
    """FLUX DiT backbone: dual-stream + single-stream transformer blocks.

    Takes packed latent patches + text embeddings + timestep conditioning
    and produces noise predictions.
    """

    def __init__(self, config: FluxConfig, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels or config.in_channels
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = inner_dim
        self.guidance_embeds = config.guidance_embeds

        self.pos_embed = FluxPosEmbed(
            theta=10000, axes_dim=config.axes_dims_rope,
        )

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if config.guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=inner_dim,
            pooled_projection_dim=config.pooled_projection_dim,
        )

        self.context_embedder = Linear(config.joint_attention_dim, inner_dim)
        self.x_embedder = Linear(config.in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                quant_config=quant_config,
            )
            for _ in range(config.num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                quant_config=quant_config,
            )
            for _ in range(config.num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6,
        )
        self.proj_out = Linear(
            inner_dim,
            config.patch_size * config.patch_size * self.out_channels,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) * 1000

        if guidance is not None:
            guidance = guidance.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        def _default_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)

        params_dict = dict(self.named_parameters())
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class FluxPipeline(nn.Module):
    """Full FLUX text-to-image pipeline.

    Composes: CLIP + T5 text encoding -> latent preparation -> denoising
    loop (flow-match Euler) -> VAE decode.
    """

    def __init__(self, config: FluxConfig, model_name: str,
                 quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.model_name = model_name

        local_files_only = os.path.isdir(model_name)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler", local_files_only=local_files_only,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", local_files_only=local_files_only,
        )
        t5_config = AutoConfig.from_pretrained(
            model_name, subfolder="text_encoder_2", local_files_only=local_files_only,
        )
        self.text_encoder_2 = T5EncoderModel(t5_config)
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", local_files_only=local_files_only,
        )
        self.transformer = FluxTransformer2DModel(config, quant_config=quant_config)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer_2", local_files_only=local_files_only,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels")
            else 8
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if self.tokenizer is not None
            else 77
        )
        self.default_sample_size = 128

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _get_clip_prompt_embeds(
        self, prompt: str | list[str], num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.vae.device), output_hidden_states=False,
        )
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.vae.device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_2(
            text_inputs.input_ids.to(self.vae.device), output_hidden_states=False,
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.vae.device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt, num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_2, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        t5_dtype = self.text_encoder_2.dtype
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=t5_dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.vae.device, dtype=t5_dtype,
        )
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # -----------------------------------------------------------------------
    # Latent preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int, height: int, width: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor, batch_size: int,
        num_channels: int, height: int, width: int,
    ) -> torch.Tensor:
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, height: int, width: int, vae_scale_factor: int,
    ) -> torch.Tensor:
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = 2 * (int(height) // (self.vae_scale_factor * 2))
        w = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, h, w)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, h, w)
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, h // 2, w // 2, device, dtype,
        )
        return latents, latent_image_ids

    def prepare_timesteps(
        self, num_inference_steps: int, sigmas: list[float] | None, image_seq_len: int,
    ) -> tuple[torch.Tensor, int]:
        sigmas_arr = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        mu = _calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, sigmas=sigmas_arr, mu=mu,
        )
        return timesteps, num_inference_steps

    # -----------------------------------------------------------------------
    # Denoising
    # -----------------------------------------------------------------------

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        latents: torch.Tensor,
        latent_image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        self.scheduler.set_begin_index(0)
        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(
                device=latents.device, dtype=latents.dtype,
            )
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # -----------------------------------------------------------------------
    # Forward (full pipeline)
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        prompts: str | list[str],
        params: DiffusionSamplingParams | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> DiffusionOutput:
        params = params or DiffusionSamplingParams()
        if isinstance(prompts, str):
            prompts = [prompts]

        height = params.height or self.default_sample_size * self.vae_scale_factor
        width = params.width or self.default_sample_size * self.vae_scale_factor
        batch_size = len(prompts)
        num_images = params.num_outputs_per_prompt
        device = self.vae.device

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompts,
            num_images_per_prompt=num_images,
            max_sequence_length=params.max_sequence_length,
        )

        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images, num_channels_latents,
            height, width,
            prompt_embeds.dtype, device, generator,
        )

        timesteps, _ = self.prepare_timesteps(
            params.num_inference_steps, params.sigmas, latents.shape[1],
        )

        if self.transformer.guidance_embeds:
            guidance = torch.full(
                [1], params.guidance_scale, dtype=prompt_embeds.dtype, device=device,
            ).expand(latents.shape[0])
        else:
            guidance = None

        latents = self.diffuse(
            prompt_embeds, pooled_prompt_embeds,
            latents, latent_image_ids, text_ids,
            timesteps, guidance,
        )

        if params.output_type == "latent":
            return DiffusionOutput(latents=latents)

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents = latents.to(dtype=self.vae.dtype)
        images = self.vae.decode(latents, return_dict=False)[0]

        image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        pil_images = image_processor.postprocess(images)

        return DiffusionOutput(images=pil_images, latents=latents)
