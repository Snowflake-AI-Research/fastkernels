"""SDXL text-to-image pipeline (L4).

Contains:
- SDXLConfig: UNet configuration dataclass
- UNet2DConditionModel: the UNet backbone wired from L3 blocks
- SDXLPipeline: full text-to-image pipeline (encode, diffuse, decode)

L4 wiring/configuration; computation lives in L1-L3 tasks.
External imports (L4 only): AutoencoderKL, EulerDiscreteScheduler, CLIPTextModel.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from torch import nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from ..L1.conv2d import Conv2d
from ..L1.group_norm import GroupNorm
from ..L1.silu import SiLU
from ..L2.sdxl_time_embedding import TextTimeEmbedding
from ..L2.timestep_embedding import Timesteps, TimestepEmbedding
from ..L3.sdxl_unet_block import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)
from ..L1.video_processor import VideoProcessor as VaeImageProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SDXLConfig:
    """Configuration for SDXL UNet2DConditionModel."""
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280)
    layers_per_block: int = 2
    attention_head_dim: tuple[int, ...] = (5, 10, 20)  # actually num_heads per block
    cross_attention_dim: int = 2048
    transformer_layers_per_block: tuple[int, ...] = (1, 2, 10)
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    use_linear_projection: bool = True
    addition_embed_type: str = "text_time"
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int = 2816
    flip_sin_to_cos: bool = True
    freq_shift: int = 0

    @classmethod
    def from_pretrained(cls, model_name: str) -> "SDXLConfig":
        from huggingface_hub import hf_hub_download
        if os.path.isdir(model_name):
            config_path = os.path.join(model_name, "unet", "config.json")
        else:
            config_path = hf_hub_download(model_name, "unet/config.json")
        with open(config_path) as f:
            data = json.load(f)
        return cls(
            in_channels=data.get("in_channels", 4),
            out_channels=data.get("out_channels", 4),
            block_out_channels=tuple(data.get("block_out_channels", [320, 640, 1280])),
            layers_per_block=data.get("layers_per_block", 2),
            attention_head_dim=tuple(data.get("attention_head_dim", [5, 10, 20])),
            cross_attention_dim=data.get("cross_attention_dim", 2048),
            transformer_layers_per_block=tuple(data.get("transformer_layers_per_block", [1, 2, 10])),
            norm_num_groups=data.get("norm_num_groups", 32),
            norm_eps=data.get("norm_eps", 1e-5),
            use_linear_projection=data.get("use_linear_projection", True),
            addition_embed_type=data.get("addition_embed_type", "text_time"),
            addition_time_embed_dim=data.get("addition_time_embed_dim", 256),
            projection_class_embeddings_input_dim=data.get("projection_class_embeddings_input_dim", 2816),
            flip_sin_to_cos=data.get("flip_sin_to_cos", True),
            freq_shift=data.get("freq_shift", 0),
        )


@dataclass
class SDXLSamplingParams:
    """Sampling parameters for SDXL generation."""
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    num_outputs_per_prompt: int = 1
    seed: int | None = None
    output_type: str = "pil"
    original_size: tuple[int, int] | None = None
    crops_coords_top_left: tuple[int, int] = (0, 0)
    target_size: tuple[int, int] | None = None


@dataclass
class SDXLOutput:
    """Output of the SDXL pipeline."""
    images: list[Any] | None = None
    latents: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# UNet2DConditionModel
# ---------------------------------------------------------------------------

class UNet2DConditionModel(nn.Module):
    """SDXL UNet backbone wired from L3 blocks.

    Architecture: conv_in -> down_blocks -> mid_block -> up_blocks -> conv_norm_out -> conv_out
    """

    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        boc = config.block_out_channels

        time_embed_dim = boc[0] * 4

        # Input conv
        self.conv_in = Conv2d(config.in_channels, boc[0], kernel_size=3, padding=1)

        # Time embedding
        self.time_proj = Timesteps(
            num_channels=boc[0],
            flip_sin_to_cos=config.flip_sin_to_cos,
            downscale_freq_shift=config.freq_shift,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=boc[0],
            time_embed_dim=time_embed_dim,
        )

        # SDXL text_time conditioning
        self.add_time_proj = Timesteps(
            num_channels=config.addition_time_embed_dim,
            flip_sin_to_cos=config.flip_sin_to_cos,
            downscale_freq_shift=0,
        )
        self.add_embedding = TextTimeEmbedding(
            text_time_input_dim=config.projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        # Down blocks: [DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D]
        # attention_head_dim is actually num_attention_heads per block
        # (diffusers naming quirk, see https://github.com/huggingface/diffusers/issues/2011).
        # SDXL values [5, 10, 20] give head_dim = ch / num_heads = 64 for all blocks.

        self.down_blocks = nn.ModuleList()

        # Block 0: DownBlock2D (320 -> 320, no attention)
        self.down_blocks.append(
            DownBlock2D(
                in_channels=boc[0],
                out_channels=boc[0],
                temb_channels=time_embed_dim,
                num_layers=config.layers_per_block,
                resnet_groups=config.norm_num_groups,
                resnet_eps=config.norm_eps,
                add_downsample=True,
            )
        )

        # Block 1: CrossAttnDownBlock2D (320 -> 640)
        self.down_blocks.append(
            CrossAttnDownBlock2D(
                in_channels=boc[0],
                out_channels=boc[1],
                temb_channels=time_embed_dim,
                num_layers=config.layers_per_block,
                transformer_layers_per_block=config.transformer_layers_per_block[1],
                num_attention_heads=config.attention_head_dim[1],
                cross_attention_dim=config.cross_attention_dim,
                resnet_groups=config.norm_num_groups,
                resnet_eps=config.norm_eps,
                add_downsample=True,
                use_linear_projection=config.use_linear_projection,
            )
        )

        # Block 2: CrossAttnDownBlock2D (640 -> 1280, no downsample)
        self.down_blocks.append(
            CrossAttnDownBlock2D(
                in_channels=boc[1],
                out_channels=boc[2],
                temb_channels=time_embed_dim,
                num_layers=config.layers_per_block,
                transformer_layers_per_block=config.transformer_layers_per_block[2],
                num_attention_heads=config.attention_head_dim[2],
                cross_attention_dim=config.cross_attention_dim,
                resnet_groups=config.norm_num_groups,
                resnet_eps=config.norm_eps,
                add_downsample=False,
                use_linear_projection=config.use_linear_projection,
            )
        )

        # Mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=boc[-1],
            temb_channels=time_embed_dim,
            transformer_layers_per_block=config.transformer_layers_per_block[-1],
            num_attention_heads=config.attention_head_dim[-1],
            cross_attention_dim=config.cross_attention_dim,
            resnet_groups=config.norm_num_groups,
            resnet_eps=config.norm_eps,
            use_linear_projection=config.use_linear_projection,
        )

        # Up blocks (reversed)
        # Block 0: CrossAttnUpBlock2D (1280, prev=1280, 2 transformer layers)
        # Block 1: CrossAttnUpBlock2D (640, prev=1280, 1 transformer layer)
        # Block 2: UpBlock2D (320, prev=640, no attention)
        reversed_boc = list(reversed(boc))
        reversed_head_dim = list(reversed(config.attention_head_dim))
        reversed_tlpb = list(reversed(config.transformer_layers_per_block))

        self.up_blocks = nn.ModuleList()

        # Up block 0: CrossAttnUpBlock2D 1280
        self.up_blocks.append(
            CrossAttnUpBlock2D(
                in_channels=reversed_boc[min(1, len(reversed_boc) - 1)],
                prev_output_channel=reversed_boc[0],
                out_channels=reversed_boc[0],
                temb_channels=time_embed_dim,
                num_layers=config.layers_per_block + 1,
                transformer_layers_per_block=reversed_tlpb[0],
                num_attention_heads=reversed_head_dim[0],
                cross_attention_dim=config.cross_attention_dim,
                resnet_groups=config.norm_num_groups,
                resnet_eps=config.norm_eps,
                add_upsample=True,
                use_linear_projection=config.use_linear_projection,
            )
        )

        # Up block 1: CrossAttnUpBlock2D 640
        self.up_blocks.append(
            CrossAttnUpBlock2D(
                in_channels=reversed_boc[min(2, len(reversed_boc) - 1)],
                prev_output_channel=reversed_boc[0],
                out_channels=reversed_boc[1],
                temb_channels=time_embed_dim,
                num_layers=config.layers_per_block + 1,
                transformer_layers_per_block=reversed_tlpb[1],
                num_attention_heads=reversed_head_dim[1],
                cross_attention_dim=config.cross_attention_dim,
                resnet_groups=config.norm_num_groups,
                resnet_eps=config.norm_eps,
                add_upsample=True,
                use_linear_projection=config.use_linear_projection,
            )
        )

        # Up block 2: UpBlock2D 320
        self.up_blocks.append(
            UpBlock2D(
                in_channels=reversed_boc[min(2, len(reversed_boc) - 1)],
                prev_output_channel=reversed_boc[1],
                out_channels=reversed_boc[2],
                temb_channels=time_embed_dim,
                num_layers=config.layers_per_block + 1,
                resnet_groups=config.norm_num_groups,
                resnet_eps=config.norm_eps,
                add_upsample=False,
            )
        )

        # Output
        self.conv_norm_out = GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=boc[0],
            eps=config.norm_eps,
        )
        self.conv_act = SiLU()
        self.conv_out = Conv2d(boc[0], config.out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        # 0. Time embedding
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])
        elif timestep.ndim == 1 and timestep.shape[0] == 1:
            timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # SDXL text_time augmentation
        if added_cond_kwargs is not None:
            text_embeds = added_cond_kwargs["text_embeds"]
            time_ids = added_cond_kwargs["time_ids"]
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape(text_embeds.shape[0], -1)
            add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(dtype=emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb

        # 1. Input
        sample = self.conv_in(sample)

        # 2. Down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if hasattr(down_block, "attentions"):
                sample, res_samples = down_block(
                    sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = down_block(sample, temb=emb)
            down_block_res_samples += res_samples

        # 3. Mid
        sample = self.mid_block(
            sample, temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        # 4. Up
        for up_block in self.up_blocks:
            n_resnets = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if hasattr(up_block, "attentions"):
                sample = up_block(
                    sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = up_block(
                    sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                )

        # 5. Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        return sample

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load diffusers checkpoint weights into this model."""
        params_dict = dict(self.named_parameters())
        for name, buf in self.named_buffers():
            params_dict[name] = buf

        loaded: set[str] = set()
        missing_in_model = []
        for name, tensor in weights:
            if name in params_dict:
                params_dict[name].data.copy_(tensor)
                loaded.add(name)
            else:
                missing_in_model.append(name)

        if missing_in_model:
            logger.warning(
                "UNet load_weights: %d checkpoint keys not found in model (first 10: %s)",
                len(missing_in_model), missing_in_model[:10],
            )

        model_keys = set(params_dict.keys())
        not_loaded = model_keys - loaded
        if not_loaded:
            logger.warning(
                "UNet load_weights: %d model keys not loaded from checkpoint (first 10: %s)",
                len(not_loaded), sorted(not_loaded)[:10],
            )

        return loaded


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SDXLPipeline(nn.Module):
    """Full SDXL text-to-image pipeline.

    Composes: dual CLIP text encoding -> latent preparation -> denoising
    loop (Euler discrete) -> VAE decode -> postprocess.
    """

    def __init__(
        self,
        config: SDXLConfig,
        model_name: str,
        torch_dtype: torch.dtype | None = None,
        variant: str | None = "fp16",
    ):
        super().__init__()
        self.config = config
        self.model_name = model_name

        local_files_only = os.path.isdir(model_name)
        hf_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if torch_dtype is not None:
            hf_kwargs["torch_dtype"] = torch_dtype
        if variant is not None:
            hf_kwargs["variant"] = variant

        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler", local_files_only=local_files_only,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer_2", local_files_only=local_files_only,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", **hf_kwargs,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder_2", **hf_kwargs,
        )
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", **hf_kwargs,
        )

        self.unet = UNet2DConditionModel(config)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels")
            else 8
        )
        self.default_sample_size = 128
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts with dual CLIP text encoders."""
        device = self.vae.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        batch_size = len(prompt)

        # Encode with CLIP text_encoder (text_encoder)
        text_inputs = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds_1 = self.text_encoder(
            text_inputs.input_ids.to(device),
            output_hidden_states=True,
        )
        prompt_embeds = prompt_embeds_1.hidden_states[-2]

        # Encode with CLIP text_encoder_2
        text_inputs_2 = self.tokenizer_2(
            prompt_2, padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds_2 = self.text_encoder_2(
            text_inputs_2.input_ids.to(device),
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds_2[0]
        prompt_embeds_2_hidden = prompt_embeds_2.hidden_states[-2]

        # Concat the two text encoder hidden states
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2_hidden], dim=-1)
        prompt_embeds = prompt_embeds.to(device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

        # Negative prompt encoding for CFG
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_classifier_free_guidance:
            zero_out = negative_prompt is None
            if zero_out:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            else:
                neg = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                neg_2 = negative_prompt_2 or neg
                neg_2 = [neg_2] if isinstance(neg_2, str) else neg_2

                neg_inputs = self.tokenizer(
                    neg, padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                )
                neg_embeds_1 = self.text_encoder(
                    neg_inputs.input_ids.to(device),
                    output_hidden_states=True,
                )
                negative_prompt_embeds = neg_embeds_1.hidden_states[-2]

                neg_inputs_2 = self.tokenizer_2(
                    neg_2, padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True, return_tensors="pt",
                )
                neg_embeds_2 = self.text_encoder_2(
                    neg_inputs_2.input_ids.to(device),
                    output_hidden_states=True,
                )
                negative_pooled_prompt_embeds = neg_embeds_2[0]
                neg_embeds_2_hidden = neg_embeds_2.hidden_states[-2]

                negative_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, neg_embeds_2_hidden], dim=-1
                )
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_images_per_prompt, 1
                ).view(bs_embed * num_images_per_prompt, seq_len, -1)

                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                    1, num_images_per_prompt
                ).view(bs_embed * num_images_per_prompt, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    # -----------------------------------------------------------------------
    # Time IDs
    # -----------------------------------------------------------------------

    def _get_add_time_ids(
        self,
        original_size: tuple[int, int],
        crops_coords_top_left: tuple[int, int],
        target_size: tuple[int, int],
        dtype: torch.dtype,
        text_encoder_projection_dim: int,
    ) -> torch.Tensor:
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # -----------------------------------------------------------------------
    # Latent preparation
    # -----------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # -----------------------------------------------------------------------
    # Forward (full pipeline)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        prompts: str | list[str],
        params: SDXLSamplingParams | None = None,
        generator: torch.Generator | None = None,
    ) -> SDXLOutput:
        params = params or SDXLSamplingParams()
        if isinstance(prompts, str):
            prompts = [prompts]

        height = params.height
        width = params.width
        batch_size = len(prompts)
        device = self.vae.device
        do_cfg = params.guidance_scale > 1.0

        original_size = params.original_size or (height, width)
        target_size = params.target_size or (height, width)

        # 1. Encode prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompts,
            num_images_per_prompt=params.num_outputs_per_prompt,
            do_classifier_free_guidance=do_cfg,
        )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(params.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latents
        num_channels_latents = self.config.in_channels
        latents = self.prepare_latents(
            batch_size * params.num_outputs_per_prompt,
            num_channels_latents,
            height, width,
            prompt_embeds.dtype,
            device, generator,
        )

        # 4. Prepare added time ids
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size, params.crops_coords_top_left, target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device)

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        else:
            add_text_embeds = pooled_prompt_embeds

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * params.num_outputs_per_prompt, 1
        )

        # 5. Denoising loop
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        for t in timesteps:
            latent_model_input = (
                torch.cat([latents] * 2) if do_cfg else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + params.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False,
            )[0]

        # 6. Decode
        if params.output_type == "latent":
            return SDXLOutput(latents=latents)

        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        pil_images = self.image_processor.postprocess(images, output_type=params.output_type)

        return SDXLOutput(images=pil_images, latents=latents)
