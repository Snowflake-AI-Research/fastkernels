"""Oasis 500M world-model pipeline wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors.torch import load_file

from ..L3.oasis_autoencoder_kl import OasisAutoencoderKL
from ..L3.oasis_dit import OasisDiT
from ..L3.oasis_rollout import OasisRollout

sigmoid_beta_schedule = OasisRollout.sigmoid_beta_schedule


def DiT_S_2(*, max_frames: int = 32) -> OasisDiT:
    return OasisDiT(patch_size=2, hidden_size=1024, depth=16, num_heads=16, max_frames=max_frames)


def ViT_L_20_Shallow_Encoder() -> OasisAutoencoderKL:
    return OasisAutoencoderKL(
        latent_dim=16,
        patch_size=20,
        enc_dim=1024,
        enc_depth=6,
        enc_heads=16,
        dec_dim=1024,
        dec_depth=12,
        dec_heads=16,
        input_height=360,
        input_width=640,
    )


@dataclass
class OasisConfig:
    dit_variant: str = "DiT-S/2"
    vae_variant: str = "vit-l-20-shallow-encoder"
    scaling_factor: float = 0.07843137255
    max_noise_level: int = 1000
    max_frames: int = 32
    stabilization_level: int = 15
    noise_abs_max: float = 20.0


@dataclass
class OasisSamplingParams:
    num_frames: int = 8
    ddim_steps: int = 4
    n_prompt_frames: int = 1
    video_offset: int | None = None
    seed: int | None = None
    output_type: str = "video"


@dataclass
class OasisOutput:
    video: torch.Tensor
    latents: torch.Tensor
    prompt_latents: torch.Tensor


class OasisPipeline(nn.Module):
    def __init__(self, config: OasisConfig):
        super().__init__()
        self.config = config
        if config.dit_variant != "DiT-S/2":
            raise ValueError(f"unsupported Oasis DiT variant: {config.dit_variant}")
        if config.vae_variant != "vit-l-20-shallow-encoder":
            raise ValueError(f"unsupported Oasis VAE variant: {config.vae_variant}")
        self.model = DiT_S_2(max_frames=config.max_frames)
        self.vae = ViT_L_20_Shallow_Encoder()
        self.rollout_engine = OasisRollout(
            scaling_factor=config.scaling_factor,
            max_noise_level=config.max_noise_level,
            stabilization_level=config.stabilization_level,
            noise_abs_max=config.noise_abs_max,
        )

    def load_weights(self, model_dir: str) -> None:
        dit_path = os.path.join(model_dir, "oasis500m.safetensors")
        vae_path = os.path.join(model_dir, "vit-l-20.safetensors")
        self.model.load_state_dict(load_file(dit_path), strict=False)
        self.vae.load_state_dict(load_file(vae_path), strict=False)

    def encode_prompt(self, prompt: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.rollout_engine.encode_prompt(self.vae, prompt, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.rollout_engine.decode_latents(self.vae, latents)

    def rollout(
        self,
        prompt: torch.Tensor,
        actions: torch.Tensor,
        params: OasisSamplingParams,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> OasisOutput:
        video, latents, prompt_latents = self.rollout_engine(
            self.model,
            self.vae,
            prompt,
            actions,
            num_frames=params.num_frames,
            ddim_steps=params.ddim_steps,
            n_prompt_frames=params.n_prompt_frames,
            seed=params.seed,
            dtype=dtype,
        )
        return OasisOutput(video=video, latents=latents, prompt_latents=prompt_latents)
