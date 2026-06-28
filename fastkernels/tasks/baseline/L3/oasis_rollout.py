"""Oasis latent encoding, decoding, and rollout sampling."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from .oasis_autoencoder_kl import OasisAutoencoderKL
from .oasis_dit import OasisDiT


class OasisRollout(nn.Module):
    def __init__(
        self,
        scaling_factor: float,
        max_noise_level: int,
        stabilization_level: int,
        noise_abs_max: float,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.max_noise_level = max_noise_level
        self.stabilization_level = stabilization_level
        self.noise_abs_max = noise_abs_max

    @staticmethod
    def _autocast(device: torch.device, dtype: torch.dtype):
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    @staticmethod
    def sigmoid_beta_schedule(
        timesteps: int,
        start: float = -3,
        end: float = 3,
        tau: float = 1,
        clamp_min: float = 0.0,
    ) -> torch.Tensor:
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, clamp_min, 0.999)

    def encode_prompt(self, vae: OasisAutoencoderKL, prompt: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        bsz, frames, channels, height, width = prompt.shape
        prompt = prompt.reshape(bsz * frames, channels, height, width)
        with torch.inference_mode(), self._autocast(prompt.device, dtype):
            posterior = vae.encode(prompt * 2 - 1)
            latents = posterior.mean * self.scaling_factor
        h = height // vae.patch_size
        w = width // vae.patch_size
        return latents.reshape(bsz, frames, h, w, latents.shape[-1]).permute(0, 1, 4, 2, 3)

    def decode_latents(self, vae: OasisAutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
        bsz, frames, channels, height, width = latents.shape
        target_dtype = vae.post_quant_conv.weight.dtype
        latents = latents.permute(0, 1, 3, 4, 2).reshape(bsz * frames, height * width, channels).to(target_dtype)
        with torch.inference_mode():
            decoded = (vae.decode(latents / self.scaling_factor) + 1) / 2
        return decoded.reshape(bsz, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])

    def forward(
        self,
        model: OasisDiT,
        vae: OasisAutoencoderKL,
        prompt: torch.Tensor,
        actions: torch.Tensor,
        *,
        num_frames: int,
        ddim_steps: int,
        n_prompt_frames: int,
        seed: int | None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = prompt.device
        prompt_latents = self.encode_prompt(vae, prompt, dtype=dtype)[:, :n_prompt_frames]
        x = prompt_latents
        noise_range = torch.linspace(-1, self.max_noise_level - 1, ddim_steps + 1, device=device)

        betas = self.sigmoid_beta_schedule(self.max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).reshape(-1, 1, 1, 1)

        generator = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)

        for index in range(n_prompt_frames, num_frames):
            chunk = torch.randn((prompt.shape[0], 1, *x.shape[-3:]), generator=generator, device=device)
            chunk = torch.clamp(chunk, -self.noise_abs_max, self.noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, index + 1 - model.max_frames)

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_ctx = torch.full(
                    (prompt.shape[0], index),
                    self.stabilization_level - 1,
                    dtype=torch.long,
                    device=device,
                )
                t = torch.full((prompt.shape[0], 1), noise_range[noise_idx], dtype=torch.long, device=device)
                t_next = torch.full((prompt.shape[0], 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                x_curr = x[:, start_frame:].clone()
                t_curr = t[:, start_frame:]
                t_next_curr = t_next[:, start_frame:]

                with torch.inference_mode(), self._autocast(prompt.device, dtype):
                    v = model(x_curr, t_curr, actions[:, start_frame:index + 1])

                x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                    1 / alphas_cumprod[t_curr] - 1
                ).sqrt()

                alpha_next = alphas_cumprod[t_next_curr]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        video = self.decode_latents(vae, x)
        return video, x, prompt_latents
