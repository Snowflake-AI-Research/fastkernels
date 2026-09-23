"""Adapt Oasis's diagonal-Gaussian sampling to VibeVoice's random scale."""

import torch


def sample_latents(mean, vae_std):
    """Retain the parent reparameterization: mean + std * Gaussian noise.

    Parent: L3/oasis_autoencoder_kl.DiagonalGaussianDistribution.sample.
    VibeVoice samples one signed scale per batch item before the latent noise,
    and both draws use the latent dtype. These draws and all pointwise work
    remain inside the measured audio encoder path.
    """
    std = vae_std * torch.randn(mean.shape[0], device=mean.device, dtype=mean.dtype)
    return mean + std[:, None, None] * torch.randn_like(mean)
