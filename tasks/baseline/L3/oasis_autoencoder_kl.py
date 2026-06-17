"""Oasis VAE autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.oasis_patch_embed import OasisPatchEmbed
from .oasis_vae_attention_block import OasisVAEAttentionBlock


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False, dim: int = 1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)

    def mode(self) -> torch.Tensor:
        return self.mean


class OasisAutoencoderKL(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        input_height: int = 360,
        input_width: int = 640,
        patch_size: int = 20,
        enc_dim: int = 1024,
        enc_depth: int = 6,
        enc_heads: int = 16,
        dec_dim: int = 1024,
        dec_depth: int = 12,
        dec_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_variational: bool = True,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size ** 2
        self.latent_dim = latent_dim
        self.use_variational = use_variational

        self.patch_embed = OasisPatchEmbed(input_height, input_width, patch_size, 3, enc_dim)
        self.encoder = nn.ModuleList(
            [
                OasisVAEAttentionBlock(
                    enc_dim,
                    enc_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(enc_depth)
            ]
        )
        self.enc_norm = LayerNorm(enc_dim, eps=1e-6)

        mult = 2 if self.use_variational else 1
        self.quant_conv = Linear(enc_dim, mult * latent_dim, bias=True)
        self.post_quant_conv = Linear(latent_dim, dec_dim, bias=True)

        self.decoder = nn.ModuleList(
            [
                OasisVAEAttentionBlock(
                    dec_dim,
                    dec_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_norm = LayerNorm(dec_dim, eps=1e-6)
        self.predictor = Linear(dec_dim, self.patch_dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init_weights(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, LayerNorm):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)

        self.apply(_init_weights)
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute(0, 3, 1, 2)
        x = x.reshape(bsz, 3, self.patch_size, self.patch_size, self.seq_h, self.seq_w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.reshape(bsz, 3, self.input_height, self.input_width)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        x = self.patch_embed(x)
        for block in self.encoder:
            x = block(x)
        x = self.enc_norm(x)
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), dim=2)
        return DiagonalGaussianDistribution(moments, deterministic=not self.use_variational, dim=2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        for block in self.decoder:
            z = block(z)
        z = self.dec_norm(z)
        z = self.predictor(z)
        return self.unpatchify(z)

    def autoencode(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution, torch.Tensor]:
        posterior = self.encode(input)
        if self.use_variational and sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z

    def get_input(self, batch: dict[str, torch.Tensor], k: str) -> torch.Tensor:
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        split: str = "train",
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution, torch.Tensor]:
        del labels, split
        rec, post, latent = self.autoencode(inputs)
        return rec, post, latent

    def get_last_layer(self) -> torch.Tensor:
        return self.predictor.weight
