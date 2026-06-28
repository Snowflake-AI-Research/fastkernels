"""Diffusion conditioning for AlphaFold3.

Produces conditioned single and pair representations from trunk outputs
and diffusion time step. Implements Fourier time embedding and optional
trunk conditioning.

Reference: openfold3/core/model/layers/diffusion_conditioning.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_input_embedder import relpos_complex
from .alphafold3_swiglu_transition import SwiGLUTransition


class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning.

    Uses random Fourier features (matching the reference's seeded initialization).

    Args:
        c: Embedding dimension (256 in the reference)
        seed: Random seed for weight initialization
    """

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        self.c = c
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer(
            "w", torch.randn(c, generator=generator),
        )
        self.register_buffer(
            "b", torch.randn(c, generator=generator),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b
        return torch.cos(2 * math.pi * x)


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module.

    Matches the reference:
    - Pair: concat([zij_trunk, relpos], dim=-1) -> LayerNorm -> Linear -> 2x SwiGLU transition
    - Single: concat([si_trunk, si_input], dim=-1) -> LayerNorm -> Linear + fourier -> 2x SwiGLU transition

    Reference: openfold3/core/model/layers/diffusion_conditioning.py

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_input: Input single representation dimension (449)
        sigma_data: Noise level scaling for Fourier embedding
        relpos_k: Maximum relative position for pair bias
        max_relative_chain: Maximum relative chain index
        c_fourier_emb: Fourier embedding dimension (256)
        seed_fourier_emb: Fourier embedding random seed
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )

        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:     Feature dictionary (needs asym_id, entity_id etc. for relpos)
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            si:  [*, N_token, c_s] conditioned single rep
            zij: [*, N_token, N_token, c_z] conditioned pair rep
        """
        if use_conditioning:
            # Pair conditioning: concat trunk pair with relpos features
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,),
                )

            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))

            # Single conditioning: concat trunk single with input
            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        # Fourier noise embedding
        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        # Apply transition layers
        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)

        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)

        return si, zij
