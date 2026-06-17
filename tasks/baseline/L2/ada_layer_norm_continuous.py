"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

Implementation copied from diffusers' ``AdaLayerNormContinuous``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
