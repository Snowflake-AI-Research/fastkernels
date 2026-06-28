"""Sinusoidal 2D position encoding for SAM3.

Generates position embeddings from spatial dimensions using sine/cosine
functions along x and y axes. Used by the FPN neck and geometry encoder.

Reference: sam3/model/model_misc.py PositionEmbeddingSine
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Sam3PositionEncoding(nn.Module):
    """2D sinusoidal position encoding.

    Produces (batch, d_model, H, W) position embeddings from a (batch, C, H, W)
    feature map. Only the spatial dimensions are used; the channel content is
    ignored.

    Also exposes ``_encode_xy`` for encoding arbitrary (x, y) coordinates and
    ``encode_boxes`` for encoding (cx, cy, w, h) boxes (used by the geometry
    encoder).
    """

    def __init__(
        self,
        d_model: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float | None = None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.half_d = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def _encode_xy(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode flattened x and y coordinates into sinusoidal embeddings.

        Args:
            x: 1-D tensor of x positions in [0, 1].
            y: 1-D tensor of y positions in [0, 1].

        Returns:
            Tuple of (enc_x, enc_y) each with shape (N, half_d).
        """
        dim_t = torch.arange(self.half_d, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.half_d)

        pos_x = x[:, None] * self.scale / dim_t[None, :]
        pos_y = y[:, None] * self.scale / dim_t[None, :]

        enc_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        enc_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return enc_x, enc_y

    def encode_boxes(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Encode box coordinates (cx, cy, w, h) each in [0, 1].

        Returns a (N, d_model + 2) tensor: [enc_x, enc_y, w, h].
        """
        enc_x, enc_y = self._encode_xy(cx, cy)
        return torch.cat([enc_y, enc_x, h[:, None], w[:, None]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate position encoding for a feature map.

        Args:
            x: Feature map of shape (batch, C, H, W).

        Returns:
            Position encoding of shape (batch, d_model, H, W).
        """
        B, _, H, W = x.shape
        device = x.device

        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1] + eps) * self.scale
            x_embed = x_embed / (x_embed[-1] + eps) * self.scale

        dim_t = torch.arange(self.half_d, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.half_d)

        pos_x = x_embed[:, None] / dim_t[None, :]  # (W, half_d)
        pos_y = y_embed[:, None] / dim_t[None, :]  # (H, half_d)

        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)  # (W, half_d)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)  # (H, half_d)

        pos = torch.cat(
            [
                pos_y[:, None, :].expand(-1, W, -1),
                pos_x[None, :, :].expand(H, -1, -1),
            ],
            dim=-1,
        )  # (H, W, d_model)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos
