"""2D Rotary Position Embedding for SAM3 ViT attention.

Computes axial complex-valued frequency tables for 2D spatial layouts, then
applies them to query/key tensors. Supports windowed (tiled) and global
(interpolated) RoPE modes.

Reference: sam3/model/vitdet.py compute_axial_cis / apply_rotary_enc
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
) -> torch.Tensor:
    """Build 2-D axial complex frequency table.

    Returns a (end_x * end_y, dim // 2) complex tensor.
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float() * scale_pos + offset
    t_y = torch.div(t, end_x, rounding_mode="floor").float() * scale_pos + offset

    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


class Sam3RoPE2D(nn.Module):
    """2-D axial rotary position embedding for SAM3 ViT.

    Pre-computes frequency tables at init time and applies rotary encoding to
    (query, key) pairs in the attention forward pass.

    Args:
        head_dim: Per-head dimension (must be divisible by 2).
        input_size: Spatial resolution seen by this attention op (H, W).
        theta: Base frequency for RoPE.
        tiled: If True, tile the frequencies from ``pt_size`` to ``input_size``
            instead of interpolating.
        pt_size: Pre-training spatial resolution for tiling/interpolation.
        interp: If True, interpolate (scale) frequencies to ``input_size``.
        cls_token: If True, prepend a zero-frequency row for the CLS token.
    """

    def __init__(
        self,
        head_dim: int,
        input_size: tuple[int, int],
        theta: float = 10000.0,
        tiled: bool = False,
        pt_size: tuple[int, int] | None = None,
        interp: bool = False,
        cls_token: bool = False,
    ):
        super().__init__()
        if pt_size is None:
            pt_size = input_size

        if pt_size != input_size and tiled:
            freqs_cis = compute_axial_cis(
                dim=head_dim, end_x=pt_size[0], end_y=pt_size[1], theta=theta,
            )
            freqs_cis = (
                freqs_cis.reshape(pt_size[0], pt_size[1], -1)
                .tile(
                    input_size[0] // pt_size[0],
                    input_size[1] // pt_size[1],
                    1,
                )
                .reshape(-1, freqs_cis.shape[-1])
            )
        else:
            scale_pos = pt_size[0] / input_size[0] if interp else 1.0
            freqs_cis = compute_axial_cis(
                dim=head_dim,
                end_x=input_size[0],
                end_y=input_size[1],
                theta=theta,
                scale_pos=scale_pos,
            )

        if cls_token:
            t = torch.zeros(head_dim // 2, dtype=torch.float32)
            cls_freqs = torch.polar(torch.ones_like(t), t)[None, :]
            freqs_cis = torch.cat([cls_freqs, freqs_cis], dim=0)

        self.register_buffer("freqs_cis", freqs_cis)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2-D rotary encoding to q and k.

        Args:
            q: (B, num_heads, L, head_dim)
            k: (B, num_heads, L, head_dim)

        Returns:
            Rotated (q, k) with same shape and dtype as inputs.
        """
        xq_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

        freqs = self.freqs_cis
        ndim = xq_.ndim
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(xq_.shape)]
        freqs = freqs.view(*shape)

        xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
        return xq_out.type_as(q), xk_out.type_as(k)
