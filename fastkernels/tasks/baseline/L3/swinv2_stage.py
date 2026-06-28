"""SwinV2 stage: optional downsample + sequence of transformer blocks (L3).

One stage of the hierarchical SwinV2 architecture. Contains an optional
PatchMerging downsample layer followed by a sequence of SwinV2Blocks
with alternating W-MSA (even blocks) and SW-MSA (odd blocks).

Reference: timm/models/swin_transformer_v2.py SwinTransformerV2Stage
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..L2.swinv2_patch_merging import SwinV2PatchMerging
from .swinv2_block import SwinV2Block


_int_or_tuple_2_t = Union[int, Tuple[int, int]]


def _to_2tuple(x: _int_or_tuple_2_t) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


class SwinV2Stage(nn.Module):
    """A single SwinV2 stage.

    Args:
        dim: Input channel dimension.
        out_dim: Output channel dimension.
        input_resolution: Spatial resolution (H, W) entering this stage.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        window_size: Local window size.
        downsample: Whether to apply PatchMerging at the start.
        mlp_ratio: MLP hidden-dim expansion ratio.
        qkv_bias: If True, add learnable bias to QKV.
        proj_drop: Projection dropout rate.
        attn_drop: Attention dropout rate.
        pretrained_window_size: Pretrained window size for CPB normalization.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: _int_or_tuple_2_t,
        downsample: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        pretrained_window_size: _int_or_tuple_2_t = 0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        )
        self.depth = depth

        window_size = _to_2tuple(window_size)
        shift_size = (window_size[0] // 2, window_size[1] // 2)

        if downsample:
            self.downsample = SwinV2PatchMerging(dim=dim, out_dim=out_dim)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList([
            SwinV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x
