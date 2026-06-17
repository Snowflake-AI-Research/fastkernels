"""ViT-Det backbone for SAM3.

Full Vision Transformer with patch embedding, absolute/relative position
encoding, windowed and global attention blocks, and optional 2D RoPE.
Outputs one or more feature maps from global attention stages.

Reference: sam3/model/vitdet.py ViT
"""

from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.conv2d import Conv2d
from ..L1.layer_norm import LayerNorm
from .sam3_vit_block import Sam3ViTBlock


class Sam3PatchEmbed(nn.Module):
    """Image to patch embedding via convolution."""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # B C H W -> B H W C


class Sam3ViT(nn.Module):
    """ViT-Det backbone for SAM3.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        embed_dim: Patch embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Bias in QKV projection.
        drop_path_rate: Stochastic depth rate.
        window_size: Window size for windowed attention blocks.
        global_att_blocks: Indices of blocks using global attention.
        use_rope: Enable 2D RoPE.
        use_tiled_rope: Tile RoPE from pt_size.
        rope_pt_size: Pre-training resolution for RoPE.
        use_interp_rope: Interpolate RoPE.
        use_abs_pos: Use absolute position embedding.
        tile_abs_pos: Tile absolute position embedding.
        pretrain_img_size: Image size of the pretrained model (for pos embed).
        pretrain_use_cls_token: Whether pretrained model had CLS token.
        retain_cls_token: Keep CLS token in the output.
        dropout: Dropout rate.
        return_interm_layers: Return features from all global attention blocks.
        ln_pre: Apply LayerNorm before blocks.
        ln_post: Apply LayerNorm after blocks.
    """

    def __init__(
        self,
        img_size: int = 1008,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        window_size: int = 14,
        global_att_blocks: Tuple[int, ...] = (7, 15, 23, 31),
        use_rope: bool = True,
        use_tiled_rope: bool = True,
        rope_pt_size: Optional[int] = None,
        use_interp_rope: bool = False,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.retain_cls_token = retain_cls_token

        spatial_size = img_size // patch_size
        window_block_indexes = [i for i in range(depth) if i not in global_att_blocks]
        self.full_attn_ids = list(global_att_blocks)

        if retain_cls_token:
            window_block_indexes = []

        if self.retain_cls_token:
            scale = embed_dim ** -0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        self.patch_embed = Sam3PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        self.use_abs_pos = use_abs_pos
        self.tile_abs_pos = tile_abs_pos
        if self.use_abs_pos:
            pt_patches = pretrain_img_size // patch_size
            num_positions = pt_patches * pt_patches
            if pretrain_use_cls_token:
                num_positions += 1
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            ws = window_size if i in window_block_indexes else 0
            rpt = (
                (window_size, window_size) if rope_pt_size is None
                else (rope_pt_size, rope_pt_size)
            )
            is_windowed = (ws > 0)
            block_rope_tiled = use_tiled_rope and is_windowed
            block_input_size = (ws, ws) if is_windowed else (spatial_size, spatial_size)
            block_rope_interp = use_interp_rope and not is_windowed

            self.blocks.append(Sam3ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                window_size=ws,
                use_rope=use_rope,
                input_size=block_input_size,
                rope_pt_size=rpt if is_windowed else block_input_size,
                rope_tiled=block_rope_tiled,
                rope_interp=block_rope_interp,
                cls_token=retain_cls_token,
                dropout=dropout,
            ))

        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids) if return_interm_layers
            else [embed_dim]
        )

        self.ln_pre = LayerNorm(embed_dim) if ln_pre else nn.Identity()
        self.ln_post = LayerNorm(embed_dim) if ln_post else nn.Identity()

    def _get_abs_pos(self, hw: Tuple[int, int]) -> torch.Tensor:
        """Interpolate/tile absolute position embedding to match spatial size."""
        h, w = hw
        abs_pos = self.pos_embed
        has_cls = self.pretrain_use_cls_token

        if has_cls:
            cls_pos = abs_pos[:, :1]
            abs_pos = abs_pos[:, 1:]

        xy_num = abs_pos.shape[1]
        size = int(math.sqrt(xy_num))

        if size != h or size != w:
            new_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
            if self.tile_abs_pos:
                new_pos = new_pos.tile(
                    [1, 1] + [x // y + 1 for x, y in zip((h, w), new_pos.shape[2:])]
                )[:, :, :h, :w]
            else:
                new_pos = F.interpolate(new_pos, size=(h, w), mode="bicubic", align_corners=False)

            if self.retain_cls_token:
                return torch.cat([cls_pos, new_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)], dim=1)
            return new_pos.permute(0, 2, 3, 1)
        else:
            if self.retain_cls_token:
                return torch.cat([cls_pos, abs_pos], dim=1)
            return abs_pos.reshape(1, h, w, -1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run ViT backbone.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            List of (B, C, h, w) feature maps from global attention stages.
        """
        x = self.patch_embed(x)  # (B, h, w, C)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            x = torch.cat([self.class_embedding.expand(x.shape[0], -1, -1), x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + self._get_abs_pos((h, w))

        x = self.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.full_attn_ids[-1]) or (self.return_interm_layers and i in self.full_attn_ids):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    feats = feats.reshape(feats.shape[0], h, w, feats.shape[-1]).permute(0, 3, 1, 2)

                outputs.append(feats)

        return outputs
