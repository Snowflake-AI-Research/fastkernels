"""Memory encoder for SAM3 tracker.

Contains:
- SimpleMaskDownSampler: Progressive mask downsampling with strided convs
- CXBlock: ConvNeXt block with depthwise conv + pointwise MLPs
- SimpleFuser: Stacks multiple CXBlocks
- Sam3MemoryEncoder (SimpleMaskEncoder): Fuses pixel features + mask features,
  projects to 64-D memory tokens, adds positional encoding

Reference: sam3/model/memory.py
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam3_prompt_encoder import LayerNorm2d


class SimpleMaskDownSampler(nn.Module):
    """Progressively downsample a mask by total_stride using strided convs.

    Reference: sam3/model/memory.py SimpleMaskDownSampler
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: type = nn.GELU,
        interpol_size: list | None = None,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride ** num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_out_chans * (stride ** 2)
            self.encoder.append(
                nn.Conv2d(mask_in_chans, mask_out_chans,
                          kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.interpol_size = list(interpol_size) if interpol_size is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(), size=self.interpol_size,
                align_corners=False, mode="bilinear", antialias=True,
            )
        return self.encoder(x)


class CXBlock(nn.Module):
    """ConvNeXt block: DwConv -> LN -> Linear -> GELU -> Linear + residual.

    Reference: sam3/model/memory.py CXBlock
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding,
            groups=dim if use_dwconv else 1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class SimpleFuser(nn.Module):
    """Stack of CXBlocks for fusing mask + pixel features.

    Reference: sam3/model/memory.py SimpleFuser
    """

    def __init__(self, layer: nn.Module, num_layers: int):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([
            type(layer)(
                dim=layer.dwconv.in_channels,
                kernel_size=layer.dwconv.kernel_size[0],
                padding=layer.dwconv.padding[0],
                layer_scale_init_value=(
                    float(layer.gamma[0].item()) if layer.gamma is not None else 0.0
                ),
                use_dwconv=(layer.dwconv.groups == layer.dwconv.in_channels),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D position encoding for memory features.

    Reference: sam3/model/position_encoding.py PositionEmbeddingSine
    Note: num_pos_feats is halved internally (the reference does // 2).
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
        precompute_resolution: int | None = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device = x.device
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Sam3MemoryEncoder(nn.Module):
    """Memory encoder: fuse pixel features + mask features, project to 64-D.

    Reference: sam3/model/memory.py SimpleMaskEncoder
    """

    def __init__(
        self,
        out_dim: int,
        mask_downsampler: nn.Module,
        fuser: nn.Module,
        position_encoding: nn.Module,
        in_dim: int = 256,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict:
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        pix_feat = pix_feat.to(masks.device)
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}
