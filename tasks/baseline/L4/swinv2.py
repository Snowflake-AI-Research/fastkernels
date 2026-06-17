"""SwinV2 Large vision encoder (L4).

Full hierarchical Swin Transformer V2 matching timm's SwinTransformerV2
for the Large/window12/192 variant. Conv2d patch embed, 4 stages with
patch merging, windowed attention with cosine similarity and CPB,
global avg pooling.

Reference: timm/models/swin_transformer_v2.py SwinTransformerV2
           timm model name: swinv2_large_window12_192.ms_in22k
"""

from __future__ import annotations

import re
from typing import Tuple

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L3.swinv2_stage import SwinV2Stage


SWINV2_LARGE_WINDOW12_192_CONFIG = dict(
    patch_size=4,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    default_resolution=192,
    num_classes=21841,
)

HF_TO_TIMM = {
    "timm/swinv2_large_window12_192.ms_in22k": "swinv2_large_window12_192.ms_in22k",
}


class SwinV2Model(nn.Module):
    """SwinV2 Large hierarchical vision encoder.

    Args:
        patch_size: Patch size for initial embedding.
        in_chans: Input image channels.
        embed_dim: Base embedding dimension (doubles each stage).
        depths: Number of blocks per stage.
        num_heads: Attention heads per stage.
        window_size: Window size for windowed attention.
        mlp_ratio: MLP hidden-dim expansion ratio.
        qkv_bias: If True, add learnable bias to QKV.
        num_classes: Output classes (0 = feature extractor only).
        pretrained_window_sizes: Per-stage pretrained window sizes for CPB.
    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depths: Tuple[int, ...] = (2, 2, 18, 2),
        num_heads: Tuple[int, ...] = (6, 12, 24, 48),
        window_size: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_classes: int = 0,
        default_resolution: int = 192,
        pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        embed_dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        grid_size = default_resolution // patch_size

        self.patch_embed = nn.Sequential()
        self.patch_embed.proj = nn.Conv2d(
            in_chans, embed_dims[0],
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.patch_embed.norm = LayerNorm(embed_dims[0])

        layers = []
        in_dim = embed_dims[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dims[i]
            layers.append(SwinV2Stage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(grid_size // scale, grid_size // scale),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=i > 0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                pretrained_window_size=pretrained_window_sizes[i],
            ))
            in_dim = out_dim
            if i > 0:
                scale *= 2

        self.layers = nn.Sequential(*layers)
        self.norm = LayerNorm(self.num_features)

        self.head = nn.Identity()
        if num_classes > 0:
            self.head = Linear(self.num_features, num_classes)

        self.global_pool = "avg"
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features: (B, C, H, W) -> (B, H', W', C')."""
        x = self.patch_embed.proj(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.patch_embed.norm(x)

        x = self.layers(x)
        x = self.norm(x)
        return x

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Global avg pool over spatial dims: (B, H, W, C) -> (B, C)."""
        return x.mean(dim=(1, 2))

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @staticmethod
    def from_timm(model_name: str = "swinv2_large_window12_192.ms_in22k") -> SwinV2Model:
        """Load weights from a timm pretrained checkpoint."""
        import timm

        timm_model = timm.create_model(model_name, pretrained=True)
        timm_sd = timm_model.state_dict()

        cfg = SWINV2_LARGE_WINDOW12_192_CONFIG
        num_classes = getattr(timm_model, "num_classes", cfg["num_classes"])
        grid_size = timm_model.patch_embed.grid_size[0] if hasattr(timm_model, "patch_embed") else cfg["default_resolution"] // cfg["patch_size"]
        default_resolution = grid_size * cfg["patch_size"]

        pretrained_window_sizes = (0, 0, 0, 0)
        if hasattr(timm_model, "layers") and len(timm_model.layers) > 0:
            blk0 = timm_model.layers[0].blocks[0]
            pws = blk0.attn.pretrained_window_size
            if pws[0] > 0:
                pretrained_window_sizes = tuple(
                    timm_model.layers[i].blocks[0].attn.pretrained_window_size[0]
                    for i in range(len(timm_model.layers))
                )

        kb_model = SwinV2Model(
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            num_classes=num_classes,
            default_resolution=default_resolution,
            pretrained_window_sizes=pretrained_window_sizes,
        )

        new_sd = _remap_timm_to_kb(timm_sd)

        missing, unexpected = kb_model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  SwinV2 load: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  SwinV2 load: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

        del timm_model
        return kb_model


def _remap_timm_to_kb(timm_sd: dict) -> dict:
    """Remap timm SwinTransformerV2 state dict keys to kb-nano SwinV2Model.

    timm native keys (after checkpoint_filter_fn):
        patch_embed.proj.weight/bias       -> patch_embed.proj.weight/bias
        patch_embed.norm.weight/bias       -> patch_embed.norm.weight/bias
        layers.N.downsample.reduction.*    -> layers.N.downsample.reduction.*
        layers.N.downsample.norm.*         -> layers.N.downsample.norm.*
        layers.N.blocks.B.attn.qkv.weight  -> layers.N.blocks.B.attn.qkv.weight
        layers.N.blocks.B.attn.q_bias      -> layers.N.blocks.B.attn.q_bias
        layers.N.blocks.B.attn.v_bias      -> layers.N.blocks.B.attn.v_bias
        layers.N.blocks.B.attn.proj.*      -> layers.N.blocks.B.attn.proj.*
        layers.N.blocks.B.attn.logit_scale -> layers.N.blocks.B.attn.logit_scale
        layers.N.blocks.B.attn.cpb_mlp.*   -> layers.N.blocks.B.attn.cpb_mlp.*
        layers.N.blocks.B.norm1/norm2.*    -> layers.N.blocks.B.norm1/norm2.*
        layers.N.blocks.B.mlp.fc1/fc2.*   -> layers.N.blocks.B.mlp.fc1/fc2.*
        norm.weight/bias                   -> norm.weight/bias
        head.fc.weight/bias                -> head.weight/bias
    """
    out = {}
    for k, v in timm_sd.items():
        # Skip non-persistent buffers
        if any(n in k for n in ("relative_position_index", "relative_coords_table", "attn_mask", "k_bias")):
            continue

        nk = k

        # head.fc -> head (our head is a plain Linear, not ClassifierHead)
        if k.startswith("head.fc."):
            nk = k.replace("head.fc.", "head.")
        elif k.startswith("head.") and not k.startswith("head.fc."):
            # head.weight -> head.weight (already matches if no ClassifierHead)
            nk = k

        out[nk] = v
    return out
