"""DINOv3 7B/16 Eva vision encoder (L4).

Full DINOv3 model matching timm's Eva architecture for the 7B/16
variant. Conv2d patch embed, CLS token, register tokens, DINOv3
2D RoPE, SwiGLU MLP, layer-scale, CLS-token pooling.

Reference: timm/models/eva.py Eva, vit_7b_patch16_dinov3
           timm model name: vit_7b_patch16_dinov3.lvd1689m
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L1.dinov3_rope import DINOv3RoPE
from ..L3.eva_block import EvaBlock


DINOV3_7B_CONFIG = dict(
    patch_size=16,
    embed_dim=4096,
    depth=40,
    num_heads=32,
    mlp_ratio=2,
    qkv_bias=False,
    init_values=1e-5,
    rope_temperature=100.0,
    num_reg_tokens=4,
    swiglu_align_to=64,
    default_resolution=256,
)

HF_TO_TIMM = {
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "vit_7b_patch16_dinov3.lvd1689m",
}


class DINOv3Model(nn.Module):
    """DINOv3 7B/16 Eva vision encoder.

    Args:
        patch_size: Patch size.
        in_chans: Input channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        mlp_ratio: SwiGLU MLP expansion ratio.
        qkv_bias: QKV projection bias.
        init_values: Layer-scale initial value.
        rope_temperature: RoPE temperature.
        num_reg_tokens: Number of register tokens.
        swiglu_align_to: Alignment for SwiGLU hidden dim.
        num_classes: Output classes (0 = feature extractor).
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 4096,
        depth: int = 40,
        num_heads: int = 32,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = False,
        init_values: float = 1e-5,
        rope_temperature: float = 100.0,
        num_reg_tokens: int = 4,
        swiglu_align_to: int = 64,
        num_classes: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 + num_reg_tokens  # CLS + register
        self.num_reg_tokens = num_reg_tokens

        # Patch embedding (conv2d, dynamic_img_size -> NHWC output)
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # CLS and register tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim))

        # No absolute position embedding (DINOv3 uses only RoPE)
        head_dim = embed_dim // num_heads
        self.rope = DINOv3RoPE(
            dim=head_dim,
            temperature=rope_temperature,
            normalize_coords="separate",
            grid_offset=0.0,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EvaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_fused=True,
                num_prefix_tokens=self.num_prefix_tokens,
                rotate_half=True,
                init_values=init_values,
                swiglu_align_to=swiglu_align_to,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = LayerNorm(embed_dim)

        # Head
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Pool config: avg pooling matches timm default for DINOv3
        self.global_pool = "avg"

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Patch embed -> (B, C, Hg, Wg) -> (B, Hg, Wg, C)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)  # NHWC
        gh, gw = x.shape[1], x.shape[2]
        x = x.reshape(B, gh * gw, self.embed_dim)

        # Get RoPE embeddings
        rope_embed = self.rope.get_embed(shape=[gh, gw])

        # Prepend CLS + register tokens
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.reg_token.expand(B, -1, -1)
        x = torch.cat([cls, reg, x], dim=1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, rope=rope_embed)

        x = self.norm(x)
        return x

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "token":
            return x[:, 0]
        elif self.global_pool == "avg":
            return x[:, self.num_prefix_tokens:].mean(dim=1)
        return x[:, 0]

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool(x)
        x = self.fc_norm(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @staticmethod
    def from_timm(model_name: str = "vit_7b_patch16_dinov3.lvd1689m") -> DINOv3Model:
        """Load weights from a timm pretrained checkpoint."""
        import timm

        timm_model = timm.create_model(model_name, pretrained=True)
        timm_sd = timm_model.state_dict()

        cfg = DINOV3_7B_CONFIG
        kb_model = DINOv3Model(
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            init_values=cfg["init_values"],
            rope_temperature=cfg["rope_temperature"],
            num_reg_tokens=cfg["num_reg_tokens"],
            swiglu_align_to=cfg["swiglu_align_to"],
            num_classes=0,
        )

        new_sd = _remap_timm_to_kb(timm_sd)

        missing, unexpected = kb_model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  DINOv3 load: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  DINOv3 load: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

        del timm_model
        return kb_model


def _remap_timm_to_kb(timm_sd: dict) -> dict:
    """Remap timm Eva state dict keys to kb-nano DINOv3Model keys.

    timm keys:
        patch_embed.proj.weight/bias -> patch_embed.weight/bias
        cls_token                    -> cls_token
        reg_token                    -> reg_token
        blocks.N.*                   -> blocks.N.*
        norm.weight/bias             -> norm.weight/bias
    """
    out = {}
    for k, v in timm_sd.items():
        # Skip RoPE buffers (non-persistent, recomputed)
        if "rope" in k:
            continue
        # Skip norm_pre (Identity in DINOv3 with bias=True patch_embed)
        if k.startswith("norm_pre."):
            continue

        nk = k
        # patch_embed.proj -> patch_embed
        if k.startswith("patch_embed.proj."):
            nk = k.replace("patch_embed.proj.", "patch_embed.")

        out[nk] = v
    return out
