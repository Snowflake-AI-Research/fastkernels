"""SwinV2 transformer block with windowed attention (L3).

Post-norm residual block with window partition, optional cyclic shift
for SW-MSA, and pre-computed attention masks for shifted windows.
Operates on NHWC tensors throughout.

Reference: timm/models/swin_transformer_v2.py SwinTransformerV2Block
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L2.swinv2_window_attention import SwinV2WindowAttention
from ..L2.vit_encoder_mlp import VitEncoderMlp


_int_or_tuple_2_t = Union[int, Tuple[int, int]]


def _to_2tuple(x: _int_or_tuple_2_t) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def window_partition(
    x: torch.Tensor,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    """Partition (B, H, W, C) into (num_windows*B, Wh, Ww, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)


def window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    img_size: Tuple[int, int],
) -> torch.Tensor:
    """Merge (num_windows*B, Wh, Ww, C) back to (B, H, W, C)."""
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)


class SwinV2Block(nn.Module):
    """SwinV2 transformer block.

    Uses windowed self-attention (W-MSA) or shifted-window self-attention
    (SW-MSA) depending on shift_size. Post-norm residual connections:
    ``x = x + norm1(attn(x)); x = x + norm2(mlp(x))``.

    Args:
        dim: Number of input channels.
        input_resolution: Input spatial resolution (H, W).
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for SW-MSA. 0 means W-MSA.
        mlp_ratio: MLP hidden-dim expansion ratio.
        qkv_bias: If True, add learnable bias to QKV.
        proj_drop: Dropout rate after projections.
        attn_drop: Attention dropout rate.
        pretrained_window_size: Pretrained window size for CPB normalization.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: _int_or_tuple_2_t = 7,
        shift_size: _int_or_tuple_2_t = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        pretrained_window_size: _int_or_tuple_2_t = 0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.window_size, self.shift_size = self._calc_window_shift(
            _to_2tuple(window_size), _to_2tuple(shift_size),
        )
        self.window_area = self.window_size[0] * self.window_size[1]

        self.attn = SwinV2WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=_to_2tuple(pretrained_window_size),
        )
        self.norm1 = LayerNorm(dim)
        self.mlp = VitEncoderMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_approximate="none",
            bias=True,
            drop=proj_drop,
        )
        self.norm2 = LayerNorm(dim)

        H, W = self.input_resolution
        self.register_buffer(
            "attn_mask",
            self._make_attn_mask(H, W),
            persistent=False,
        )

    def _calc_window_shift(
        self,
        target_window_size: Tuple[int, int],
        target_shift_size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        window_size = tuple(
            r if r <= w else w
            for r, w in zip(self.input_resolution, target_window_size)
        )
        shift_size = tuple(
            0 if r <= w else s
            for r, w, s in zip(self.input_resolution, window_size, target_shift_size)
        )
        return window_size, shift_size

    def _make_attn_mask(
        self,
        H: int,
        W: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        if not any(self.shift_size):
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
        cnt = 0
        for h in (
            (0, -self.window_size[0]),
            (-self.window_size[0], -self.shift_size[0]),
            (-self.shift_size[0], None),
        ):
            for w in (
                (0, -self.window_size[1]),
                (-self.window_size[1], -self.shift_size[1]),
                (-self.shift_size[1], None),
            ):
                img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_area)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = F.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_area, C)

        if (H, W) == self.input_resolution:
            attn_mask = self.attn_mask
        else:
            attn_mask = self._make_attn_mask(Hp, Wp, device=x.device, dtype=x.dtype)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, (Hp, Wp))
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x + self.norm1(self._attn(x))
        x = x.reshape(B, -1, C)
        x = x + self.norm2(self.mlp(x))
        x = x.reshape(B, H, W, C)
        return x
