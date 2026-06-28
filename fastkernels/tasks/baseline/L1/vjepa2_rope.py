"""V-JEPA 2 rotary position helper.

Applies the factorized spatio-temporal rotary embedding used by V-JEPA 2:
separate frame / height / width rotations over disjoint head-dim slices.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VJEPA2RotaryEmbedding(nn.Module):
    """Apply V-JEPA 2's frame/height/width rotary embedding."""

    def __init__(
        self,
        crop_size: int,
        patch_size: int,
        frames_per_clip: int,
        tubelet_size: int,
        head_dim: int,
    ):
        super().__init__()
        self.grid_size = crop_size // patch_size
        self.grid_depth = max(1, frames_per_clip // tubelet_size)
        axis_dim = 2 * ((head_dim // 3) // 2)
        self.d_dim = axis_dim
        self.h_dim = axis_dim
        self.w_dim = axis_dim
        self.head_dim = head_dim

    def _get_frame_pos(self, ids: torch.Tensor) -> torch.Tensor:
        tokens_per_frame = self.grid_size * self.grid_size
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: torch.Tensor) -> torch.Tensor:
        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        return ids // self.grid_size

    def get_position_ids(
        self,
        seq_len: int,
        device: torch.device,
        num_heads: int,
        position_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_mask is None:
            ids = torch.arange(seq_len, device=device)
        else:
            ids = position_mask.to(device).unsqueeze(1).expand(-1, num_heads, -1)

        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        height_ids = self._get_height_pos(ids)
        width_ids = (ids - tokens_per_frame * frame_ids) - self.grid_size * height_ids
        return frame_ids, height_ids, width_ids

    def _apply_axis_rotary(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        half_dim = dim // 2
        if half_dim == 0:
            return x

        omega = torch.arange(half_dim, device=x.device, dtype=x.dtype)
        omega /= half_dim
        omega = 1.0 / (10000 ** omega)

        freq = pos.to(x.device).unsqueeze(-1) * omega
        emb_sin = freq.sin()
        emb_cos = freq.cos()
        repeat_shape = [1] * emb_sin.dim()
        repeat_shape[-1] = 2
        emb_sin = emb_sin.repeat(*repeat_shape)
        emb_cos = emb_cos.repeat(*repeat_shape)

        rotated = x.unflatten(-1, (-1, 2))
        x1, x2 = rotated.unbind(dim=-1)
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * emb_cos + rotated * emb_sin

    def forward(
        self,
        qk: torch.Tensor,
        position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame_ids, height_ids, width_ids = self.get_position_ids(
            qk.shape[-2], qk.device, qk.shape[1], position_mask=position_mask,
        )

        pieces: list[torch.Tensor] = []
        start = 0
        for size, pos in (
            (self.d_dim, frame_ids),
            (self.h_dim, height_ids),
            (self.w_dim, width_ids),
        ):
            if size <= 0:
                continue
            pieces.append(self._apply_axis_rotary(qk[..., start:start + size], pos))
            start += size

        if start < self.head_dim:
            pieces.append(qk[..., start:])

        return torch.cat(pieces, dim=-1)
