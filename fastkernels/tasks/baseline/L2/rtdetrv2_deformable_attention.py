"""RTDetrV2 multi-scale deformable attention composite module."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.softmax import Softmax
from ..L1.rtdetrv2_deformable_attention import MultiScaleDeformableAttentionV2


class RTDetrV2MultiscaleDeformableAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_heads = config.decoder_attention_heads
        n_points = config.decoder_n_points

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "RTDetrV2MultiscaleDeformableAttention is most efficient when dim per head is a power of 2."
            )

        self.d_model = config.d_model
        self.n_levels = config.decoder_n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = Linear(config.d_model, config.d_model)
        self.output_proj = Linear(config.d_model, config.d_model)
        self._softmax = Softmax(dim=-1)
        self.msdeform_attn = MultiScaleDeformableAttentionV2()

        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

        self.n_points_list = [self.n_points for _ in range(self.n_levels)]
        n_points_scale = [1 / n for n in self.n_points_list for _ in range(n)]
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        del level_start_index
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        _, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError("Spatial shapes must match encoder sequence length.")

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], 0.0)
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = self._softmax(attention_weights)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.msdeform_attn(
            value,
            spatial_shapes_list,
            sampling_locations,
            attention_weights,
            self.n_points_list,
            self.method,
        )
        output = self.output_proj(output)
        return output, attention_weights
