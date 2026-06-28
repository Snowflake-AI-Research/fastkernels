"""RTDetrV2 multi-scale deformable attention primitive.

Contains the low-level sampling kernel only.  The composite module
(RTDetrV2MultiscaleDeformableAttention) lives in L2/rtdetrv2_deformable_attention.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttentionV2(nn.Module):
    """RTDetrV2 multi-scale deformable attention sampling primitive.

    Stateless ``nn.Module`` wrapper around the bilinear/discrete sampling
    + weighted aggregation pipeline.  Used by
    ``L2/rtdetrv2_deformable_attention.py``.
    """

    def forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        num_points_list: list[int],
        method: str = "default",
    ) -> torch.Tensor:
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, _, _, _ = sampling_locations.shape
        value_list = (
            value.permute(0, 2, 3, 1)
            .flatten(0, 1)
            .split([height * width for height, width in value_spatial_shapes], dim=-1)
        )
        if method == "default":
            sampling_grids = 2 * sampling_locations - 1
        elif method == "discrete":
            sampling_grids = sampling_locations
        else:
            raise ValueError(f"Unsupported decoder method: {method}")
        sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_grids = sampling_grids.split(num_points_list, dim=-2)

        sampling_value_list = []
        for level_id, (height, width) in enumerate(value_spatial_shapes):
            value_l = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
            sampling_grid_l = sampling_grids[level_id]
            if method == "default":
                sampling_value_l = F.grid_sample(
                    value_l,
                    sampling_grid_l,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            else:
                sampling_coord = (sampling_grid_l * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                    torch.int64
                )
                sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
                sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)
                sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
                sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
                sampling_idx = torch.arange(sampling_coord.shape[0], device=value.device).unsqueeze(-1)
                sampling_idx = sampling_idx.repeat(1, sampling_coord.shape[1])
                sampling_value_l = value_l[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
                sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                    batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
                )
            sampling_value_list.append(sampling_value_l)

        attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
            batch_size * num_heads, 1, num_queries, sum(num_points_list)
        )
        output = (
            (torch.concat(sampling_value_list, dim=-1) * attention_weights)
            .sum(-1)
            .view(batch_size, num_heads * hidden_dim, num_queries)
        )
        return output.transpose(1, 2).contiguous()
