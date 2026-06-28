"""Compute M-RoPE 3D position indices for text+vision token sequences.

Builds a (3, seq_len) position tensor where each row encodes temporal,
height, and width positions respectively. Text tokens get identical
positions across all three dimensions; vision tokens get 3D grid indices.

For Qwen3-VL videos, each frame is a separate block of video_token_id
tokens interleaved with timestamp/vision_start/vision_end tokens, so
video_offsets contains one entry per frame (not per video).

For Qwen2-VL videos, all frames are contiguous so video_offsets has one
entry per video, and the full (t, h, w) grid is used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MRopeInputPositions(nn.Module):
    """Stateless module that computes M-RoPE 3D positions from token layout."""

    def forward(
        self,
        input_tokens: list[int],
        spatial_merge_size: int,
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
    ) -> tuple[torch.Tensor, int]:
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0

        media_items: list[tuple[int, int, int, int]] = []
        if image_grid_thw and image_offsets:
            for i, (t, h, w) in enumerate(image_grid_thw):
                merged_h = h // spatial_merge_size
                merged_w = w // spatial_merge_size
                media_items.append((image_offsets[i], t, merged_h, merged_w))

        if video_grid_thw and video_offsets:
            total_frames = sum(thw[0] for thw in video_grid_thw)
            per_frame = len(video_offsets) == total_frames and total_frames > len(video_grid_thw)
            if per_frame:
                frame_offset_idx = 0
                for t, h, w in video_grid_thw:
                    merged_h = h // spatial_merge_size
                    merged_w = w // spatial_merge_size
                    for _ in range(t):
                        media_items.append(
                            (video_offsets[frame_offset_idx], 1, merged_h, merged_w)
                        )
                        frame_offset_idx += 1
            else:
                for i, (t, h, w) in enumerate(video_grid_thw):
                    merged_h = h // spatial_merge_size
                    merged_w = w // spatial_merge_size
                    media_items.append((video_offsets[i], t, merged_h, merged_w))

        media_items.sort(key=lambda x: x[0])

        for offset, grid_t, grid_h, grid_w in media_items:
            text_len = offset - st
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )
            grid_indices = np.indices((grid_t, grid_h, grid_w))
            llm_pos_ids_list.append(grid_indices.reshape(3, -1) + text_len + st_idx)
            st = offset + grid_t * grid_h * grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        if not llm_pos_ids_list:
            positions = np.broadcast_to(
                np.arange(len(input_tokens)), (3, len(input_tokens))
            )
            return torch.from_numpy(positions), 0

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = int(llm_positions.max() + 1 - len(input_tokens))
        return torch.from_numpy(llm_positions), mrope_position_delta
