"""V-JEPA 2 patch embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv3d import Conv3d


class VJEPA2PatchEmbeddings3D(nn.Module):
    """Tubelet patch embedding."""

    def __init__(self, config, hidden_size: int = 1024):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.hidden_size = hidden_size
        self.proj = Conv3d(
            in_channels=config.in_chans,
            out_channels=hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
            bias=True,
        )

    @staticmethod
    def num_patches(config) -> int:
        return (
            (config.frames_per_clip // config.tubelet_size)
            * (config.crop_size // config.patch_size)
            * (config.crop_size // config.patch_size)
        )

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.proj(pixel_values_videos).flatten(2).transpose(1, 2)


class VJEPA2Embeddings(nn.Module):
    """Video-to-patch embedding wrapper."""

    def __init__(self, config, hidden_size: int = 1024):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size)
        self.num_patches = self.patch_embeddings.num_patches(config)
        self.patch_size = config.patch_size

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        num_frames = pixel_values_videos.shape[1]
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)
        if num_frames < self.config.tubelet_size:
            pixel_values_videos = pixel_values_videos.repeat(1, 1, self.config.tubelet_size, 1, 1)
        target_dtype = self.patch_embeddings.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
        return self.patch_embeddings(pixel_values_videos)
