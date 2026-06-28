"""VAE output post-processor for images and video.

Handles tensor -> PIL/numpy conversion after VAE decoding: denormalize
from [-1, 1] to [0, 1], permute to HWC, convert to numpy/PIL.

``postprocess``: 4D ``(B, C, H, W)`` image tensors.
``postprocess_video``: 5D ``(B, C, T, H, W)`` video tensors.

Mirrors diffusers' ``VaeImageProcessor.postprocess`` and
``VideoProcessor.postprocess_video``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class VideoProcessor(nn.Module):
    """Post-processor for VAE-decoded image and video tensors."""

    def __init__(self, vae_scale_factor: int = 8, do_normalize: bool = True):
        super().__init__()
        self.vae_scale_factor = vae_scale_factor
        self.do_normalize = do_normalize

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
    ) -> list[Image.Image] | np.ndarray | torch.Tensor:
        """Post-process a 4D ``(B, C, H, W)`` image tensor."""
        if output_type == "latent" or output_type == "pt":
            return image

        if self.do_normalize:
            image = (image * 0.5 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "np":
            return image

        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in images]

    def postprocess_video(
        self,
        video: torch.Tensor,
        output_type: str = "np",
    ) -> np.ndarray | torch.Tensor | list:
        """Post-process a 5D ``(B, C, T, H, W)`` video tensor."""
        batch_size = video.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = self.postprocess(batch_vid, output_type)
            outputs.append(batch_output)

        if output_type == "np":
            return np.stack(outputs)
        elif output_type == "pt":
            return torch.stack(outputs)
        elif output_type == "pil":
            return outputs
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
