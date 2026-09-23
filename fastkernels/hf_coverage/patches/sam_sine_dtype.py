"""SAM3 memory position encoding with HF's input-dtype metadata arithmetic.

PositionEmbeddingSine builds this same fixed grid in FP32. HF constructs its
grid, frequencies and sin/cos in the feature dtype. No image values participate;
the patch changes rounding, not the axes, projection or memory dependencies.
"""

import torch
from fastkernels.tasks.baseline.L2.sam3_memory_encoder import PositionEmbeddingSine


class SamSineDtype(PositionEmbeddingSine):
    def forward(self, x):
        batch, _, height, width = x.shape
        y = torch.arange(1, height + 1, dtype=x.dtype, device=x.device)[None, :, None].expand(batch, -1, width)
        xx = torch.arange(1, width + 1, dtype=x.dtype, device=x.device)[None, None, :].expand(batch, height, -1)
        if self.normalize:
            y = y / (y[:, -1:, :] + 1e-6) * self.scale
            xx = xx / (xx[:, :, -1:] + 1e-6) * self.scale
        frequencies = torch.arange(self.num_pos_feats, device=x.device).to(x.dtype)
        frequencies = self.temperature ** (2 * torch.div(frequencies, 2, rounding_mode="floor") / self.num_pos_feats)
        px, py = xx[..., None] / frequencies, y[..., None] / frequencies
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1).flatten(3)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((py, px), dim=-1).permute(0, 3, 1, 2)
