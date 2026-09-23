"""SAM3 decoder/scoring adaptations that retain native activation precision.

The parents are Sam3Decoder and Sam3DotProductScoring in L4/sam3.py. Native HF
builds the box-dependent sine/RPB grids and masked-text reductions in the input
dtype; the parents promote these paths to FP32. The formulas, reductions,
intermediates, and dependencies below are unchanged. Boxes and pooled text are
learned activations, so these changes are explicit numerical patches.
"""

import math
import torch

from fastkernels.tasks.baseline.L4.sam3 import Sam3Decoder, Sam3DotProductScoring


class Sam3DecoderDtype(Sam3Decoder):
    def _gen_sineembed(self, pos, num_feats=None):
        half = (num_feats or self.d_model) // 2
        dim_t = torch.arange(half, dtype=torch.int64, device=pos.device).to(pos.dtype)
        dim_t = 10000.0 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / half)
        encoded = []
        for index in (1, 0, 2, 3):
            values = (pos[..., index] * (2 * math.pi))[..., None] / dim_t
            encoded.append(torch.stack((values[..., 0::2].sin(), values[..., 1::2].cos()), dim=-1).flatten(-2))
        return torch.cat(encoded, dim=-1)

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        height, width = feat_size
        batch, queries, _ = reference_boxes.shape
        boxes = self._box_cxcywh_to_xyxy(reference_boxes)
        rows = torch.arange(height, device=boxes.device, dtype=boxes.dtype) / height
        cols = torch.arange(width, device=boxes.device, dtype=boxes.dtype) / width
        dy = (rows.view(1, -1, 1) - boxes.reshape(-1, 1, 4)[:, :, 1:4:2]).view(batch, queries, -1, 2)
        dx = (cols.view(1, -1, 1) - boxes.reshape(-1, 1, 4)[:, :, 0:3:2]).view(batch, queries, -1, 2)
        dx, dy = dx * 8, dy * 8
        dx = torch.sign(dx) * torch.log2(torch.abs(dx) + 1.0) / math.log2(8)
        dy = torch.sign(dy) * torch.log2(torch.abs(dy) + 1.0) / math.log2(8)
        dx, dy = self.boxRPB_embed_x(dx), self.boxRPB_embed_y(dy)
        return (dy.unsqueeze(3) + dx.unsqueeze(2)).flatten(2, 3).permute(0, 3, 1, 2).contiguous()


class Sam3ScoringDtype(Sam3DotProductScoring):
    def mean_pool_text(self, prompt, prompt_mask):
        valid = (~prompt_mask).to(prompt.dtype).permute(1, 0)[..., None]
        count = torch.clamp(torch.sum(valid, dim=0), min=1.0)
        return (prompt * valid).sum(dim=0) / count
