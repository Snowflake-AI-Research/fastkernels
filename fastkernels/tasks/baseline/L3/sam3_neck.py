"""FPN neck for SAM3.

Takes ViT backbone features and produces multi-scale feature maps via the
SimpleFPN architecture. Optionally produces a dual set of features for the
SAM2-style interactive path.

Reference: sam3/model/necks.py Sam3DualViTDetNeck
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..L1.sam3_position_encoding import Sam3PositionEncoding
from ..L2.sam3_fpn_conv import Sam3FPNConvStage
from .sam3_vit import Sam3ViT


class Sam3Neck(nn.Module):
    """SimpleFPN neck for SAM3.

    Runs the ViT trunk, takes the last feature map, and applies multi-scale
    convolution stages to produce FPN outputs at different resolutions.

    Args:
        trunk: ViT backbone.
        d_model: Output channel dimension.
        scale_factors: Spatial scale factors for each FPN level.
        add_sam2_neck: If True, create a duplicate neck with separate weights.
    """

    def __init__(
        self,
        trunk: Sam3ViT,
        d_model: int = 256,
        scale_factors: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = Sam3PositionEncoding(d_model)

        in_dim = trunk.channel_list[-1]
        self.convs = nn.ModuleList([
            Sam3FPNConvStage(in_dim, d_model, sf) for sf in scale_factors
        ])

        self.sam2_convs: nn.ModuleList | None = None
        if add_sam2_neck:
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        """Run trunk + FPN neck.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            (sam3_out, sam3_pos, sam2_out, sam2_pos):
                sam3_out: List of (B, d_model, H_i, W_i) feature maps.
                sam3_pos: Corresponding position encodings.
                sam2_out/sam2_pos: Duplicate outputs from second neck, or None.
        """
        xs = self.trunk(x)
        feat = xs[-1]

        sam3_out, sam3_pos = [], []
        sam2_out: Optional[List[torch.Tensor]] = None
        sam2_pos: Optional[List[torch.Tensor]] = None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        for i, conv in enumerate(self.convs):
            out = conv(feat)
            sam3_out.append(out)
            sam3_pos.append(self.position_encoding(out).to(out.dtype))

            if self.sam2_convs is not None:
                s2_out = self.sam2_convs[i](feat)
                sam2_out.append(s2_out)
                sam2_pos.append(self.position_encoding(s2_out).to(s2_out.dtype))

        return sam3_out, sam3_pos, sam2_out, sam2_pos
