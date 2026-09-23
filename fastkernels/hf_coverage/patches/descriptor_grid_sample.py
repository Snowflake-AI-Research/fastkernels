"""RTDetrV2 bilinear sampler specialized to one level/point and unit weight.

The parent's sampling backend is unchanged. SuperPoint supplies an already
normalized grid and uses align_corners=True; exposing those choices avoids an
extra BF16 coordinate roundtrip. The unit-weight aggregation is an identity.
"""

from torch import nn
from torch.nn import functional as F


class DescriptorGridSample(nn.Module):
    def forward(self, descriptors, normalized_grid):
        return F.grid_sample(descriptors, normalized_grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)
