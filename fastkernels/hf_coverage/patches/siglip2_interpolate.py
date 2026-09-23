"""Expose the library's antialiased bilinear resize for learned position tables.

The call exists in L2/sam3_memory_encoder.SimpleMaskDownSampler.forward.
This adapter exposes size as an argument and retains the caller's GPU dtype;
the parent always passes FP32 masks. CPU upcasting matches HF's table path.
The interpolation backend, filter, coordinate convention and spatial reduction
are unchanged. This is an implemented interface/dtype adaptation, not unchanged
reuse of the complete mask downsampler.
"""

import torch.nn as nn
import torch.nn.functional as F


class PositionTableResize(nn.Module):
    def forward(self, table, size):
        dtype = table.dtype
        source = table.float() if table.device.type == "cpu" else table
        return F.interpolate(source, size=size, mode="bilinear", align_corners=False,
                             antialias=True).to(dtype)
