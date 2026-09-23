"""SAM3 random Fourier position encoding with HF's pre-matmul dtype boundary.

The coordinate projection, sin/cos evaluation, dependencies and storage are the
same as PositionEmbeddingRandom._pe_encoding. HF casts after normalization,
which matters when point padding/grid construction produces FP32 coordinates.
"""

import math
import torch
from fastkernels.tasks.baseline.L2.sam3_prompt_encoder import PositionEmbeddingRandom


class SamPositionDtype(PositionEmbeddingRandom):
    def _pe_encoding(self, coords):
        coords = (2 * coords - 1).to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat((torch.sin(coords), torch.cos(coords)), dim=-1)
