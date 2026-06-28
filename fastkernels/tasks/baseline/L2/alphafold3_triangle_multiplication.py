"""Triangle multiplicative update for AlphaFold3 (L2).

Implements AF3 Algorithms 12 (outgoing) and 13 (incoming).
Core operation: gated einsum("...ij,...jk->...ik") on pair representations
with permutation controlling outgoing vs incoming orientation.

Reference: openfold3/core/model/layers/triangular_multiplicative_update.py
           TriangleMultiplicativeUpdate
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class TriangleMultiplicativeUpdate(nn.Module):
    """AF3 Algorithms 12/13: Triangle multiplicative update.

    Args:
        c_z: Input channel dimension
        c_hidden: Hidden channel dimension
        _outgoing: If True, outgoing (Alg 12); else incoming (Alg 13)
    """

    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = Linear(c_z, c_hidden, bias=False)
        self.linear_b_p = Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = Linear(c_z, c_hidden, bias=False)

        self.linear_g = Linear(c_z, c_z, bias=False)
        self.linear_z = Linear(c_hidden, c_z, bias=False)

        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def _combine_projections(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> torch.Tensor:
        if self._outgoing:
            a = _permute_final_dims(a, (2, 0, 1))
            b = _permute_final_dims(b, (2, 1, 0))
        else:
            a = _permute_final_dims(a, (2, 1, 0))
            b = _permute_final_dims(b, (2, 0, 1))

        p = torch.einsum("...ij,...jk->...ik", a, b)
        return _permute_final_dims(p, (1, 2, 0))

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            z:    [*, N_res, N_res, C_z] input tensor
            mask: [*, N_res, N_res] input mask

        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z_ln = self.layer_norm_in(z)

        a = mask * torch.sigmoid(self.linear_a_g(z_ln)) * self.linear_a_p(z_ln)
        b = mask * torch.sigmoid(self.linear_b_g(z_ln)) * self.linear_b_p(z_ln)

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        x = x * torch.sigmoid(self.linear_g(z_ln))

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 12."""

    def __init__(self, c_z: int, c_hidden: int):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 13."""

    def __init__(self, c_z: int, c_hidden: int):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=False)
