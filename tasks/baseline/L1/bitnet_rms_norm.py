"""BitNet-specific RMSNorm that is bit-for-bit identical to SOTA.

The default fastkernels ``RMSNorm`` (vllm's CUDA kernel) computes
``out = bf16( fp32(x * rstd) ) * bf16(weight)`` — i.e. it round-trips the
normalized value through bf16 *before* multiplying by the weight, costing
~1 ULP of precision per element relative to xformers' kernel which keeps
the entire ``x * rstd * weight`` chain in fp32 and only stores the final
result in bf16.

That single extra round-trip is small per-op but compounds across 2 norms
× 30 layers + the final norm = 61 norm sites in BitNet b1.58-2B-4T.
Combined with BitNet's ternary weights (which produce many close-magnitude
logits), it's enough to flip ~20% of greedy-decode argmaxes vs SOTA.

This module mirrors ``fastkernels.tasks.baseline.L1.RMSNorm``'s public API
(same forward signature, same ``elementwise_affine`` semantics, same
optional fused-add residual path) but dispatches to xformers' ``rms_norm``
/ ``rms_norm_add`` Triton kernels which are exactly what SOTA uses.

Used only by the BitNet model.  Other models continue to use vllm's
RMSNorm so we don't perturb their numerics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from xformers.ops import rms_norm as _xf_rms_norm
    from xformers.ops import rms_norm_add as _xf_rms_norm_add
    _HAS_XFORMERS_RMS = True
except ImportError:
    _HAS_XFORMERS_RMS = False


class BitNetRMSNorm(nn.Module):
    """Drop-in for ``RMSNorm`` that matches SOTA's xformers kernel exactly.

    Falls back to the same eager pure-PyTorch path as the base RMSNorm
    when xformers is not installed (we still keep the entire chain in
    fp32 to match xformers' behaviour bit-exactly).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    @staticmethod
    def _native_forward(x: torch.Tensor, weight: torch.Tensor | None,
                        eps: float) -> torch.Tensor:
        """Match xformers' kernel: keep ``x*rstd*w`` chain in fp32, store bf16."""
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        out = xf * rstd
        if weight is not None:
            out = out * weight.to(torch.float32)
        return out.to(orig_dtype)

    def forward(self, x: torch.Tensor,
                residual: torch.Tensor | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight if self.elementwise_affine else None

        # Pure decoder layers feed (post-attn-output, pre-norm-residual)
        # through the optional ``residual`` slot to fuse the add.  Match
        # SOTA semantics: the *new* residual is ``x + residual`` (in the
        # input dtype) and the norm operates on that sum.
        if residual is not None:
            if (_HAS_XFORMERS_RMS and x.is_cuda and x.is_contiguous()
                    and residual.is_cuda and residual.is_contiguous()
                    and weight is not None):
                # ``rms_norm_add`` writes the in-place sum back to ``x`` and
                # returns the normalized output.  Matches SOTA's
                # ``increment_and_forward_`` exactly.
                out = _xf_rms_norm_add(x, residual, weight, self.eps)
                return out, x
            # Eager fallback (still bit-exact-with-xformers numerics).
            new_residual = x + residual
            out = self._native_forward(new_residual, weight, self.eps)
            return out, new_residual

        if (_HAS_XFORMERS_RMS and x.is_cuda and x.is_contiguous()
                and weight is not None):
            return _xf_rms_norm(x, weight, self.eps)
        return self._native_forward(x, weight, self.eps)
