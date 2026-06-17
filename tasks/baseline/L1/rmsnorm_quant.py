"""Fused RMSNorm + per-token-group FP8 quantization.

Single CUDA kernel that combines rmsnorm and FP8 quantization, eliminating
the intermediate BF16 tensor and one kernel launch per decoder layer.

Also registers the fused ops in the ``kb_nano_norm`` torch.library namespace
so Inductor fusion passes can reference them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .csrc import _C

_FP8_GROUP_SIZE = 128

# ---------------------------------------------------------------------------
# Register fused ops in kb_nano_norm namespace for Inductor fusion passes.
# These use the same Library as rms_norm.py (IMPL mode to extend it).
# ---------------------------------------------------------------------------

_fused_lib = torch.library.Library("kb_nano_norm", "DEF")

_fused_lib.define(
    "rmsnorm_fp8_quant(Tensor! output_fp8, Tensor! output_scales, "
    "Tensor input, Tensor weight, float eps) -> ()"
)

def _rmsnorm_fp8_quant_impl(output_fp8, output_scales, input, weight, eps):
    _C.rmsnorm_fp8_quant(output_fp8, output_scales, input, weight, eps)

_fused_lib.impl("rmsnorm_fp8_quant", _rmsnorm_fp8_quant_impl, "CUDA")

@torch.library.impl(_fused_lib, "rmsnorm_fp8_quant", "Meta")
def _rmsnorm_fp8_quant_meta(output_fp8, output_scales, input, weight, eps):
    pass

_fused_lib.define(
    "fused_add_rmsnorm_fp8_quant(Tensor! output_fp8, Tensor! output_scales, "
    "Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> ()"
)

def _fused_add_rmsnorm_fp8_quant_impl(output_fp8, output_scales,
                                       input, residual, weight, eps):
    _C.fused_add_rmsnorm_fp8_quant(output_fp8, output_scales,
                                    input, residual, weight, eps)

_fused_lib.impl("fused_add_rmsnorm_fp8_quant",
                _fused_add_rmsnorm_fp8_quant_impl, "CUDA")

@torch.library.impl(_fused_lib, "fused_add_rmsnorm_fp8_quant", "Meta")
def _fused_add_rmsnorm_fp8_quant_meta(output_fp8, output_scales,
                                       input, residual, weight, eps):
    pass


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

class RMSNormFP8Quant(nn.Module):
    """Fused RMSNorm + per-token-group FP8 quantization.

    Stateless wrapper around ``torch.ops.kb_nano_norm.rmsnorm_fp8_quant``.
    The RMSNorm weight is owned by the parent module and passed through
    ``forward``.
    """

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RMSNorm + per-token-group FP8 quantization.

        Returns:
            (output_fp8, output_scales) where output_fp8 is
            [num_tokens, hidden_size] in float8_e4m3fn and output_scales is
            [num_tokens, num_groups] in float32 with column-major layout.
        """
        hidden_size = input.size(-1)
        num_tokens = input.numel() // hidden_size
        num_groups = (hidden_size + _FP8_GROUP_SIZE - 1) // _FP8_GROUP_SIZE

        output_fp8 = torch.empty(
            num_tokens, hidden_size,
            dtype=torch.float8_e4m3fn, device=input.device,
        )
        output_scales = torch.empty(
            num_groups, num_tokens, dtype=torch.float32, device=input.device,
        ).transpose(0, 1)

        torch.ops.kb_nano_norm.rmsnorm_fp8_quant(
            output_fp8, output_scales, input, weight, eps,
        )
        return output_fp8, output_scales


class FusedAddRMSNormFP8Quant(nn.Module):
    """Fused residual-add + RMSNorm + per-token-group FP8 quantization.

    Stateless wrapper around
    ``torch.ops.kb_nano_norm.fused_add_rmsnorm_fp8_quant``.
    """

    def forward(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Modifies *input* and *residual* in-place."""
        hidden_size = input.size(-1)
        num_tokens = input.numel() // hidden_size
        num_groups = (hidden_size + _FP8_GROUP_SIZE - 1) // _FP8_GROUP_SIZE

        output_fp8 = torch.empty(
            num_tokens, hidden_size,
            dtype=torch.float8_e4m3fn, device=input.device,
        )
        output_scales = torch.empty(
            num_groups, num_tokens, dtype=torch.float32, device=input.device,
        ).transpose(0, 1)

        torch.ops.kb_nano_norm.fused_add_rmsnorm_fp8_quant(
            output_fp8, output_scales, input, residual, weight, eps,
        )
        return output_fp8, output_scales, residual
