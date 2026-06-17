"""RMSNorm with dual dispatch: CUDA custom op (eager) and pure-PyTorch (compiled).

Mirrors vLLM's ``CustomOp`` dispatch pattern:
  - ``forward_cuda``: calls vLLM's ``torch.ops._C.rms_norm`` /
    ``torch.ops._C.fused_add_rms_norm`` CUDA kernels for bitwise-identical
    numerics with vLLM.  Falls back to ``kb_nano_norm.*`` if vLLM ops
    are not available.
  - ``forward_native``: pure PyTorch implementation (f32 promotion, variance,
    rsqrt, weight multiply).  Used when torch.compile is active so Inductor
    can inline, fuse, and optimise the norm with adjacent ops — this is the
    key mechanism that enables RMSNorm+FP8-quant fusion.

The ``forward`` method dispatches based on ``torch.compiler.is_compiling()``.

Known limitations of the CUDA kernel (forward_cuda path):
  - Produces incorrect output for hidden sizes that aren't multiples of 32
    (verified empirically: hidden=16 and hidden=80 give max-abs error ~1e3
    on random unit-variance input vs the reference math; hidden=32, 64, 128
    are correct).
  - Has no ``torch.autograd`` backward registered, so the norm silently
    drops gradient under ``torch.func.grad``.

Use :class:`L1.rms_norm_native.RMSNormNative` instead when you need either
of those properties — odd head_dims (e.g. TTT-E2E qk_norm at head_dim=16,
or any model with head_dim that isn't a multiple of 32), or autograd /
torch.func.grad support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csrc import _C

try:
    import vllm._C  # noqa: F401 — registers torch.ops._C.rms_norm etc.
    _VLLM_NORM_AVAILABLE = True
except ImportError:
    _VLLM_NORM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Register _C ops as torch.library custom ops for torch.compile compatibility.
# These are used in eager mode and as CUDA graph replay targets.
# ---------------------------------------------------------------------------

_lib = torch.library.Library("kb_nano_norm", "DEF")

_lib.define("rmsnorm(Tensor! result, Tensor input, Tensor weight, float eps) -> ()")

def _rmsnorm_impl(result, input, weight, eps):
    _C.rmsnorm(result, input, weight, eps)

_lib.impl("rmsnorm", _rmsnorm_impl, "CUDA")

@torch.library.impl(_lib, "rmsnorm", "Meta")
def _rmsnorm_meta(result, input, weight, eps):
    pass

_lib.define(
    "fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, "
    "Tensor weight, float eps) -> ()"
)

def _fused_add_rmsnorm_impl(input, residual, weight, eps):
    _C.fused_add_rmsnorm(input, residual, weight, eps)

_lib.impl("fused_add_rmsnorm", _fused_add_rmsnorm_impl, "CUDA")

@torch.library.impl(_lib, "fused_add_rmsnorm", "Meta")
def _fused_add_rmsnorm_meta(input, residual, weight, eps):
    pass


# ---------------------------------------------------------------------------
# RMSNorm module
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            # Match vLLM's has_weight=False path: use the same CUDA RMSNorm
            # kernel with a non-persistent unit scale instead of falling back
            # to torch.nn.functional.rms_norm in eager/CUDA-graph decode.
            self.register_buffer(
                "_unit_weight",
                torch.ones(hidden_size),
                persistent=False,
            )

    # -- Pure PyTorch path (used under torch.compile so Inductor can fuse) --

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        hidden_size: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch RMSNorm matching vLLM's forward_static."""
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        return x, residual

    # -- CUDA kernel path (used in eager mode / CUDA graph replay) --
    # Prefers vLLM's CUDA kernels for bitwise-identical numerics;
    # falls back to kb-nano's own kernels when vLLM is unavailable.

    @staticmethod
    def forward_cuda(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if weight is not None:
            if _VLLM_NORM_AVAILABLE:
                if residual is None:
                    out = torch.empty_like(x)
                    torch.ops._C.rms_norm(out, x, weight, eps)
                    return out
                else:
                    torch.ops._C.fused_add_rms_norm(x, residual, weight, eps)
                    return x, residual
            else:
                if residual is None:
                    out = torch.empty_like(x)
                    torch.ops.kb_nano_norm.rmsnorm(out, x, weight, eps)
                    return out
                else:
                    torch.ops.kb_nano_norm.fused_add_rmsnorm(
                        x, residual, weight, eps,
                    )
                    return x, residual
        else:
            if residual is None:
                return F.rms_norm(x, (x.size(-1),), eps=eps)
            else:
                x = x + residual
                residual = x
                return F.rms_norm(x, (x.size(-1),), eps=eps), residual

    def forward(self, x, residual=None):
        if torch.compiler.is_compiling():
            return self.forward_native(
                x, self.weight if self.elementwise_affine else None,
                self.eps, self.hidden_size, residual,
            )
        weight = self.weight if self.elementwise_affine else self._unit_weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        return self.forward_cuda(
            x, weight, self.eps, residual,
        )
