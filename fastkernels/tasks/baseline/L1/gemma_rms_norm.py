"""GemmaRMSNorm: RMSNorm where the stored weight is an offset from 1.0.

Ports vLLM's ``GemmaRMSNorm`` (``vllm/model_executor/layers/layernorm.py``)
to avoid a runtime dependency on ``sgl_kernel``.

Differences from a vanilla RMSNorm:
  1. The scale applied at runtime is ``(1 + weight)`` rather than ``weight``
     (the checkpoint stores values near zero).
  2. The cast back to the original dtype happens *after* the weight multiply
     -- ``(x * w).to(orig_dtype)`` instead of ``x.to(orig_dtype) * w``.
     See https://github.com/huggingface/transformers/pull/29402.

Implementation strategy mirrors vLLM:
  - ``forward_native``: pure PyTorch (f32 promotion + variance + rsqrt + scale).
  - ``forward_cuda``: same pure-PyTorch helpers but lazily wrapped with
    ``torch.compile`` so Inductor can fuse them. Skipped when already inside
    a compiled region (``torch.compiler.is_compiling()``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GemmaRMSNorm(nn.Module):
    """RMSNorm with weight stored as offset from 1.0 (Gemma convention).

    The checkpoint stores values near zero; the scale ``(1 + weight)`` is
    materialized at runtime, matching vLLM's GemmaRMSNorm semantics exactly.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype)

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # Match vLLM: promote to f32 only when the residual add would lose
        # precision (i.e. fp16 inputs); otherwise add in the input dtype.
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x.to(orig_dtype) if x.dtype != orig_dtype else x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype), residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._forward_static_no_residual(
                self.weight.data, self.variance_epsilon, x,
            )
        return self._forward_static_with_residual(
            self.weight.data, self.variance_epsilon, x, residual,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self._forward_static_no_residual = torch.compile(  # type: ignore[method-assign]
                self._forward_static_no_residual,
            )
            self._forward_static_with_residual = torch.compile(  # type: ignore[method-assign]
                self._forward_static_with_residual,
            )
            self._is_compiled = True
        return self.forward_native(x, residual)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)
