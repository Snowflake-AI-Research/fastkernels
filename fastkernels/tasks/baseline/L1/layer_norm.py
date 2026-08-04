"""Standard LayerNorm wrapping F.layer_norm with optional affine parameters.

Supports create_scale and create_offset flags matching the reference
openfold3/core/model/primitives/normalization.py LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

        # fp32 views of weight/bias, filled on the first forward (see there).
        # A separate flag rather than a None check on _w32, so a module with no
        # affine params does not retry the cast every call.
        self._cast_done = False
        self._w32: torch.Tensor | None = None
        self._b32: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.promote_fp32:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )

        # Promote to fp32 for the reduction to match vLLM's
        # ``vllm/model_executor/layers/layernorm.py:LayerNorm`` which keeps
        # ``weight`` / ``bias`` in fp32 and runs the reduction in fp32.
        # Matters for the DeepSeek-V3.2 indexer ``k_norm`` — running the
        # reduction in bf16 biases the variance enough to shift the
        # FP8-quantized indexer K cache, which in turn changes the top-2048
        # selection in every sparse layer.
        orig_dtype = x.dtype
        # Cast weight/bias to fp32 ONCE, not per call. These are parameters, so
        # the cast is loop-invariant, but re-running it cost ~50 kernel launches
        # per decode step across the 21 indexer compute layers -- the same defect
        # as the indexer rope re-casting its cos/sin cache every call. Weight
        # loading completes before the first forward, so a lazy cache is safe.
        if not self._cast_done:
            w, b = self.weight, self.bias
            self._w32 = (w.float()
                         if w is not None and w.dtype != torch.float32 else w)
            self._b32 = (b.float()
                         if b is not None and b.dtype != torch.float32 else b)
            self._cast_done = True
        weight, bias = self._w32, self._b32
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)
