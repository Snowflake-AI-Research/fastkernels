"""Pure-PyTorch RMSNorm L1 op (fp32-internal, autograd-friendly).

Distinct from :class:`L1.rms_norm.RMSNorm`:

  - That op dispatches to a CUDA kernel (vLLM's ``_C.rms_norm`` or fastkernels's
    ``fastkernels_norm.rmsnorm``) by default. vLLM's kernel rounds differently
    from the reference math -- 1 bf16 ULP at every hidden size measured (16, 32,
    80, 256, 288) -- and neither has an autograd backward registered.
  - This op stays in pure PyTorch, computes variance + rsqrt in fp32 for
    numerical stability, and casts back to the input dtype before the
    weight multiply. Matches JAX/Equinox's ``nn.RMSNorm`` with
    ``promote_dtype(x, weight, dtype=compute_dtype)`` semantics.

Use this op (not :class:`RMSNorm`) wherever:
  - the head_dim or hidden size may not be a multiple of 32 (e.g. TTT-E2E
    qk_norm at head_dim=16, Llama-3 3B-style head_dim=80), OR
  - the norm sits in a ``torch.func.grad`` gradient path (e.g. an inner-loop
    SGD over a subset of params).

``allow_cuda_kernel=True`` opts a call site into a single-launch CUDA path for
inference (see :meth:`RMSNormNative.forward`). It is off by default because it
is not bit-exact with the PyTorch sequence at every size, and because the
gradient paths above need the PyTorch one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNormNative(nn.Module):
    """RMSNorm in fp32-internal PyTorch, with an optional single-kernel path.

    The PyTorch sequence costs 8 kernel launches (cast, square, mean, add-eps,
    rsqrt, scale, cast back, weight multiply). That is invisible under
    torch.compile, which fuses it -- and is how the reference library runs it --
    but it is the dominant cost of a small norm in an eager or CUDA-graph decode
    step. Jamba, which applies three of these per Mamba layer to (dt, B, C),
    spent 0.84 ms of a 6.58 ms batch-1 decode step on 84 norms' worth of
    launches: ~500 kernels doing almost no work.

    ``allow_cuda_kernel=True`` replaces the sequence with one launch of
    ``fastkernels_norm.rmsnorm`` when no gradient is being tracked. That kernel
    computes the same fp32 mean-square, rsqrt, and round-then-scale, and was
    verified bit-identical to the PyTorch sequence at every hidden size and row
    count a decode step uses (16 and 256 wide; 1..256 rows; input scales 1e-3 to
    1e2). At prefill row counts (1024+) it differs on a small number of elements
    by 1 bf16 ULP, so a call site that opts in should be checked for output
    parity against its reference, not assumed equivalent.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        allow_cuda_kernel: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.allow_cuda_kernel = allow_cuda_kernel
        if allow_cuda_kernel:
            # Importing the sibling op is what registers the
            # ``fastkernels_norm`` torch.library namespace.
            from .rms_norm import RMSNorm  # noqa: F401

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xn = xf * torch.rsqrt(var + self.eps)
        return xn.to(orig_dtype) * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.allow_cuda_kernel
            and x.is_cuda
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            # The kernel indexes rows at ``hidden_size`` stride, so a strided
            # view (dt/B/C are slices of one x_proj output) must be made dense
            # first -- the same ``.contiguous()`` the reference does before its
            # own norms.
            xc = x.reshape(-1, x.shape[-1]).contiguous()
            out = torch.empty_like(xc)
            torch.ops.fastkernels_norm.rmsnorm(out, xc, self.weight, self.eps)
            return out.view(x.shape)
        return self.forward_native(x)
