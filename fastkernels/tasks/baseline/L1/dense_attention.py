"""Dense (non-paged) multi-head attention.

Unlike the paged attention ops (FlashAttnPrefill/Decode) which use KV cache
and varlen APIs, this op handles full dense attention with a standard
(batch, seq_len, num_heads, head_dim) layout. Supports both causal and
non-causal modes.

Backend selection is controlled via the ``backend`` parameter:

  ``"auto"`` (default) — picks the fastest available backend:
    Ampere / Hopper (cc 8.x–9.x):
      FA3 via ``fa3_fwd_interface`` > ``flash_attn_interface`` > FA2 via
      ``flash_attn`` > PyTorch SDPA.
    Blackwell+ (cc >= 10.0) or pre-Ampere:
      PyTorch SDPA (dispatches to cuDNN flash attention on supported GPUs).

  ``"sdpa"`` — always use ``F.scaled_dot_product_attention``.  Fully
    ``torch.compile``-friendly and produces numerically identical results
    to diffusers' ``AttnProcessor2_0``.

  ``"flash_attn"`` — always use the flash-attention fallback chain
    (FA3 > FA2); raises if none is installed.

  ``"cudnn"`` — pin the cuDNN flash attention backend via
    ``torch.nn.attention.sdpa_kernel``. Required to actually get cuDNN
    selection through ``torch.compile``: without the context, Inductor
    bakes ``mem_efficient`` (cutlass FMHA, ``sm80`` fallback on
    Blackwell) into the compiled graph and runtime
    ``enable_*_sdp`` toggles don't override it. cuDNN's
    ``sdpa_sm100_flash_*`` kernels are ~2.7× faster than cutlass FMHA
    at typical (1024×9216 with mask) attention shapes on B200.
    ``MATH`` is included as a last-resort fallback for masks cuDNN
    can't handle.

Used by diffusion models (FLUX, SDXL) and any architecture that needs
stateless multi-head attention without KV cache, including encoder-style
bidirectional attention.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_flash_attn_func():
    """Return the best available flash-attention callable, or None."""
    for import_path in (
        ("fa3_fwd_interface", "flash_attn_func"),
        ("flash_attn_interface", "flash_attn_func"),
        ("flash_attn", "flash_attn_func"),
    ):
        try:
            mod = __import__(import_path[0], fromlist=[import_path[1]])
            return getattr(mod, import_path[1])
        except ImportError:
            continue
    return None


class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``
            (PyTorch's heuristic chooses among flash/cuDNN/mem_eff/math).
            ``"flash_attn"`` always uses the flash-attention package.
            ``"cudnn"`` pins the cuDNN flash backend via
            ``torch.nn.attention.sdpa_kernel`` (with MATH fallback for
            masks cuDNN can't handle). Required to get cuDNN flash
            through ``torch.compile`` on Blackwell.
    """

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            if self.fa_func is None:
                raise ImportError(
                    "backend='flash_attn' requested but no flash-attention "
                    "package is installed (tried fa3_fwd_interface, "
                    "flash_attn_interface, flash_attn)"
                )
            return

        # backend == "auto"
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        # SDPA handles both the masked case and the plain causal/non-causal case.
        # Custom masks force is_causal=False; FlashAttn does not support arbitrary masks.
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if self.use_flex_kernel:
            # FlexAttention generates a fused Triton fwd+bwd kernel autotuned
            # for the exact (B, H, S_q, S_kv, D) shape and the user-provided
            # mask. ``attn_mask`` here is repurposed to accept a
            # ``BlockMask`` (from ``create_block_mask``) instead of a dense
            # bool tensor. On B200 with chunked-suffix shapes
            # (Q=1024, KV=9216, D=64), the fused fwd+bwd is ~1.37x faster
            # than cuDNN flash with the equivalent dense mask
            # (microbenched). Same numerical agreement vs the fp32 MATH
            # reference (~1e-2 max-abs-diff in bf16, identical to cuDNN).
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            out = self._flex_fn(
                q, k, v,
                block_mask=attn_mask,
                scale=softmax_scale,
            )
        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # cuDNN flash needs contiguous tensors. The permute above
            # produces non-contiguous strides; without ``.contiguous()``
            # cuDNN silently falls back to whatever's next on the list.
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            # Try strict cuDNN first. Adding MATH as a fallback in the
            # ``sdpa_kernel`` list causes PyTorch's selection heuristic to
            # pick MATH over cuDNN (~10× slower) for inputs both can
            # handle. If cuDNN rejects (e.g. head_dim=16, fp32, or some
            # mask shape it doesn't support), fall back through MATH.
            try:
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            except RuntimeError:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask.to(dtype=q.dtype) if attn_mask is not None else None,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)
