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

# cuDNN's SDPA kernels are limited to head_dim <= 128 ("head_dim should be no
# more than 128" in sdp_utils.cpp); larger heads must use EFFICIENT/MATH.
_CUDNN_MAX_HEAD_DIM = 128


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

        # backend == "auto": flash-attn on Ampere/Hopper (80<=cc<100); cuDNN flash
        # on Blackwell (cc>=100), where PyTorch's SDPA heuristic otherwise picks
        # FA2 (~3.6x slower than cuDNN for large joint-attention shapes on B200).
        # This mirrors vllm-omni's platform selector, which pins cuDNN/TRTLLM on
        # Blackwell. The cuDNN forward path already falls back to mem-efficient/MATH
        # for shapes/masks cuDNN rejects, so this is safe as a default.
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

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
            # An explicit mask plus is_causal=True is ambiguous, and the two code
            # paths here would resolve it differently: this branch would hand both
            # to SDPA (which applies the causal mask *on top of* attn_mask), while
            # the non-cuDNN branch below drops is_causal and treats attn_mask as
            # authoritative. SDPA itself accepts the combination on this backend
            # rather than rejecting it, so nothing would surface the disagreement
            # -- reject it here instead of silently masking twice.
            if attn_mask is not None and causal:
                raise ValueError(
                    "DenseAttention: pass either attn_mask or causal=True, not both "
                    "(an explicit mask must already encode causality). Got "
                    f"attn_mask={tuple(attn_mask.shape)} with causal=True."
                )
            # The sdpa_kernel context below FORCES cuDNN, and on Blackwell (sm100,
            # cuDNN 9.19) the cuDNN flash kernel accepts the permuted, non-contiguous
            # q/k/v views directly -- so we skip the q/k/v .contiguous() clones (they
            # were a real cost: 3 clones/block x54 blocks). Verified bit-identical and
            # faster; if cuDNN ever rejects a layout it raises -> MATH fallback below.
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            # Try strict cuDNN first. Adding MATH as a fallback in the
            # ``sdpa_kernel`` list causes PyTorch's selection heuristic to
            # pick MATH over cuDNN (~10× slower) for inputs both can
            # handle. If cuDNN rejects (e.g. head_dim=16, fp32, or some
            # mask shape it doesn't support), fall back through MATH.
            #
            # head_dim > 128 is rejected by cuDNN unconditionally ("head_dim
            # should be no more than 128"), so route it straight to the backends
            # that can serve it. The try/except below only recovers in eager --
            # under torch.compile the RuntimeError surfaces during fake-tensor
            # tracing and aborts the whole graph rather than taking the handler,
            # which is how a head_dim=256 model (Gemma-2B in Pi0) failed to
            # compile at all.
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                # EFFICIENT_ATTENTION requires an additive bias in the query's
                # dtype ("invalid dtype for bias - should match query's dtype");
                # cuDNN tolerated an fp32 mask against bf16 q/k/v. A bool mask is
                # passed through -- coercing it would turn True/False into a
                # 1.0/0.0 additive bias.
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            else:
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
            # SDPA accepts a boolean mask (True = attend) directly; only a float
            # (additive) mask needs dtype coercion. Coercing a bool mask to q.dtype
            # would turn True/False into a 1.0/0.0 additive bias (wrong semantics) --
            # e.g. the HunyuanVideo key-padding mask would then fail to mask padding
            # on non-cuDNN backends.
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)
