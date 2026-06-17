"""TTT-E2E transformer block (L2).

Mirrors ``ttt/model/transformer.py:Block`` exactly. Layout per block:

    x0 = x
    x  = pre_norm(seq_norm) -> SWA -> post_norm(seq_post_norm)
    x  = x + x0

    [if has_prime (suffix blocks only):
        x0 = x
        x  = pre_norm(ffn_prime_norm) -> SwiGLU_prime -> post_norm(ffn_prime_post_norm)
        x  = x + x0
    ]

    x0 = x
    x  = pre_norm(ffn_norm) -> SwiGLU -> post_norm(ffn_post_norm)
    x  = x + x0

The "prime" FFN's weights are present only on the last ``suffix_len`` blocks,
and are test-time-trained chunk-by-chunk by the L4 pipeline.

Composed entirely of L1 ops + the local L2 SWA / SwiGLU sub-modules:
  - :class:`RMSNormNative` (L1) for all norms — autograd-friendly so the
    inner-loop SGD path through ``torch.func.grad`` is well-defined.
  - :class:`TTTE2ESWA` (L2) for sliding-window attention.
  - :class:`TTTE2ESwiGLU` (L2) for both the regular and prime FFN.
"""

from __future__ import annotations

import torch
from torch import nn

from ..L1.rms_norm_native import RMSNormNative
from .ttt_e2e_swa import TTTE2ESWA
from .ttt_e2e_swiglu import TTTE2ESwiGLU


class TTTE2EBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        window_size: int,
        chunk_size: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        has_prime: bool = False,
        attention_backend: str = "cudnn",
    ):
        super().__init__()
        self.has_prime = has_prime
        self.hidden_size = hidden_size

        self.seq_modeling_block = TTTE2ESWA(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            chunk_size=chunk_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            rms_norm_eps=rms_norm_eps,
            attention_backend=attention_backend,
        )
        self.feed_forward = TTTE2ESwiGLU(hidden_size, intermediate_size)

        self.seq_norm = RMSNormNative(hidden_size, eps=rms_norm_eps)
        self.ffn_norm = RMSNormNative(hidden_size, eps=rms_norm_eps)
        self.seq_post_norm = RMSNormNative(hidden_size, eps=rms_norm_eps)
        self.ffn_post_norm = RMSNormNative(hidden_size, eps=rms_norm_eps)

        if has_prime:
            self.feed_forward_prime = TTTE2ESwiGLU(hidden_size, intermediate_size)
            self.ffn_prime_norm = RMSNormNative(hidden_size, eps=rms_norm_eps)
            self.ffn_prime_post_norm = RMSNormNative(hidden_size, eps=rms_norm_eps)
        else:
            self.feed_forward_prime = None
            self.ffn_prime_norm = None
            self.ffn_prime_post_norm = None

    # ------------------------------------------------------------- residual FFN

    def _residual_ffn(self, x: torch.Tensor, ffn: nn.Module, pre: nn.Module, post: nn.Module) -> torch.Tensor:
        return x + post(ffn(pre(x)))

    # ---------------------------------------------------------------- prefix path

    def forward_prefix(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = self.seq_modeling_block.forward_prefix(self.seq_norm(x), position_ids=position_ids)
        x = x + self.seq_post_norm(h)

        if self.has_prime:
            x = self._residual_ffn(x, self.feed_forward_prime, self.ffn_prime_norm, self.ffn_prime_post_norm)

        x = self._residual_ffn(x, self.feed_forward, self.ffn_norm, self.ffn_post_norm)
        return x

    # ---------------------------------------------------------------- suffix path

    def forward_suffix_chunk(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        chunk_id: int | torch.Tensor,
        prime_override: dict | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one chunk through this suffix block.

        ``prime_override``: optional dict ``{"w1": Tensor, "w2": Tensor,
        "w3": Tensor}`` of fresh prime-FFN weights (kept in fp32 by the L4
        inner loop). When provided, the prime FFN is called via
        ``torch.func.functional_call`` with these weights instead of the
        stored Parameters; the cast to ``x.dtype`` is differentiable so the
        gradient path back to the fp32 source is preserved.
        """
        b, c, _ = x.shape
        assert c == self.seq_modeling_block.chunk_size

        h, new_cache = self.seq_modeling_block.forward_suffix_chunk(
            self.seq_norm(x), kv_cache, chunk_id,
        )
        x = x + self.seq_post_norm(h)

        if self.has_prime:
            assert self.feed_forward_prime is not None
            h_in = self.ffn_prime_norm(x)
            if prime_override is None:
                h = self.feed_forward_prime(h_in)
            else:
                # JAX reference: prime params live in state_dtype (fp32) and
                # are differentiated in fp32; the forward casts them to
                # compute_dtype (bf16) via promote_dtype right before the
                # matmul. ``v.to(h_in.dtype)`` preserves the gradient
                # connection back to the fp32 ``prime_override`` tensors.
                h = torch.func.functional_call(
                    self.feed_forward_prime,
                    {f"{k}.weight": v.to(h_in.dtype) for k, v in prime_override.items()},
                    (h_in,),
                )
            x = x + self.ffn_prime_post_norm(h)

        x = self._residual_ffn(x, self.feed_forward, self.ffn_norm, self.ffn_post_norm)
        return x, new_cache
