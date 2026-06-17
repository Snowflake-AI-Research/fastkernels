"""Cross-attention module for SAM3 encoder and decoder layers.

Performs multi-head cross-attention where queries attend to a separate
key/value memory (e.g., text prompts attending to image features, or decoder
queries attending to encoder memory).

Reference: sam3/model/encoder.py TransformerEncoderLayer cross_attn_image
           sam3/model/decoder.py TransformerDecoderLayer cross-attn paths
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear


class Sam3CrossAttention(nn.Module):
    """Multi-head cross-attention for SAM3.

    Standard cross-attention: Q from one source, K/V from another.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        bias: Whether linear projections have bias.
    """

    def __init__(self, d_model: int, n_head: int, bias: bool = True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.attn = DenseAttention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend query to key/value.

        Args:
            query: (B, Lq, D) query tensor.
            key: (B, Lk, D) key tensor.
            value: (B, Lk, D) value tensor.
            key_padding_mask: (B, Lk) boolean mask (True = padded, ignored).
            attn_mask: (B*H, Lq, Lk) additive float mask (e.g. boxRPB bias).

        Returns:
            (B, Lq, D) attended output.
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query).reshape(B, Lq, self.n_head, self.head_dim)
        k = self.k_proj(key).reshape(B, Lk, self.n_head, self.head_dim)
        v = self.v_proj(value).reshape(B, Lk, self.n_head, self.head_dim)

        use_sdpa = (
            (key_padding_mask is not None and key_padding_mask.any())
            or attn_mask is not None
        )

        if use_sdpa:
            q_sdpa = q.permute(0, 2, 1, 3)  # (B, H, Lq, D)
            k_sdpa = k.permute(0, 2, 1, 3)
            v_sdpa = v.permute(0, 2, 1, 3)

            combined_mask: Optional[torch.Tensor] = None
            if attn_mask is not None:
                combined_mask = attn_mask.reshape(B, self.n_head, Lq, Lk)

            if key_padding_mask is not None and key_padding_mask.any():
                pad_mask = ~key_padding_mask[:, None, None, :]
                pad_mask = pad_mask.expand(-1, self.n_head, Lq, -1)
                pad_bias = torch.zeros_like(pad_mask, dtype=q.dtype)
                pad_bias.masked_fill_(~pad_mask, float("-inf"))
                if combined_mask is not None:
                    combined_mask = combined_mask + pad_bias
                else:
                    combined_mask = pad_bias

            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=combined_mask,
                dropout_p=0.0,
            )
            out = out.permute(0, 2, 1, 3)  # (B, Lq, H, D)
        else:
            out = self.attn(q, k, v)

        return self.out_proj(out.reshape(B, Lq, -1))
