"""Fusion encoder layer for SAM3.

Self-attention over flattened image tokens followed by cross-attention to
text/prompt features, then a feed-forward network. Pre-norm architecture
with ReLU activation (matching reference).

Reference: sam3/model/encoder.py TransformerEncoderLayer (pre_norm path)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.sam3_cross_attention import Sam3CrossAttention


class Sam3EncoderLayer(nn.Module):
    """Single fusion encoder layer for SAM3.

    Pre-norm: LN -> self-attn -> residual -> LN -> cross-attn -> residual ->
    LN -> FFN -> residual.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = Sam3CrossAttention(d_model, n_head)
        self.cross_attn = Sam3CrossAttention(d_model, n_head)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.linear1 = Linear(d_model, dim_feedforward, bias=True)
        self.linear2 = Linear(dim_feedforward, d_model, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: (B, L, D) image tokens.
            memory: (B, S, D) text/prompt tokens.
            tgt_key_padding_mask: (B, L) padding mask for tgt.
            memory_key_padding_mask: (B, S) padding mask for memory.
            query_pos: (B, L, D) positional encoding for tgt.
            pos: (B, S, D) positional encoding for memory (unused in base).

        Returns:
            (B, L, D) processed tokens.
        """
        # Self-attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if query_pos is not None else tgt2
        tgt2 = self.self_attn(q, k, tgt2, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention to prompt/text
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)

        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt
