"""Detection decoder layer for SAM3.

Post-norm architecture matching reference: self-attn → text cross-attn →
image cross-attn → FFN. Position embeddings are added transiently inside
attention (via with_pos_embed) rather than accumulated into hidden states.

Reference: sam3/model/decoder.py TransformerDecoderLayer
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.sam3_cross_attention import Sam3CrossAttention


class Sam3DecoderLayer(nn.Module):
    """Single detection decoder layer for SAM3 (post-norm).

    Order: self-attn → text cross-attn → image cross-attn → FFN.
    All attention uses with_pos_embed for positional conditioning.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
        text_cross_attention: Whether to include text cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        text_cross_attention: bool = True,
    ):
        super().__init__()
        self.self_attn = Sam3CrossAttention(d_model, n_head)
        self.cross_attn = Sam3CrossAttention(d_model, n_head)

        self.has_text_cross_attn = text_cross_attention
        if text_cross_attention:
            self.ca_text = Sam3CrossAttention(d_model, n_head)
            self.catext_norm = LayerNorm(d_model)
            self.catext_dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.linear1 = Linear(d_model, dim_feedforward, bias=True)
        self.linear2 = Linear(dim_feedforward, d_model, bias=True)
        self.activation = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    @staticmethod
    def _with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_text: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        presence_token: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        """Forward pass (post-norm, matching reference decoder layer).

        Args:
            tgt: (B, Q, D) object queries.
            memory: (B, L, D) encoder memory.
            tgt_query_pos: (B, Q, D) positional encoding for queries.
            memory_pos: (B, L, D) positional encoding for memory keys.
            tgt_key_padding_mask: (B, Q) padding mask for queries.
            memory_key_padding_mask: (B, L) padding mask for memory.
            memory_text: (B, S, D) text features for text cross-attention.
            text_attention_mask: (B, S) padding mask for text.
            cross_attn_mask: (B*H, Q, L) additive mask for image cross-attn (boxRPB).
            presence_token: (B, 1, D) presence token or None.

        Returns:
            (tgt, presence_token_out): tgt is (B, Q, D), presence_token_out is (B, 1, D) or None.
        """
        tgt_query_pos_sa = tgt_query_pos

        if presence_token is not None:
            tgt = torch.cat([presence_token, tgt], dim=1)
            zero_pos = torch.zeros_like(presence_token)
            tgt_query_pos_sa = torch.cat([zero_pos, tgt_query_pos], dim=1)
            tgt_query_pos = torch.cat([zero_pos, tgt_query_pos], dim=1)

        # Self-attention with positional encoding on q, k
        q = k = self._with_pos_embed(tgt, tgt_query_pos_sa)
        tgt2 = self.self_attn(q, k, tgt, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Text cross-attention (before image, matching reference order)
        if self.has_text_cross_attn and memory_text is not None:
            tgt2 = self.ca_text(
                self._with_pos_embed(tgt, tgt_query_pos),
                memory_text, memory_text,
                key_padding_mask=text_attention_mask,
            )
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        # Prepend presence token zero-row to cross_attn_mask so it has no RPB bias
        if presence_token is not None and cross_attn_mask is not None:
            pres_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = torch.cat([pres_mask, cross_attn_mask], dim=1)

        # Image cross-attention with pos on both q and k
        tgt2 = self.cross_attn(
            self._with_pos_embed(tgt, tgt_query_pos),
            self._with_pos_embed(memory, memory_pos),
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # FFN (ReLU activation, matching reference)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:, :1, :]
            tgt = tgt[:, 1:, :]

        return tgt, presence_token_out
