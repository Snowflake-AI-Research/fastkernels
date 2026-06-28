"""Memory attention transformer for SAM3 tracker.

TransformerEncoderCrossAttention: applies self-attention (with RoPE) + cross-attention
to condition current-frame features on memory features from previous frames.

TransformerDecoderLayerv2: Pre-norm layer with self-attention + cross-attention + FFN,
used as the building block for the memory attention transformer.

Reference: sam3/model/decoder.py TransformerEncoderCrossAttention + TransformerDecoderLayerv2
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.sam3_rope_attention import Sam3RoPEAttention


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    from copy import deepcopy
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Sam3MemoryAttentionLayer(nn.Module):
    """Pre-norm transformer layer with RoPE self-attention + RoPE cross-attention + FFN.

    Reference: sam3/model/decoder.py TransformerDecoderLayerv2
    """

    def __init__(
        self,
        *,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        num_heads: int,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        cross_attention_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        return tgt + self.dropout1(tgt2)

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        pos: torch.Tensor,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        if self.cross_attn_image is None:
            return tgt

        kwds = {}
        if num_k_exclude_rope > 0:
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        return tgt + self.dropout2(tgt2)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dac: bool = False,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        num_k_exclude_rope: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class Sam3MemoryAttention(nn.Module):
    """Memory attention transformer: condition current frame on past memories.

    Reference: sam3/model/decoder.py TransformerEncoderCrossAttention
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        prompt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        prompt_key_padding_mask: Optional[torch.Tensor] = None,
        src_pos: Optional[torch.Tensor] = None,
        prompt_pos: Optional[torch.Tensor] = None,
        feat_sizes: Optional[list] = None,
        num_obj_ptr_tokens: int = 0,
    ) -> dict:
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = src[0], src_key_padding_mask[0], src_pos[0]

        assert src.shape[1] == prompt.shape[1]

        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, Sam3RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                **kwds,
            )
            normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
        }
