"""HunyuanVideo-1.5 transformer block (L3 composite).

Dual-stream DiT block with AdaLayerNormZero conditioning, joint
attention (video + encoder), and separate FFNs for each stream.

Mirrors vllm-omni's ``HunyuanVideo15TransformerBlock`` in
``vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.ada_layer_norm import AdaLayerNormZero
from ..L2.flux_feedforward import FeedForward
from ..L2.hunyuan_video_attention import HunyuanVideo15Attention


class HunyuanVideo15TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = HunyuanVideo15Attention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            eps=1e-6,
        )

        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

        self.norm2_context = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states
