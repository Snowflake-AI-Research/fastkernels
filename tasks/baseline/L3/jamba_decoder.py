"""Jamba decoder layer wiring -- mirrors vLLM's JambaAttentionDecoderLayer
and JambaMambaDecoderLayer (and our project's LlamaDecoderLayer).

Each Jamba layer is one of two flavours:

  * **Attention layer**  -- fused-residual pre-norm + multi-head attention
                            + fused-residual pre-FF norm + (MLP or sparse MoE).
  * **Mamba layer**      -- fused-residual pre-norm + Mamba mixer
                            + fused-residual pre-FF norm + (MLP or sparse MoE).

The attn/Mamba choice and the MLP-vs-MoE choice are encoded in the
config:

    config.layers_block_type[layer_idx] in {"attention", "mamba"}
    config.layers_num_experts[layer_idx]   # 1 -> dense MLP, >1 -> MoE

Forward signature follows the project convention (matches Llama, vLLM):

    forward(self, positions, hidden_states, residual) -> (hidden_states, residual)

**Residual handling: Llama-style fused (matching vLLM's Jamba).**
``self.input_layernorm(hidden_states, residual)`` returns
``(normed, residual + hidden_states)`` in a single CUDA kernel
(vLLM's ``fused_add_rms_norm``) -- the residual is the *running residual
stream* and ``hidden_states`` carries the per-block *delta*.  The L4
model's final ``self.final_layernorm(hidden_states, residual)`` folds
the trailing residual into the output.

This is bf16-identical with vLLM at the kernel level; the previous
non-fused HF-style residual (``h0 + norm(h)``) added a small drift
relative to vLLM that compounded over 32 layers and showed up as
~25/128 match-tokens in the bench.

Hidden-states layout: flat ``[N, hidden]`` (vLLM convention).  The
engine packs prefill prompts as flat varlen ``[total_tokens, hidden]``
and decode steps as ``[B, hidden]``; both go through the same forward.

L1 ops used: ``RMSNorm`` (with the fused-add path).
L2 ops used: ``JambaAttention``, ``JambaMambaMixer``, ``JambaMLP``,
             ``JambaMoE``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.jamba_attention import JambaAttention
from ..L2.jamba_mamba_mixer import JambaMambaMixer
from ..L2.jamba_mlp import JambaMLP
from ..L2.jamba_moe import JambaMoE


def _make_feed_forward(config, num_experts: int) -> nn.Module:
    if num_experts > 1:
        return JambaMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
        )
    return JambaMLP(config.hidden_size, config.intermediate_size)


class JambaAttentionDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mamba_layer = False
        self.self_attn = JambaAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            layer_idx=layer_idx,
        )
        num_experts = config.layers_num_experts[layer_idx]
        self.feed_forward = _make_feed_forward(config, num_experts)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor | None,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused add + RMSNorm (Llama-style, matching vLLM).  See module
        # docstring for why this matters numerically vs the HF non-fused
        # residual.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # Fully-connected (MLP / MoE) block with fused residual.
        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class JambaMambaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mamba_layer = True
        self.mamba = JambaMambaMixer(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=config.mamba_expand * config.hidden_size,
            time_step_rank=config.mamba_dt_rank,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            rms_norm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
        )
        num_experts = config.layers_num_experts[layer_idx]
        self.feed_forward = _make_feed_forward(config, num_experts)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor | None,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused add + RMSNorm (Llama-style, matching vLLM).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.mamba(positions, hidden_states)

        # Fully-connected (MLP / MoE) block with fused residual.
        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual
