"""Mamba2 decoder layer: RMSNorm + Mamba2Mixer with vLLM-style residual.

Mirrors ``vllm/model_executor/models/mamba2.Mamba2DecoderLayer`` and
follows kb_nano's ``LlamaDecoderLayer`` residual convention:

  - first layer (residual is None): residual = hidden_states;
    hidden_states = norm(hidden_states)
  - subsequent layers: hidden_states, residual = norm(hidden_states, residual)
    (fused add + RMS norm)

Returns ``(mixer_output, residual)``; the model wraps the final residual
add into ``norm_f`` outside the layer loop.

Weight names match HuggingFace mamba2 checkpoints:
  layers.{i}.norm.weight    [hidden_size]
  layers.{i}.mixer.*        (see mamba2_mixer.py)
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.mamba2_mixer import Mamba2Mixer


class Mamba2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        intermediate = getattr(config, "intermediate_size", config.expand * config.hidden_size)
        self.mixer = Mamba2Mixer(
            hidden_size=config.hidden_size,
            ssm_state_size=config.state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=intermediate,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.layer_norm_epsilon,
            chunk_size=config.chunk_size,
            layer_idx=layer_idx,
            quant_config=quant_config,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, positions, hidden_states, residual):
        # ``positions`` is unused for Mamba but kept in the signature so
        # the layer is interchangeable with attention layers.
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual
