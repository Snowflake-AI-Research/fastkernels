"""Mamba v1 decoder layer: RMSNorm + MambaMixer with vLLM-style residual.

Mirrors ``vllm/model_executor/models/mamba.MambaDecoderLayer`` and
follows kb_nano's ``LlamaDecoderLayer`` residual convention:

  - first layer (residual is None): residual = hidden_states;
    hidden_states = norm(hidden_states)
  - subsequent layers: hidden_states, residual = norm(hidden_states, residual)
    (fused add + RMS norm)

Returns ``(mixer_output, residual)``.

Weight names match HuggingFace checkpoint:
  layers.{i}.norm.weight    [hidden_size]
  layers.{i}.mixer.*        (see mamba_mixer.py)
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.mamba_mixer import MambaMixer


class MambaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.mixer = MambaMixer(
            hidden_size=config.hidden_size,
            ssm_state_size=config.state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.intermediate_size,
            time_step_rank=config.time_step_rank,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            layer_idx=layer_idx,
            quant_config=quant_config,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, positions, hidden_states, residual):
        # positions unused for Mamba; kept for layer-uniform signature.
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual
