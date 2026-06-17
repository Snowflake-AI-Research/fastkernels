"""V-JEPA 2 transformer layer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L2.vjepa2_attention import VJEPA2RopeAttention
from ..L2.vjepa2_mlp import VJEPA2MLP


def _drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1.0 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    return input.div(keep_prob) * random_tensor


class VJEPA2DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _drop_path(hidden_states, self.drop_prob, self.training)


class VJEPA2Layer(nn.Module):
    """Single V-JEPA 2 transformer block."""

    def __init__(
        self,
        config,
        drop_path_rate: float = 0.0,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2RopeAttention(config, hidden_size, num_attention_heads)
        self.drop_path = VJEPA2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states,
            position_mask=position_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.drop_path(self_attention_outputs[0]) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        outputs = self_attention_outputs[1:]
        return (hidden_states,) + outputs
