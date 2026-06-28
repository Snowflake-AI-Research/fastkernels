"""RWKV7 feed-forward network: token-shift + key -> sqrelu -> value.

Built exclusively from L1 ops:
  ``token_shift`` (zero-pad + lerp), ``Linear`` x2, ``SquaredReLU``.

Weight names match the FLA checkpoint format:
  ``x_k`` (per-channel mix vector), ``key.weight``, ``value.weight``.

For cached decode, the FFN keeps a per-sequence ``conv_state`` (the last
hidden vector of the previous call) so the token-shift sees the right
"previous token" instead of zero-padding on every T=1 step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.squared_relu import SquaredReLU


class RWKV7FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = Linear(hidden_size, intermediate_size, bias=False)
        self.value = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SquaredReLU()

    def forward(
        self,
        x: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        prev_shift = None
        if past_key_values is not None:
            cs = getattr(past_key_values, "conv_states", None)
            if cs is not None:
                prev_shift = cs.get(id(self))
        shifted = torch.empty_like(x)
        if prev_shift is not None:
            shifted[:, 0] = prev_shift
        else:
            shifted[:, 0].zero_()
        if T > 1:
            shifted[:, 1:] = x[:, :-1]
        delta = shifted - x
        xk = torch.addcmul(x, delta, self.x_k)

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "conv_states"):
                past_key_values.conv_states = {}
            past_key_values.conv_states[id(self)] = x[:, -1].detach()

        return self.value(self.act(self.key(xk)))
