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
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        prev_shift = None
        if past_key_values is not None:
            cs = getattr(past_key_values, "conv_states", None)
            if cs is not None:
                prev_shift = cs.get(id(self))
        shifted = torch.empty_like(x)
        if T > 1:
            shifted[:, 1:] = x[:, :-1]
        if cu_seqlens is not None:
            # Packed varlen [1, total_T, d]: token-shift must not cross
            # sequence boundaries. Each sequence's first token shifts in that
            # sequence's own previous token (its stored conv_state), or zero.
            starts = cu_seqlens[:-1].to(torch.long)
            if prev_shift is not None:
                shifted[0].index_copy_(0, starts, prev_shift.to(shifted.dtype))
            else:
                shifted[0, starts] = 0
        else:
            shifted[:, 0] = prev_shift if prev_shift is not None else 0
        delta = shifted - x
        xk = torch.addcmul(x, delta, self.x_k)

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "conv_states"):
                past_key_values.conv_states = {}
            if cu_seqlens is not None:
                ends = (cu_seqlens[1:] - 1).to(torch.long)
                past_key_values.conv_states[id(self)] = x[0].index_select(0, ends).detach()
            else:
                past_key_values.conv_states[id(self)] = x[:, -1].detach()

        return self.value(self.act(self.key(xk)))
