"""Pre-lookahead convolutional layer for CosyVoice3 causal processing.

Adopted from vllm-omni CosyVoice3 code2wav_core/layers.py.
Matches the PreLookaheadLayer interface from the reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreLookaheadLayer(nn.Module):

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        else:
            assert not self.training
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(
                torch.concat([outputs, context], dim=2),
                (0, self.pre_lookahead_len - context.size(2)),
                mode="constant",
                value=0.0,
            )
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs
