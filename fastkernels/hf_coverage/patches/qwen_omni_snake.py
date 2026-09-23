"""CosyVoice's Snake activation with an independent magnitude parameter."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L2.cosyvoice3_hifigan import Snake


class SnakeBeta(Snake):
    """Retain Snake's pointwise sine-square graph and use beta in its divisor.

    Qwen Omni stores both frequency and magnitude in log space. The parent
    already supports log-frequency; the only semantic change is replacing
    its tied magnitude with a separate learned log-magnitude parameter.
    """

    def __init__(self, channels):
        super().__init__(channels, alpha_logscale=True)
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden):
        alpha = self.alpha[None, :, None].exp()
        beta = self.beta[None, :, None].exp()
        return hidden + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden * alpha), 2
        )
