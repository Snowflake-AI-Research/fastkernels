"""Affine input adaptation of an existing Conv2d operation."""

from torch import nn


class InputBiasConv2d(nn.Module):
    """Add a fixed learned input bias before the unchanged convolution.

    The bias module only arranges checkpoint parameters. The actual addition
    executes here, preserving the parent's convolution reduction and layout.
    """

    def __init__(self, convolution, input_bias):
        super().__init__()
        self.convolution = convolution
        self.input_bias = input_bias

    def forward(self, inputs):
        return self.convolution(inputs + self.input_bias(inputs))
