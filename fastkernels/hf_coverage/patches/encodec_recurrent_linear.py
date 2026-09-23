"""EnCodec recurrent Linear with separate FP32 accumulation of gate biases."""

from fastkernels.tasks.baseline.L1.linear import Linear


class EncodecRecurrentLinear(Linear):
    """Retain Linear's GEMM, adding the precomputed input branch before biases.

    Native BF16 LSTM GEMMs round before the cell's FP32 bias/gate arithmetic.
    The matrix reduction and recurrent state dependency remain unchanged.
    """

    def forward(self, hidden, input_projection, input_bias):
        recurrent = self.matmul(hidden, self.weight)
        return (input_projection.float() + recurrent.float()
                + input_bias.float() + self.bias.float())
