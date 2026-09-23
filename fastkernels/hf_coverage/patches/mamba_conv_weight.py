"""Select FP32 products in the existing L1 causal convolution kernels.

Parent operations: ``L1.causal_conv1d.causal_conv1d_fn`` and
``causal_conv1d_update``. Their ``matrix_x * matrix_w`` expression multiplies
in the operands' promoted dtype before adding to the FP32 accumulator. With
two BF16 operands that rounds each product early, unlike pinned HF's kernels.

An FP32 copy of the already-rounded weights selects FP32 multiplication in
those unchanged kernels. Inputs, outputs, cached activations, convolution
reductions and state updates retain their existing layouts and algorithms.
Only the small, static convolution-weight representation gains storage.
"""


def fp32_causal_conv_weight(weight):
    """Prepare once after loading; the selected mixed-dtype kernel runs timed."""
    return weight.detach().float().view(weight.shape[0], -1)
