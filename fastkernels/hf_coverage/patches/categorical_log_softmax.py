"""LogSoftmax with Categorical's explicit normalization rounding boundary."""

import torch

from fastkernels.tasks.baseline.L1.softmax import LogSoftmax


def _subtract_and_clamp(logits, normalizer):
    normalized = logits - normalizer
    return normalized, normalized.clamp_min(torch.finfo(logits.dtype).min)


_compiled_epilogue = torch.compile(_subtract_and_clamp, fullgraph=True)


class CategoricalLogSoftmax(LogSoftmax):
    """Retain row log-normalization, exposing its intermediate rounding.

    HF Categorical first materializes logsumexp in the input dtype, then
    subtracts it. Fused LogSoftmax can round differently and change BLT's
    predicted patch boundaries. The ATen reduction remains outside the compiled
    epilogue so that its dtype boundary cannot disappear through fusion.
    The unclamped result feeds Softmax; the clamped result feeds entropy's
    product, matching Categorical's handling of negative infinity.
    """

    def forward(self, logits):
        normalizer = torch.logsumexp(logits, dim=self.dim, keepdim=True)
        epilogue = _compiled_epilogue if logits.is_cuda else _subtract_and_clamp
        return epilogue(logits, normalizer)
