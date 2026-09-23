"""Expose TopKSoftmax's first-index argmax reduction for raw codec scores."""

import torch
from torch import nn
from fastkernels.infra.cuda_ext import lazy_op

_CUDA = lazy_op("hf_coverage_codec_top1", "codec_top1.cu")


class CodecTop1(nn.Module):
    """Same CUB reduction; negative infinity replaces the probability sentinel.

    The parent mutates its input during selection, so the copy belongs to the
    measured implementation. CPU argmax is only a development comparison.
    """

    def forward(self, scores):
        if not scores.is_cuda:
            return scores.argmax(dim=-1)
        rows = scores.reshape(-1, scores.shape[-1]).float().contiguous().clone()
        return _CUDA.codec_top1(rows).long().reshape(scores.shape[:-1])
