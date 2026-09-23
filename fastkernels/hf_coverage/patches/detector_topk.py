"""Expose TopKSoftmax's existing repeated CUB selection for raw proposal scores."""

from torch import nn
from fastkernels.infra.cuda_ext import lazy_op

_CUDA = lazy_op("hf_coverage_detector_topk", "detector_topk.cu")


class DetectorTopK(nn.Module):
    """Descending scores, first-index ties; at least k finite entries per row.

    The parent's initial and selected-score sentinels become negative infinity.
    Its reduction, k-step dependency, and scratch storage remain unchanged.
    Input cloning and conversion are included in execution. HF's tie order can
    differ; this is not a promise of identical ordering for tied proposals.
    """
    def forward(self, scores, k):
        if not scores.is_cuda:
            # Native CPU selection is a development backend, not CUDA evidence.
            return scores.float().topk(k, dim=-1)
        rows = scores.reshape(-1, scores.shape[-1]).float().contiguous().clone()
        values, indices = _CUDA.raw_topk(rows, k)
        shape = (*scores.shape[:-1], k)
        return values.reshape(shape), indices.long().reshape(shape)
