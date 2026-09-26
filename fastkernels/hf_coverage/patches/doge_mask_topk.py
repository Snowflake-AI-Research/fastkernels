"""Extract GroupedTopK's native raw-score selection without router scoring.

Parent: L1/grouped_topk.py fallback torch.topk(scores_for_choice, ..., sorted=False).
Doge already supplies its dynamic attention scores, so sigmoid/softmax and
expert grouping are absent. The unchanged native GPU top-k retains its actual
BF16 tie policy; DetectorTopK's first-index policy is not interchangeable here.
No new reduction kernel, score conversion, or synthetic matrix is introduced.
"""
import torch
from torch import nn


class DogeMaskTopK(nn.Module):
    def forward(self, scores, k):
        return torch.topk(scores, k, dim=-1, largest=True, sorted=False)
