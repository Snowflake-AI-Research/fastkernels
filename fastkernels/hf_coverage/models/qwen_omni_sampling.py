"""Omni generation filters composed from existing selection operations."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.fla_engine import FLAEngine
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.top_k_per_row import TopKPerRow
from ..patches.codec_top1 import CodecTop1
from ..patches.qwen_omni_sampling import sample_nucleus


class OmniSampling(nn.Module):
    """Preserve repetition, temperature, top-k, top-p, then sampling order."""

    def __init__(self, top_k, top_p, temperature, repetition_penalty):
        super().__init__()
        self.top_k, self.top_p = top_k, top_p
        self.temperature, self.repetition_penalty = temperature, repetition_penalty
        self.topk, self.compare = TopKPerRow(), CodecTop1()
        self.minimum = MaxPool2d((1, top_k)) if top_k else None

    def filter(self, logits, history):
        logits = logits.float().clone()
        if self.repetition_penalty != 1 and history.numel():
            scores = logits.gather(-1, history)
            negative = self.compare(torch.stack((scores, torch.zeros_like(scores)), -1)).bool()
            penalized = torch.where(
                negative, scores * self.repetition_penalty, scores / self.repetition_penalty
            )
            logits.scatter_(-1, history, penalized)
        logits = logits / self.temperature
        if self.top_k:
            rows, width = logits.shape
            indices = self.topk.forward_prefill(
                logits, torch.zeros(rows, dtype=torch.int32, device=logits.device),
                torch.full((rows,), width, dtype=torch.int32, device=logits.device), self.top_k,
            )
            selected = logits.gather(-1, indices.long())
            threshold = -self.minimum(-selected[:, None, None, :]).reshape(rows, 1)
            # CodecTop1 chooses index zero at a tie, retaining every kth tie.
            remove = self.compare(torch.stack((logits, threshold.expand_as(logits)), -1)).bool()
            logits.masked_fill_(remove, -float("inf"))
        return logits

    def forward(self, logits, history):
        filtered = self.filter(logits, history)
        if self.top_p < 1:
            return sample_nucleus(filtered, self.top_p).squeeze(-1)
        samples = [
            FLAEngine._sample(None, row, SimpleNamespace(temperature=1.0, top_p=1.0))
            for row in filtered
        ]
        return torch.tensor(samples, dtype=torch.long, device=logits.device)
