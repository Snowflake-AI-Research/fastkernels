"""ColBERTv2 embedding and MaxSim ops."""

from __future__ import annotations

import torch
import torch.nn as nn


class ColBERTv2TokenMask(nn.Module):
    def __init__(self, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        skiplist: set[int] | None = None,
    ) -> torch.Tensor:
        blocked = {self.pad_token_id}
        if skiplist:
            blocked |= {int(token_id) for token_id in skiplist}
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in blocked:
            mask &= input_ids.ne(token_id)
        return mask


class ColBERTv2ScoreReduce(nn.Module):
    def forward(
        self,
        scores_padded: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        doc_padding = ~doc_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
        scores_padded = scores_padded.masked_fill(doc_padding.unsqueeze(-1), -9999)
        return scores_padded.max(1).values.sum(-1)


class ColBERTv2MaxSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_reduce = ColBERTv2ScoreReduce()

    def forward(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = doc_vecs @ query_vecs.to(dtype=doc_vecs.dtype).permute(0, 2, 1)
        return self.score_reduce(scores, doc_mask)
