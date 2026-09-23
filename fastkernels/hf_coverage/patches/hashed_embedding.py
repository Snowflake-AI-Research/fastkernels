"""Embedding adaptation: transform each supplied character ID before lookup."""

import torch
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.embedding import Embedding


def _lookup(input_ids, weight, prime, buckets):
    return F.embedding(((input_ids + 1) * prime) % buckets, weight)


_compiled_lookup = torch.compile(_lookup, fullgraph=True)


class HashedEmbedding(Embedding):
    """Keep the parent's independent gather and table/output storage.

    Only the source-row index changes, using integer arithmetic before the
    gather. There is no reduction or communication between characters. The
    CUDA path executes compiled code; the CPU path supports diagnosis.
    """

    def __init__(self, num_embeddings, embedding_dim, prime):
        super().__init__(num_embeddings, embedding_dim)
        self.prime = prime

    def forward(self, input_ids):
        lookup = _compiled_lookup if input_ids.is_cuda else _lookup
        return lookup(input_ids, self.emb.weight, self.prime, self.emb.num_embeddings)
