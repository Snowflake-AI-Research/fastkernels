"""FLAEngine's nucleus sampler with Hugging Face's boundary convention."""

import torch


def sample_nucleus(logits, top_p):
    """Adapt ``infra.fla_engine.FLAEngine._sample``'s enabled top-p branch.

    Retain its full sort, softmax, cumulative sum, threshold mask, scatter,
    final softmax, and multinomial. Return the sampled device tensor instead
    of converting the index to a host integer. HF sorts ascending and
    excludes cumulative mass <= 1-p; the parent sorts descending and excludes
    preceding mass >= p. The changed boundary convention matters at ties and
    floating-point cutoffs. Temperature and existing top-k operations are
    composed by the caller before this operation.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative <= (1 - top_p)
    remove[..., -1:] = False
    sorted_logits[remove] = -float("inf")
    filtered = logits.scatter(-1, sorted_indices, sorted_logits)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1)
