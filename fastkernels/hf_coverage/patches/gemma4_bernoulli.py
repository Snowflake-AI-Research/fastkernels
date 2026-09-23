"""VJEPA2 stochastic-depth mask with tensor probabilities and explicit ties."""
import torch


def acceptance_mask(probabilities):
    """Adapt L3.vjepa2_layer._drop_path's uniform draw and threshold stage.

    Preserve one uniform draw per independent mask element and the pointwise
    probability threshold. Accept per-element probabilities instead of one
    keep probability, return the mask without input rescaling, and use the
    speculative sampler's inclusive <= boundary. Values above one accept;
    negative values reject. Zero accepts only a uniform draw exactly zero.
    This patch does not implement ratios, normalization or prefix reduction.
    """
    draws = torch.rand_like(probabilities)
    return draws <= probabilities
