"""Match VITS's requested sampling dtype without changing Gaussian sampling."""

import torch
from fastkernels.tasks.baseline.L3.oasis_autoencoder_kl import DiagonalGaussianDistribution


class TypedDiagonalGaussian(DiagonalGaussianDistribution):
    """Parent: L3/oasis_autoencoder_kl.py, DiagonalGaussianDistribution.sample.

    Preserve the parent's normal draw and mean/std affine mapping. Preserve the
    mean dtype and layout with randn_like, as native VITS does. The parent
    uses the global dtype and contiguous draws; both can change noise values.
    """

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)
