"""DP3 robot-state (proprioception) MLP builder.

Mirrors the ``create_mlp`` helper used by
``diffusion_policy_3d.model.vision.pointnet_extractor.DP3Encoder`` to build
its ``state_mlp`` Sequential. Returns a plain ``nn.Sequential`` so the
caller can install it under whatever attribute name matches the reference
checkpoint (``obs_encoder.state_mlp``).

For ``hidden_sizes=(64, 64)``, ``in_dim=9`` the expansion is:
    Linear(9, 64) -> ReLU -> Linear(64, 64)

The Sequential indices match the reference exactly so checkpoint keys
load by name.
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.linear import Linear
from ..L1.relu import ReLU


def build_state_mlp(in_dim: int,
                    hidden_sizes: tuple[int, ...] = (64, 64)) -> nn.Sequential:
    """Build the DP3 state MLP, replicating ``create_mlp`` from the reference.

    Reference signature: ``create_mlp(input_dim, output_dim, net_arch,
    activation_fn)`` where ``net_arch`` is the list of hidden sizes and
    ``output_dim`` is the final layer (no activation after the final layer).
    The reference's DP3Encoder calls it with
    ``output_dim = state_mlp_size[-1]``, ``net_arch = state_mlp_size[:-1]``.
    """
    if len(hidden_sizes) == 0:
        raise RuntimeError("state_mlp hidden_sizes is empty")
    if len(hidden_sizes) == 1:
        net_arch: tuple[int, ...] = ()
    else:
        net_arch = hidden_sizes[:-1]
    output_dim = hidden_sizes[-1]

    modules: list[nn.Module] = []
    if len(net_arch) > 0:
        modules.append(Linear(in_dim, net_arch[0]))
        modules.append(ReLU())
        for i in range(len(net_arch) - 1):
            modules.append(Linear(net_arch[i], net_arch[i + 1]))
            modules.append(ReLU())
        last_layer_dim = net_arch[-1]
    else:
        last_layer_dim = in_dim
    modules.append(Linear(last_layer_dim, output_dim))
    return nn.Sequential(*modules)
