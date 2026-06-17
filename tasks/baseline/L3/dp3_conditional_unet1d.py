"""DP3 1-D conditional U-Net (denoising network).

Mirrors ``diffusion_policy_3d.model.diffusion.conditional_unet1d.ConditionalUnet1D``
restricted to:
    - ``condition_type="film"`` (the only mode used by released DP3 configs)
    - ``local_cond=None`` (used only if ``local_cond_dim`` is configured)
    - ``use_down_condition=True``, ``use_mid_condition=True``,
      ``use_up_condition=True`` (defaults — no released configs disable them)

Module names (down_modules / mid_modules / up_modules / final_conv /
diffusion_step_encoder) are kept identical to the reference so checkpoint
state_dict keys load with no remapping.

Wiring/composition only — all computation lives in L1/L2 ops.
"""

from __future__ import annotations

from typing import Union

import einops
import torch
import torch.nn as nn

from ..L2.dp3_conditional_residual_block import ConditionalResidualBlock1D
from ..L2.dp3_conv1d_block import Conv1dBlock, Downsample1d, Upsample1d
from ..L2.dp3_diffusion_step_embedding import build_diffusion_step_encoder


class ConditionalUnet1D(nn.Module):
    """1-D U-Net with FiLM conditioning on diffusion-step + global obs cond.

    Args:
        input_dim: action dim (channels of the trajectory tensor).
        global_cond_dim: encoder feature dim concatenated with the time embed.
            ``None`` ⇒ no global conditioning.
        diffusion_step_embed_dim: time-embed dim (``dsed``). Default 256
            (DP3) / 128 (Simple-DP3 and the released ``dp3.yaml``).
        down_dims: channel sizes of the down-stack (e.g. [128,256,384] for
            Simple-DP3, [512,1024,2048] for full DP3).
        kernel_size: Conv1d kernel size in residual blocks (default 5 for DP3).
        n_groups: GroupNorm groups (default 8).
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.diffusion_step_encoder = build_diffusion_step_encoder(dsed)
        cond_dim = dsed + (global_cond_dim if global_cond_dim is not None else 0)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Mid stack: two res-blocks at bottleneck channel.
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
            ),
        ])

        # Down stack: per-stage [resnet, resnet2, downsample (or Identity)].
        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # Up stack: per-stage [resnet (cat skip → in), resnet2, upsample (or Identity)].
        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.down_modules = down_modules
        self.up_modules = up_modules

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, T, input_dim) trajectory.
            timestep: (B,) or scalar diffusion timestep.
            global_cond: (B, global_cond_dim) or None.
        Returns:
            (B, T, input_dim) — predicted x_0 (DP3 prediction_type="sample").
        """
        # Reference rearrange: 'b h t -> b t h' moves channels to the middle so
        # Conv1d operates over the time axis.
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device,
            )
        elif torch.is_tensor(timesteps) and timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
        else:
            global_feature = timestep_embed

        x = sample
        h: list[torch.Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
