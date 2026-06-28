"""Diffusion module for AlphaFold3.

Implements AF3 Algorithm 20 (denoising) and Algorithm 18 (sampling).
Pipeline: DiffusionConditioning -> AtomAttentionEncoder -> DiffusionTransformer
-> AtomAttentionDecoder -> EDM-style output combination.

Reference: openfold3/core/model/structure/diffusion_module.py DiffusionModule
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.alphafold3_atom_attention import AtomAttentionDecoder, AtomAttentionEncoder
from ..L2.alphafold3_diffusion_conditioning import DiffusionConditioning
from .alphafold3_diffusion_transformer import DiffusionTransformer


def _quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [*, 4] to rotation matrix [*, 3, 3]."""
    q = quat[..., None] * quat[..., None, :]
    r00 = q[..., 0, 0] + q[..., 1, 1] - q[..., 2, 2] - q[..., 3, 3]
    r01 = 2 * (q[..., 1, 2] - q[..., 0, 3])
    r02 = 2 * (q[..., 1, 3] + q[..., 0, 2])
    r10 = 2 * (q[..., 1, 2] + q[..., 0, 3])
    r11 = q[..., 0, 0] - q[..., 1, 1] + q[..., 2, 2] - q[..., 3, 3]
    r12 = 2 * (q[..., 2, 3] - q[..., 0, 1])
    r20 = 2 * (q[..., 1, 3] - q[..., 0, 2])
    r21 = 2 * (q[..., 2, 3] + q[..., 0, 1])
    r22 = q[..., 0, 0] - q[..., 1, 1] - q[..., 2, 2] + q[..., 3, 3]
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def centre_random_augmentation(
    xl: torch.Tensor, atom_mask: torch.Tensor, scale_trans: float = 1.0,
) -> torch.Tensor:
    """AF3 Algorithm 19: random rotation + translation + centering."""
    q = torch.randn(*xl.shape[:-2], 4, dtype=xl.dtype, device=xl.device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    rots = _quat_to_rot(q)

    trans = scale_trans * torch.randn(
        (*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device,
    )

    mask_sum = torch.sum(atom_mask[..., None], dim=-2, keepdim=True).clamp(min=1.0)
    mean_xl = torch.sum(xl * atom_mask[..., None], dim=-2, keepdim=True) / mask_sum
    pos_centered = xl - mean_xl
    pos_out = pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]
    return pos_out * atom_mask[..., None]


class DiffusionModule(nn.Module):
    """AF3 Algorithm 20: Diffusion module.

    Full version with atom-level attention encoder/decoder matching the
    reference OpenFold3 DiffusionModule.

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_token: Token diffusion channel dimension
        c_s_input: Input single representation dimension
        sigma_data: Data variance constant
        no_diff_blocks: Number of diffusion transformer blocks
        no_diff_heads: Number of diffusion attention heads
        c_diff_hidden: Per-head hidden dim for diffusion attention
        n_diff_transition: Transition scale
        relpos_k: Relative position max index
        max_relative_chain: Max relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        atom_attn_n_query: Block height for atom attention
        atom_attn_n_key: Block width for atom attention
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_token: int = 768,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        no_diff_blocks: int = 24,
        no_diff_heads: int = 16,
        c_diff_hidden: int = 48,
        n_diff_transition: int = 2,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        atom_attn_n_query: int = 32,
        atom_attn_n_key: int = 128,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_token = c_token
        self.sigma_data = sigma_data

        self.diffusion_conditioning = DiffusionConditioning(
            c_s=c_s, c_z=c_z, c_s_input=c_s_input,
            sigma_data=sigma_data,
            relpos_k=relpos_k, max_relative_chain=max_relative_chain,
        )

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=True,
            c_s=c_s,
            c_z=c_z,
            c_hidden=32,
            no_heads=4,
            no_blocks=3,
            n_transition=2,
            n_query=atom_attn_n_query,
            n_key=atom_attn_n_key,
            use_ada_layer_norm=True,
        )

        self.diffusion_transformer = DiffusionTransformer(
            c_a=c_token, c_s=c_s, c_z=c_z,
            c_hidden=c_diff_hidden, no_heads=no_diff_heads,
            no_blocks=no_diff_blocks,
            n_transition=n_diff_transition,
            use_ada_layer_norm=True,
        )

        self.atom_attn_dec = AtomAttentionDecoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=32,
            no_heads=4,
            no_blocks=3,
            n_transition=2,
            n_query=atom_attn_n_query,
            n_key=atom_attn_n_key,
            use_ada_layer_norm=True,
        )

        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_token, bias=False)

        self.layer_norm_a = LayerNorm(c_token, create_offset=False)

    def forward(
        self,
        batch: dict,
        xl_noisy: torch.Tensor,
        token_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            batch:     Feature dictionary
            xl_noisy:  [*, N_atom, 3] noisy atom positions
            token_mask:[*, N_token] token mask
            atom_mask: [*, N_atom] atom mask
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            [*, N_atom, 3] denoised atom positions
        """
        si, zij = self.diffusion_conditioning(
            batch=batch, t=t,
            si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
            use_conditioning=use_conditioning,
        )

        xl_noisy = xl_noisy * atom_mask[..., None]

        rl_noisy = xl_noisy / torch.sqrt(t[..., None, None] ** 2 + self.sigma_data ** 2)

        ai, ql, cl, plm = self.atom_attn_enc(
            batch=batch,
            rl=rl_noisy,
            si_trunk=si,
            zij_trunk=zij,
        )

        ai = ai + self.linear_s(self.layer_norm_s(si))

        ai = self.diffusion_transformer(
            a=ai, s=si, z=zij, mask=token_mask,
        )

        ai = self.layer_norm_a(ai)

        rl_update = self.atom_attn_dec(
            batch=batch, ai=ai, ql=ql, cl=cl, plm=plm,
        )

        # EDM-style combination
        xl_out = (
            self.sigma_data ** 2
            / (self.sigma_data ** 2 + t[..., None, None] ** 2)
            * xl_noisy
            + self.sigma_data
            * t[..., None, None]
            / torch.sqrt(self.sigma_data ** 2 + t[..., None, None] ** 2)
            * rl_update
        )

        xl_out = xl_out * atom_mask[..., None]

        return xl_out


class SampleDiffusion(nn.Module):
    """AF3 Algorithm 18: Diffusion sampling.

    Args:
        gamma_0: Schedule controlling factor
        gamma_min: Minimum schedule threshold
        noise_scale: Noise scaling factor
        step_scale: Step scaling factor
        diffusion_module: Instantiated DiffusionModule
    """

    def __init__(
        self,
        gamma_0: float,
        gamma_min: float,
        noise_scale: float,
        step_scale: float,
        diffusion_module: DiffusionModule,
    ):
        super().__init__()
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.diffusion_module = diffusion_module

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        no_rollout_samples: int,
        use_conditioning: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            batch:           Feature dictionary
            si_input:        [*, N_token, c_s_input] input embedding
            si_trunk:        [*, N_token, c_s] trunk single rep
            zij_trunk:       [*, N_token, N_token, c_z] trunk pair rep
            noise_schedule:  [no_rollout_steps + 1] noise schedule
            no_rollout_samples: Number of diffusion samples

        Returns:
            [*, N_samples, N_atom, 3] sampled atom positions
        """
        atom_mask = batch["atom_mask"]
        batch_dim = atom_mask.shape[0]
        num_atoms = atom_mask.shape[-1]

        xl = noise_schedule[0] * torch.randn(
            (batch_dim, no_rollout_samples, num_atoms, 3),
            device=atom_mask.device, dtype=atom_mask.dtype,
        )

        for tau, c_tau in enumerate(noise_schedule[1:]):
            xl = centre_random_augmentation(xl=xl, atom_mask=atom_mask)

            gamma = self.gamma_0 if c_tau > self.gamma_min else 0
            t = noise_schedule[tau] * (gamma + 1)

            noise = (
                self.noise_scale
                * torch.sqrt(t ** 2 - noise_schedule[tau] ** 2)
                * torch.randn_like(xl)
            )
            xl_noisy = xl + noise

            xl_denoised = self.diffusion_module(
                batch=batch,
                xl_noisy=xl_noisy,
                token_mask=batch["token_mask"],
                atom_mask=atom_mask,
                t=t.to(xl_noisy.device),
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=use_conditioning,
            )

            delta = (xl_noisy - xl_denoised) / t
            dt = c_tau - t
            xl = xl_noisy + self.step_scale * dt * delta

        return xl
