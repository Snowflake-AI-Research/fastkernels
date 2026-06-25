"""OpenFold3 / AlphaFold3 structure prediction pipeline (L4 pipeline).

Contains:
- OpenFold3Config: model configuration dataclass.
- OpenFold3Model: full AF3 model wiring input embedder, MSA module,
  PairFormer, diffusion module, and auxiliary heads.
- load_openfold3_checkpoint: weight remapping from OpenFold3 HF checkpoint.

L4 wiring/configuration; computation lives in L1-L3 tasks.

Reference: openfold3/projects/of3_all_atom/model.py OpenFold3
           openfold3/projects/of3_all_atom/config/model_config.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.alphafold3_input_embedder import InputEmbedder
from ..L2.alphafold3_msa_module_embedder import MSAModuleEmbedder
from ..L2.alphafold3_template_embedder import TemplateEmbedder
from ..L3.alphafold3_diffusion_module import DiffusionModule, SampleDiffusion
from ..L3.alphafold3_heads import AuxiliaryHeads
from ..L3.alphafold3_msa_module import MSAModuleStack
from ..L3.alphafold3_pairformer import PairFormerStack


@dataclass
class OpenFold3Config:
    """Configuration for AlphaFold3 model.

    Default values match openfold3/projects/of3_all_atom/config/model_config.py.
    """

    # Channel dimensions
    c_s: int = 384
    c_z: int = 128
    c_m: int = 64
    c_atom: int = 128
    c_atom_pair: int = 16
    c_token_embedder: int = 384
    c_token_diffusion: int = 768
    c_s_input: int = 449
    c_t: int = 64
    max_atoms_per_token: int = 23

    # PairFormer
    pairformer_no_blocks: int = 48
    pairformer_c_hidden_mul: int = 128
    pairformer_c_hidden_pair_att: int = 32
    pairformer_no_heads_pair: int = 4
    pairformer_c_hidden_pair_bias: int = 24
    pairformer_no_heads_pair_bias: int = 16
    pairformer_transition_n: int = 4
    pairformer_pair_dropout: float = 0.0

    # MSA module
    msa_no_blocks: int = 4
    msa_c_hidden_msa_att: int = 8
    msa_c_hidden_opm: int = 32
    msa_c_hidden_mul: int = 128
    msa_c_hidden_pair_att: int = 32
    msa_no_heads_msa: int = 8
    msa_no_heads_pair: int = 4
    msa_transition_n: int = 4
    msa_opm_first: bool = True
    msa_dropout: float = 0.0
    msa_pair_dropout: float = 0.0

    # Diffusion module
    diff_no_blocks: int = 24
    diff_no_heads: int = 16
    diff_c_hidden: int = 48
    diff_n_transition: int = 2
    sigma_data: float = 16.0

    # Noise schedule
    no_rollout_steps: int = 200
    s_max: float = 160.0
    s_min: float = 4e-4
    noise_schedule_p: int = 7

    # Sampling
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    # Recycling
    num_recycles: int = 3

    # Input
    relpos_k: int = 32
    max_relative_chain: int = 2

    # Atom attention
    atom_attn_n_query: int = 32
    atom_attn_n_key: int = 128

    @classmethod
    def from_pretrained(cls, model_name: str) -> "OpenFold3Config":
        """Create config for known OpenFold3 models."""
        return cls()


def _create_noise_schedule(
    no_rollout_steps: int,
    sigma_data: float,
    s_max: float,
    s_min: float,
    p: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """AF3 noise schedule (Page 24 of supplement)."""
    t = torch.arange(0, 1 + no_rollout_steps, dtype=dtype, device=device) / no_rollout_steps
    return sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p


class OpenFold3Model(nn.Module):
    """Full AlphaFold3 / OpenFold3 model.

    Wires input embedder -> template embedder -> MSA module -> PairFormer
    -> diffusion module -> auxiliary heads.

    Reference: openfold3/projects/of3_all_atom/model.py OpenFold3

    Args:
        config: OpenFold3Config with all hyperparameters.
    """

    def __init__(self, config: OpenFold3Config):
        super().__init__()
        self.config = config

        # Input embedding (includes AtomAttentionEncoder)
        self.input_embedder = InputEmbedder(
            c_s_input=config.c_s_input,
            c_s=config.c_s,
            c_z=config.c_z,
            relpos_k=config.relpos_k,
            max_relative_chain=config.max_relative_chain,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token_embedder,
        )

        # Recycle projections (z first, then s, matching reference ordering)
        self.layer_norm_z = LayerNorm(config.c_z)
        self.linear_z = Linear(config.c_z, config.c_z, bias=False)

        # Template embedder
        self.template_embedder = TemplateEmbedder(
            c_t=config.c_t,
            c_z=config.c_z,
        )

        # MSA module embedder
        self.msa_module_embedder = MSAModuleEmbedder(
            c_m_feats=34,
            c_m=config.c_m,
            c_s_input=config.c_s_input,
        )

        # MSA module
        self.msa_module = MSAModuleStack(
            c_m=config.c_m,
            c_z=config.c_z,
            c_hidden_msa_att=config.msa_c_hidden_msa_att,
            c_hidden_opm=config.msa_c_hidden_opm,
            c_hidden_mul=config.msa_c_hidden_mul,
            c_hidden_pair_att=config.msa_c_hidden_pair_att,
            no_heads_msa=config.msa_no_heads_msa,
            no_heads_pair=config.msa_no_heads_pair,
            no_blocks=config.msa_no_blocks,
            transition_n=config.msa_transition_n,
            msa_dropout=config.msa_dropout,
            pair_dropout=config.msa_pair_dropout,
            opm_first=config.msa_opm_first,
        )

        # Recycle single projection (after template/MSA, before pairformer)
        self.layer_norm_s = LayerNorm(config.c_s)
        self.linear_s = Linear(config.c_s, config.c_s, bias=False)

        # PairFormer stack
        self.pairformer_stack = PairFormerStack(
            c_s=config.c_s,
            c_z=config.c_z,
            c_hidden_pair_bias=config.pairformer_c_hidden_pair_bias,
            no_heads_pair_bias=config.pairformer_no_heads_pair_bias,
            c_hidden_mul=config.pairformer_c_hidden_mul,
            c_hidden_pair_att=config.pairformer_c_hidden_pair_att,
            no_heads_pair=config.pairformer_no_heads_pair,
            no_blocks=config.pairformer_no_blocks,
            transition_n=config.pairformer_transition_n,
            pair_dropout=config.pairformer_pair_dropout,
        )

        # Diffusion module (includes AtomAttentionEncoder/Decoder)
        self.diffusion_module = DiffusionModule(
            c_s=config.c_s,
            c_z=config.c_z,
            c_token=config.c_token_diffusion,
            c_s_input=config.c_s_input,
            sigma_data=config.sigma_data,
            no_diff_blocks=config.diff_no_blocks,
            no_diff_heads=config.diff_no_heads,
            c_diff_hidden=config.diff_c_hidden,
            n_diff_transition=config.diff_n_transition,
            relpos_k=config.relpos_k,
            max_relative_chain=config.max_relative_chain,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            atom_attn_n_query=config.atom_attn_n_query,
            atom_attn_n_key=config.atom_attn_n_key,
        )

        self.sample_diffusion = SampleDiffusion(
            gamma_0=config.gamma_0,
            gamma_min=config.gamma_min,
            noise_scale=config.noise_scale,
            step_scale=config.step_scale,
            diffusion_module=self.diffusion_module,
        )

        # Auxiliary heads (includes PairformerEmbedding + ExperimentallyResolved)
        self.aux_heads = AuxiliaryHeads(
            c_s=config.c_s,
            c_z=config.c_z,
            c_s_input=config.c_s_input,
            max_atoms_per_token=config.max_atoms_per_token,
        )

    def run_trunk(
        self,
        batch: dict,
        num_recycles: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the trunk: input embedding -> MSA module -> PairFormer.

        Args:
            batch: Feature dictionary with token_features, residue_index,
                   msa (optional), token_mask, pair_mask, msa_mask.
            num_recycles: Number of recycling iterations.

        Returns:
            (s_input, s, z):
                s_input: [*, N_token, C_s_input]
                s: [*, N_token, C_s]
                z: [*, N_token, N_token, C_z]
        """
        if num_recycles is None:
            num_recycles = self.config.num_recycles

        token_features = batch["token_features"]
        residue_index = batch["residue_index"]
        token_mask = batch["token_mask"]
        pair_mask = batch.get("pair_mask", token_mask[..., :, None] * token_mask[..., None, :])

        s_input, s_init, z_init = self.input_embedder(
            token_features, residue_index, batch=batch,
        )

        s = s_init
        z = z_init

        for cycle in range(num_recycles + 1):
            if cycle > 0:
                z = z_init + self.linear_z(self.layer_norm_z(z))
                s = s_init + self.linear_s(self.layer_norm_s(s))

            # Template embedder
            if "template_distogram" in batch:
                t_emb = self.template_embedder(
                    batch=batch, z=z, pair_mask=pair_mask,
                )
                z = z + t_emb

            # MSA module
            if "msa" in batch and batch["msa"] is not None:
                m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)
                _, z = self.msa_module(
                    m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask,
                )

            # PairFormer
            s, z = self.pairformer_stack(
                s=s, z=z,
                single_mask=token_mask,
                pair_mask=pair_mask,
            )

        return s_input, s, z

    def forward(self, batch: dict) -> tuple[dict, dict]:
        """Full forward pass.

        Args:
            batch: Feature dictionary. Required keys:
                - token_features: [*, N_token, C_token] per-token features
                - residue_index:  [*, N_token] residue indices
                - token_mask:     [*, N_token] token mask
                - atom_mask:      [*, N_atom] atom mask
                Optional:
                - msa:       [*, N_seq, N_res, C_m] MSA features
                - msa_mask:  [*, N_seq, N_res] MSA mask
                - pair_mask: [*, N_token, N_token] pair mask

        Returns:
            (outputs, aux_outputs):
                outputs: dict with atom_positions_predicted, head logits
                aux_outputs: dict with trunk representations
        """
        s_input, s, z = self.run_trunk(batch)

        # Auxiliary heads
        head_outputs = self.aux_heads(s=s, z=z)

        # Diffusion sampling
        noise_schedule = _create_noise_schedule(
            no_rollout_steps=self.config.no_rollout_steps,
            sigma_data=self.config.sigma_data,
            s_max=self.config.s_max,
            s_min=self.config.s_min,
            p=self.config.noise_schedule_p,
            dtype=s.dtype,
            device=s.device,
        )

        with torch.no_grad():
            atom_positions = self.sample_diffusion(
                batch=batch,
                si_input=s_input,
                si_trunk=s,
                zij_trunk=z,
                noise_schedule=noise_schedule,
                no_rollout_samples=1,
                use_conditioning=True,
            )

        outputs = {
            "atom_positions_predicted": atom_positions,
            **head_outputs,
        }

        aux_outputs = {
            "s_input": s_input,
            "s_trunk": s,
            "z_trunk": z,
        }

        return outputs, aux_outputs


def load_openfold3_checkpoint(
    model: OpenFold3Model, checkpoint_path: str,
) -> None:
    """Load an OpenFold3 checkpoint into a fastkernels OpenFold3Model.

    The fastkernels model architecture is a 1:1 match with the reference
    OpenFold/OpenFold3 checkpoint, so strict loading is used.

    Args:
        model: fastkernels OpenFold3Model instance.
        checkpoint_path: Path to the checkpoint file.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
