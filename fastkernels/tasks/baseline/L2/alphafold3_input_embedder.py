"""Input embedder for AlphaFold3.

Produces initial single (s) and pair (z) representations from token and
atom features.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           InputEmbedderAllAtom
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import OneHot, Pad
from .alphafold3_atom_attention import AtomAttentionEncoder


def _binned_one_hot(
    x: torch.Tensor, boundaries: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with bin boundaries (matches reference binned_one_hot)."""
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Build relative position features matching the reference implementation.

    Produces 139 features when max_relative_idx=32, max_relative_chain=2:
      66 (rel_pos) + 66 (rel_token) + 1 (same_entity) + 6 (rel_chain)

    Reference: openfold3/core/utils/relpos.py relpos_complex
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int,
    ) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device,
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Matches InputEmbedderAllAtom: runs AtomAttentionEncoder to get a
    token-level representation, concatenates with restype/profile/deletion_mean
    to form s_input (449 dims), then projects to s and z.

    Args:
        c_s_input: Input single representation dimension (449 for all-atom)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative residue position
        max_relative_chain: Maximum relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        c_token: Token dim for atom attention encoder output
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int | None = None,
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain
        self._one_hot = OneHot()
        self._pad = Pad()

        if c_token is None:
            c_token = c_s

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, bias=False)
        self.linear_z_i = Linear(c_s_input, c_z, bias=False)
        self.linear_z_j = Linear(c_s_input, c_z, bias=False)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = Linear(n_relpos_features, c_z, bias=False)

        self.linear_token_bonds = Linear(1, c_z, bias=False)

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input] per-token features.
                If batch contains ref_pos (atom features), only restype/profile/deletion_mean
                are expected here and atom_attn_enc produces the remaining features.
                Otherwise, treated as pre-built s_input.
            residue_index:  [*, N_token] residue indices
            batch: Feature dict for relpos and atom attention.

        Returns:
            s_input: [*, N_token, c_s_input] input single representation
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        if batch is not None and "ref_pos" in batch:
            a, _, _, _ = self.atom_attn_enc(batch=batch)
            s_input = torch.cat(
                [
                    a,
                    batch.get("restype", token_features[..., :32]),
                    batch.get("profile", token_features[..., 32:64]),
                    batch.get("deletion_mean", token_features[..., -1:]).unsqueeze(-1)
                    if batch.get("deletion_mean") is not None and batch["deletion_mean"].dim() == token_features.dim() - 1
                    else batch.get("deletion_mean", token_features[..., -1:]),
                ],
                dim=-1,
            )
        else:
            s_input = token_features

        s = self.linear_s(s_input)

        z_i = self.linear_z_i(s_input)[..., :, None, :]
        z_j = self.linear_z_j(s_input)[..., None, :, :]
        z = z_i + z_j

        if batch is not None and "asym_id" in batch:
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.relpos_k,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
        else:
            d = residue_index[..., :, None] - residue_index[..., None, :]
            d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
            n_bins = 2 * self.relpos_k + 2
            relpos_feats = self._one_hot(d.long(), n_bins).to(
                dtype=z.dtype,
            )
            n_relpos_in = self.linear_relpos.weight.shape[-1]
            if relpos_feats.shape[-1] < n_relpos_in:
                pad_size = n_relpos_in - relpos_feats.shape[-1]
                relpos_feats = self._pad(relpos_feats, (0, pad_size))

        z = z + self.linear_relpos(relpos_feats)

        if batch is not None and "token_bonds" in batch:
            token_bonds_emb = self.linear_token_bonds(
                batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype)
            )
            z = z + token_bonds_emb

        return s_input, s, z
