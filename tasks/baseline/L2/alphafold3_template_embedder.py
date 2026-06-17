"""Template embedder for AlphaFold3 (Algorithm 16).

TemplatePairEmbedder -> TemplatePairStack -> average -> ReLU -> linear_t.

Reference: openfold3/core/model/feature_embedders/template_embedders.py
           openfold3/core/model/latent/template_module.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.relu import ReLU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_pair_block import PairBlock
from .alphafold3_swiglu_transition import SwiGLUTransition


class TemplatePairEmbedder(nn.Module):
    """Embeds template pair features (Algorithm 16, lines 1-5).

    Args:
        c_in: Pair embedding input dimension (c_z)
        c_dgram: Distogram feature dim
        c_aatype: Template aatype feature dim
        c_out: Output template embedding dim (c_t)
    """

    def __init__(
        self,
        c_in: int = 128,
        c_dgram: int = 39,
        c_aatype: int = 32,
        c_out: int = 64,
    ):
        super().__init__()
        self.dgram_linear = Linear(c_dgram, c_out, bias=False)
        self.aatype_linear_1 = Linear(c_aatype, c_out, bias=False)
        self.aatype_linear_2 = Linear(c_aatype, c_out, bias=False)
        self.pseudo_beta_mask_linear = Linear(1, c_out, bias=False)
        self.x_linear = Linear(1, c_out, bias=False)
        self.y_linear = Linear(1, c_out, bias=False)
        self.z_linear = Linear(1, c_out, bias=False)
        self.backbone_mask_linear = Linear(1, c_out, bias=False)

        self.layer_norm_z = LayerNorm(c_in)
        self.linear_z = Linear(c_in, c_out, bias=False)

    def forward(
        self, batch: dict, z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            batch: needs template_distogram, template_restype,
                   template_pseudo_beta_mask, template_unit_vector,
                   template_backbone_frame_mask, asym_id
            z: [*, N_token, N_token, C_z]

        Returns:
            [*, N_templ, N_token, N_token, C_t]
        """
        dtype = batch["template_unit_vector"].dtype

        multichain_pair_mask = (
            batch["asym_id"][..., None] == batch["asym_id"][..., None, :]
        )
        multichain_pair_mask = multichain_pair_mask[..., None, :, :, None]

        pseudo_beta_pair_mask = (
            batch["template_pseudo_beta_mask"][..., None]
            * batch["template_pseudo_beta_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        backbone_frame_pair_mask = (
            batch["template_backbone_frame_mask"][..., None]
            * batch["template_backbone_frame_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_unit_vector = batch["template_unit_vector"]
        x, y, z_coord = template_unit_vector.unbind(dim=-1)

        template_restype = batch["template_restype"]
        n_token = template_restype.shape[-2]
        template_restype_ti = template_restype[..., None, :].expand(
            *template_restype.shape[:-2], -1, n_token, -1
        )
        template_restype_tj = template_restype[..., None, :, :].expand(
            *template_restype.shape[:-2], n_token, -1, -1
        )

        t = (
            self.dgram_linear(batch["template_distogram"].to(dtype=dtype))
            * pseudo_beta_pair_mask
        )
        t = t + self.aatype_linear_1(template_restype_ti.to(dtype=dtype))
        t = t + self.aatype_linear_2(template_restype_tj.to(dtype=dtype))
        t = t + (
            self.pseudo_beta_mask_linear(
                batch["template_pseudo_beta_mask"][..., None, None].to(dtype=dtype).expand(
                    *batch["template_pseudo_beta_mask"].shape, n_token, 1
                )
            )
            * multichain_pair_mask
        )
        t = t + self.x_linear(x[..., None]) * backbone_frame_pair_mask
        t = t + self.y_linear(y[..., None]) * backbone_frame_pair_mask
        t = t + self.z_linear(z_coord[..., None]) * backbone_frame_pair_mask
        t = t + (
            self.backbone_mask_linear(
                batch["template_backbone_frame_mask"][..., None, None].to(dtype=dtype).expand(
                    *batch["template_backbone_frame_mask"].shape, n_token, 1
                )
            )
            * multichain_pair_mask
        )

        z_emb = self.linear_z(self.layer_norm_z(z))
        t = t + z_emb[..., None, :, :, :]

        return t


class TemplatePairStackBlock(PairBlock):
    """Single block of the template pair stack.

    Inherits directly from PairBlock to avoid extra key nesting.
    """

    def __init__(
        self,
        c_t: int = 64,
        c_hidden_tri_att: int = 16,
        c_hidden_tri_mul: int = 64,
        no_heads: int = 4,
        transition_n: int = 2,
        dropout_rate: float = 0.25,
        tri_mul_first: bool = True,
        inf: float = 1e9,
    ):
        super().__init__(
            c_z=c_t,
            c_hidden_mul=c_hidden_tri_mul,
            c_hidden_pair_att=c_hidden_tri_att,
            no_heads_pair=no_heads,
            transition_n=transition_n,
            pair_dropout=dropout_rate,
            inf=inf,
        )


class TemplatePairStack(nn.Module):
    """Template pair stack.

    Args:
        c_t: Template pair dim
        c_hidden_tri_att: Triangle attention hidden dim
        c_hidden_tri_mul: Triangle multiplication hidden dim
        no_blocks: Number of blocks
        no_heads: Number of attention heads
        transition_n: Transition block scale
        dropout_rate: Dropout rate
        tri_mul_first: Triangle multiplication before attention
    """

    def __init__(
        self,
        c_t: int = 64,
        c_hidden_tri_att: int = 16,
        c_hidden_tri_mul: int = 64,
        no_blocks: int = 2,
        no_heads: int = 4,
        transition_n: int = 2,
        dropout_rate: float = 0.25,
        tri_mul_first: bool = True,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemplatePairStackBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                transition_n=transition_n,
                dropout_rate=dropout_rate,
                tri_mul_first=tri_mul_first,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])
        self.layer_norm = LayerNorm(c_t)

    def forward(
        self,
        t: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        n_templ = t.shape[-4]
        for block in self.blocks:
            for i in range(n_templ):
                t[..., i, :, :, :] = block(
                    z=t[..., i, :, :, :], pair_mask=pair_mask[..., 0, :, :],
                )
        return self.layer_norm(t)


class TemplateEmbedder(nn.Module):
    """AF3 Algorithm 16: Template embedder.

    Args:
        c_t: Template pair embedding dim
        c_z: Pair representation dim
        c_dgram: Distogram feature dim
        c_aatype: Template aatype feature dim
        c_hidden_tri_att: Triangle attention hidden dim
        c_hidden_tri_mul: Triangle multiplication hidden dim
        no_blocks: Number of pair stack blocks
        no_heads: Number of attention heads
        transition_n: Transition scale
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        c_t: int = 64,
        c_z: int = 128,
        c_dgram: int = 39,
        c_aatype: int = 32,
        c_hidden_tri_att: int = 16,
        c_hidden_tri_mul: int = 64,
        no_blocks: int = 2,
        no_heads: int = 4,
        transition_n: int = 2,
        dropout_rate: float = 0.25,
        tri_mul_first: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.relu = ReLU()
        self.template_pair_embedder = TemplatePairEmbedder(
            c_in=c_z, c_dgram=c_dgram, c_aatype=c_aatype, c_out=c_t,
        )
        self.template_pair_stack = TemplatePairStack(
            c_t=c_t,
            c_hidden_tri_att=c_hidden_tri_att,
            c_hidden_tri_mul=c_hidden_tri_mul,
            no_blocks=no_blocks,
            no_heads=no_heads,
            transition_n=transition_n,
            dropout_rate=dropout_rate,
            tri_mul_first=tri_mul_first,
            inf=inf,
        )
        self.linear_t = Linear(c_t, c_z, bias=False)

    def forward(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            batch: Template features
            z: [*, N_token, N_token, C_z] pair embedding
            pair_mask: [*, N_token, N_token] pair mask

        Returns:
            [*, N_token, N_token, C_z] template embedding added to pair rep
        """
        template_embeds = self.template_pair_embedder(batch, z)
        n_templ = template_embeds.shape[-4]

        pair_mask_4d = pair_mask[..., None, :, :].to(dtype=z.dtype)

        t = self.template_pair_stack(
            template_embeds, pair_mask_4d,
        )

        t = torch.sum(t, dim=-4) / n_templ
        t = self.relu(t)
        t = self.linear_t(t)

        return t
