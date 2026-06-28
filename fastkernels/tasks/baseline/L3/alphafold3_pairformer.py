"""PairFormer stack for AlphaFold3.

48-block PairFormer: each block runs a PairBlock on pair (z) then
AttentionPairBias + SwiGLUTransition on single (s).

Reference: openfold3/core/model/latent/pairformer.py PairFormerStack
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition


class PairFormerBlock(nn.Module):
    """Single block of AF3 Algorithm 17.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

        self.attn_pair_bias = AttentionPairBias(
            c_q=c_s, c_k=c_s, c_v=c_s,
            c_s=c_s, c_z=c_z,
            c_hidden=c_hidden_pair_bias,
            no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False,
            gating=True,
            inf=inf,
        )

        self.single_transition = SwiGLUTransition(c_in=c_s, n=transition_n)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        single_trans_mask = single_mask if _mask_trans else None

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)

        s = s + self.single_transition(s, mask=single_trans_mask)

        return s, z


class PairFormerStack(nn.Module):
    """AF3 Algorithm 17: PairFormer stack.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of PairFormer blocks
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                c_s=c_s, c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        for block in self.blocks:
            s, z = block(
                s=s, z=z,
                single_mask=single_mask,
                pair_mask=pair_mask,
            )

        return s, z
