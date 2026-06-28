from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm
from .mla_attention_impl import MLAAttention
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class KimiMLAAttention(nn.Module):
    """Kimi MLA path matching vLLM's latent-attention formulation."""

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // tp
        self.scaling = self.qk_head_dim ** -0.5

        assert self.q_lora_rank is None
        assert getattr(config, "mla_use_nope", True)

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            is_sparse=False,
        )
        object.__setattr__(self.attn, "_kv_b_proj", self.kv_b_proj)

    def compute_absorbed_weights(self):
        """Compute absorbed MLA decode weights from ``kv_b_proj``."""
        weight = self.kv_b_proj.weight.data
        if hasattr(self.kv_b_proj, "use_fp8") and self.kv_b_proj.use_fp8:
            scale = self.kv_b_proj.weight_scale_inv.data
            weight = self._dequant_fp8_block(weight, scale)
        else:
            weight = weight.to(torch.bfloat16)

        weight = weight.T
        latent = self.kv_lora_rank
        heads = self.num_local_heads
        nope = self.qk_nope_head_dim
        value = self.v_head_dim
        weight = weight.view(latent, heads, nope + value)
        w_uk = weight[:, :, :nope]
        w_uv = weight[:, :, nope:]
        self.attn.W_UV = w_uv.permute(1, 0, 2).contiguous()
        self.attn.W_UK_T = w_uk.permute(1, 2, 0).contiguous()

    @staticmethod
    def _dequant_fp8_block(
        w_fp8: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        import math

        n, k = w_fp8.shape
        sn = math.ceil(n / block_size)
        sk = math.ceil(k / block_size)
        scale = scale_inv[:sn, :sk]
        scale_expanded = scale.repeat_interleave(block_size, dim=0)[:n]
        scale_expanded = scale_expanded.repeat_interleave(block_size, dim=1)[:, :k]
        return (w_fp8.float() * scale_expanded).to(torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        del positions, state_manager
        num_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        q = q.view(num_tokens, self.num_local_heads, self.qk_head_dim)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)

        attn_output = self.attn(
            q,
            kv_c,
            k_pe,
            output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        return self.o_proj(attn_output)
