"""EAGLE-3 draft model for Llama-3.x family.

Mirrors sglang's ``LlamaForCausalLMEagle3`` (apache-2.0):
- ``fc`` linear fuses three target hidden states into one draft hidden.
- A single ``midlayer`` Llama decoder block where ``qkv_proj`` consumes the
  concat ``[input_layernorm(embeds) | hidden_norm(hidden_states)]`` (in_features
  = ``2 * hidden_size``).
- ``lm_head`` operates over a smaller ``draft_vocab_size`` and the ``d2t``
  buffer maps draft ids back to the target vocab.

The draft model's ``embed_tokens`` is sized over the full target vocab and is
expected to be tied with the target's embedding table at engine init time via
``set_embed_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L2.parallel_linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from ..L1.silu_and_mul import SiluAndMul
from ..L2.attention_impl import Attention


@dataclass
class LlamaEagle3Config:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128256
    draft_vocab_size: int = 32000
    target_hidden_size: int = 4096
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192
    dtype: torch.dtype = torch.bfloat16
    qkv_bias: bool = False
    num_hidden_layers: int = 1
    tie_word_embeddings: bool = False

    @classmethod
    def from_pretrained(cls, draft_repo: str, target_config) -> "LlamaEagle3Config":
        hf = AutoConfig.from_pretrained(draft_repo)

        target_hidden_size = getattr(hf, "target_hidden_size", hf.hidden_size)
        draft_vocab_size = getattr(hf, "draft_vocab_size", hf.vocab_size)

        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            vocab_size=hf.vocab_size,
            draft_vocab_size=draft_vocab_size,
            target_hidden_size=target_hidden_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=target_config.rope_theta,
            rope_scaling_factor=target_config.rope_scaling_factor,
            rope_low_freq_factor=target_config.rope_low_freq_factor,
            rope_high_freq_factor=target_config.rope_high_freq_factor,
            rope_original_max_position_embeddings=(
                target_config.rope_original_max_position_embeddings
            ),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )


class _Eagle3Attention(nn.Module):
    """Attention layer where qkv_proj input dim = 2 * hidden_size.

    Otherwise identical to a standard Llama GQA attention with RoPE.
    """

    def __init__(self, config: LlamaEagle3Config, rotary_emb: nn.Module):
        super().__init__()
        from ...baseline.L1.rms_norm import RMSNorm  # noqa: F401  (kept for parity)

        from ....infra.tp import _tp_size
        tp = _tp_size()
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.rotary_emb = rotary_emb

        self.qkv_proj = QKVParallelLinear(
            2 * config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.attn = Attention(
            self.num_heads, config.head_dim, self.scale,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        N = hidden_states.shape[0]
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


class _Eagle3MLP(nn.Module):
    def __init__(self, config: LlamaEagle3Config):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)


class _Eagle3MidLayer(nn.Module):
    def __init__(self, config: LlamaEagle3Config, rotary_emb: nn.Module):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = _Eagle3Attention(config, rotary_emb=rotary_emb)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _Eagle3MLP(config)

    def forward(self, positions, embeds, hidden_states):
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaEagle3Model(nn.Module):
    def __init__(self, config: LlamaEagle3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_original_max_position_embeddings=config.rope_original_max_position_embeddings,
        )
        self.fc = nn.Linear(
            config.target_hidden_size * 3,
            config.hidden_size,
            bias=False,
        )
        self.midlayer = _Eagle3MidLayer(config, rotary_emb=self.rotary_emb)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, hidden_states):
        embeds = self.embed_tokens(input_ids)
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        hidden_states, residual = self.midlayer(positions, embeds, hidden_states)

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual,
        )
        return hidden_states_to_logits, hidden_states_to_aux


class LlamaForCausalLMEagle3(nn.Module):
    """EAGLE-3 draft model (single midlayer + fc fusion + d2t remap)."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaEagle3Config):
        super().__init__()
        self.config = config
        self.model = LlamaEagle3Model(config)

        self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size)

        # Draft-to-target token id remapping. Populated from the d2t buffer in
        # the checkpoint as ``hot_token_id = d2t + arange(draft_vocab_size)``.
        self.register_buffer(
            "hot_token_id",
            torch.empty(config.draft_vocab_size, dtype=torch.long),
            persistent=False,
        )
        self._has_hot_token_id = False

    def set_embed_tokens(self, target_embed: VocabParallelEmbedding):
        """Tie this draft's input embedding to the target's embedding table.

        sglang materializes a fresh ``embed_tokens`` from the target weights on
        every draft step; tying once at init is equivalent and saves memory.
        """
        self.model.embed_tokens = target_embed

    def forward_draft(self, input_ids, positions, hidden_states):
        """Run the single-layer draft forward.

        Returns ``(hidden_to_logits, hidden_to_next)`` where ``hidden_to_next``
        is the pre-norm hidden state used as the input to the next draft step.
        """
        return self.model(input_ids, positions, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states).float()

    def remap_draft_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocab ids back to target-vocab ids."""
        if not self._has_hot_token_id:
            return draft_ids
        return self.hot_token_id.to(draft_ids.device)[draft_ids]
