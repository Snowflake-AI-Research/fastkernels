"""Standalone RWKV7 model implementation.

Matches FLA checkpoint weight names exactly:
  model.embeddings.weight
  model.layers.0.pre_norm.{weight,bias}
  model.layers.{i}.attn_norm.{weight,bias}
  model.layers.{i}.attn.x_{r,w,k,v,a,g}
  model.layers.{i}.attn.{r,k,v,o}_proj.weight
  model.layers.{i}.attn.k_k, k_a, r_k
  model.layers.{i}.attn.{w,a,g}_lora.lora.{0,2}.{weight,bias}
  model.layers.{i}.attn.v_lora.lora.{0,2}.{weight,bias}  (layers > 0)
  model.layers.{i}.attn.g_norm.{weight,bias}
  model.layers.{i}.ffn_norm.{weight,bias}
  model.layers.{i}.ffn.x_k
  model.layers.{i}.ffn.{key,value}.weight
  model.norm.{weight,bias}
  lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L3.rwkv7_decoder import RWKV7Block
from .recurrent_cache import CausalLMOutputWithPast, RecurrentCache


@dataclass
class RWKV7Config:
    hidden_size: int = 2560
    head_dim: int = 64
    num_heads: int | None = None
    num_hidden_layers: int = 32
    vocab_size: int = 65536
    decay_low_rank_dim: int = 96
    gate_low_rank_dim: int = 320
    a_low_rank_dim: int = 96
    v_low_rank_dim: int = 64
    hidden_ratio: float = 4.0
    intermediate_size: int | None = None
    norm_bias: bool = True
    norm_eps: float = 1e-5
    norm_first: bool = True
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.num_heads is None:
            self.num_heads = self.hidden_size // self.head_dim
        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * self.hidden_ratio)
            self.intermediate_size = 32 * ((intermediate + 31) // 32)

    @classmethod
    def from_dict(cls, data: dict) -> "RWKV7Config":
        keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "dtype"}
        kwargs = {k: data[k] for k in keys if k in data}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "RWKV7Config":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class RWKV7Model(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RWKV7Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LayerNorm(
            config.hidden_size, eps=config.norm_eps,
            create_offset=config.norm_bias,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RecurrentCache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, RecurrentCache | None]:
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = RecurrentCache()

        v_first = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states, v_first, _, past_key_values = layer(
                hidden_states,
                v_first,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        return self.norm(hidden_states), past_key_values


class RWKV7ForCausalLM(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.model = RWKV7Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.emb.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RecurrentCache | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        # When the engine only needs the last token's logits (every
        # generation call), restrict the lm_head + fp32 upcast to a
        # single position. For batched prefill at B=200, T=1024 with
        # vocab=65k this saves ~50 GB of fp32 logits memory and the
        # corresponding compute.
        if num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(
            logits=logits, past_key_values=past_key_values, loss=loss,
        )
