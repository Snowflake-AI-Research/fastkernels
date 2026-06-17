"""TP-aware T5 encoder model for diffusion pipelines (L4).

Mirrors vllm-omni's T5EncoderModel built with TP-aware linear layers.
Used by FLUX.1-dev for the text_encoder_2 (T5-XXL).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import T5Config

from ..L1.t5_layer_norm import T5LayerNorm
from ..L3.t5_block import T5Block


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, shared: nn.Embedding):
        super().__init__()
        self.embed_tokens = shared
        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            extended_mask = (1.0 - extended_mask) * torch.finfo(hidden_states.dtype).min
        else:
            extended_mask = None

        position_bias = None
        for block in self.block:
            hidden_states, position_bias = block(
                hidden_states, mask=extended_mask, position_bias=position_bias,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5EncoderModel(nn.Module):
    """TP-aware T5 encoder model for diffusion pipelines."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.encoder(input_ids, attention_mask=attention_mask)
        return (hidden_states,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q", "q"),
            ("qkv_proj", "k", "k"),
            ("qkv_proj", "v", "v"),
            ("wi", "wi_0", 0),
            ("wi", "wi_1", 1),
        ]

        def _default_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            if "relative_attention_bias.weight" in name:
                lookup_name = name.replace(
                    "relative_attention_bias.weight",
                    "relative_attention_bias.emb.weight",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                lookup_name = name.replace(f".{weight_name}.", f".{param_name}.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
