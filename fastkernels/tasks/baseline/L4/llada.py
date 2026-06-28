"""Standalone LLaDA masked diffusion language model.

First implementation targets single-GPU inference for `GSAI-ML/LLaDA-8B-Instruct`.
The runtime path is handled by ``infra/dllm_engine.py`` instead of the AR engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.embedding import Embedding
from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L2.parallel_linear import ReplicatedLinear
from ..L3.llada_block import LLaDABlock


@dataclass
class LLaDAConfig:
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    n_layers: int = 32
    mlp_hidden_size: int = 12288
    vocab_size: int = 126464
    embedding_size: int = 126464
    max_sequence_length: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_full_precision: bool = False
    include_bias: bool = False
    include_qkv_bias: bool = False
    weight_tying: bool = False
    mask_token_id: int = 126336
    pad_token_id: int = 126080
    eos_token_id: int = 126081
    dtype: torch.dtype = torch.bfloat16

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LLaDAConfig":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cls(
            d_model=hf.d_model,
            n_heads=hf.n_heads,
            n_kv_heads=hf.n_kv_heads or hf.n_heads,
            n_layers=hf.n_layers,
            mlp_hidden_size=hf.mlp_hidden_size,
            vocab_size=hf.vocab_size,
            embedding_size=getattr(hf, "embedding_size", hf.vocab_size),
            max_sequence_length=hf.max_sequence_length,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_theta,
            rope_full_precision=getattr(hf, "rope_full_precision", False),
            include_bias=hf.include_bias,
            include_qkv_bias=hf.include_qkv_bias,
            weight_tying=hf.weight_tying,
            mask_token_id=hf.mask_token_id,
            pad_token_id=hf.pad_token_id,
            eos_token_id=hf.eos_token_id,
        )


class LLaDAOutput(NamedTuple):
    logits: torch.Tensor
    hidden_states: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class LLaDAModel(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_sequence_length,
            config.rope_theta,
        )
        self.transformer = nn.ModuleDict(
            {
                "wte": Embedding(config.embedding_size, config.d_model),
                "blocks": nn.ModuleList(
                    [LLaDABlock(config, rotary_emb=self.rotary_emb) for _ in range(config.n_layers)]
                ),
                "ln_f": RMSNorm(config.d_model, eps=config.rms_norm_eps),
                "ff_out": ReplicatedLinear(
                    config.d_model,
                    config.embedding_size or config.vocab_size,
                    bias=config.include_bias,
                ),
            }
        )

    def rebuild_rope_cache_fp32(self) -> None:
        if not self.config.rope_full_precision:
            return
        device = self.rotary_emb.cos_sin_cache.device
        head_dim = self.rotary_emb.head_dim
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(
            self.config.max_sequence_length,
            device=device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        full_positions = torch.cat((freqs, freqs), dim=-1)
        pos_cos = full_positions.cos()[None, None, :, :]
        pos_sin = full_positions.sin()[None, None, :, :]
        half = head_dim // 2
        self.rotary_emb.cos_sin_cache = torch.cat(
            (pos_cos[0, 0, :, :half], pos_sin[0, 0, :, :half]),
            dim=-1,
        )
        self.rotary_emb.llada_pos_cos_cache = pos_cos
        self.rotary_emb.llada_pos_sin_cache = pos_sin

    @property
    def device(self) -> torch.device:
        return self.transformer["ln_f"].weight.device

    def _build_attention_bias(
        self,
        attention_mask: torch.Tensor | None,
        attention_bias: torch.Tensor | None,
        batch_size: int,
        past_length: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_bias is None and attention_mask is None:
            return None

        if attention_bias is None:
            attention_bias = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.float32)
        elif attention_bias.dtype != torch.float32:
            attention_bias = attention_bias.to(torch.float32)

        if attention_mask is not None:
            if attention_mask.dtype != torch.float32:
                attention_mask = attention_mask.to(torch.float32)
            if not bool(torch.all(attention_mask > 0.0)):
                mask = attention_mask.view(batch_size, 1, 1, seq_len)
                attention_bias = attention_bias[:, :, :seq_len, :seq_len] + (
                    (1.0 - mask) * torch.finfo(torch.float32).min
                )
            else:
                return None
        return attention_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        replace_position: torch.Tensor | None = None,
    ) -> LLaDAOutput:
        batch_size, seq_len = input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        attention_bias = self._build_attention_bias(
            attention_mask, attention_bias, batch_size, past_length, seq_len, input_ids.device,
        )

        hidden_states = self.transformer["wte"](input_ids)
        caches = [] if use_cache else None
        for block_idx, block in enumerate(self.transformer["blocks"]):
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            hidden_states, cache = block(
                hidden_states,
                attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                replace_position=replace_position,
            )
            if caches is not None:
                assert cache is not None
                caches.append(cache)
        hidden_states = self.transformer["ln_f"](hidden_states)
        logits = self.transformer["ff_out"](hidden_states).float()
        return LLaDAOutput(logits=logits, hidden_states=hidden_states, past_key_values=caches)


class LLaDAModelLM(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.config = config
        self.model = LLaDAModel(config)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        replace_position: torch.Tensor | None = None,
    ) -> LLaDAOutput:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            replace_position=replace_position,
        )
