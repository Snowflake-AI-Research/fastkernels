"""BGE-M3 vLLM-compatible embedding model wiring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.l2_norm import L2Norm
from ..L1.linear import Linear
from ..L1.relu import ReLU
from ..L3.xlm_roberta_model import XLMRobertaModel


@dataclass
class BGEM3Config:
    model_type: str = "xlm-roberta"
    vocab_size: int = 250002
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 8194
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: str = "absolute"
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "BGEM3Config":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cls(
            model_type=getattr(hf, "model_type", "xlm-roberta"),
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            intermediate_size=hf.intermediate_size,
            max_position_embeddings=hf.max_position_embeddings,
            type_vocab_size=hf.type_vocab_size,
            layer_norm_eps=hf.layer_norm_eps,
            hidden_act=hf.hidden_act,
            hidden_dropout_prob=hf.hidden_dropout_prob,
            attention_probs_dropout_prob=hf.attention_probs_dropout_prob,
            pad_token_id=hf.pad_token_id,
            bos_token_id=getattr(hf, "bos_token_id", 0),
            eos_token_id=getattr(hf, "eos_token_id", 2),
            position_embedding_type=getattr(hf, "position_embedding_type", "absolute"),
        )


class BgeM3EmbeddingModel(nn.Module):
    is_pooling_model = True

    def __init__(self, config: BGEM3Config):
        super().__init__()
        self.config = config
        self.model = XLMRobertaModel(config)
        self.sparse_linear = Linear(config.hidden_size, 1, bias=True)
        self.colbert_linear = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.relu = ReLU()
        self.norm = L2Norm(dim=-1)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def forward_with_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.forward_with_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

    def forward_varlen(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.forward_varlen(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def token_embed(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.colbert_linear.weight.dtype)
        vecs = self.colbert_linear(hidden_states[:, 1:])
        if attention_mask is not None:
            vecs = vecs * attention_mask[:, 1:].unsqueeze(-1).to(dtype=vecs.dtype)
        return self.norm(vecs)

    def token_classify(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.sparse_linear.weight.dtype)
        weights = self.relu(self.sparse_linear(hidden_states))
        if attention_mask is not None:
            weights = weights * attention_mask.unsqueeze(-1).to(dtype=weights.dtype)
        return weights
