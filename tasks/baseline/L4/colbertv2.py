"""ColBERT vLLM-compatible embedding model wiring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.l2_norm import L2Norm
from ..L1.linear import Linear
from ..L2.colbertv2_ops import ColBERTv2MaxSim, ColBERTv2ScoreReduce, ColBERTv2TokenMask
from ..L3.bert_model import BertModel


@dataclass
class ColBERTv2Config:
    model_type: str = "bert"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    dtype: torch.dtype = torch.bfloat16
    dim: int = 128

    @classmethod
    def from_pretrained(cls, model_name: str) -> "ColBERTv2Config":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cls(
            model_type=getattr(hf, "model_type", "bert"),
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
            position_embedding_type=getattr(hf, "position_embedding_type", "absolute"),
            dim=(
                getattr(hf, "colbert_dim", None)
                or getattr(hf, "dim", None)
                or getattr(hf, "projection_dim", 128)
            ),
        )


class ColBERTModel(nn.Module):
    is_pooling_model = True

    def __init__(self, config: ColBERTv2Config):
        super().__init__()
        self.config = config
        self.model = BertModel(config)
        self.colbert_linear = Linear(config.hidden_size, config.dim, bias=False)
        self.norm = L2Norm(dim=-1)
        self.token_mask = ColBERTv2TokenMask(config.pad_token_id)
        self.maxsim = ColBERTv2MaxSim()
        self.score_reducer = ColBERTv2ScoreReduce()

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
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.colbert_linear.weight.dtype)
        vecs = self.colbert_linear(hidden_states)
        if token_mask is not None:
            vecs = vecs * token_mask.unsqueeze(-1).to(dtype=vecs.dtype)
        return self.norm(vecs)

    def mask(
        self,
        input_ids: torch.Tensor,
        skiplist: set[int] | None = None,
    ) -> torch.Tensor:
        return self.token_mask(input_ids, skiplist=skiplist)

    def score(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.maxsim(query_vecs, doc_vecs, doc_mask)
