"""Standalone Mamba2 model implementation (Codestral / SSD architecture).

Follows kb_nano's ``LlamaForCausalLM`` interface:
``forward(input_ids, positions)`` returning hidden states; ``compute_logits``
performs the LM-head projection separately.

Per-batch SSM state and metadata (state slot indices, prefill/decode
split, chunk metadata, ...) are read from the global ``Context``
(``infra/context.py``) by the mixer; the engine populates them via
``set_forward_context``.

Matches HuggingFace mamba2 / Codestral checkpoint weight names exactly:
  backbone.embeddings.weight                    [vocab_size, hidden_size]
  backbone.layers.{i}.norm.weight               [hidden_size]
  backbone.layers.{i}.mixer.in_proj.weight      [in_proj_size, hidden_size]
  backbone.layers.{i}.mixer.conv1d.weight       [conv_dim, 1, conv_kernel]
  backbone.layers.{i}.mixer.conv1d.bias         [conv_dim]
  backbone.layers.{i}.mixer.A_log               [num_heads]
  backbone.layers.{i}.mixer.D                   [num_heads]
  backbone.layers.{i}.mixer.dt_bias             [num_heads]
  backbone.layers.{i}.mixer.norm.weight         [intermediate_size]
  backbone.layers.{i}.mixer.out_proj.weight     [hidden_size, intermediate_size]
  backbone.norm_f.weight                        [hidden_size]
  lm_head.weight                                [vocab_size, hidden_size]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.mamba2_decoder import Mamba2DecoderLayer


@dataclass
class Mamba2Config:
    model_type: str = "mamba2"
    hidden_size: int = 4096
    num_hidden_layers: int = 64
    intermediate_size: int = 8192
    state_size: int = 128
    conv_kernel: int = 4
    expand: int = 2
    n_groups: int = 8
    num_heads: int = 128
    head_dim: int = 64
    chunk_size: int = 256
    vocab_size: int = 32768
    use_bias: bool = False
    use_conv_bias: bool = True
    tie_word_embeddings: bool = False
    layer_norm_epsilon: float = 1e-5
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Mamba2Config":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        hidden_size = data.get("hidden_size", 4096)
        expand = data.get("expand", 2)
        num_heads = data.get("num_heads", 128)
        head_dim = data.get("head_dim", 64)
        intermediate_size = data.get("intermediate_size", num_heads * head_dim)

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=data.get("num_hidden_layers", data.get("n_layer", 64)),
            intermediate_size=intermediate_size,
            state_size=data.get("state_size", 128),
            conv_kernel=data.get("conv_kernel", 4),
            expand=expand,
            n_groups=data.get("n_groups", 8),
            num_heads=num_heads,
            head_dim=head_dim,
            chunk_size=data.get("chunk_size", 256),
            vocab_size=data.get("vocab_size", 32768),
            use_bias=data.get("use_bias", False),
            use_conv_bias=data.get("use_conv_bias", True),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            layer_norm_epsilon=data.get("layer_norm_epsilon", 1e-5),
        )


class Mamba2Model(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Mamba2DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class Mamba2ForCausalLM(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.backbone = Mamba2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.embedding_op.emb.weight = (
                self.backbone.embeddings.embedding_op.emb.weight
            )

    def forward(self, input_ids, positions):
        return self.backbone(input_ids, positions)

    def compute_logits(self, hidden_states):
        """Project pre-selected per-seq hidden states to vocab logits.

        ``infra/engine.py``'s ``run_mamba`` extracts the last hidden
        state per sequence before calling here, so we bypass
        ``ParallelLMHead.project``'s context-driven slicing (which
        relies on attention-only ``cu_seqlens_q`` / ``logit_indices``).
        """
        partial = self.lm_head.linear_op(
            hidden_states, self.lm_head.embedding_op.emb.weight,
        )
        logits = self.lm_head.gather_logits(partial)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, hidden_states):
        return self.compute_logits(hidden_states)
