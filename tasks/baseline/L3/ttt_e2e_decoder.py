"""TTT-E2E transformer (L3).

Wires up: token embedding -> N transformer blocks (with prime on the last
``suffix_len``) -> final RMSNorm -> tied LM head.

Mirrors ``ttt/model/transformer.py:TransformerModel`` and ``CausalLM``.

The L3 boundary here is simple wiring; the inner-loop SGD lives at L4.
"""

from __future__ import annotations

import torch
from torch import nn

from ..L1.embedding import Embedding
from ..L1.rms_norm_native import RMSNormNative
from ..L2.ttt_e2e_block import TTTE2EBlock


class TTTE2EDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        window_size: int,
        chunk_size: int,
        suffix_len: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        attention_backend: str = "cudnn",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.suffix_len = suffix_len
        self.tie_word_embeddings = tie_word_embeddings

        self.wte = Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            has_prime = (i >= num_hidden_layers - suffix_len)
            self.layers.append(
                TTTE2EBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    window_size=window_size,
                    chunk_size=chunk_size,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                    qk_norm=qk_norm,
                    rms_norm_eps=rms_norm_eps,
                    has_prime=has_prime,
                    attention_backend=attention_backend,
                )
            )

        self.ln_f = RMSNormNative(hidden_size, eps=rms_norm_eps)

        if not tie_word_embeddings:
            from ..L1.linear import Linear
            self.lm_head = Linear(hidden_size, vocab_size, bias=False)
        else:
            self.lm_head = None

    # Convenience selectors --------------------------------------------------

    @property
    def prefix_layers(self):
        return self.layers[: self.num_hidden_layers - self.suffix_len]

    @property
    def suffix_layers(self):
        return self.layers[self.num_hidden_layers - self.suffix_len :]

    # Forward helpers --------------------------------------------------------

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tie_word_embeddings:
            return hidden_states @ self.wte.emb.weight.t()
        return self.lm_head(hidden_states)
