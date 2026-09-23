"""Gemma v1 causal LM composed from existing decoder, norm, and GELU ops."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import (
    load_state_dict_into as load_llama_weights,
    make_workloads,
)
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM, LlamaModel


class GemmaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
            layer.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
            layer.mlp.act_fn = GeluAndMul(approximate="tanh")
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer(
            "embed_scale", torch.tensor(config.hidden_size**0.5, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # HF rounds this fixed scale to the embedding weight dtype before
            # multiplying, including when sqrt(hidden_size) is not integral.
            inputs_embeds = inputs_embeds * self.embed_scale.to(inputs_embeds.dtype)
        return super().forward(input_ids, positions, inputs_embeds=inputs_embeds)


class GemmaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight


def build_from_config(config, device, dtype) -> GemmaForCausalLM:
    if _tp_size() != 1:
        raise ValueError("The Gemma coverage workload requires tensor parallel size 1")
    if (
        config.hidden_act != "gelu_pytorch_tanh"
        or config.attention_bias
        or not config.tie_word_embeddings
        or config.use_bidirectional_attention
    ):
        raise ValueError("The Gemma v1 pilot requires tanh-GELU, bias-free causal attention, and a tied head")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The Gemma v1 pilot requires default RoPE")
    if config.num_attention_heads != config.num_key_value_heads:
        raise ValueError("The Gemma constructor-default pilot preserves multi-head attention")
    if config.num_attention_heads * config.head_dim * 3 != config.hidden_size * 4:
        raise ValueError("The Gemma constructor-default pilot preserves attention width 4/3 of hidden size")

    fk_config = LlamaConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype,
        qkv_bias=False,
    )
    model = GemmaForCausalLM(fk_config).to(device=device, dtype=dtype).eval()
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config) -> None:
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("The Gemma tied embedding and output-head weights must agree")
    load_llama_weights(model, state_dict, config)
