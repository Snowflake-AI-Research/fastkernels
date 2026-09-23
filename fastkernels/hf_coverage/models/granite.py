"""Granite causal LM with checkpoint-defined embedding, residual, attention, and logit scales."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import load_state_dict_into as load_llama_weights
from fastkernels.hf_coverage.models.llama import make_workloads as llama_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L3.llama_decoder import LlamaDecoderLayer
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM, LlamaModel


class GraniteDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, rotary, source):
        super().__init__(config, rotary_emb=rotary)
        attention = self.self_attn.attn
        self.self_attn.attn = Attention(
            attention.num_heads, attention.head_size, source.attention_multiplier,
            num_kv_heads=attention.num_kv_heads,
        )
        self.residual_multiplier = source.residual_multiplier

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states) * self.residual_multiplier
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states) * self.residual_multiplier
        return hidden_states, residual


class GraniteModel(LlamaModel):
    def __init__(self, config, source):
        nn.Module.__init__(self)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.layers = nn.ModuleList([
            GraniteDecoderLayer(config, self.rotary_emb, source)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embedding_multiplier = source.embedding_multiplier
        self.capture_aux_hidden_states = False
        self.aux_layer_ids = []

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * self.embedding_multiplier
        return super().forward(input_ids, positions, inputs_embeds=inputs_embeds)


class GraniteForCausalLM(LlamaForCausalLM):
    def __init__(self, config, source):
        nn.Module.__init__(self)
        self.config = config
        self.model = GraniteModel(config, source)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("The Granite coverage workload requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.attention_bias or config.mlp_bias or not config.tie_word_embeddings:
        raise ValueError("The Granite-3.0-8b-base pilot requires bias-free SiLU layers and a tied head")
    if config.num_attention_heads != 4 * config.num_key_value_heads:
        raise ValueError("The Granite checkpoint preserves 4:1 grouped-query attention")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The Granite pilot requires default RoPE")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = GraniteForCausalLM(fk_config, config).to(device=device, dtype=dtype).eval()
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("The Granite tied embedding and output-head weights must agree")
    load_llama_weights(model, state_dict, model.config)


def make_workloads(model, inputs, config, *, case=None):
    def project(hidden):
        logits = model.lm_head.linear_op(hidden, model.lm_head.embedding_op.emb.weight)
        return {"logits": logits / config.logits_scaling}

    return llama_workloads(model, inputs, config, case=case, output_projection=project)
