"""Zamba2 constructor-default hybrid with shared weights and per-use adapters."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.bamba import copy_parameter, hybrid_workloads, load_mixer
from fastkernels.infra.context import AttnBackendConfig, set_attn_backend_config
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.mamba2_mixer import Mamba2Mixer
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding


class Zamba2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_size = config.num_attention_heads * config.attention_head_dim
        self.kv_size = config.num_key_value_heads * config.attention_head_dim
        self.q_proj = Linear(config.attention_hidden_size, self.q_size, bias=False)
        self.k_proj = Linear(config.attention_hidden_size, self.kv_size, bias=False)
        self.v_proj = Linear(config.attention_hidden_size, self.kv_size, bias=False)
        self.o_proj = Linear(self.q_size, config.hidden_size, bias=False)
        self.attention = Attention(
            config.num_attention_heads, config.attention_head_dim,
            scale=(config.attention_head_dim / 2) ** -0.5, num_kv_heads=config.num_key_value_heads,
        )

    def forward(self, hidden):
        return self.o_proj(self.attention(self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)))


class Zamba2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.add_bias_linear)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=config.add_bias_linear)
        self.activation = GeluAndMul(approximate="none")
        self.gate_up_proj_adapter_list = nn.ModuleList([
            nn.Sequential(
                Linear(config.hidden_size, config.adapter_rank, bias=False),
                Linear(config.adapter_rank, 2 * config.intermediate_size, bias=False),
            ) for _ in config.hybrid_layer_ids
        ])

    def forward(self, hidden, adapter_index):
        projected = self.gate_up_proj(hidden) + self.gate_up_proj_adapter_list[adapter_index](hidden)
        return self.down_proj(self.activation(projected))


class Zamba2SharedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNormNative(config.attention_hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Zamba2Attention(config)
        self.pre_ff_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = Zamba2MLP(config)

    def forward(self, hidden, original, adapter_index):
        hidden = self.input_layernorm(torch.cat((hidden, original), dim=-1))
        hidden = self.pre_ff_layernorm(self.self_attn(hidden))
        return self.feed_forward(hidden, adapter_index)


class Zamba2Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.input_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = Mamba2Mixer(
            hidden_size=config.hidden_size, ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv, intermediate_size=config.mamba_expand * config.hidden_size,
            use_conv_bias=config.use_conv_bias, use_bias=config.add_bias_linear,
            n_groups=config.mamba_ngroups, num_heads=config.n_mamba_heads,
            head_dim=config.mamba_headdim, rms_norm_eps=1e-5,
            activation="silu", chunk_size=config.chunk_size, layer_idx=index,
        )
        self.hybrid = config.layers_block_type[index] == "hybrid"
        if self.hybrid:
            self.shared_transformer = Zamba2SharedTransformer(config)
            self.linear = Linear(config.hidden_size, config.hidden_size, bias=False)
            self.adapter_index = config.hybrid_layer_ids.index(index)

    def forward(self, hidden, original):
        residual = hidden
        if self.hybrid:
            hidden = hidden + self.linear(self.shared_transformer(hidden, original, self.adapter_index))
        return residual + self.mamba(self.input_layernorm(hidden))


class Zamba2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Zamba2Layer(config, index) for index in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding_op.emb.weight
        shared = None
        for layer in self.layers:
            if not layer.hybrid:
                continue
            if shared is None:
                shared = layer.shared_transformer
            else:
                # Actual parameter aliases preserve sharing; attention objects
                # retain independent KV caches for each use of the shared block.
                for name, parameter in shared.named_parameters():
                    module_name, _, field = name.rpartition(".")
                    setattr(layer.shared_transformer.get_submodule(module_name), field, parameter)

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids)
        original = hidden.clone()
        for layer in self.layers:
            hidden = layer(hidden, original)
        return self.lm_head(self.final_layernorm(hidden))


def build_from_config(config, device, dtype):
    if config.num_mem_blocks != 1 or config.use_shared_attention_adapter or config.use_mem_rope:
        raise ValueError("The Zamba2 constructor-default case has one memory block, no attention adapter, and no RoPE")
    if config.use_mem_eff_path or not config.use_mamba_kernels or config.hidden_act != "gelu":
        raise ValueError("The selected Zamba2 inference path uses separate Mamba kernels and exact GELU")
    if config.time_step_limit is not None and tuple(config.time_step_limit) != (0.0, float("inf")):
        raise ValueError("The selected Zamba2 case has unrestricted time steps")
    if len(config.layers_block_type) != config.num_hidden_layers:
        raise ValueError("Zamba2 layer pattern must match num_hidden_layers")
    set_attn_backend_config(AttnBackendConfig(backend="flash_attn", block_size=256, kv_layout="NHD"))
    model = Zamba2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        layer.mamba.A.data = layer.mamba.A.data.float()
    return model


def load_state_dict_into(model, weights, config):
    copy_parameter(model.embed_tokens.embedding_op.emb.weight, weights["model.embed_tokens.weight"])
    copy_parameter(model.lm_head.weight, weights.get("lm_head.weight", weights["model.embed_tokens.weight"]))
    copy_parameter(model.final_layernorm.weight, weights["model.final_layernorm.weight"])
    first_hybrid = config.hybrid_layer_ids[0]
    shared_prefix = f"model.layers.{first_hybrid}.shared_transformer."
    shared_loaded = False
    for index, layer in enumerate(model.layers):
        prefix = f"model.layers.{index}."
        if layer.hybrid:
            copy_parameter(layer.linear.weight, weights[prefix + "linear.weight"])
            if not shared_loaded:
                for name, parameter in layer.shared_transformer.named_parameters():
                    copy_parameter(parameter, weights[shared_prefix + name])
                shared_loaded = True
            prefix += "mamba_decoder."
        copy_parameter(layer.input_layernorm.weight, weights[prefix + "input_layernorm.weight"])
        load_mixer(layer.mamba, weights, prefix + "mamba.")


def make_workloads(model, inputs, config, *, case=None):
    return hybrid_workloads(
        model, inputs, config, [layer.mamba for layer in model.layers],
        [layer.shared_transformer.self_attn.attention for layer in model.layers if layer.hybrid], config.chunk_size,
        case=case, attention_indices=config.hybrid_layer_ids,
    )
