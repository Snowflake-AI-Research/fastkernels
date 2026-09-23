"""Documented Nemotron-H 8B: alternating Mamba2, dense ReLU² and NoPE blocks."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.hf_coverage.models.bamba import copy_parameter, hybrid_workloads, load_mixer
from fastkernels.infra.context import AttnBackendConfig, set_attn_backend_config
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.mamba2_mixer import Mamba2Mixer
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding


class NemotronHMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.activation = SquaredReLU()
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden):
        return self.down_proj(self.activation(self.up_proj(hidden)))


class NemotronHBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.block_type = config.layers_block_type[index]
        self.norm = RMSNormNative(config.hidden_size, eps=config.layer_norm_epsilon)
        if self.block_type == 'mamba':
            self.mixer = Mamba2Mixer(
                hidden_size=config.hidden_size, ssm_state_size=config.ssm_state_size,
                conv_kernel_size=config.conv_kernel,
                intermediate_size=config.mamba_num_heads * config.mamba_head_dim,
                use_conv_bias=config.use_conv_bias, use_bias=config.use_bias,
                n_groups=config.n_groups, num_heads=config.mamba_num_heads, head_dim=config.mamba_head_dim,
                rms_norm_eps=config.layer_norm_epsilon, activation=config.mamba_hidden_act,
                chunk_size=config.chunk_size, layer_idx=index,
            )
        elif self.block_type == 'attention':
            self.mixer = LlamaAttention(config.hidden_size, config.num_attention_heads,
                                       config.num_key_value_heads, config.head_dim,
                                       rotary_emb=None, bias=False, layer_idx=index)
        else:
            self.mixer = NemotronHMLP(config)

    def forward(self, hidden, positions):
        normalized = self.norm(hidden.to(self.norm.weight.dtype))
        mixed = self.mixer(positions, normalized) if self.block_type == 'attention' else self.mixer(normalized)
        return hidden + mixed


class NemotronHForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHBlock(config, i) for i in range(len(config.layers_block_type))])
        self.final_layernorm = RMSNormNative(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.final_layernorm(hidden)).float()


def build_from_config(config, device, dtype):
    if set(config.layers_block_type) != {'mamba', 'attention', 'mlp'}:
        raise ValueError('Documented Nemotron-H 8B uses Mamba, attention and dense MLP blocks')
    if config.mlp_hidden_act != 'relu2' or config.mamba_hidden_act != 'silu':
        raise ValueError('Documented Nemotron-H 8B uses ReLU² MLPs and SiLU convolution')
    if config.sliding_window is not None or config.tie_word_embeddings or config.use_bias or config.mlp_bias:
        raise ValueError('Documented Nemotron-H 8B uses full attention and untied bias-free projections')
    if config.num_nextn_predict_layers or tuple(config.time_step_limit) != (0.0, float('inf')):
        raise ValueError('Documented Nemotron-H 8B disables MTP and has unrestricted time steps')
    set_attn_backend_config(AttnBackendConfig(backend='flash_attn', block_size=256, kv_layout='NHD'))
    model = NemotronHForCausalLM(config).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        if layer.block_type == 'mamba':
            layer.mixer.A.data = layer.mixer.A.data.float()
    return model


def load_state_dict_into(model, weights, config):
    copy_parameter(model.embed_tokens.embedding_op.emb.weight, weights['model.embeddings.weight'])
    copy_parameter(model.final_layernorm.weight, weights['model.norm_f.weight'])
    copy_parameter(model.lm_head.weight, weights['lm_head.weight'])
    for i, layer in enumerate(model.layers):
        prefix = f'model.layers.{i}.'
        copy_parameter(layer.norm.weight, weights[prefix + 'norm.weight'])
        if layer.block_type == 'mamba':
            load_mixer(layer.mixer, weights, prefix + 'mixer.')
        elif layer.block_type == 'attention':
            for shard in ('q', 'k', 'v'):
                parameter = layer.mixer.qkv_proj.weight
                parameter.weight_loader(parameter, weights[prefix + f'mixer.{shard}_proj.weight'], shard)
            copy_parameter(layer.mixer.o_proj.weight, weights[prefix + 'mixer.o_proj.weight'])
        else:
            for name in ('up_proj', 'down_proj'):
                copy_parameter(getattr(layer.mixer, name).weight, weights[prefix + f'mixer.{name}.weight'])


def make_workloads(model, inputs, config, *, case=None):
    # HF derives depth from its block list; to_dict omits that property.
    dimensions = SimpleNamespace(num_hidden_layers=len(model.layers), vocab_size=config.vocab_size)
    return hybrid_workloads(
        model, inputs, dimensions,
        [layer.mixer for layer in model.layers if layer.block_type == 'mamba'],
        [layer.mixer.attn for layer in model.layers if layer.block_type == 'attention'], config.chunk_size,
        case=case, attention_indices=[i for i, layer in enumerate(model.layers) if layer.block_type == 'attention'],
    )
