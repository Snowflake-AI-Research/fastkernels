"""Falcon-H1's parallel attention/Mamba2 branches with checkpoint scales."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.bamba import copy_parameter, hybrid_workloads, load_mixer
from fastkernels.hf_coverage.patches.falcon_h1_projection import FalconH1ProjectionScale
from fastkernels.hf_coverage.patches.falcon_h1_scan import FalconH1Mixer
from fastkernels.infra.context import AttnBackendConfig, set_attn_backend_config
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_gated import RMSNormGated
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.parallel_linear import MergedColumnParallelLinear


class FalconH1Attention(LlamaAttention):
    def __init__(self, config, index, rotary):
        super().__init__(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                         config.head_dim, rotary_emb=rotary, bias=config.attention_bias,
                         o_proj_bias=config.attention_bias, layer_idx=index)
        self.key_multiplier = config.key_multiplier

    def forward(self, positions, hidden):
        q, k, v = self.qkv_proj(hidden).split(
            [self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim,
             self.num_kv_heads * self.head_dim], dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k * self.key_multiplier)
        return self.o_proj(self.attn(q, k, v))


class FalconH1MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size, config.intermediate_size], bias=config.mlp_bias,
        )
        self.activation = SiluAndMul()
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def forward(self, hidden):
        gate, up = self.gate_up_proj(hidden).chunk(2, dim=-1)
        hidden = self.activation(torch.cat((gate * self.gate_multiplier, up), dim=-1))
        return self.down_proj(hidden) * self.down_multiplier


class FalconH1Layer(nn.Module):
    def __init__(self, config, index, rotary):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = FalconH1Attention(config, index, rotary)
        self.feed_forward = FalconH1MLP(config)
        width = config.mamba_d_ssm if config.mamba_d_ssm is not None else config.mamba_expand * config.hidden_size
        self.mamba = FalconH1Mixer(
            hidden_size=config.hidden_size, ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv, intermediate_size=width,
            use_conv_bias=config.mamba_conv_bias, use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups, num_heads=config.mamba_n_heads, head_dim=config.mamba_d_head,
            rms_norm_eps=config.rms_norm_eps, activation=config.hidden_act,
            chunk_size=config.mamba_chunk_size, layer_idx=index,
        )
        # HF multiplies the normalized FP32 values by weight before the final
        # cast; the existing standalone gated norm retains that order.
        self.mamba.norm = RMSNormGated(width, eps=config.rms_norm_eps, norm_before_gate=False)
        group_width = config.mamba_n_groups * config.mamba_d_state
        self.mamba.in_proj = FalconH1ProjectionScale(
            self.mamba.in_proj, [width, width, group_width, group_width, config.mamba_n_heads], config.ssm_multipliers,
        )

    def forward(self, hidden, positions):
        x = self.input_layernorm(hidden)
        recurrent = self.mamba(x * self.config.ssm_in_multiplier) * self.config.ssm_out_multiplier
        attention = self.self_attn(positions, x * self.config.attention_in_multiplier) * self.config.attention_out_multiplier
        hidden = hidden + (recurrent + attention)
        return hidden + self.feed_forward(self.pre_ff_layernorm(hidden))


class FalconH1ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_parameters['rope_theta'])
        self.layers = nn.ModuleList([FalconH1Layer(config, i, self.rotary) for i in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids) * self.config.embedding_multiplier
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.final_layernorm(hidden)) * self.config.lm_head_multiplier


def build_from_config(config, device, dtype):
    if not config.mamba_rms_norm or config.mamba_norm_before_gate or config.mamba_n_groups != 1:
        raise ValueError('Selected Falcon-H1 example uses one-group RMS norm after its gate')
    if config.hidden_act != 'silu' or config.rope_parameters['rope_type'] != 'default':
        raise ValueError('Selected Falcon-H1 example uses SiLU and default RoPE')
    if config.projectors_bias or config.mamba_proj_bias or config.attention_bias or config.mlp_bias or config.tie_word_embeddings:
        raise ValueError('Selected Falcon-H1 example has bias-free projections and an untied head')
    if tuple(config.time_step_limit) != (0.0, float('inf')):
        raise ValueError('Selected Falcon-H1 example has unrestricted time steps')
    set_attn_backend_config(AttnBackendConfig(backend='flash_attn', block_size=256, kv_layout='NHD'))
    model = FalconH1ForCausalLM(config).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        layer.mamba.A.data = layer.mamba.A.data.float()
    return model


def load_state_dict_into(model, weights, config):
    copy_parameter(model.embed_tokens.embedding_op.emb.weight, weights['model.embed_tokens.weight'])
    copy_parameter(model.lm_head.weight, weights['lm_head.weight'])
    copy_parameter(model.final_layernorm.weight, weights['model.final_layernorm.weight'])
    for index, layer in enumerate(model.layers):
        prefix = f'model.layers.{index}.'
        for name in ('input_layernorm', 'pre_ff_layernorm'):
            copy_parameter(getattr(layer, name).weight, weights[prefix + name + '.weight'])
        load_mixer(layer.mamba, weights, prefix + 'mamba.')
        for shard, name in enumerate(('gate', 'up')):
            parameter = layer.feed_forward.gate_up_proj.weight
            parameter.weight_loader(parameter, weights[prefix + f'feed_forward.{name}_proj.weight'], shard)
        copy_parameter(layer.feed_forward.down_proj.weight, weights[prefix + 'feed_forward.down_proj.weight'])
        for shard in ('q', 'k', 'v'):
            parameter = layer.self_attn.qkv_proj.weight
            parameter.weight_loader(parameter, weights[prefix + f'self_attn.{shard}_proj.weight'], shard)
        copy_parameter(layer.self_attn.o_proj.weight, weights[prefix + 'self_attn.o_proj.weight'])


def make_workloads(model, inputs, config, *, case=None):
    return hybrid_workloads(model, inputs, config, [layer.mamba for layer in model.layers],
                            [layer.self_attn.attn for layer in model.layers], config.mamba_chunk_size,
                            case=case, attention_indices=list(range(config.num_hidden_layers)))
