"""Granite 4.0 Hybrid's NoPE/Mamba2 blocks and routed plus shared experts."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.bamba import copy_parameter, hybrid_workloads, load_mixer, make_mixer
from fastkernels.infra.context import AttnBackendConfig, set_attn_backend_config
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE


class GraniteMoeHybridLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        self.input_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_multiplier = config.residual_multiplier
        if self.layer_type == 'mamba':
            self.mamba = make_mixer(config, index)
        else:
            self.self_attn = LlamaAttention(
                config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
                rotary_emb=None, bias=False, layer_idx=index,
            )
            self.self_attn.attn.scale = config.attention_multiplier
        self.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size, num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok, moe_intermediate_size=config.intermediate_size,
            routing='softmax', renormalize=True,
            shared_expert_intermediate_size=config.shared_intermediate_size, shared_expert_gate=False,
        )
        # Select the existing grouped GEMMs with materialized BF16 intermediates.
        # Shared-input checks track HF's expert arithmetic more closely here.
        self.mlp.use_trtllm = False

    def forward(self, hidden, positions):
        normalized = self.input_layernorm(hidden)
        mixed = self.mamba(normalized) if self.layer_type == 'mamba' else self.self_attn(positions, normalized)
        hidden = hidden + mixed * self.residual_multiplier
        return hidden + self.mlp(self.post_attention_layernorm(hidden)) * self.residual_multiplier


class GraniteMoeHybridForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GraniteMoeHybridLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.embedding_op.emb.weight

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids) * self.config.embedding_multiplier
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.final_layernorm(hidden)) / self.config.logits_scaling


def build_from_config(config, device, dtype):
    if config.position_embedding_type != 'nope' or config.hidden_act != 'silu' or config.attention_bias:
        raise ValueError('Selected Granite Hybrid example uses NoPE, SiLU and bias-free attention')
    if not config.tie_word_embeddings or config.num_local_experts <= 0 or config.output_router_logits:
        raise ValueError('Selected Granite Hybrid example uses tied embeddings and active MoE without router outputs')
    if set(config.layer_types) != {'mamba', 'attention'} or config.mamba_n_groups != 1:
        raise ValueError('Selected Granite Hybrid case retains both mixer types and one SSM group')
    if tuple(config.time_step_limit) != (0.0, float('inf')):
        raise ValueError('Selected Granite Hybrid example has unrestricted time steps')
    set_attn_backend_config(AttnBackendConfig(backend='flash_attn', block_size=256, kv_layout='NHD'))
    model = GraniteMoeHybridForCausalLM(config).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        if layer.layer_type == 'mamba':
            layer.mamba.A.data = layer.mamba.A.data.float()
    return model


def load_state_dict_into(model, weights, config):
    copy_parameter(model.embed_tokens.embedding_op.emb.weight, weights['model.embed_tokens.weight'])
    copy_parameter(model.final_layernorm.weight, weights['model.norm.weight'])
    for i, layer in enumerate(model.layers):
        prefix = f'model.layers.{i}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            copy_parameter(getattr(layer, name).weight, weights[prefix + name + '.weight'])
        if layer.layer_type == 'mamba':
            load_mixer(layer.mamba, weights, prefix + 'mamba.')
        else:
            for shard in ('q', 'k', 'v'):
                parameter = layer.self_attn.qkv_proj.weight
                parameter.weight_loader(parameter, weights[prefix + f'self_attn.{shard}_proj.weight'], shard)
            copy_parameter(layer.self_attn.o_proj.weight, weights[prefix + 'self_attn.o_proj.weight'])
        mlp = layer.mlp
        copy_parameter(mlp.gate.weight, weights[prefix + 'block_sparse_moe.router.layer.weight'])
        mlp.w13.data.copy_(weights[prefix + 'block_sparse_moe.input_linear.weight'])
        mlp.w2.data.copy_(weights[prefix + 'block_sparse_moe.output_linear.weight'])
        copy_parameter(mlp.shared_expert.gate_up_proj.weight, weights[prefix + 'shared_mlp.input_linear.weight'])
        copy_parameter(mlp.shared_expert.down_proj.weight, weights[prefix + 'shared_mlp.output_linear.weight'])
        mlp.process_weights_after_loading()


def make_workloads(model, inputs, config, *, case=None):
    return hybrid_workloads(
        model, inputs, config,
        [layer.mamba for layer in model.layers if layer.layer_type == 'mamba'],
        [layer.self_attn.attn for layer in model.layers if layer.layer_type == 'attention'], config.mamba_chunk_size,
        case=case, attention_indices=[i for i, layer in enumerate(model.layers) if layer.layer_type == 'attention'],
    )
