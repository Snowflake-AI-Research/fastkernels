"""GLM4.5V dense/shared-routed text layers and GLM learned-position vision."""

import torch

from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from ..patches.grouped_topk_normalization import GroupedTopKNormalization
from . import glm4v
from .dots1 import NormalizedExperts
from .olmo2 import decoder_config


def build_from_config(config, device, dtype):
    text = config.text_config
    if (not text.attention_bias or text.use_qk_norm or not text.norm_topk_prob
            or text.hidden_act != 'silu' or text.rope_parameters['rope_type'] != 'default'):
        raise ValueError('GLM4.5V uses biased QKV without Q/K norms, normalized sigmoid routing and SiLU')
    language = LlamaForCausalLM(decoder_config(text, dtype))
    for i, layer in enumerate(language.model.layers):
        if i >= text.first_k_dense_replace:
            layer.mlp = NormalizedExperts(
                hidden_size=text.hidden_size, num_experts=text.n_routed_experts,
                top_k=text.num_experts_per_tok, moe_intermediate_size=text.moe_intermediate_size,
                routing='sigmoid', keep_router_weights_fp32=True,
                num_expert_group=text.n_group, topk_group=text.topk_group,
                shared_expert_intermediate_size=text.n_shared_experts * text.moe_intermediate_size,
                normalizer=GroupedTopKNormalization(scoring_func='sigmoid', epsilon=1e-20,
                                                    scale=text.routed_scaling_factor))
    model = glm4v.Model(language, config).to(device=device, dtype=dtype).eval()
    glm4v.attach_positions(model, config, device, interleaved=False)
    for layer in model.model.layers[text.first_k_dense_replace:]:
        layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    backbone, text = model.model.text, config.text_config
    direct = {'model.language_model.embed_tokens.weight': backbone.embed_tokens.embedding_op.emb.weight,
              'model.language_model.norm.weight': backbone.norm.weight,
              'lm_head.weight': model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(backbone.layers):
        p = f'model.language_model.layers.{i}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            direct[p + name + '.weight'] = getattr(layer, name).weight
        direct[p + 'self_attn.o_proj.weight'] = layer.self_attn.o_proj.weight
        for shard in ('q', 'k', 'v'):
            for kind in ('weight', 'bias'):
                packed[p + f'self_attn.{shard}_proj.{kind}'] = (getattr(layer.self_attn.qkv_proj, kind), shard)
        if i < text.first_k_dense_replace:
            target, stem = layer.mlp, p + 'mlp.'
        else:
            expert = layer.mlp
            direct[p + 'mlp.gate.weight'] = expert.gate.weight
            direct[p + 'mlp.gate.e_score_correction_bias'] = expert.gate.e_score_correction_bias
            direct[p + 'mlp.experts.gate_up_proj'] = expert.w13
            direct[p + 'mlp.experts.down_proj'] = expert.w2
            target, stem = expert.shared_expert, p + 'mlp.shared_experts.'
        direct[stem + 'down_proj.weight'] = target.down_proj.weight
        for shard, name in enumerate(('gate_proj', 'up_proj')):
            packed[stem + name + '.weight'] = (target.gate_up_proj.weight, shard)
    for name, parameter in direct.items():
        value = remaining.pop(name)
        if parameter.shape != value.shape or parameter.dtype != value.dtype:
            raise ValueError(f'GLM4v-MoE shape/dtype mismatch: {name}')
        parameter.copy_(value)
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, remaining.pop(name), shard)
    glm4v.load_vision(model.model.vision, remaining)
    if remaining:
        raise KeyError(f'Unmapped GLM4v-MoE state: {sorted(remaining)}')


make_workloads = glm4v.make_workloads
