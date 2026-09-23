"""GLM-4.5's biased grouped-query attention, head norms and routed experts."""
import torch
from fastkernels.hf_coverage.models.dots1 import NormalizedExperts
from fastkernels.hf_coverage.models.git import GitAttention
from fastkernels.hf_coverage.models.qwen2_precision import DenseCachedAttention
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.models.stablelm import PartialRotary
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class GroupedEagerAttention(GitAttention):
    """Reuse score-rounded BMM/softmax attention with HF's grouped KV expansion.

    The existing vision variant multiplies BF16 scores by the attention scale,
    evaluates softmax in FP32, then stores probabilities in BF16 before BMM.
    GLM4's eager reference uses that same order; fused attention does not.
    """

    def __init__(self):
        super().__init__(vision=True)

    def forward(self, query, key, value, causal=False):
        repeats = query.shape[2] // key.shape[2]
        key, value = (tensor.repeat_interleave(repeats, dim=2) for tensor in (key, value))
        return super().forward(query, key, value, causal=causal)


class NativeRotary(RotaryEmbedding):
    """Select existing RoPE with the reference's BF16 intermediate stores."""

    def forward(self, positions, query, key):
        return self.forward_native(positions, query, key, self.head_dim, self.cos_sin_cache)


def build_from_config(config, device, dtype):
    if not config.attention_bias or not config.use_qk_norm or config.rope_parameters['rope_type'] != 'default':
        raise ValueError('The GLM-4.5 checkpoint uses biased QKV, per-head Q/K norms and default partial RoPE')
    model = LlamaForCausalLM(decoder_config(config, dtype))
    rotary_dim = int(config.head_dim * config.rope_parameters['partial_rotary_factor'])
    rotary = PartialRotary(config.head_dim, rotary_dim, config.max_position_embeddings, config.rope_parameters['rope_theta'])
    rotary.rotary = NativeRotary(rotary_dim, config.max_position_embeddings, config.rope_parameters['rope_theta'], is_neox_style=True)
    model.model.rotary_emb = rotary
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.rotary_emb = rotary
        layer.self_attn.attn = DenseCachedAttention(
            config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        )
        layer.self_attn.attn.attention = GroupedEagerAttention()
        layer.self_attn.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        layer.self_attn.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        if i >= config.first_k_dense_replace:
            layer.mlp = NormalizedExperts(
                hidden_size=config.hidden_size, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                routing='sigmoid', keep_router_weights_fp32=True,
                num_expert_group=config.n_group, topk_group=config.topk_group,
                shared_expert_intermediate_size=config.n_shared_experts * config.moe_intermediate_size,
                normalizer=GroupedTopKNormalization(scoring_func='sigmoid', epsilon=1e-20, scale=config.routed_scaling_factor))
    model.to(device=device, dtype=dtype)
    for layer in model.model.layers[config.first_k_dense_replace:]:
        layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {'model.embed_tokens.weight': model.model.embed_tokens.embedding_op.emb.weight,
              'model.norm.weight': model.model.norm.weight, 'lm_head.weight': model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f'model.layers.{i}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            direct[p + name + '.weight'] = getattr(layer, name).weight
        for name in ('q_norm', 'k_norm', 'o_proj'):
            direct[p + 'self_attn.' + name + '.weight'] = getattr(layer.self_attn, name).weight
        for shard in ('q', 'k', 'v'):
            for kind in ('weight', 'bias'):
                packed[p + f'self_attn.{shard}_proj.{kind}'] = (getattr(layer.self_attn.qkv_proj, kind), shard)
        if i < config.first_k_dense_replace:
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
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f'GLM4-MoE weight mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}')
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f'GLM4-MoE shape/dtype mismatch: {name}')
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
