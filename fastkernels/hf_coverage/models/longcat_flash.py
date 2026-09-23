"""LongCat's two attention sublayers and routed identity-expert shortcut."""
from copy import copy
import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.models.deepseek_v2 import ExpandedAttention
from fastkernels.hf_coverage.models.llama import make_workloads as decoder_workloads
from fastkernels.hf_coverage.models.olmo2 import PostNormModel, decoder_config
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.grouped_topk import GroupedTopK
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class ScaledNorm(RMSNorm):
    def __init__(self, width, scale):
        super().__init__(width, 1e-6)
        self.scale = scale

    def forward(self, x):
        return super().forward(x) * self.scale


class LatentQuery(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q_a_proj = Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, 1e-6)
        self.q_b_proj = Linear(c.q_lora_rank, c.num_attention_heads*c.qk_head_dim, bias=False)
        self.scale = (c.hidden_size/c.q_lora_rank)**.5

    def forward(self, x):
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))) * self.scale


class InterleavedInputRotary(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.width = c.qk_rope_head_dim
        self.rotary = RotaryEmbedding(self.width, c.max_position_embeddings, c.rope_parameters['rope_theta'])

    def forward(self, positions, q, k):
        q_shape, k_shape = q.shape, k.shape
        n = q.shape[0]
        q = q.reshape(n, -1, self.width//2, 2).transpose(-1, -2).reshape(n, -1).contiguous()
        k = k.reshape(n, -1, self.width//2, 2).transpose(-1, -2).reshape(n, -1).contiguous()
        q, k = self.rotary(positions, q, k)
        return q.reshape(q_shape), k.reshape(k_shape)


class IdentityExperts(nn.Module):
    """Run selected learned experts and retain the separate identity branch."""
    def __init__(self, c):
        super().__init__()
        self.real, self.total, self.topk = c.n_routed_experts, c.n_routed_experts+c.zero_expert_num, c.moe_topk
        self.scale = c.routed_scaling_factor
        self.router = nn.Module()
        self.router.classifier = Linear(c.hidden_size, self.total, bias=False)
        self.router.register_buffer('e_score_correction_bias', torch.zeros(self.total))
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.empty(self.total, 2*c.expert_ffn_hidden_size, c.hidden_size))
        self.experts.down_proj = nn.Parameter(torch.empty(self.real, c.hidden_size, c.expert_ffn_hidden_size))
        self.linear = Matmul()
        self.route = GroupedTopK(scoring_func='softmax', renormalize=False, routed_scaling_factor=self.scale)
        self.activation, self.product, self.reduce = SiluAndMul(), ProductGate(), MoeSum()

    def forward(self, x):
        logits = self.linear(x.float(), self.router.classifier.weight.float())
        weights, indices = self.route(logits, self.router.e_score_correction_bias.float(), 1, 1, self.topk)
        slots = torch.empty(x.shape[0], self.topk, x.shape[-1], device=x.device, dtype=x.dtype)
        # These are the model's expert branches; only discrete routing controls selection.
        for expert in range(self.total):
            rows, positions = torch.where(indices == expert)
            if rows.numel() == 0:
                continue
            selected = x[rows]
            if expert < self.real:
                selected = self.activation(self.linear(selected, self.experts.gate_up_proj[expert]))
                selected = self.linear(selected, self.experts.down_proj[expert])
            weighted = self.product(torch.cat((selected.float(), weights[rows, positions, None].expand_as(selected)), -1)).to(x.dtype)
            slots[rows, positions] = weighted
        return self.reduce(slots.reshape(-1, x.shape[-1]), self.topk)


class DualLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.input_layernorm, self.post_attention_layernorm = nn.ModuleList(), nn.ModuleList()
        for _ in range(2):
            a = ExpandedAttention(c)
            a.q_proj = LatentQuery(c)
            a.kv_a_layernorm = ScaledNorm(c.kv_lora_rank, (c.hidden_size/c.kv_lora_rank)**.5)
            a.rotary_emb = InterleavedInputRotary(c)
            self.self_attn.append(a)
            self.mlps.append(LlamaMLP(c, hidden_size=c.hidden_size, intermediate_size=c.ffn_hidden_size))
            self.input_layernorm.append(RMSNorm(c.hidden_size, c.rms_norm_eps))
            self.post_attention_layernorm.append(RMSNorm(c.hidden_size, c.rms_norm_eps))
        self.mlp = IdentityExperts(c)

    def forward(self, positions, hidden, residual=None):
        hidden = hidden+self.self_attn[0](positions, self.input_layernorm[0](hidden))
        normalized = self.post_attention_layernorm[0](hidden)
        shortcut = self.mlp(normalized)
        hidden = hidden+self.mlps[0](normalized)
        hidden = hidden+self.self_attn[1](positions, self.input_layernorm[1](hidden))
        hidden = hidden+self.mlps[1](self.post_attention_layernorm[1](hidden))+shortcut
        return hidden, None


def build_from_config(config, device, dtype):
    if config.attention_bias or config.q_lora_rank is None or config.rope_parameters['rope_type'] != 'default':
        raise ValueError('Selected LongCat uses bias-free latent queries and default RoPE')
    carrier = copy(config)
    carrier.intermediate_size, carrier.num_hidden_layers = config.ffn_hidden_size, config.num_layers
    carrier.head_dim, carrier.num_key_value_heads = config.qk_head_dim, config.num_attention_heads
    fk = decoder_config(carrier, dtype)
    model = LlamaForCausalLM(fk)
    model.model = PostNormModel(fk)
    model.model.layers = nn.ModuleList([DualLayer(config) for _ in range(config.num_layers)])
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {'model.embed_tokens.weight': model.model.embed_tokens.embedding_op.emb.weight,
              'model.norm.weight': model.model.norm.weight, 'lm_head.weight': model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f'model.layers.{i}.'
        for j, a in enumerate(layer.self_attn):
            for name in ('q_a_proj', 'q_a_layernorm', 'q_b_proj'):
                direct[p+f'self_attn.{j}.{name}.weight'] = getattr(a.q_proj, name).weight
            for name in ('kv_a_proj_with_mqa', 'kv_a_layernorm', 'kv_b_proj', 'o_proj'):
                direct[p+f'self_attn.{j}.{name}.weight'] = getattr(a, name).weight
            for name in ('input_layernorm', 'post_attention_layernorm'):
                direct[p+f'{name}.{j}.weight'] = getattr(layer, name)[j].weight
            direct[p+f'mlps.{j}.down_proj.weight'] = layer.mlps[j].down_proj.weight
            for shard, name in enumerate(('gate_proj', 'up_proj')):
                packed[p+f'mlps.{j}.{name}.weight'] = (layer.mlps[j].gate_up_proj.weight, shard)
        for name, tensor in layer.mlp.state_dict(keep_vars=True).items():
            direct[p+'mlp.'+name] = tensor
    if state_dict.keys() != direct.keys()|packed.keys():
        raise KeyError(f'LongCat state mismatch: {sorted(state_dict.keys() ^ (direct.keys()|packed.keys()))}')
    for name, target in direct.items():
        if target.shape != state_dict[name].shape or target.dtype != state_dict[name].dtype:
            raise ValueError(f'LongCat shape/dtype mismatch: {name}')
        target.copy_(state_dict[name])
    for name, (target, shard) in packed.items():
        target.weight_loader(target, state_dict[name], shard)


def make_workloads(model, inputs, config, *, case=None):
    attentions = [a.attn for layer in model.model.layers for a in layer.self_attn]
    workloads = decoder_workloads(model, inputs, config, attentions=attentions, case=case)
    if case is None or case.get('workload') != 'causal_lm_continuation':
        return workloads
    for name, work in list(workloads.items()):
        def collect(output, parent=work.collect):
            output = parent(output)
            # ExpandedAttention pads physical values to the query/key width.
            # Expose only HF's logical value dimensions for state comparison.
            return {key: value[..., :config.v_head_dim] if key.endswith('.value') else value
                    for key, value in output.items()}
        workloads[name] = Workload(run=work.run, prepare=work.prepare, collect=collect)
    return workloads
