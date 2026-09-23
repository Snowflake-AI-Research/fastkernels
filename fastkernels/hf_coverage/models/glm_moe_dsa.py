"""GLM-5 BF16 sparse attention with explicit, timed index-score computation."""
import torch
from torch import nn

from fastkernels.hf_coverage.models.dots1 import NormalizedExperts
from fastkernels.hf_coverage.models.glm4_moe import NativeRotary
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.top_k_per_row import TopKPerRow
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class Indexer(nn.Module):
    """Compose the pinned reference's unquantized index scores and raw top-k.

    The existing full sparse indexer quantizes Q/K to FP8. The selected HF
    default instead keeps these inputs in BF16 and computes scores in FP32.
    """
    def __init__(self, c):
        super().__init__()
        self.heads, self.width, self.rope_width = c.index_n_heads, c.index_head_dim, c.qk_rope_head_dim
        self.topk = c.index_topk
        self.wq_b = Linear(c.q_lora_rank, self.heads*self.width, False)
        self.wk = Linear(c.hidden_size, self.width, False)
        self.k_norm = LayerNorm(self.width, eps=1e-6, promote_fp32=False)
        self.weights_proj = Linear(c.hidden_size, self.heads, False)
        self.bmm, self.relu, self.select = BMM(), ReLU(), TopKPerRow()

    def forward(self, hidden, latent, positions, rotary, offset):
        batch, length = hidden.shape[:2]
        q = self.wq_b(latent).reshape(batch*length, self.heads, self.width)
        k = self.k_norm(self.wk(hidden)).reshape(batch*length, 1, self.width)
        qrot, krot = rotary(positions, q[..., :self.rope_width].contiguous(), k[..., :self.rope_width].contiguous())
        q = torch.cat((qrot.view(batch, length, self.heads, self.rope_width),
                       q[..., self.rope_width:].reshape(batch, length, self.heads, -1)), -1)
        k = torch.cat((krot.view(batch, length, self.rope_width),
                       k[..., self.rope_width:].reshape(batch, length, -1)), -1)
        end = offset+length
        self.keys[:, offset:end] = k
        keys = self.keys[:, :end]
        scores = self.bmm(q.float().reshape(batch, length*self.heads, self.width), keys.float().transpose(-1, -2))
        scores = self.relu(scores.reshape(batch, length, self.heads, end)*self.width**-.5)
        weights = self.weights_proj(hidden).float()*self.heads**-.5
        scores = self.bmm(weights.reshape(batch*length, 1, self.heads), scores.reshape(batch*length, self.heads, end))
        scores = scores.reshape(batch, length, end)
        causal = torch.arange(end, device=hidden.device)[None, :] > torch.arange(offset, end, device=hidden.device)[:, None]
        scores = scores.masked_fill(causal, torch.finfo(hidden.dtype).min)
        rows = scores.reshape(batch*length, end).contiguous()
        starts = torch.zeros(rows.shape[0], dtype=torch.int32, device=hidden.device)
        ends = torch.full_like(starts, end)
        indices = self.select.forward_prefill(rows, starts, ends, min(self.topk, end))
        return indices.long().reshape(batch, length, -1), causal


class SparseAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.nope, self.rope_width, self.value = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        self.rank = c.kv_lora_rank
        self.q_a_proj = Linear(c.hidden_size, c.q_lora_rank, False)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, 1e-6)
        self.q_b_proj = Linear(c.q_lora_rank, self.heads*(self.nope+self.rope_width), False)
        self.kv_a_proj_with_mqa = Linear(c.hidden_size, self.rank+self.rope_width, False)
        self.kv_a_layernorm = RMSNorm(self.rank, 1e-6)
        self.kv_b_proj = Linear(self.rank, self.heads*(self.nope+self.value), False)
        self.o_proj = Linear(self.heads*self.value, c.hidden_size, False)
        self.indexer = Indexer(c)
        self.rotary_emb = NativeRotary(self.rope_width, c.max_position_embeddings, c.rope_parameters['rope_theta'])
        self.attend = DenseAttention(backend='sdpa')

    def forward(self, positions, hidden):
        count = hidden.shape[0]
        batch, length, offset = self.batch, count//self.batch, self.offset
        latent = self.q_a_layernorm(self.q_a_proj(hidden))
        q = self.q_b_proj(latent).reshape(count, self.heads, self.nope+self.rope_width)
        kv, krot = self.kv_a_proj_with_mqa(hidden).split((self.rank, self.rope_width), -1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv.contiguous())).reshape(count, self.heads, self.nope+self.value)
        kplain, value = kv.split((self.nope, self.value), -1)
        qrot, krot = self.rotary_emb(positions, q[..., self.nope:].contiguous(), krot.contiguous())
        q = torch.cat((q[..., :self.nope], qrot.reshape(count, self.heads, self.rope_width)), -1).reshape(batch, length, self.heads, -1)
        key = torch.cat((kplain, krot.reshape(count, 1, self.rope_width).expand(-1, self.heads, -1)), -1)
        end = offset+length
        self.keys[:, offset:end] = key.reshape(batch, length, self.heads, -1)
        self.values[:, offset:end] = value.reshape(batch, length, self.heads, self.value)
        indices, causal = self.indexer(hidden.reshape(batch, length, -1), latent.reshape(batch, length, -1), positions, self.rotary_emb, offset)
        mask = torch.full((batch, length, end), float('-inf'), dtype=hidden.dtype, device=hidden.device)
        mask.scatter_(-1, indices, 0.)
        mask = mask.masked_fill(causal, float('-inf'))
        output = self.attend(q, self.keys[:, :end], self.values[:, :end],
                             softmax_scale=(self.nope+self.rope_width)**-.5, attn_mask=mask[:, None])
        return self.o_proj(output.reshape(count, -1))


def build_from_config(config, device, dtype):
    if (config.attention_bias or config.q_lora_rank is None or config.rope_parameters['rope_type'] != 'default'
            or any(kind != 'full' for kind in config.indexer_types)):
        raise ValueError('The GLM-5 checkpoint computes a fresh default-RoPE sparse index in every layer')
    model = LlamaForCausalLM(decoder_config(config, dtype))
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = SparseAttention(config)
        if i >= config.first_k_dense_replace:
            layer.mlp = NormalizedExperts(
                hidden_size=config.hidden_size, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                routing='sigmoid', keep_router_weights_fp32=True,
                num_expert_group=config.n_group, topk_group=config.topk_group,
                shared_expert_intermediate_size=config.n_shared_experts*config.moe_intermediate_size,
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
            direct[p+name+'.weight'] = getattr(layer, name).weight
        for name, parameter in layer.self_attn.named_parameters():
            direct[p+'self_attn.'+name] = parameter
        if i < config.first_k_dense_replace:
            target, stem = layer.mlp, p+'mlp.'
        else:
            expert = layer.mlp
            direct[p+'mlp.gate.weight'] = expert.gate.weight
            direct[p+'mlp.gate.e_score_correction_bias'] = expert.gate.e_score_correction_bias
            direct[p+'mlp.experts.gate_up_proj'] = expert.w13
            direct[p+'mlp.experts.down_proj'] = expert.w2
            target, stem = expert.shared_expert, p+'mlp.shared_experts.'
        direct[stem+'down_proj.weight'] = target.down_proj.weight
        for shard, name in enumerate(('gate_proj', 'up_proj')):
            packed[stem+name+'.weight'] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys()|packed.keys():
        raise KeyError(f'GLM-5 state mismatch: {sorted(state_dict.keys() ^ (direct.keys()|packed.keys()))}')
    for name, target in direct.items():
        if target.shape != state_dict[name].shape or target.dtype != state_dict[name].dtype:
            raise ValueError(f'GLM-5 shape/dtype mismatch: {name}')
        target.copy_(state_dict[name])
    for name, (target, shard) in packed.items():
        target.weight_loader(target, state_dict[name], shard)


def make_workloads(model, inputs, config):
    tokens = inputs['input_ids']
    batch, total = tokens.shape
    prompt = total-1
    attentions = [layer.self_attn for layer in model.model.layers]
    parameter = next(model.parameters())
    for attention in attentions:
        attention.batch = batch
        attention.keys = torch.zeros(batch, total, config.num_attention_heads, config.qk_head_dim, device=tokens.device, dtype=parameter.dtype)
        attention.values = torch.zeros(batch, total, config.num_attention_heads, config.v_head_dim, device=tokens.device, dtype=parameter.dtype)
        attention.indexer.keys = torch.zeros(batch, total, config.index_head_dim, device=tokens.device, dtype=parameter.dtype)

    def run(ids, offset):
        length = ids.shape[1]
        for attention in attentions:
            attention.offset = offset
        positions = torch.arange(offset, offset+length, device=tokens.device).repeat(batch)
        hidden = model.model(ids.reshape(-1), positions)
        logits = model.lm_head.linear_op(hidden, model.lm_head.embedding_op.emb.weight)
        return {'logits': logits.reshape(batch, length, -1)}

    def prepare_prefill():
        for attention in attentions:
            attention.keys.zero_()
            attention.values.zero_()
            attention.indexer.keys.zero_()

    def prepare_decode():
        prepare_prefill()
        run(tokens[:, :prompt], 0)

    return {'prefill': Workload(run=lambda: run(tokens[:, :prompt], 0), prepare=prepare_prefill),
            'decode': Workload(run=lambda: run(tokens[:, -1:], prompt), prepare=prepare_decode)}
