"""Aria's vision-query projector and routed/shared-expert conditional decoder."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.topk_softmax import TopKSoftmax
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from . import deepseek_vl, llama


class AriaMoE(SharedExpertMoE):
    def __init__(self, config):
        super().__init__(hidden_size=config.hidden_size, num_experts=config.moe_num_experts,
                         top_k=config.moe_topk, moe_intermediate_size=config.intermediate_size,
                         shared_expert_intermediate_size=config.intermediate_size * config.moe_num_shared_experts)
        self.select, self.softmax = TopKSoftmax(), Softmax()
        # Preserve native softmax of selected logits in their storage dtype.
        self.use_trtllm = False

    def _route(self, router_logits):
        _, indices = self.select(router_logits, self.top_k)
        weights = self.softmax(router_logits.gather(-1, indices.long()))
        return weights.float(), indices


class QueryProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision, text = config.vision_config, config.text_config
        width = vision.hidden_size
        self.heads = vision.num_attention_heads
        self.query_counts = {int(k): v for k, v in config.projector_patch_to_query_dict.items()}
        self.query = nn.Parameter(torch.empty(max(self.query_counts.values()), width))
        self.layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.layer_norm_kv = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.q_proj, self.k_proj, self.v_proj = (Linear(width, width, bias=False) for _ in range(3))
        self.inner_q, self.inner_k, self.inner_v = (Linear(width, width) for _ in range(3))
        self.out_proj, self.linear = Linear(width, width), Linear(width, width)
        self.final_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.linear_in = Linear(width, text.hidden_size, bias=False)
        self.linear_out = Linear(text.hidden_size, text.hidden_size, bias=False)
        self.activation, self.bmm, self.softmax = NewGELUActivation(), BMM(), Softmax()
        self.average_attention = GlobalAvgPool2d()

    def forward(self, features):
        batch, patches, width = features.shape
        count = self.query_counts[patches]
        query = self.q_proj(self.layer_norm(self.query[:count].unsqueeze(0).expand(batch, -1, -1)))
        normalized = self.layer_norm_kv(features)
        key, value = self.k_proj(normalized), self.v_proj(normalized)
        dim = width // self.heads
        query, key, value = [projection(tensor).reshape(batch, -1, self.heads, dim).transpose(1, 2)
                             for projection, tensor in ((self.inner_q, query), (self.inner_k, key), (self.inner_v, value))]
        scores = self.bmm(query * dim ** -0.5, key.transpose(-1, -2))
        probabilities = self.softmax(scores)
        attended = self.bmm(probabilities, value).transpose(1, 2).reshape(batch, count, width)
        # MultiheadAttention's ordinary call also computes averaged head weights,
        # even though Aria discards that returned tensor.
        self.average_attention(probabilities.permute(0, 2, 3, 1).reshape(batch * count, patches, self.heads, 1))
        hidden = self.final_norm(self.linear(self.out_proj(attended)))
        return self.linear_out(self.activation(self.linear_in(hidden)))


class AriaBackbone(deepseek_vl.DeepseekBackbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text = text
        self.vision = deepseek_vl.make_vision(config.vision_config)
        for layer in self.vision.layers:
            layer.mlp.act = GELU('tanh')
        self.projector = QueryProjector(config)
        self.image_token_id = config.image_token_index
        self.pixel_values = self.image_hidden_states = None

    def features(self, pixels):
        hidden = self.vision.patch_embedding(pixels).flatten(2).transpose(1, 2) + self.vision.position_embedding
        for layer in self.vision.layers:
            hidden = layer(hidden)
        # HF selects the final hidden state before the executed post-normalization.
        self.vision.post_layernorm(hidden)
        return self.projector(hidden)


class AriaModel(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = AriaBackbone(language.model, config)


def build_from_config(config, device, dtype):
    if (config.vision_feature_layer != -1 or config.text_config.attention_bias
            or config.text_config.mlp_bias or config.vision_config.hidden_act != 'gelu_pytorch_tanh'):
        raise ValueError('Preserve Aria final vision features, tanh GELU and bias-free expert decoder')
    language = llama.build_from_config(config.text_config, device, dtype)
    for layer in language.model.layers:
        layer.mlp = AriaMoE(config.text_config)
    return AriaModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    def copy(parameter, name):
        value = remaining.pop(name)
        if parameter.shape != value.shape:
            raise ValueError(f'Aria shape mismatch for {name}')
        parameter.copy_(value)

    backbone = model.model.text
    copy(backbone.embed_tokens.embedding_op.emb.weight, 'model.language_model.embed_tokens.weight')
    copy(backbone.norm.weight, 'model.language_model.norm.weight')
    copy(model.lm_head.embedding_op.emb.weight, 'lm_head.weight')
    for i, layer in enumerate(backbone.layers):
        prefix = f'model.language_model.layers.{i}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            copy(getattr(layer, name).weight, prefix + name + '.weight')
        attention = layer.self_attn
        for shard in ('q', 'k', 'v'):
            attention.qkv_proj.weight.weight_loader(attention.qkv_proj.weight,
                remaining.pop(prefix + f'self_attn.{shard}_proj.weight'), shard)
        copy(attention.o_proj.weight, prefix + 'self_attn.o_proj.weight')
        expert = layer.mlp
        prefix = f'model.language_model.layers.{i}.mlp.'
        expert.gate.weight.copy_(remaining.pop(prefix+'router.weight'))
        expert.w13.copy_(remaining.pop(prefix+'experts.fc1.weight').transpose(1, 2))
        expert.w2.copy_(remaining.pop(prefix+'experts.fc2.weight').transpose(1, 2))
        for shard, name in enumerate(('gate_proj','up_proj')):
            expert.shared_expert.gate_up_proj.weight.weight_loader(expert.shared_expert.gate_up_proj.weight,
                remaining.pop(prefix+'shared_experts.'+name+'.weight'), shard)
        expert.shared_expert.down_proj.weight.copy_(remaining.pop(prefix+'shared_experts.down_proj.weight'))
    deepseek_vl.load_vision(model.model.vision, remaining, 'model.vision_tower.')
    projector = model.model.projector
    mapped = {}
    for name in projector.state_dict():
        if name == 'query': source = name
        elif name.startswith('final_norm.'):source = name.replace('final_norm.', 'layer_norm.')
        elif name.startswith(('linear_in.','linear_out.')):source = 'feed_forward.'+name
        elif name.startswith(('inner_q.','inner_k.','inner_v.')):
            continue
        elif name.startswith('out_proj.'):source = 'cross_attn.multihead_attn.'+name
        else:source = 'cross_attn.'+name
        mapped[name] = remaining.pop('model.multi_modal_projector.'+source)
    for field in ('weight','bias'):
        parts = remaining.pop('model.multi_modal_projector.cross_attn.multihead_attn.in_proj_'+field).chunk(3,dim=0)
        for name, value in zip(('inner_q','inner_k','inner_v'),parts):mapped[name+'.'+field]=value
    projector.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f'Unmapped Aria weights: {sorted(remaining)}')


def make_workloads(model, inputs, config):
    # The pinned conditional-generation wrapper does not populate its declared
    # image_hidden_states field. The entire frontend still runs during prefill.
    model.model.pixel_values = inputs['pixel_values']
    return llama.make_workloads(model, {'input_ids': inputs['input_ids']}, model.config)
