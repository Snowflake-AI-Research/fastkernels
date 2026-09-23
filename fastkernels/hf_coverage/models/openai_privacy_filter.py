"""Privacy Filter's bidirectional sink attention and FP32 selected experts."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.bitnet_rms_norm import BitNetRMSNorm as RMSNorm
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.topk_softmax import TopKSoftmax
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from ..patches.oai_swiglu import OaiSwiGLU
from ..patches.product_gate import ProductGate
from ..runner import Workload


class SelectedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.count, self.topk = config.num_local_experts, config.num_experts_per_tok
        self.router = Linear(config.hidden_size, self.count)
        self.gate_up_proj = nn.Parameter(torch.empty(self.count, config.hidden_size, 2 * config.intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.count, 2 * config.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(self.count, config.intermediate_size, config.hidden_size))
        self.down_proj_bias = nn.Parameter(torch.empty(self.count, config.hidden_size))
        self.route, self.linear = TopKSoftmax(), Matmul()
        self.activation, self.product, self.reduce = OaiSwiGLU(), ProductGate(), MoeSum()

    def forward(self, hidden):
        original_shape, original_dtype = hidden.shape, hidden.dtype
        hidden = hidden.reshape(-1, hidden.shape[-1]).float()
        scores = self.linear(hidden, self.router.weight.float(), self.router.bias.float())
        weights, indices = self.route(scores, self.topk, renormalize=True)
        weights = weights / self.topk
        outputs = torch.empty(hidden.shape[0], self.topk, hidden.shape[-1], device=hidden.device, dtype=torch.float32)
        # Routing indices are computed by the existing router. Each selected
        # token/expert pair is projected once; this does not densify all experts.
        for expert in range(self.count):
            tokens, slots = torch.where(indices == expert)
            if not tokens.numel():
                continue
            projected = self.linear(hidden[tokens], self.gate_up_proj[expert].float().T,
                                    self.gate_up_proj_bias[expert].float())
            values = self.linear(self.activation(projected), self.down_proj[expert].float().T,
                                 self.down_proj_bias[expert].float())
            routing = weights[tokens, slots, None].expand_as(values)
            outputs[tokens, slots] = self.product(torch.cat((values, routing), dim=-1))
        return (self.reduce(outputs.reshape(-1, hidden.shape[-1]), self.topk)
                .to(original_dtype) * self.topk).reshape(original_shape)


class SinkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.width = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.window = config.sliding_window
        self.q_proj = Linear(config.hidden_size, self.heads * self.width)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.width)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.width)
        self.o_proj = Linear(self.heads * self.width, config.hidden_size)
        self.sinks = nn.Parameter(torch.empty(self.heads, dtype=torch.float32))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, positions, table):
        batch, length, _ = hidden.shape
        q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
        q, k = RotaryEmbedding.forward_native_interleaved(
            positions, q.reshape(batch * length, -1), k.reshape(batch * length, -1),
            self.width, table.to(hidden.dtype))
        q = q.reshape(batch, length, self.heads, self.width).transpose(1, 2) * self.width**-0.25
        k = k.reshape(batch, length, self.kv_heads, self.width).transpose(1, 2) * self.width**-0.25
        v = v.reshape(batch, length, self.kv_heads, self.width).transpose(1, 2)
        k, v = (x.repeat_interleave(self.heads // self.kv_heads, dim=1) for x in (k, v))
        # Materialize the HF BF16 score boundary before FP32 sink softmax.
        # Existing fused unified attention failed the complete model check;
        # this composition matches the isolated reference attention exactly.
        scores = self.bmm(q.reshape(-1, length, self.width), k.reshape(-1, length, self.width).transpose(1, 2))
        axis = torch.arange(length, device=hidden.device)
        scores = scores.masked_fill((axis[:, None] - axis[None, :]).abs() > self.window,
                                    torch.finfo(hidden.dtype).min)
        sinks = self.sinks[None, :, None, None].expand(batch, -1, length, 1).reshape(-1, length, 1)
        probabilities = self.softmax(torch.cat((scores, sinks), dim=-1))[..., :-1].to(v.dtype)
        context = self.bmm(probabilities, v.reshape(-1, length, self.width))
        context = context.reshape(batch, self.heads, length, self.width).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(context)


class PrivacyLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn, self.mlp = SinkAttention(config), SelectedExperts(config)

    def forward(self, hidden, positions, table):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, table)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class PrivacyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(PrivacyLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = config.rope_parameters
        # This library constructor multiplies its cache length by the factor.
        cache_base = (config.max_position_embeddings + int(rope['factor']) - 1) // int(rope['factor'])
        self.rotary_emb = YaRNRotaryEmbedding(config.head_dim, cache_base,
            rope['rope_theta'], rope['factor'], rope['original_max_position_embeddings'],
            rope['beta_fast'], rope['beta_slow'], rope['truncate'])

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(input_ids.shape[0])
        for layer in self.layers:
            hidden = layer(hidden, positions, self.rotary_emb.cos_sin_cache)
        return self.norm(hidden)


def build_from_config(config, device, dtype):
    if not config.attention_bias or config.rope_parameters['rope_type'] != 'yarn':
        raise ValueError('Selected Privacy Filter path uses biased projections and YaRN')
    model = PrivacyModel(config).to(device=device, dtype=dtype)
    for layer in model.layers:
        layer.self_attn.sinks.data = layer.self_attn.sinks.data.float()
    return model.eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for name, value in state_dict.items():
        target = name.replace('embed_tokens.weight', 'embed_tokens.emb.weight').replace('.mlp.experts.', '.mlp.')
        mapped[target] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: {'last_hidden_state': model(inputs['input_ids'])})}
