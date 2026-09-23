"""Llama4's text-only path: chunked attention and input-weighted sparse experts.

The multimodal tower is outside this explicitly selected text-only workload.
All numerical work uses existing operations; native rotary/SwiGLU callables are
reused unchanged. No claim is made about image-conditioned execution.
"""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.detector_topk import DetectorTopK
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.sam3_rope_attention import _apply_rotary_enc
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm

from .cohere2 import Cache, make_workloads


class UnweightedNorm(GemmaRMSNorm):
    """Zero-offset existing Gemma norm implements HF's unweighted RMS norm."""

    def __init__(self, width, eps):
        super().__init__(width, eps)
        self.weight.requires_grad_(False)

    def forward(self, x):
        return self.forward_native(x)


class MLP(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate_proj = Linear(width, intermediate, bias=False)
        self.up_proj = Linear(width, intermediate, bias=False)
        self.down_proj = Linear(intermediate, width, bias=False)

    def forward(self, hidden):
        packed = torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), dim=-1)
        return self.down_proj(SiluAndMul.forward_native(packed))


class InputWeightedMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList(MLP(config.hidden_size, config.intermediate_size)
                                    for _ in range(config.num_local_experts))
        self.shared_expert = MLP(config.hidden_size, config.intermediate_size)
        self.select, self.sigmoid, self.product = DetectorTopK(), Sigmoid(), ProductGate()

    def forward(self, hidden):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits = self.router(flat)
        values, indices = self.select(logits, 1)
        scores = self.sigmoid(values.float()).to(hidden.dtype)
        routed = torch.zeros_like(flat)
        # Routing IDs are metadata. Top-one gives disjoint destinations, so no
        # reduction or raw activation comparison is hidden in this dispatch.
        for index, expert in enumerate(self.experts):
            rows = torch.where(indices[:, 0] == index)[0]
            if rows.numel():
                selected = flat[rows]
                weighted = self.product(torch.cat((selected, scores[rows].expand_as(selected)), dim=-1))
                routed[rows] = expert(weighted)
        return (self.shared_expert(flat) + routed).reshape(shape)


class Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        width = config.hidden_size
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.use_rope = bool(config.no_rope_layers[index])
        self.chunk = config.attention_chunk_size if config.layer_types[index] == 'chunked_attention' else None
        self.temperature = config.attn_temperature_tuning and not self.use_rope
        self.floor_scale, self.attn_scale = config.floor_scale, config.attn_scale
        self.input_layernorm = T5LayerNorm(width, config.rms_norm_eps)
        self.post_attention_layernorm = T5LayerNorm(width, config.rms_norm_eps)
        self.self_attn = nn.ModuleDict({
            'q_proj': Linear(width, self.heads * self.head_dim, bias=False),
            'k_proj': Linear(width, self.kv_heads * self.head_dim, bias=False),
            'v_proj': Linear(width, self.kv_heads * self.head_dim, bias=False),
            'o_proj': Linear(self.heads * self.head_dim, width, bias=False),
        })
        self.qk_norm = UnweightedNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm and self.use_rope else None
        self.attention = DenseAttention(backend='sdpa')
        self.temperature_product = ProductGate()
        self.feed_forward = InputWeightedMoE(config) if index in config.moe_layers else MLP(width, config.intermediate_size_mlp)

    def forward(self, hidden, positions, rotary_cache, previous=None):
        batch, length, _ = hidden.shape
        normed = self.input_layernorm(hidden)
        query, key, value = (self.self_attn[name + '_proj'](normed) for name in ('q', 'k', 'v'))
        query = query.reshape(batch, length, self.heads, self.head_dim)
        key = key.reshape(batch, length, self.kv_heads, self.head_dim)
        if self.use_rope:
            # Existing complex rotary callable has HF's complex multiply/cast
            # order; transpose only adapts its [B,H,S,D] layout.
            query, key = _apply_rotary_enc(query.transpose(1, 2), key.transpose(1, 2), rotary_cache[positions])
            query, key = query.transpose(1, 2), key.transpose(1, 2)
        if self.qk_norm is not None:
            query, key = self.qk_norm(query), self.qk_norm(key)
        if self.temperature:
            # Compute metadata scales here; apply them through the admitted product.
            scale = torch.log1p(torch.floor((positions.float() + 1) / self.floor_scale)) * self.attn_scale + 1
            scales = scale[None, :, None, None].expand_as(query)
            query = self.temperature_product(torch.cat((query.float(), scales), dim=-1)).to(hidden.dtype)
        key = key.transpose(1, 2)
        value = value.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        past_length = 0 if previous is None else previous[0].shape[2]
        if previous is not None:
            key, value = torch.cat((previous[0], key), dim=2), torch.cat((previous[1], value), dim=2)
        keys = torch.arange(key.shape[2], device=hidden.device) + positions[0] - past_length
        mask = keys[None, :] <= positions[:, None]
        if self.chunk is not None:
            mask = mask & ((keys[None, :] // self.chunk) == (positions[:, None] // self.chunk))
            state = (key[:, :, -(self.chunk - 1):].clone(), value[:, :, -(self.chunk - 1):].clone())
        else:
            state = (key, value)
        repeats = self.heads // self.kv_heads
        if repeats != 1:
            key, value = key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask)
        hidden = hidden + self.self_attn['o_proj'](context.reshape(batch, length, -1))
        return hidden + self.feed_forward(self.post_attention_layernorm(hidden)), state


class TextForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(Layer(config, index) for index in range(config.num_hidden_layers))
        self.norm = T5LayerNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None):
        start = 0 if past_key_values is None else past_key_values.seen_tokens
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + start
        hidden = self.embed_tokens(input_ids)
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, positions, self.rotary_cache,
                                  None if past_key_values is None else past_key_values.layers[index])
            states.append(state)
        return {'logits': self.lm_head(self.norm(hidden)),
                'past_key_values': Cache(tuple(states), start + input_ids.shape[1])}


def build_from_config(config, device, dtype):
    if (config.hidden_act != 'silu' or config.attention_bias or config.tie_word_embeddings
            or not config.use_cache or config.num_experts_per_tok != 1
            or config.output_router_logits or config.rope_parameters['rope_type'] != 'default'
            or config.head_dim % 2 or config.num_attention_heads % config.num_key_value_heads
            or config.attention_chunk_size is None or config.attention_chunk_size < 2
            or len(config.layer_types) != config.num_hidden_layers
            or len(config.no_rope_layers) != config.num_hidden_layers
            or set(config.layer_types) - {'full_attention', 'chunked_attention'}):
        raise ValueError('Llama4 text coverage requires default RoPE, cached bias-free SiLU and untied top-one experts')
    model = TextForCausalLM(config).to(device=device, dtype=dtype).eval()
    dimensions = torch.arange(0, config.head_dim, 2, dtype=torch.float32, device='cpu')
    frequencies = 1.0 / (config.rope_parameters['rope_theta'] ** (dimensions / config.head_dim))
    angles = torch.outer(torch.arange(config.max_position_embeddings, device=device, dtype=torch.float32), frequencies.to(device))
    # HF constructs complex coefficients in FP32; this table is metadata only.
    coefficients = torch.polar(torch.ones_like(angles), angles)
    model.register_buffer('rotary_cache', coefficients, persistent=False)
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped = {}
    consumed = set()
    def put(target, source, tensor=None):
        mapped[target] = state_dict[source] if tensor is None else tensor
        consumed.add(source)
    put('embed_tokens.emb.weight', 'model.embed_tokens.weight')
    put('norm.weight', 'model.norm.weight')
    put('lm_head.weight', 'lm_head.weight')
    for index, layer in enumerate(model.layers):
        p, h = f'layers.{index}.', f'model.layers.{index}.'
        for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            put(p + name, h + name)
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            put(p + 'self_attn.' + name + '.weight', h + 'self_attn.' + name + '.weight')
        if layer.qk_norm is not None:
            mapped[p + 'qk_norm.weight'] = torch.zeros_like(layer.qk_norm.weight)
        f = h + 'feed_forward.'
        if isinstance(layer.feed_forward, InputWeightedMoE):
            put(p + 'feed_forward.router.weight', f + 'router.weight')
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                put(p + 'feed_forward.shared_expert.' + name + '.weight', f + 'shared_expert.' + name + '.weight')
            packed = state_dict[f + 'experts.gate_up_proj']
            down = state_dict[f + 'experts.down_proj']
            for expert in range(config.num_local_experts):
                base = p + f'feed_forward.experts.{expert}.'
                put(base + 'gate_proj.weight', f + 'experts.gate_up_proj', packed[expert, :, :config.intermediate_size].T)
                put(base + 'up_proj.weight', f + 'experts.gate_up_proj', packed[expert, :, config.intermediate_size:].T)
                put(base + 'down_proj.weight', f + 'experts.down_proj', down[expert].T)
        else:
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                put(p + 'feed_forward.' + name + '.weight', f + name + '.weight')
    if consumed != set(state_dict):
        raise KeyError(f'Llama4 unexpected state keys: {sorted(set(state_dict) - consumed)}')
    model.load_state_dict(mapped, strict=True)
