"""ModernBERT's alternating global/local attention and gated GELU encoder."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L3.gemma_dense_decoder_layer import GemmaMLP


def normalization(config):
    return LayerNorm(config.hidden_size, eps=config.norm_eps, create_offset=config.norm_bias, promote_fp32=False)


class ModernLayer(nn.Module):
    def __init__(self, config, index, causal):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        self.causal = causal
        self.window = config.local_attention // 2 if config.layer_types[index] == 'sliding_attention' else None
        self.attn_norm = nn.Identity() if index == 0 else normalization(config)
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.out_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention = DenseAttention(backend='sdpa')
        self.mlp_norm = normalization(config)
        self.mlp = GemmaMLP(config)
        rope = config.rope_parameters[config.layer_types[index]]
        self.rotary = RotaryEmbedding(self.width, config.max_position_embeddings, rope['rope_theta'])

    def forward(self, hidden, positions, past=None, attention_mask=None):
        batch, length = hidden.shape[:2]
        q, k, v = self.qkv(self.attn_norm(hidden)).chunk(3, dim=-1)
        # The reference rounds trig values to the model dtype, then rotates in FP32.
        cache = self.rotary.cos_sin_cache.to(q.dtype).float()
        index = positions.expand(batch, -1).reshape(-1)
        q, k = RotaryEmbedding.forward_native(index, q.reshape(-1, self.heads * self.width).float(),
                                             k.reshape(-1, self.heads * self.width).float(), self.width, cache)
        shape = (batch, length, self.heads, self.width)
        q, k, v = q.to(hidden.dtype).view(shape), k.to(hidden.dtype).view(shape), v.view(shape)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=1), torch.cat((past[1], v), dim=1)
        key_start = positions[0] - (0 if past is None else past[0].shape[1])
        keys = torch.arange(k.shape[1], device=hidden.device) + key_start
        mask = None
        if self.causal:
            mask = keys[None, :] <= positions[:, None]
            if self.window is not None:
                mask = mask & (keys[None, :] > positions[:, None] - self.window)
        elif self.window is not None:
            mask = (keys[None, :] - positions[:, None]).abs() <= self.window
        if attention_mask is not None:
            padding = attention_mask[:, None, None, :].bool()
            mask = padding if mask is None else mask & padding
        attended = self.attention(q, k, v, attn_mask=mask)
        hidden = hidden + self.out_proj(attended.reshape(batch, length, -1))
        if self.causal and self.window is not None:
            k, v = k[:, -(self.window - 1):].clone(), v[:, -(self.window - 1):].clone()
        return hidden + self.mlp(self.mlp_norm(hidden)), (k, v)


class ModernBertLM(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        self.embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embedding_norm = normalization(config)
        self.layers = nn.ModuleList([ModernLayer(config, i, causal) for i in range(config.num_hidden_layers)])
        self.final_norm = normalization(config)
        self.head_dense = Linear(config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.head_activation = GELU()
        self.head_norm = normalization(config)
        self.decoder = Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        if config.tie_word_embeddings:
            self.decoder.weight = self.embeddings.emb.weight

    def forward(self, input_ids, positions=None, past=None):
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embedding_norm(self.embeddings(input_ids))
        caches = []
        for i, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, positions, None if past is None else past[i])
            caches.append(cache)
        logits = self.decoder(self.head_norm(self.head_activation(self.head_dense(self.final_norm(hidden)))))
        return logits, caches


def build_modern(config, device, dtype, *, causal):
    if config.hidden_activation != 'gelu' or config.classifier_activation != 'gelu' or config.mlp_bias:
        raise ValueError('ModernBERT cases preserve the documented bias-free gated GELU MLP')
    if any(rope['rope_type'] != 'default' for rope in config.rope_parameters.values()):
        raise ValueError('ModernBERT case requires static per-layer-type rotary embeddings')
    model = ModernBertLM(config, causal).to(device=device, dtype=dtype).eval()
    # Position coefficients originate in FP32 regardless of model parameter dtype.
    for i, layer in enumerate(model.layers):
        layer.rotary = RotaryEmbedding(layer.width, config.max_position_embeddings,
                                       config.rope_parameters[config.layer_types[i]]['rope_theta']).to(device=device)
    return model


def build_from_config(config, device, dtype):
    return build_modern(config, device, dtype, causal=False)


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    weights = {}
    names = {'embeddings.emb': 'model.embeddings.tok_embeddings', 'embedding_norm': 'model.embeddings.norm',
             'final_norm': 'model.final_norm', 'head_dense': 'head.dense', 'head_norm': 'head.norm', 'decoder': 'decoder'}
    for name in model.state_dict():
        module, field = name.rsplit('.', 1)
        if name.startswith('layers.'):
            _, index, local = module.split('.', 2)
            prefix = f'model.layers.{index}.'
            if local == 'qkv' and prefix + 'attn.Wqkv.' + field not in remaining:
                weights[name] = torch.cat([remaining.pop(prefix + f'attn.{part}_proj.{field}') for part in ('q', 'k', 'v')])
                continue
            local = {'qkv': 'attn.Wqkv', 'out_proj': 'attn.Wo', 'mlp.gate_up_proj': 'mlp.Wi',
                     'mlp.down_proj': 'mlp.Wo'}.get(local, local)
            source = prefix + local + '.' + field
        else:
            source = names[module] + '.' + field
            if module in ('head_dense', 'head_norm') and source not in remaining:
                source = source.replace('head.', 'lm_head.', 1)
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped ModernBERT weights: {sorted(remaining)}')
    if config.tie_word_embeddings and not torch.equal(weights['decoder.weight'], weights['embeddings.emb.weight']):
        raise ValueError('Tied ModernBERT weights disagree')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    from ..runner import Workload
    return {'forward': Workload(run=lambda: {'logits': model(inputs['input_ids'])[0]})}
