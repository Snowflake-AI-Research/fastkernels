"""Fuyu patch projection and Persimmon decoder using existing child operations."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear
from . import persimmon
from .stablelm import PartialRotary
from .qwen2_precision import NativeRotaryEmbedding
from ..runner import Workload


class FuyuForCausalLM(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        text = config.text_config
        self.text = persimmon.build_from_config(text, device, dtype)
        self.vision_embed_tokens = Linear(config.patch_size**2 * config.num_channels, config.hidden_size)
        self.image_token_id = config.image_token_id
        self.heads, self.width = text.num_attention_heads, text.hidden_size // text.num_attention_heads
        self.attention = DenseAttention(backend='cudnn')
        self.use_cache = text.use_cache

    def forward(self, input_ids, image_patches):
        backbone = self.text.model
        hidden = backbone.embed_tokens(input_ids)
        images = self.vision_embed_tokens(image_patches)
        hidden[input_ids == self.image_token_id] = images.reshape(-1, images.shape[-1])
        batch, length, width = hidden.shape
        positions = torch.arange(length, device=hidden.device).repeat(batch)
        outputs = {}
        for index, layer in enumerate(backbone.layers):
            a = layer.self_attn
            query, key, value = a.qkv_proj(layer.input_layernorm(hidden)).chunk(3, dim=-1)
            query = a.q_norm(query.reshape(batch * length, self.heads, self.width)).reshape(batch * length, width)
            key = a.k_norm(key.reshape(batch * length, self.heads, self.width)).reshape(batch * length, width)
            query, key = a.rotary_emb(positions, query, key)
            query = query.view(batch, length, self.heads, self.width)
            key = key.view(batch, length, self.heads, self.width)
            value = value.view(batch, length, self.heads, self.width)
            hidden = hidden + a.o_proj(self.attention(query, key, value, causal=True).reshape(batch, length, width))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            if self.use_cache:
                outputs[f'past_key_values.{index}.key'] = key.transpose(1, 2)
                outputs[f'past_key_values.{index}.value'] = value.transpose(1, 2)
        hidden = backbone.norm(hidden)
        head = self.text.lm_head
        outputs['logits'] = head.linear_op(hidden, head.embedding_op.emb.weight)
        return outputs


def build_from_config(config, device, dtype):
    if config.hidden_size != config.text_config.hidden_size or config.tie_word_embeddings:
        raise ValueError('Fuyu case preserves the matching projection width and untied language head')
    model = FuyuForCausalLM(config, device, dtype).to(device=device, dtype=dtype).eval()
    text = config.text_config
    width = text.hidden_size // text.num_attention_heads
    rotary = PartialRotary(width, width // 2, text.max_position_embeddings,
                            text.rope_parameters['rope_theta']).to(device=device)
    rotary.rotary = NativeRotaryEmbedding(width // 2, text.max_position_embeddings,
                                          text.rope_parameters['rope_theta']).to(device=device)
    for layer in model.text.model.layers:
        layer.self_attn.rotary_emb = rotary
    return model


def load_state_dict_into(model, state_dict, config):
    text = {}
    for name, value in state_dict.items():
        if name.startswith('model.language_model.'):
            text[name.replace('model.language_model.', 'model.', 1)] = value
        elif name == 'lm_head.weight':
            text[name] = value
        elif name not in ('model.vision_embed_tokens.weight', 'model.vision_embed_tokens.bias'):
            raise KeyError(f'Unmapped Fuyu weight: {name}')
    persimmon.load_state_dict_into(model.text, text, config.text_config)
    model.vision_embed_tokens.load_state_dict({field: state_dict['model.vision_embed_tokens.' + field]
                                             for field in ('weight', 'bias')}, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
