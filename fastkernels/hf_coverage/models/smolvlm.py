"""SmolVLM image encoding and native greedy text generation from existing ops."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from ..patches.codec_top1 import CodecTop1
from ..runner import Workload
from . import deepseek_vl, idefics3
from .modernvbert import ImagePaddingFilter
from .qwen2_5_omni import TextMLP


class TextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = Linear(config.hidden_size, self.heads * self.dim, bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias)
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, bias=config.attention_bias)
        self.attention = DenseAttention(backend='sdpa')
        self.key = self.value = None

    def forward(self, hidden, positions, rotary):
        batch, length, _ = hidden.shape
        q, k = self.q_proj(hidden), self.k_proj(hidden)
        q, k = rotary.forward_native(positions.repeat(batch), q.reshape(batch * length, -1),
                                     k.reshape(batch * length, -1), self.dim, rotary.cos_sin_cache.to(q.dtype))
        q = q.reshape(batch, length, self.heads, self.dim)
        k = k.reshape(batch, length, self.kv_heads, self.dim)
        v = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.dim)
        fresh = self.key is None
        self.key = k if fresh else torch.cat((self.key, k), dim=1)
        self.value = v if fresh else torch.cat((self.value, v), dim=1)
        key = self.key.repeat_interleave(self.heads // self.kv_heads, dim=2)
        value = self.value.repeat_interleave(self.heads // self.kv_heads, dim=2)
        output = self.attention(q, key, value, causal=fresh and length > 1)
        return self.o_proj(output.reshape(batch, length, -1))


class TextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.mlp = TextAttention(config), TextMLP(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions, rotary):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, rotary)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TextDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != 'silu' or config.mlp_bias:
            raise ValueError('Selected decoder requires bias-free SiLU MLP')
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(TextLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        rope = config.rope_parameters
        if rope['rope_type'] not in ('default', 'llama3'):
            raise ValueError('Unsupported decoder rotary configuration')
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, rope['rope_theta'],
            rope_scaling_factor=rope.get('factor', 1.), rope_low_freq_factor=rope.get('low_freq_factor', 1.),
            rope_high_freq_factor=rope.get('high_freq_factor', 1.),
            rope_original_max_position_embeddings=rope.get('original_max_position_embeddings', config.max_position_embeddings))

    def reset(self):
        for layer in self.layers:
            layer.self_attn.key = layer.self_attn.value = None

    def forward(self, hidden, positions):
        for layer in self.layers:
            hidden = layer(hidden, positions, self.rotary)
        return self.norm(hidden)


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = idefics3.vision_model(config)
        self.attention = DenseAttention(backend='cudnn')

    def forward(self, pixels, pixel_mask=None):
        c = self.config
        if pixel_mask is None:
            pixel_mask = torch.ones(pixels.shape[0], *pixels.shape[-2:], dtype=torch.bool, device=pixels.device)
        patch_mask = pixel_mask.unfold(1, c.patch_size, c.patch_size).unfold(2, c.patch_size, c.patch_size).sum((-1, -2)) > 0
        batch, height, width = patch_mask.shape
        side = c.image_size // c.patch_size
        boundaries = torch.arange(1 / side, 1., 1 / side, device=pixels.device)
        # All arithmetic below depends only on the supplied pixel mask.
        hstep = 1.0 / patch_mask[:, :, 0].sum(1)
        wstep = 1.0 / patch_mask[:, 0, :].sum(1)
        hcoords = torch.arange(height, device=pixels.device).float()[None] * hstep[:, None]
        wcoords = torch.arange(width, device=pixels.device).float()[None] * wstep[:, None]
        hids = torch.bucketize(hcoords.clamp(max=1.-1e-6).to(pixels.dtype), boundaries, right=True)
        wids = torch.bucketize(wcoords.clamp(max=1.-1e-6).to(pixels.dtype), boundaries, right=True)
        ids = (hids[:, :, None] * side + wids[:, None, :]).masked_fill(~patch_mask, 0).flatten(1)
        hidden = self.core.patch_embedding(pixels).flatten(2).transpose(1, 2)
        hidden = hidden + self.core.position_embedding[0][ids]
        # Native SDPA omits an all-valid mask; even a zero mask changes its
        # numerical execution. Keep genuine padding in the measured forward.
        mask = None
        if not bool(patch_mask.all()):
            mask = hidden.new_zeros(batch, 1, 1, height * width).masked_fill(
                ~patch_mask.flatten(1)[:, None, None], torch.finfo(hidden.dtype).min,
            )
        for layer in self.core.layers:
            norm = layer.layer_norm1(hidden)
            attn = layer.self_attn
            shape = (batch, height * width, c.num_attention_heads, c.hidden_size // c.num_attention_heads)
            q, k, v = (getattr(attn, name)(norm).reshape(shape) for name in ('q_proj', 'k_proj', 'v_proj'))
            hidden = hidden + attn.out_proj(self.attention(q, k, v, attn_mask=mask).reshape_as(hidden))
            hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
        return self.core.post_layernorm(hidden)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = Vision(config.vision_config)
        self.connector = idefics3.Connector(config)
        self.text_model = TextDecoder(config.text_config)
        self.lm_head = Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.filter, self.select = ImagePaddingFilter(), CodecTop1()

    def features(self, pixels, pixel_attention_mask=None):
        flat = pixels.flatten(0, 1)
        if pixel_attention_mask is not None:
            # Reuse the same selection operation on image values to retain
            # corresponding supplied masks, including the empty-image fallback.
            indices = torch.arange(flat.shape[0], device=flat.device)
            # Filter exposes selected values, so identify masks via exact
            # discrete indices obtained by running its existing gate operations.
            values = self.filter.relu(flat) + self.filter.relu(-flat)
            offsets = indices.new_tensor([i * flat[0].numel() for i in range(flat.shape[0] + 1)])
            maxima = self.filter.reduce(values.flatten(), offsets, reduce='max')
            nonzero = self.filter.top1(torch.stack((torch.zeros_like(maxima), maxima), -1))
            maximum = self.filter.reduce(nonzero.float(), offsets.new_tensor([0, flat.shape[0]]), reduce='max')
            any_image = self.filter.top1(torch.stack((torch.zeros_like(maximum), maximum), -1)).bool()
            keep = nonzero.bool()
            keep[0] |= ~any_image[0]
            flat = flat[keep].contiguous()
            pixel_attention_mask = pixel_attention_mask.flatten(0, 1)[keep]
        else:
            flat = self.filter(flat)
        return self.connector(self.vision_model(flat, pixel_attention_mask))

    def generate(self, input_ids, pixel_values, pixel_attention_mask=None, max_new_tokens=4,
                 eos_token_id=49279, pad_token_id=2, use_cache=True, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('The selected public generation case has one prompt with multiple images')
        self.text_model.reset()
        ids = input_ids.clone()
        outputs = {}
        for step in range(max_new_tokens):
            current = ids if step == 0 or not use_cache else ids[:, -1:]
            hidden = self.text_model.embed_tokens(current)
            if step == 0 or not use_cache:
                if not use_cache:
                    self.text_model.reset()
                images = self.features(pixel_values, pixel_attention_mask)
                hidden = hidden.masked_scatter((current == self.config.image_token_id)[..., None].expand_as(hidden), images)
            start = 0 if step == 0 or not use_cache else ids.shape[1] - 1
            positions = torch.arange(start, ids.shape[1], device=ids.device)
            hidden = self.text_model(hidden, positions)
            logits = self.lm_head(hidden[:, -1:])[:, -1].float()
            outputs[f'logits.{step}'] = logits
            token = self.select(logits).reshape(1, 1)
            ids = torch.cat((ids, token), dim=1)
            if int(token[0, 0]) == eos_token_id:
                break
        outputs['sequences'] = ids
        if use_cache:
            for index, layer in enumerate(self.text_model.layers):
                outputs[f'past_key_values.{index}.key'] = layer.self_attn.key.transpose(1, 2)
                outputs[f'past_key_values.{index}.value'] = layer.self_attn.value.transpose(1, 2)
        return outputs


def build_from_config(config, device, dtype):
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    deepseek_vl.load_vision(model.vision_model.core, remaining, 'model.vision_model.')
    model.connector.proj.load_state_dict({'weight': remaining.pop('model.connector.modality_projection.proj.weight')}, strict=True)
    mapped = {}
    for name in model.text_model.state_dict():
        mapped[name] = remaining.pop('model.text_model.' + name.replace('.emb.weight', '.weight'))
    model.text_model.load_state_dict(mapped, strict=True)
    model.lm_head.load_state_dict({'weight': remaining.pop('lm_head.weight')}, strict=True)
    if remaining:
        raise KeyError(f'Unmapped SmolVLM weights: {sorted(remaining)}')


def make_workloads(model, inputs, config, case=None):
    options = {} if case is None else dict(case['generation_kwargs'])
    if options.pop('do_sample', False):
        raise ValueError('Selected public workload uses greedy generation')
    return {'generate': Workload(run=lambda: model.generate(**inputs, **options))}
