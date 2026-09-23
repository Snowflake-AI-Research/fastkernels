"""Idefics2: SigLIP, gated projection, grouped-query resampler and Mistral."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.silu import SiLU
from ..patches.product_gate import ProductGate
from ..runner import Workload
from . import deepseek_vl, idefics3, llama, mistral
from .idefics3 import square_position_ids


class GatedMLP(nn.Module):
    def __init__(self, width, intermediate, output):
        super().__init__()
        self.gate_proj, self.up_proj = (Linear(width, intermediate, bias=False) for _ in range(2))
        self.down_proj = Linear(intermediate, output, bias=False)
        self.activation, self.product = SiLU(), ProductGate()

    def forward(self, hidden):
        packed = torch.cat((self.activation(self.gate_proj(hidden)), self.up_proj(hidden)), dim=-1)
        return self.down_proj(self.product(packed))


class ResamplerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads, self.kv_heads, self.dim = config.resampler_n_heads, config.num_key_value_heads, config.resampler_head_dim
        self.q_proj = Linear(width, self.heads * self.dim, bias=False)
        self.k_proj, self.v_proj = (Linear(width, self.kv_heads * self.dim, bias=False) for _ in range(2))
        self.o_proj = Linear(self.heads * self.dim, width, bias=False)
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, latents, context):
        batch = latents.shape[0]
        joined = torch.cat((context, latents), dim=1)
        query = self.q_proj(latents).reshape(batch, -1, self.heads, self.dim)
        key, value = [projection(joined).reshape(batch, -1, self.kv_heads, self.dim)
                      .repeat_interleave(self.heads // self.kv_heads, dim=2)
                      for projection in (self.k_proj, self.v_proj)]
        return self.o_proj(self.attention(query, key, value).reshape(batch, latents.shape[1], -1))


class ResamplerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, eps = config.hidden_size, config.rms_norm_eps
        self.input_latents_norm = RMSNormNative(width, eps)
        self.input_context_norm = RMSNormNative(width, eps)
        self.post_attention_layernorm = RMSNormNative(width, eps)
        self.self_attn = ResamplerAttention(config)
        self.mlp = GatedMLP(width, width * 4, width)

    def forward(self, latents, context):
        latents = latents + self.self_attn(self.input_latents_norm(latents), self.input_context_norm(context))
        return latents + self.mlp(self.post_attention_layernorm(latents))


class Resampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latents = nn.Parameter(torch.empty(config.resampler_n_latents, config.hidden_size))
        self.layers = nn.ModuleList(ResamplerLayer(config) for _ in range(config.resampler_depth))
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, context):
        hidden = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)
        for layer in self.layers:
            hidden = layer(hidden, context)
        return self.norm(hidden)


class Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modality_projection = GatedMLP(config.vision_config.hidden_size,
            config.text_config.intermediate_size, config.text_config.hidden_size)
        self.perceiver_resampler = Resampler(config.perceiver_config)

    def forward(self, hidden):
        return self.perceiver_resampler(self.modality_projection(hidden))


class Backbone(idefics3.Backbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text = text
        self.real_images = idefics3.RealImages(3, config.vision_config.image_size)
        self.vision = idefics3.vision_model(config.vision_config)
        self.connector = Connector(config)
        self.image_token_id = config.image_token_id
        self.pixel_values = self.image_hidden_states = None

    def features(self, pixels):
        return super().features(pixels).flatten(0, 1)


def build_from_config(config, device, dtype):
    if config.perceiver_config.hidden_act != 'silu':
        raise ValueError('Preserve the SiLU gated resampler')
    language = mistral.build_from_config(config.text_config, device, dtype)
    return idefics3.Model(language, config, Backbone).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace('model.text_model.', 'model.'): remaining.pop(name)
            for name in list(remaining) if name.startswith('model.text_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    deepseek_vl.load_vision(model.model.vision, remaining, 'model.vision_model.')
    # The selected square image has fixed patch metadata. Native Idefics2 rounds
    # fractional coordinates through pixel dtype before selecting position rows.
    # Prepare that fixed row selection once; retain the existing vision forward.
    positions = model.model.vision.position_embedding
    device = 'cpu' if positions.is_meta else positions.device
    indices = square_position_ids(config.vision_config, positions.dtype, device).to(positions.device)
    positions.copy_(positions.index_select(1, indices))
    model.model.connector.load_state_dict({name: remaining.pop('model.connector.' + name)
        for name in model.model.connector.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f'Unmapped Idefics2 weights: {sorted(remaining)}')


def make_workloads(model, inputs, config, *, case=None):
    expected = (config.vision_config.image_size, config.vision_config.image_size)
    if tuple(inputs['pixel_values'].shape[-2:]) != expected:
        raise ValueError('This Idefics2 workload requires its selected full square image geometry')
    if 'pixel_attention_mask' in inputs and not inputs['pixel_attention_mask'].bool().all():
        raise ValueError('This Idefics2 workload requires fully valid image patches')
    model.model.pixel_values = inputs['pixel_values']
    workloads = llama.make_workloads(
        model, {'input_ids': inputs['input_ids']}, model.config, case=case,
        cache_windows=[config.text_config.sliding_window] * config.text_config.num_hidden_layers,
    )
    prefill = workloads['prefill']

    def run_prefill():
        return {**prefill.run(), 'image_hidden_states': model.model.image_hidden_states}

    workloads['prefill'] = Workload(run=run_prefill, prepare=prefill.prepare, collect=prefill.collect)
    return workloads
