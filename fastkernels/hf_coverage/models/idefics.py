"""Idefics: CLIP vision, normalized resampler, gated cross attention and text."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..runner import Workload, config_values
from . import llama
from .clip import ClipVisionModel
from .idefics2 import GatedMLP


class PerceiverAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, perceiver = config.vision_config.hidden_size, config.perceiver_config
        self.heads, self.dim = perceiver.resampler_n_heads, perceiver.resampler_head_dim
        for name in ('context_layer_norm', 'latents_layer_norm'):
            setattr(self, name, LayerNorm(width, eps=1e-5, promote_fp32=False))
        self.q_layer_norm = LayerNorm(self.dim, eps=1e-5, promote_fp32=False)
        self.k_layer_norm = LayerNorm(self.dim, eps=1e-5, promote_fp32=False)
        self.q_proj, self.k_proj, self.v_proj = (Linear(width, self.heads * self.dim, bias=False) for _ in range(3))
        self.output_proj = Linear(self.heads * self.dim, width, bias=False)
        self.bmm, self.softmax = BMM(), Softmax()
        count = (config.vision_config.image_size // config.vision_config.patch_size) ** 2 + 1
        self.maximum = MaxPool2d((1, count + perceiver.resampler_n_latents))

    def forward(self, context, latents):
        context, latents = self.context_layer_norm(context), self.latents_layer_norm(latents)
        joined = torch.cat((context, latents), dim=1)
        batch = context.shape[0]
        query = self.q_proj(latents).reshape(batch, -1, self.heads, self.dim).transpose(1, 2)
        key, value = [projection(joined).reshape(batch, -1, self.heads, self.dim).transpose(1, 2)
                      for projection in (self.k_proj, self.v_proj)]
        query, key = self.q_layer_norm(query), self.k_layer_norm(key)
        scores = self.bmm(query * self.dim ** -0.5, key.transpose(-1, -2))
        # Native Idefics explicitly rounds the max subtraction before softmax.
        probabilities = self.softmax(scores - self.maximum(scores))
        return self.output_proj(self.bmm(probabilities, value).transpose(1, 2).flatten(2))


class PerceiverMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.fc, self.c_proj = Linear(width, width * 4, bias=False), Linear(width * 4, width, bias=False)
        self.act = ReLU()

    def forward(self, hidden):
        return self.c_proj(self.act(self.fc(self.ln(hidden))))


class Resampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, p = config.vision_config.hidden_size, config.perceiver_config
        self.latents = nn.Parameter(torch.empty(p.resampler_n_latents, width))
        self.blocks = nn.ModuleList(nn.ModuleList((PerceiverAttention(config), PerceiverMLP(width)))
                                   for _ in range(p.resampler_depth))
        self.layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)

    def forward(self, context):
        hidden = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)
        for attention, mlp in self.blocks:
            hidden = hidden + attention(context, hidden)
            hidden = hidden + mlp(hidden)
        return self.layer_norm(hidden)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads, self.dim = config.num_attention_heads, width // config.num_attention_heads
        self.q_proj, self.o_proj = Linear(width, width, bias=False), Linear(width, width, bias=False)
        self.k_proj, self.v_proj = (Linear(config.vision_config.hidden_size, width, bias=False) for _ in range(2))
        self.q_layer_norm = RMSNormNative(self.dim, config.rms_norm_eps)
        self.k_layer_norm = RMSNormNative(self.dim, config.rms_norm_eps)
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, images, mask):
        batch, length, width = hidden.shape
        query = self.q_layer_norm(self.q_proj(hidden).reshape(batch, length, self.heads, self.dim))
        key = self.k_layer_norm(self.k_proj(images).reshape(batch, -1, self.heads, self.dim))
        value = self.v_proj(images).reshape(batch, -1, self.heads, self.dim)
        return self.o_proj(self.attention(query, key, value, attn_mask=mask[:, None]).reshape(batch, length, width))


class CrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.input_layernorm = RMSNormNative(width, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(width, config.rms_norm_eps)
        self.cross_attn = CrossAttention(config)
        self.mlp = GatedMLP(width, config.intermediate_size, width)
        self.alpha_cross_attn, self.alpha_dense = nn.Parameter(torch.empty(1)), nn.Parameter(torch.empty(1))
        self.register_buffer('cross_scale', torch.empty(1), persistent=False)
        self.register_buffer('dense_scale', torch.empty(1), persistent=False)

    def forward(self, hidden, images, mask):
        attended = self.cross_attn(self.input_layernorm(hidden), images, mask)
        attended = attended.masked_fill(~mask.any(dim=-1, keepdim=True), 0)
        hidden = hidden + attended * self.cross_scale
        return hidden + self.mlp(self.post_attention_layernorm(hidden)) * self.dense_scale


class Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision = ClipVisionModel(config.vision_config)
        for module in self.vision.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
        for layer in self.vision.encoder.layers:
            layer.mlp_act = GELU()
        self.perceiver_resampler = Resampler(config)
        self.gated_cross_attn_layers = nn.ModuleList(CrossLayer(config)
            for _ in range(config.num_hidden_layers // config.cross_layer_interval))
        self.interval = config.cross_layer_interval
        self.pixel_values = self.image_attention_mask = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, pixels):
        batch, count = pixels.shape[:2]
        hidden, _ = self.vision(pixels.flatten(0, 1))
        return self.perceiver_resampler(hidden).reshape(batch, count, -1, hidden.shape[-1])

    def forward(self, input_ids, positions):
        if get_context().is_prefill:
            self.image_hidden_states = self.features(self.pixel_values)
        batch = self.image_attention_mask.shape[0]
        indices = positions.reshape(batch, -1).unsqueeze(-1)
        mask = self.image_attention_mask.gather(1, indices.expand(-1, -1, self.image_attention_mask.shape[-1]))
        images = self.image_hidden_states
        mask = mask.repeat_interleave(images.shape[2], dim=-1)
        images = images.flatten(1, 2)
        hidden = self.text.embed_tokens(input_ids)
        residual = None
        for index, layer in enumerate(self.layers):
            if index % self.interval == 0:
                hidden = hidden if residual is None else hidden + residual
                hidden = self.gated_cross_attn_layers[index // self.interval](
                    hidden.reshape(images.shape[0], -1, hidden.shape[-1]), images, mask).flatten(0, 1)
                residual = None
            hidden, residual = layer(positions, hidden, residual)
        return self.text.norm(hidden, residual)[0]


class Model(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = Backbone(language.model, config)


def build_from_config(config, device, dtype):
    config = config_values(config.to_dict())
    config.vision_config.hidden_size = config.vision_config.embed_dim
    if (not config.use_resampler or not config.qk_layer_norms
            or not config.perceiver_config.qk_layer_norms_perceiver
            or config.alpha_type != 'float' or config.hidden_act != 'silu'
            or config.vision_config.hidden_act != 'gelu' or config.tie_word_embeddings):
        raise ValueError('Preserve the documented scalar-gated QK-normalized Idefics architecture')
    native = LlamaConfig(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_attention_heads, head_dim=config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size + config.additional_vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=10000., rope_scaling_factor=1., dtype=dtype)
    return Model(LlamaForCausalLM(native), config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name: remaining.pop(name) for name in list(remaining)
            if name.startswith(('model.layers.', 'model.norm.'))}
    text['model.embed_tokens.weight'] = torch.cat((remaining.pop('model.embed_tokens.weight'),
        remaining.pop('model.embed_tokens.additional_embedding.weight')))
    text['lm_head.weight'] = torch.cat((remaining.pop('lm_head.weight'), remaining.pop('lm_head.additional_fc.weight')))
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, model.config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name.replace('.emb.weight', '.weight').replace('embeddings.patch_embedding.proj.', 'embeddings.patch_embedding.')
        source = source.replace('.ln_1.', '.layer_norm1.').replace('.ln_2.', '.layer_norm2.')
        source = source.replace('.mlp_fc1.', '.mlp.fc1.').replace('.mlp_fc2.', '.mlp.fc2.')
        for field in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            source = source.replace('.' + field + '.', '.self_attn.' + field + '.')
        mapped[name] = remaining.pop('model.vision_model.' + source)
    model.model.vision.load_state_dict(mapped, strict=True)
    for field in ('perceiver_resampler', 'gated_cross_attn_layers'):
        module = getattr(model.model, field)
        module.load_state_dict({name: remaining.pop('model.' + field + '.' + name)
                               for name in module.state_dict()}, strict=True)
    for layer in model.model.gated_cross_attn_layers:
        # Fixed scalar transforms are prepared once after loading, outside timing.
        layer.cross_scale.copy_(layer.alpha_cross_attn.tanh())
        layer.dense_scale.copy_(layer.alpha_dense.tanh())
    if remaining:
        raise KeyError(f'Unmapped Idefics weights: {sorted(remaining)}')


def make_workloads(model, inputs, config, *, case=None):
    model.model.pixel_values = inputs['pixel_values']
    model.model.image_attention_mask = inputs['image_attention_mask']
    workloads = llama.make_workloads(model, {'input_ids': inputs['input_ids']}, model.config, case=case)
    for name in workloads:
        workload = workloads[name]
        def run(workload=workload):
            return {**workload.run(), 'image_hidden_states': model.model.image_hidden_states}
        workloads[name] = Workload(run=run, prepare=workload.prepare, collect=workload.collect)
    return workloads
