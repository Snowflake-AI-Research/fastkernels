"""GLM vision-language composition using existing normalization/gating/attention."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.conv3d import Conv3d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.dinov3_rope import apply_rot_embed_cat
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from ..patches.product_gate import ProductGate
from . import glm4, llama, qwen2, qwen2_vl
from .olmo2 import decoder_config
from .qwen2_precision import DenseCachedAttention, SeparateQKV


class GatedMLP(nn.Module):
    def __init__(self, width, intermediate, bias=False):
        super().__init__()
        self.gate_proj = Linear(width, intermediate, bias=bias)
        self.up_proj = Linear(width, intermediate, bias=bias)
        self.down_proj = Linear(intermediate, width, bias=bias)
        self.activation, self.product = SiLU(), ProductGate()

    def forward(self, hidden):
        packed = torch.cat((self.activation(self.gate_proj(hidden)), self.up_proj(hidden)), dim=-1)
        return self.down_proj(self.product(packed))


class VisionAttention(nn.Module):
    def __init__(self, config, ocr):
        super().__init__()
        self.heads = config.num_heads
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.proj = Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias if ocr else False)
        self.q_norm = RMSNormNative(config.hidden_size // self.heads, config.rms_norm_eps) if ocr else nn.Identity()
        self.k_norm = RMSNormNative(config.hidden_size // self.heads, config.rms_norm_eps) if ocr else nn.Identity()
        self.attention = DenseAttention('sdpa')

    def forward(self, hidden, rotary, lengths):
        q, k, v = self.qkv(hidden).reshape(len(hidden), 3, self.heads, -1).unbind(1)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rot_embed_cat(q.float(), rotary[:, None], half=True).to(hidden.dtype)
        k = apply_rot_embed_cat(k.float(), rotary[:, None], half=True).to(hidden.dtype)
        outputs = [self.attention(a[None], b[None], c[None])[0]
                   for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))]
        return self.proj(torch.cat(outputs).reshape(len(hidden), -1))


class VisionBlock(nn.Module):
    def __init__(self, config, ocr):
        super().__init__()
        self.norm1 = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.attn = VisionAttention(config, ocr)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size if ocr else config.out_hidden_size,
                            config.attention_bias if ocr else False)

    def forward(self, hidden, rotary, lengths):
        hidden = hidden + self.attn(self.norm1(hidden), rotary, lengths)
        return hidden + self.mlp(self.norm2(hidden))


class Merger(GatedMLP):
    def __init__(self, config, ocr):
        width = config.out_hidden_size
        super().__init__(width, width * config.in_channels if ocr else config.intermediate_size)
        self.proj = Linear(width, width, bias=False)
        self.post_projection_norm = LayerNorm(width, promote_fp32=False)
        self.act1 = GELU()

    def forward(self, hidden):
        return super().forward(self.act1(self.post_projection_norm(self.proj(hidden))))


class Vision(nn.Module):
    def __init__(self, config, ocr=False):
        super().__init__()
        self.config, self.ocr = config, ocr
        patch = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.patch_embed = Conv3d(config.in_channels, config.hidden_size, patch, bias=True)
        if not ocr:
            self.position_embedding = nn.Parameter(torch.empty((config.image_size // config.patch_size) ** 2, config.hidden_size))
            self.interpolate = Interpolate()
            self.post_conv_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.blocks = nn.ModuleList(VisionBlock(config, ocr) for _ in range(config.depth))
        self.post_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.downsample = Conv2d(config.hidden_size, config.out_hidden_size, config.spatial_merge_size,
                                 stride=config.spatial_merge_size, bias=True)
        self.merger = Merger(config, ocr)
        self.rotary = None  # Attach FP32 position cache after the requested parameter cast.

    def forward(self, pixels, grid):
        config, merge = self.config, self.config.spatial_merge_size
        grid = grid.tolist()
        hidden = self.patch_embed(pixels.reshape(-1, config.in_channels, config.temporal_patch_size,
                                                  config.patch_size, config.patch_size)).flatten(1)
        if not self.ocr:
            hidden = self.post_conv_layernorm(hidden)
            side = config.image_size // config.patch_size
            weight = self.position_embedding.float().reshape(side, side, -1).permute(2, 0, 1)[None]
            positions = []
            for t, h, w in grid:
                resized = self.interpolate(weight, size=(h, w), mode='bicubic', align_corners=False)
                resized = resized[0].permute(1, 2, 0).reshape(h // merge, merge, w // merge, merge, -1)
                positions.append(resized.permute(0, 2, 1, 3, 4).reshape(h * w, -1).repeat(t, 1))
            hidden = hidden + torch.cat(positions).to(hidden.dtype)
        cos, sin = self.rotary(grid, merge, torch.float32, hidden.device)
        rotary = torch.cat((sin, sin, cos, cos), dim=-1)
        lengths = [h * w for t, h, w in grid for _ in range(t)]
        for block in self.blocks:
            hidden = block(hidden, rotary, lengths)
        hidden = self.post_layernorm(hidden).reshape(-1, merge, merge, config.hidden_size).permute(0, 3, 1, 2)
        hidden = self.downsample(hidden).flatten(1)
        return self.merger(hidden)


class MultimodalRotary(nn.Module):
    """Position preparation plus the unchanged DINOv3 rotation helper."""
    def __init__(self, config, interleaved=True):
        super().__init__()
        rope = config.rope_parameters
        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.dim = int(head_dim * rope.get('partial_rotary_factor', 1.0))
        self.sections, self.interleaved = rope['mrope_section'], interleaved
        self.register_buffer('inv_freq', 1.0 / (rope['rope_theta'] **
                             (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)), persistent=False)

    def forward(self, positions, query, key):
        angles = positions[:, :, None].float() * self.inv_freq[None, None]
        angles = torch.cat([chunk[i % 3] for i, chunk in enumerate(angles.split(self.sections, dim=-1))], dim=-1)
        if self.interleaved:
            cos, sin = angles.cos().to(query.dtype).repeat_interleave(2, -1), angles.sin().to(query.dtype).repeat_interleave(2, -1)
        else:
            cos, sin = angles.cos().to(query.dtype).repeat(1, 2), angles.sin().to(query.dtype).repeat(1, 2)
        emb = torch.cat((sin, cos), dim=-1)[:, None]
        outputs = []
        for value in (query, key):
            shape = value.shape
            value = value.reshape(value.shape[0], -1, self.head_dim)
            rotated = apply_rot_embed_cat(value[..., :self.dim], emb, half=not self.interleaved)
            outputs.append(torch.cat((rotated, value[..., self.dim:]), dim=-1).reshape(shape))
        return tuple(outputs)


class TextBackbone(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.embed_tokens, self.layers, self.norm = native.embed_tokens, native.layers, native.norm
        self.rotary_emb = native.rotary_emb

    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden, _ = layer(positions, hidden)
        return self.norm(hidden)


class Backbone(nn.Module):
    def __init__(self, text, config, vision=None):
        super().__init__()
        self.text, self.config = text, config
        self.vision = Vision(config.vision_config, config.model_type == 'glm_ocr') if vision is None else vision
        self.inputs = self.rope_delta = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        if not get_context().is_prefill:
            return self.text(input_ids, positions[None].expand(3, -1) + self.rope_delta)
        embeddings = self.text.embed_tokens(input_ids)
        features = self.vision(self.inputs['pixel_values'], self.inputs['image_grid_thw'])
        embeddings[input_ids == self.config.image_token_id] = features
        positions, delta = qwen2_vl.multimodal_positions(input_ids,
            self.inputs['mm_token_type_ids'][0, :input_ids.numel()], self.inputs, self.config)
        self.rope_delta = delta.reshape(1, 1)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class Model(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = Backbone(language.model, config)


def build_from_config(config, device, dtype):
    text = config.text_config
    if text.hidden_act != 'silu' or text.rope_parameters['rope_type'] != 'default':
        raise ValueError('Selected GLM multimodal decoder uses SiLU and default multidimensional RoPE')
    language = make_text(text, dtype, native_attention=config.model_type != 'glm_ocr')
    model = Model(language, config).to(device=device, dtype=dtype).eval()
    attach_positions(model, config, device)
    return model


def make_text(text, dtype, native_attention=False):
    native = decoder_config(text, dtype)
    language = LlamaForCausalLM(native)
    language.model.layers = nn.ModuleList(glm4.Glm4Layer(layer, text) for layer in language.model.layers)
    for layer in language.model.layers:
        for name in ('input_layernorm', 'post_attention_layernorm', 'post_self_attn_layernorm', 'post_mlp_layernorm'):
            setattr(layer, name, RMSNormNative(text.hidden_size, text.rms_norm_eps))
        if native_attention:
            attention = layer.self_attn
            sizes = [native.num_attention_heads * native.head_dim,
                     native.num_key_value_heads * native.head_dim,
                     native.num_key_value_heads * native.head_dim]
            attention.qkv_proj = SeparateQKV(attention.qkv_proj, sizes)
            # Common native Q/K/V still diverged in the original paged kernel;
            # unchanged cache-storage plus existing grouped SDPA matched exactly.
            attention.attn = DenseCachedAttention(native.num_attention_heads,
                                                  native.num_key_value_heads, native.head_dim)
    language.model.norm = RMSNormNative(text.hidden_size, text.rms_norm_eps)
    language.model = TextBackbone(language.model)
    return language


def attach_positions(model, config, device, interleaved=True):
    text = config.text_config
    rotary = MultimodalRotary(text, interleaved).to(device=device)
    model.model.text.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
    vision = config.vision_config
    model.model.vision.rotary = VisionRotaryEmbedding(vision.hidden_size // vision.num_heads // 2).to(device=device)


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    load_text(model, remaining, config)
    load_vision(model.model.vision, remaining)
    if remaining:
        raise KeyError(f'Unmapped GLM multimodal weights: {sorted(remaining)}')


def load_text(model, remaining, config):
    text = {name.replace('model.language_model.', 'model.'): remaining.pop(name)
            for name in list(remaining) if name.startswith('model.language_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    carrier = SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config)
    for i, layer in enumerate(model.model.layers):
        for name in ('post_self_attn_layernorm', 'post_mlp_layernorm'):
            target = getattr(layer, name).weight
            source = text.pop(f'model.layers.{i}.{name}.weight')
            if target.shape != source.shape:
                raise ValueError(f'GLM text normalization shape mismatch: {i}.{name}')
            target.copy_(source)
        packed = text.pop(f'model.layers.{i}.mlp.gate_up_proj.weight')
        for part, value in zip(('gate', 'up'), packed.chunk(2, dim=0)):
            text[f'model.layers.{i}.mlp.{part}_proj.weight'] = value
    loader = qwen2.load_state_dict_into if config.text_config.attention_bias else llama.load_state_dict_into
    loader(carrier, text, model.config)


def load_vision(vision, remaining):
    mapped = {}
    for name, value in vision.state_dict().items():
        source = name.replace('patch_embed.conv.', 'patch_embed.proj.').replace('downsample.conv.', 'downsample.')
        if source == 'position_embedding':
            source = 'embeddings.position_embedding.weight'
        source = 'model.visual.' + source
        supplied = remaining.pop(source)
        if value.shape != supplied.shape:
            raise ValueError(f'GLM vision shape mismatch: {source}')
        mapped[name] = supplied
    vision.load_state_dict(mapped, strict=True)


make_workloads = qwen2_vl.make_workloads
