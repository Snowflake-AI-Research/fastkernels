"""PaddleOCR's patch-grid vision, packed projector and uncached text forward."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.dinov3_rope import apply_rot_embed_cat
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from ..runner import Workload
from . import glm4v, llama
from .olmo2 import decoder_config


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.attention = DenseAttention('sdpa')

    def forward(self, hidden, rotary, lengths):
        q, k, v = [getattr(self, name)(hidden).reshape(len(hidden), self.heads, -1)
                   for name in ('q_proj', 'k_proj', 'v_proj')]
        q = apply_rot_embed_cat(q.float(), rotary[:, None], half=True).to(hidden.dtype)
        k = apply_rot_embed_cat(k.float(), rotary[:, None], half=True).to(hidden.dtype)
        outputs = [self.attention(a[None], b[None], c[None])[0]
                   for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))]
        return self.out_proj(torch.cat(outputs).reshape(len(hidden), -1))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        self.layer_norm2 = LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        self.self_attn = Attention(config)
        self.mlp = nn.ModuleDict({'fc1': Linear(config.hidden_size, config.intermediate_size),
                                  'fc2': Linear(config.intermediate_size, config.hidden_size)})
        self.activation = GELU('tanh')

    def forward(self, hidden, rotary, lengths):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden), rotary, lengths)
        return hidden + self.mlp['fc2'](self.activation(self.mlp['fc1'](self.layer_norm2(hidden))))


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        self.position_embedding = nn.Parameter(torch.empty((config.image_size // config.patch_size) ** 2, config.hidden_size))
        self.interpolate = Interpolate()
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.post_layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        self.rotary = None

    def forward(self, pixels, grid):
        config, grids = self.config, grid.tolist()
        hidden = self.patch_embedding(pixels.reshape(-1, config.num_channels, config.patch_size, config.patch_size)).flatten(1)
        side = config.image_size // config.patch_size
        weight = self.position_embedding.reshape(side, side, -1).permute(2, 0, 1)[None]
        positions = [self.interpolate(weight, size=(h, w), mode='bilinear', align_corners=False)[0]
                     .permute(1, 2, 0).reshape(h * w, -1).repeat(t, 1) for t, h, w in grids]
        hidden = hidden + torch.cat(positions)
        cos, sin = self.rotary(grids, 1, torch.float32, hidden.device)
        rotary = torch.cat((sin, sin, cos, cos), dim=-1)
        lengths = [t * h * w for t, h, w in grids]
        for layer in self.layers:
            hidden = layer(hidden, rotary, lengths)
        return self.post_layernorm(hidden)


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.merge = config.vision_config.spatial_merge_size
        width = config.vision_config.hidden_size
        self.pre_norm = LayerNorm(width, 1e-5, promote_fp32=False)
        self.linear_1 = Linear(width * self.merge ** 2, width * self.merge ** 2)
        self.linear_2 = Linear(width * self.merge ** 2, config.text_config.hidden_size)
        self.act = GELU()

    def forward(self, features, grid):
        grids, merge = grid.tolist(), self.merge
        outputs = []
        for value, (t, h, w) in zip(features.split([t * h * w for t, h, w in grids]), grids):
            value = self.pre_norm(value).reshape(t, h // merge, merge, w // merge, merge, -1)
            value = value.transpose(2, 3).reshape(t * h * w // merge ** 2, -1)
            outputs.append(self.linear_2(self.act(self.linear_1(value))))
        return torch.cat(outputs)


class Frontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision, self.projector = Vision(config.vision_config), Projector(config)

    def forward(self, pixels, grid):
        return self.projector(self.vision(pixels, grid), grid)


def build_from_config(config, device, dtype):
    text = config.text_config
    if (text.use_cache or text.use_bias or text.hidden_act != 'silu' or getattr(text, 'sliding_window', None) is not None
            or text.rope_parameters['rope_type'] != 'default' or config.vision_config.hidden_act != 'gelu_pytorch_tanh'):
        raise ValueError('Selected PaddleOCR uses uncached bias-free full text attention and tanh-GELU vision')
    language = LlamaForCausalLM(decoder_config(text, dtype))
    # Construct the generic fusion backbone without instantiating a GLM vision tower.
    model = nn.Module()
    model.config, model.lm_head = language.config, language.lm_head
    model.model = glm4v.Backbone(language.model, config, Frontend(config))
    model.to(device=device, dtype=dtype).eval()
    rotary = glm4v.MultimodalRotary(text, interleaved=False).to(device=device)
    language.model.rotary_emb = rotary
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = rotary
    vision = config.vision_config
    model.model.vision.vision.rotary = VisionRotaryEmbedding(vision.hidden_size // vision.num_attention_heads // 2).to(device=device)
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace('model.language_model.', 'model.'): remaining.pop(name)
            for name in list(remaining) if name.startswith('model.language_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, model.config)
    frontend = model.model.vision
    mapped = {}
    for name, value in frontend.state_dict().items():
        if name.startswith('projector.'):
            source = 'model.' + name
        else:
            source = name.removeprefix('vision.')
            if source.startswith('layers.'):
                source = 'encoder.' + source
            elif source.startswith('patch_embedding.'):
                source = 'embeddings.' + source
            elif source == 'position_embedding':
                source = 'embeddings.position_embedding.weight'
            source = 'model.visual.vision_model.' + source
        supplied = remaining.pop(source)
        if supplied.shape != value.shape:
            raise ValueError(f'PaddleOCR vision weight mismatch: {source}')
        mapped[name] = supplied
    frontend.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f'Unmapped PaddleOCR weights: {sorted(remaining)}')


def make_workloads(model, inputs, config):
    model.model.inputs = inputs
    original = llama.make_workloads(model, inputs, model.config, cached_decode=False)['forward']

    def run():
        outputs = original.run()
        outputs['rope_deltas'] = model.model.rope_delta
        return outputs

    return {'forward': Workload(run=run, prepare=original.prepare)}
