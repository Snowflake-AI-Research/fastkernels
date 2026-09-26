"""Emu3 image-chat generation with its spatial/temporal VQ image encoder."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.conv3d import Conv3d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from ..runner import Workload, config_values
from .dac import VectorQuantizer
from .smolvlm import TextDecoder


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid, self.product = Sigmoid(), ProductGate()

    def forward(self, hidden):
        return self.product(torch.cat((hidden, self.sigmoid(hidden)), dim=-1))


class CausalConv3d(nn.Module):
    def __init__(self, source, target, kernel, stride):
        super().__init__()
        sizes = [k - s for k, s in zip(kernel[1:], stride[1:])]
        self.padding = tuple(v for size in sizes[::-1] for v in (size // 2 + size % 2, size // 2)) + (2, 0)
        self.conv = Conv3d(source, target, kernel, stride=stride, bias=True)

    def forward(self, hidden):
        return self.conv(F.pad(hidden, self.padding))


class Residual(nn.Module):
    def __init__(self, source, target, temporal=False):
        super().__init__()
        self.norm1 = BatchNorm2d(source) if temporal else GroupNorm(32, source, eps=1e-6)
        self.norm2 = BatchNorm2d(target) if temporal else GroupNorm(32, target, eps=1e-6)
        self.conv1 = CausalConv3d(source, target, (3, 3, 3), (1, 1, 1)) if temporal else Conv2d(source, target, 3, padding=1)
        self.conv2 = CausalConv3d(target, target, (3, 3, 3), (1, 1, 1)) if temporal else Conv2d(target, target, 3, padding=1)
        if source != target:
            self.nin_shortcut = Conv3d(source, target, (1, 1, 1), stride=(1, 1, 1), bias=True) if temporal else Conv2d(source, target, 1)
        self.activation = GatedActivation()

    def forward(self, hidden):
        residual = self.nin_shortcut(hidden) if hasattr(self, 'nin_shortcut') else hidden
        hidden = self.conv1(self.activation(self.norm1(hidden)))
        return residual + self.conv2(self.activation(self.norm2(hidden)))


class ImageAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self, name, Linear(self.width, self.width))
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden):
        shape = (*hidden.shape[:2], self.heads, self.width // self.heads)
        q, k, v = (getattr(self, name)(hidden).reshape(shape) for name in ('q_proj', 'k_proj', 'v_proj'))
        return self.out_proj(self.attention(q, k, v).reshape_as(hidden))


def spatial_attention(hidden, normalization, attention):
    residual = hidden
    hidden = normalization(hidden)
    batch, channels, height, width = hidden.shape
    hidden = attention(hidden.view(batch, channels, height * width).transpose(1, 2))
    return residual + hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class Downsample(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = Conv2d(width, width, 3, stride=2)

    def forward(self, hidden):
        return self.conv(F.pad(hidden, (0, 1, 0, 1)))


class DownBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down = nn.ModuleList()
        source = config.base_channels
        for index, multiplier in enumerate(config.channel_multiplier):
            target = config.base_channels * multiplier
            level = nn.Module()
            level.block, level.attn, level.attn_norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            for _ in range(config.num_res_blocks):
                level.block.append(Residual(source, target))
                source = target
                if index in config.attn_resolutions:
                    level.attn.append(ImageAttention(config))
                    level.attn_norms.append(GroupNorm(32, target, eps=1e-6))
            if index + 1 < len(config.channel_multiplier):
                level.downsample = Downsample(target)
            self.down.append(level)

    def forward(self, hidden):
        for level in self.down:
            for index, block in enumerate(level.block):
                hidden = block(hidden)
                if len(level.attn):
                    hidden = spatial_attention(hidden, level.attn_norms[index], level.attn[index])
            if hasattr(level, 'downsample'):
                hidden = level.downsample(hidden)
        return hidden


class MiddleBlock(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.block_1, self.block_2 = Residual(width, width), Residual(width, width)
        self.attn_1, self.attn_norm = ImageAttention(config), GroupNorm(32, width, eps=1e-6)

    def forward(self, hidden):
        return self.block_2(spatial_attention(self.block_1(hidden), self.attn_norm, self.attn_1))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.base_channels * config.channel_multiplier[-1]
        latent = config.latent_channels * (2 if config.double_latent else 1)
        self.conv_in = Conv2d(config.in_channels, config.base_channels, 3, padding=1)
        self.down_block, self.middle_block = DownBlock(config), MiddleBlock(config, width)
        self.norm_out, self.conv_out = GroupNorm(32, width, eps=1e-6), Conv2d(width, latent, 3, padding=1)
        self.time_conv = nn.ModuleList()
        for _ in range(int(math.log2(config.temporal_downsample_factor))):
            layer = nn.Module()
            layer.conv = CausalConv3d(latent, latent, (4, 3, 3), (2, 1, 1))
            self.time_conv.append(layer)
        self.time_res_stack = nn.ModuleList(Residual(latent, latent, temporal=True) for _ in range(config.num_res_blocks))
        self.activation = GatedActivation()

    def forward(self, pixels):
        batch, temporal = pixels.shape[:2]
        hidden = self.middle_block(self.down_block(self.conv_in(pixels.flatten(0, 1))))
        hidden = self.conv_out(self.activation(self.norm_out(hidden)))
        hidden = hidden.reshape(batch, temporal, *hidden.shape[1:]).permute(0, 2, 1, 3, 4)
        for layer in self.time_conv:
            hidden = self.activation(layer.conv(hidden))
        for layer in self.time_res_stack:
            hidden = layer(hidden)
        return hidden.permute(0, 2, 1, 3, 4)


class Quantizer(nn.Module):
    squared_row_norm = VectorQuantizer.squared_row_norm

    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.codebook_size, config.embed_dim)
        self.product, self.reduce = ProductGate(), SegmentCSR()
        self.matmul, self.select = BMM(), CodecTop1()

    def forward(self, hidden):
        batch, temporal, channels, height, width = hidden.shape
        rows = hidden.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, channels)
        codes = self.embedding.emb.weight
        dot = 2 * self.matmul(rows, codes.T)
        distances = self.squared_row_norm(rows) + self.squared_row_norm(codes).T - dot
        return self.select(-distances).reshape(batch, temporal, height, width)


class VQEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.quant_conv = CausalConv3d(config.latent_channels, config.embed_dim, (3, 1, 1), (1, 1, 1))
        self.quantize = Quantizer(config)

    def forward(self, pixels, image_sizes):
        if pixels.ndim != 4:
            raise ValueError('Selected Emu3 public chat task supplies images')
        pixels = pixels[:, None].repeat(1, self.config.temporal_downsample_factor, 1, 1, 1)
        hidden = self.encoder(pixels)
        quantized = self.quant_conv(hidden.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        codes = self.quantize(quantized).squeeze(1)
        factor = 2 ** (len(self.config.channel_multiplier) - 1)
        return [code[:int(size[0]) // factor, :int(size[1]) // factor] for code, size in zip(codes, image_sizes)]


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        text = config_values(dict(config.text_config))
        text.head_dim = text.get('head_dim', text.hidden_size // text.num_attention_heads)
        text.attention_bias, text.mlp_bias = text.get('attention_bias', False), text.get('mlp_bias', False)
        self.text_model, self.vqmodel = TextDecoder(text), VQEncoder(config.vq_config)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        self.select = CodecTop1()
        mapping = {int(name[-8:-2]): value for name, value in config.vocabulary_map.items() if name.startswith('<|visual token')}
        table = torch.zeros(max(mapping) + 1, dtype=torch.long)
        for key, value in mapping.items():
            table[key] = value
        self.register_buffer('image_to_bpe', table, persistent=False)
        self.inactive_state = nn.Module()

    def image_tokens(self, pixels, image_sizes):
        codes = self.vqmodel(pixels, image_sizes)
        end = self.config.vocabulary_map['<|extra_200|>']
        return torch.cat([torch.cat((self.image_to_bpe[code], code.new_full((code.shape[0], 1), end)), dim=-1).flatten() for code in codes]).to(torch.int32)

    def generate(self, input_ids, pixel_values, image_sizes, max_new_tokens=4, *, return_logits=False):
        self.text_model.reset()
        ids = input_ids.clone()
        scores = []
        for step in range(max_new_tokens):
            current = ids if step == 0 else ids[:, -1:]
            hidden = self.text_model.embed_tokens(current)
            if step == 0:
                image = self.text_model.embed_tokens(self.image_tokens(pixel_values, image_sizes))
                hidden = hidden.masked_scatter((current == self.config.image_token_id)[..., None].expand_as(hidden), image)
            start = 0 if step == 0 else ids.shape[1] - 1
            hidden = self.text_model(hidden, torch.arange(start, ids.shape[1], device=ids.device))
            logits = self.lm_head(hidden[:, -1:]).float()
            if return_logits:
                scores.append(logits[:, -1].clone())
            token = self.select(logits).reshape(1, 1)
            ids = torch.cat((ids, token), dim=1)
            if int(token[0, 0]) == self.config.text_config.eos_token_id:
                break
        return (ids, scores) if return_logits else ids


def build_from_config(config, device, dtype):
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for module, prefix in ((model.text_model, 'model.text_model.'), (model.vqmodel, 'model.vqmodel.')):
        values = {}
        for name in module.state_dict():
            source = name.replace('.emb.weight', '.weight')
            if module is model.vqmodel:
                source = source.replace('.conv.conv.', '.conv.')
            values[name] = remaining.pop(prefix + source)
        module.load_state_dict(values, strict=True)
    model.lm_head.load_state_dict({'weight': remaining.pop('lm_head.weight')}, strict=True)
    unknown = [name for name in remaining if not name.startswith(('model.vqmodel.decoder.', 'model.vqmodel.post_quant_conv.'))]
    if unknown:
        raise KeyError(f'Unmapped Emu3 state: {unknown}')
    model.inactive_state = nn.Module()
    for name, value in remaining.items():
        model.inactive_state.register_buffer(name.replace('.', '__'), value.clone())


def make_workloads(model, inputs, config, case=None):
    options = {} if case is None else dict(case['generation_kwargs'])
    if options.pop('do_sample', False):
        raise ValueError('Selected public workload uses greedy image-conditioned text generation')
    def run():
        sequences, scores = model.generate(**inputs, **options, return_logits=True)
        return {'sequences': sequences, **{f'logits.{i}': value for i, value in enumerate(scores)}}

    return {'generate': Workload(run=run)}
