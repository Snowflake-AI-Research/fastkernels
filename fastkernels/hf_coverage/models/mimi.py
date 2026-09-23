"""Mimi's default non-streaming codec, retaining both residual quantizer groups."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.models.encodec import Codebook, elu
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp


class CausalConv(nn.Module):
    def __init__(self, source, target, kernel, stride=1, dilation=1, groups=1,
                 bias=True, transpose=False, replicate=False):
        super().__init__()
        op = ConvTranspose1d if transpose else Conv1dNative
        self.conv = op(source, target, kernel, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.total = (kernel - 1) * dilation + 1 - stride
        self.stride, self.transpose, self.replicate = stride, transpose, replicate
        self.pad = Pad()

    def forward(self, hidden):
        if self.transpose:
            output = self.conv(hidden)
            return output[..., :-self.total] if self.total else output
        extra = (-hidden.shape[-1]) % self.stride
        if self.replicate:
            hidden = torch.cat((hidden[..., :1].expand(*hidden.shape[:-1], self.total), hidden,
                                hidden[..., -1:].expand(*hidden.shape[:-1], extra)), dim=-1)
        else:
            hidden = self.pad(hidden, (self.total, extra))
        return self.conv(hidden)


class Residual(nn.Module):
    def __init__(self, config, width, dilation):
        super().__init__()
        small = width // config.compress
        self.block = nn.ModuleList([elu(), CausalConv(width, small, config.residual_kernel_size, dilation=dilation),
                                    elu(), CausalConv(small, width, 1)])
        self.shortcut = CausalConv(width, width, 1) if config.use_conv_shortcut else nn.Identity()

    def forward(self, hidden):
        output = hidden
        for layer in self.block:
            output = layer(output)
        return self.shortcut(hidden) + output


class Stack(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        width = config.num_filters * (2 ** len(config.upsampling_ratios) if decoder else 1)
        layers = [CausalConv(config.hidden_size if decoder else config.audio_channels, width, config.kernel_size)]
        for ratio in config.upsampling_ratios if decoder else reversed(config.upsampling_ratios):
            if decoder:
                layers += [elu(), CausalConv(width, width // 2, 2 * ratio, stride=ratio, transpose=True)]
                width //= 2
            for index in range(config.num_residual_layers):
                layers.append(Residual(config, width, config.dilation_growth_rate ** index))
            if not decoder:
                layers += [elu(), CausalConv(width, width * 2, 2 * ratio, stride=ratio)]
                width *= 2
        layers += [elu(), CausalConv(width, config.audio_channels if decoder else config.hidden_size, config.last_kernel_size)]
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class LayerScale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(width))
        self.product = ProductGate()

    def forward(self, hidden):
        return self.product(torch.cat((hidden, self.scale.expand_as(hidden)), dim=-1))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = Linear(config.hidden_size, self.heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = Linear(self.heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_parameters['rope_theta'])
        self.matmul, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, mask):
        batch, length, _ = hidden.shape
        query, key = self.q_proj(hidden), self.k_proj(hidden)
        positions = torch.arange(length, device=hidden.device).expand(batch, -1).reshape(-1)
        query, key = query.reshape(batch * length, -1), key.reshape(batch * length, -1)
        # The parent's native callable preserves HF's separate BF16 product
        # rounding. Its CUDA callable instead promotes the paired products.
        query, key = self.rotary_emb.forward_native(
            positions, query, key, self.head_dim, self.rotary_emb.cos_sin_cache.to(hidden.dtype))
        query = query.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.heads != self.kv_heads:
            key = key.repeat_interleave(self.heads // self.kv_heads, dim=1)
            value = value.repeat_interleave(self.heads // self.kv_heads, dim=1)
        query, key, value = [t.reshape(batch * self.heads, length, self.head_dim) for t in (query,key,value)]
        scores = self.matmul(query, key.transpose(1, 2)) * (self.head_dim ** -0.5)
        weights = self.softmax((scores + mask).float()).to(hidden.dtype)
        output = self.matmul(weights, value).reshape(batch, self.heads, length, self.head_dim)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.norm_eps, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, bias=False)
        self.self_attn_layer_scale, self.mlp_layer_scale = LayerScale(config.hidden_size), LayerScale(config.hidden_size)

    def forward(self, hidden, mask):
        hidden = hidden + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(hidden), mask))
        return hidden + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(hidden)))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(TransformerLayer(config) for _ in range(config.num_hidden_layers))
        self.window = config.sliding_window

    def forward(self, hidden):
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - self.window)
        mask = torch.zeros(allowed.shape, dtype=hidden.dtype, device=hidden.device).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden


class FloatDistanceCodebook(Codebook):
    def encode(self, hidden):
        batch, width, length = hidden.shape
        rows = hidden.transpose(1, 2).reshape(-1, width).float()
        codes = self.embedding.emb.weight.float()
        dot = self.matmul((2 * rows).unsqueeze(0), codes.T.unsqueeze(0))[0]
        squared_distance = self.squared_row_norm(rows) - dot + self.squared_row_norm(codes).T
        # Squared Euclidean distance has the same nearest centroid as distance.
        # Exact integer comparison against HF remains mandatory for tested inputs.
        return self.select(-squared_distance).reshape(batch, length)


class QuantizerGroup(nn.Module):
    def __init__(self, config, count):
        super().__init__()
        self.layers = nn.ModuleList(FloatDistanceCodebook(config) for _ in range(count))
        self.input_proj = Conv1dNative(config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False)
        self.output_proj = Conv1dNative(config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False)

    def encode(self, hidden):
        residual = self.input_proj(hidden)
        codes = []
        for layer in self.layers:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            codes.append(indices)
        return torch.stack(codes, dim=1)

    def decode(self, codes):
        quantized = torch.tensor(0.0, device=codes.device)
        for index, layer in enumerate(self.layers):
            quantized = quantized + layer.decode(codes[:, index])
        return self.output_proj(quantized)


class Mimi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder, self.decoder = Stack(config), Stack(config, decoder=True)
        self.encoder_transformer, self.decoder_transformer = Transformer(config), Transformer(config)
        frame_rate = getattr(config, '_frame_rate', None) or (config.sampling_rate / (2 * math.prod(config.upsampling_ratios)))
        kernel = 2 * int(math.ceil(config.sampling_rate / math.prod(config.upsampling_ratios)) / frame_rate)
        self.downsample = CausalConv(config.hidden_size, config.hidden_size, kernel, stride=2, bias=False, replicate=True)
        self.upsample = CausalConv(config.hidden_size, config.hidden_size, kernel, stride=2, bias=False,
                                   groups=config.upsample_groups, transpose=True)
        self.semantic = QuantizerGroup(config, config.num_semantic_quantizers)
        self.acoustic = QuantizerGroup(config, config.num_quantizers - config.num_semantic_quantizers)
        self.semantic_count = config.num_semantic_quantizers

    def forward(self, input_values):
        encoded = self.downsample(self.encoder_transformer(self.encoder(input_values).transpose(1, 2)).transpose(1, 2))
        codes = torch.cat((self.semantic.encode(encoded), self.acoustic.encode(encoded)), dim=1)
        quantized = self.semantic.decode(codes[:, :self.semantic_count]) + self.acoustic.decode(codes[:, self.semantic_count:])
        decoded = self.decoder_transformer(self.upsample(quantized).transpose(1, 2)).transpose(1, 2)
        return {'audio_codes': codes, 'audio_values': self.decoder(decoded)[..., :input_values.shape[-1]]}


def build_from_config(config, device, dtype):
    if (config.use_streaming or config.use_cache or config.pad_mode != 'constant' or not config.use_causal_conv
            or config.hidden_act != 'gelu' or config.vector_quantization_hidden_dimension == config.hidden_size
            or config.rope_parameters['rope_type'] != 'default'):
        raise ValueError('Mimi case preserves the published non-streaming architecture and quantizer projections')
    return Mimi(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for name, target in model.state_dict().items():
        source = name.replace('semantic.', 'quantizer.semantic_residual_vector_quantizer.', 1) if name.startswith('semantic.') else name
        source = source.replace('acoustic.', 'quantizer.acoustic_residual_vector_quantizer.', 1) if name.startswith('acoustic.') else source
        if '.embedding.emb.weight' in source:
            prefix = source.replace('.embedding.emb.weight', '.codebook.')
            embed, usage = prefix + 'embed_sum', prefix + 'cluster_usage'
            value = state[embed].to(device=target.device, dtype=target.dtype) / state[usage].to(device=target.device, dtype=target.dtype).clamp(min=1e-5)[:, None]
            consumed.update((embed, usage, prefix + 'initialized'))
        else:
            value = state[source]
            consumed.add(source)
        if value.shape != target.shape:
            raise ValueError(f'Mimi weight mismatch: {source}')
        mapped[name] = value
    if consumed != set(state):
        raise ValueError(f'Mimi unmapped state: {set(state) - consumed}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
