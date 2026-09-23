"""Parakeet CTC with its full convolutional subsampling and relative attention."""

import math
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.silu import SiLU
from ..patches.audio_query_bias import BiasedQueryAttention
from ..patches.query_bias_bmm import BiasedQueryBMM
from .wav2vec2_conformer import GLU, make_workloads


class Subsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.count = int(math.log2(config.subsampling_factor))
        width, kernel, stride = (config.subsampling_conv_channels, config.subsampling_conv_kernel_size,
                                 config.subsampling_conv_stride)
        self.layers = nn.ModuleList([Conv2d(1, width, kernel, stride=stride, padding=(kernel - 1) // 2), ReLU()])
        for _ in range(self.count - 1):
            self.layers.extend([Conv2d(width, width, kernel, stride=stride, padding=(kernel - 1) // 2,
                                       groups=width), Conv2d(width, width, 1), ReLU()])
        self.linear = Linear(width * (config.num_mel_bins // stride**self.count), config.hidden_size)

    def forward(self, features, attention_mask):
        hidden = features[:, None]
        lengths = attention_mask.sum(-1) if attention_mask is not None else None
        for layer in self.layers:
            hidden = layer(hidden)
            if isinstance(layer, Conv2d) and lengths is not None:
                if layer.stride[0] != 1:
                    lengths = (lengths + sum(layer.padding) - layer.weight.shape[-1]) // layer.stride[0] + 1
                mask = torch.arange(hidden.shape[2], device=hidden.device)[None] < lengths[:, None]
                hidden = hidden.masked_fill(~mask[:, None, :, None], 0)
        hidden = hidden.transpose(1, 2).flatten(2)
        mask = (torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
                if lengths is not None else None)
        return self.linear(hidden), mask


class RelativePositions(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        rate = 1 / (10000.0 ** (torch.arange(0, config.hidden_size, 2, dtype=torch.float32,
                                             device=device) / config.hidden_size))
        self.register_buffer("inv_freq", rate, persistent=False)

    def forward(self, hidden):
        length = hidden.shape[1]
        positions = torch.arange(length - 1, -length, -1, device=hidden.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq[None]
        table = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return table[None].expand(hidden.shape[0], -1, -1).to(hidden.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias))
        self.relative_k_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.bias_u = nn.Parameter(torch.empty(self.heads, self.width))
        self.bias_v = nn.Parameter(torch.empty(self.heads, self.width))
        self.relative_score = BiasedQueryBMM()
        # Optional kernel imports disable global cuDNN dispatch; select the
        # existing backend explicitly to retain native HF attention rounding.
        self.attention = BiasedQueryAttention(backend="cudnn")

    def forward(self, hidden, positions, mask):
        batch, length, _ = hidden.shape
        q, k, v = (getattr(self, name)(hidden).reshape(batch, length, self.heads, self.width)
                   for name in ("q_proj", "k_proj", "v_proj"))
        relative = self.relative_k_proj(positions).reshape(batch, -1, self.heads, self.width)
        relative = relative.transpose(1, 2).reshape(batch * self.heads, -1, self.width)
        query = q.transpose(1, 2).reshape(-1, length, self.width)
        bias = self.bias_v[None].expand(batch, -1, -1).reshape(-1, 1, self.width)
        scores = self.relative_score(query, relative.transpose(1, 2), bias)
        scores = torch.cat((torch.zeros_like(scores[..., :1]), scores), dim=-1)
        scores = scores.reshape(-1, 2 * length, length)[:, 1:]
        scores = scores.reshape(batch, self.heads, length, 2 * length - 1)[..., :length] * self.width**-0.5
        if mask is not None:
            allowed = mask[:, None, :, None] & mask[:, None, None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
        output = self.attention(q, k, v, self.bias_u[None, None], attn_mask=scores,
                                softmax_scale=self.width**-0.5)
        return self.o_proj(output.reshape(batch, length, -1))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.activation = SiLU()
        self.linear2 = Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden):
        return self.linear2(self.activation(self.linear1(hidden)))


class Convolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.hidden_size, config.conv_kernel_size
        self.pointwise_conv1 = Conv1dNative(width, 2 * width, 1, bias=config.convolution_bias)
        self.glu = GLU()
        self.depthwise_conv = Conv1dNative(width, width, kernel, padding=(kernel - 1) // 2,
                                         groups=width, bias=config.convolution_bias)
        self.norm = BatchNorm2d(width)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1dNative(width, width, 1, bias=config.convolution_bias)

    def forward(self, hidden, mask):
        hidden = self.glu(self.pointwise_conv1(hidden.transpose(1, 2)))
        if mask is not None:
            hidden = hidden.masked_fill(~mask[:, None, :], 0)
        hidden = self.activation(self.norm(self.depthwise_conv(hidden)))
        return self.pointwise_conv2(hidden).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward1, self.feed_forward2 = FeedForward(config), FeedForward(config)
        self.self_attn, self.conv = Attention(config), Convolution(config)
        for name in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            setattr(self, name, LayerNorm(config.hidden_size, promote_fp32=False))

    def forward(self, hidden, positions, mask):
        hidden = hidden + 0.5 * self.feed_forward1(self.norm_feed_forward1(hidden))
        hidden = hidden + self.self_attn(self.norm_self_att(hidden), positions, mask)
        hidden = hidden + self.conv(self.norm_conv(hidden), mask)
        hidden = hidden + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden))
        return self.norm_out(hidden)


class Parakeet(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder = config.encoder_config
        self.input_scale = math.sqrt(encoder.hidden_size) if encoder.scale_input else 1.0
        self.encoder = nn.Module()
        self.encoder.subsampling = Subsampling(encoder)
        self.encoder.layers = nn.ModuleList(Block(encoder) for _ in range(encoder.num_hidden_layers))
        self.ctc_head = Conv1dNative(encoder.hidden_size, config.vocab_size, 1)

    def forward(self, input_features, attention_mask=None):
        hidden, mask = self.encoder.subsampling(input_features, attention_mask)
        hidden = hidden * self.input_scale
        positions = self.encoder.encode_positions(hidden)
        for layer in self.encoder.layers:
            hidden = layer(hidden, positions, mask)
        return {"logits": self.ctc_head(hidden.transpose(1, 2)).transpose(1, 2)}


def build_from_config(config, device, dtype):
    encoder = config.encoder_config
    if encoder.hidden_act != "silu" or encoder.num_key_value_heads != encoder.num_attention_heads:
        raise ValueError("Selected Parakeet checkpoint uses SiLU and ordinary multi-head attention")
    model = Parakeet(config).to(device=device, dtype=dtype)
    # HF retains FP32 inverse frequencies independently of its parameter dtype.
    model.encoder.encode_positions = RelativePositions(encoder, device)
    return model.eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)
