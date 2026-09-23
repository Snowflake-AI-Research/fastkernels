"""LASR CTC constructor configuration, composed from existing operations."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU

from .parakeet import FeedForward
from .wav2vec2_conformer import GLU, make_workloads


def normalization(config):
    return LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                     create_offset=False, promote_fp32=False)


class Subsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, channels = config.hidden_size, config.subsampling_conv_channels
        kernel, stride = config.subsampling_conv_kernel_size, config.subsampling_conv_stride
        self.dense_0 = Linear(config.num_mel_bins, width)
        self.conv_0 = Conv1dNative(width, width, kernel, stride=stride)
        self.conv_1 = Conv1dNative(width, channels, kernel, stride=stride)
        self.dense_1 = Linear(channels, width)
        self.activation = ReLU()

    def forward(self, features):
        hidden = self.activation(self.dense_0(features)).transpose(1, 2)
        hidden = self.activation(self.conv_0(hidden))
        hidden = self.activation(self.conv_1(hidden))
        return self.dense_1(hidden.transpose(1, 2))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size,
                                       bias=config.attention_bias))
        # The GLU dependency imports vLLM, which disables global cuDNN SDPA.
        # Select the existing backend explicitly to retain HF's native rounding.
        self.attention = DenseAttention(backend="cudnn")

    def forward(self, hidden, positions, rotary_table):
        batch, length, width = hidden.shape
        query, key, value = (getattr(self, name)(hidden).reshape(-1, width)
                             for name in ("q_proj", "k_proj", "v_proj"))
        query, key = RotaryEmbedding.forward_native(
            positions, query, key, self.head_dim, rotary_table,
        )
        shape = (batch, length, self.heads, self.head_dim)
        output = self.attention(query.reshape(shape), key.reshape(shape), value.reshape(shape))
        return self.o_proj(output.reshape(batch, length, width))


class Convolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, bias = config.hidden_size, config.convolution_bias
        self.pointwise_conv1 = Conv1dNative(width, 2 * width, 1, bias=bias)
        self.glu = GLU()
        # Native convolution accepts "same", including the default even kernel.
        self.depthwise_conv = Conv1dNative(width, width, config.conv_kernel_size,
                                         padding="same", groups=width, bias=bias)
        # This operation's F.batch_norm also accepts [batch, channels, length].
        self.norm = BatchNorm2d(width, momentum=config.batch_norm_momentum)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1dNative(width, width, 1, bias=bias)

    def forward(self, hidden):
        hidden = self.glu(self.pointwise_conv1(hidden.transpose(1, 2)))
        hidden = self.activation(self.norm(self.depthwise_conv(hidden)))
        return self.pointwise_conv2(hidden).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward1, self.feed_forward2 = FeedForward(config), FeedForward(config)
        self.self_attn, self.conv = Attention(config), Convolution(config)
        for name in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            setattr(self, name, normalization(config))
        self.ff_weights = config.feed_forward_residual_weights
        self.conv_weights = config.conv_residual_weights

    def forward(self, hidden, positions, rotary_table):
        hidden = self.ff_weights[0] * hidden + self.ff_weights[1] * self.feed_forward1(
            self.norm_feed_forward1(hidden),
        )
        hidden = hidden + self.self_attn(self.norm_self_att(hidden), positions, rotary_table)
        hidden = self.conv_weights[0] * hidden + self.conv_weights[1] * self.conv(self.norm_conv(hidden))
        hidden = self.ff_weights[0] * hidden + self.ff_weights[1] * self.feed_forward2(
            self.norm_feed_forward2(hidden),
        )
        return self.norm_out(hidden)


class Lasr(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder = config.encoder_config
        self.encoder = nn.Module()
        self.encoder.subsampler = Subsampling(encoder)
        self.encoder.layers = nn.ModuleList(Block(encoder) for _ in range(encoder.num_hidden_layers))
        self.encoder.out_norm = normalization(encoder)
        self.ctc_head = Conv1dNative(encoder.hidden_size, config.vocab_size, 1)

    def forward(self, input_features):
        hidden = self.encoder.subsampler(input_features)
        batch, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device).repeat(batch)
        table = self.encoder.rotary_emb.cos_sin_cache.to(hidden.dtype)
        for layer in self.encoder.layers:
            hidden = layer(hidden, positions, table)
        hidden = self.encoder.out_norm(hidden)
        return {"logits": self.ctc_head(hidden.transpose(1, 2)).transpose(1, 2)}


def build_from_config(config, device, dtype):
    encoder = config.encoder_config
    if (encoder.hidden_act != "silu" or encoder.num_key_value_heads != encoder.num_attention_heads
            or encoder.rope_parameters["rope_type"] != "default"
            or config.output_attentions or config.output_hidden_states):
        raise ValueError("LASR coverage preserves the constructor's SiLU, ordinary RoPE attention and logits output")
    model = Lasr(config).to(device=device, dtype=dtype)
    # Keep the position table in FP32 until the forward's explicit dtype boundary.
    model.encoder.rotary_emb = RotaryEmbedding(
        encoder.hidden_size // encoder.num_attention_heads, encoder.max_position_embeddings,
        encoder.rope_parameters["rope_theta"],
    ).to(device=device)
    return model.eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)
