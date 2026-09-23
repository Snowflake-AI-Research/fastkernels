"""Relative-position Conformer with the published layer-normalized waveform frontend."""

import math
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.qwen3_next_attention import _gate_mul_inplace
from ..patches.product_gate import ProductGate
from ..patches.query_bias_bmm import BiasedQueryBMM
from ..runner import Workload
from .wav2vec2 import FeatureConv, FeatureProjection


class GLU(nn.Module):
    def __init__(self, *, precise_sigmoid=False):
        super().__init__()
        self.precise_sigmoid = precise_sigmoid
        self.sigmoid, self.product = Sigmoid(), ProductGate()

    def forward(self, hidden):
        if hidden.is_cuda and not self.precise_sigmoid:
            # Unchanged internal GLU retains FP32 intermediates and one final
            # store. It is not separately exposed as a benchmark task.
            value, gate = hidden.chunk(2, dim=1)
            return _gate_mul_inplace(value.contiguous(), gate.contiguous())
        # ATen's sigmoid avoids the Triton exponential approximation. Keep the
        # product in FP32 and round once, as native GLU does for BF16 inputs.
        value, gate = hidden.float().transpose(1, 2).chunk(2, dim=-1)
        return self.product(torch.cat((value, self.sigmoid(gate)), dim=-1)).to(hidden.dtype).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = SiLU()
        self.output_dense = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden):
        return self.output_dense(self.intermediate_act_fn(self.intermediate_dense(hidden)))


class RelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.linear_pos = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.heads, self.width))
        self.pos_bias_v = nn.Parameter(torch.empty(self.heads, self.width))
        self.affine_bmm, self.bmm, self.softmax = BiasedQueryBMM(), BatchMatMul(), Softmax()

    def forward(self, hidden, positions, mask=None):
        batch, length, _ = hidden.shape
        q, k, v = (getattr(self, name)(hidden).reshape(batch, length, self.heads, self.width)
                   .transpose(1, 2).reshape(-1, length, self.width)
                   for name in ("linear_q", "linear_k", "linear_v"))
        pos = self.linear_pos(positions).reshape(1, -1, self.heads, self.width).transpose(1, 2)
        pos = pos.expand(batch, -1, -1, -1).reshape(batch * self.heads, -1, self.width)
        bias_u, bias_v = (x[None].expand(batch, -1, -1).reshape(-1, 1, self.width)
                          for x in (self.pos_bias_u, self.pos_bias_v))
        content = self.affine_bmm(q, k.transpose(1, 2), bias_u)
        relative = self.affine_bmm(q, pos.transpose(1, 2), bias_v)
        # HF's relative shift is a layout change over the computed score rows.
        relative = torch.cat((torch.zeros_like(relative[..., :1]), relative), dim=-1)
        relative = relative.reshape(-1, 2 * length, length)[:, 1:]
        relative = relative.reshape(-1, length, 2 * length - 1)[..., :length]
        scores = (content + relative) * self.width**-0.5
        if mask is not None:
            scores = scores.reshape(batch, self.heads, length, length).masked_fill(
                ~mask[:, None, None, :], torch.finfo(scores.dtype).min).reshape(-1, length, length)
        output = self.bmm(self.softmax(scores), v).reshape(batch, self.heads, length, self.width)
        return self.linear_out(output.transpose(1, 2).reshape(batch, length, -1))


class Convolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.hidden_size, config.conv_depthwise_kernel_size
        self.layer_norm = LayerNorm(width, promote_fp32=False)
        self.pointwise_conv1 = Conv1dNative(width, 2 * width, 1, bias=False)
        self.glu = GLU(precise_sigmoid=True)
        self.depthwise_conv = Conv1dNative(width, width, kernel, padding=(kernel - 1) // 2,
                                         groups=width, bias=False)
        self.batch_norm = BatchNorm2d(width)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1dNative(width, width, 1, bias=False)

    def forward(self, hidden, mask=None):
        hidden = self.glu(self.pointwise_conv1(self.layer_norm(hidden).transpose(1, 2)))
        hidden = self.activation(self.batch_norm(self.depthwise_conv(hidden)))
        return self.pointwise_conv2(hidden).transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, config, attention=RelativeAttention, convolution=Convolution, norm_eps=1e-5):
        super().__init__()
        for name in ("ffn1_layer_norm", "self_attn_layer_norm", "ffn2_layer_norm", "final_layer_norm"):
            setattr(self, name, LayerNorm(config.hidden_size, eps=norm_eps, promote_fp32=False))
        self.ffn1, self.ffn2 = FeedForward(config), FeedForward(config)
        self.self_attn, self.conv_module = attention(config), convolution(config)

    def forward(self, hidden, positions, mask):
        hidden = self.ffn1(self.ffn1_layer_norm(hidden)) * 0.5 + hidden
        hidden = self.self_attn(self.self_attn_layer_norm(hidden), positions, mask) + hidden
        hidden = hidden + self.conv_module(hidden, mask)
        hidden = self.ffn2(self.ffn2_layer_norm(hidden)) * 0.5 + hidden
        return self.final_layer_norm(hidden)


class ConformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = nn.Module()
        self.feature_extractor.conv_layers = nn.ModuleList(FeatureConv(config, i) for i in range(len(config.conv_dim)))
        self.feature_projection = FeatureProjection(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(ConformerLayer(config) for _ in range(config.num_hidden_layers))
        self.encoder.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        if config.mask_time_prob > 0 or config.mask_feature_prob > 0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))
        # HF initializes this constant table on CPU. GPU transcendental rounding
        # changes some BF16 entries and then the relative-attention scores.
        with torch.device("cpu"):
            position = torch.arange(config.max_source_positions, dtype=torch.float32)[:, None]
            rate = torch.exp(torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
                             * -(math.log(10000.0) / config.hidden_size))
            positive = torch.stack(((position * rate).sin(), (position * rate).cos()), dim=-1).flatten(1)
            negative = torch.stack(((-position * rate).sin(), (-position * rate).cos()), dim=-1).flatten(1)
        self.register_buffer("position_table", torch.cat((positive.flip(0), negative[1:]))[None], persistent=False)

    def forward(self, input_values, attention_mask=None):
        hidden = input_values[:, None]
        for layer in self.feature_extractor.conv_layers:
            hidden = layer(hidden)
        hidden, features = self.feature_projection(hidden.transpose(1, 2))
        mask = None
        if attention_mask is not None:
            lengths = attention_mask.sum(-1)
            for kernel, stride in zip(self.config.conv_kernel, self.config.conv_stride):
                lengths = (lengths - kernel) // stride + 1
            mask = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
            hidden = hidden.masked_fill(~mask[:, :, None], 0)
        length, middle = hidden.shape[1], self.position_table.shape[1] // 2
        positions = self.position_table[:, middle - length + 1:middle + length].to(hidden.dtype)
        for layer in self.encoder.layers:
            hidden = layer(hidden, positions, mask)
        return {"last_hidden_state": self.encoder.layer_norm(hidden), "extract_features": features}


def build_from_config(config, device, dtype):
    if (config.position_embeddings_type != "relative" or config.feat_extract_norm != "layer"
            or config.feat_extract_activation != "gelu" or config.hidden_act != "swish" or config.add_adapter):
        raise ValueError("Selected Conformer checkpoint uses relative attention, layer-normalized frontend and swish")
    return ConformerModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    # Pinned HF constructs this positional convolution but never calls it in
    # the Conformer encoder forward. The active relative-position path is above.
    state = {name: value for name, value in state_dict.items() if not name.startswith("encoder.pos_conv_embed.")}
    model.load_state_dict(state, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
