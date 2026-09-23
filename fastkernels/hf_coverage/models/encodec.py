"""EnCodec 24kHz default waveform encode/decode composition."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.models.dac import VectorQuantizer
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.encodec_recurrent_linear import EncodecRecurrentLinear
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.cosyvoice3_hifigan import CausalConvRNNF0Predictor


def elu():
    # Reuse an unchanged callable child; the unused parent exists only at construction.
    return CausalConvRNNF0Predictor(in_channels=1, cond_channels=1).condnet[1]


class CausalConv(nn.Module):
    def __init__(self, source, target, kernel, stride=1, dilation=1, transpose=False):
        super().__init__()
        op = ConvTranspose1d if transpose else Conv1dNative
        self.conv = op(source, target, kernel, stride=stride, dilation=dilation)
        self.total = (kernel - 1) * dilation + 1 - stride
        self.stride, self.transpose, self.pad = stride, transpose, Pad()

    def forward(self, hidden):
        if self.transpose:
            output = self.conv(hidden)
            return output[..., :-self.total] if self.total else output
        extra = (-hidden.shape[-1]) % self.stride
        length = hidden.shape[-1]
        extension = max(0, max(self.total, extra) - length + 1)
        if extension:
            hidden = self.pad(hidden, (0, extension))
        # Native reflection padding is a fixed permutation/copy, not activation math.
        indices = torch.cat((torch.arange(self.total, 0, -1, device=hidden.device),
                             torch.arange(hidden.shape[-1], device=hidden.device),
                             torch.arange(hidden.shape[-1] - 2, hidden.shape[-1] - extra - 2,
                                          -1, device=hidden.device)))
        hidden = hidden.index_select(-1, indices)
        if extension:
            hidden = hidden[..., :-extension]
        return self.conv(hidden)


class LSTM(nn.Module):
    def __init__(self, width, count):
        super().__init__()
        self.input = nn.ModuleList([Linear(width, 4 * width) for _ in range(count)])
        self.recurrent = nn.ModuleList([EncodecRecurrentLinear(width, 4 * width) for _ in range(count)])
        self.sigmoid, self.tanh, self.product = Sigmoid(), Tanh(), ProductGate()

    def multiply(self, left, right):
        return self.product(torch.cat((left, right), dim=-1))

    def forward(self, hidden):
        sequence = hidden.transpose(1, 2)
        residual = sequence
        for input_op, recurrent_op in zip(self.input, self.recurrent):
            # Retain BF16 GEMM rounding; native cell arithmetic adds biases in FP32.
            projected = input_op.matmul(sequence, input_op.weight)
            state = sequence.new_zeros(sequence.shape[0], sequence.shape[-1])
            cell, outputs = torch.zeros_like(state), []
            for step in range(sequence.shape[1]):
                i, f, g, o = recurrent_op(state, projected[:, step], input_op.bias).chunk(4, dim=-1)
                next_cell = (self.multiply(self.sigmoid(f), cell.float())
                             + self.multiply(self.sigmoid(i), self.tanh(g)))
                # h uses the unrounded new cell; only the carried states round.
                state = self.multiply(self.sigmoid(o), self.tanh(next_cell)).to(sequence.dtype)
                cell = next_cell.to(sequence.dtype)
                outputs.append(state)
            sequence = torch.stack(outputs, dim=1)
        return (sequence + residual).transpose(1, 2)


class Residual(nn.Module):
    def __init__(self, config, width, dilation):
        super().__init__()
        smaller = width // config.compress
        self.block = nn.ModuleList([elu(), CausalConv(width, smaller, config.residual_kernel_size, dilation=dilation),
                                    elu(), CausalConv(smaller, width, 1)])
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
        if decoder:
            layers.append(LSTM(width, config.num_lstm_layers))
        for ratio in config.upsampling_ratios if decoder else reversed(config.upsampling_ratios):
            if decoder:
                layers += [elu(), CausalConv(width, width // 2, 2 * ratio, stride=ratio, transpose=True)]
                width //= 2
            for index in range(config.num_residual_layers):
                layers.append(Residual(config, width, config.dilation_growth_rate ** index))
            if not decoder:
                layers += [elu(), CausalConv(width, width * 2, 2 * ratio, stride=ratio)]
                width *= 2
        if not decoder:
            layers.append(LSTM(width, config.num_lstm_layers))
        layers += [elu(), CausalConv(width, config.audio_channels if decoder else config.hidden_size, config.last_kernel_size)]
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class Codebook(nn.Module):
    squared_row_norm = VectorQuantizer.squared_row_norm

    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.codebook_size, config.codebook_dim)
        self.product, self.reduce = ProductGate(), SegmentCSR()
        self.matmul, self.select = BatchMatMul(), CodecTop1()

    def encode(self, hidden):
        batch, width, length = hidden.shape
        rows = hidden.transpose(1, 2).reshape(-1, width)
        codes = self.embedding.emb.weight
        dot = self.matmul((2 * rows).unsqueeze(0), codes.T.unsqueeze(0))[0]
        distance = -(self.squared_row_norm(rows) - dot + self.squared_row_norm(codes).T)
        return self.select(distance).reshape(batch, length)

    def decode(self, indices):
        return self.embedding(indices).transpose(1, 2)


class Encodec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder, self.decoder = Stack(config), Stack(config, decoder=True)
        self.mask_product = ProductGate()
        frame_rate = math.ceil(config.sampling_rate / math.prod(config.upsampling_ratios))
        count = int(1000 * config.target_bandwidths[-1] // (frame_rate * math.ceil(math.log2(config.codebook_size))))
        self.codebooks = nn.ModuleList([Codebook(config) for _ in range(count)])
        self.active = max(1, math.floor(config.target_bandwidths[0] * 1000 /
                                      (math.log2(config.codebook_size) * frame_rate)))

    def forward(self, input_values, padding_mask=None):
        if padding_mask is not None:
            # HF applies the processor's supplied mask before encoding, even
            # when normalization and chunking are disabled in the 24kHz model.
            mask = padding_mask.reshape(padding_mask.shape[0], -1, padding_mask.shape[-1])
            mask = mask.bool().to(input_values.dtype).expand_as(input_values)
            input_values = self.mask_product(torch.cat((input_values, mask), dim=-1))
        residual = self.encoder(input_values)
        indices = []
        for codebook in self.codebooks[:self.active]:
            codes = codebook.encode(residual)
            residual = residual - codebook.decode(codes)
            indices.append(codes)
        quantized = torch.tensor(0.0, device=input_values.device)
        for codebook, codes in zip(self.codebooks, indices):
            quantized = quantized + codebook.decode(codes)
        audio = self.decoder(quantized)[..., :input_values.shape[-1]]
        return {'audio_codes': torch.stack(indices, dim=1).unsqueeze(0), 'audio_values': audio}


def build_from_config(config, device, dtype):
    if (config.normalize or config.chunk_length_s is not None or not config.use_causal_conv
            or config.pad_mode != 'reflect' or config.norm_type != 'weight_norm' or config.trim_right_ratio != 1):
        raise ValueError('EnCodec case requires the published 24kHz default computation')
    return Encodec(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for name, target in model.state_dict().items():
        source = name
        if name.startswith('codebooks.'):
            source = 'quantizer.layers.' + name.split('.')[1] + '.codebook.embed'
        elif '.input.' in name or '.recurrent.' in name:
            marker = '.input.' if '.input.' in name else '.recurrent.'
            prefix, tail = name.split(marker)
            index, kind = tail.split('.')
            source = f'{prefix}.lstm.{kind}_{"ih" if marker == ".input." else "hh"}_l{index}'
        if source.endswith('.conv.weight'):
            prefix = source[:-len('weight')] + 'parametrizations.weight.'
            v, g = prefix + 'original1', prefix + 'original0'
            # Fixed inference weights: preserve native requested-dtype weight normalization.
            value = torch._weight_norm(state[v].to(device=target.device, dtype=target.dtype), state[g].to(device=target.device, dtype=target.dtype), 0)
            consumed.update((v, g))
        else:
            value = state[source]
            consumed.add(source)
        if value.shape != target.shape:
            raise ValueError(f'EnCodec weight mismatch: {source}')
        mapped[name] = value
    unused = set(state) - consumed
    allowed = {f'quantizer.layers.{i}.codebook.{field}' for i in range(len(model.codebooks))
               for field in ('inited', 'cluster_size', 'embed_avg')}
    if unused != allowed:
        raise ValueError(f'EnCodec unexpected unused state: {unused ^ allowed}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
