"""UnivNet vocoder with location-dependent kernels composed through BMM."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.tensor_ops import Pad


class LeakyReLU(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope, self.relu = slope, ReLU()

    def forward(self, hidden):
        # Disjoint positive and negative branches avoid cancellation.
        return self.relu(hidden) - self.relu(-hidden) * self.slope


class PredictorResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.model_in_channels, config.kernel_predictor_conv_size
        self.conv1 = Conv1dNative(width, width, kernel, padding=kernel // 2)
        self.conv2 = Conv1dNative(width, width, kernel, padding=kernel // 2)
        self.activation = LeakyReLU(config.leaky_relu_slope)

    def forward(self, hidden):
        return hidden + self.activation(self.conv2(self.activation(self.conv1(hidden))))


class KernelPredictor(nn.Module):
    def __init__(self, config, kernel, layers):
        super().__init__()
        self.width, self.kernel, self.count = config.model_hidden_channels, kernel, layers
        width, size = config.kernel_predictor_hidden_channels, config.kernel_predictor_conv_size
        self.input_conv = Conv1dNative(config.num_mel_bins, width, 5, padding=2)
        self.resblocks = nn.ModuleList(PredictorResidual(config) for _ in range(config.kernel_predictor_num_blocks))
        self.kernel_conv = Conv1dNative(width, self.width * self.width * 2 * kernel * layers, size, padding=size // 2)
        self.bias_conv = Conv1dNative(width, 2 * self.width * layers, size, padding=size // 2)
        self.activation = LeakyReLU(config.leaky_relu_slope)

    def forward(self, spectrogram):
        batch, _, length = spectrogram.shape
        hidden = self.activation(self.input_conv(spectrogram))
        for block in self.resblocks:
            hidden = block(hidden)
        kernels = self.kernel_conv(hidden).reshape(batch, self.count, self.width, 2 * self.width, self.kernel, length)
        biases = self.bias_conv(hidden).reshape(batch, self.count, 2 * self.width, length)
        return kernels, biases


class LocalResidual(nn.Module):
    def __init__(self, config, kernel, dilation):
        super().__init__()
        width = config.model_hidden_channels
        self.conv = Conv1dNative(width, width, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation)
        self.activation = LeakyReLU(config.leaky_relu_slope)
        self.bmm, self.pad = BatchMatMul(), Pad()
        self.sigmoid, self.tanh, self.product = Sigmoid(), Tanh(), ProductGate()

    def forward(self, hidden, kernels, bias, hop):
        residual = hidden
        hidden = self.activation(self.conv(self.activation(hidden)))
        batch, source, length = hidden.shape
        _, _, target, kernel, frames = kernels.shape
        # Each frame has one predicted kernel, shared by its hop output samples.
        windows = self.pad(hidden, (kernel // 2, kernel // 2)).unfold(2, kernel, 1)
        windows = windows.reshape(batch, source, frames, hop, kernel).permute(0, 2, 3, 1, 4)
        weights = kernels.permute(0, 4, 1, 3, 2).reshape(batch * frames, source * kernel, target)
        output = self.bmm(windows.reshape(batch * frames, hop, source * kernel), weights)
        output = output.reshape(batch, frames, hop, target).permute(0, 3, 1, 2)
        output = (output + bias[..., None]).reshape(batch, target, length)
        gate, value = output.chunk(2, dim=1)
        packed = torch.cat((self.sigmoid(gate).transpose(1, 2), self.tanh(value).transpose(1, 2)), dim=-1)
        return residual + self.product(packed).transpose(1, 2)


class VocoderBlock(nn.Module):
    def __init__(self, config, index, hop):
        super().__init__()
        width, stride = config.model_hidden_channels, config.resblock_stride_sizes[index]
        kernel, dilations = config.resblock_kernel_sizes[index], config.resblock_dilation_sizes[index]
        self.hop = hop
        self.convt_pre = ConvTranspose1d(width, width, 2 * stride, stride=stride,
                                        padding=stride // 2 + stride % 2, output_padding=stride % 2)
        self.kernel_predictor = KernelPredictor(config, kernel, len(dilations))
        self.resblocks = nn.ModuleList(LocalResidual(config, kernel, dilation) for dilation in dilations)
        self.activation = LeakyReLU(config.leaky_relu_slope)

    def forward(self, hidden, spectrogram):
        hidden = self.convt_pre(self.activation(hidden))
        kernels, biases = self.kernel_predictor(spectrogram)
        for index, block in enumerate(self.resblocks):
            hidden = block(hidden, kernels[:, index], biases[:, index], self.hop)
        return hidden


class UnivNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_pre = Conv1dNative(config.model_in_channels, config.model_hidden_channels, 7)
        self.conv_post = Conv1dNative(config.model_hidden_channels, 1, 7)
        self.resblocks = nn.ModuleList()
        hop = 1
        for index, stride in enumerate(config.resblock_stride_sizes):
            hop *= stride
            self.resblocks.append(VocoderBlock(config, index, hop))
        self.activation, self.tanh, self.pad = LeakyReLU(config.leaky_relu_slope), Tanh(), Pad()

    def forward(self, input_features, noise_sequence):
        hidden = self.conv_pre(self.reflect(noise_sequence.transpose(1, 2)))
        spectrogram = input_features.transpose(1, 2)
        for block in self.resblocks:
            hidden = block(hidden, spectrogram)
        return self.tanh(self.conv_post(self.reflect(self.activation(hidden)))).squeeze(1)

    @staticmethod
    def reflect(hidden):
        return torch.cat((hidden[..., 1:4].flip(-1), hidden, hidden[..., -4:-1].flip(-1)), dim=-1)


def build_from_config(config, device, dtype):
    return UnivNet(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"waveforms": model(inputs["input_features"], inputs["noise_sequence"])})}
