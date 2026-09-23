"""The documented deterministic VibeVoice acoustic encode/decode computation."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp


class CausalConv(nn.Module):
    def __init__(self, source, target, kernel, stride=1, groups=1):
        super().__init__()
        self.conv = Conv1dNative(source, target, kernel, stride=stride, groups=groups)
        self.left_pad = kernel - stride
        self.pad = Pad()

    def forward(self, hidden):
        return self.conv(self.pad(hidden, (self.left_pad, 0)))


class CausalUpsample(nn.Module):
    def __init__(self, source, target, ratio):
        super().__init__()
        self.convtr = ConvTranspose1d(source, target, ratio * 2, stride=ratio)
        self.right_trim = ratio

    def forward(self, hidden):
        return self.convtr(hidden)[..., :-self.right_trim]


class ConvNextBlock(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.norm = T5LayerNorm(width, eps=config.rms_norm_eps)
        self.ffn_norm = T5LayerNorm(width, eps=config.rms_norm_eps)
        self.ffn = VitEncoderMlp(width, config.ffn_expansion * width)
        self.mixer = CausalConv(width, width, config.kernel_size, groups=width)
        self.gamma = nn.Parameter(torch.empty(width))
        self.ffn_gamma = nn.Parameter(torch.empty(width))
        self.product = ProductGate()

    def scaled(self, hidden, scale):
        return self.product(torch.cat((hidden, scale.expand_as(hidden)), dim=-1))

    def forward(self, hidden):
        normalized = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        mixed = self.mixer(normalized).transpose(1, 2)
        hidden = hidden + self.scaled(mixed, self.gamma).transpose(1, 2)
        mixed = self.ffn(self.ffn_norm(hidden.transpose(1, 2)))
        return hidden + self.scaled(mixed, self.ffn_gamma).transpose(1, 2)


class Stage(nn.Module):
    def __init__(self, config, convolution, width, depth):
        super().__init__()
        self.conv = convolution
        self.stage = nn.ModuleList(ConvNextBlock(config, width) for _ in range(depth))

    def forward(self, hidden):
        hidden = self.conv(hidden)
        for block in self.stage:
            hidden = block(hidden)
        return hidden


class Stack(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()
        ratios, depths = list(config.downsampling_ratios), list(config.depths)
        widths = [config.num_filters * 2**index for index in range(len(depths))]
        if decoder:
            ratios, depths, widths = ratios[::-1], depths[::-1], widths[::-1]
        self.stem = Stage(config, CausalConv(
            config.hidden_size if decoder else config.channels, widths[0], config.kernel_size,
        ), widths[0], depths[0])
        self.conv_layers = nn.ModuleList()
        for index, ratio in enumerate(ratios):
            conv = (CausalUpsample(widths[index], widths[index + 1], ratio) if decoder
                    else CausalConv(widths[index], widths[index + 1], ratio * 2, stride=ratio))
            self.conv_layers.append(Stage(config, conv, widths[index + 1], depths[index + 1]))
        self.head = CausalConv(widths[-1], config.channels if decoder else config.hidden_size,
                               config.kernel_size)

    def forward(self, hidden):
        hidden = self.stem(hidden)
        for stage in self.conv_layers:
            hidden = stage(hidden)
        return self.head(hidden)


class VibeVoiceAcousticTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Stack(config, decoder=False)
        self.decoder = Stack(config, decoder=True)

    def forward(self, input_values):
        latents = self.encoder(input_values).transpose(1, 2)
        return {"audio": self.decoder(latents.transpose(1, 2)), "latents": latents}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or len(config.depths) != len(config.downsampling_ratios) + 1:
        raise ValueError("VibeVoice's documented codec requires GELU and all configured stages")
    return VibeVoiceAcousticTokenizer(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for target, parameter in model.state_dict().items():
        source = target.replace(".ffn.fc1.", ".ffn.linear1.").replace(".ffn.fc2.", ".ffn.linear2.")
        parts = source.split(".")
        if parts[0:2] == ["decoder", "conv_layers"] and parts[3] == "conv":
            parts[3] = "convtr"
            source = ".".join(parts)
        if state[source].shape != parameter.shape:
            raise ValueError(f"VibeVoice weight shape mismatch: {source}")
        mapped[target] = state[source]
        consumed.add(source)
    if consumed != set(state):
        raise ValueError(f"VibeVoice unmapped state: {sorted(set(state) - consumed)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
