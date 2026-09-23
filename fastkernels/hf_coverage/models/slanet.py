"""SLANet table recognition, including native greedy recurrent decoding."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from .pp_lcnet import ConvLayer, divisible, build_from_config as build_backbone


class Depthwise(nn.Module):
    def __init__(self, width, kernel, stride=1):
        super().__init__()
        self.depthwise_convolution = ConvLayer(width, width, kernel, stride=stride, groups=width)
        self.pointwise_convolution = ConvLayer(width, width, 1)

    def forward(self, hidden):
        return self.pointwise_convolution(self.depthwise_convolution(hidden))


class Bottleneck(nn.Module):
    def __init__(self, width, kernel):
        super().__init__()
        self.conv1, self.conv2 = ConvLayer(width, width, 1), Depthwise(width, kernel)

    def forward(self, hidden):
        return self.conv2(self.conv1(hidden))


class CSP(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.conv1, self.conv2 = ConvLayer(2 * width, width // 2, 1), ConvLayer(2 * width, width // 2, 1)
        self.conv3 = ConvLayer(width, width, 1)
        self.bottlenecks = nn.ModuleList([Bottleneck(width // 2, config.csp_kernel_size)
                                        for _ in range(config.csp_num_blocks)])

    def forward(self, hidden):
        residual, hidden = self.conv1(hidden), self.conv2(hidden)
        for block in self.bottlenecks:
            hidden = block(hidden)
        return self.conv3(torch.cat((hidden, residual), dim=1))


class Pyramid(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        width = config.post_conv_out_channels
        self.channel_projector = nn.ModuleList([ConvLayer(channel, width, 1) for channel in channels])
        self.top_down_blocks = nn.ModuleList([CSP(config, width) for _ in channels[1:]])
        self.bottom_up_blocks = nn.ModuleList([CSP(config, width) for _ in channels[1:]])
        self.downsamples = nn.ModuleList([Depthwise(width, config.csp_kernel_size, stride=2) for _ in channels[1:]])
        self.resize = Interpolate()

    def forward(self, features):
        features = [project(feature) for project, feature in zip(self.channel_projector, features)]
        top = [features[-1]]
        for block, low in zip(self.top_down_blocks, reversed(features[:-1])):
            top.append(block(torch.cat((self.resize(top[-1], size=low.shape[-2:], mode="nearest"), low), dim=1)))
        top.reverse()
        hidden = top[0]
        for down, block, high in zip(self.downsamples, self.bottom_up_blocks, top[1:]):
            hidden = block(torch.cat((down(hidden), high), dim=1))
        return hidden.flatten(2).transpose(1, 2)


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, classes):
        super().__init__()
        self.input_to_hidden = Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden, self.score = Linear(hidden_size, hidden_size), Linear(hidden_size, 1, bias=False)
        self.input_gates, self.hidden_gates = Linear(input_size + classes, 3 * hidden_size), Linear(hidden_size, 3 * hidden_size)
        self.softmax, self.tanh, self.sigmoid = Softmax(dim=1), Tanh(), Sigmoid()
        self.matmul, self.product = BatchMatMul(), ProductGate()

    def multiply(self, left, right):
        return self.product(torch.cat((left, right), dim=-1))

    def forward(self, state, memory, token):
        attention = self.score(self.tanh(self.input_to_hidden(memory) + self.hidden_to_hidden(state).unsqueeze(1)))
        probs = self.softmax(attention.float()).to(attention.dtype).transpose(1, 2)
        context = self.matmul(probs, memory).squeeze(1)
        ir, iz, inn = self.input_gates(torch.cat((context, token), dim=1)).chunk(3, dim=-1)
        hr, hz, hn = self.hidden_gates(state).chunk(3, dim=-1)
        reset, update = self.sigmoid(ir + hr), self.sigmoid(iz + hz)
        candidate = self.tanh(inn + self.multiply(reset, hn))
        return self.multiply(1 - update, candidate) + self.multiply(update, state)


class StructureHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.structure_attention_cell = AttentionGRU(config.post_conv_out_channels, config.hidden_size, config.out_channels)
        self.structure_generator = nn.Module()
        self.structure_generator.fc1 = Linear(config.hidden_size, config.hidden_size)
        self.structure_generator.fc2 = Linear(config.hidden_size, config.out_channels)
        self.select, self.softmax = CodecTop1(), Softmax()
        self.finished_reduce = SegmentCSR()

    def all_finished(self, finished):
        offsets = torch.tensor([0, finished.numel()], device=finished.device, dtype=torch.long)
        minimum = self.finished_reduce(finished.float(), offsets, reduce="min")
        # Top1 returns discrete 0/1 indices; a zero minimum ties at index zero.
        flag = self.select(torch.stack((torch.zeros_like(minimum), minimum), dim=-1))
        return bool(flag[0])

    def forward(self, memory):
        config, batch = self.config, memory.shape[0]
        state = torch.zeros(batch, config.hidden_size, dtype=torch.float32, device=memory.device)
        token_ids = torch.zeros(batch, dtype=torch.long, device=memory.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        outputs = []
        for _ in range(config.max_text_length + 1):
            token = torch.zeros(batch, config.out_channels, dtype=torch.float32, device=memory.device)
            token.scatter_(1, token_ids[:, None], 1)
            state = self.structure_attention_cell(state, memory.float(), token)
            scores = self.structure_generator.fc2(self.structure_generator.fc1(state))
            token_ids = self.select(scores)
            outputs.append(scores)
            finished = finished | (token_ids == config.out_channels - 1)
            if self.all_finished(finished):
                break
        return self.softmax(torch.stack(outputs, dim=1).float()).to(memory.dtype)


class SLANet(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.vision_backbone = backbone
        b = config.backbone_config
        channels = [divisible(stage[-1][2] * b.scale, b.divisor) for stage in b.block_configs]
        self.backbone.post_csp_pan = Pyramid(config, channels[1:])
        self.head = StructureHead(config)

    def forward(self, pixel_values):
        features = list(self.backbone.vision_backbone(pixel_values).values())
        return {"last_hidden_state": self.head(self.backbone.post_csp_pan(features))}


def build_from_config(config, device, dtype):
    if config.hidden_act != "hardswish":
        raise ValueError("The selected SLANet checkpoint uses hard-swish convolution blocks")
    backbone = build_backbone(config.backbone_config, device, dtype)
    return SLANet(config, backbone).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        source = destination
        for target, origin in (("input_gates.weight", "rnn.weight_ih"), ("input_gates.bias", "rnn.bias_ih"),
                               ("hidden_gates.weight", "rnn.weight_hh"), ("hidden_gates.bias", "rnn.bias_hh")):
            source = source.replace(target, origin)
        mapped[destination] = remaining.pop(source)
    if remaining:
        raise ValueError(f"Unmapped table-recognition weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
