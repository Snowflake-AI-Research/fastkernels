"""TextNet's unfused convolution branches and both default base-model outputs."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU


class _Stem(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = Conv2d(c.stem_num_channels, c.stem_out_channels, c.stem_kernel_size,
                           stride=c.stem_stride, padding=c.stem_kernel_size // 2, bias=False)
        self.batch_norm = BatchNorm2d(c.stem_out_channels, c.batch_norm_eps)
        self.activation = ReLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class _BranchBlock(nn.Module):
    def __init__(self, c, incoming, outgoing, kernel, stride):
        super().__init__()
        h, w = kernel
        self.main_conv = Conv2d(incoming, outgoing, (h, w), stride, ((h-1)//2, (w-1)//2), bias=False)
        self.main_batch_norm = BatchNorm2d(outgoing, c.batch_norm_eps)
        self.vertical_conv = Conv2d(incoming, outgoing, (h, 1), stride, ((h-1)//2, 0), bias=False) if w != 1 else None
        self.vertical_batch_norm = BatchNorm2d(outgoing, c.batch_norm_eps) if w != 1 else None
        self.horizontal_conv = Conv2d(incoming, outgoing, (1, w), stride, (0, (w-1)//2), bias=False) if h != 1 else None
        self.horizontal_batch_norm = BatchNorm2d(outgoing, c.batch_norm_eps) if h != 1 else None
        self.rbr_identity = BatchNorm2d(incoming, c.batch_norm_eps) if incoming == outgoing and stride == 1 else None
        self.activation = ReLU()

    def forward(self, x):
        y = self.main_batch_norm(self.main_conv(x))
        if self.vertical_conv is not None:
            y = y + self.vertical_batch_norm(self.vertical_conv(x))
        if self.horizontal_conv is not None:
            y = y + self.horizontal_batch_norm(self.horizontal_conv(x))
        if self.rbr_identity is not None:
            y = y + self.rbr_identity(x)
        return self.activation(y)


class _Stage(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.stage = nn.ModuleList([
            _BranchBlock(c, c.hidden_sizes[index] if j == 0 else c.hidden_sizes[index+1],
                         c.hidden_sizes[index+1], kernel, stride)
            for j, (kernel, stride) in enumerate(zip(c.conv_layer_kernel_sizes[index], c.conv_layer_strides[index]))
        ])

    def forward(self, x):
        for layer in self.stage:
            x = layer(x)
        return x


class TextNetModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = _Stem(c)
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList([_Stage(c, i) for i in range(len(c.conv_layer_kernel_sizes))])
        self.pool = GlobalAvgPool2d(keepdim=True)

    def forward(self, pixel_values):
        x = self.stem(pixel_values)
        for stage in self.encoder.stages:
            x = stage(x)
        # Adaptive 2x2 pooling: each actual bin uses the existing spatial mean.
        h, w = x.shape[-2:]
        rows = []
        for i in range(2):
            rows.append(torch.cat([self.pool(x[..., i*h//2:((i+1)*h+1)//2,
                                                 j*w//2:((j+1)*w+1)//2]) for j in range(2)], dim=-1))
        return {"last_hidden_state": x, "pooler_output": torch.cat(rows, dim=-2)}


def build_from_config(config, device, dtype):
    if config.stem_act_func != "relu":
        raise ValueError("The declared TextNet checkpoint uses a ReLU stem")
    return TextNetModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
