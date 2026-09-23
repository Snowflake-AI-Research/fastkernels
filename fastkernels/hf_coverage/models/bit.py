"""BiT's default preactivation residual network with fixed standardized weights."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU

from ..runner import Workload


class _Norm(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm = GroupNorm(groups, channels, eps=1e-5)
        self.act = ReLU()

    def forward(self, x):
        return self.act(self.norm(x))


def _conv(source, target, kernel, stride=1):
    return Conv2d(source, target, kernel, stride=stride,
                  padding=((stride - 1) + kernel - 1) // 2, bias=False)


class _Block(nn.Module):
    def __init__(self, source, target, groups, stride, first):
        super().__init__()
        middle = target // 4
        self.norm1 = _Norm(source, groups)
        self.norm2 = _Norm(middle, groups)
        self.norm3 = _Norm(middle, groups)
        self.conv1 = _conv(source, middle, 1)
        self.conv2 = _conv(middle, middle, 3, stride)
        self.conv3 = _conv(middle, target, 1)
        self.downsample = _conv(source, target, 1, stride) if first else None

    def forward(self, x):
        preactivated = self.norm1(x)
        residual = x if self.downsample is None else self.downsample(preactivated)
        x = self.conv1(preactivated)
        x = self.conv2(self.norm2(x))
        return self.conv3(self.norm3(x)) + residual


class BitModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stem = _conv(config.num_channels, config.embedding_size, 7, 2)
        self.stem_pool = MaxPool2d(3, stride=2, padding=0)
        self.stages = nn.ModuleList()
        source = config.embedding_size
        for index, (width, depth) in enumerate(zip(config.hidden_sizes, config.depths)):
            blocks = nn.ModuleList()
            for j in range(depth):
                blocks.append(_Block(source, width, config.num_groups, 2 if index and not j else 1, j == 0))
                source = width
            self.stages.append(blocks)
        self.norm = _Norm(source, config.num_groups)
        self.pool = GlobalAvgPool2d(keepdim=True)

    def forward(self, pixel_values):
        hidden = self.stem(pixel_values)
        # HF pads with zeros before unpadded max pooling, rather than -inf.
        padded = hidden.new_zeros(hidden.shape[0], hidden.shape[1], hidden.shape[2] + 2, hidden.shape[3] + 2)
        padded[:, :, 1:-1, 1:-1] = hidden
        hidden = self.stem_pool(padded)
        for stage in self.stages:
            for block in stage:
                hidden = block(hidden)
        hidden = self.norm(hidden)
        return {"last_hidden_state": hidden, "pooler_output": self.pool(hidden)}


def build_from_config(config, device, dtype):
    if (config.layer_type != "preactivation" or config.hidden_act != "relu" or
            config.global_padding is not None or config.embedding_dynamic_padding or
            config.width_factor != 1 or config.output_stride != 32):
        raise ValueError("Preserve BiT's default preactivation, static padding and stride32 path")
    return BitModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name, parameter in model.named_parameters():
        if name.startswith("stem."):
            source = name.replace("stem.", "embedder.convolution.")
        else:
            source = name
            if source.startswith("stages."):
                parts = source.split(".")
                source = ".".join(("encoder", "stages", parts[1], "layers", *parts[2:]))
            source = source.replace(".norm.", ".").replace("downsample.", "downsample.conv.")
        value = remaining.pop(source).to(device=parameter.device, dtype=parameter.dtype)
        if value.ndim == 4:
            # Weights are constant during inference. Existing BatchNorm with no
            # running statistics performs HF's per-output-filter standardization.
            standardize = BatchNorm2d(value.shape[0], eps=1e-8, momentum=0, affine=False, track_running_stats=False)
            value = standardize(value.reshape(1, value.shape[0], -1)).reshape_as(value)
        mapped[name] = value
    if remaining:
        raise KeyError(f"Unmapped BiT state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
