"""PP-OCRv5 mobile text probability maps with every native backbone branch."""

import torch
from torch import nn
from .pp_lcnet_v3 import build_from_config as build_backbone
from .pp_lcnet import divisible, make_workloads
from .depth_anything import NonoverlappingTransposeConv
from ..patches.dfine_clamp import DFineClamp
from ..patches.product_gate import ProductGate
from ..patches.linear import PostBiasLinear
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.interpolate import Interpolate


class _Transpose(NonoverlappingTransposeConv):
    def forward(self, hidden):
        return super().forward(hidden.contiguous(memory_format=torch.channels_last))


class _Squeeze(nn.Module):
    def __init__(self, width, reduction):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d(keepdim=True)
        self.conv1, self.conv2 = Conv2d(width, width // reduction, 1), Conv2d(width // reduction, width, 1)
        self.act_fn, self.clamp, self.product = ReLU(), DFineClamp(), ProductGate()

    def forward(self, hidden):
        gate = self.clamp(.2 * self.conv2(self.act_fn(self.conv1(self.avg_pool(hidden)))) + .5, 0., 1.)
        return self.product(torch.cat((hidden, gate.expand_as(hidden)), dim=-1))


class _RSE(nn.Module):
    def __init__(self, source, target, kernel, reduction):
        super().__init__()
        self.in_conv = Conv2d(source, target, kernel, padding=kernel // 2, bias=False)
        self.squeeze_excitation_block = _Squeeze(target, reduction)

    def forward(self, hidden):
        hidden = self.in_conv(hidden)
        return hidden + self.squeeze_excitation_block(hidden)


class _Neck(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.neck_out_channels
        self.insert_conv = nn.ModuleList([_RSE(x, width, 1, config.reduction) for x in config.layer_list_out_channels])
        self.input_conv = nn.ModuleList([_RSE(width, width // 4, 3, config.reduction) for _ in range(4)])
        self.interpolate, self.mode = Interpolate(), config.interpolate_mode

    def forward(self, features):
        fused = [layer(x) for layer, x in zip(self.insert_conv, features)]
        for i in range(2, -1, -1):
            fused[i] = fused[i] + self.interpolate(fused[i + 1], scale_factor=2, mode=self.mode)
        features = [layer(x) for layer, x in zip(self.input_conv, fused)]
        return torch.cat([self.interpolate(x, scale_factor=2**i, mode=self.mode) if i else x
                          for i, x in enumerate(features)][::-1], dim=1)


class _ConvNorm(nn.Module):
    def __init__(self, source, target, kernel, stride=1, padding=1, bias=False, transpose=False):
        super().__init__()
        self.convolution = (_Transpose(source, target, kernel) if transpose else
                            Conv2d(source, target, kernel, stride=stride, padding=padding, bias=bias))
        self.norm, self.act_fn = BatchNorm2d(target), ReLU()

    def forward(self, hidden):
        return self.act_fn(self.norm(self.convolution(hidden)))


class _Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.neck_out_channels
        self.conv_down = _ConvNorm(width, width // 4, config.kernel_list[0], padding=config.kernel_list[0] // 2)
        self.conv_up = _ConvNorm(width // 4, width // 4, 2, transpose=True)
        self.conv_final = _Transpose(width // 4, 1, 2)
        self.sigmoid = Sigmoid()

    def forward(self, hidden):
        return self.sigmoid(self.conv_final(self.conv_up(self.conv_down(hidden))))


class _Detector(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = build_backbone(config.backbone_config, device, dtype)
        c = config.backbone_config
        channels = [divisible(c.block_configs[index - 1][-1][2] * c.scale, c.divisor) for index in c.out_indices]
        self.model.layer = nn.ModuleList([Conv2d(source, target, 1) for source, target in zip(channels, config.layer_list_out_channels)])
        self.model.neck = _Neck(config)
        self.head = _Head(config)

    def forward(self, pixel_values):
        features = list(self.model.backbone(pixel_values).values())
        features = [layer(x) for layer, x in zip(self.model.layer, features)]
        return {'last_hidden_state': self.head(self.model.neck(features))}


def build_from_config(config, device, dtype):
    if list(config.kernel_list) != [3, 2, 2] or config.backbone_config.model_type != 'pp_lcnet_v3':
        raise ValueError('Preserve the checkpoint mobile backbone and nonoverlapping transpose head')
    model = _Detector(config, device, dtype)
    configure_transposes(model, device)
    return model.to(device=device, dtype=dtype).eval()


def configure_transposes(model, device):
    # Native CUDA transpose convolution adds bias after rounding its product;
    # CPU's oneDNN implementation fuses it. Preserve the selected device's
    # numerical boundary with existing Linear/PostBiasLinear operations.
    if torch.device(device).type == 'cuda':
        for module in model.modules():
            if isinstance(module, _Transpose):
                source = module.projection.weight.shape[1]
                module.projection = PostBiasLinear(source, module.output_width * module.factor**2)


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for key, value in state_dict.items():
        module_name, field = key.rsplit('.', 1)
        module = model.get_submodule(module_name)
        if isinstance(module, NonoverlappingTransposeConv):
            if field == 'weight':
                value = value.permute(1, 2, 3, 0).reshape(-1, value.shape[0]).contiguous()
            elif field == 'bias':
                value = value.repeat_interleave(module.factor**2)
            key = module_name + '.projection.' + field
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
