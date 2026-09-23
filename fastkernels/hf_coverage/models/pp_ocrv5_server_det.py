"""PP-OCRv5 server text detector, including native repeated interpolation."""

import torch
from torch import nn
from .hgnet_v2 import build_from_config as build_backbone
from .pp_ocrv5_mobile_det import _ConvNorm, _Head, configure_transposes, load_state_dict_into, make_workloads
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid


class _Intraclass(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.neck_out_channels // 4
        reduced = width // config.reduce_factor
        specs = config.intraclass_block_config
        self.conv_reduce_channel = Conv2d(width, reduced, specs['reduce_channel'][0], stride=specs['reduce_channel'][1], padding=specs['reduce_channel'][2])
        for direction in ('vertical_long_to_small', 'horizontal_small_to_long', 'symmetric_conv_long'):
            for ratio in ('longratio', 'midratio', 'shortratio'):
                name = f'{direction}_conv_{ratio}' if direction != 'symmetric_conv_long' else f'{direction}_{ratio}'
                kernel, stride, padding = specs[name]
                self.add_module(name, Conv2d(reduced, reduced, kernel, stride=stride, padding=padding))
        kernel, stride, padding = specs['return_channel']
        self.conv_final = _ConvNorm(reduced, width, kernel, stride=stride, padding=padding, bias=True)

    def forward(self, hidden):
        residual = hidden
        hidden = self.conv_reduce_channel(hidden)
        for ratio in ('longratio', 'midratio', 'shortratio'):
            hidden = (getattr(self, f'symmetric_conv_long_{ratio}')(hidden)
                      + getattr(self, f'vertical_long_to_small_conv_{ratio}')(hidden)
                      + getattr(self, f'horizontal_small_to_long_conv_{ratio}')(hidden))
        return residual + self.conv_final(hidden)


class _Neck(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.neck_out_channels
        self.input_channel_adjustment_convolution = nn.ModuleList([Conv2d(x, width, 1, bias=False) for x in config.backbone_config.stage_out_channels])
        self.input_feature_projection_convolution = nn.ModuleList([Conv2d(width, width // 4, 9, padding=4, bias=False) for _ in range(4)])
        self.path_aggregation_head_convolution = nn.ModuleList([Conv2d(width // 4, width // 4, 3, stride=2, padding=1, bias=False) for _ in range(3)])
        self.path_aggregation_lateral_convolution = nn.ModuleList([Conv2d(width // 4, width // 4, 9, padding=4, bias=False) for _ in range(4)])
        self.intraclass_blocks = nn.ModuleList([_Intraclass(config) for _ in range(config.intraclass_block_number)])
        self.interpolate, self.mode, self.scales = Interpolate(), config.interpolate_mode, config.scale_factor_list

    def forward(self, features):
        adjusted = [layer(x) for layer, x in zip(self.input_channel_adjustment_convolution, features)]
        fused = list(adjusted)
        for i in range(2, -1, -1):
            fused[i] = adjusted[i] + self.interpolate(fused[i + 1], scale_factor=2, mode=self.mode)
        projected = [layer(x) for layer, x in zip(self.input_feature_projection_convolution, fused)]
        fused = list(projected)
        for i in range(1, 4):
            fused[i] = projected[i] + self.path_aggregation_head_convolution[i - 1](fused[i - 1])
        refined = [block(layer(x)) for block, layer, x in zip(self.intraclass_blocks, self.path_aggregation_lateral_convolution, fused)]
        # The pinned native forward executes these upsamples twice, discarding
        # its first list. Keep that otherwise redundant executed workload.
        for _ in range(2):
            upsampled = [self.interpolate(x, scale_factor=scale, mode=self.mode) if scale > 1 else x
                         for x, scale in zip(refined, self.scales)]
        return torch.cat(upsampled[::-1], dim=1)


class _Binarize(_Head):
    def forward(self, hidden):
        feature = self.conv_up(self.conv_down(hidden))
        return self.sigmoid(self.conv_final(feature)), feature


class _HeadRefine(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.neck_out_channels // 4
        self.binarize_head = _Binarize(config)
        self.local_refinement_module = nn.Module()
        self.local_refinement_module.convolution_backbone = _ConvNorm(width + 1, width, 3)
        self.local_refinement_module.convolution_final = Conv2d(width, 1, 1)
        self.interpolate, self.sigmoid = Interpolate(), Sigmoid()
        self.scale, self.mode = config.scale_factor, config.interpolate_mode

    def forward(self, hidden):
        initial, feature = self.binarize_head(hidden)
        feature = self.interpolate(feature, scale_factor=self.scale, mode=self.mode)
        local = self.local_refinement_module
        logits = local.convolution_final(local.convolution_backbone(torch.cat((initial, feature), dim=1)))
        return .5 * (initial + self.sigmoid(logits))


class _Detector(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = build_backbone(config.backbone_config, device, dtype)
        self.model.neck = _Neck(config)
        self.head = _HeadRefine(config)

    def forward(self, pixel_values):
        features = list(self.model.backbone(pixel_values).values())
        return {'last_hidden_state': self.head(self.model.neck(features))}


def build_from_config(config, device, dtype):
    if (list(config.kernel_list) != [3, 2, 2] or config.hidden_act != 'relu'
            or config.backbone_config.model_type != 'hgnet_v2' or config.intraclass_block_number != 4):
        raise ValueError('Preserve the native HGNet backbone and all four intraclass branches')
    model = _Detector(config, device, dtype)
    configure_transposes(model, device)
    return model.to(device=device, dtype=dtype).eval()
