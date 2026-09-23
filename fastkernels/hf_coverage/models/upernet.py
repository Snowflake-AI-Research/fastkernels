"""UperNet's ConvNeXt backbone, pyramid decoder, and default auxiliary head."""

import torch
from torch import nn

from .convnext import ConvNextModel, load_state_dict_into as load_backbone
from .vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.relu import ReLU


def _resize(hidden, size):
    return Interpolate()(hidden, size=size, mode="bilinear", align_corners=False)


class _Backbone(ConvNextModel):
    def __init__(self, config):
        super().__init__(config)
        del self.pooler, self.layernorm
        self.hidden_states_norms = nn.ModuleDict({
            f"stage{i + 1}": LayerNorm2d(width, eps=1e-6) for i, width in enumerate(config.hidden_sizes)
        })

    def forward(self, pixels):
        hidden, features = self.embeddings(pixels), []
        for index, stage in enumerate(self.encoder.stages, start=1):
            hidden = stage(hidden)
            features.append(self.hidden_states_norms[f"stage{index}"](hidden))
        return features


class _ConvModule(nn.Module):
    def __init__(self, source, target, kernel, padding=0):
        super().__init__()
        self.conv = Conv2d(source, target, kernel, padding=padding, bias=False)
        self.batch_norm, self.activation = BatchNorm2d(target), ReLU()

    def forward(self, hidden):
        return self.activation(self.batch_norm(self.conv(hidden)))


class _AdaptivePool(nn.Module):
    """Each native adaptive bin is one existing average-pooling operation.

    Bin bounds depend only on image dimensions. Bins may overlap exactly as in
    HF; there is no added recurrence or padded expansion of the input tensor.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, hidden):
        height, width = hidden.shape[-2:]
        rows = []
        for row in range(self.size):
            top = row * height // self.size
            bottom = ((row + 1) * height + self.size - 1) // self.size
            columns = []
            for col in range(self.size):
                left = col * width // self.size
                right = ((col + 1) * width + self.size - 1) // self.size
                columns.append(AvgPool2d((bottom - top, right - left))(hidden[:, :, top:bottom, left:right]))
            rows.append(torch.cat(columns, dim=-1))
        return torch.cat(rows, dim=-2)


class _DecodeHead(nn.Module):
    def __init__(self, config, widths):
        super().__init__()
        width = config.hidden_size
        self.classifier = Conv2d(width, len(config.id2label), 1)
        self.psp_modules = nn.ModuleList([
            nn.Sequential(_AdaptivePool(scale), _ConvModule(widths[-1], width, 1))
            for scale in config.pool_scales
        ])
        self.bottleneck = _ConvModule(widths[-1] + len(config.pool_scales) * width, width, 3, padding=1)
        self.lateral_convs = nn.ModuleList([_ConvModule(source, width, 1) for source in widths[:-1]])
        self.fpn_convs = nn.ModuleList([_ConvModule(width, width, 3, padding=1) for _ in widths[:-1]])
        self.fpn_bottleneck = _ConvModule(len(widths) * width, width, 3, padding=1)

    def forward(self, features):
        laterals = [layer(feature) for layer, feature in zip(self.lateral_convs, features[:-1])]
        pooled = [features[-1], *[_resize(block(features[-1]), features[-1].shape[-2:])
                                 for block in self.psp_modules]]
        laterals.append(self.bottleneck(torch.cat(pooled, dim=1)))
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + _resize(laterals[i], laterals[i - 1].shape[-2:])
        outputs = [layer(feature) for layer, feature in zip(self.fpn_convs, laterals[:-1])]
        outputs.append(laterals[-1])
        for i in range(len(outputs) - 1, 0, -1):
            outputs[i] = _resize(outputs[i], outputs[0].shape[-2:])
        return self.classifier(self.fpn_bottleneck(torch.cat(outputs, dim=1)))


class _AuxiliaryHead(nn.Module):
    def __init__(self, config, widths):
        super().__init__()
        source = widths[2] if config.auxiliary_in_channels is None else config.auxiliary_in_channels
        width = config.auxiliary_channels
        self.convs = nn.Sequential(*[
            _ConvModule(source if i == 0 else width, width, 3, padding=1)
            for i in range(config.auxiliary_num_convs)
        ])
        if config.auxiliary_concat_input:
            self.conv_cat = _ConvModule(source + width, width, 3, padding=1)
        self.classifier = Conv2d(width, len(config.id2label), 1)

    def forward(self, features):
        hidden = self.convs(features[2])
        if hasattr(self, "conv_cat"):
            hidden = self.conv_cat(torch.cat((features[2], hidden), dim=1))
        return self.classifier(hidden)


class UperNetForSemanticSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = _Backbone(config.backbone_config)
        widths = config.backbone_config.hidden_sizes
        self.decode_head = _DecodeHead(config, widths)
        self.auxiliary_head = _AuxiliaryHead(config, widths) if config.use_auxiliary_head else None

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        logits = _resize(self.decode_head(features), pixel_values.shape[-2:])
        if self.auxiliary_head is not None:
            # HF computes and resizes this head even without labels, then omits
            # it from public outputs. Eager execution retains its actual cost.
            _resize(self.auxiliary_head(features), pixel_values.shape[-2:])
        return {"logits": logits}


def build_from_config(config, device, dtype):
    backbone = config.backbone_config
    if backbone.model_type != "convnext" or backbone.out_features != [f"stage{i}" for i in range(1, 5)]:
        raise ValueError("The documented UperNet checkpoint returns all four ConvNeXt stages")
    return UperNetForSemanticSegmentation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    state = {key[9:]: remaining.pop(key) for key in list(remaining) if key.startswith("backbone.")}
    load_backbone(model.backbone, state, config.backbone_config)
    mapped = {"backbone." + key: value for key, value in model.backbone.state_dict().items()}
    model.load_state_dict({**mapped, **remaining}, strict=True)
