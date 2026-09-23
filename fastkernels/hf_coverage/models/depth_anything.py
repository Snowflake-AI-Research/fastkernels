"""Depth Anything's DINOv2 backbone and complete four-resolution depth head."""

import torch
from torch import nn

from .dinov2 import Dinov2Model, _Dinov2Embeddings, load_state_dict_into as load_backbone
from .vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid


class DepthEmbeddings(_Dinov2Embeddings):
    """Keep the learned DINOv2 grid while accepting processor-sized images."""

    def __init__(self, config):
        super().__init__(config)
        self.resize = Interpolate()

    def forward(self, pixel_values):
        height, width = pixel_values.shape[-2:]
        pixel_values = pixel_values.to(self.patch_embeddings.proj.weight.dtype)
        patches = self.patch_embeddings(pixel_values, random_sample=True)
        hidden = torch.cat((self.cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)
        positions = self.position_embeddings
        if patches.shape[1] != positions.shape[1] - 1 or height != width:
            grid_h, grid_w = self.patch_embeddings.grid_size
            patch_h, patch_w = self.patch_embeddings.patch_size
            spatial = positions[:, 1:].reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
            spatial = self.resize(spatial.float(), size=(height // patch_h, width // patch_w),
                                  mode="bicubic", align_corners=False).to(positions.dtype)
            positions = torch.cat((positions[:, :1], spatial.flatten(2).transpose(1, 2)), dim=1)
        return hidden + positions


class NonoverlappingTransposeConv(nn.Module):
    """Kernel=stride, padding=0: one linear projection per pixel, then tile layout.

    Each projected value is one required output pixel. There is no zero insertion,
    overlapping accumulation, or expansion beyond the actual output tensor.
    """

    def __init__(self, input_width, output_width, factor, bias=True):
        super().__init__()
        self.factor, self.output_width = factor, output_width
        self.projection = Linear(input_width, output_width * factor * factor, bias=bias)

    def forward(self, hidden):
        batch, _, height, width = hidden.shape
        factor = self.factor
        hidden = self.projection(hidden.permute(0, 2, 3, 1))
        hidden = hidden.reshape(batch, height, width, self.output_width, factor, factor)
        return hidden.permute(0, 3, 1, 4, 2, 5).reshape(batch, self.output_width, height * factor, width * factor)


class _ReassembleLayer(nn.Module):
    def __init__(self, config, width, factor):
        super().__init__()
        self.projection = Conv2d(config.reassemble_hidden_size, width, 1)
        if factor > 1:
            self.resize = NonoverlappingTransposeConv(width, width, int(factor))
        elif factor < 1:
            self.resize = Conv2d(width, width, 3, stride=int(1 / factor), padding=1)
        else:
            self.resize = nn.Identity()

    def forward(self, hidden):
        return self.resize(self.projection(hidden))


class _Reassemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([_ReassembleLayer(config, width, factor)
                                    for width, factor in zip(config.neck_hidden_sizes, config.reassemble_factors)])

    def forward(self, features, height, width):
        return [layer(hidden[:, 1:].transpose(1, 2).reshape(hidden.shape[0], -1, height, width))
                for layer, hidden in zip(self.layers, features)]


class _Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.activation1, self.activation2 = ReLU(), ReLU()
        self.convolution1 = Conv2d(width, width, 3, padding=1)
        self.convolution2 = Conv2d(width, width, 3, padding=1)

    def forward(self, hidden):
        return hidden + self.convolution2(self.activation2(self.convolution1(self.activation1(hidden))))


class _Fusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = Conv2d(width, width, 1)
        self.residual_layer1, self.residual_layer2 = _Residual(width), _Residual(width)
        self.interpolate = Interpolate()

    def forward(self, hidden, residual=None, size=None):
        if residual is not None:
            if residual.shape != hidden.shape:
                residual = self.interpolate(residual, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
            hidden = hidden + self.residual_layer1(residual)
        hidden = self.residual_layer2(hidden)
        return self.projection(self.interpolate(hidden, size=size, scale_factor=2 if size is None else None,
                                               mode="bilinear", align_corners=True))


class _FusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([_Fusion(config.fusion_hidden_size) for _ in config.neck_hidden_sizes])

    def forward(self, features):
        features, outputs, hidden = features[::-1], [], None
        for index, (feature, layer) in enumerate(zip(features, self.layers)):
            size = features[index + 1].shape[-2:] if index + 1 < len(features) else None
            hidden = layer(feature, size=size) if hidden is None else layer(hidden, feature, size=size)
            outputs.append(hidden)
        return outputs


class _Neck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reassemble_stage = _Reassemble(config)
        self.convs = nn.ModuleList([Conv2d(width, config.fusion_hidden_size, 3, padding=1, bias=False)
                                   for width in config.neck_hidden_sizes])
        self.fusion_stage = _FusionStage(config)

    def forward(self, features, height, width):
        features = self.reassemble_stage(features, height, width)
        return self.fusion_stage([conv(feature) for conv, feature in zip(self.convs, features)])


class _Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.index, self.patch_size, self.max_depth = config.head_in_index, config.patch_size, config.max_depth
        width = config.fusion_hidden_size
        self.conv1 = Conv2d(width, width // 2, 3, padding=1)
        self.conv2 = Conv2d(width // 2, config.head_hidden_size, 3, padding=1)
        self.conv3 = Conv2d(config.head_hidden_size, 1, 1)
        self.activation1 = ReLU()
        self.activation2 = ReLU() if config.depth_estimation_type == "relative" else Sigmoid()
        self.interpolate = Interpolate()

    def forward(self, features, height, width):
        hidden = self.conv1(features[self.index])
        hidden = self.interpolate(hidden, size=(height * self.patch_size, width * self.patch_size),
                                  mode="bilinear", align_corners=True)
        return (self.activation2(self.conv3(self.activation1(self.conv2(hidden)))) * self.max_depth).squeeze(1)


class DepthAnythingForDepthEstimation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Dinov2Model(config.backbone_config)
        self.backbone.embeddings = DepthEmbeddings(config.backbone_config)
        self.out_indices = config.backbone_config.out_indices
        self.apply_layernorm = config.backbone_config.apply_layernorm
        self.patch_size = config.patch_size
        self.neck, self.head = _Neck(config), _Head(config)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        hidden = self.backbone.embeddings(pixel_values)
        features = []
        for index, layer in enumerate(self.backbone.encoder, start=1):
            hidden = layer(hidden)
            if index in self.out_indices:
                features.append(self.backbone.layernorm(hidden) if self.apply_layernorm else hidden)
        height, width = (size // self.patch_size for size in pixel_values.shape[-2:])
        return {"predicted_depth": self.head(self.neck(features, height, width), height, width)}


def build_from_config(config, device, dtype):
    if config.backbone_config.model_type != "dinov2" or config.backbone_config.reshape_hidden_states:
        raise ValueError("The selected checkpoint uses unreshaped DINOv2 backbone features")
    if config.depth_estimation_type not in ("relative", "metric") or 0 in config.backbone_config.out_indices:
        raise ValueError("The selected checkpoint uses four encoder outputs and relative or metric depth")
    return DepthAnythingForDepthEstimation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    backbone = {name.removeprefix("backbone."): value for name, value in state_dict.items() if name.startswith("backbone.")}
    load_backbone(model.backbone, backbone, config.backbone_config)
    mapped = {name: value for name, value in state_dict.items() if not name.startswith("backbone.")}
    for index, layer in enumerate(model.neck.reassemble_stage.layers):
        if isinstance(layer.resize, NonoverlappingTransposeConv):
            prefix = f"neck.reassemble_stage.layers.{index}.resize."
            weight = mapped.pop(prefix + "weight")
            mapped[prefix + "projection.weight"] = weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0]).contiguous()
            mapped[prefix + "projection.bias"] = mapped.pop(prefix + "bias").repeat_interleave(layer.resize.factor ** 2)
    mapped.update({"backbone." + name: value for name, value in model.backbone.state_dict().items()})
    model.load_state_dict(mapped, strict=True)
