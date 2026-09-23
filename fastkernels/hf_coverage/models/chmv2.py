"""CHMv2 canopy depth with default DINOv3 readouts and mixed depth bins."""

import torch
from torch import nn

from .depth_anything import NonoverlappingTransposeConv, _Fusion
from .dinov3_vit import build_from_config as build_backbone, load_state_dict_into as load_backbone
from .vit_msn import make_workloads
from ..patches.chmv2_normalization import PositiveFloor, PositiveL1Norm
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR


class _ReassembleLayer(nn.Module):
    def __init__(self, config, width, factor):
        super().__init__()
        self.projection = Conv2d(config.backbone_config.hidden_size, width, 1)
        self.resize = (NonoverlappingTransposeConv(width, width, int(factor)) if factor > 1
                       else Conv2d(width, width, 3, stride=int(1 / factor), padding=1) if factor < 1
                       else nn.Identity())

    def forward(self, hidden):
        return self.resize(self.projection(hidden))


class _Reassemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.backbone_config.hidden_size
        self.layers = nn.ModuleList([_ReassembleLayer(config, channels, factor)
                                    for channels, factor in zip(config.post_process_channels, config.reassemble_factors)])
        self.readout_projects = nn.ModuleList([nn.Sequential(Linear(2 * width, width), GELU()) for _ in self.layers])

    def forward(self, features):
        outputs = []
        for (feature, cls), layer, readout in zip(features, self.layers, self.readout_projects):
            batch, width, height, length = feature.shape
            hidden = feature.flatten(2).transpose(1, 2)
            hidden = readout(torch.cat((hidden, cls[:, None].expand_as(hidden)), dim=-1))
            outputs.append(layer(hidden.transpose(1, 2).reshape(batch, width, height, length)))
        return outputs


class _Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolate = Interpolate()

    def forward(self, hidden):
        return self.interpolate(hidden, scale_factor=2, mode="bilinear", align_corners=True)


class _ConvDepth(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.fusion_hidden_size
        self.head = nn.Sequential(Conv2d(width, width // 2, 3, padding=1), _Upsample(),
                                  Conv2d(width // 2, config.head_hidden_size, 3, padding=1), ReLU(),
                                  Conv2d(config.head_hidden_size, config.number_output_channels, 1))

    def forward(self, hidden):
        return self.head(hidden)


class _Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reassemble_stage = _Reassemble(config)
        self.convs = nn.ModuleList([Conv2d(width, config.fusion_hidden_size, 3, padding=1, bias=False)
                                   for width in config.post_process_channels])
        self.fusion_layers = nn.ModuleList([_Fusion(config.fusion_hidden_size) for _ in self.convs])
        # HF does not instantiate the unused first residual branch.
        del self.fusion_layers[0].residual_layer1
        self.conv_depth = _ConvDepth(config)

    def forward(self, features):
        features = [conv(feature) for conv, feature in zip(self.convs, self.reassemble_stage(features))][::-1]
        hidden = self.fusion_layers[0](features[0])
        for feature, layer in zip(features[1:], self.fusion_layers[1:]):
            hidden = layer(hidden, feature)
        return self.conv_depth(hidden)


class _FeaturesToDepth(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.minimum, self.maximum, self.count = config.min_depth, config.max_depth, config.number_output_channels
        self.relu, self.normalize, self.floor = ReLU(), PositiveL1Norm(), PositiveFloor()
        self.product, self.reduce = ProductGate(), SegmentCSR()

    def forward(self, logits):
        # Bins depend only on configuration metadata; preserve HF's FP32 construction.
        linear = torch.linspace(self.minimum, self.maximum / 8, self.count, device=logits.device)
        log = torch.exp(torch.linspace(torch.log(torch.tensor(self.minimum, device=logits.device)),
                                     torch.log(torch.tensor(self.maximum / 8, device=logits.device)),
                                     self.count, device=logits.device))
        fraction = torch.linspace(1, 0, self.count, device=logits.device)
        bins = self.floor(fraction * log + (1 - fraction) * linear)
        weights = self.normalize(self.relu(logits)).float().permute(0, 2, 3, 1)
        products = self.product(torch.cat((weights, bins.expand_as(weights)), dim=-1))
        batch, height, width, channels = products.shape
        offsets = torch.arange(0, products.numel() + 1, channels, device=logits.device)
        depth = self.reduce(products.reshape(-1), offsets, reduce="sum").reshape(batch, height, width)
        return self.floor(depth) * 8


class CHMv2ForDepthEstimation(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.backbone = build_backbone(config.backbone_config, device, dtype)
        self.out_indices = config.backbone_config.out_indices
        self.head = _Head(config).to(device=device, dtype=dtype)
        self.features_to_depth = _FeaturesToDepth(config)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        backbone, features = self.backbone, []
        batch, _, height, width = pixel_values.shape
        height, width = height // backbone.patch_size, width // backbone.patch_size
        hidden = backbone.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden = torch.cat((backbone.cls_token.expand(batch, -1, -1),
                            backbone.reg_token.expand(batch, -1, -1), hidden), dim=1)
        rope = backbone.rope.get_embed([height, width]).to(pixel_values.dtype)
        for index, block in enumerate(backbone.blocks, start=1):
            hidden = block(hidden, rope=rope)
            if index in self.out_indices:
                normalized = backbone.norm(hidden)
                feature = normalized[:, backbone.num_prefix_tokens:].transpose(1, 2).reshape(batch, -1, height, width)
                features.append((feature, normalized[:, 0]))
        return {"predicted_depth": self.features_to_depth(self.head(features))}


def build_from_config(config, device, dtype):
    if config.readout_type != "project" or config.norm_strategy != "chmv2_mixlog" or config.bins_strategy != "chmv2_mixlog":
        raise ValueError("This case retains the default project readout and mixed-log bin normalization")
    return CHMv2ForDepthEstimation(config, device, dtype).eval()


def load_state_dict_into(model, state_dict, config):
    load_backbone(model.backbone, {k.removeprefix("backbone."): v for k, v in state_dict.items()
                                   if k.startswith("backbone.")}, config.backbone_config)
    mapped = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
    for index, layer in enumerate(model.head.reassemble_stage.layers):
        if isinstance(layer.resize, NonoverlappingTransposeConv):
            prefix = f"head.reassemble_stage.layers.{index}.resize."
            weight = mapped.pop(prefix + "weight")
            mapped[prefix + "projection.weight"] = weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0]).contiguous()
            mapped[prefix + "projection.bias"] = mapped.pop(prefix + "bias").repeat_interleave(layer.resize.factor ** 2)
    mapped.update({"backbone." + k: v for k, v in model.backbone.state_dict().items()})
    model.load_state_dict(mapped, strict=True)
