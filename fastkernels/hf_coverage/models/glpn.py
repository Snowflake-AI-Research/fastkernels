"""GLPN depth estimation with all four encoder and selective-fusion stages."""

import torch
from torch import nn

from .segformer import SpatialReductionEncoder, check_encoder_config, encoder_state_name
from .vit_msn import make_workloads
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid


class _SelectiveFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.convolutional_layer1 = nn.Sequential(
            Conv2d(2 * width, width, 3, padding=1), BatchNorm2d(width), ReLU())
        self.convolutional_layer2 = nn.Sequential(
            Conv2d(width, width // 2, 3, padding=1), BatchNorm2d(width // 2), ReLU())
        self.convolutional_layer3 = Conv2d(width // 2, 2, 3, padding=1)
        self.sigmoid = Sigmoid()
        self.product = ProductGate()

    def forward(self, local, global_features):
        features = self.convolutional_layer1(torch.cat((local, global_features), dim=1))
        gates = self.sigmoid(self.convolutional_layer3(self.convolutional_layer2(features)))
        local = self.product(torch.cat((local, gates[:, :1].expand_as(local)), dim=-1))
        global_features = self.product(torch.cat((global_features, gates[:, 1:].expand_as(global_features)), dim=-1))
        return local + global_features


class _DecoderStage(nn.Module):
    def __init__(self, input_width, width, first):
        super().__init__()
        self.convolution = Conv2d(input_width, width, 1) if input_width != width else nn.Identity()
        self.fusion = None if first else _SelectiveFusion(width)
        self.upsample = Interpolate()

    def forward(self, hidden, residual):
        hidden = self.convolution(hidden)
        if residual is not None:
            hidden = self.fusion(hidden, residual)
        return self.upsample(hidden, scale_factor=2, mode="bilinear", align_corners=False)


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList([
            _DecoderStage(width, config.decoder_hidden_size, index == 0)
            for index, width in enumerate(config.hidden_sizes[::-1])])
        self.final_upsample = Interpolate()

    def forward(self, features):
        outputs, hidden = [], None
        for feature, stage in zip(features[::-1], self.stages):
            hidden = stage(feature, hidden)
            outputs.append(hidden)
        outputs[-1] = self.final_upsample(hidden, scale_factor=2, mode="bilinear", align_corners=False)
        return outputs


class _DepthHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.index, self.max_depth = config.head_in_index, config.max_depth
        width = config.decoder_hidden_size
        self.head = nn.Sequential(Conv2d(width, width, 3, padding=1), ReLU(), Conv2d(width, 1, 3, padding=1))
        self.sigmoid = Sigmoid()

    def forward(self, features):
        return (self.sigmoid(self.head(features[self.index])) * self.max_depth).squeeze(1)


class GLPNForDepthEstimation(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pinned HF uses nn.LayerNorm's epsilon throughout the GLPN encoder.
        self.encoder = SpatialReductionEncoder(config, norm_eps=1e-5)
        self.decoder = _Decoder(config)
        self.head = _DepthHead(config)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        return {"predicted_depth": self.head(self.decoder(self.encoder(pixel_values, all_stages=True)))}


def build_from_config(config, device, dtype):
    check_encoder_config(config)
    return GLPNForDepthEstimation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for name, value in state_dict.items():
        if name.startswith("glpn.encoder."):
            name = name.removeprefix("glpn.encoder.")
            for index in range(config.num_encoder_blocks):
                name = name.replace(f"patch_embeddings.{index}.proj.", f"encoder.stages.{index}.projection.")
                name = name.replace(f"patch_embeddings.{index}.layer_norm.", f"encoder.stages.{index}.embedding_norm.")
                name = name.replace(f"layer_norm.{index}.", f"encoder.stages.{index}.norm.")
                name = name.replace(f"block.{index}.", f"encoder.stages.{index}.blocks.")
            name = name.replace(".attention.self.", ".attention.").replace(".attention.output.dense.", ".attention.proj.")
            name = encoder_state_name(name)
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
