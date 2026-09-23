"""MaskFormer's Swin backbone, feature pyramid and complete query decoder."""

import math
import re

import torch
from torch import nn

from .detr import _Decoder, make_workloads
from .mask2former import _SwinBackbone, _load_swin_backbone
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR


class _Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = _SwinBackbone(config)
        self.layernorm = LayerNorm(config.embed_dim * 8, eps=config.layer_norm_eps, promote_fp32=False)
        self.pool = SegmentCSR()

    def forward(self, pixels):
        backbone = self.features
        hidden, features = backbone.embed_pixels(pixels), []
        for index, stage in enumerate(backbone.stages, start=1):
            for block in stage.blocks:
                hidden = block(hidden)
            features.append(backbone.hidden_states_norms[f"stage{index}"](hidden).permute(0, 3, 1, 2).contiguous())
            hidden = backbone.downsample(hidden, stage)
        # MaskFormerSwinModel computes these even though its backbone caller
        # consumes the earlier stage features instead of the pooled result.
        normalized = self.layernorm(hidden).flatten(1, 2).transpose(1, 2).contiguous()
        offsets = torch.arange(normalized.shape[0] * normalized.shape[1] + 1,
                               device=hidden.device) * normalized.shape[-1]
        self.pool(normalized.flatten(), offsets, reduce="mean")
        return features


def _conv(source, target):
    return nn.Sequential(Conv2d(source, target, 3, padding=1, bias=False), GroupNorm(32, target, eps=1e-5), ReLU())


class _FPNLayer(nn.Module):
    def __init__(self, width, source):
        super().__init__()
        self.proj = nn.Sequential(Conv2d(source, width, 1, bias=False), GroupNorm(32, width, eps=1e-5))
        self.block, self.interpolate = _conv(width, width), Interpolate()

    def forward(self, hidden, feature):
        feature = self.proj(feature)
        return self.block(self.interpolate(hidden, size=feature.shape[-2:], mode="nearest") + feature)


class _PixelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = [config.backbone_config.embed_dim * 2**i for i in range(4)]
        self.fpn = nn.Module()
        self.fpn.stem = _conv(channels[-1], config.fpn_feature_size)
        self.fpn.layers = nn.ModuleList([_FPNLayer(config.fpn_feature_size, source) for source in channels[-2::-1]])
        self.mask_projection = Conv2d(config.fpn_feature_size, config.mask_feature_size, 3, padding=1)

    def forward(self, features):
        hidden = self.fpn.stem(features[-1])
        for layer, feature in zip(self.fpn.layers, features[-2::-1]):
            hidden = layer(hidden, feature)
        return self.mask_projection(hidden)


def _positions(feature, width):
    batch, _, height, columns = feature.shape
    valid = torch.ones(batch, height, columns, device=feature.device, dtype=feature.dtype)
    y, x = valid.cumsum(1), valid.cumsum(2)
    y, x = y / (y[:, -1:] + 1e-6) * (2 * math.pi), x / (x[:, :, -1:] + 1e-6) * (2 * math.pi)
    dim = torch.arange(width // 2, device=feature.device).to(feature.dtype)
    frequency = 10000 ** (2 * torch.div(dim, 2, rounding_mode="floor") / (width // 2))
    x, y = x[..., None] / frequency, y[..., None] / frequency
    x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(3)
    y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(3)
    return torch.cat((y, x), dim=-1).flatten(1, 2)


class _Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.decoder_config.d_model
        self.input_projection = Conv2d(config.backbone_config.embed_dim * 8, width, 1)
        self.queries_embedder = Embedding(config.decoder_config.num_queries, width)
        self.decoder = _Decoder(config.decoder_config)

    def forward(self, feature):
        feature = self.input_projection(feature)
        positions = _positions(feature, feature.shape[1])
        memory = feature.flatten(2).transpose(1, 2)
        query_positions = self.queries_embedder.emb.weight[None].expand(feature.shape[0], -1, -1)
        return self.decoder(torch.zeros_like(query_positions), query_positions, memory, positions, None)


class _MaskFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.pixel_level_module = nn.Module()
        self.model.pixel_level_module.encoder = _Backbone(config.backbone_config)
        self.model.pixel_level_module.decoder = _PixelDecoder(config)
        self.model.transformer_module = _Transformer(config)
        width = config.decoder_config.d_model
        self.class_predictor = Linear(width, len(config.id2label) + 1)
        self.mask_embedder = nn.Sequential(nn.Sequential(Linear(width, width), ReLU()),
                                           nn.Sequential(Linear(width, width), ReLU()),
                                           nn.Sequential(Linear(width, config.mask_feature_size), nn.Identity()))
        self.criterion = nn.Module()
        self.criterion.register_buffer("empty_weight", torch.empty(len(config.id2label) + 1))
        self.bmm = BMM()

    def forward(self, pixel_values, pixel_mask=None):
        features = self.model.pixel_level_module.encoder(pixel_values)
        pixels = self.model.pixel_level_module.decoder(features)
        hidden = self.model.transformer_module(features[-1])
        masks = self.bmm(self.mask_embedder(hidden), pixels.flatten(2)).reshape(
            hidden.shape[0], hidden.shape[1], *pixels.shape[-2:])
        return {"class_queries_logits": self.class_predictor(hidden), "masks_queries_logits": masks,
                "encoder_last_hidden_state": features[-1], "pixel_decoder_last_hidden_state": pixels,
                "transformer_decoder_last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.use_auxiliary_loss or config.decoder_config.auxiliary_loss:
        raise ValueError("The published ADE MaskFormer checkpoint disables auxiliary prediction")
    if config.backbone_config.model_type not in ("swin", "maskformer-swin"):
        raise ValueError("This case preserves the published MaskFormer Swin backbone")
    return _MaskFormer(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    prefix = "model.pixel_level_module.encoder."
    state, mapped = {}, {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            name = key[len(prefix):]
            if name.startswith("model.layernorm."):
                mapped[prefix + name[len("model."):]] = value
            elif name.startswith("model."):
                state[name[len("model."):]] = value
            else:
                name = re.sub(r"hidden_states_norms\.(\d+)\.", lambda match: f"hidden_states_norms.stage{int(match[1]) + 1}.", name)
                state[name] = value
        else:
            key = key.replace("queries_embedder.weight", "queries_embedder.emb.weight")
            key = key.replace(".self_attn.o_proj.", ".self_attn.out_proj.").replace(".encoder_attn.o_proj.", ".encoder_attn.out_proj.")
            mapped[key] = value
    backbone = model.model.pixel_level_module.encoder.features
    _load_swin_backbone(backbone, state, config.backbone_config)
    mapped.update({prefix + "features." + key: value for key, value in backbone.state_dict().items()})
    model.load_state_dict(mapped, strict=True)
