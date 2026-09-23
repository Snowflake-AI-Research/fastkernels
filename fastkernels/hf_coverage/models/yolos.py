"""YOLOS tiny's default image encoder and both object-detection heads."""

import torch
from torch import nn

from .vit import _encoder_block
from .vit_msn import load_state_dict_into as load_encoder, make_workloads
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead


class _PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)

    def forward(self, pixels):
        return self.proj(pixels).flatten(2).transpose(1, 2)


class _Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.grid = tuple(size // config.patch_size for size in config.image_size)
        self.patch_embeddings = _PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.detection_tokens = nn.Parameter(torch.empty(1, config.num_detection_tokens, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(
            1, self.grid[0] * self.grid[1] + config.num_detection_tokens + 1, config.hidden_size))
        self.interpolate = Interpolate()

    def forward(self, pixels):
        batch, _, height, width = pixels.shape
        hidden = torch.cat((self.cls_token.expand(batch, -1, -1), self.patch_embeddings(pixels),
                            self.detection_tokens.expand(batch, -1, -1)), dim=1)
        count = self.detection_tokens.shape[1]
        positions = self.position_embeddings
        patch_positions = positions[:, 1:-count].transpose(1, 2).reshape(1, -1, *self.grid)
        patch_positions = self.interpolate(patch_positions, size=(height // self.patch_size, width // self.patch_size),
                                           mode="bicubic", align_corners=False).flatten(2).transpose(1, 2)
        return hidden + torch.cat((positions[:, :1], patch_positions, positions[:, -count:]), dim=1)


class _VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = _Embeddings(config)
        self.encoder = nn.ModuleList([_encoder_block(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixels):
        hidden = self.embeddings(pixels)
        for layer in self.encoder:
            hidden = layer(hidden)
        return self.layernorm(hidden)


class YolosForObjectDetection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = _VisionEncoder(config)
        self.num_detection_tokens = config.num_detection_tokens
        self.class_labels_classifier = RTDetrV2MLPPredictionHead(
            config, config.hidden_size, config.hidden_size, len(config.id2label) + 1, 3)
        self.bbox_predictor = RTDetrV2MLPPredictionHead(config, config.hidden_size, config.hidden_size, 4, 3)
        self.sigmoid = Sigmoid()

    def forward(self, pixel_values):
        hidden = self.vit(pixel_values)
        objects = hidden[:, -self.num_detection_tokens:]
        return {"logits": self.class_labels_classifier(objects),
                "pred_boxes": self.sigmoid(self.bbox_predictor(objects)), "last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.use_mid_position_embeddings or config.hidden_act != "gelu":
        raise ValueError("The documented YOLOS tiny checkpoint uses GELU without intermediate position embeddings")
    return YolosForObjectDetection(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    state = {key[4:]: remaining.pop(key) for key in list(remaining) if key.startswith("vit.")}
    load_encoder(model.vit, state, config)
    mapped = {"vit." + key: value for key, value in model.vit.state_dict().items()}
    model.load_state_dict({**mapped, **remaining}, strict=True)
