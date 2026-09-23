"""OneVision's SigLIP features, bounded any-resolution packing and Qwen2."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionEncoder
from . import llava_next, llava_next_video, qwen2
from .siglip import configure_encoder


class OnevisionBackbone(llava_next_video.NextVideoBackbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text = text
        self.vision = SigLIPVisionEncoder(config.vision_config)
        configure_encoder(self.vision.layers)
        self.vision.post_layernorm.promote_fp32 = False
        self.linear_1 = Linear(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        self.activation = GELU()
        self.resize = Interpolate()
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.crop_size = config.vision_config.image_size
        self.patch_grid = self.crop_size // config.vision_config.patch_size
        self.grid_pinpoints = config.image_grid_pinpoints
        self.max_patches = int(config.vision_aspect_ratio.removeprefix("anyres_max_"))
        self.image_token_id, self.video_token_id = config.image_token_index, config.video_token_index
        self.pixel_values = self.pixel_values_videos = self.image_sizes = None
        self.image_hidden_states = self.video_hidden_states = None

    def projected_features(self, pixels):
        hidden = self.vision.patch_embedding(pixels).flatten(2).transpose(1, 2)
        hidden = hidden + self.vision.position_embedding
        for layer in self.vision.layers:
            hidden = layer(hidden)
        # Selection -1 uses the final encoder state before the tower's final norm.
        self.vision.post_layernorm(hidden)
        return self.linear_2(self.activation(self.linear_1(hidden)))

    def image_features(self, pixels, sizes):
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.tolist()
        grids = [llava_next.crop_grid(size, self.grid_pinpoints, self.crop_size) for size in sizes]
        counts = [height * width + 1 for height, width in grids]
        features = self.projected_features(torch.cat([image[:count] for image, count in zip(pixels, counts)]))
        outputs = []
        for feature, (height, width), size in zip(features.split(counts), grids, sizes):
            base, tiles = feature[0], feature[1:]
            tiles = tiles.view(height, width, self.patch_grid, self.patch_grid, -1)
            tiles = tiles.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
            tiles = llava_next.unpad_features(tiles, size)
            height, width = tiles.shape[1:]
            ratio = math.sqrt(height * width / (self.max_patches * self.patch_grid**2))
            if ratio > 1.1:
                tiles = self.resize(tiles[None], size=(int(height // ratio), int(width // ratio)),
                                    mode="bilinear", align_corners=False)[0]
            newline = self.image_newline[:, None, None].expand(tiles.shape[0], tiles.shape[1], 1)
            tiles = torch.cat((tiles, newline), dim=-1).flatten(1, 2).transpose(0, 1)
            outputs.append(torch.cat((base, tiles)))
        return torch.cat(outputs)

    def video_features(self, pixels):
        batch, frames = pixels.shape[:2]
        projected = self.projected_features(pixels.flatten(0, 1))
        spatial = projected.view(batch * frames, self.patch_grid, self.patch_grid, -1).permute(0, 3, 1, 2).contiguous()
        size = (math.ceil(self.patch_grid / 2),) * 2
        pooled = self.resize(spatial, size=size, mode="bilinear", align_corners=False)
        pooled = pooled.permute(0, 2, 3, 1).reshape(batch, -1, projected.shape[-1])
        return torch.cat((pooled, self.image_newline[None, None].expand(batch, 1, -1)), dim=1).flatten(0, 1)


class OnevisionModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = OnevisionBackbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    if (config.vision_feature_layer != -1 or config.vision_feature_select_strategy != "full"
            or config.vision_aspect_ratio != "anyres_max_9" or config.vision_config.vision_use_head
            or config.vision_config.hidden_act != "gelu_pytorch_tanh" or config.projector_hidden_act != "gelu"):
        raise ValueError("The documented OneVision checkpoint uses final SigLIP encoder features and anyres_max_9")
    return OnevisionModel(qwen2.build_from_config(config.text_config, device, dtype), config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name.replace("layers.", "encoder.layers.")
        source = source.replace("patch_embedding.", "embeddings.patch_embedding.")
        if name == "position_embedding":
            source = "embeddings.position_embedding.weight"
        value = remaining.pop("model.vision_tower." + source)
        mapped[name] = value[None] if name == "position_embedding" else value
    model.model.vision.load_state_dict(mapped, strict=True)
    model.model.image_newline.copy_(remaining.pop("model.image_newline"))
    for name in ("linear_1", "linear_2"):
        getattr(model.model, name).load_state_dict({field: remaining.pop(f"model.multi_modal_projector.{name}.{field}")
                                                  for field in ("weight", "bias")})
    if remaining:
        raise KeyError(f"Unmapped OneVision state: {sorted(remaining)}")


make_workloads = llava_next_video.make_workloads
