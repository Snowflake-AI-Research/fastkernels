"""VidEoMT video segmentation with per-frame queries propagated through Linear."""

import torch
from torch import nn

from fastkernels.hf_coverage.models import eomt
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.sam3_fpn_conv import Sam3FPNConvStage


class Videomt(eomt._Eomt):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings.mask_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.query_updater = Linear(config.hidden_size, config.hidden_size)
        for block in self.upscale_block.block:
            # Reuse the actual transpose-convolution child of the existing FPN
            # stage. Its forward is unchanged; configure output channels and
            # parameter shapes for VidEoMT's equal-width upsampling layer.
            # This child has no standalone benchmark optimization interface.
            conv = Sam3FPNConvStage(config.hidden_size, config.hidden_size, 2.0).conv[0]
            conv.out_channels = config.hidden_size
            conv.weight = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size, 2, 2))
            conv.bias = nn.Parameter(torch.empty(config.hidden_size))
            block.conv1 = conv

    def forward(self, pixel_values_videos):
        batch, frames, channels, height, width = pixel_values_videos.shape
        hidden = self.embeddings(pixel_values_videos.reshape(batch * frames, channels, height, width))
        split = self.config.num_hidden_layers - self.config.num_blocks
        for layer in self.layers[:split]:
            hidden = layer(hidden)
        hidden = hidden.reshape(batch, frames, *hidden.shape[1:])
        masks, classes, states = [], [], []
        previous = None
        for index in range(frames):
            queries = self.query.emb.weight[None].expand(batch, -1, -1)
            if previous is not None:
                queries = self.query_updater(previous) + queries
            frame = torch.cat((queries, hidden[:, index]), dim=1)
            for layer in self.layers[split:]:
                frame = layer(frame)
            normalized = self.layernorm(frame)
            frame_masks, frame_classes = self.predict(normalized)
            masks.append(frame_masks)
            classes.append(frame_classes)
            states.append(normalized)
            # Native propagates the unnormalized final query features.
            previous = frame[:, :self.config.num_queries]
        return {"masks_queries_logits": torch.cat(masks),
                "class_queries_logits": torch.cat(classes),
                "last_hidden_state": torch.cat(states)}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.use_swiglu_ffn:
        raise ValueError("The declared VidEoMT checkpoint uses the exact-GELU MLP")
    return Videomt(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for key, value in state_dict.items():
        key = key.replace("query.weight", "query.emb.weight")
        key = key.replace("position_embeddings.weight", "position_embeddings.emb.weight")
        key = key.replace(".attention.out_proj.", ".attention.proj.")
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
