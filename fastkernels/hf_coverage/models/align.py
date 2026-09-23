"""ALIGN's BERT/EfficientNet towers with their original poolers and temperature."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L3.bert_model import BertModel
from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModelOutput

from .clip import make_workloads
from .efficientnet import _PaddedConv, _batch_norm, _block, _round_filters
from .efficientnet import load_state_dict_into as load_vision_state
from .layoutlm import LayoutAttention, Pooler


class AlignTextModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        # Native ALIGN stores low-precision scores before its FP32 softmax.
        for layer in self.encoder.layer:
            layer.attention = LayoutAttention(config)
        self.pooler = Pooler(config)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = super().forward(input_ids, positions.unsqueeze(0).expand_as(input_ids))
        return CLIPTextModelOutput(hidden, self.pooler(hidden))


class AlignVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        stem_width = _round_filters(config, 32)
        self.stem = _PaddedConv(config.num_channels, stem_width, 3, 2)
        self.stem_bn = _batch_norm(config, stem_width, activation=True)
        self.blocks = nn.ModuleList()
        for stage, repeats in enumerate(config.num_block_repeats):
            in_channels = _round_filters(config, config.in_channels[stage])
            out_channels = _round_filters(config, config.out_channels[stage])
            for index in range(math.ceil(repeats * config.depth_coefficient)):
                self.blocks.append(_block(
                    config, in_channels if index == 0 else out_channels, out_channels,
                    config.kernel_sizes[stage], config.strides[stage] if index == 0 else 1,
                    config.expand_ratios[stage], index == 0, len(self.blocks),
                ))
        # ALIGN ends at the MBConv blocks; it has no EfficientNet top projection.
        self.pooler = AvgPool2d(config.hidden_dim, ceil_mode=True)

    def forward(self, pixel_values):
        hidden = self.stem_bn(self.stem(pixel_values))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden, self.pooler(hidden).reshape(hidden.shape[:2])


class AlignModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AlignTextModel(config.text_config)
        self.vision_model = AlignVisionModel(config.vision_config)
        self.text_projection = Linear(config.text_config.hidden_size, config.projection_dim)
        self.temperature = nn.Parameter(torch.empty(()))
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values):
        vision_hidden, vision_pool = self.vision_model(pixel_values)
        text_output = self.text_model(input_ids)
        images = self.normalize(vision_pool)
        # The default text pooler runs, but similarity uses the raw first token.
        texts = self.normalize(self.text_projection(text_output.last_hidden_state[:, 0]))
        logits = self.matmul(texts, images.t()) / self.temperature
        return {
            "logits_per_text": logits, "logits_per_image": logits.t(),
            "text_embeds": texts, "image_embeds": images,
            "text_model_output.last_hidden_state": text_output.last_hidden_state,
            "text_model_output.pooler_output": text_output.pooler_output,
            "vision_model_output.last_hidden_state": vision_hidden,
            "vision_model_output.pooler_output": vision_pool,
        }


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if (text.hidden_act != "gelu" or text.is_decoder or text.add_cross_attention
            or text.chunk_size_feed_forward or text.position_embedding_type != "absolute"
            or vision.hidden_act != "swish" or vision.pooling_type != "mean"):
        raise ValueError("ALIGN coverage preserves its bidirectional GELU text and SiLU vision towers")
    if len(vision.num_block_repeats) != 7 or config.projection_dim != _round_filters(vision, vision.out_channels[-1]):
        raise ValueError("ALIGN requires all seven vision stages and matching image/text projection widths")
    if any(getattr(tower, name, False) for tower in (text, vision)
           for name in ("output_hidden_states", "output_attentions")):
        raise ValueError("ALIGN coverage returns the complete ordinary inference outputs")
    return AlignModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    vision_state = {key.removeprefix("vision_model."): remaining.pop(key)
                    for key in list(remaining) if key.startswith("vision_model.")}
    load_vision_state(model.vision_model, vision_state, config.vision_config)
    mapped = {}
    for name in model.text_model.state_dict():
        source = "text_model." + name.replace(".emb.weight", ".weight")
        for projection, native in (("q_proj", "self.query"), ("k_proj", "self.key"),
                                   ("v_proj", "self.value"), ("out_proj", "output.dense")):
            source = source.replace(f".attention.core.{projection}.", f".attention.{native}.")
        source = source.replace(".attention.LayerNorm.", ".attention.output.LayerNorm.")
        mapped[name] = remaining.pop(source)
    model.text_model.load_state_dict(mapped, strict=True)
    for name in ("weight", "bias"):
        getattr(model.text_projection, name).copy_(remaining.pop("text_projection." + name))
    model.temperature.copy_(remaining.pop("temperature"))
    if remaining:
        raise KeyError(f"Unmapped ALIGN state entries: {sorted(remaining)}")
