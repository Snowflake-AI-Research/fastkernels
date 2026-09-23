"""The published one-stage Deformable DETR, including all intermediate heads."""

import math

import torch
from torch import nn

from .detr import _Attention, _MLP, _ResNetFeatures, make_workloads
from ..patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rtdetrv2_deformable_attention import MultiScaleDeformableAttentionV2
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead
from fastkernels.tasks.baseline.L3.rtdetrv2_decoder import inverse_sigmoid


class _SamplingAttention(nn.Module):
    """Existing RT-DETR sampling core with explicit per-level coordinate wiring.

    The L2 parent flattens levels and points before constructing coordinates;
    these models instead supply a different reference point for every level.
    The L1 accepts exactly the resulting flattened sampling coordinates.
    """
    def __init__(self, width, heads, levels, points):
        super().__init__()
        self.width, self.heads, self.levels, self.points = width, heads, levels, points
        self.sampling_offsets = Linear(width, heads * levels * points * 2)
        self.attention_weights = Linear(width, heads * levels * points)
        self.value_proj, self.output_proj = Linear(width, width), Linear(width, width)
        self.softmax, self.sampling, self.product = Softmax(dim=-1), MultiScaleDeformableAttentionV2(), ProductGate()

    def forward(self, hidden, memory, positions, references, shapes, mask=None):
        hidden = hidden + positions
        batch, count = hidden.shape[:2]
        value = self.value_proj(memory)
        if mask is not None:
            value = value.masked_fill(~mask[..., None], 0)
        value = value.reshape(batch, -1, self.heads, self.width // self.heads)
        offsets = self.sampling_offsets(hidden).reshape(batch, count, self.heads, self.levels, self.points, 2)
        weights = self.softmax(self.attention_weights(hidden).reshape(batch, count, self.heads, -1))
        locations = []
        for level, (height, width) in enumerate(shapes):
            offset = offsets[:, :, :, level]
            if references.shape[-1] == 2:
                # Each divisor is fixed shape metadata; retain native division rounding.
                offset = torch.stack((offset[..., 0] / width, offset[..., 1] / height), dim=-1)
            else:
                offset = offset / self.points
                size = references[:, :, None, level, None, 2:].expand_as(offset)
                offset = self.product(torch.cat((offset, size), dim=-1)) * 0.5
            locations.append(references[:, :, None, level, None, :2] + offset)
        output = self.sampling(value, shapes, torch.cat(locations, dim=-2), weights, [self.points] * self.levels)
        return self.output_proj(output)


class _Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.backbone_config.backbone != "resnet50" or config.backbone_config.out_indices != [2, 3, 4]:
            raise ValueError("The published default returns the last three ResNet50 stages")
        self.model, self.interpolate = _ResNetFeatures("resnet50"), Interpolate()

    def forward(self, pixels, mask):
        return [(feature, self.interpolate(mask[None].float(), size=feature.shape[-2:]).bool()[0])
                for feature in self.model(pixels)[1:]]


class _EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.self_attn = _SamplingAttention(width, config.encoder_attention_heads, config.num_feature_levels, config.encoder_n_points)
        self.self_attn_layer_norm, self.final_layer_norm = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(2))
        self.mlp = _MLP(width, config.encoder_ffn_dim)

    def forward(self, hidden, positions, references, shapes, mask):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, hidden, positions, references, shapes, mask))
        return self.final_layer_norm(hidden + self.mlp(hidden))


class _DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.self_attn = _Attention(width, config.decoder_attention_heads)
        self.encoder_attn = _SamplingAttention(width, config.decoder_attention_heads, config.num_feature_levels, config.decoder_n_points)
        self.self_attn_layer_norm, self.encoder_attn_layer_norm, self.final_layer_norm = (
            LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(3))
        self.mlp = _MLP(width, config.decoder_ffn_dim)

    def forward(self, hidden, positions, memory, references, shapes, mask):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden, positions))
        hidden = self.encoder_attn_layer_norm(hidden + self.encoder_attn(hidden, memory, positions, references, shapes, mask))
        return self.final_layer_norm(hidden + self.mlp(hidden))


def _valid_ratios(mask, dtype):
    """Padding metadata only, preserving the native axis and dtype choices."""
    height, width = mask.shape[-2:]
    return torch.stack((mask[:, 0, :].sum(1).to(dtype) / width,
                        mask[:, :, 0].sum(1).to(dtype) / height), dim=-1)


def _centered_positions(mask, width, dtype):
    """This HF position formula subtracts half a pixel before normalization."""
    y, x = mask.cumsum(1, dtype=dtype), mask.cumsum(2, dtype=dtype)
    y = (y - .5) / (y[:, -1:, :] + 1e-6) * (2 * math.pi)
    x = (x - .5) / (x[:, :, -1:] + 1e-6) * (2 * math.pi)
    dim = torch.arange(width // 2, device=mask.device, dtype=torch.int64).to(dtype)
    frequencies = 10000 ** (2 * torch.div(dim, 2, rounding_mode="floor") / (width // 2))
    x, y = x[..., None] / frequencies, y[..., None] / frequencies
    x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(3)
    y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(3)
    return torch.cat((y, x), dim=-1).flatten(1, 2)


def _encoder_references(shapes, ratios):
    references = []
    for level, (height, width) in enumerate(shapes):
        y, x = torch.meshgrid(torch.linspace(.5, height - .5, height, dtype=ratios.dtype, device=ratios.device),
                              torch.linspace(.5, width - .5, width, dtype=ratios.dtype, device=ratios.device), indexing="ij")
        x = x.flatten()[None] / (ratios[:, None, level, 0] * width)
        y = y.flatten()[None] / (ratios[:, None, level, 1] * height)
        references.append(torch.stack((x, y), dim=-1))
    return torch.cat(references, dim=1)[:, :, None] * ratios[:, None]


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.width, self.backbone = width, _Backbone(config)
        self.input_proj = nn.ModuleList([nn.Sequential(Conv2d(source, width, 1), GroupNorm(32, width, eps=1e-5))
                                        for source in (512, 1024, 2048)])
        self.input_proj.append(nn.Sequential(Conv2d(2048, width, 3, stride=2, padding=1), GroupNorm(32, width, eps=1e-5)))
        self.query_position_embeddings = Embedding(config.num_queries, 2 * width)
        self.reference_points = Linear(width, 2)
        self.level_embed = nn.Parameter(torch.zeros(config.num_feature_levels, width))
        self.encoder, self.decoder = nn.Module(), nn.Module()
        self.encoder.layers = nn.ModuleList([_EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.decoder.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.interpolate, self.sigmoid, self.product = Interpolate(), Sigmoid(), ProductGate()

    def forward(self, pixels, pixel_mask=None):
        if pixel_mask is None:
            pixel_mask = torch.ones(pixels.shape[0], *pixels.shape[-2:], device=pixels.device)
        features = self.backbone(pixels, pixel_mask)
        sources = [projection(feature) for projection, (feature, _) in zip(self.input_proj, features)]
        masks = [mask for _, mask in features]
        sources.append(self.input_proj[-1](features[-1][0]))
        masks.append(self.interpolate(pixel_mask[None].to(pixels.dtype), size=sources[-1].shape[-2:]).bool()[0])
        positions = [_centered_positions(mask, self.width, pixels.dtype) + self.level_embed[index].view(1, 1, -1)
                     for index, mask in enumerate(masks)]
        shapes = [tuple(source.shape[-2:]) for source in sources]
        ratios = torch.stack([_valid_ratios(mask, pixels.dtype) for mask in masks], dim=1)
        mask = torch.cat([mask.flatten(1) for mask in masks], dim=1)
        hidden = torch.cat([source.flatten(2).transpose(1, 2) for source in sources], dim=1)
        positions = torch.cat(positions, dim=1)
        references = _encoder_references(shapes, ratios)
        for layer in self.encoder.layers:
            hidden = layer(hidden, positions, references, shapes, mask)
        memory = hidden
        queries = self.query_position_embeddings.emb.weight.unsqueeze(0).expand(pixels.shape[0], -1, -1)
        positions, hidden = queries.split(self.width, dim=-1)
        initial = self.sigmoid(self.reference_points(positions))
        reference = initial[:, :, None].expand(-1, -1, len(shapes), -1)
        reference = self.product(torch.cat((reference, ratios[:, None].expand_as(reference)), dim=-1))
        intermediate = []
        for layer in self.decoder.layers:
            hidden = layer(hidden, positions, memory, reference, shapes, mask)
            intermediate.append(hidden)
        return hidden, memory, torch.stack(intermediate, dim=1), initial


class _Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.model = _Model(config)
        self.class_embed = nn.ModuleList([Linear(width, len(config.id2label))] * config.decoder_layers)
        self.bbox_embed = nn.ModuleList([RTDetrV2MLPPredictionHead(config, width, width, 4, 3)] * config.decoder_layers)
        self.sigmoid = Sigmoid()

    def forward(self, pixel_values, pixel_mask=None):
        hidden, memory, intermediate, initial = self.model(pixel_values, pixel_mask)
        logits, boxes = [], []
        for index in range(intermediate.shape[1]):
            logits.append(self.class_embed[index](intermediate[:, index]))
            delta = self.bbox_embed[index](intermediate[:, index])
            delta = torch.cat((delta[..., :2] + inverse_sigmoid(initial), delta[..., 2:]), dim=-1)
            boxes.append(self.sigmoid(delta))
        return {"logits": torch.stack(logits)[-1], "pred_boxes": torch.stack(boxes)[-1],
                "last_hidden_state": hidden, "encoder_last_hidden_state": memory,
                "intermediate_hidden_states": intermediate, "init_reference_points": initial,
                "intermediate_reference_points": torch.stack([initial] * intermediate.shape[1], dim=1)}


def build_from_config(config, device, dtype):
    if (config.two_stage or config.with_box_refine or config.num_feature_levels != 4
            or config.activation_function != "relu" or config.position_embedding_type != "sine" or config.dilation):
        raise ValueError("This case preserves the published one-stage four-level Deformable DETR default")
    return _Detector(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {key.replace("query_position_embeddings.weight", "query_position_embeddings.emb.weight")
              .replace(".o_proj.", ".out_proj."): value for key, value in state_dict.items()}
    model.load_state_dict(mapped, strict=True)
