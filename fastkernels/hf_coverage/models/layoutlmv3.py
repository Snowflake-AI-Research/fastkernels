"""LayoutLMv3 paired text/image encoding with both relative-position biases."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention


class TextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        for name, count, width in (
            ("word", config.vocab_size, config.hidden_size),
            ("token_type", config.type_vocab_size, config.hidden_size),
            ("position", config.max_position_embeddings, config.hidden_size),
            ("x_position", config.max_2d_position_embeddings, config.coordinate_size),
            ("y_position", config.max_2d_position_embeddings, config.coordinate_size),
            ("h_position", config.max_2d_position_embeddings, config.shape_size),
            ("w_position", config.max_2d_position_embeddings, config.shape_size),
        ):
            setattr(self, name + "_embeddings", Embedding(count, width))
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, ids, bbox):
        # Token positions and integer boxes are metadata, independent of activations.
        mask = ids.ne(self.padding_idx).int()
        positions = (mask.cumsum(1).type_as(mask) * mask).long() + self.padding_idx
        hidden = self.word_embeddings(ids) + self.token_type_embeddings(torch.zeros_like(ids))
        hidden = hidden + self.position_embeddings(positions)
        spatial = torch.cat((
            self.x_position_embeddings(bbox[..., 0]), self.y_position_embeddings(bbox[..., 1]),
            self.x_position_embeddings(bbox[..., 2]), self.y_position_embeddings(bbox[..., 3]),
            self.h_position_embeddings((bbox[..., 3] - bbox[..., 1]).clamp(0, 1023)),
            self.w_position_embeddings((bbox[..., 2] - bbox[..., 0]).clamp(0, 1023)),
        ), dim=-1)
        return self.LayerNorm(hidden + spatial)


class SpatialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        for name in ("query", "key", "value"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.matmul, self.softmax, self.reduce = BatchMatMul(), Softmax(), SegmentCSR()

    def forward(self, hidden, bias):
        batch, length, width = hidden.shape
        q, k, v = (getattr(self, name)(hidden).reshape(batch, length, self.heads, self.width)
                   .transpose(1, 2).reshape(batch * self.heads, length, self.width)
                   for name in ("query", "key", "value"))
        scores = self.matmul(q / math.sqrt(self.width), k.transpose(1, 2))
        scores = scores + bias.reshape(batch * self.heads, length, length) / math.sqrt(self.width)
        # Native CogView stabilization has an explicit rounded row maximum before softmax.
        scaled = scores / 32
        offsets = torch.arange(0, scaled.numel() + 1, length, device=hidden.device)
        maximum = self.reduce(scaled.flatten(), offsets, reduce="max").reshape(*scaled.shape[:-1], 1)
        probs = self.softmax((scaled - maximum) * 32)
        return self.matmul(probs, v).reshape(batch, self.heads, length, self.width).transpose(1, 2).reshape(batch, length, width)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = SpatialAttention(config)
        self.attention.output = EncoderSelfOutput(config)
        self.intermediate, self.output = EncoderIntermediate(config), EncoderOutput(config)

    def forward(self, hidden, bias):
        hidden = self.attention.output(self.attention.self(hidden, bias), hidden)
        return self.output(self.intermediate(hidden), hidden)


class LayoutLMv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TextEmbeddings(config)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        size = config.input_size // config.patch_size
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.empty(1, size * size + 1, config.hidden_size))
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.norm = LayerNorm(config.hidden_size, eps=1e-6, promote_fp32=False)
        positions = torch.arange(size + 1) * 1000 // size
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        boxes = torch.stack((positions[x], positions[y], positions[x + 1], positions[y + 1]), dim=-1).reshape(-1, 4)
        self.register_buffer("visual_bbox", torch.cat((torch.tensor([[1, 1, 999, 999]]), boxes)), persistent=False)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.encoder.rel_pos_bias = Embedding(config.rel_pos_bins, config.num_attention_heads)
        self.encoder.rel_pos_x_bias = Embedding(config.rel_2d_pos_bins, config.num_attention_heads)
        self.encoder.rel_pos_y_bias = Embedding(config.rel_2d_pos_bins, config.num_attention_heads)

    def relative_bias(self, positions, name, bins, distance):
        differences = positions.unsqueeze(-2) - positions.unsqueeze(-1)
        buckets = T5SelfAttention._relative_position_bucket(differences, num_buckets=bins, max_distance=distance)
        return getattr(self.encoder, name)(buckets).permute(0, 3, 1, 2).contiguous()

    def forward(self, input_ids, bbox, pixel_values):
        batch, length = input_ids.shape
        text = self.embeddings(input_ids, bbox)
        visual = self.patch_embed.proj(pixel_values).flatten(2).transpose(1, 2)
        visual = self.norm(torch.cat((self.cls_token.expand(batch, -1, -1), visual), dim=1) + self.pos_embed)
        hidden = self.LayerNorm(torch.cat((text, visual), dim=1))
        positions = torch.cat((torch.arange(length, device=input_ids.device),
                               torch.arange(visual.shape[1], device=input_ids.device))).expand(batch, -1)
        boxes = torch.cat((bbox, self.visual_bbox.expand(batch, -1, -1)), dim=1)
        config = self.config
        relative = self.relative_bias(positions, "rel_pos_bias", config.rel_pos_bins, config.max_rel_pos)
        spatial_x = self.relative_bias(boxes[..., 0], "rel_pos_x_bias", config.rel_2d_pos_bins, config.max_rel_2d_pos)
        spatial_y = self.relative_bias(boxes[..., 3], "rel_pos_y_bias", config.rel_2d_pos_bins, config.max_rel_2d_pos)
        bias = relative + (spatial_x + spatial_y)
        for layer in self.encoder.layer:
            hidden = layer(hidden, bias)
        return {"last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if not all((config.text_embed, config.visual_embed, config.has_relative_attention_bias, config.has_spatial_attention_bias)):
        raise ValueError("The selected LayoutLMv3 path includes text, image and both spatial biases")
    return LayoutLMv3(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        source = destination.replace(".emb.weight", ".weight")
        value = remaining.pop(source)
        if source.startswith("encoder.rel_pos"):
            value = value.T
        mapped[destination] = value
    if remaining:
        raise ValueError(f"Unmapped LayoutLMv3 weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
