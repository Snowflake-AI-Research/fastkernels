"""ViTMatte's relative/window attention backbone and full detail-capture head."""

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.sam3_prompt_encoder import LayerNorm2d
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.swinv2_block import window_partition, window_reverse


class SpatialRelativeAttention(nn.Module):
    """Separate spatial-axis relative projections through existing batched matmul."""

    def __init__(self, width, heads, shape, qkv_bias=True, fp32_softmax=False):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.qkv, self.proj = Linear(width, width*3, bias=qkv_bias), Linear(width, width)
        self.rel_pos_h = nn.Parameter(torch.empty(2*shape[0]-1, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.empty(2*shape[1]-1, self.head_dim))
        self.bmm, self.softmax, self.resize = BMM(), Softmax(), Interpolate()
        self.fp32_softmax = fp32_softmax

    def relative_table(self, table, length):
        resized = self.resize(table.T[None], size=2*length-1, mode="linear", align_corners=False)[0].T
        positions = torch.arange(length, device=table.device)
        return resized[positions[:, None] - positions[None, :] + length - 1]

    def forward(self, x):
        batch, h, w, width = x.shape
        qkv = self.qkv(x).reshape(batch, h*w, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, batch*self.heads, h*w, self.head_dim).unbind(0)
        scores = self.bmm(q * self.head_dim**-0.5, k.transpose(-1, -2))
        query = q.reshape(batch*self.heads, h, w, self.head_dim)
        rh = self.relative_table(self.rel_pos_h, h)
        rw = self.relative_table(self.rel_pos_w, w)
        # Batched axes are H or W, with the other spatial axis and batch as rows.
        rel_h = self.bmm(query.permute(1, 0, 2, 3).reshape(h, -1, self.head_dim), rh.transpose(1, 2))
        rel_h = rel_h.reshape(h, batch*self.heads, w, h).permute(1, 0, 2, 3)
        rel_w = self.bmm(query.permute(2, 0, 1, 3).reshape(w, -1, self.head_dim), rw.transpose(1, 2))
        rel_w = rel_w.reshape(w, batch*self.heads, h, w).permute(1, 2, 0, 3)
        scores = scores.reshape(batch*self.heads, h, w, h, w) + rel_h[..., None] + rel_w[..., None, :]
        scores = scores.reshape(batch*self.heads, h*w, h*w)
        probabilities = self.softmax(scores.float() if self.fp32_softmax else scores).to(q.dtype)
        context = self.bmm(probabilities, v).reshape(batch, self.heads, h, w, self.head_dim)
        return self.proj(context.permute(0, 2, 3, 1, 4).reshape(batch, h, w, width))


class _ResidualBottleneck(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = Conv2d(width, width//2, 1, bias=False)
        self.norm1, self.act1 = LayerNorm2d(width//2), GELU()
        self.conv2 = Conv2d(width//2, width//2, 3, padding=1, bias=False)
        self.norm2, self.act2 = LayerNorm2d(width//2), GELU()
        self.conv3 = Conv2d(width//2, width, 1, bias=False)
        self.norm3 = LayerNorm2d(width)

    def forward(self, x):
        hidden = self.act1(self.norm1(self.conv1(x)))
        hidden = self.act2(self.norm2(self.conv2(hidden)))
        return x + self.norm3(self.conv3(hidden))


class _Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.window = c.window_size if index in c.window_block_indices else 0
        shape = (self.window, self.window) if self.window else (c.image_size//c.patch_size,)*2
        self.norm1 = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.attention = SpatialRelativeAttention(c.hidden_size, c.num_attention_heads, shape, c.qkv_bias)
        self.mlp = VitEncoderMlp(c.hidden_size, int(c.hidden_size*c.mlp_ratio), c.hidden_size)
        self.residual = _ResidualBottleneck(c.hidden_size) if index in c.residual_block_indices else nn.Identity()

    def forward(self, x):
        hidden = x.permute(0, 2, 3, 1)
        normalized = self.norm1(hidden)
        if self.window:
            _, h, w, _ = normalized.shape
            ph, pw = (-h) % self.window, (-w) % self.window
            normalized = F.pad(normalized, (0, 0, 0, pw, 0, ph))
            normalized = window_partition(normalized, (self.window, self.window))
            attended = self.attention(normalized)
            attended = window_reverse(attended, (self.window, self.window), (h+ph, w+pw))[:, :h, :w]
        else:
            attended = self.attention(normalized)
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.norm2(hidden))
        return self.residual(hidden.permute(0, 3, 1, 2))


class _Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.projection = Conv2d(c.num_channels, c.hidden_size, c.patch_size, stride=c.patch_size)
        self.pretrain_grid = c.pretrain_image_size // c.patch_size
        self.embeddings.position_embeddings = nn.Parameter(torch.empty(1, self.pretrain_grid**2+1, c.hidden_size))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Layer(c, i) for i in range(c.num_hidden_layers)])
        self.resize = Interpolate()

    def forward(self, pixels):
        hidden = self.embeddings.projection(pixels)
        positions = self.embeddings.position_embeddings[:, 1:].reshape(1, self.pretrain_grid, self.pretrain_grid, -1).permute(0, 3, 1, 2)
        positions = self.resize(positions, size=hidden.shape[-2:], mode="bicubic", align_corners=False)
        hidden = hidden + positions
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return hidden


class _Conv(nn.Module):
    def __init__(self, c, incoming, outgoing, stride=2):
        super().__init__()
        self.conv = Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.batch_norm, self.relu = BatchNorm2d(outgoing, eps=c.batch_norm_eps), ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


class _Fusion(nn.Module):
    def __init__(self, c, incoming, outgoing):
        super().__init__()
        self.conv, self.resize = _Conv(c, incoming, outgoing, stride=1), Interpolate()

    def forward(self, x, detail):
        return self.conv(torch.cat((detail, self.resize(x, scale_factor=2, mode="bilinear", align_corners=False)), dim=1))


class VitMatteForImageMatting(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.backbone = _Backbone(c.backbone_config)
        self.decoder = nn.Module()
        self.decoder.convstream = nn.Module()
        channels = [c.backbone_config.num_channels] + list(c.convstream_hidden_sizes)
        self.decoder.convstream.convs = nn.ModuleList([_Conv(c, i, o) for i, o in zip(channels, channels[1:])])
        fusion = [c.hidden_size] + list(c.fusion_hidden_sizes)
        self.decoder.fusion_blocks = nn.ModuleList([_Fusion(c, i+channels[-j-1], o)
            for j, (i, o) in enumerate(zip(fusion, fusion[1:]))])
        self.decoder.matting_head = nn.Module()
        self.decoder.matting_head.matting_convs = nn.Sequential(
            Conv2d(fusion[-1], 16, 3, padding=1), BatchNorm2d(16), ReLU(), Conv2d(16, 1, 1))
        self.sigmoid = Sigmoid()

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        details = [pixel_values]
        for conv in self.decoder.convstream.convs:
            details.append(conv(details[-1]))
        for fusion, detail in zip(self.decoder.fusion_blocks, reversed(details)):
            features = fusion(features, detail)
        return {"alphas": self.sigmoid(self.decoder.matting_head.matting_convs(features))}


def build_from_config(config, device, dtype):
    c = config.backbone_config
    if (c.model_type != "vitdet" or not c.use_relative_position_embeddings or not c.use_absolute_position_embeddings
            or c.hidden_act != "gelu" or tuple(c.out_indices) != (c.num_hidden_layers,)):
        raise ValueError("The documented matting checkpoint uses the final relative-position ViTDet feature map")
    return VitMatteForImageMatting(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
