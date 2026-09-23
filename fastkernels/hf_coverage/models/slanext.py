"""SLANeXt window/global image encoding and shared recurrent table head."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from .slanet import StructureHead, load_state_dict_into, make_workloads


class SpatialAttention(nn.Module):
    def __init__(self, config, window):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        size = window or config.image_size // config.patch_size
        self.qkv, self.proj = Linear(config.hidden_size, 3 * config.hidden_size, bias=config.qkv_bias), Linear(config.hidden_size, config.hidden_size)
        self.rel_pos_h = nn.Parameter(torch.empty(2 * size - 1, self.width))
        self.rel_pos_w = nn.Parameter(torch.empty(2 * size - 1, self.width))
        self.matmul, self.softmax, self.resize = BatchMatMul(), Softmax(), Interpolate()

    def positions(self, size, table):
        table = self.resize(table.T.unsqueeze(0), size=2 * size - 1, mode="linear").squeeze(0).T
        index = torch.arange(size, device=table.device)
        return table[index[:, None] - index[None, :] + size - 1]

    def forward(self, hidden):
        batch, height, width, channels = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, height * width, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.reshape(3, batch * self.heads, height * width, self.width).unbind(0)
        scores = self.matmul(query * self.width ** -0.5, key.transpose(1, 2))
        spatial = query.reshape(batch * self.heads, height, width, self.width)
        rows = spatial.permute(1, 0, 2, 3).reshape(height, -1, self.width)
        rel_h = self.matmul(rows, self.positions(height, self.rel_pos_h).transpose(1, 2))
        rel_h = rel_h.reshape(height, batch * self.heads, width, height).permute(1, 0, 2, 3)
        columns = spatial.permute(2, 0, 1, 3).reshape(width, -1, self.width)
        rel_w = self.matmul(columns, self.positions(width, self.rel_pos_w).transpose(1, 2))
        rel_w = rel_w.reshape(width, batch * self.heads, height, width).permute(1, 2, 0, 3)
        relative = rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)
        scores = scores + relative.reshape_as(scores)
        probs = self.softmax(scores.float()).to(query.dtype)
        hidden = self.matmul(probs, value).reshape(batch, self.heads, height, width, self.width)
        return self.proj(hidden.permute(0, 2, 3, 1, 4).reshape(batch, height, width, channels))


class VisionLayer(nn.Module):
    def __init__(self, config, window):
        super().__init__()
        self.window = window
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.attn = SpatialAttention(config, window)
        self.mlp = nn.Module()
        self.mlp.lin1, self.mlp.lin2 = Linear(config.hidden_size, config.mlp_dim), Linear(config.mlp_dim, config.hidden_size)
        self.mlp.act, self.pad = GELU(), Pad()

    def forward(self, hidden):
        residual, hidden = hidden, self.layer_norm1(hidden)
        batch, height, width, channels = hidden.shape
        if self.window:
            window = self.window
            ph, pw = (-height) % window, (-width) % window
            hidden = self.pad(hidden, (0, 0, 0, pw, 0, ph))
            hidden = hidden.reshape(batch, (height + ph) // window, window, (width + pw) // window, window, channels)
            hidden = hidden.permute(0, 1, 3, 2, 4, 5).reshape(-1, window, window, channels)
        hidden = self.attn(hidden)
        if self.window:
            hidden = hidden.reshape(batch, (height + ph) // window, (width + pw) // window, window, window, channels)
            hidden = hidden.permute(0, 1, 3, 2, 4, 5).reshape(batch, height + ph, width + pw, channels)
            hidden = hidden[:, :height, :width].contiguous()
        hidden = residual + hidden
        return hidden + self.mlp.lin2(self.mlp.act(self.mlp.lin1(self.layer_norm2(hidden))))


class ChannelNorm(LayerNorm):
    def forward(self, hidden):
        return super().forward(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.projection = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        side = config.image_size // config.patch_size
        self.pos_embed = nn.Parameter(torch.empty(1, side, side, config.hidden_size))
        self.layers = nn.ModuleList([VisionLayer(config, 0 if i in config.global_attn_indexes else config.window_size)
                                     for i in range(config.num_hidden_layers)])
        self.neck = nn.Module()
        self.neck.conv1 = Conv2d(config.hidden_size, config.output_channels, 1, bias=False)
        self.neck.conv2 = Conv2d(config.output_channels, config.output_channels, 3, padding=1, bias=False)
        self.neck.layer_norm1 = ChannelNorm(config.output_channels, eps=1e-6, promote_fp32=False)
        self.neck.layer_norm2 = ChannelNorm(config.output_channels, eps=1e-6, promote_fp32=False)

    def forward(self, pixels):
        hidden = self.patch_embed.projection(pixels).permute(0, 2, 3, 1) + self.pos_embed
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.neck.layer_norm1(self.neck.conv1(hidden.permute(0, 3, 1, 2)))
        return self.neck.layer_norm2(self.neck.conv2(hidden))


class SLANeXt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.vision_tower = Vision(config.vision_config)
        self.backbone.post_conv = Conv2d(config.post_conv_in_channels, config.post_conv_out_channels, 3, stride=2, padding=1, bias=False)
        self.head = StructureHead(config)

    def forward(self, pixel_values):
        hidden = self.backbone.post_conv(self.backbone.vision_tower(pixel_values)).flatten(2).transpose(1, 2)
        return {"last_hidden_state": self.head(hidden)}


def build_from_config(config, device, dtype):
    if not config.vision_config.use_abs_pos or not config.vision_config.use_rel_pos:
        raise ValueError("The selected SLANeXt vision path includes absolute and relative positions")
    model = SLANeXt(config).to(device=device, dtype=dtype).eval()
    # Pinned HF keeps both recurrent attention and structure projection in FP32.
    model.head.float()
    return model
