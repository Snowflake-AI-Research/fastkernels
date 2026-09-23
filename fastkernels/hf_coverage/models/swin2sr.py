"""Swin2SR base backbone through existing SwinV2 blocks and convolutions."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.frozen_batch_norm2d import FrozenBatchNorm2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L3.swinv2_block import SwinV2Block


class _Stage(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinV2Block(c.embed_dim, (c.image_size, c.image_size), c.num_heads[index],
                        window_size=c.window_size, shift_size=c.window_size//2 if j % 2 else 0,
                        mlp_ratio=c.mlp_ratio, qkv_bias=c.qkv_bias)
            for j in range(c.depths[index])])
        for block in self.layers:
            if block.attn_mask is not None:
                # Pinned HF adds its configuration-derived shift mask twice.
                block.attn_mask.mul_(2)
            block.norm1.eps = block.norm2.eps = c.layer_norm_eps
        self.conv = Conv2d(c.embed_dim, c.embed_dim, 3, padding=1)
        self.patch_embed = nn.Module()
        self.patch_embed.projection = Conv2d(c.embed_dim, c.embed_dim, 1)

    def forward(self, x):
        residual = x
        for block in self.layers:
            x = block(x)
        x = self.patch_embed.projection(self.conv(x.permute(0, 3, 1, 2)))
        return residual + x.permute(0, 2, 3, 1)


class Swin2SRModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.image_size = c.image_size
        self.normalize = FrozenBatchNorm2d(3, eps=0.)
        self.normalize.running_mean.copy_(torch.tensor([0.4488, 0.4371, 0.4040]))
        self.img_range = c.img_range
        self.first_convolution = Conv2d(3, c.embed_dim, 3, padding=1)
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = Conv2d(c.embed_dim, c.embed_dim, 1)
        self.embeddings.patch_embeddings.layernorm = LayerNorm(c.embed_dim, promote_fp32=False)
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList([_Stage(c, i) for i in range(len(c.depths))])
        self.layernorm = LayerNorm(c.embed_dim, eps=c.layer_norm_eps, promote_fp32=False)
        self.conv_after_body = Conv2d(c.embed_dim, c.embed_dim, 3, padding=1)
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False

    def forward(self, pixel_values):
        if tuple(pixel_values.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError("This declared workload uses window-divisible configured images")
        embedded = self.first_convolution(self.normalize(pixel_values) * self.img_range)
        x = self.embeddings.patch_embeddings.projection(embedded).permute(0, 2, 3, 1)
        x = self.embeddings.patch_embeddings.layernorm(x)
        for stage in self.encoder.stages:
            x = stage(x)
        x = self.conv_after_body(self.layernorm(x).permute(0, 3, 1, 2)) + embedded
        return {"last_hidden_state": x}


def build_from_config(config, device, dtype):
    if (config.patch_size != 1 or config.resi_connection != "1conv" or config.use_absolute_embeddings
            or config.hidden_act != "gelu" or config.num_channels != 3 or config.num_channels_out != 3
            or config.image_size % config.window_size):
        raise ValueError("This case retains the documented default RGB, patch1, one-convolution Swin2SR backbone")
    return Swin2SRModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name, value in model.state_dict().items():
        if name.startswith("normalize."):
            mapped[name] = value
            continue
        source = name
        source = source.replace(".attn.proj.", ".attention.output.dense.")
        source = source.replace(".norm1.", ".layernorm_before.").replace(".norm2.", ".layernorm_after.")
        source = source.replace(".mlp.fc1.", ".intermediate.dense.").replace(".mlp.fc2.", ".output.dense.")
        source = source.replace(".attn.cpb_mlp.", ".attention.self.continuous_position_bias_mlp.")
        source = source.replace(".attn.logit_scale", ".attention.self.logit_scale")
        source = source.replace(".attn.q_bias", ".attention.self.query.bias").replace(".attn.v_bias", ".attention.self.value.bias")
        if source.endswith(".attn.qkv.weight"):
            mapped[name] = torch.cat([remaining.pop(source.replace(".attn.qkv.weight", f".attention.self.{q}.weight"))
                                      for q in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    if remaining:
        raise ValueError(f"Unmapped Swin2SR weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
