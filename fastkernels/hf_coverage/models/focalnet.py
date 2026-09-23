"""Default FocalNet, including context modulation and the public mean pooler."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from ..patches.product_gate import ProductGate
from ..runner import Workload


class PatchEmbedding(nn.Module):
    def __init__(self, source, target, patch, eps, norm):
        super().__init__()
        self.projection = Conv2d(source, target, patch, stride=patch)
        self.norm = LayerNorm(target, eps=eps) if norm else nn.Identity()
        self.patch = patch
        self.pad = Pad()

    def forward(self, x):
        x = self.pad(x, (0, -x.shape[-1] % self.patch, 0, -x.shape[-2] % self.patch))
        x = self.projection(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), (height, width)


class Modulation(nn.Module):
    def __init__(self, config, index, width):
        super().__init__()
        self.width = width
        self.levels = config.focal_levels[index]
        self.projection_in = Linear(width, 2 * width + self.levels + 1)
        self.projection_context = Conv2d(width, width, 1)
        self.projection_out = Linear(width, width)
        self.focal_layers = nn.ModuleList()
        for i in range(self.levels):
            kernel = 2 * i + config.focal_windows[index]
            self.focal_layers.append(nn.Sequential(Conv2d(width, width, kernel, padding=kernel // 2,
                                                          groups=width, bias=False), GELU()))
        self.activation = GELU()
        self.pool = GlobalAvgPool2d(keepdim=True)
        self.product = ProductGate()

    def multiply(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        return self.product(torch.cat((x, y), dim=-1))

    def forward(self, x):
        projected = self.projection_in(x).permute(0, 3, 1, 2).contiguous()
        query, context, gates = projected.split((self.width, self.width, self.levels + 1), dim=1)
        combined = None
        for level, layer in enumerate(self.focal_layers):
            context = layer(context)
            term = self.multiply(context, gates[:, level:level + 1])
            combined = term if combined is None else combined + term
        # Preserve HF's two sequential means, including the intermediate dtype boundary.
        by_column = self.pool(context.transpose(-1, -2).unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
        global_context = self.activation(self.pool(by_column))
        combined = combined + self.multiply(global_context, gates[:, self.levels:])
        hidden = self.multiply(query, self.projection_context(combined))
        return self.projection_out(hidden.permute(0, 2, 3, 1).contiguous())


class Block(nn.Module):
    def __init__(self, config, index, width):
        super().__init__()
        self.norm1 = LayerNorm(width, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(width, eps=config.layer_norm_eps)
        self.modulation = Modulation(config, index, width)
        self.mlp = nn.Module()
        self.mlp.fc1 = Linear(width, int(width * config.mlp_ratio))
        self.mlp.fc2 = Linear(int(width * config.mlp_ratio), width)
        self.activation = GELU()

    def forward(self, x, dimensions):
        height, width = dimensions
        hidden = self.norm1(x).reshape(x.shape[0], height, width, x.shape[-1])
        x = x + self.modulation(hidden).reshape_as(x)
        return x + self.mlp.fc2(self.activation(self.mlp.fc1(self.norm2(x))))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = PatchEmbedding(config.num_channels, config.embed_dim,
                                                          config.patch_size, config.layer_norm_eps, False)
        self.embeddings.norm = LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList()
        for i, depth in enumerate(config.depths):
            width = config.embed_dim * 2 ** i
            stage = nn.Module()
            stage.layers = nn.ModuleList([Block(config, i, width) for _ in range(depth)])
            stage.downsample = (PatchEmbedding(width, 2 * width, 2, config.layer_norm_eps, True)
                                if i < len(config.depths) - 1 else None)
            self.encoder.stages.append(stage)
        self.layernorm = LayerNorm(width, eps=config.layer_norm_eps)
        self.pool = GlobalAvgPool2d()

    def forward(self, pixel_values):
        x, dimensions = self.embeddings.patch_embeddings(pixel_values)
        x = self.embeddings.norm(x)
        for stage in self.encoder.stages:
            for block in stage.layers:
                x = block(x, dimensions)
            if stage.downsample is not None:
                image = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], *dimensions)
                x, dimensions = stage.downsample(image)
        x = self.layernorm(x)
        return {"last_hidden_state": x, "pooler_output": self.pool(x.transpose(1, 2).unsqueeze(-1))}


def build_from_config(config, device, dtype):
    flags = (config.use_conv_embed, config.use_layerscale, config.use_post_layernorm,
             config.use_post_layernorm_in_modulation, config.normalize_modulator)
    if any(flags) or config.hidden_act != "gelu":
        raise ValueError("Preserve the default FocalNet modulation and normalization branches")
    return Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
