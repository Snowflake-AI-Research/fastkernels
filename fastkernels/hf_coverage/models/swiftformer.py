"""SwiftFormer's default four-stage feature extractor and additive attention."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax

from ..patches.product_gate import ProductGate
from .vit_msn import make_workloads


def _channel_scale(gate, hidden_states, scale):
    # Pack whole NCHW samples so the existing gate processes contiguous pairs.
    packed = torch.cat((scale.expand_as(hidden_states), hidden_states), dim=1).flatten(1)
    return gate(packed).reshape(hidden_states.shape)


class _PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.embed_dims[0]
        self.patch_embedding = nn.Sequential(
            Conv2d(config.num_channels, width // 2, 3, stride=2, padding=1),
            BatchNorm2d(width // 2, eps=config.batch_norm_eps), ReLU(),
            Conv2d(width // 2, width, 3, stride=2, padding=1),
            BatchNorm2d(width, eps=config.batch_norm_eps), ReLU(),
        )

    def forward(self, pixel_values):
        return self.patch_embedding(pixel_values)


class _Downsample(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.proj = Conv2d(config.embed_dims[index], config.embed_dims[index + 1],
                           config.down_patch_size, stride=config.down_stride,
                           padding=config.down_pad)
        self.norm = BatchNorm2d(config.embed_dims[index + 1], eps=config.batch_norm_eps)

    def forward(self, hidden_states):
        return self.norm(self.proj(hidden_states))


class _ConvEncoder(nn.Module):
    """The shared convolution graph; the local-representation variant has no expansion."""

    def __init__(self, config, width, expand=True):
        super().__init__()
        intermediate = int(width * config.mlp_ratio) if expand else width
        self.depth_wise_conv = Conv2d(width, width, 3, padding=1, groups=width)
        self.norm = BatchNorm2d(width, eps=config.batch_norm_eps)
        self.point_wise_conv1 = Conv2d(width, intermediate, 1)
        self.act = GELU(approximate="none")
        self.point_wise_conv2 = Conv2d(intermediate, width, 1)
        self.layer_scale = nn.Parameter(torch.empty(width, 1, 1))
        self.gate = ProductGate()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(self.depth_wise_conv(hidden_states))
        hidden_states = self.point_wise_conv2(self.act(self.point_wise_conv1(hidden_states)))
        return residual + _channel_scale(self.gate, hidden_states, self.layer_scale)


class _Mlp(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.norm1 = BatchNorm2d(width, eps=config.batch_norm_eps)
        self.fc1 = Conv2d(width, int(width * config.mlp_ratio), 1)
        self.act = GELU(approximate="none")
        self.fc2 = Conv2d(int(width * config.mlp_ratio), width, 1)

    def forward(self, hidden_states):
        return self.fc2(self.act(self.fc1(self.norm1(hidden_states))))


class _AdditiveAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.to_query = Linear(width, width)
        self.to_key = Linear(width, width)
        self.w_g = nn.Parameter(torch.empty(width, 1))
        self.scale_factor = width ** -0.5
        self.proj = Linear(width, width)
        self.final = Linear(width, width)
        self.normalize = L2Norm(dim=-1, eps=1e-12)
        self.matmul = BMM()
        self.softmax = Softmax(dim=-1)
        self.gate = ProductGate()

    def forward(self, hidden_states):
        query = self.normalize(self.to_query(hidden_states))
        key = self.normalize(self.to_key(hidden_states))
        # Keep pinned HF's singleton-axis softmax and its preceding projection.
        weights = self.softmax(self.matmul(query, self.w_g) * self.scale_factor)
        global_query = self.matmul(weights.transpose(1, 2), query)
        gated = self.gate(torch.cat((global_query.expand_as(key), key), dim=-1))
        return self.final(self.proj(gated) + query)


class _EncoderBlock(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.local_representation = _ConvEncoder(config, width, expand=False)
        self.attn = _AdditiveAttention(width)
        self.linear = _Mlp(config, width)
        self.layer_scale_1 = nn.Parameter(torch.empty(width, 1, 1))
        self.layer_scale_2 = nn.Parameter(torch.empty(width, 1, 1))
        self.gate = ProductGate()

    def forward(self, hidden_states):
        hidden_states = self.local_representation(hidden_states)
        batch, channels, height, width = hidden_states.shape
        tokens = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        attended = self.attn(tokens).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = hidden_states + _channel_scale(self.gate, attended, self.layer_scale_1)
        return hidden_states + _channel_scale(self.gate, self.linear(hidden_states), self.layer_scale_2)


class _Stage(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        width = config.embed_dims[index]
        self.blocks = nn.ModuleList(
            [_ConvEncoder(config, width) for _ in range(config.depths[index] - 1)]
            + [_EncoderBlock(config, width)]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        network = []
        for index in range(len(config.depths)):
            network.append(_Stage(config, index))
            if index < len(config.depths) - 1:
                network.append(_Downsample(config, index))
        self.network = nn.ModuleList(network)

    def forward(self, hidden_states):
        for module in self.network:
            hidden_states = module(hidden_states)
        return hidden_states


class SwiftFormerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = _PatchEmbedding(config)
        self.encoder = _Encoder(config)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("SwiftFormer coverage supports inference only")
        return {"last_hidden_state": self.encoder(self.patch_embed(pixel_values))}


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or not config.use_layer_scale
            or config.drop_path_rate or config.drop_mlp_rate or config.drop_conv_encoder_rate):
        raise ValueError("This case preserves default exact GELU, layer scales and zero dropout")
    if (len(config.depths) != 4 or len(config.embed_dims) != 4
            or any(depth < 2 for depth in config.depths) or not all(config.downsamples)):
        raise ValueError("All four stages retain convolution and additive-attention blocks with downsampling")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("This case returns the ordinary final feature map")
    return SwiftFormerModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    model.load_state_dict(state_dict, strict=True)
