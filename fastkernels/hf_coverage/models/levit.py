"""Default LeViT stages, subsampling attention and mean pooler."""

import itertools
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from ..patches.hard_activations import HardSwish
from ..runner import Workload


class Attention(nn.Module):
    """Existing operations with LeViT's observed BF16 score-rounding boundary."""
    def __init__(self):
        super().__init__()
        self.matmul = BatchMatMul()
        self.softmax = Softmax()

    def forward(self, query, key, value, scale, bias):
        batch, query_tokens, heads, head_dim = query.shape
        key_tokens, value_dim = key.shape[1], value.shape[-1]
        query = query.transpose(1, 2).reshape(-1, query_tokens, head_dim)
        key = key.transpose(1, 2).reshape(-1, key_tokens, head_dim)
        value = value.transpose(1, 2).reshape(-1, key_tokens, value_dim)
        scores = self.matmul(query, key.transpose(-1, -2)).reshape(batch, heads, query_tokens, key_tokens)
        scores = scores * scale + bias
        context = self.matmul(self.softmax(scores).reshape(-1, query_tokens, key_tokens), value)
        return context.reshape(batch, heads, query_tokens, value_dim).transpose(1, 2)

class BatchNorm1dViaBatchNorm2d(BatchNorm2d):
    """Eval-mode BatchNorm1d over [batch, tokens, channels] via kb BatchNorm2d."""

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim != 3:
            raise RuntimeError(f"expected [batch, tokens, channels], got {tuple(hidden_state.shape)}")
        batch, tokens, channels = hidden_state.shape
        hidden_state = hidden_state.reshape(batch * tokens, channels, 1, 1)
        hidden_state = super().forward(hidden_state)
        return hidden_state.reshape(batch, tokens, channels)


class LevitConvEmbeddings(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.convolution = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = BatchNorm2d(out_channels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.convolution(pixel_values))


class LevitPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        h0 = int(config.hidden_sizes[0])
        self.embedding_layer_1 = LevitConvEmbeddings(
            int(config.num_channels), h0 // 8, int(config.kernel_size), int(config.stride), int(config.padding)
        )
        self.activation_layer_1 = HardSwish()
        self.embedding_layer_2 = LevitConvEmbeddings(
            h0 // 8, h0 // 4, int(config.kernel_size), int(config.stride), int(config.padding)
        )
        self.activation_layer_2 = HardSwish()
        self.embedding_layer_3 = LevitConvEmbeddings(
            h0 // 4, h0 // 2, int(config.kernel_size), int(config.stride), int(config.padding)
        )
        self.activation_layer_3 = HardSwish()
        self.embedding_layer_4 = LevitConvEmbeddings(
            h0 // 2, h0, int(config.kernel_size), int(config.stride), int(config.padding)
        )
        self.num_channels = int(config.num_channels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if int(pixel_values.shape[1]) != self.num_channels:
            raise RuntimeError(f"expected {self.num_channels} image channels, got {int(pixel_values.shape[1])}")
        hidden_state = self.embedding_layer_1(pixel_values)
        hidden_state = self.activation_layer_1(hidden_state)
        hidden_state = self.embedding_layer_2(hidden_state)
        hidden_state = self.activation_layer_2(hidden_state)
        hidden_state = self.embedding_layer_3(hidden_state)
        hidden_state = self.activation_layer_3(hidden_state)
        hidden_state = self.embedding_layer_4(hidden_state)
        return hidden_state.flatten(2).transpose(1, 2)


class MLPLayerWithBN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = Linear(int(input_dim), int(output_dim), bias=False)
        self.batch_norm = BatchNorm1dViaBatchNorm2d(int(output_dim))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear(hidden_state)
        return self.batch_norm(hidden_state)


class LevitSubsample(nn.Module):
    def __init__(self, stride: int, resolution: int):
        super().__init__()
        self.stride = int(stride)
        self.resolution = int(resolution)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _tokens, channels = hidden_state.shape
        grid = hidden_state.view(batch_size, self.resolution, self.resolution, channels)
        hidden_state = grid[:, :: self.stride, :: self.stride]
        return hidden_state.reshape(batch_size, -1, channels)


def _attention_bias_indices(resolution: int) -> tuple[list[int], int]:
    points = list(itertools.product(range(int(resolution)), range(int(resolution))))
    attention_offsets: dict[tuple[int, int], int] = {}
    indices: list[int] = []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if offset not in attention_offsets:
                attention_offsets[offset] = len(attention_offsets)
            indices.append(attention_offsets[offset])
    return indices, len(attention_offsets)


def _subsample_attention_bias_indices(resolution_in: int, resolution_out: int, stride: int) -> tuple[list[int], int]:
    points_in = list(itertools.product(range(int(resolution_in)), range(int(resolution_in))))
    points_out = list(itertools.product(range(int(resolution_out)), range(int(resolution_out))))
    attention_offsets: dict[tuple[float, float], int] = {}
    indices: list[int] = []
    for p1 in points_out:
        for p2 in points_in:
            offset = (abs(p1[0] * stride - p2[0]), abs(p1[1] * stride - p2[1]))
            if offset not in attention_offsets:
                attention_offsets[offset] = len(attention_offsets)
            indices.append(attention_offsets[offset])
    return indices, len(attention_offsets)


class LevitAttention(nn.Module):
    def __init__(self, hidden_size: int, key_dim: int, num_attention_heads: int, attention_ratio: int, resolution: int):
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.scale = int(key_dim) ** -0.5
        self.key_dim = int(key_dim)
        self.attention_ratio = int(attention_ratio)
        self.out_dim_keys_values = (
            self.attention_ratio * self.key_dim * self.num_attention_heads
            + self.key_dim * self.num_attention_heads * 2
        )
        self.out_dim_projection = self.attention_ratio * self.key_dim * self.num_attention_heads

        self.queries_keys_values = MLPLayerWithBN(int(hidden_size), self.out_dim_keys_values)
        self.activation = HardSwish()
        self.projection = MLPLayerWithBN(self.out_dim_projection, int(hidden_size))
        self.dense_attn = Attention()

        indices, num_offsets = _attention_bias_indices(int(resolution))
        len_points = int(resolution) * int(resolution)
        self.attention_biases = nn.Parameter(torch.zeros(self.num_attention_heads, num_offsets))
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(indices).view(len_points, len_points), persistent=False
        )

    def _bias(self) -> torch.Tensor:
        return self.attention_biases[:, self.attention_bias_idxs].unsqueeze(0)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_state.shape
        qkv = self.queries_keys_values(hidden_state)
        query, key, value = qkv.view(batch_size, seq_length, self.num_attention_heads, -1).split(
            [self.key_dim, self.key_dim, self.attention_ratio * self.key_dim], dim=3
        )
        context = self.dense_attn(query, key, value, self.scale, self._bias())
        context = context.reshape(batch_size, seq_length, self.out_dim_projection)
        return self.projection(self.activation(context))


class LevitAttentionSubsample(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key_dim: int,
        num_attention_heads: int,
        attention_ratio: int,
        stride: int,
        resolution_in: int,
        resolution_out: int,
    ):
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.scale = int(key_dim) ** -0.5
        self.key_dim = int(key_dim)
        self.attention_ratio = int(attention_ratio)
        self.out_dim_keys_values = (
            self.attention_ratio * self.key_dim * self.num_attention_heads
            + self.key_dim * self.num_attention_heads
        )
        self.out_dim_projection = self.attention_ratio * self.key_dim * self.num_attention_heads
        self.resolution_out = int(resolution_out)

        self.keys_values = MLPLayerWithBN(int(input_dim), self.out_dim_keys_values)
        self.queries_subsample = LevitSubsample(int(stride), int(resolution_in))
        self.queries = MLPLayerWithBN(int(input_dim), self.key_dim * self.num_attention_heads)
        self.activation = HardSwish()
        self.projection = MLPLayerWithBN(self.out_dim_projection, int(output_dim))
        self.dense_attn = Attention()

        indices, num_offsets = _subsample_attention_bias_indices(int(resolution_in), int(resolution_out), int(stride))
        self.attention_biases = nn.Parameter(torch.zeros(self.num_attention_heads, num_offsets))
        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(indices).view(self.resolution_out**2, int(resolution_in) ** 2),
            persistent=False,
        )

    def _bias(self) -> torch.Tensor:
        return self.attention_biases[:, self.attention_bias_idxs].unsqueeze(0)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_state.shape
        key, value = self.keys_values(hidden_state).view(batch_size, seq_length, self.num_attention_heads, -1).split(
            [self.key_dim, self.attention_ratio * self.key_dim], dim=3
        )
        query = self.queries(self.queries_subsample(hidden_state))
        query = query.view(batch_size, self.resolution_out**2, self.num_attention_heads, self.key_dim)
        context = self.dense_attn(query, key, value, self.scale, self._bias())
        context = context.reshape(batch_size, -1, self.out_dim_projection)
        hidden_state = self.projection(self.activation(context))
        return hidden_state


class LevitMLPLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_up = MLPLayerWithBN(int(input_dim), int(hidden_dim))
        self.activation = HardSwish()
        self.linear_down = MLPLayerWithBN(int(hidden_dim), int(input_dim))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_down(self.activation(self.linear_up(hidden_state)))


class LevitResidualLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state + self.module(hidden_state)


class LevitStage(nn.Module):
    def __init__(
        self,
        config,
        idx: int,
        hidden_size: int,
        key_dim: int,
        depth: int,
        num_attention_heads: int,
        attention_ratio: int,
        mlp_ratio: int,
        down_ops: list,
        resolution_in: int,
    ):
        super().__init__()
        self.resolution_in = int(resolution_in)
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.append(
                LevitResidualLayer(
                    LevitAttention(hidden_size, key_dim, num_attention_heads, attention_ratio, self.resolution_in)
                )
            )
            if int(mlp_ratio) > 0:
                layers.append(LevitResidualLayer(LevitMLPLayer(hidden_size, int(hidden_size) * int(mlp_ratio))))

        if down_ops and down_ops[0] == "Subsample":
            stride = int(down_ops[5])
            self.resolution_out = (self.resolution_in - 1) // stride + 1
            hidden_sizes = [int(x) for x in config.hidden_sizes]
            layers.append(
                LevitAttentionSubsample(
                    hidden_sizes[idx],
                    hidden_sizes[idx + 1],
                    key_dim=int(down_ops[1]),
                    num_attention_heads=int(down_ops[2]),
                    attention_ratio=int(down_ops[3]),
                    stride=stride,
                    resolution_in=self.resolution_in,
                    resolution_out=self.resolution_out,
                )
            )
            self.resolution_in = self.resolution_out
            if int(down_ops[4]) > 0:
                hidden_dim = hidden_sizes[idx + 1] * int(down_ops[4])
                layers.append(LevitResidualLayer(LevitMLPLayer(hidden_sizes[idx + 1], hidden_dim)))

        self.layers = nn.ModuleList(layers)

    def get_resolution(self) -> int:
        return self.resolution_in

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class LevitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        resolution = int(config.image_size) // int(config.patch_size)
        ops = [list(op) for op in config.down_ops[:len(config.depths)]]
        ops += [[""]] * (len(config.depths) - len(ops))
        self.stages = nn.ModuleList()
        for stage_idx in range(len(config.depths)):
            stage = LevitStage(
                config,
                stage_idx,
                int(config.hidden_sizes[stage_idx]),
                int(config.key_dim[stage_idx]),
                int(config.depths[stage_idx]),
                int(config.num_attention_heads[stage_idx]),
                int(config.attention_ratio[stage_idx]),
                int(config.mlp_ratio[stage_idx]),
                ops[stage_idx],
                resolution,
            )
            resolution = stage.get_resolution()
            self.stages.append(stage)
        self.final_resolution = int(resolution)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            hidden_state = stage(hidden_state)
        return hidden_state



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = LevitPatchEmbeddings(config)
        self.encoder = LevitEncoder(config)
        self.pool = GlobalAvgPool2d()

    def forward(self, pixel_values):
        hidden = self.encoder(self.patch_embeddings(pixel_values))
        return {"last_hidden_state": hidden, "pooler_output": self.pool(hidden.transpose(1, 2).unsqueeze(-1))}

def build_from_config(config, device, dtype):
    return Model(config).to(device=device, dtype=dtype).eval()

def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)

def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
