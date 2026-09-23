"""Hiera image encoder with local/global attention, query pooling and default pooler."""

import math
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.softmax import Softmax
from ..runner import Workload

def _as_pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)

class HieraPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = int(config.num_channels)
        self.image_size = _as_pair(config.image_size)
        self.projection = Conv2d(
            self.num_channels,
            int(config.embed_dim),
            kernel_size=_as_pair(config.patch_size),
            stride=_as_pair(config.patch_stride),
            padding=_as_pair(config.patch_padding),
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = pixel_values.shape
        if channels != self.num_channels:
            raise RuntimeError(f"expected {self.num_channels} channels, got {channels}")
        if (height, width) != self.image_size:
            raise RuntimeError(f"fixed-size artifact expected image {self.image_size}, got {(height, width)}")
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class HieraEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = HieraPatchEmbeddings(config)
        tokens_spatial_shape = [i // s for i, s in zip(config.image_size, config.patch_stride)]
        self.num_tokens = math.prod(tokens_spatial_shape)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, int(config.embed_dim)))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        return embeddings + self.position_embeddings.to(dtype=embeddings.dtype, device=embeddings.device)


def _unroll(hidden_states: torch.Tensor, image_shape, patch_stride, schedule) -> torch.Tensor:
    batch_size, _, hidden_size = hidden_states.shape
    size = [i // s for i, s in zip(image_shape, patch_stride)]
    current_size = size
    hidden_states = hidden_states.view(batch_size, *current_size, hidden_size)

    for strides in schedule:
        current_size = [i // s for i, s in zip(current_size, strides)]
        new_shape = [item for pair in zip(current_size, strides) for item in pair]
        new_shape = [batch_size] + new_shape + [hidden_size]
        hidden_states = hidden_states.view(new_shape)
        num_dims = len(new_shape)
        permute = [0] + list(range(2, num_dims - 1, 2)) + list(range(1, num_dims - 1, 2)) + [num_dims - 1]
        hidden_states = hidden_states.permute(permute)
        hidden_states = hidden_states.flatten(0, len(strides))
        batch_size *= math.prod(strides)

    return hidden_states.reshape(-1, math.prod(size), hidden_size)


class HieraMlp(nn.Module):
    def __init__(self, config, dim: int):
        super().__init__()
        self.fc1 = Linear(dim, int(dim * float(config.mlp_ratio)), bias=True)
        self.act = GELU(approximate="none")
        self.fc2 = Linear(int(dim * float(config.mlp_ratio)), dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class HieraMaskUnitAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_size_output: int,
        num_heads: int,
        query_stride: int,
        window_size: int,
        use_mask_unit_attn: bool,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.query_stride = int(query_stride)
        self.hidden_size_output = int(hidden_size_output)
        self.head_dim = self.hidden_size_output // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = int(window_size)
        self.use_mask_unit_attn = bool(use_mask_unit_attn)
        self.qkv = Linear(int(hidden_size), 3 * self.hidden_size_output, bias=True)
        self.proj = Linear(self.hidden_size_output, self.hidden_size_output, bias=True)
        self.matmul = BatchMatMul()
        self.softmax = Softmax()
        self.q_pool = (
            MaxPool2d(kernel_size=(self.query_stride, 1), stride=(self.query_stride, 1))
            if self.query_stride > 1 else None
        )

    def _pool_query_stride(self, query: torch.Tensor) -> torch.Tensor:
        # HF uses query.view(..., query_stride, -1, head_dim).max(dim=3).
        batch, heads, num_windows, key_tokens, head_dim = query.shape
        query = query.view(batch, heads, num_windows, self.query_stride, -1, head_dim)
        query_tokens = query.shape[4]
        query = query.permute(0, 1, 2, 4, 5, 3).reshape(-1, head_dim, self.query_stride, 1)
        query = self.q_pool(query).squeeze(-1).squeeze(-1)
        return query.view(batch, heads, num_windows, query_tokens, head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        num_windows = 1
        if self.use_mask_unit_attn:
            num_windows = seq_len // (self.query_stride * self.window_size)

        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, -1, num_windows, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 2, 1, 5)
        query, key, value = qkv.unbind(0)

        if self.query_stride > 1:
            query = self._pool_query_stride(query)

        batch, heads, windows, q_tokens, head_dim = query.shape
        k_tokens = key.shape[3]
        # HF rounds the scaled query before QK and stores scores/probabilities
        # in the input dtype. Fused SDPA skips those tested rounding boundaries.
        query = (query * self.scale).reshape(-1, q_tokens, head_dim)
        key = key.reshape(-1, k_tokens, head_dim)
        value = value.reshape(-1, k_tokens, head_dim)
        scores = self.matmul(query, key.transpose(-1, -2))
        attn_output = self.matmul(self.softmax(scores), value)
        attn_output = attn_output.reshape(batch, heads, windows, q_tokens, head_dim)
        attn_output = attn_output.transpose(1, 3).reshape(batch, -1, self.hidden_size_output)
        return self.proj(attn_output)


class HieraLayer(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        hidden_size_output: int,
        num_heads: int,
        query_stride: int,
        window_size: int,
        use_mask_unit_attn: bool,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.hidden_size_output = int(hidden_size_output)
        self.query_stride = int(query_stride)
        self.layernorm_before = LayerNorm(self.hidden_size, eps=float(config.layer_norm_eps))
        self.attn = HieraMaskUnitAttention(
            self.hidden_size,
            self.hidden_size_output,
            num_heads,
            self.query_stride,
            window_size,
            use_mask_unit_attn,
        )
        self.layernorm_after = LayerNorm(self.hidden_size_output, eps=float(config.layer_norm_eps))
        self.mlp = HieraMlp(config, self.hidden_size_output)
        self.res_pool = (
            MaxPool2d(kernel_size=(self.query_stride, 1), stride=(self.query_stride, 1))
            if self.query_stride > 1 else None
        )
        if self.hidden_size != self.hidden_size_output:
            self.proj = Linear(self.hidden_size, self.hidden_size_output, bias=True)
        else:
            self.proj = None

    def _pool_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, self.query_stride, -1, hidden_size)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.res_pool(hidden_states).squeeze(2)
        return hidden_states.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states_norm)
            hidden_states = self._pool_residual(hidden_states)

        attn_output = self.attn(hidden_states_norm)
        hidden_states = hidden_states + attn_output

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class HieraStage(nn.Module):
    def __init__(
        self,
        config,
        depth: int,
        hidden_size: int,
        hidden_size_output: int,
        num_heads: int,
        query_stride: list[int],
        window_size: int,
        use_mask_unit_attn: bool,
        stage_num: int,
    ):
        super().__init__()
        previous_stage_used_masked_attention = config.masked_unit_attention[stage_num - 1 if stage_num > 0 else 0]
        self.layers = nn.ModuleList(
            [
                HieraLayer(
                    config=config,
                    hidden_size=hidden_size if i == 0 else hidden_size_output,
                    hidden_size_output=hidden_size_output,
                    num_heads=num_heads,
                    query_stride=query_stride[i],
                    window_size=window_size,
                    use_mask_unit_attn=use_mask_unit_attn or (previous_stage_used_masked_attention and i == 0),
                )
                for i in range(depth)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class HieraBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = HieraEmbeddings(config)
        self.unroll_schedule = [config.query_stride] * len(config.depths[:-1])

        self.pool = GlobalAvgPool2d()
        self.pooler = nn.Module()
        self.pooler.layernorm = LayerNorm(int(config.embed_dim * config.embed_dim_multiplier ** (len(config.depths) - 1)), eps=config.layer_norm_eps)
        total_depth = sum(config.depths)
        cumulative_depths = torch.tensor(config.depths, device="cpu").cumsum(0).tolist()
        query_pool_layer = cumulative_depths[: config.num_query_pool]
        query_strides = [math.prod(config.query_stride) if i in query_pool_layer else 1 for i in range(total_depth)]

        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList()
        hidden_size = int(config.embed_dim)
        stage_ends = [0] + cumulative_depths
        masked_unit_area = math.prod(config.masked_unit_size)
        query_stride_area = math.prod(config.query_stride)
        for idx_stage, depth in enumerate(config.depths):
            hidden_size_output = int(config.embed_dim * config.embed_dim_multiplier ** idx_stage)
            self.encoder.stages.append(
                HieraStage(
                    config=config,
                    depth=depth,
                    hidden_size=hidden_size,
                    hidden_size_output=hidden_size_output,
                    num_heads=config.num_heads[idx_stage],
                    query_stride=query_strides[stage_ends[idx_stage] : stage_ends[idx_stage + 1]],
                    window_size=int(masked_unit_area * query_stride_area ** -idx_stage),
                    use_mask_unit_attn=config.masked_unit_attention[idx_stage],
                    stage_num=idx_stage,
                )
            )
            hidden_size = hidden_size_output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        hidden_states = self.embeddings(pixel_values)
        hidden_states = _unroll(
            hidden_states,
            image_shape=(height, width),
            patch_stride=self.config.patch_stride,
            schedule=self.unroll_schedule,
        )
        for stage in self.encoder.stages:
            hidden_states = stage(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler.layernorm(
            self.pool(hidden_states.transpose(1, 2).unsqueeze(-1)))}



def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu":
        raise ValueError("Preserve the default GELU MLP")
    return HieraBackbone(config).to(device=device, dtype=dtype).eval()

def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)

def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
