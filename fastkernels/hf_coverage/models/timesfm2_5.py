"""TimesFM2.5 forecasts, retaining running statistics and both forecast heads."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.patches.forecast_masked_stats import MaskedPatchStats
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.forecast_revin import ForecastNormalize, ZeroSafeVarianceNormalize
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.silu import SiLU


class ResidualBlock(nn.Module):
    def __init__(self, input_width, hidden_width, output_width, bias=False):
        super().__init__()
        self.input_layer = Linear(input_width, hidden_width, bias=bias)
        self.output_layer = Linear(hidden_width, output_width, bias=bias)
        self.residual_layer = Linear(input_width, output_width, bias=bias)
        self.activation = SiLU()

    def forward(self, values):
        values = values.to(self.input_layer.weight.dtype)
        return self.output_layer(self.activation(self.input_layer(values))) + self.residual_layer(values)


class RunningStats(nn.Module):
    """Centered variance updates preserving HF's intermediate dtype stores."""
    def __init__(self):
        super().__init__()
        self.stats, self.product = MaskedPatchStats(0.), ProductGate()
        self.sqrt = ZeroSafeVarianceNormalize()
        self.divide = ForecastNormalize()

    def multiply(self, left, right):
        return self.product(torch.stack((left, right), dim=-1)).squeeze(-1)

    def square(self, values):
        return self.multiply(values, values)

    def forward(self, patches, padding):
        count = patches.new_zeros(patches.shape[0])
        mean, std = count.clone(), count.clone()
        means, stds = [], []
        for step in range(patches.shape[1]):
            increment = (~padding[:, step]).sum(-1).to(patches.dtype)
            new_mean, _, new_std = self.stats(patches[:, step], padding[:, step].to(patches.dtype))
            total = count + increment
            zero = torch.zeros_like(total)
            numerator = self.multiply(count, mean) + self.multiply(increment, new_mean)
            merged_mean = self.divide(numerator[:, None], zero, total).flatten()
            # MergeState's fused weighted average is mathematically equivalent,
            # but skips these BF16 stores and fails the full-model comparison.
            term1 = self.multiply(count, self.square(std))
            term2 = self.multiply(increment, self.square(new_std))
            term3 = self.multiply(count, self.square(mean - merged_mean))
            term4 = self.multiply(increment, self.square(new_mean - merged_mean))
            numerator = term1 + term2 + term3 + term4
            variance = self.divide(numerator[:, None], zero, total).flatten()
            # Patched frozen normalization computes var/sqrt(var) in FP32,
            # regularizing only the removable zero/zero singularity.
            var32 = variance.float()
            std = self.sqrt(var32, torch.zeros_like(var32), var32).to(variance.dtype)
            count, mean = total, merged_mean
            means.append(mean)
            stds.append(std)
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.self_attn = nn.Module()
        self.self_attn.scaling = nn.Parameter(torch.empty(config.head_dim))
        self.register_buffer("query_scale", torch.empty(config.head_dim), persistent=False)
        for name in ("q_proj", "k_proj", "v_proj"):
            heads = self.heads if name == "q_proj" else self.kv_heads
            setattr(self.self_attn, name, Linear(width, heads * self.head_dim, bias=config.attention_bias))
        self.self_attn.o_proj = Linear(self.heads * self.head_dim, width, bias=config.attention_bias)
        self.self_attn.q_norm = RMSNormNative(self.head_dim, config.rms_norm_eps)
        self.self_attn.k_norm = RMSNormNative(self.head_dim, config.rms_norm_eps)
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            setattr(self, name, RMSNormNative(width, config.rms_norm_eps))
        self.mlp = nn.Module()
        self.mlp.fc1 = Linear(width, config.intermediate_size, bias=config.use_bias)
        self.mlp.fc2 = Linear(config.intermediate_size, width, bias=config.use_bias)
        self.activation, self.attention, self.product = SiLU(), DenseAttention(backend="cudnn"), ProductGate()

    def forward(self, hidden, positions, rotary_table, mask):
        batch, length, width = hidden.shape
        norm = self.input_layernorm(hidden)
        query = self.self_attn.q_proj(norm).reshape(batch * length, -1)
        key = self.self_attn.k_proj(norm).reshape(batch * length, -1)
        query, key = RotaryEmbedding.forward_native(positions.flatten(), query, key, self.head_dim, rotary_table)
        query = self.self_attn.q_norm(query.reshape(batch, length, self.heads, self.head_dim)).transpose(1, 2)
        key = self.self_attn.k_norm(key.reshape(batch, length, self.kv_heads, self.head_dim)).transpose(1, 2)
        value = self.self_attn.v_proj(norm).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query = self.product(torch.cat((query, self.query_scale.expand_as(query)), dim=-1))
        groups = self.heads // self.kv_heads
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
        context = self.attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                 softmax_scale=1., attn_mask=mask).reshape(batch, length, -1)
        hidden = hidden + self.post_attention_layernorm(self.self_attn.o_proj(context))
        projected = self.mlp.fc2(self.activation(self.mlp.fc1(self.pre_feedforward_layernorm(hidden))))
        return hidden + self.post_feedforward_layernorm(projected)


class TimesFm2_5ModelForPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.model = nn.Module()
        self.model.input_ff_layer = ResidualBlock(2 * config.patch_length, width, width, bias=True)
        self.model.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.output_projection_point = ResidualBlock(width, width, config.horizon_length * (1 + len(config.quantiles)))
        self.output_projection_quantiles = ResidualBlock(width, width, config.output_quantile_len * (1 + len(config.quantiles)))
        self.stats, self.global_stats = RunningStats(), MaskedPatchStats(0.)
        self.normalize, self.product, self.relu, self.minimum = ForecastNormalize(), ProductGate(), ReLU(), SegmentCSR()
        self.select = CodecTop1()
        patches = config.context_length // config.patch_length
        # HF initializes frequencies on CPU before moving them to the model's
        # device. GPU power evaluation changes some rounded rotary values.
        inverse = 1. / (config.rope_parameters["rope_theta"] ** (
            torch.arange(0, config.head_dim, 2, dtype=torch.float32, device="cpu") / config.head_dim))
        positions = torch.arange(-patches, patches, dtype=torch.float32)
        angles = positions[:, None] * inverse.to(positions.device)[None, :]
        self.register_buffer("rotary_table", torch.cat((angles.cos(), angles.sin()), dim=-1), persistent=False)

    def denormalize(self, values, mean, std):
        scale = std[..., None].expand_as(values)
        return self.product(torch.cat((values, scale), dim=-1)) + mean[..., None]

    def decode(self, values, padding):
        config = self.config
        batch = values.shape[0]
        patches = values.reshape(batch, -1, config.patch_length)
        padded = padding.reshape_as(patches)
        mean, std = self.stats(patches, padded)
        normalized = self.normalize(patches, mean, std).masked_fill(padded, 0.)
        hidden = self.model.input_ff_layer(torch.cat((normalized, padded.to(normalized.dtype)), dim=-1))
        patch_padding = padded[..., -1]
        length = hidden.shape[1]
        position = torch.arange(length, device=hidden.device)
        positions = position[None, :] - patch_padding.to(torch.int32).sum(-1, keepdim=True)
        # Native create_causal_mask supplies a Boolean allowed-position mask.
        # Preserve that representation for the native cuDNN attention path.
        mask = ~patch_padding[:, None, None, :] & (position[None, :] <= position[:, None])
        for layer in self.model.layers:
            hidden = layer(hidden, positions + length, self.rotary_table, mask)
        point = self.denormalize(self.output_projection_point(hidden), mean, std)
        quantiles = self.denormalize(self.output_projection_quantiles(hidden), mean, std)
        count = 1 + len(config.quantiles)
        return (point.reshape(batch, length, config.horizon_length, count)[:, -1],
                quantiles.reshape(batch, length, config.output_quantile_len, count)[:, -1], hidden)

    def forward(self, past_values):
        config = self.config
        series = [row[-config.context_length:] for row in past_values]
        joined = torch.cat(series)
        input_min = self.minimum(joined.float(), torch.tensor([0, joined.numel()], device=joined.device), reduce="min")
        values, masks = [], []
        for row in series:
            missing = config.context_length - row.shape[0]
            values.append(torch.cat((row.new_zeros(missing), row)))
            masks.append(torch.arange(config.context_length, device=row.device) < missing)
        values, padding = torch.stack(values), torch.stack(masks)
        mean, _, std = self.global_stats(values, torch.zeros_like(values))
        # Convert population to sample standard deviation; the divisor comes
        # from the fixed context length, not an activation-dependent estimate.
        std = std * math.sqrt(config.context_length / (config.context_length - 1))
        normalized = self.normalize(values, mean, std)
        point, quantiles, hidden = self.decode(normalized, padding)
        flipped_point, flipped_quantiles, _ = self.decode(-normalized, padding)
        def flip(values):
            return torch.cat((values[..., :1], values[..., 1:].flip(-1)), dim=-1)
        point, quantiles = (point - flip(flipped_point)) / 2, (quantiles - flip(flipped_quantiles)) / 2
        forecast = point.clone()
        median = config.decode_index
        horizon = min(config.horizon_length, config.output_quantile_len)
        for index in range(1, len(config.quantiles) + 1):
            if index != median:
                forecast[:, :horizon, index] = quantiles[:, :horizon, index] - quantiles[:, :horizon, median] + forecast[:, :horizon, median]
        forecast = self.denormalize(forecast, mean[:, None], std[:, None])
        # Existing top-1 selection uses index0 on ties: choose clipped outputs
        # exactly when the minimum input is nonnegative, including zero.
        choice = self.select(torch.stack((input_min, torch.zeros_like(input_min)), dim=-1))
        alternatives = torch.stack((self.relu(forecast), forecast), dim=0)
        forecast = alternatives.index_select(0, choice.reshape(-1).long()).squeeze(0)
        return {"mean_predictions": forecast[..., median], "full_predictions": forecast, "last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if not (config.force_flip_invariance and config.use_continuous_quantile_head and config.infer_is_positive
            and not config.use_bias and config.activation == "swish" and config.rope_parameters["rope_type"] == "default"):
        raise ValueError("Expected the documented TimesFM2.5 checkpoint computation")
    return TimesFm2_5ModelForPrediction(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.query_scale.copy_(torch.nn.functional.softplus(layer.self_attn.scaling) * (1.442695041 / math.sqrt(config.head_dim)))


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
