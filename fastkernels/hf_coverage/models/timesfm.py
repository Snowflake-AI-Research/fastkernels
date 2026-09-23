"""TimesFM mean/quantile forecasts with selected-patch normalization."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.patches.forecast_masked_stats import MaskedPatchStats
from fastkernels.hf_coverage.patches.forecast_revin import ForecastNormalize
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.silu import SiLU


class ResidualBlock(nn.Module):
    def __init__(self, input_width, hidden_width, output_width):
        super().__init__()
        self.input_layer = Linear(input_width, hidden_width)
        self.output_layer = Linear(hidden_width, output_width)
        self.residual_layer = Linear(input_width, output_width)
        self.activation = SiLU()

    def forward(self, hidden):
        return self.output_layer(self.activation(self.input_layer(hidden))) + self.residual_layer(hidden)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        self.input_layernorm = RMSNormNative(width, config.rms_norm_eps)
        self.self_attn = nn.Module()
        self.self_attn.scaling = nn.Parameter(torch.empty(config.head_dim))
        self.register_buffer("query_scale", torch.empty(config.head_dim), persistent=False)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, Linear(width, width))
        self.mlp = nn.Module()
        self.mlp.layer_norm = LayerNorm(width, eps=1e-6, promote_fp32=False)
        self.mlp.gate_proj = Linear(width, config.intermediate_size)
        self.mlp.down_proj = Linear(config.intermediate_size, width)
        self.attention = DenseAttention(backend="cudnn")
        self.product, self.relu = ProductGate(), ReLU()

    def forward(self, hidden, mask, padding):
        norm = self.input_layernorm(hidden)
        batch, length, width = hidden.shape
        query, key, value = (getattr(self.self_attn, name)(norm).reshape(batch, length, self.heads, self.head_dim)
                             for name in ("q_proj", "k_proj", "v_proj"))
        scale = self.query_scale.expand_as(query)
        query = self.product(torch.cat((query, scale), dim=-1))
        # The native default selects cuDNN SDPA, including its zero output for
        # fully padded queries. Reuse that existing attention operation.
        context = self.attention(query, key, value, softmax_scale=1., attn_mask=mask).reshape(batch, length, width)
        hidden = hidden + self.self_attn.o_proj(context)
        projected = self.mlp.down_proj(self.relu(self.mlp.gate_proj(self.mlp.layer_norm(hidden))))
        projected = projected.masked_fill(padding[:, :, None], 0.)
        return hidden + projected


class TimesFmModelForPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = nn.Module()
        self.decoder.input_ff_layer = ResidualBlock(2 * config.patch_length, config.intermediate_size, config.hidden_size)
        self.decoder.freq_emb = Embedding(config.freq_size, config.hidden_size)
        self.decoder.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.horizon_ff_layer = ResidualBlock(config.hidden_size, config.intermediate_size,
                                             config.horizon_length * (1 + len(config.quantiles)))
        self.stats = MaskedPatchStats(config.tolerance)
        # Statistics already clamp std to the native tolerance. Preserve the
        # native subtract store before dividing by the rounded standard deviation.
        self.normalization = ForecastNormalize(tolerance=0.)
        self.product = ProductGate()
        self.padding_relu, self.padding_top1, self.padding_reduce = ReLU(), CodecTop1(), SegmentCSR()

    def prepare_padding(self, patched, padding):
        # Sentinel detection depends on series values, so use numerical
        # operations before treating the resulting discrete indices as masks.
        difference = patched - self.config.pad_val
        magnitude = self.padding_relu(difference) + self.padding_relu(-difference)
        tolerance = torch.full_like(magnitude, self.config.tolerance)
        # First-index ties implement strict magnitude < tolerance.
        sentinel = self.padding_top1(torch.stack((magnitude, tolerance), dim=-1)).bool()
        padding = padding.masked_fill(sentinel, 1.)
        width = patched.shape[-1]
        offsets = torch.arange(0, patched.numel() + 1, width, device=patched.device)
        counts = self.padding_reduce((1. - padding.float()).flatten(), offsets, reduce="sum")
        counts = counts.reshape(patched.shape[:-1])
        # A tie at three valid values is eligible; ties among eligible patches
        # select the first. If none qualifies, native uses the last patch.
        eligible = ~self.padding_top1(torch.stack((counts, torch.full_like(counts, 3.)), dim=-1)).bool()
        indices = self.padding_top1(eligible.float())
        row_offsets = torch.arange(0, eligible.numel() + 1, eligible.shape[-1], device=patched.device)
        any_eligible = self.padding_reduce(eligible.float().flatten(), row_offsets, reduce="max")
        has_eligible = self.padding_top1(torch.stack((torch.zeros_like(any_eligible), any_eligible), dim=-1)).bool()
        indices = torch.where(has_eligible, indices, indices.new_full(indices.shape, patched.shape[1] - 1))
        patch_padding = ~self.padding_top1(torch.stack((torch.zeros_like(counts), counts), dim=-1)).bool()
        return padding, indices, patch_padding

    def forward(self, past_values, freq=None):
        config = self.config
        series = [row[-config.context_length:] for row in past_values]
        values, pads = [], []
        for row in series:
            missing = config.context_length - row.shape[0]
            values.append(torch.cat((row.new_zeros(missing), row)))
            pads.append(torch.cat((row.new_ones(missing), row.new_zeros(row.shape[0]))))
        values = torch.stack(values)
        padding = torch.stack(pads).reshape(len(series), -1, config.patch_length)
        patched = values.reshape_as(padding)
        padding, indices, patch_padding = self.prepare_padding(patched, padding)
        rows = torch.arange(len(series), device=values.device)
        mean, _, std = self.stats(patched[rows, indices], padding[rows, indices])
        normalized = self.normalization(values, mean, std).reshape_as(patched)
        normalized = normalized.masked_fill(padding.bool(), 0.)
        hidden = self.decoder.input_ff_layer(torch.cat((normalized, padding), dim=-1))
        if freq is None:
            frequency = torch.zeros(len(series), 1, dtype=torch.int32, device=values.device)
        else:
            frequency = torch.as_tensor(freq, device=values.device, dtype=torch.int32).reshape(-1, 1)
        hidden = hidden + self.decoder.freq_emb(frequency)
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        mask = torch.zeros(len(series), 1, length, length, dtype=hidden.dtype, device=hidden.device)
        mask.masked_fill_(patch_padding[:, None, None, :] | (positions[None, :] > positions[:, None]), torch.finfo(hidden.dtype).min)
        for layer in self.decoder.layers:
            hidden = layer(hidden, mask, patch_padding)
        forecast = self.horizon_ff_layer(hidden).reshape(len(series), length, config.horizon_length, -1)
        scale = std[:, None, None, None].expand_as(forecast)
        forecast = self.product(torch.cat((forecast, scale), dim=-1)) + mean[:, None, None, None]
        forecast = forecast[:, -1]
        return {"mean_predictions": forecast[..., 0], "full_predictions": forecast, "last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.use_positional_embedding or config.hidden_size != config.num_attention_heads * config.head_dim:
        raise ValueError("Expected the documented TimesFM2.0 checkpoint computation")
    return TimesFmModelForPrediction(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = dict(state_dict)
    weights["decoder.freq_emb.emb.weight"] = weights.pop("decoder.freq_emb.weight")
    model.load_state_dict(weights, strict=True)
    with torch.no_grad():
        for layer in model.decoder.layers:
            # Inference-constant learned per-channel scale; preserve native dtype
            # softplus and scalar multiply before storing it for forward reuse.
            layer.query_scale.copy_(torch.nn.functional.softplus(layer.self_attn.scaling) * (1.442695041 / math.sqrt(config.head_dim)))


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
