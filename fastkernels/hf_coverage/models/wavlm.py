"""WavLMModel with its default activation-dependent relative attention bias."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention

from ..patches.wavlm import WavLMGatedPositionBias
from .mbart import PreNormEncoderAttention
from .unispeech import GroupFeatureConv
from .wav2vec2 import FeatureProjection, PositionConv, make_workloads


class RelativeAttention(nn.Module):
    def __init__(self, config, first_layer):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_bucket_distance
        self.attention = VitEncoderAttention(config.hidden_size, self.heads, qkv_bias=True)
        self.gru_rel_pos_linear = Linear(self.head_dim, 8)
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.heads, 1, 1))
        self.reduce_four = AvgPool2d(kernel_size=(1, 4))
        self.gated_bias = WavLMGatedPositionBias()
        if first_layer:
            self.rel_attn_embed = Embedding(config.num_buckets, self.heads)

    def compute_bias(self, length, device):
        # Only integer position metadata is computed here. Keep HF's FP32
        # logarithmic bucket calculation and cast order, including its cap.
        positions = torch.arange(length, dtype=torch.long)
        relative = positions[None, :] - positions[:, None]
        half = self.num_buckets // 2
        buckets = (relative > 0).long() * half
        distance = relative.abs()
        exact = half // 2
        large = torch.log(distance.float() / exact)
        large = large / math.log(self.max_distance / exact) * (half - exact)
        large = (exact + large).long().clamp(max=half - 1)
        buckets = buckets + torch.where(distance < exact, distance, large)
        return self.rel_attn_embed(buckets.to(device)).permute(2, 0, 1)

    def forward(self, hidden_states, position_bias):
        batch, length, _ = hidden_states.shape
        if position_bias is None:
            position_bias = self.compute_bias(length, hidden_states.device)
            position_bias = position_bias.unsqueeze(0).repeat(batch, 1, 1, 1)
            position_bias = position_bias.reshape(batch * self.heads, length, length)
        per_head = hidden_states.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        projected = self.gru_rel_pos_linear(per_head)
        # Sum after the projection, preserving its rounded eight values.
        sums = self.reduce_four(projected.reshape(-1, 1, 2, 4)) * 4
        sums = sums.reshape(batch, self.heads, length, 2)
        bias = self.gated_bias(sums, self.gru_rel_pos_const, position_bias)
        output = self.attention(hidden_states, bias.view(batch, self.heads, length, length))
        return output, position_bias


class EncoderLayer(nn.Module):
    def __init__(self, config, first_layer):
        super().__init__()
        self.attention = RelativeAttention(config, first_layer)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)

    def forward(self, hidden_states, position_bias):
        attended, position_bias = self.attention(hidden_states, position_bias)
        hidden_states = self.layer_norm(hidden_states + attended)
        hidden_states = self.output(self.intermediate(hidden_states), hidden_states)
        return hidden_states, position_bias


class WavLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = nn.ModuleList(
            GroupFeatureConv(config, index) for index in range(len(config.conv_dim))
        )
        self.feature_projection = FeatureProjection(config)
        self.position_conv = PositionConv(config)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList(
            EncoderLayer(config, index == 0) for index in range(config.num_hidden_layers)
        )
        if config.mask_time_prob > 0 or config.mask_feature_prob > 0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, input_values):
        if self.training:
            raise RuntimeError("WavLM coverage supports unmasked inference only")
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, waveform samples]")
        hidden_states = input_values[:, None]
        for layer in self.feature_extractor:
            hidden_states = layer(hidden_states)
        hidden_states, features = self.feature_projection(hidden_states.transpose(1, 2))
        hidden_states = self.layer_norm(hidden_states + self.position_conv(hidden_states))
        position_bias = None
        for layer in self.layers:
            hidden_states, position_bias = layer(hidden_states, position_bias)
        return {"last_hidden_state": hidden_states, "extract_features": features}


def build_from_config(config, device, dtype):
    if (config.feat_extract_norm != "group" or config.do_stable_layer_norm
            or config.hidden_act != "gelu" or config.feat_extract_activation != "gelu"
            or config.add_adapter):
        raise ValueError("This case requires WavLM's default group-norm frontend and post-norm encoder")
    if config.output_attentions or config.output_hidden_states:
        raise ValueError("This case returns the ordinary final model outputs")
    model = WavLMModel(config)
    # Reuse the projection/attention composition that explicitly selects
    # cuDNN, preserving native HF dispatch despite optional dependency imports.
    for layer in model.layers:
        layer.attention.attention = PreNormEncoderAttention(layer.attention.attention)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination, source):
        mapped[destination] = remaining.pop(source)

    if hasattr(model, "masked_spec_embed"):
        copy("masked_spec_embed", "masked_spec_embed")
    for index, layer in enumerate(model.feature_extractor):
        for name in layer.state_dict():
            copy(f"feature_extractor.{index}.{name}", f"feature_extractor.conv_layers.{index}.{name}")
    for name in model.feature_projection.state_dict():
        copy(f"feature_projection.{name}", f"feature_projection.{name}")
    for field in ("weight", "bias"):
        copy(f"layer_norm.{field}", f"encoder.layer_norm.{field}")
    copy("position_conv.conv.bias", "encoder.pos_conv_embed.conv.bias")
    prefix = "encoder.pos_conv_embed.conv.parametrizations.weight."
    device = model.position_conv.conv.weight.device
    weight_g = remaining.pop(prefix + "original0").to(device=device)
    weight_v = remaining.pop(prefix + "original1").to(device=device)
    mapped["position_conv.conv.weight"] = torch._weight_norm(weight_v, weight_g, 2)

    for index in range(len(model.layers)):
        destination, source = f"layers.{index}.", f"encoder.layers.{index}."
        for field in ("weight", "bias"):
            mapped[destination + f"attention.attention.qkv.{field}"] = torch.cat([
                remaining.pop(source + f"attention.{projection}_proj.{field}")
                for projection in ("q", "k", "v")
            ], dim=0)
            copy(destination + f"attention.gru_rel_pos_linear.{field}", source + f"attention.gru_rel_pos_linear.{field}")
        copy(destination + "attention.gru_rel_pos_const", source + "attention.gru_rel_pos_const")
        if index == 0:
            copy(destination + "attention.rel_attn_embed.emb.weight", source + "attention.rel_attn_embed.weight")
        for target, origin in (
            ("attention.attention.proj", "attention.out_proj"),
            ("layer_norm", "layer_norm"),
            ("intermediate.dense", "feed_forward.intermediate_dense"),
            ("output.dense", "feed_forward.output_dense"),
            ("output.LayerNorm", "final_layer_norm"),
        ):
            for field in ("weight", "bias"):
                copy(destination + f"{target}.{field}", source + f"{origin}.{field}")
    if remaining:
        raise KeyError(f"Unmapped WavLM state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
