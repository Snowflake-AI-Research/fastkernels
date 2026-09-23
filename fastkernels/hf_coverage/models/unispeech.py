"""UniSpeechModel's public default waveform encoder and both final outputs."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer

from .wav2vec2 import FeatureProjection, PositionConv, make_workloads


class GroupFeatureConv(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        channels = config.conv_dim[index]
        self.conv = Conv1dNative(
            config.conv_dim[index - 1] if index else 1,
            channels, config.conv_kernel[index],
            stride=config.conv_stride[index], bias=config.conv_bias,
        )
        self.layer_norm = GroupNorm(channels, channels, eps=1e-5) if index == 0 else nn.Identity()
        self.activation = GELU()

    def forward(self, hidden_states):
        return self.activation(self.layer_norm(self.conv(hidden_states)))


class PostNormWaveformModel(nn.Module):
    def __init__(self, config, feature_layers, position_conv):
        super().__init__()
        self.feature_extractor = nn.ModuleList(feature_layers)
        self.feature_projection = FeatureProjection(config)
        self.position_conv = position_conv
        self.layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )
        self.layers = nn.ModuleList(BertLayer(config) for _ in range(config.num_hidden_layers))
        if config.mask_time_prob > 0 or config.mask_feature_prob > 0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, input_values):
        if self.training:
            raise RuntimeError("Waveform coverage supports unmasked inference only")
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, waveform samples]")
        hidden_states = input_values[:, None]
        for layer in self.feature_extractor:
            hidden_states = layer(hidden_states)
        hidden_states, features = self.feature_projection(hidden_states.transpose(1, 2))
        hidden_states = self.layer_norm(hidden_states + self.position_conv(hidden_states))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states, "extract_features": features}


def build_from_config(config, device, dtype):
    if (config.feat_extract_norm != "group" or config.do_stable_layer_norm
            or config.feat_extract_activation != "gelu" or config.hidden_act != "gelu"):
        raise ValueError("This case requires the public default group-norm frontend and post-norm encoder")
    if config.output_attentions or config.output_hidden_states:
        raise ValueError("This case returns the ordinary final model outputs")
    model = PostNormWaveformModel(
        config,
        [GroupFeatureConv(config, index) for index in range(len(config.conv_dim))],
        PositionConv(config),
    )
    # UniSpeech-SAT owns the mask parameter even if its masking probabilities
    # are zero; ordinary inference never applies it without explicit indices.
    if config.model_type == "unispeech-sat" and not hasattr(model, "masked_spec_embed"):
        model.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))
    # Optional varlen imports disable global cuDNN dispatch. Select the
    # existing native backend locally for UniSpeech and UniSpeech-SAT.
    for layer in model.layers:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
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

    if isinstance(model.position_conv, PositionConv):
        copy("position_conv.conv.bias", "encoder.pos_conv_embed.conv.bias")
        prefix = "encoder.pos_conv_embed.conv.parametrizations.weight."
        device = model.position_conv.conv.weight.device
        weight_g = remaining.pop(prefix + "original0").to(device=device)
        weight_v = remaining.pop(prefix + "original1").to(device=device)
        mapped["position_conv.conv.weight"] = torch._weight_norm(weight_v, weight_g, 2)
    else:
        for name in model.position_conv.state_dict():
            copy(f"position_conv.{name}", f"encoder.pos_conv_embed.{name}")

    for index in range(len(model.layers)):
        destination, source = f"layers.{index}.", f"encoder.layers.{index}."
        for field in ("weight", "bias"):
            mapped[destination + f"attention.self.qkv.{field}"] = torch.cat([
                remaining.pop(source + f"attention.{projection}_proj.{field}")
                for projection in ("q", "k", "v")
            ], dim=0)
        for target, origin in (
            ("attention.output.dense", "attention.out_proj"),
            ("attention.output.LayerNorm", "layer_norm"),
            ("intermediate.dense", "feed_forward.intermediate_dense"),
            ("output.dense", "feed_forward.output_dense"),
            ("output.LayerNorm", "final_layer_norm"),
        ):
            for field in ("weight", "bias"):
                copy(destination + f"{target}.{field}", source + f"{origin}.{field}")
    if remaining:
        raise KeyError(f"Unmapped post-norm waveform state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
