"""SEWModel with its convolutional frontend and squeezed waveform encoder."""

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer

from .unispeech import GroupFeatureConv
from .wav2vec2 import PositionConv, make_workloads


class Upsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor = config.squeeze_factor
        self.projection = Linear(config.hidden_size, config.hidden_size * self.factor)
        self.activation = GELU()

    def forward(self, hidden_states):
        batch, length, width = hidden_states.shape
        return self.activation(self.projection(hidden_states)).reshape(batch, length * self.factor, width)


class SqueezedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_conv_embed = PositionConv(config)
        self.pos_conv_embed.conv.stride = (config.squeeze_factor,)
        self.pool = AvgPool2d((1, config.squeeze_factor))
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList(BertLayer(config) for _ in range(config.num_hidden_layers))
        self.upsample = Upsampling(config)

    def squeeze(self, hidden_states):
        positions = self.pos_conv_embed(hidden_states)
        pooled = self.pool(hidden_states.transpose(1, 2).unsqueeze(2)).squeeze(2).transpose(1, 2)
        length = min(pooled.shape[1], positions.shape[1])
        return pooled[:, :length] + positions[:, :length]

    def forward(self, hidden_states):
        original_length = hidden_states.shape[1]
        hidden_states = self.layer_norm(self.squeeze(hidden_states))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.upsample(hidden_states)
        if hidden_states.shape[1] < original_length:
            hidden_states = F.pad(hidden_states, (0, 0, 0, original_length - hidden_states.shape[1]))
        return hidden_states


class SqueezedWaveformModel(nn.Module):
    def __init__(self, config, encoder, feature_norm_eps):
        super().__init__()
        self.feature_extractor = nn.ModuleList(
            GroupFeatureConv(config, index) for index in range(len(config.conv_dim))
        )
        self.layer_norm = LayerNorm(config.conv_dim[-1], eps=feature_norm_eps, promote_fp32=False)
        if config.conv_dim[-1] != config.hidden_size:
            self.feature_projection = Linear(config.conv_dim[-1], config.hidden_size)
        if config.mask_time_prob > 0 or config.mask_feature_prob > 0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))
        self.encoder = encoder

    def forward(self, input_values):
        if self.training or input_values.ndim != 2:
            raise ValueError("SEW coverage expects unmasked inference on [batch, waveform samples]")
        hidden_states = input_values[:, None]
        for layer in self.feature_extractor:
            hidden_states = layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states.transpose(1, 2))
        if hasattr(self, "feature_projection"):
            hidden_states = self.feature_projection(hidden_states)
        return {"last_hidden_state": self.encoder(hidden_states)}


def check_frontend(config):
    if (config.feat_extract_norm != "group" or config.feat_extract_activation != "gelu"
            or config.squeeze_factor != 2 or config.num_conv_pos_embeddings != 128
            or config.num_conv_pos_embedding_groups != 16
            or tuple(config.conv_kernel) != (10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)
            or tuple(config.conv_stride) != (5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)
            or config.output_attentions or config.output_hidden_states):
        raise ValueError("This case preserves SEW's default frontend and final unmasked output")


def build_from_config(config, device, dtype):
    check_frontend(config)
    if config.hidden_act != "gelu":
        raise ValueError("SEW's default transformer uses exact GELU")
    model = SqueezedWaveformModel(config, SqueezedEncoder(config), config.layer_norm_eps)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    remaining = dict(state_dict)
    prefix = "encoder.pos_conv_embed.conv."
    device = model.encoder.pos_conv_embed.conv.weight.device
    weight_g = remaining.pop(prefix + "parametrizations.weight.original0").to(device=device)
    weight_v = remaining.pop(prefix + "parametrizations.weight.original1").to(device=device)
    remaining[prefix + "weight"] = torch._weight_norm(weight_v, weight_g, 2)
    mapped = {}
    for destination in model.state_dict():
        source = destination.replace("feature_extractor.", "feature_extractor.conv_layers.")
        source = source.replace(".emb.weight", ".weight")
        if source.startswith("encoder.layers."):
            if ".attention.self.qkv." in source:
                mapped[destination] = torch.cat([
                    remaining.pop(source.replace(".attention.self.qkv.", f".attention.{part}_proj."))
                    for part in ("q", "k", "v")
                ])
                continue
            for target, origin in (
                ("attention.output.dense", "attention.out_proj"),
                ("attention.output.LayerNorm", "layer_norm"),
                ("intermediate.dense", "feed_forward.intermediate_dense"),
                ("output.dense", "feed_forward.output_dense"),
                ("output.LayerNorm", "final_layer_norm"),
            ):
                source = source.replace("." + target + ".", "." + origin + ".")
        mapped[destination] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped squeezed waveform state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
