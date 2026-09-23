"""PatchTSMixer's default unmasked base model, including scaling and gates."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.forecast_std_scaler import UnmaskedStdScaler
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax


class Mixer(nn.Module):
    def __init__(self, config, patches):
        super().__init__()
        self.patches = patches
        width = config.num_patches if patches else config.d_model
        self.norm = nn.Module()
        self.norm.norm = LayerNorm(config.d_model, eps=config.norm_eps, promote_fp32=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = Linear(width, width * config.expansion_factor)
        self.mlp.fc2 = Linear(width * config.expansion_factor, width)
        self.gating_block = nn.Module()
        self.gating_block.attn_layer = Linear(width, width)
        self.gelu, self.softmax, self.product = GELU(), Softmax(dim=-1), ProductGate()

    def forward(self, hidden):
        residual = hidden
        hidden = self.norm.norm(hidden)
        if self.patches:
            hidden = hidden.transpose(2, 3)
        hidden = self.mlp.fc2(self.gelu(self.mlp.fc1(hidden)))
        weights = self.softmax(self.gating_block.attn_layer(hidden))
        hidden = self.product(torch.cat((hidden, weights), dim=-1))
        if self.patches:
            hidden = hidden.transpose(2, 3)
        return hidden + residual


class PatchTSMixerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_length, self.patch_stride = config.patch_length, config.patch_stride
        self.start = config.context_length - config.patch_length - config.patch_stride * (config.num_patches - 1)
        self.scaler = UnmaskedStdScaler(getattr(config, "minimum_scale", 1e-5))
        self.encoder = nn.Module()
        self.encoder.patcher = Linear(config.patch_length, config.d_model)
        self.encoder.mlp_mixer_encoder = nn.Module()
        layers = self.encoder.mlp_mixer_encoder.mixers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = nn.Module()
            layer.patch_mixer, layer.feature_mixer = Mixer(config, True), Mixer(config, False)
            layers.append(layer)

    def forward(self, past_values):
        scaled, loc, scale = self.scaler(past_values)
        patches = scaled[:, self.start:].unfold(1, self.patch_length, self.patch_stride).transpose(1, 2).contiguous()
        hidden = self.encoder.patcher(patches)
        for layer in self.encoder.mlp_mixer_encoder.mixers:
            hidden = layer.feature_mixer(layer.patch_mixer(hidden))
        return {"last_hidden_state": hidden, "patch_input": patches, "loc": loc, "scale": scale}


def build_from_config(config, device, dtype):
    if not (config.scaling in ("std", True) and config.mode == "common_channel" and config.gated_attn
            and not config.self_attn and not config.use_positional_encoding and "batch" not in config.norm_mlp.lower()):
        raise ValueError("Expected the documented PatchTSMixer base-model checkpoint computation")
    return PatchTSMixerModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
