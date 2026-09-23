"""Wav2Vec2Model's documented large encoder, built from existing operations."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from ..runner import Workload


class FeatureConv(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        channels = config.conv_dim[index]
        self.conv = Conv1dNative(
            config.conv_dim[index - 1] if index else 1,
            channels,
            config.conv_kernel[index],
            stride=config.conv_stride[index],
            bias=config.conv_bias,
        )
        # HF's convolutional LayerNorm uses its own default epsilon.
        self.layer_norm = LayerNorm(channels, eps=1e-5, promote_fp32=False)
        self.activation = GELU()

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states).transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states).transpose(1, 2)
        return self.activation(hidden_states)


class FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = LayerNorm(
            config.conv_dim[-1], eps=config.layer_norm_eps, promote_fp32=False
        )
        self.projection = Linear(config.conv_dim[-1], config.hidden_size)

    def forward(self, hidden_states):
        features = self.layer_norm(hidden_states)
        return self.projection(features), features


class PositionConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel = config.num_conv_pos_embeddings
        self.conv = Conv1dNative(
            config.hidden_size, config.hidden_size, kernel,
            padding=kernel // 2, groups=config.num_conv_pos_embedding_groups,
        )
        self.remove_last = kernel % 2 == 0
        self.activation = GELU()

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states.transpose(1, 2))
        if self.remove_last:
            hidden_states = hidden_states[:, :, :-1]
        return self.activation(hidden_states).transpose(1, 2)


class WaveformModel(nn.Module):
    """Shared layer-normalized frontend and pre-normalized waveform encoder."""

    def __init__(self, config, return_features):
        super().__init__()
        self.return_features = return_features
        self.feature_extractor = nn.ModuleList(
            FeatureConv(config, index) for index in range(len(config.conv_dim))
        )
        self.feature_projection = FeatureProjection(config)
        self.position_conv = PositionConv(config)
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            block = VitEncoderBlock(
                dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                mlp_ratio=config.intermediate_size / config.hidden_size,
                qkv_bias=True, proj_bias=True, act_approximate="none",
                norm_eps=config.layer_norm_eps,
            )
            if block.mlp.fc1.weight.shape[0] != config.intermediate_size:
                block.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size)
            self.layers.append(block)
        self.layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )
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
        hidden_states = hidden_states + self.position_conv(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        outputs = {"last_hidden_state": self.layer_norm(hidden_states)}
        if self.return_features:
            outputs["extract_features"] = features
        return outputs


def build_waveform_model(config, device, dtype, *, return_features):
    if (config.feat_extract_norm != "layer" or not config.do_stable_layer_norm
            or config.feat_extract_activation != "gelu" or config.hidden_act != "gelu"
            or getattr(config, "add_adapter", False)
            or getattr(config, "adapter_attn_dim", None) is not None
            or getattr(config, "conv_pos_batch_norm", False)
            or not getattr(config, "feat_proj_layer_norm", True)):
        raise ValueError("This case requires the documented large model's normalized convolution and pre-norm encoder")
    if getattr(config, "output_attentions", False) or getattr(config, "output_hidden_states", False):
        raise ValueError("This case returns the ordinary final model outputs")
    return WaveformModel(config, return_features).to(device=device, dtype=dtype).eval()


def build_from_config(config, device, dtype):
    return build_waveform_model(config, device, dtype, return_features=True)


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

    # In inference these weights are fixed. Fold precisely the same PyTorch
    # parametrization on the execution device, preserving the stored dtypes.
    prefix = "encoder.pos_conv_embed.conv.parametrizations.weight."
    device = model.position_conv.conv.weight.device
    weight_g = remaining.pop(prefix + "original0").to(device=device)
    weight_v = remaining.pop(prefix + "original1").to(device=device)
    mapped["position_conv.conv.weight"] = torch._weight_norm(weight_v, weight_g, 2)

    for index, layer in enumerate(model.layers):
        destination, source = f"layers.{index}.", f"encoder.layers.{index}."
        for field in ("weight", "bias"):
            mapped[destination + f"attn.qkv.{field}"] = torch.cat([
                remaining.pop(source + f"attention.{projection}_proj.{field}")
                for projection in ("q", "k", "v")
            ], dim=0)
        for target, origin in (
            ("attn.proj", "attention.out_proj"),
            ("mlp.fc1", "feed_forward.intermediate_dense"),
            ("mlp.fc2", "feed_forward.output_dense"),
            ("norm1", "layer_norm"),
            ("norm2", "final_layer_norm"),
        ):
            for field in ("weight", "bias"):
                copy(destination + f"{target}.{field}", source + f"{origin}.{field}")
    if remaining:
        raise KeyError(f"Unmapped waveform model state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    del config
    if set(inputs) != {"input_values"}:
        raise ValueError("Default waveform inference expects only input_values")
    return {"forward": Workload(run=lambda: model(inputs["input_values"]))}
