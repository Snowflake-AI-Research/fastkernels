"""PatchTST's unmasked base encoder with the documented checkpoint settings."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.mvp import EagerAttention
from fastkernels.hf_coverage.patches.forecast_std_scaler import UnmaskedStdScaler
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear


class TimeBatchNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batchnorm = BatchNorm2d(config.d_model, eps=config.norm_eps)

    def forward(self, hidden):
        return self.batchnorm(hidden.transpose(1, 2)).transpose(1, 2)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, Linear(config.d_model, config.d_model))
        self.attention = EagerAttention(prescale_query=False)
        self.norm_sublayer1, self.norm_sublayer3 = TimeBatchNorm(config), TimeBatchNorm(config)
        self.ff = nn.Sequential(Linear(config.d_model, config.ffn_dim, bias=config.bias), GELU(),
                                nn.Identity(), Linear(config.ffn_dim, config.d_model, bias=config.bias))

    def forward(self, hidden):
        batch, channels, length, width = hidden.shape
        hidden = hidden.reshape(batch * channels, length, width)
        norm = self.norm_sublayer1(hidden)
        query, key, value = (getattr(self.self_attn, name)(norm).reshape(batch * channels, length, self.heads, -1)
                             for name in ("q_proj", "k_proj", "v_proj"))
        context = self.attention(query, key, value).reshape(batch * channels, length, width)
        hidden = hidden + self.self_attn.out_proj(context)
        hidden = hidden + self.ff(self.norm_sublayer3(hidden))
        return hidden.reshape(batch, channels, length, width)


class PatchTSTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_length, self.patch_stride = config.patch_length, config.patch_stride
        patches = (config.context_length - config.patch_length) // config.patch_stride + 1
        self.start = config.context_length - config.patch_length - config.patch_stride * (patches - 1)
        self.scaler = UnmaskedStdScaler(getattr(config, "minimum_scale", 1e-5))
        self.encoder = nn.Module()
        self.encoder.embedder = nn.Module()
        self.encoder.embedder.input_embedding = Linear(config.patch_length, config.d_model)
        self.encoder.positional_encoder = nn.Module()
        position = self.encoder.positional_encoder
        position.cls_token = nn.Parameter(torch.empty(1, 1, 1, config.d_model))
        position.position_enc = nn.Parameter(torch.empty(patches + 1, config.d_model), requires_grad=False)
        self.encoder.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, past_values):
        scaled, loc, scale = self.scaler(past_values)
        patches = scaled[:, self.start:].unfold(1, self.patch_length, self.patch_stride).transpose(1, 2).contiguous()
        position = self.encoder.positional_encoder
        hidden = self.encoder.embedder.input_embedding(patches) + position.position_enc[1:]
        cls = (position.cls_token + position.position_enc[:1]).expand(hidden.shape[0], hidden.shape[1], -1, -1)
        hidden = torch.cat((cls, hidden), dim=2)
        for layer in self.encoder.layers:
            hidden = layer(hidden)
        return {"last_hidden_state": hidden, "patch_input": patches, "loc": loc, "scale": scale}


def build_from_config(config, device, dtype):
    if not (config.scaling == "std" and config.norm_type == "batchnorm" and config.pre_norm
            and not config.channel_attention and config.share_embedding and config.use_cls_token
            and not config.do_mask_input and config.activation_function == "gelu"):
        raise ValueError("Expected the documented PatchTST base-model checkpoint computation")
    return PatchTSTModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
