"""SEWDModel with squeezed audio and shared disentangled attention blocks."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm

from ..patches.gelu_python import PythonGELU
from .deberta_v2 import DebertaV2Layer
from .sew import SqueezedEncoder, SqueezedWaveformModel, Upsampling, check_frontend, load_state_dict_into
from .wav2vec2 import PositionConv, make_workloads


class RelativeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.buckets = config.position_buckets
        self.max_positions = config.max_position_embeddings
        self.rel_embeddings = Embedding(2 * self.buckets, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        attention_config = copy.copy(config)
        attention_config.attention_head_size = config.hidden_size // config.num_attention_heads
        self.layer = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = DebertaV2Layer(attention_config)
            layer.intermediate.intermediate_act_fn = PythonGELU()
            self.layer.append(layer)

    def forward(self, hidden_states):
        # Integer position metadata; the same logarithmic buckets as DeBERTa-v2.
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        relative = positions[:, None] - positions[None, :]
        mid = self.buckets // 2
        magnitude = torch.where((relative < mid) & (relative > -mid), mid - 1, relative.abs())
        denominator = torch.log(torch.tensor((self.max_positions - 1) / mid, device=hidden_states.device))
        log_position = torch.ceil(torch.log(magnitude / mid) / denominator * (mid - 1)) + mid
        bucket = torch.where(magnitude <= mid, relative, log_position * relative.sign()).long()
        c2p = (bucket + self.buckets).clamp(0, 2 * self.buckets - 1).unsqueeze(0)
        p2c = (-bucket + self.buckets).clamp(0, 2 * self.buckets - 1).unsqueeze(0)
        embeddings = self.LayerNorm(self.rel_embeddings.emb.weight).unsqueeze(0)
        for layer in self.layer:
            hidden_states = layer(hidden_states, embeddings, c2p, p2c)
        return hidden_states


class SqueezedRelativeEncoder(SqueezedEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.pos_conv_embed = PositionConv(config)
        self.pos_conv_embed.conv.stride = (config.squeeze_factor,)
        self.pool = AvgPool2d((1, config.squeeze_factor))
        self.encoder = RelativeEncoder(config)
        self.upsample = Upsampling(config)

    def forward(self, hidden_states):
        original_length = hidden_states.shape[1]
        hidden_states = self.upsample(self.encoder(self.squeeze(hidden_states)))
        if hidden_states.shape[1] < original_length:
            hidden_states = F.pad(hidden_states, (0, 0, 0, original_length - hidden_states.shape[1]))
        return hidden_states


def build_from_config(config, device, dtype):
    check_frontend(config)
    if (config.hidden_act != "gelu_python" or not config.relative_attention or not config.share_att_key
            or set(config.pos_att_type) != {"c2p", "p2c"} or config.norm_rel_ebd != "layer_norm"
            or config.position_buckets != 256 or config.max_position_embeddings != 512
            or getattr(config, "max_relative_positions", -1) != -1
            or getattr(config, "conv_kernel_size", 0) != 0
            or getattr(config, "attention_head_size", config.hidden_size // config.num_attention_heads)
            != config.hidden_size // config.num_attention_heads):
        raise ValueError("SEW-D coverage preserves its default relative attention and Python GELU")
    model = SqueezedWaveformModel(config, SqueezedRelativeEncoder(config), config.feature_layer_norm_eps)
    return model.to(device=device, dtype=dtype).eval()
