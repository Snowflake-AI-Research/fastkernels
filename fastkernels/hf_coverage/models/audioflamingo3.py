"""AudioFlamingo3's masked Whisper encoder, temporal pool and Qwen2 decoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.whisper import WhisperConfig, WhisperEncoder
from . import llama, qwen2
from .whisper import WhisperAttention
from ..runner import Workload


class AudioAttention(WhisperAttention):
    def forward(self, hidden):
        batch, length = hidden.shape[:2]
        query = (self.q_proj(hidden) * self.head_dim ** -0.5).view(batch, length, self.heads, self.head_dim)
        key = self.k_proj(hidden).view_as(query)
        value = self.v_proj(hidden).view_as(query)
        context = self.attention(query, key, value, softmax_scale=1.0, attn_mask=self.mask)
        return self.out_proj(context.reshape(batch, length, -1))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        carrier = WhisperConfig(d_model=config.hidden_size, num_mel_bins=config.num_mel_bins,
                                 max_source_positions=config.max_source_positions,
                                 encoder_layers=config.num_hidden_layers,
                                 encoder_attention_heads=config.num_attention_heads,
                                 encoder_ffn_dim=config.intermediate_size)
        self.encoder = WhisperEncoder(carrier)
        for layer in self.encoder.layers:
            layer.self_attn = AudioAttention(config.hidden_size, config.num_attention_heads)
        self.pool = AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, features, feature_mask):
        encoder = self.encoder
        hidden = encoder.gelu(encoder.conv1(features))
        hidden = encoder.gelu(encoder.conv2(hidden)).transpose(1, 2)
        hidden = hidden + encoder.embed_positions.emb.weight
        lengths = (feature_mask.sum(-1) - 1) // 2 + 1
        valid = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
        mask = valid[:, None, None, :].expand(-1, 1, hidden.shape[1], -1)
        for layer in encoder.layers:
            layer.self_attn.mask = mask
            hidden = layer(hidden)
        hidden = self.pool(hidden.transpose(1, 2).unsqueeze(2)).squeeze(2).transpose(1, 2)
        return encoder.layer_norm(hidden), lengths // 2


class AudioBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text, self.audio = text, AudioEncoder(config.audio_config)
        self.projector = nn.Sequential(Linear(config.audio_config.hidden_size, config.text_config.hidden_size,
                                               bias=config.projector_bias), GELU(),
                                       Linear(config.text_config.hidden_size, config.text_config.hidden_size,
                                               bias=config.projector_bias))
        self.audio_token_id = config.audio_token_id
        self.inputs = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, features, mask):
        hidden, lengths = self.audio(features, mask)
        projected = self.projector(hidden)
        valid = torch.arange(projected.shape[1], device=projected.device)[None] < lengths[:, None]
        return hidden, projected[valid]

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            _, features = self.features(self.inputs["input_features"], self.inputs["input_features_mask"])
            embeddings = embeddings.masked_scatter((input_ids == self.audio_token_id)[:, None].expand_as(embeddings), features)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class AudioFlamingoModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config, self.lm_head = text.config, text.lm_head
        self.model = AudioBackbone(text.model, config)


def build_from_config(config, device, dtype):
    if config.projector_hidden_act != "gelu" or config.audio_config.activation_function != "gelu" or config.text_config.use_cache:
        raise ValueError("The selected AudioFlamingo3 checkpoint uses GELU and disables language caching")
    language = qwen2.build_from_config(config.text_config, device, dtype)
    return AudioFlamingoModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.removeprefix("language_model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("language_model.")}
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    mapped = {}
    for name in model.model.audio.encoder.state_dict():
        source = name.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias").replace(".emb.weight", ".weight").replace(".mlp.fc", ".fc")
        mapped[name] = remaining.pop("audio_tower." + source)
    model.model.audio.encoder.load_state_dict(mapped, strict=True)
    model.model.projector.load_state_dict({name: remaining.pop("multi_modal_projector." + ("linear_1" if name[0] == "0" else "linear_2") + name[1:])
                                          for name in model.model.projector.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped AudioFlamingo3 state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    model.model.inputs = inputs
    return llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config, cached_decode=False)
