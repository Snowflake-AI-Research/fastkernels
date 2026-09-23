"""Qwen2-Audio's encoder, audio projection, and cached Qwen2 decoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.whisper import WhisperConfig, WhisperEncoder
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, qwen2
from ..runner import Workload


class AudioAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q_proj = Linear(width, width)
        self.k_proj = Linear(width, width, bias=False)
        self.v_proj = Linear(width, width)
        self.out_proj = Linear(width, width)
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden, mask):
        batch, length, width = hidden.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = (self.q_proj(hidden) * self.head_dim**-0.5).view(shape)
        key, value = self.k_proj(hidden).view(shape), self.v_proj(hidden).view(shape)
        context = self.attention(query, key, value, softmax_scale=1.0, attn_mask=mask)
        return self.out_proj(context.reshape(batch, length, width))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        carrier = WhisperConfig(
            **{
                name: getattr(config, name)
                for name in (
                    "d_model",
                    "encoder_layers",
                    "encoder_attention_heads",
                    "encoder_ffn_dim",
                    "num_mel_bins",
                    "max_source_positions",
                )
            }
        )
        self.encoder = WhisperEncoder(carrier)
        for layer in self.encoder.layers:
            layer.self_attn = AudioAttention(
                config.d_model, config.encoder_attention_heads
            )
        self.pool = AvgPool2d((1, 2), stride=(1, 2))

    def forward(self, features, feature_mask):
        encoder = self.encoder
        hidden = encoder.gelu(encoder.conv1(features))
        hidden = encoder.gelu(encoder.conv2(hidden)).transpose(1, 2)
        hidden = hidden + encoder.embed_positions.emb.weight
        lengths = (feature_mask.sum(-1) - 1) // 2 + 1
        mask = (
            torch.arange(hidden.shape[1], device=hidden.device)[None, :]
            < lengths[:, None]
        )
        mask = mask[:, None, None, :]
        for layer in encoder.layers:
            hidden = hidden + layer.self_attn(layer.self_attn_layer_norm(hidden), mask)
            hidden = hidden + layer.mlp(layer.final_layer_norm(hidden))
        hidden = (
            self.pool(hidden.transpose(1, 2).unsqueeze(2)).squeeze(2).transpose(1, 2)
        )
        return encoder.layer_norm(hidden), lengths // 2


class AudioBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.audio_tower = AudioEncoder(config.audio_config)
        self.projector = Linear(
            config.audio_config.d_model, config.text_config.hidden_size
        )
        self.audio_token_id = config.audio_token_index
        self.inputs = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        if not get_context().is_prefill:
            return self.text(input_ids, positions)
        embeddings = self.text.embed_tokens(input_ids)
        features, lengths = self.audio_tower(
            self.inputs["input_features"], self.inputs["feature_attention_mask"]
        )
        features = self.projector(features)
        valid = (
            torch.arange(features.shape[1], device=features.device)[None, :]
            < lengths[:, None]
        )
        embeddings[input_ids == self.audio_token_id] = features[valid]
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class QwenAudio(nn.Module):
    def __init__(self, native, config):
        super().__init__()
        self.config, self.lm_head = native.config, native.lm_head
        self.model = AudioBackbone(native.model, config)


def build_from_config(config, device, dtype):
    if config.audio_config.activation_function != "gelu":
        raise ValueError("The selected Qwen2-Audio checkpoint uses GELU")
    text = config.text_config
    if text.hidden_act != "silu" or text.tie_word_embeddings or text.use_sliding_window:
        raise ValueError(
            "The selected audio checkpoint uses full-attention SiLU with an untied head"
        )
    values = {
        name: getattr(text, name)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
        )
    }
    native = LlamaForCausalLM(
        LlamaConfig(
            **values,
            head_dim=text.hidden_size // text.num_attention_heads,
            rope_theta=text.rope_parameters["rope_theta"],
            qkv_bias=True,
            dtype=dtype,
            rope_scaling_factor=1.0,
            rope_low_freq_factor=1.0,
            rope_high_freq_factor=1.0,
            rope_original_max_position_embeddings=text.max_position_embeddings,
        )
    )
    return QwenAudio(native, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)
    text = {
        name.removeprefix("language_model."): remaining.pop(name)
        for name in list(remaining)
        if name.startswith("language_model.")
    }
    carrier = SimpleNamespace(
        model=model.model.text, lm_head=model.lm_head, config=model.config
    )
    qwen2.load_state_dict_into(carrier, text, model.config)
    encoder = model.model.audio_tower.encoder
    mapped = {}
    for target in encoder.state_dict():
        source = (
            target.replace(".conv.", ".")
            .replace(".emb.", ".")
            .replace(".mlp.fc", ".fc")
        )
        mapped[target] = remaining.pop("audio_tower." + source)
    encoder.load_state_dict(mapped, strict=True)
    model.model.projector.load_state_dict(
        {
            name: remaining.pop("multi_modal_projector.linear." + name)
            for name in ("weight", "bias")
        },
        strict=True,
    )
    if remaining:
        raise KeyError(f"Unmapped Qwen2-Audio state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("The Qwen2-Audio development workload uses one sequence")
    model.model.inputs = inputs
    workloads = llama.make_workloads(model, inputs, model.config)
    for phase, workload in list(workloads.items()):

        def run(workload=workload, phase=phase):
            result = workload.run()
            length = inputs["input_ids"].shape[1] - (phase == "prefill")
            for index, layer in enumerate(model.model.layers):
                attention = layer.self_attn.attn
                for name, cache in (
                    ("key", attention.k_cache),
                    ("value", attention.v_cache),
                ):
                    if attention.kv_layout == "HND":
                        cache = cache.transpose(1, 2)
                    result[f"past_key_values.{index}.{name}"] = (
                        cache.flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0)
                    )
            return result

        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
