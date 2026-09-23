"""GLM-ASR audio encoding, partial rotary attention and greedy transcription."""

from types import SimpleNamespace
import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.whisper import WhisperConfig, WhisperEncoder
from . import llama
from .voxtral import build_language, make_workloads


class AudioAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.width = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.rotary_width = int(self.width * config.rope_parameters["partial_rotary_factor"])
        self.q_proj = Linear(config.hidden_size, self.heads * self.width)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.width, bias=False)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.width)
        self.o_proj = Linear(self.heads * self.width, config.hidden_size)
        self.rotary = RotaryEmbedding(self.rotary_width, config.max_position_embeddings, config.rope_parameters["rope_theta"])
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).reshape(batch * length, self.heads, self.width)
        key = self.k_proj(hidden).reshape(batch * length, self.kv_heads, self.width)
        positions = torch.arange(length, device=hidden.device).expand(batch, -1).reshape(-1)
        qrot, krot = self.rotary.forward_native(positions, query[..., :self.rotary_width].reshape(batch * length, -1),
                                               key[..., :self.rotary_width].reshape(batch * length, -1),
                                               self.rotary_width, self.rotary.cos_sin_cache.to(hidden.dtype))
        query = torch.cat((qrot.reshape(batch * length, self.heads, -1), query[..., self.rotary_width:]), -1)
        key = torch.cat((krot.reshape(batch * length, self.kv_heads, -1), key[..., self.rotary_width:]), -1)
        query = query.reshape(batch, length, self.heads, self.width)
        key = key.reshape(batch, length, self.kv_heads, self.width)
        value = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.width)
        key, value = (tensor.repeat_interleave(self.heads // self.kv_heads, dim=2) for tensor in (key, value))
        output = self.attention(query, key, value)
        return self.o_proj(output.reshape(batch, length, -1))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        carrier = WhisperConfig(d_model=config.hidden_size, num_mel_bins=config.num_mel_bins,
                                max_source_positions=config.max_position_embeddings,
                                encoder_layers=config.num_hidden_layers, encoder_attention_heads=config.num_attention_heads,
                                encoder_ffn_dim=config.intermediate_size)
        self.encoder = WhisperEncoder(carrier)
        del self.encoder.embed_positions
        for layer in self.encoder.layers:
            layer.self_attn = AudioAttention(config)

    def forward(self, features):
        encoder = self.encoder
        hidden = encoder.gelu(encoder.conv1(features))
        hidden = encoder.gelu(encoder.conv2(hidden)).transpose(1, 2)
        for layer in encoder.layers:
            hidden = layer(hidden)
        return encoder.layer_norm(hidden)


class Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text, self.audio = text, AudioEncoder(config.audio_config)
        self.linear_1 = Linear(config.audio_config.intermediate_size, 2 * config.text_config.hidden_size)
        self.linear_2 = Linear(2 * config.text_config.hidden_size, config.text_config.hidden_size)
        self.activation = GELU()
        self.audio_token_id = config.audio_token_id
        self.inputs = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, ids, positions):
        hidden = self.text.embed_tokens(ids)
        if get_context().is_prefill:
            features = self.inputs["input_features"]
            audio = self.audio(features).reshape(features.shape[0], -1, self.linear_1.weight.shape[1])
            audio = self.linear_2(self.activation(self.linear_1(audio)))
            # Integer processor-mask lengths determine valid frame groups.
            lengths = ((self.inputs["input_features_mask"].sum(-1) + 1) // 2) // 4
            valid = torch.arange(audio.shape[1], device=audio.device)[None] < lengths[:, None]
            hidden = hidden.masked_scatter((ids == self.audio_token_id)[:, None].expand_as(hidden), audio[valid])
        return self.text(ids, positions, inputs_embeds=hidden)


class GlmAsr(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config, self.lm_head = text.config, text.lm_head
        self.model = Backbone(text.model, config)
        self.top1 = CodecTop1()


def build_from_config(config, device, dtype):
    text, audio = config.text_config, config.audio_config
    if (config.projector_hidden_act != "gelu" or audio.hidden_act != "gelu"
            or audio.intermediate_size != 4 * audio.hidden_size or text.tie_word_embeddings
            or text.hidden_act != "silu" or text.attention_bias or text.mlp_bias
            or not text.use_cache or audio.rope_parameters["partial_rotary_factor"] != 0.5):
        raise ValueError("GLM-ASR requires half-head audio RoPE, four-frame GELU projection and cached untied Llama")
    return GlmAsr(build_language(text, dtype), config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.removeprefix("language_model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("language_model.")}
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    encoder, mapped = model.model.audio.encoder, {}
    for name in encoder.state_dict():
        source = name.replace(".conv.", ".").replace("self_attn_layer_norm", "input_layernorm").replace("final_layer_norm", "post_attention_layernorm")
        if source.startswith("layer_norm."):
            source = source.replace("layer_norm.", "norm.", 1)
        mapped[name] = remaining.pop("audio_tower." + source)
    encoder.load_state_dict(mapped, strict=True)
    for name in ("linear_1", "linear_2"):
        module = getattr(model.model, name)
        module.load_state_dict({field: remaining.pop(f"multi_modal_projector.{name}.{field}") for field in module.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped GLM-ASR state: {sorted(remaining)}")
