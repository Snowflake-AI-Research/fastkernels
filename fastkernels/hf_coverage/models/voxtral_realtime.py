"""VoxtralRealtime's ordinary array-input, chunked speech generation."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.patches.voxtral_realtime_time import VoxtralRealtimeTimeEmbedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from .dia import AudioMLP
from .mimi import CausalConv


class Attention(nn.Module):
    def __init__(self, config, audio):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.head_dim
        self.kv_heads = self.heads if audio else config.num_key_value_heads
        self.window = config.sliding_window
        self.q_proj = Linear(config.hidden_size, self.heads * self.width, bias=audio)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.width, bias=False)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.width, bias=audio)
        self.o_proj = Linear(self.heads * self.width, config.hidden_size, bias=audio)
        self.rotary = RotaryEmbedding(self.width, config.max_position_embeddings, config.rope_parameters["rope_theta"])
        self.attention = DenseAttention(backend="sdpa")
        self.reset()

    def reset(self):
        self.key = self.value = None
        self.seen = 0

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        positions = torch.arange(self.seen, self.seen + length, device=hidden.device)
        query, key = (projection(hidden).reshape(batch * length, -1) for projection in (self.q_proj, self.k_proj))
        query, key = self.rotary.forward_native(positions.expand(batch, -1).reshape(-1), query, key,
                                               self.width, self.rotary.cos_sin_cache.to(hidden.dtype))
        query = query.reshape(batch, length, self.heads, self.width)
        key = key.reshape(batch, length, self.kv_heads, self.width).transpose(1, 2)
        value = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.width).transpose(1, 2)
        if self.key is not None:
            key, value = torch.cat((self.key, key), 2), torch.cat((self.value, value), 2)
        else:
            key, value = key.contiguous(), value.contiguous()
        source_positions = torch.arange(self.seen + length - key.shape[2], self.seen + length, device=hidden.device)
        mask = (source_positions[None] <= positions[:, None]) & (source_positions[None] > positions[:, None] - self.window)
        self.key, self.value = key[:, :, -self.window + 1:], value[:, :, -self.window + 1:]
        self.seen += length
        keys, values = (tensor.repeat_interleave(self.heads // self.kv_heads, dim=1).transpose(1, 2) for tensor in (key, value))
        output = self.attention(query, keys, values, attn_mask=mask)
        return self.o_proj(output.reshape(batch, length, -1))


class Layer(nn.Module):
    def __init__(self, config, audio):
        super().__init__()
        self.self_attn = Attention(config, audio)
        self.input_layernorm = T5LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = T5LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AudioMLP(config)
        if audio:
            self.mlp.down_proj = Linear(config.intermediate_size, config.hidden_size)
        else:
            self.ada_linear1 = Linear(config.hidden_size, 32, bias=False)
            self.ada_linear2 = Linear(32, config.hidden_size, bias=False)
            self.gelu, self.product = GELU(), ProductGate()

    def forward(self, hidden, condition=None):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        normalized = self.post_attention_layernorm(hidden)
        if condition is not None:
            gate = 1 + self.ada_linear2(self.gelu(self.ada_linear1(condition)))
            normalized = self.product(torch.cat((normalized, gate.expand_as(normalized)), -1))
        return hidden + self.mlp(normalized)


class VoxtralRealtime(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        audio, text = config.audio_config, config.text_config
        self.conv1 = CausalConv(audio.num_mel_bins, audio.hidden_size, 3)
        self.conv2 = CausalConv(audio.hidden_size, audio.hidden_size, 3, stride=2)
        self.gelu = GELU()
        self.audio_layers = nn.ModuleList(Layer(audio, True) for _ in range(audio.num_hidden_layers))
        self.audio_norm = T5LayerNorm(audio.hidden_size, eps=audio.rms_norm_eps)
        self.linear_1 = Linear(audio.hidden_size * config.downsample_factor, text.hidden_size, bias=False)
        self.linear_2 = Linear(text.hidden_size, text.hidden_size, bias=False)
        self.embed_tokens = Embedding(text.vocab_size, text.hidden_size)
        self.text_layers = nn.ModuleList(Layer(text, False) for _ in range(text.num_hidden_layers))
        self.text_norm = T5LayerNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.emb.weight
        self.time_embedding = VoxtralRealtimeTimeEmbedding(text.hidden_size)
        self.top1 = CodecTop1()

    def generate(self, inputs, generation, steps):
        for layer in (*self.audio_layers, *self.text_layers):
            layer.self_attn.reset()
        features, sequences = inputs["input_features"], inputs["input_ids"]
        if features.shape[-1] % self.config.audio_length_per_tok:
            raise ValueError("This array-input case uses complete audio-token groups")
        encoded = self.gelu(self.conv1(features))
        encoded = self.gelu(self.conv2(encoded)).transpose(1, 2)
        maximum = min(steps, features.shape[-1] // self.config.audio_length_per_tok - sequences.shape[1])
        eos = generation.get("eos_token_id")
        eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
        eos_ids = torch.tensor(eos, device=features.device, dtype=torch.long)
        outputs, seen = {}, 0
        for step in range(maximum):
            ids = sequences if step == 0 else sequences[:, -1:]
            start, end = seen * self.config.downsample_factor, (seen + ids.shape[1]) * self.config.downsample_factor
            audio = encoded[:, start:end]
            for layer in self.audio_layers:
                audio = layer(audio)
            audio = self.audio_norm(audio).reshape(features.shape[0], ids.shape[1], -1)
            audio = self.linear_2(self.gelu(self.linear_1(audio)))
            hidden = self.embed_tokens(ids) + audio
            delay = inputs.get("num_delay_tokens", self.config.default_num_delay_tokens)
            timestep = hidden.new_full((1,), delay)
            condition = self.time_embedding(timestep)[:, None]
            for layer in self.text_layers:
                hidden = layer(hidden, condition)
            hidden = self.text_norm(hidden)
            logits = self.lm_head(hidden[:, -1:])[:, -1].float()
            outputs[f"logits.{step}"] = logits
            next_ids = self.top1(logits).reshape(1, 1)
            sequences = torch.cat((sequences, next_ids), 1)
            seen += ids.shape[1]
            if eos and torch.any(next_ids == eos_ids).item():
                break
        for index, layer in enumerate(self.text_layers):
            outputs[f"past_key_values.{index}.key"] = layer.self_attn.key
            outputs[f"past_key_values.{index}.value"] = layer.self_attn.value
        return {"sequences": sequences, **outputs}


def build_from_config(config, device, dtype):
    if (not config.text_config.tie_word_embeddings or config.downsample_factor != 4
            or config.audio_length_per_tok != 8 or config.projector_hidden_act != "gelu"
            or any(tower.hidden_act != "silu" or tower.rope_parameters["rope_type"] != "default"
                   for tower in (config.audio_config, config.text_config))):
        raise ValueError("VoxtralRealtime requires native tied-head SiLU towers and four-frame audio grouping")
    return VoxtralRealtime(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    if not torch.equal(remaining["language_model.lm_head.weight"], remaining["language_model.model.embed_tokens.weight"]):
        raise ValueError("VoxtralRealtime checkpoint requires tied text embedding/head")
    for name in model.state_dict():
        if name.startswith(("conv1.", "conv2.")):
            source = "audio_tower.embedder." + name.replace(".conv.", ".")
        elif name.startswith("audio_norm."):
            source = name.replace("audio_norm.", "audio_tower.norm.")
        elif name.startswith("audio_layers."):
            source = name.replace("audio_layers.", "audio_tower.layers.")
            source = source.replace(".input_layernorm.", ".self_attn_layer_norm.").replace(".post_attention_layernorm.", ".final_layer_norm.")
        elif name.startswith("text_layers."):
            source = name.replace("text_layers.", "language_model.model.layers.")
            source = source.replace(".ada_linear1.", ".ada_rms_norm.linear1.").replace(".ada_linear2.", ".ada_rms_norm.linear2.")
        elif name.startswith("text_norm."):
            source = name.replace("text_norm.", "language_model.model.norm.")
        elif name.startswith("embed_tokens."):
            source = "language_model.model." + name.replace(".emb.", ".")
        elif name.startswith("lm_head."):
            source = "language_model." + name
        else:
            source = "multi_modal_projector." + name
        if ".mlp.gate_up_proj." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace("gate_up_proj", part)) for part in ("gate_proj", "up_proj")], 0)
        else:
            mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped VoxtralRealtime state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case):
    generation = case["reference"]["generation_config"]
    if generation.get("do_sample", False) or generation.get("num_beams", 1) != 1 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("The native array-input development case uses one greedy transcription request")
    return {"generate": Workload(run=lambda: model.generate(inputs, generation, case["generation_kwargs"]["max_new_tokens"]))}
