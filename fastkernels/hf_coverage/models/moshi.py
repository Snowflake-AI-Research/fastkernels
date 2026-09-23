"""Moshi's public text-logit forward from text and two audio-code streams."""

from copy import copy
import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bitnet_rms_norm import BitNetRMSNorm
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from .dia import AudioMLP


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.width = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.window = config.sliding_window
        self.q_proj = Linear(config.hidden_size, self.heads * self.width, bias=False)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.width, bias=False)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.width, bias=False)
        self.o_proj = Linear(self.heads * self.width, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.width, config.max_position_embeddings, config.rope_parameters["rope_theta"])
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden, mask):
        batch, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device).expand(batch, -1).reshape(-1)
        query, key = (op(hidden).reshape(batch * length, -1) for op in (self.q_proj, self.k_proj))
        query, key = self.rotary.forward_native(positions, query, key, self.width, self.rotary.cos_sin_cache.to(hidden.dtype))
        query = query.reshape(batch, length, self.heads, self.width)
        key = key.reshape(batch, length, self.kv_heads, self.width).transpose(1, 2).contiguous()
        value = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.width).transpose(1, 2).contiguous()
        # DynamicSlidingWindowLayer returns the full first prefill to attention,
        # while retaining only window-1 entries for the returned cache.
        cache = tuple(tensor[:, :, -(self.window - 1):].contiguous() for tensor in (key, value))
        keys, values = (tensor.repeat_interleave(self.heads // self.kv_heads, dim=1).transpose(1, 2) for tensor in (key, value))
        output = self.attention(query, keys, values, attn_mask=mask)
        return self.o_proj(output.reshape(batch, length, -1)), cache


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        carrier = copy(config)
        carrier.intermediate_size = config.ffn_dim // 2
        self.mlp = AudioMLP(carrier)

    def forward(self, hidden, mask):
        attention, cache = self.self_attn(self.input_layernorm(hidden), mask)
        hidden = hidden + attention
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), cache


class Moshi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.ModuleList(Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(2 * config.num_codebooks))
        self.text_embedding = Embedding(config.vocab_size + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.norm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, user_audio_codes, moshi_audio_codes, attention_mask):
        audio = torch.cat((moshi_audio_codes, user_audio_codes), dim=1)
        hidden = sum(embedding(audio[:, index]) for index, embedding in enumerate(self.embed_tokens)) + self.text_embedding(input_ids)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        # Pinned MoshiModel creates a full causal mask before creating its
        # sliding cache. Consequently its initial public forward has no window
        # restriction, although the returned cache is truncated to window-1.
        mask = positions[None, :] <= positions[:, None]
        mask = mask[None, None] & attention_mask[:, None, None].bool()
        output = {}
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, mask)
            for prefix in ("past_key_values", "depth_past_key_values"):
                for name, tensor in zip(("key", "value"), cache):
                    output[f"{prefix}.{index}.{name}"] = tensor
        hidden = self.norm(hidden)
        return {**output, "logits": self.lm_head(hidden), "last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or not config.use_cache or config.rope_parameters["rope_type"] != "default":
        raise ValueError("Moshi forward requires native SiLU, rotary positions and sliding caches")
    return Moshi(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    model.inactive_state_names = sorted(name for name in remaining if name.startswith(("audio_encoder.", "depth_decoder.")))
    for name in model.inactive_state_names:
        remaining.pop(name)
    mapped = {}
    for name in model.state_dict():
        if name.startswith("embed_tokens."):
            source = name.replace(".emb.", ".")
        elif name == "text_embedding.emb.weight":
            source = "decoder.model.embed_tokens.weight"
        elif name.startswith("lm_head."):
            source = "decoder." + name
        else:
            source = "decoder.model." + name
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                source = source.replace(f".{projection}.", f".{projection}.linear.")
            source = source.replace(".mlp.gate_up_proj.", ".mlp.fc1.").replace(".mlp.down_proj.", ".mlp.fc2.")
        mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped active Moshi state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
