"""Higgs Audio's documented mixed text/audio forward, including every KV state."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from ..runner import Workload


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act = SiluAndMul()

    def forward(self, hidden):
        return self.down_proj(self.act.forward_native(torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), -1)))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        for name, heads in (("q", self.heads), ("k", self.kv_heads), ("v", self.kv_heads)):
            setattr(self, name + "_proj", Linear(config.hidden_size, heads * self.head_dim, bias=config.attention_bias))
        self.o_proj = Linear(self.heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, hidden, rotary, positions, cache=None):
        batch, length = hidden.shape[:2]
        query, key, value = [getattr(self, name + "_proj")(hidden).view(batch, length, heads, self.head_dim)
                             for name, heads in (("q", self.heads), ("k", self.kv_heads), ("v", self.kv_heads))]
        query, key = RotaryEmbedding.forward_native(positions.reshape(-1), query.reshape(batch * length, -1),
            key.reshape(batch * length, -1), self.head_dim, rotary.cos_sin_cache.to(hidden.dtype))
        query = query.view(batch, length, self.heads, self.head_dim)
        key = key.view(batch, length, self.kv_heads, self.head_dim)
        if cache is not None:
            key, value = torch.cat((cache[0], key), 1), torch.cat((cache[1], value), 1)
        state = key, value
        key, value = (x.repeat_interleave(self.heads // self.kv_heads, 2) for x in (key, value))
        mask = None
        if cache is not None and length > 1:
            mask = torch.arange(key.shape[1], device=hidden.device)[None] <= positions.reshape(-1, 1)
        output = self.attn(query, key, value, causal=cache is None and length > 1, attn_mask=mask)
        return self.o_proj(output.reshape(batch, length, -1)), state


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp, self.audio_mlp = MLP(config), MLP(config)
        for name in ("input_layernorm", "post_attention_layernorm", "audio_input_layernorm", "audio_post_attention_layernorm"):
            setattr(self, name, RMSNormNative(config.hidden_size, config.rms_norm_eps))

    def forward(self, hidden, audio_mask, rotary, positions, cache=None):
        normalized = torch.empty_like(hidden)
        normalized[audio_mask] = self.audio_input_layernorm(hidden[audio_mask])
        normalized[~audio_mask] = self.input_layernorm(hidden[~audio_mask])
        attention, state = self.self_attn(normalized, rotary, positions, cache)
        hidden = hidden + attention
        result = hidden.clone()
        result[audio_mask] = result[audio_mask] + self.audio_mlp(self.audio_post_attention_layernorm(hidden[audio_mask]))
        result[~audio_mask] = result[~audio_mask] + self.mlp(self.post_attention_layernorm(hidden[~audio_mask]))
        return result, state


class AudioEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_audio_tokens = Embedding(config.num_codebooks * config.codebook_size, config.hidden_size)
        self.register_buffer("offsets", torch.arange(config.num_codebooks) * config.codebook_size, persistent=False)
        self.reduce = SegmentCSR()

    def forward(self, ids):
        values = self.embed_audio_tokens(ids + self.offsets)
        count = ids.shape[-1]
        # Native sum accumulates BF16 operands in FP32 before the output cast.
        return self.reduce(values.movedim(-2, 0).contiguous().float(),
                           torch.tensor([0, count], device=ids.device), "sum")[0].to(values.dtype)


class HiggsAudio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.model.embed_audio_tokens = AudioEmbeddings(config)
        self.model.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.model.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.audio_lm_head = Linear(config.hidden_size, config.num_codebooks * config.codebook_size, bias=False)
        rope = config.rope_parameters
        self.model.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, rope["rope_theta"],
            rope.get("factor", 1.), rope.get("low_freq_factor", 1.), rope.get("high_freq_factor", 1.),
            rope.get("original_max_position_embeddings", config.max_position_embeddings))

    def forward(self, input_ids, audio_input_ids, audio_input_ids_mask):
        hidden = self.model.embed_tokens(input_ids)
        mask = (input_ids == self.config.audio_token_id) | (input_ids == self.config.audio_delay_token_id)
        hidden[mask] = self.model.embed_audio_tokens(audio_input_ids)[audio_input_ids_mask]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
        outputs = {}
        for index, layer in enumerate(self.model.layers):
            hidden, (key, value) = layer(hidden, mask, self.model.rotary_emb, positions)
            outputs[f"past_key_values.{index}.key"] = key.transpose(1, 2)
            outputs[f"past_key_values.{index}.value"] = value.transpose(1, 2)
        outputs["logits"] = self.audio_lm_head(self.model.norm(hidden))
        return outputs


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.rope_parameters["rope_type"] not in ("llama3", "default"):
        raise ValueError("Preserve Higgs' SiLU and native Llama3 scaled rotary configuration")
    return HiggsAudio(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, weights, config):
    mapped, consumed = {}, set()
    for name, value in model.state_dict().items():
        source = name.replace(".emb.weight", ".weight")
        if weights[source].shape != value.shape:
            raise ValueError(f"Higgs state shape mismatch: {source}")
        mapped[name] = weights[source]
        consumed.add(source)
    if consumed != set(weights):
        raise ValueError(f"Higgs unmapped state: {sorted(set(weights) - consumed)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
