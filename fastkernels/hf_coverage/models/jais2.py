"""Jais2's affine LayerNorm decoder and squared-ReLU feed-forward blocks."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU

from .cohere2 import Cache, make_workloads
from .olmo2 import NativeFP32Rotary


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.head_dim
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn = nn.ModuleDict({
            "q_proj": Linear(width, self.heads * self.head_dim, bias=config.attention_bias),
            "k_proj": Linear(width, self.kv_heads * self.head_dim, bias=config.attention_bias),
            "v_proj": Linear(width, self.kv_heads * self.head_dim, bias=config.attention_bias),
            "o_proj": Linear(self.heads * self.head_dim, width, bias=config.attention_bias),
        })
        self.attention = DenseAttention(backend="sdpa")
        self.mlp = nn.ModuleDict({
            "up_proj": Linear(width, config.intermediate_size, bias=config.mlp_bias),
            "down_proj": Linear(config.intermediate_size, width, bias=config.mlp_bias),
        })
        self.activation = SquaredReLU()

    def forward(self, hidden, positions, rotary_cache, previous=None):
        batch, length, _ = hidden.shape
        normalized = self.input_layernorm(hidden)
        query = self.self_attn["q_proj"](normalized)
        key = self.self_attn["k_proj"](normalized)
        value = self.self_attn["v_proj"](normalized)
        indices = positions.expand(batch, -1).reshape(-1)
        # Reuse the native rotary callable: HF rounds coefficients and both
        # products in the activation dtype before adding the products.
        query, key = RotaryEmbedding.forward_native(
            indices, query.reshape(batch * length, -1),
            key.reshape(batch * length, -1), self.head_dim, rotary_cache,
        )
        query = query.reshape(batch, length, self.heads, self.head_dim)
        key = key.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if previous is not None:
            key = torch.cat((previous[0], key), dim=2)
            value = torch.cat((previous[1], value), dim=2)
        state = (key, value)
        repeats = self.heads // self.kv_heads
        if repeats != 1:
            key, value = key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)
        # The selected unpadded workload uses native SDPA's implicit causal
        # prefill and unrestricted single-token continuation.
        mask = None
        causal = previous is None and length > 1
        if previous is not None and length > 1:
            mask = torch.arange(key.shape[2], device=hidden.device)[None, :] <= positions[:, None]
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2),
                                 causal=causal, attn_mask=mask)
        hidden = hidden + self.self_attn["o_proj"](context.reshape(batch, length, -1))
        normalized = self.post_attention_layernorm(hidden)
        return hidden + self.mlp["down_proj"](self.activation(self.mlp["up_proj"](normalized))), state


class Jais2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_positions = config.max_position_embeddings
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None):
        if self.training:
            raise RuntimeError("Jais2 coverage supports inference only")
        start = 0 if past_key_values is None else past_key_values.seen_tokens
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or start + input_ids.shape[1] > self.max_positions:
            raise ValueError("Jais2 requires nonempty token batches within its configured position limit")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + start
        hidden = self.embed_tokens(input_ids)
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, positions, self.rotary.cos_sin_cache,
                                  None if past_key_values is None else past_key_values.layers[index])
            states.append(state)
        return {"logits": self.lm_head(self.norm(hidden)),
                "past_key_values": Cache(tuple(states), start + input_ids.shape[1])}


def build_from_config(config, device, dtype):
    if (config.hidden_act != "relu2" or not config.attention_bias or not config.mlp_bias
            or config.tie_word_embeddings or not config.use_cache
            or config.rope_parameters["rope_type"] != "default"
            or config.head_dim % 2 or config.hidden_size != config.num_attention_heads * config.head_dim
            or config.num_attention_heads % config.num_key_value_heads):
        raise ValueError("Selected Jais2 requires affine squared-ReLU blocks, untied embeddings and default RoPE")
    model = Jais2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    with torch.device("cpu"):
        model.rotary = NativeFP32Rotary(config.head_dim, config.max_position_embeddings,
                                      config.rope_parameters["rope_theta"], device)
    # Only reuse table construction here; Jais2 does not promote rotary products.
    model.rotary.cos_sin_cache = model.rotary.cos_sin_cache.to(dtype)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped = {name.removeprefix("model."): tensor for name, tensor in state_dict.items()}
    mapped["embed_tokens.emb.weight"] = mapped.pop("embed_tokens.weight")
    model.load_state_dict(mapped, strict=True)
