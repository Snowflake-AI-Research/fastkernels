"""Gemma2 constructor graph under the explicitly selected HF SDPA backend.

HF SDPA omits attention-score softcapping. The final-logit cap is preserved.
Native norm/GELU/rotary callables are unchanged existing operations; these
callables have no standalone optimization interface in this adapter.
"""
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from .cohere2 import Cache, make_workloads
from .gemma3 import MLP, NativeNorm, PositionRotary


class Attention(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.heads, self.kv_heads, self.dim = (
            config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        self.window = config.sliding_window if kind == "sliding_attention" else None
        self.scale = config.query_pre_attn_scalar ** -0.5
        for name, heads in (("q", self.heads), ("k", self.kv_heads), ("v", self.kv_heads)):
            setattr(self, name + "_proj", Linear(config.hidden_size, heads * self.dim, bias=False))
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, bias=False)
        rotary_config = SimpleNamespace(head_dim=self.dim, rope_parameters={"default": config.rope_parameters})
        self.rotary = PositionRotary(rotary_config, "default")
        self.core = DenseAttention(backend="sdpa")

    def forward(self, hidden, positions, previous):
        shape = (*hidden.shape[:2], -1, self.dim)
        query, key, value = (getattr(self, name + "_proj")(hidden).reshape(shape)
                             for name in ("q", "k", "v"))
        query, key = self.rotary(query, key, positions)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        old_length = 0 if previous is None else previous[0].shape[2]
        if previous is not None:
            key, value = (torch.cat((old, new), dim=2) for old, new in zip(previous, (key, value)))
        keys = torch.arange(key.shape[2], device=hidden.device) + positions[0, 0] - old_length
        mask = keys[None, :] <= positions[0, :, None]
        if self.window is not None:
            mask = mask & (keys[None, :] > positions[0, :, None] - self.window)
            retained = tuple(x[:, :, 1-self.window:].clone() for x in (key, value))
        else:
            retained = (key, value)
        groups = self.heads // self.kv_heads
        key, value = (x.transpose(1, 2).repeat_interleave(groups, dim=2) for x in (key, value))
        output = self.core(query, key, value, softmax_scale=self.scale, attn_mask=mask)
        return self.o_proj(output.reshape(*hidden.shape[:2], -1)), retained


class Layer(nn.Module):
    def __init__(self, config, kind, *, post_branch_norms):
        super().__init__()
        self.self_attn, self.mlp = Attention(config, kind), MLP(config)
        self.input_layernorm = NativeNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = NativeNorm(config.hidden_size, config.rms_norm_eps)
        self.post_branch_norms = post_branch_norms
        if post_branch_norms:
            self.post_attention_layernorm = NativeNorm(config.hidden_size, config.rms_norm_eps)
            self.post_feedforward_layernorm = NativeNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions, previous):
        attention, state = self.self_attn(self.input_layernorm(hidden), positions, previous)
        if self.post_branch_norms:
            attention = self.post_attention_layernorm(attention)
        hidden = hidden + attention
        feedforward = self.mlp(self.pre_feedforward_layernorm(hidden))
        if self.post_branch_norms:
            feedforward = self.post_feedforward_layernorm(feedforward)
        return hidden + feedforward, state


class Gemma2ForCausalLM(nn.Module):
    def __init__(self, config, *, post_branch_norms=True):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.register_buffer("embed_scale", torch.tensor(config.hidden_size ** 0.5, dtype=torch.float32), persistent=False)
        self.layers = nn.ModuleList(Layer(config, kind, post_branch_norms=post_branch_norms)
                                    for kind in config.layer_types)
        self.norm = NativeNorm(config.hidden_size, config.rms_norm_eps)
        with torch.device("meta"):
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.emb.weight
        self.final_cap = config.final_logit_softcapping
        self.tanh = Tanh()

    def forward(self, input_ids, past_key_values=None):
        if self.training:
            raise RuntimeError("Gemma2/VaultGemma coverage supports inference only")
        start = 0 if past_key_values is None else past_key_values.seen_tokens
        positions = (torch.arange(input_ids.shape[1], device=input_ids.device) + start)[None].expand(input_ids.shape[0], -1)
        hidden = self.embed_tokens(input_ids) * self.embed_scale
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, positions, None if past_key_values is None else past_key_values.layers[index])
            states.append(state)
        logits = self.lm_head(self.norm(hidden))
        if self.final_cap is not None:
            logits = self.tanh(logits / self.final_cap) * self.final_cap
        return {"logits": logits, "past_key_values": Cache(tuple(states), start + input_ids.shape[1])}


def build_from_config(config, device, dtype, *, post_branch_norms=True):
    if (config.hidden_activation != "gelu_pytorch_tanh" or config.attention_bias
            or not config.tie_word_embeddings or not config.use_cache
            or getattr(config, "use_bidirectional_attention", False)
            or config.rope_parameters["rope_type"] != "default"
            or config.head_dim % 2 or config.num_attention_heads % config.num_key_value_heads
            or config.sliding_window is None or config.sliding_window < 2
            or len(config.layer_types) != config.num_hidden_layers
            or set(config.layer_types) - {"sliding_attention", "full_attention"}):
        raise ValueError("Gemma2/VaultGemma coverage requires selected SDPA, tied cached causal tanh-GELU and default RoPE")
    return Gemma2ForCausalLM(config, post_branch_norms=post_branch_norms).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("Gemma2/VaultGemma tied embedding and vocabulary weights disagree")
    mapped = {name.removeprefix("model."): tensor for name, tensor in state_dict.items()}
    mapped["embed_tokens.emb.weight"] = mapped.pop("embed_tokens.weight")
    model.load_state_dict(mapped, strict=True)
