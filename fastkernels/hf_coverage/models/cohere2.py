"""Cohere2 parallel residuals, interleaved sliding-layer RoPE and native cache tails."""

from dataclasses import dataclass

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul

from .olmo2 import NativeFP32Rotary
from ..runner import Workload


@dataclass
class Cache:
    layers: tuple
    seen_tokens: int


class ParallelLayer(nn.Module):
    def __init__(self, config, index, *, rotary_on_all_layers=False):
        super().__init__()
        width = config.hidden_size
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", width // self.heads)
        layer_types = getattr(config, "layer_types", None)
        self.window = config.sliding_window if layer_types and layer_types[index] == "sliding_attention" else None
        self.apply_rotary = rotary_on_all_layers or self.window is not None
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, create_offset=False)
        self.self_attn = nn.ModuleDict({
            "q_proj": Linear(width, self.heads * self.head_dim, bias=False),
            "k_proj": Linear(width, self.kv_heads * self.head_dim, bias=False),
            "v_proj": Linear(width, self.kv_heads * self.head_dim, bias=False),
            "o_proj": Linear(self.heads * self.head_dim, width, bias=False),
        })
        self.attention = DenseAttention(backend="sdpa")
        self.mlp = nn.ModuleDict({
            "gate_proj": Linear(width, config.intermediate_size, bias=False),
            "up_proj": Linear(width, config.intermediate_size, bias=False),
            "down_proj": Linear(config.intermediate_size, width, bias=False),
        })

    def forward(self, hidden, positions, rotary_cache, previous=None):
        batch, length, _ = hidden.shape
        normed = self.input_layernorm(hidden)
        query = self.self_attn["q_proj"](normed)
        key = self.self_attn["k_proj"](normed)
        value = self.self_attn["v_proj"](normed)
        if self.apply_rotary:
            indices = positions.expand(batch, -1).reshape(-1)
            query, key = RotaryEmbedding.forward_native_interleaved(
                indices, query.reshape(batch * length, -1).float(),
                key.reshape(batch * length, -1).float(), self.head_dim, rotary_cache,
            )
            query, key = query.to(hidden.dtype), key.to(hidden.dtype)
        query = query.reshape(batch, length, self.heads, self.head_dim)
        key = key.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        past_length = 0 if previous is None else previous[0].shape[2]
        if previous is not None:
            key = torch.cat((previous[0], key), dim=2)
            value = torch.cat((previous[1], value), dim=2)
        # Cache position is cumulative; retained sliding tensors alone cannot recover it.
        keys = torch.arange(key.shape[2], device=hidden.device) + positions[0] - past_length
        mask = keys[None, :] <= positions[:, None]
        if self.window is not None:
            mask = mask & (keys[None, :] > positions[:, None] - self.window)
            state = (key[:, :, -(self.window - 1):].clone(), value[:, :, -(self.window - 1):].clone())
        else:
            state = (key, value)
        repeats = self.heads // self.kv_heads
        if repeats != 1:
            key, value = key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask)
        attention = self.self_attn["o_proj"](context.reshape(batch, length, -1))
        # This unchanged operation callable is the implementation on CPU and GPU.
        activated = SiluAndMul.forward_native(torch.cat((self.mlp["gate_proj"](normed),
                                                        self.mlp["up_proj"](normed)), dim=-1))
        return hidden + attention + self.mlp["down_proj"](activated), state


class Cohere2ForCausalLM(nn.Module):
    def __init__(self, config, *, rotary_on_all_layers=False):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(ParallelLayer(config, i, rotary_on_all_layers=rotary_on_all_layers)
                                    for i in range(config.num_hidden_layers))
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, create_offset=False)
        # Avoid allocating a second full vocabulary matrix before tying it.
        with torch.device("meta"):
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.emb.weight
        self.logit_scale = config.logit_scale

    def forward(self, input_ids, past_key_values=None):
        if self.training:
            raise RuntimeError("Cohere2 coverage supports inference only")
        start = 0 if past_key_values is None else past_key_values.seen_tokens
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + start
        hidden = self.embed_tokens(input_ids)
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, positions, self.rotary.cos_sin_cache,
                                  None if past_key_values is None else past_key_values.layers[index])
            states.append(state)
        logits = self.lm_head(self.norm(hidden)) * self.logit_scale
        return {"logits": logits, "past_key_values": Cache(tuple(states), start + input_ids.shape[1])}


def build_from_config(config, device, dtype):
    if (config.hidden_act != "silu" or config.attention_bias or not config.tie_word_embeddings
            or not config.use_cache or config.rope_parameters["rope_type"] != "default"
            or config.sliding_window is None or config.sliding_window < 2
            or config.head_dim % 2 or config.hidden_size != config.num_attention_heads * config.head_dim
            or config.num_attention_heads % config.num_key_value_heads
            or len(config.layer_types) != config.num_hidden_layers
            or set(config.layer_types) - {"sliding_attention", "full_attention"}):
        raise ValueError("Cohere2 coverage requires cached bias-free SiLU, tied embeddings and default interleaved RoPE")
    model = Cohere2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    with torch.device("cpu"):
        model.rotary = NativeFP32Rotary(config.head_dim, config.max_position_embeddings,
                                      config.rope_parameters["rope_theta"], device)
    # Native HF rounds trig coefficients to model dtype, then rotates Q/K in FP32.
    model.rotary.cos_sin_cache = model.rotary.cos_sin_cache.to(dtype).float()
    return model


def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("Cohere2 tied embedding and vocabulary weights disagree")
    mapped = {name.removeprefix("model."): tensor for name, tensor in state_dict.items()}
    mapped["embed_tokens.emb.weight"] = mapped.pop("embed_tokens.weight")
    model.load_state_dict(mapped, strict=True)
    # Existing LayerNorm caches inference-constant FP32 parameter views lazily.
    for module in model.modules():
        if isinstance(module, LayerNorm):
            module._cast_done = False


def make_workloads(model, inputs, config, *, case=None):
    if set(inputs) != {"input_ids"}:
        raise ValueError("The selected Cohere2 workload uses unpadded token inputs")
    ids = inputs["input_ids"]

    def flatten(output):
        result = {"logits": output["logits"]}
        for index, (key, value) in enumerate(output["past_key_values"].layers):
            result[f"past_key_values.{index}.key"] = key
            result[f"past_key_values.{index}.value"] = value
        return result

    if case is None or case.get("workload") != "causal_lm_continuation":
        return {"forward": Workload(run=lambda: flatten(model(ids)))}
    prefix = ids.shape[1] - 2
    if ids.ndim != 2 or prefix < 1:
        raise ValueError("Cohere2 continuation requires a prefix and two supplied tokens")
    state = {}

    def initial():
        return model(ids[:, :prefix])

    def advance(index, previous):
        return model(ids[:, prefix + index:prefix + index + 1], past_key_values=previous)

    def prepare(index):
        previous = initial()["past_key_values"]
        for step in range(index):
            previous = advance(step, previous)["past_key_values"]
        state["previous"] = previous

    return {
        "prefill": Workload(run=lambda: flatten(initial())),
        "decode_1": Workload(run=lambda: flatten(advance(0, state["previous"])), prepare=lambda: prepare(0)),
        "decode_2": Workload(run=lambda: flatten(advance(1, state["previous"])), prepare=lambda: prepare(1)),
    }
