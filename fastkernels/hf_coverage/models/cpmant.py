"""CPMAnt's learned prompt, segment positions and cached bidirectional attention."""

import math
import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from fastkernels.tasks.baseline.L2.t5_dense import T5DenseGatedActDense
from ..runner import Config, Workload
from .t5 import _fresh_cache


class CPMAntLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.num_attention_heads, config.dim_head
        width = self.heads * self.head_dim
        self.attention_norm = T5LayerNorm(config.hidden_size, eps=config.eps)
        self.qkv = Linear(config.hidden_size, 3 * width, bias=False)
        self.output = Linear(width, config.hidden_size, bias=False)
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)
        self.ffn_norm = T5LayerNorm(config.hidden_size, eps=config.eps)
        self.ffn = T5DenseGatedActDense(Config(d_model=config.hidden_size, d_ff=config.dim_ff, dense_act_fn="gelu"))

    def forward(self, hidden, mask, bias, cache=None):
        batch, length = hidden.shape[:2]
        q, k, v = (part.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for part in self.qkv(self.attention_norm(hidden)).chunk(3, -1))
        if cache is None:
            k, v = _fresh_cache(k, v)
        else:
            k, v = torch.cat([cache[0], k], -2), torch.cat([cache[1], v], -2)
        scores = self.bmm(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
        probabilities = self.softmax(scores.masked_fill(~mask[:, None], float("-inf")))
        probabilities = probabilities.masked_fill(~mask[:, None], 0)
        attention = self.bmm(probabilities, v).transpose(1, 2).contiguous().view(batch, length, -1)
        hidden = hidden + self.output(attention)
        return hidden + self.ffn(self.ffn_norm(hidden)), (k, v)


class CPMAnt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        size = config.vocab_size + config.prompt_types * config.prompt_length
        self.embedding = Embedding(size, config.hidden_size)
        self.segment_embedding = Embedding(config.segment_types, config.hidden_size)
        self.position_bias = Embedding(config.segment_types ** 2 + config.position_bias_num_buckets, config.num_attention_heads)
        self.layers = nn.ModuleList([CPMAntLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = T5LayerNorm(config.hidden_size, eps=config.eps)
        self.head = Linear(config.hidden_size, size, bias=False)
        self.head.weight = self.embedding.emb.weight

    def forward(self, input_ids, cache=None):
        config = self.config
        batch, length = input_ids.shape
        prompt = torch.arange(config.vocab_size + 2 * config.prompt_length,
                              config.vocab_size + 3 * config.prompt_length, device=input_ids.device)
        segments = torch.where(input_ids != 0, 2, 0)
        lengths = (segments != 0).sum(-1)
        ids = torch.cat([prompt.expand(batch, -1), input_ids], -1)
        segments = torch.cat([torch.zeros(batch, config.prompt_length, dtype=torch.long, device=ids.device), segments], -1)
        past = 0 if cache is None else cache[0][0].shape[-2]
        segment_states = self.segment_embedding(segments if past == 0 else segments[:, -1:])
        hidden = (self.embedding(ids) + segment_states)[:, past:]

        # The public default marks all tokens as context. Its mask is therefore
        # bidirectional, with learned prompts retained and ordinary left padding.
        valid = torch.arange(length - 1, -1, -1, device=ids.device)[None, :] < lengths[:, None]
        valid = torch.cat([torch.ones(batch, config.prompt_length, device=ids.device, dtype=torch.bool), valid], -1)
        mask = valid[:, past:, None] & valid[:, None, :]
        positions = torch.arange(ids.shape[1], device=ids.device)
        buckets = T5SelfAttention._relative_position_bucket(positions[None, :] - positions[:, None],
                    bidirectional=True, num_buckets=config.position_bias_num_buckets,
                    max_distance=config.position_bias_max_distance)
        cross_segments = segments[:, :, None] * config.segment_types + segments[:, None, :] + config.position_bias_num_buckets
        buckets = torch.where(segments[:, :, None] == segments[:, None, :], buckets, cross_segments)
        bias = self.position_bias(buckets).permute(0, 3, 1, 2).contiguous()[:, :, past:, :]
        updated = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, mask, bias, None if cache is None else cache[index])
            updated.append(state)
        hidden = self.norm(hidden)
        if past == 0:
            hidden = hidden[:, config.prompt_length:]
        return self.head(hidden), updated


def build_from_config(config, device, dtype):
    if _tp_size() != 1 or not config.tie_word_embeddings or not config.use_cache:
        raise ValueError("CPMAnt's default path uses tied embeddings and cached decoding")
    return CPMAnt(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {"embedding.emb.weight": remaining.pop("cpmant.input_embedding.weight"),
              "segment_embedding.emb.weight": remaining.pop("cpmant.segment_embedding.weight"),
              "position_bias.emb.weight": remaining.pop("cpmant.position_bias.relative_attention_bias"),
              "norm.weight": remaining.pop("cpmant.encoder.output_layernorm.weight"),
              "head.weight": remaining.pop("lm_head.weight")}
    if not torch.equal(mapped["embedding.emb.weight"], mapped["head.weight"]):
        raise ValueError("CPMAnt's tied embedding and head disagree")
    for index in range(config.num_hidden_layers):
        src, dst = f"cpmant.encoder.layers.{index}.", f"layers.{index}."
        mapped[dst + "qkv.weight"] = torch.cat([
            remaining.pop(src + f"self_att.self_attention.project_{part}.weight") for part in ("q", "k", "v")])
        mapped[dst + "ffn.wi.weight"] = torch.cat([
            remaining.pop(src + f"ffn.ffn.w_in.w_{part}.weight") for part in (0, 1)])
        for target, source in (("attention_norm", "self_att.layernorm_before_attention"),
                               ("output", "self_att.self_attention.attention_out"),
                               ("ffn_norm", "ffn.layernorm_before_ffn"), ("ffn.wo", "ffn.ffn.w_out")):
            mapped[dst + target + ".weight"] = remaining.pop(src + source + ".weight")
    if remaining:
        raise KeyError(f"Unmapped CPMAnt state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    ids, state = inputs["input_ids"], {}
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    steps = 2 if continuation else 1
    prompt_length = ids.shape[1] - steps

    def prefill():
        logits, state["cache"] = model(ids[:, :prompt_length])
        return {"logits": logits}

    def decode(step):
        # Native CPMAnt expects the complete token history and slices off the
        # cached prefix internally, including its learned prompt tokens.
        logits, state["cache"] = model(ids[:, :prompt_length + step + 1], state["cache"])
        return {"logits": logits}

    def prepare_decode(step):
        prefill()
        for previous in range(step):
            decode(previous)

    def collect(output):
        result = dict(output)
        for index, (key, value) in enumerate(state["cache"]):
            result[f"past_key_values.{index}.key"] = key
            result[f"past_key_values.{index}.value"] = value
        return result

    workloads = {"prefill": Workload(run=prefill)}
    for step in range(steps):
        name = f"decode_{step + 1}" if continuation else "decode"
        workloads[name] = Workload(run=lambda step=step: decode(step),
                                   prepare=lambda step=step: prepare_decode(step))
    if continuation:
        for workload in workloads.values():
            workload.collect = collect
    return workloads
