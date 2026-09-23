"""DebertaForMaskedLM with microsoft/deberta-base computational settings."""

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput

from ..patches.linear import PostBiasLinear
from ..patches.normalization import DebertaLayerNorm
from ..runner import Workload
from .bert import MaskedLMHead, load_mlm_head


class DisentangledSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.in_proj = PostBiasLinear(config.hidden_size, 3 * config.hidden_size)
        self.pos_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_q_proj = Linear(config.hidden_size, config.hidden_size)
        self.matmul = BMM()
        self.attention = DenseAttention(backend="sdpa")
        # HF rounds the content-query divisor to the model dtype. Its position
        # query uses the FP32 scalar instead, including for BF16 execution.
        scale = torch.tensor(3 * self.head_dim, dtype=torch.float32, device="cpu")
        self.position_scale = torch.sqrt(scale).item()
        self.register_buffer("query_scale", torch.tensor(self.position_scale), persistent=False)

    def forward(self, hidden_states, rel_embeddings, c2p_index, p2c_index):
        batch, length, width = hidden_states.shape
        packed = self.in_proj(hidden_states).view(batch, length, self.num_heads, 3 * self.head_dim)
        query, key, value = packed.chunk(3, dim=-1)
        query = query / self.query_scale
        query_heads = query.transpose(1, 2)
        key_heads = key.transpose(1, 2)

        pos_key = self.pos_proj(rel_embeddings).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        c2p = self.matmul(query_heads, pos_key.transpose(-1, -2))
        c2p = torch.gather(c2p, -1, c2p_index.expand(batch, self.num_heads, length, length))

        pos_query = self.pos_q_proj(rel_embeddings).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        pos_query = pos_query / self.position_scale
        p2c = self.matmul(key_heads, pos_query.transpose(-1, -2))
        p2c = torch.gather(p2c, -1, p2c_index.expand(batch, self.num_heads, length, length)).transpose(-1, -2)

        context = self.attention(query, key, value, softmax_scale=1.0, attn_mask=c2p + p2c)
        return context.reshape(batch, length, width)


class DebertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = EncoderSelfOutput(config)
        self.output.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, hidden_states, rel_embeddings, c2p_index, p2c_index):
        context = self.self(hidden_states, rel_embeddings, c2p_index, p2c_index)
        return self.output(context, hidden_states)


class DebertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaAttention(config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)
        self.output.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, hidden_states, rel_embeddings, c2p_index, p2c_index):
        hidden_states = self.attention(hidden_states, rel_embeddings, c2p_index, p2c_index)
        return self.output(self.intermediate(hidden_states), hidden_states)


class DebertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, input_ids):
        return self.LayerNorm(self.word_embeddings(input_ids))


class DebertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_relative_positions = config.max_relative_positions
        if self.max_relative_positions < 1:
            self.max_relative_positions = config.max_position_embeddings
        self.rel_embeddings = Embedding(2 * self.max_relative_positions, config.hidden_size)
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        length = hidden_states.shape[1]
        positions = torch.arange(length, device=hidden_states.device)
        relative = positions[:, None] - positions[None, :]
        span = min(length, self.max_relative_positions)
        c2p_index = (relative + span).clamp(0, 2 * span - 1)[None, None]
        p2c_index = (-relative + span).clamp(0, 2 * span - 1)[None, None]
        center = self.max_relative_positions
        rel_embeddings = self.rel_embeddings.emb.weight[center - span:center + span]
        for layer in self.layer:
            hidden_states = layer(hidden_states, rel_embeddings, c2p_index, p2c_index)
        return hidden_states


class DebertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)

    def forward(self, input_ids):
        return self.encoder(self.embeddings(input_ids))


class DebertaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deberta = DebertaModel(config)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.deberta.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        return self.lm_head(self.deberta(input_ids))


def build_from_config(config, device, dtype):
    pos_types = config.pos_att_type
    if isinstance(pos_types, str):
        pos_types = [part.strip() for part in pos_types.lower().split("|")]
    if (config.hidden_act != "gelu" or not config.legacy
            or not config.relative_attention or set(pos_types or []) != {"c2p", "p2c"}
            or config.position_biased_input or config.type_vocab_size != 0
            or getattr(config, "embedding_size", config.hidden_size) != config.hidden_size
            or getattr(config, "talking_head", False)):
        raise ValueError("DeBERTa coverage requires microsoft/deberta-base computational settings")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    return DebertaForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.deberta.state_dict():
        source = "deberta." + name.replace(".emb.weight", ".weight")
        if source.endswith(".in_proj.bias"):
            prefix = source.removesuffix("in_proj.bias")
            query = state_dict[prefix + "q_bias"].view(config.num_attention_heads, -1)
            value = state_dict[prefix + "v_bias"].view(config.num_attention_heads, -1)
            weights[name] = torch.stack((query, torch.zeros_like(query), value), dim=1).flatten()
        else:
            weights[name] = state_dict[source]
    model.deberta.load_state_dict(weights)
    load_mlm_head(model.lm_head, state_dict)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(inputs["input_ids"])})}
