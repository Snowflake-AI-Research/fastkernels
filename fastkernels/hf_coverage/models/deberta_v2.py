"""DebertaV2ForMaskedLM with shared positional projections and input convolution."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer

from .bert import MaskedLMHead, load_mlm_head, make_workloads


class DisentangledSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_size
        self.query_proj = Linear(config.hidden_size, config.hidden_size)
        self.key_proj = Linear(config.hidden_size, config.hidden_size)
        self.value_proj = Linear(config.hidden_size, config.hidden_size)
        self.matmul = BMM()
        self.softmax = Softmax()
        scale = torch.sqrt(torch.tensor(3 * self.head_dim, dtype=torch.float32))
        self.register_buffer("scale", scale, persistent=False)

    def heads_view(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2).reshape(
            batch * self.heads, length, self.head_dim,
        )

    def forward(self, hidden_states, relative_embeddings, c2p_index, p2c_index):
        batch, length, width = hidden_states.shape
        query = self.heads_view(self.query_proj(hidden_states))
        key = self.heads_view(self.key_proj(hidden_states))
        value = self.heads_view(self.value_proj(hidden_states))
        position_query = self.heads_view(self.query_proj(relative_embeddings)).repeat(batch, 1, 1)
        position_key = self.heads_view(self.key_proj(relative_embeddings)).repeat(batch, 1, 1)
        scores = self.matmul(query, key.transpose(-1, -2) / self.scale)
        c2p = self.matmul(query, position_key.transpose(-1, -2))
        c2p = torch.gather(c2p, -1, c2p_index.expand(batch * self.heads, length, length)) / self.scale
        p2c = self.matmul(key, position_query.transpose(-1, -2))
        p2c = torch.gather(p2c, -1, p2c_index.expand(batch * self.heads, length, length)).transpose(-1, -2)
        relative_scores = c2p + p2c / self.scale
        context = self.matmul(self.softmax(scores + relative_scores), value)
        return context.view(batch, self.heads, length, self.head_dim).transpose(1, 2).reshape(batch, length, width)


class DebertaV2Layer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention.self = DisentangledSelfAttention(config)

    def forward(self, hidden_states, relative_embeddings, c2p_index, p2c_index):
        context = self.attention.self(hidden_states, relative_embeddings, c2p_index, p2c_index)
        attended = self.attention.output(context, hidden_states)
        return self.output(self.intermediate(attended), attended)


class DebertaV2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, input_ids):
        return self.LayerNorm(self.word_embeddings(input_ids))


class InputConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = Conv1dNative(config.hidden_size, config.hidden_size, config.conv_kernel_size,
                                 padding=(config.conv_kernel_size - 1) // 2)
        self.activation = GELU()
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, input_states, residual_states):
        convolved = self.conv(input_states.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return self.LayerNorm(residual_states + self.activation(convolved))


class DebertaV2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_buckets = config.position_buckets
        self.max_positions = config.max_position_embeddings
        self.rel_embeddings = Embedding(2 * config.position_buckets, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.conv = InputConvolution(config)

    def forward(self, hidden_states):
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        relative = positions[:, None] - positions[None, :]
        mid = self.position_buckets // 2
        magnitude = torch.where((relative < mid) & (relative > -mid), mid - 1, relative.abs())
        denominator = torch.log(torch.tensor((self.max_positions - 1) / mid, device=hidden_states.device))
        log_position = torch.ceil(torch.log(magnitude / mid) / denominator * (mid - 1)) + mid
        bucket = torch.where(magnitude <= mid, relative, log_position * relative.sign()).long()
        c2p_index = (bucket + self.position_buckets).clamp(0, 2 * self.position_buckets - 1).unsqueeze(0)
        p2c_index = (-bucket + self.position_buckets).clamp(0, 2 * self.position_buckets - 1).unsqueeze(0)
        relative_embeddings = self.LayerNorm(self.rel_embeddings.emb.weight).unsqueeze(0)
        input_states = hidden_states
        for index, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states, relative_embeddings, c2p_index, p2c_index)
            if index == 0:
                hidden_states = self.conv(input_states, hidden_states)
        return hidden_states


class DebertaV2ForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        return self.lm_head(self.encoder(self.embeddings(input_ids)))


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or not config.legacy or not config.relative_attention
            or set(config.pos_att_type) != {"p2c", "c2p"} or not config.share_att_key
            or config.position_biased_input or config.type_vocab_size != 0
            or getattr(config, "embedding_size", config.hidden_size) != config.hidden_size
            or config.hidden_size != config.num_attention_heads * config.attention_head_size
            or config.norm_rel_ebd != "layer_norm" or config.position_buckets != 256
            or config.max_relative_positions != -1 or config.conv_kernel_size != 3
            or config.conv_act != "gelu" or getattr(config, "conv_groups", 1) != 1):
        raise ValueError("DeBERTa-v2 coverage preserves its xlarge checkpoint's default masked-LM graph")
    return DebertaV2ForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    for name, parameter in model.named_parameters():
        if not name.startswith("lm_head."):
            source = "deberta." + name.replace(".emb.weight", ".weight")
            parameter.copy_(state_dict[source])
    load_mlm_head(model.lm_head, state_dict)
