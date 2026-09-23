"""MPNetForMaskedLM with its learned relative-position bias and BERT blocks."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfAttention
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from .bert import MaskedLMHead, make_workloads


class MPNetSelfAttention(EncoderSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.matmul = BMM()
        self.softmax = Softmax()

    def forward_with_attention_mask(self, hidden_states, attention_mask=None):
        batch, length, width = hidden_states.shape
        query, key, value = (
            tensor.view(batch, length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
            for tensor in self._project_qkv(hidden_states)
        )
        scores = self.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        context = self.matmul(self.softmax(scores + attention_mask), value)
        return context.transpose(1, 2).reshape(batch, length, width)


class MPNetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=1)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=1)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, input_ids):
        mask = input_ids.ne(1).int()
        positions = (torch.cumsum(mask, dim=1) * mask).long() + 1
        return self.LayerNorm(self.word_embeddings(input_ids) + self.position_embeddings(positions))


class MPNetForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = MPNetEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.encoder.relative_attention_bias = Embedding(32, config.num_attention_heads)
        for layer in self.encoder.layer:
            layer.attention.self = MPNetSelfAttention(config)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        distance = positions[:, None] - positions[None, :]
        magnitude = distance.abs()
        large = 8 + (torch.log(magnitude.float() / 8) / math.log(128 / 8) * 8).long()
        buckets = (distance < 0).long() * 16 + torch.where(magnitude < 8, magnitude, large.clamp(max=15))
        bias = self.encoder.relative_attention_bias(buckets).permute(2, 0, 1).unsqueeze(0)
        bias = bias.expand(input_ids.shape[0], -1, length, length).contiguous()
        return self.lm_head(self.encoder.forward_with_attention_mask(hidden_states, bias))


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.relative_attention_num_buckets != 32 or config.pad_token_id != 1:
        raise ValueError("MPNet coverage preserves the masked-LM checkpoint's default computation")
    return MPNetForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("lm_head."):
            source = source.replace(".LayerNorm.", ".layer_norm.")
        else:
            source = "mpnet." + source
        if ".qkv." in source:
            parameter.copy_(torch.cat([
                state_dict[source.replace(".attention.self.qkv.", f".attention.attn.{projection}.")]
                for projection in ("q", "k", "v")
            ]))
        else:
            source = source.replace(".attention.output.dense.", ".attention.attn.o.")
            source = source.replace(".attention.output.LayerNorm.", ".attention.LayerNorm.")
            parameter.copy_(state_dict[source])
