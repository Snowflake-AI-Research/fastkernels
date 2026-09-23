"""ConvBertForMaskedLM with dense attention and the dynamic span-convolution branch."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_model import BertModel

from ..patches.product_gate import ProductGate
from .bert import MaskedLMHead, make_workloads


class ConvBertEmbeddings(BertEmbeddings):
    def forward_with_token_type_ids(self, input_ids, position_ids, token_type_ids=None, inputs_embeds=None):
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :input_ids.shape[1]].expand_as(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + self.position_embeddings(position_ids)
        return self.LayerNorm(embeddings + self.token_type_embeddings(token_type_ids))


class SeparableConv1D(nn.Module):
    def __init__(self, input_width, output_width, kernel_size):
        super().__init__()
        self.depthwise = Conv1dNative(input_width, input_width, kernel_size,
                                     groups=input_width, padding=kernel_size // 2, bias=False)
        self.pointwise = Conv1dNative(input_width, output_width, 1, bias=False)
        self.bias = nn.Parameter(torch.empty(output_width, 1))

    def forward(self, hidden_states):
        return self.pointwise(self.depthwise(hidden_states)) + self.bias


class ConvBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads // config.head_ratio
        self.head_dim = config.hidden_size // self.heads // 2
        self.width = self.heads * self.head_dim
        self.kernel_size = config.conv_kernel_size
        self.query = Linear(config.hidden_size, self.width)
        self.key = Linear(config.hidden_size, self.width)
        self.value = Linear(config.hidden_size, self.width)
        self.key_conv_attn_layer = SeparableConv1D(config.hidden_size, self.width, self.kernel_size)
        self.conv_kernel_layer = Linear(self.width, self.heads * self.kernel_size)
        self.conv_out_layer = Linear(config.hidden_size, self.width)
        self.gate = ProductGate()
        self.matmul = BMM()
        self.softmax = Softmax()
        self.kernel_softmax = Softmax(dim=1)

    def forward_with_attention_mask(self, hidden_states, attention_mask=None):
        batch, length, _ = hidden_states.shape
        query = self.query(hidden_states)
        key = self.key(hidden_states).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        conv_key = self.key_conv_attn_layer(hidden_states.transpose(1, 2)).transpose(1, 2)
        gated = self.gate(torch.cat((conv_key, query), dim=-1))
        kernels = self.conv_kernel_layer(gated).reshape(-1, self.kernel_size, 1)
        kernels = self.kernel_softmax(kernels)
        projected = self.conv_out_layer(hidden_states).transpose(1, 2).contiguous().unsqueeze(-1)
        windows = F.unfold(projected, kernel_size=(self.kernel_size, 1),
                           padding=((self.kernel_size - 1) // 2, 0))
        windows = windows.transpose(1, 2).reshape(batch, length, self.width, self.kernel_size)
        convolved = self.matmul(windows.reshape(-1, self.head_dim, self.kernel_size), kernels)
        convolved = convolved.reshape(batch, length, self.heads, self.head_dim)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scores = self.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        context = self.matmul(self.softmax(scores), value).transpose(1, 2)
        return torch.cat((context, convolved), dim=2).reshape(batch, length, 2 * self.width)


class ConvBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convbert = BertModel(config)
        self.convbert.embeddings = ConvBertEmbeddings(config)
        for layer in self.convbert.encoder.layer:
            layer.attention.self = ConvBertSelfAttention(config)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.convbert.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        return self.lm_head(self.convbert.forward_with_attention_mask(input_ids))


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.embedding_size != config.hidden_size
            or config.num_groups != 1 or config.head_ratio != 2 or config.conv_kernel_size != 9
            or config.num_attention_heads < 2 or config.num_attention_heads % 2
            or config.is_decoder or config.add_cross_attention):
        raise ValueError("ConvBERT coverage preserves its base checkpoint's masked-LM computation")
    return ConvBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        source = source.replace("lm_head.dense.", "generator_predictions.dense.")
        source = source.replace("lm_head.LayerNorm.", "generator_predictions.LayerNorm.")
        source = source.replace("lm_head.decoder.", "generator_lm_head.")
        parameter.copy_(state_dict[source])
