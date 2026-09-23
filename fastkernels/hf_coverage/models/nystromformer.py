"""The published checkpoint's dense-attention branch and convolutional skip."""

import copy
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from .bert import MaskedLMHead, make_workloads


class NystromLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size)
        self.matmul = BatchMatMul()
        self.softmax = Softmax()
        self.conv = Conv2d(self.heads, self.heads, (config.conv_kernel_size, 1),
                           padding=(config.conv_kernel_size // 2, 0), groups=self.heads, bias=False)
        self.proj = Linear(config.hidden_size, config.hidden_size)
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, act_approximate='tanh')
        self.mlp.act = NewGELUActivation()

    def forward(self, hidden):
        batch, length = hidden.shape[:2]
        q, k, v = [part.reshape(batch, length, self.heads, self.width).transpose(1, 2)
                   for part in self.qkv(hidden).chunk(3, -1)]
        q, k = q / self.width**0.25, k / self.width**0.25
        scores = self.matmul(q.reshape(-1, length, self.width), k.reshape(-1, length, self.width).transpose(1, 2))
        context = self.matmul(self.softmax(scores), v.reshape(-1, length, self.width))
        context = context.reshape(batch, self.heads, length, self.width) + self.conv(v)
        hidden = self.norm1(hidden + self.proj(context.transpose(1, 2).reshape(batch, length, -1)))
        return self.norm2(hidden + self.mlp(hidden))


class NystromformerForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = copy.copy(config)
        embedding_config.max_position_embeddings += 2
        self.embeddings = BertEmbeddings(embedding_config)
        self.layers = nn.ModuleList([NystromLayer(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = MaskedLMHead(config)
        self.lm_head.activation = NewGELUActivation()
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] + 2
        hidden = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def build_from_config(config, device, dtype):
    if (config.num_landmarks != config.segment_means_seq_len or config.hidden_act != 'gelu_new'
            or config.conv_kernel_size is None or config.add_cross_attention):
        raise ValueError('Published nystromformer-512 config selects equal landmark/segment counts and a convolutional skip')
    return NystromformerForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    if not torch.equal(remaining['cls.predictions.bias'], remaining['cls.predictions.decoder.bias']):
        raise ValueError('Masked-LM bias aliases disagree')
    remaining.pop('cls.predictions.bias')
    names = {'conv': 'attention.self.conv', 'proj': 'attention.output.dense',
             'norm1': 'attention.output.LayerNorm', 'norm2': 'output.LayerNorm',
             'mlp.fc1': 'intermediate.dense', 'mlp.fc2': 'output.dense'}
    weights = {}
    for name in model.state_dict():
        if name.startswith('layers.'):
            _, index, rest = name.split('.', 2)
            part, field = rest.rsplit('.', 1)
            prefix = f'nystromformer.encoder.layer.{index}.'
            if part == 'qkv':
                weights[name] = torch.cat([remaining.pop(prefix + f'attention.self.{p}.{field}') for p in ('query', 'key', 'value')])
                continue
            source = prefix + names[part] + '.' + field
        elif name.startswith('lm_head.'):
            rest = name[len('lm_head.'):]
            source = ('cls.predictions.' if rest.startswith('decoder.') else 'cls.predictions.transform.') + rest
        else:
            source = 'nystromformer.' + name.replace('.emb.weight', '.weight')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped Nystromformer weights: {sorted(remaining)}')
    if config.tie_word_embeddings and not torch.equal(weights['lm_head.decoder.weight'], weights['embeddings.word_embeddings.emb.weight']):
        raise ValueError('Tied Nystromformer weights disagree')
    model.load_state_dict(weights, strict=True)
