"""Local-attention Reformer MLM from its documented illustrative checkpoint."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from .bert import make_workloads


class LocalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.attention_head_size
        self.chunk = config.local_attn_chunk_length
        self.before, self.after = config.local_num_chunks_before, config.local_num_chunks_after
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.qkv = Linear(config.hidden_size, 3 * self.heads * self.width, bias=False)
        self.proj = Linear(self.heads * self.width, config.hidden_size, bias=False)
        self.bmm, self.softmax = BatchMatMul(), Softmax()
        self.mlp = VitEncoderMlp(config.hidden_size, config.feed_forward_size)

    def forward(self, left, right):
        batch, length = right.shape[:2]
        q, k, v = [x.reshape(batch, length, self.heads, self.width).transpose(1, 2)
                   for x in self.qkv(self.norm1(right)).chunk(3, -1)]
        k = k / self.width**0.5
        if length > self.chunk:
            q, k, v = [x.reshape(batch, self.heads, -1, self.chunk, self.width) for x in (q, k, v)]
            # HF includes cyclic neighboring chunks, including last-to-first.
            indices = torch.arange(q.shape[2], device=q.device)
            gather = torch.stack([(indices + offset) % q.shape[2]
                                  for offset in range(-self.before, self.after + 1)], dim=1)
            k = k[:, :, gather].flatten(3, 4)
            v = v[:, :, gather].flatten(3, 4)
        qlength, klength = q.shape[-2], k.shape[-2]
        scores = self.bmm(q.reshape(-1, qlength, self.width), k.reshape(-1, klength, self.width).transpose(1, 2))
        attended = self.bmm(self.softmax(scores), v.reshape(-1, klength, self.width))
        attended = attended.reshape(batch, self.heads, length, self.width).transpose(1, 2).reshape(batch, length, -1)
        left = left + self.proj(attended)
        return left, right + self.mlp(self.norm2(left))


class ReformerForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.axial_shape = tuple(config.axial_pos_shape)
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.axial_weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.axial_shape[0], 1, config.axial_pos_embds_dim[0])),
            nn.Parameter(torch.empty(1, self.axial_shape[1], config.axial_pos_embds_dim[1])),
        ])
        self.layers = nn.ModuleList([LocalLayer(config) for _ in config.attn_layers])
        self.final_norm = LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.decoder = Linear(2 * config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        length = input_ids.shape[1]
        if length % self.layers[0].chunk or length > self.axial_shape[0] * self.axial_shape[1]:
            raise ValueError('Selected workload requires complete attention chunks within the axial position table')
        positions = torch.cat([w.expand(*self.axial_shape, w.shape[-1]) for w in self.axial_weights], -1).flatten(0, 1)
        hidden = self.word_embeddings(input_ids) + positions[:length]
        left, right = hidden, hidden
        for layer in self.layers:
            left, right = layer(left, right)
        return self.decoder(self.final_norm(torch.cat((left, right), dim=-1)))


def build_from_config(config, device, dtype):
    if (set(config.attn_layers) != {'local'} or not config.axial_pos_embds or config.is_decoder
            or config.hidden_act != 'gelu' or config.tie_word_embeddings or len(config.axial_pos_shape) != 2):
        raise ValueError('The documented masked-LM checkpoint uses local encoder attention and two axial position factors')
    return ReformerForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    # Pinned ReformerOnlyLMHead registers this bias but never uses it in forward.
    remaining.pop('lm_head.bias')
    names = {'norm1': 'attention.layer_norm', 'norm2': 'feed_forward.layer_norm',
             'proj': 'attention.output.dense', 'mlp.fc1': 'feed_forward.dense.dense',
             'mlp.fc2': 'feed_forward.output.dense'}
    for name in model.state_dict():
        if name.startswith('layers.'):
            _, index, rest = name.split('.', 2)
            part, field = rest.rsplit('.', 1)
            prefix = f'reformer.encoder.layers.{index}.'
            if part == 'qkv':
                weights[name] = torch.cat([remaining.pop(prefix + f'attention.self_attention.{p}.{field}') for p in ('query', 'key', 'value')])
                continue
            source = prefix + names[part] + '.' + field
        elif name.startswith('axial_weights.'):
            source = 'reformer.embeddings.position_embeddings.weights.' + name.split('.')[-1]
        elif name.startswith('final_norm.'):
            source = 'reformer.encoder.layer_norm.' + name.split('.')[-1]
        elif name.startswith('decoder.'):
            source = 'lm_head.' + name
        else:
            source = 'reformer.embeddings.' + name.replace('.emb.weight', '.weight')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped Reformer parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)
