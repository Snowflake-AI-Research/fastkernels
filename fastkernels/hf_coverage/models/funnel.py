"""Funnel MLM: relative attention, progressive pooling and decoder expansion."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from .bert import make_workloads
from ..patches.query_bias_bmm import BiasedQueryBMM


def position_tables(length, width, blocks, device, dtype):
    """Fixed position metadata, including HF's dtype-dependent table rounding."""
    frequency = torch.arange(width // 2, device=device).to(dtype)
    frequency = 1 / (10000 ** (frequency / (width // 2)))
    positions = torch.arange(-2 * length, 2 * length, device=device).to(dtype)
    angles = positions[:, None] * frequency[None]
    table = torch.cat((angles.sin(), angles.cos()), dim=-1)
    positions = torch.arange(length, device=device).to(dtype)
    tables = []

    def select(original, pooled, stride, shift):
        high = pooled[0] - original[0] + shift * len(pooled) * stride
        low = pooled[0] - original[-1]
        indices = torch.arange(high, low - 1, -stride, dtype=torch.long, device=device)
        return table[indices + 2 * length]

    for index in range(blocks):
        cross = None
        if index:
            pooled = torch.cat((positions.new_tensor([1 - 2**index]), positions[1:-1:2]))
            cross = select(positions, pooled, 2**(index - 1), 2)
            positions = pooled
        tables.append((select(positions, positions, 2**index, 1), cross))
    return tables


class FunnelLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.n_head, config.d_head
        self.q_head = Linear(config.d_model, self.heads * self.width, bias=False)
        self.k_head = Linear(config.d_model, self.heads * self.width)
        self.v_head = Linear(config.d_model, self.heads * self.width)
        self.r_w_bias = nn.Parameter(torch.empty(self.heads, self.width))
        self.r_r_bias = nn.Parameter(torch.empty(self.heads, self.width))
        self.r_s_bias = nn.Parameter(torch.empty(self.heads, self.width))
        self.r_kernel = nn.Parameter(torch.empty(config.d_model, self.heads, self.width))
        self.seg_embed = nn.Parameter(torch.empty(2, self.heads, self.width))
        self.bmm, self.matmul, self.softmax = BatchMatMul(), Matmul(), Softmax()
        self.biased_product = BiasedQueryBMM()
        self.post_proj = Linear(self.heads * self.width, config.d_model)
        self.attn_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.d_model, config.d_inner, act_approximate='tanh')
        # Preserve the intermediate BF16 rounding of HF's GELU-new formula.
        self.mlp.act = NewGELUActivation()
        self.ffn_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps, promote_fp32=False)

    def product(self, a, b, bias):
        # Each input is [batch, sequence, head, width].
        batch = a.shape[0]
        left = a.transpose(1, 2).reshape(batch * self.heads, a.shape[1], a.shape[-1])
        right = b.expand(batch, -1, -1, -1).transpose(1, 2).reshape(batch * self.heads, b.shape[1], b.shape[-1])
        bias = bias[None, :, None].expand(batch, -1, -1, -1).reshape(batch * self.heads, 1, self.width)
        return self.biased_product(left, right.transpose(1, 2), bias).reshape(batch, self.heads, a.shape[1], b.shape[1])

    def forward(self, query, key, relative):
        batch, qlength = query.shape[:2]
        klength = key.shape[1]
        q = self.q_head(query).reshape(batch, qlength, self.heads, self.width) * self.width**-0.5
        k = self.k_head(key).reshape(batch, klength, self.heads, self.width)
        v = self.v_head(key).reshape(batch, klength, self.heads, self.width)
        content = self.product(q, k, self.r_w_bias * self.width**-0.5)
        rweight = self.r_kernel.permute(1, 2, 0).reshape(self.heads * self.width, -1)
        r = self.matmul(relative, rweight).reshape(1, relative.shape[0], self.heads, self.width)
        positional = self.product(q, r, self.r_r_bias * self.width**-0.5)
        shift = 1 if qlength == klength else 2
        positional = positional.reshape(batch, self.heads, -1, qlength)[:, :, shift:]
        positional = positional.reshape(batch, self.heads, qlength, -1)[..., :klength]
        # Omitted token_type_ids are all zero, so HF selects the same-segment bin.
        segment = self.product(q, self.seg_embed[None], self.r_s_bias * self.width**-0.5)
        segment = segment[..., 1:].expand(batch, self.heads, qlength, klength)
        cls = (torch.arange(qlength, device=q.device)[:, None] == 0) | (torch.arange(klength, device=q.device)[None, :] == 0)
        scores = content + positional.masked_fill(cls, 0) + segment.masked_fill(cls, 0)
        probabilities = self.softmax(scores)
        context = self.bmm(probabilities.reshape(-1, qlength, klength),
                           v.transpose(1, 2).reshape(-1, klength, self.width))
        context = context.reshape(batch, self.heads, qlength, self.width).transpose(1, 2).reshape(batch, qlength, -1)
        hidden = self.attn_norm(query + self.post_proj(context))
        return self.ffn_norm(hidden + self.mlp(hidden))


class FunnelForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.d_model
        self.word_embeddings = Embedding(config.vocab_size, config.d_model)
        self.embed_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps, promote_fp32=False)
        self.blocks = nn.ModuleList([nn.ModuleList([FunnelLayer(config) for _ in range(size)]) for size in config.block_sizes])
        self.decoder = nn.ModuleList([FunnelLayer(config) for _ in range(config.num_decoder_layers)])
        self.pool = AvgPool2d((2, 1), stride=(2, 1), ceil_mode=True)
        self.lm_head = Linear(config.d_model, config.vocab_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.word_embeddings.emb.weight

    def forward(self, input_ids):
        length = input_ids.shape[1]
        hidden = self.embed_norm(self.word_embeddings(input_ids))
        tables = position_tables(length, self.width, len(self.blocks), hidden.device, hidden.dtype)
        for index, block in enumerate(self.blocks):
            pooled = self.pool(torch.cat((hidden[:, :1], hidden[:, :-1]), dim=1)[:, None])[:, 0] if index else hidden
            for layer_index, layer in enumerate(block):
                cross = index > 0 and layer_index == 0
                hidden = layer(pooled if cross else hidden, hidden, tables[index][1 if cross else 0])
            if not index:
                first = hidden
        stride = 2**(len(self.blocks) - 1)
        repeated = hidden[:, 1:].repeat_interleave(stride, dim=1)
        repeated = torch.cat((repeated, repeated.new_zeros(repeated.shape[0], stride - 1, self.width)), dim=1)
        hidden = torch.cat((hidden[:, :1], repeated[:, :length - 1]), dim=1) + first
        for layer in self.decoder:
            hidden = layer(hidden, hidden, tables[0][0])
        return self.lm_head(hidden)


def build_from_config(config, device, dtype):
    if (config.attention_type != 'relative_shift' or config.pooling_type != 'mean' or not config.pool_q_only
            or not config.separate_cls or not config.truncate_seq or config.hidden_act != 'gelu_new'
            or any(repeat != 1 for repeat in config.block_repeats)):
        raise ValueError('Selected Funnel default uses relative-shift attention, separate CLS, and mean query pooling')
    return FunnelForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    for name in model.state_dict():
        if name == 'word_embeddings.emb.weight':
            source = 'funnel.embeddings.word_embeddings.weight'
        elif name.startswith('embed_norm.'):
            source = name.replace('embed_norm.', 'funnel.embeddings.layer_norm.')
        elif name.startswith('lm_head.'):
            source = name
        else:
            parts = name.split('.')
            if parts[0] == 'blocks':
                prefix, rest = 'funnel.encoder.' + '.'.join(parts[:3]), '.'.join(parts[3:])
            else:
                prefix, rest = 'funnel.decoder.layers.' + parts[1], '.'.join(parts[2:])
            if rest.startswith('mlp.'):
                source = prefix + '.' + rest.replace('mlp.fc1', 'ffn.linear_1').replace('mlp.fc2', 'ffn.linear_2')
            elif rest.startswith('ffn_norm.'):
                source = prefix + '.' + rest.replace('ffn_norm', 'ffn.layer_norm')
            else:
                source = prefix + '.attention.' + rest.replace('attn_norm', 'layer_norm')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped Funnel parameters: {sorted(remaining)}')
    if config.tie_word_embeddings and not torch.equal(weights['lm_head.weight'], weights['word_embeddings.emb.weight']):
        raise ValueError('Tied Funnel weights disagree')
    model.load_state_dict(weights, strict=True)
