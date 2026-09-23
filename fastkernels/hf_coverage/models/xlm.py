"""XLM masked language modeling from existing attention, MLP and mask operations."""

import math

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from ..patches.product_gate import ProductGate
from .bert import make_workloads


class XLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.width = config.emb_dim // config.n_heads
        self.qkv = Linear(config.emb_dim, 3 * config.emb_dim)
        self.out_lin = Linear(config.emb_dim, config.emb_dim)
        self.attention = DenseAttention(backend='sdpa')
        self.norm1 = LayerNorm(config.emb_dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(config.emb_dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.emb_dim, 4 * config.emb_dim)

    def forward(self, hidden, mask):
        batch, length = hidden.shape[:2]
        q, k, v = [x.reshape(batch, length, self.heads, self.width) for x in self.qkv(hidden).chunk(3, -1)]
        context = self.attention(q / math.sqrt(self.width), k, v,
                                 softmax_scale=1.0, attn_mask=mask[:, None, None, :])
        hidden = self.norm1(hidden + self.out_lin(context.reshape(batch, length, -1)))
        return self.norm2(hidden + self.mlp(hidden))


class XLMWithLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embeddings = Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.emb_dim)
        self.layer_norm_emb = LayerNorm(config.emb_dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList([XLMLayer(config) for _ in range(config.n_layers)])
        self.mask_product = ProductGate()
        self.proj = Linear(config.emb_dim, config.vocab_size)
        if config.tie_word_embeddings:
            self.proj.weight = self.embeddings.emb.weight

    def forward(self, input_ids):
        # HF's omitted lengths count nonpadding IDs, then mask a contiguous suffix.
        lengths = (input_ids != self.padding_idx).sum(dim=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        mask = positions[None, :] < lengths[:, None]
        hidden = self.layer_norm_emb(self.embeddings(input_ids) + self.position_embeddings(positions))
        values = mask.unsqueeze(-1).expand_as(hidden).to(hidden.dtype)
        hidden = self.mask_product(torch.cat((hidden, values), dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, mask)
            hidden = self.mask_product(torch.cat((hidden, values), dim=-1))
        return self.proj(hidden)


def build_from_config(config, device, dtype):
    if (config.causal or not config.is_encoder or config.asm or not config.gelu_activation
            or config.sinusoidal_embeddings or config.n_langs != 1 or getattr(config, 'pre_norm', False)):
        raise ValueError('Selected XLM/Flaubert checkpoint requires post-norm, learned positions and monolingual MLM')
    return XLMWithLMHeadModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    if config.tie_word_embeddings and not torch.equal(state_dict['pred_layer.proj.weight'], state_dict['transformer.embeddings.weight']):
        raise ValueError('Tied XLM embedding and head weights disagree')
    remaining = dict(state_dict)
    weights = {}
    for name in model.state_dict():
        if name.startswith('layers.'):
            _, index, rest = name.split('.', 2)
            part, field = rest.rsplit('.', 1)
            if part == 'qkv':
                weights[name] = torch.cat([remaining.pop(f'transformer.attentions.{index}.{p}_lin.{field}')
                                          for p in ('q', 'k', 'v')])
                continue
            module = {'out_lin': f'attentions.{index}.out_lin', 'norm1': f'layer_norm1.{index}',
                      'norm2': f'layer_norm2.{index}',
                      'mlp.fc1': f'ffns.{index}.lin1', 'mlp.fc2': f'ffns.{index}.lin2'}[part]
            source = f'transformer.{module}.{field}'
        elif name.startswith('proj.'):
            source = 'pred_layer.' + name
        else:
            source = 'transformer.' + name.replace('.emb.weight', '.weight')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped XLM weights: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)
