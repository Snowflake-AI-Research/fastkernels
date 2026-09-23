"""Longformer local windows; the shared attention also supports LED's global start token."""

import math
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_embeddings import XLMRobertaEmbeddings
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from .bert import MaskedLMHead
from ..runner import Workload


class WindowAttention(nn.Module):
    """Block queries of width w attend to three adjacent blocks, with exact window masks.

    Only three copies of K/V are viewed per block; no per-query K/V expansion.
    Global-first mode preserves the independent global Q/K/V projections.
    """
    def __init__(self, hidden_size, heads, window, global_first=False):
        super().__init__()
        self.heads, self.width, self.window = heads, hidden_size // heads, window // 2
        self.global_first = global_first
        names = ('query', 'key', 'value')
        if global_first:
            names += ('query_global', 'key_global', 'value_global')
        for name in names:
            setattr(self, name, Linear(hidden_size, hidden_size))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def project(self, module, hidden):
        return module(hidden).view(hidden.shape[0], hidden.shape[1], self.heads, self.width).transpose(1, 2)

    def forward(self, hidden, valid_length):
        batch, length, _ = hidden.shape
        w, heads, width = self.window, self.heads, self.width
        blocks = length // w
        q = self.project(self.query, hidden) / math.sqrt(width)
        k, v = self.project(self.key, hidden), self.project(self.value, hidden)
        query = q.reshape(batch * heads * blocks, w, width)
        def windows(tensor):
            padded = torch.cat((tensor.new_zeros(batch, heads, w, width), tensor,
                                tensor.new_zeros(batch, heads, w, width)), dim=2)
            return padded.unfold(2, 3 * w, w).transpose(-1, -2).reshape(-1, 3 * w, width)
        keys, values = windows(k), windows(v)
        scores = self.bmm(query, keys.transpose(1, 2)).view(batch, heads, blocks, w, 3 * w)
        qi = torch.arange(length, device=hidden.device).view(blocks, w, 1)
        ki = torch.arange(3 * w, device=hidden.device)[None, None] + torch.arange(blocks, device=hidden.device)[:, None, None] * w - w
        mask = (ki < 0) | (ki >= valid_length) | ((qi - ki).abs() > w)
        if self.global_first:
            mask = mask | ki.eq(0)
        scores = scores.masked_fill(mask, -torch.inf)
        if self.global_first:
            global_scores = self.bmm(q.reshape(batch * heads, length, width),
                                     k[:, :, :1].reshape(batch * heads, 1, width).transpose(1, 2))
            scores = torch.cat((global_scores.view(batch, heads, blocks, w, 1), scores), dim=-1)
        if self.global_first:
            # Reduce exactly the native local window plus global token. Extra
            # masked block positions change softmax rounding in deep LED stacks.
            indices = (torch.arange(w, device=hidden.device)[:, None]
                       + torch.arange(2 * w + 1, device=hidden.device)[None] + 1)
            indices = indices[None, None, None].expand(batch, heads, blocks, -1, -1)
            packed = torch.cat((scores[..., :1], scores.gather(-1, indices)), dim=-1)
            packed = packed.reshape(batch, heads, length, -1).transpose(1, 2).contiguous()
            packed = self.softmax(packed.float()).to(scores.dtype)
            packed = packed.transpose(1, 2).reshape(batch, heads, blocks, w, -1)
            probability = torch.zeros_like(scores)
            probability[..., :1] = packed[..., :1]
            probability.scatter_(-1, indices, packed[..., 1:])
        else:
            probability = self.softmax(scores.float()).to(scores.dtype)
        probability = probability.masked_fill((qi >= valid_length)[None, None], 0)
        local_probability = probability[..., 1:] if self.global_first else probability
        if not self.global_first:
            # HF's diagonal-to-window layout leaves one unused column per row.
            # Preserve that stride: a contiguous BMM selects different rounding.
            storage = local_probability.new_empty(*local_probability.shape[:-1], 3 * w + 1)
            storage[..., :-1] = local_probability
            local_probability = storage[..., :-1]
        context = self.bmm(local_probability.reshape(-1, w, 3 * w), values).view(batch, heads, length, width)
        if self.global_first:
            global_context = self.bmm(probability[..., :1].reshape(batch * heads, length, 1),
                                      v[:, :, :1].reshape(batch * heads, 1, width)).view_as(context)
            context = global_context + context
            global_q = self.project(self.query_global, hidden[:, :1]) / math.sqrt(width)
            global_k, global_v = self.project(self.key_global, hidden), self.project(self.value_global, hidden)
            global_scores = self.bmm(global_q.reshape(batch * heads, 1, width),
                                     global_k.reshape(batch * heads, length, width).transpose(1, 2))
            global_scores = global_scores.masked_fill(torch.arange(length, device=hidden.device) >= valid_length,
                                                       torch.finfo(global_scores.dtype).min)
            global_probability = self.softmax(global_scores.float()).to(global_scores.dtype)
            context[:, :, :1] = self.bmm(global_probability, global_v.reshape(batch * heads, length, width)).view(batch, heads, 1, width)
        return context.transpose(1, 2).reshape_as(hidden)


class LongformerLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.attn = WindowAttention(config.hidden_size, config.num_attention_heads, config.attention_window[index])
        self.attn_output = EncoderSelfOutput(config)
        self.intermediate, self.output = EncoderIntermediate(config), EncoderOutput(config)

    def forward(self, hidden, valid_length):
        hidden = self.attn_output(self.attn(hidden, valid_length), hidden)
        return self.output(self.intermediate(hidden), hidden)


class LongformerForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_id, self.window = config.pad_token_id, max(config.attention_window)
        self.embeddings = XLMRobertaEmbeddings(config)
        self.layers = nn.ModuleList([LongformerLayer(config, index) for index in range(config.num_hidden_layers)])
        self.lm_head = MaskedLMHead(config)
        self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        length = input_ids.shape[1]
        pad = (-length) % self.window
        ids = torch.cat((input_ids, input_ids.new_full((input_ids.shape[0], pad), self.pad_id)), dim=1)
        nonpadding = ids.ne(self.pad_id).long()
        positions = nonpadding.cumsum(-1) * nonpadding + self.pad_id
        hidden = self.embeddings.LayerNorm(self.embeddings.word_embeddings(ids)
                    + self.embeddings.position_embeddings(positions)
                    + self.embeddings.token_type_embeddings(torch.zeros_like(ids)))
        for layer in self.layers:
            hidden = layer(hidden, length)
        return {'logits': self.lm_head(hidden[:, :length])}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or not config.tie_word_embeddings:
        raise ValueError('Selected Longformer MLM uses GELU and its tied word decoder')
    return LongformerForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    # Global attention is omitted by the documented masked-LM call, so these are uncalled.
    for index in range(config.num_hidden_layers):
        for projection in ('query_global', 'key_global', 'value_global'):
            for field in ('weight', 'bias'):
                remaining.pop(f'longformer.encoder.layer.{index}.attention.self.{projection}.{field}')
    alias = remaining.pop('lm_head.bias')
    if not torch.equal(alias, remaining['lm_head.decoder.bias']):
        raise ValueError('Longformer tied decoder biases disagree')
    if not torch.equal(remaining['lm_head.decoder.weight'], remaining['longformer.embeddings.word_embeddings.weight']):
        raise ValueError('Longformer tied decoder weights disagree')
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if source.startswith('embeddings.'):
            source = 'longformer.' + source
        elif source.startswith('layers.'):
            source = source.replace('layers.', 'longformer.encoder.layer.', 1).replace('.attn.', '.attention.self.')
            source = source.replace('.attn_output.', '.attention.output.')
        else:
            source = source.replace('.LayerNorm.', '.layer_norm.')
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped Longformer parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
