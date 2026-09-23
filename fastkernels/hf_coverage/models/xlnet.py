"""XLNet's bidirectional content stream with reusable, updated layer memories."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.gelu import GELU
from ..patches.query_bias_bmm import BiasedQueryBMM
from ..runner import Workload


class XLNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.n_head, config.d_head
        for name in ('q', 'k', 'v', 'r', 'o'):
            setattr(self, name, Linear(config.d_model, config.d_model, bias=False))
        self.content_bias = nn.Parameter(torch.empty(config.n_head, config.d_head))
        self.relative_bias = nn.Parameter(torch.empty(config.n_head, config.d_head))
        self.attn_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps, promote_fp32=False)
        self.ffn_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps, promote_fp32=False)
        self.fc1, self.fc2 = Linear(config.d_model, config.d_inner), Linear(config.d_inner, config.d_model)
        self.activation, self.softmax, self.bmm = GELU(), Softmax(), BatchMatMul()
        self.biased_product = BiasedQueryBMM()

    def forward(self, hidden, relative, memory=None):
        batch, length = hidden.shape[:2]
        def heads(tensor):
            return tensor.view(batch, -1, self.heads, self.width).transpose(1, 2).reshape(batch * self.heads, -1, self.width)
        context = hidden if memory is None else torch.cat((memory.transpose(0, 1), hidden), dim=1)
        q, k, v = heads(self.q(hidden)), heads(self.k(context)), heads(self.v(context))
        r = heads(self.r(relative.to(self.r.weight.dtype)).expand(batch, -1, -1))
        def bias(tensor):
            return tensor[None, :, None].expand(batch, -1, -1, -1).reshape(batch * self.heads, 1, self.width)
        content = self.biased_product(q, k.transpose(1, 2), bias(self.content_bias))
        positional = self.biased_product(q, r.transpose(1, 2), bias(self.relative_bias))
        relative_length = relative.shape[1]
        positional = positional.reshape(batch * self.heads, relative_length, length)[:, 1:]
        positional = positional.reshape(batch * self.heads, length, relative_length - 1)[:, :, :k.shape[1]]
        scores = (content + positional) * self.width**-0.5
        context = self.bmm(self.softmax(scores), v)
        context = context.view(batch, self.heads, length, self.width).transpose(1, 2).reshape_as(hidden)
        hidden = self.attn_norm(self.o(context) + hidden)
        return self.ffn_norm(self.fc2(self.activation(self.fc1(hidden))) + hidden)


class XLNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width, self.mem_len, self.reuse_len = config.d_model, config.mem_len, config.reuse_len
        self.word_embedding = Embedding(config.vocab_size, config.d_model)
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])

    def forward(self, input_ids, mems=None):
        hidden = self.word_embedding(input_ids)
        length = input_ids.shape[1]
        memory_length = 0 if mems is None else mems[0].shape[0]
        # Relative positions are metadata, with HF's explicit FP32 construction.
        frequency = torch.arange(0, self.width, 2, device=input_ids.device).float()
        inverse = 1 / (10000 ** (frequency / self.width))
        positions = torch.arange(length + memory_length, -length, -1, device=input_ids.device).float()
        phase = positions[:, None] * inverse[None]
        relative = torch.cat((phase.sin(), phase.cos()), dim=-1)[None]
        output = {}
        for index, layer in enumerate(self.layer):
            memory = hidden.transpose(0, 1)
            if self.reuse_len is not None and self.reuse_len > 0:
                memory = memory[:self.reuse_len]
            previous = None if mems is None else mems[index]
            if previous is not None:
                memory = torch.cat((previous, memory), dim=0)
            if self.mem_len:
                memory = memory[-self.mem_len:]
            output[f'mems.{index}'] = memory.detach()
            hidden = layer(hidden, relative, previous)
        output['last_hidden_state'] = hidden
        return output


def build_from_config(config, device, dtype):
    if (config.attn_type != 'bi' or config.bi_data or config.clamp_len > 0
            or config.ff_activation != 'gelu' or not config.use_mems_eval):
        raise ValueError('Selected XLNet path is bidirectional GELU content attention with default output memories')
    return XLNetModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    remaining.pop('mask_emb')  # Used only by optional target_mapping/two-stream calls.
    for index in range(config.n_layer):
        remaining.pop(f'layer.{index}.rel_attn.r_s_bias')
        remaining.pop(f'layer.{index}.rel_attn.seg_embed')  # Optional token_type_ids absent.
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if name.startswith('layer.'):
            prefix, part = name.rsplit('.', 1)
            for ours, original in (('.attn_norm.', '.rel_attn.layer_norm.'), ('.ffn_norm.', '.ff.layer_norm.'),
                                   ('.fc1.', '.ff.layer_1.'), ('.fc2.', '.ff.layer_2.')):
                source = source.replace(ours, original)
            if part in ('content_bias', 'relative_bias'):
                source = prefix + '.rel_attn.' + ('r_w_bias' if part == 'content_bias' else 'r_r_bias')
            elif prefix.rsplit('.', 1)[-1] in ('q', 'k', 'v', 'r', 'o'):
                base, projection = prefix.rsplit('.', 1)
                tensor = remaining.pop(f'{base}.rel_attn.{projection}').reshape(config.d_model, config.d_model)
                weights[name] = tensor if projection == 'o' else tensor.t().contiguous()
                continue
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped XLNet parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if not case or case['workload'] != 'memory_continuation':
        return {'forward': Workload(run=lambda: model(**inputs))}

    ids = inputs['input_ids']
    prefix, tokens = ids[:, :-2], ids[:, -2:]
    state = {}

    def call(token_ids, previous=None):
        output = model(token_ids, previous)
        state['memory'] = tuple(output[f'mems.{index}'] for index in range(config.n_layer))
        return output

    def prepare(index):
        call(prefix)
        for earlier in range(index):
            call(tokens[:, earlier:earlier + 1], state['memory'])

    workloads = {'forward': Workload(run=lambda: call(prefix))}
    for index in range(2):
        workloads[f'continuation_{index + 1}'] = Workload(
            run=lambda index=index: call(tokens[:, index:index + 1], state['memory']),
            prepare=lambda index=index: prepare(index),
        )
    return workloads
