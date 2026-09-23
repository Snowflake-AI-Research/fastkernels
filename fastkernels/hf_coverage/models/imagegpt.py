"""ImageGPT hidden states and cached continuation using existing operations."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from ..patches.imagegpt_rms_norm import InputDtypeRMSNorm
from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU
from .mvp import EagerAttention
from ..runner import Workload


class ImageGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_head
        self.width = config.n_embd // config.n_head
        self.ln_1 = InputDtypeRMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = InputDtypeRMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.qkv, self.proj = Linear(config.n_embd, 3 * config.n_embd), Linear(config.n_embd, config.n_embd)
        self.fc1 = Linear(config.n_embd, config.n_inner or 4 * config.n_embd)
        self.fc2 = Linear(config.n_inner or 4 * config.n_embd, config.n_embd)
        self.activation = QuickGELU()
        self.attention = EagerAttention(prescale_query=False, divide_scores=True)

    def forward(self, hidden, past_key_value=None):
        batch, length = hidden.shape[:2]
        q, k, v = (part.view(batch, length, self.heads, self.width) for part in self.qkv(self.ln_1(hidden)).chunk(3, dim=-1))
        cache = (k.transpose(1, 2).clone(memory_format=torch.contiguous_format),
                 v.transpose(1, 2).clone(memory_format=torch.contiguous_format))
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        if past_key_value is not None:
            cache = tuple(torch.cat((previous, current), dim=2)
                          for previous, current in zip(past_key_value, cache))
        queries = torch.arange(length, device=hidden.device) + past_length
        keys = torch.arange(cache[0].shape[2], device=hidden.device)
        mask = hidden.new_zeros(length, keys.numel()).masked_fill(
            keys[None, :] > queries[:, None], torch.finfo(hidden.dtype).min)
        context = self.attention(q, cache[0].transpose(1, 2), cache[1].transpose(1, 2),
                                 attn_mask=mask)
        hidden = hidden + self.proj(context.reshape_as(hidden))
        hidden = hidden + self.fc2(self.activation(self.fc1(self.ln_2(hidden))))
        return hidden, cache


class ImageGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte, self.wpe = Embedding(config.vocab_size, config.n_embd), Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([ImageGPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = InputDtypeRMSNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, past_key_values=None):
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + past_length
        hidden = self.wte(input_ids) + self.wpe(positions)
        caches = []
        for index, layer in enumerate(self.h):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, cache = layer(hidden, previous)
            caches.append(cache)
        return {'last_hidden_state': self.ln_f(hidden), 'past_key_values': tuple(caches)}


def build_from_config(config, device, dtype):
    if (config.activation_function != 'quick_gelu' or config.add_cross_attention
            or not config.use_cache or not config.scale_attn_weights
            or config.scale_attn_by_inverse_layer_idx or config.reorder_and_upcast_attn):
        raise ValueError('Selected ImageGPT uses quick GELU, standard causal attention and caches')
    return ImageGPTModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        for ours, original in (('.qkv.', '.attn.c_attn.'), ('.proj.', '.attn.c_proj.'),
                               ('.fc1.', '.mlp.c_fc.'), ('.fc2.', '.mlp.c_proj.')):
            source = source.replace(ours, original)
        tensor = remaining.pop(source)
        if source.endswith('.weight') and ('.attn.' in source or '.mlp.' in source):
            tensor = tensor.t().contiguous()
        weights[name] = tensor
    if remaining:
        raise KeyError(f'Unmapped ImageGPT parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    state = {}

    def retain(output):
        state['output'] = output
        return {'last_hidden_state': output['last_hidden_state']}

    def collect(_output):
        output = state.pop('output')
        result = {'last_hidden_state': output['last_hidden_state']}
        for index, (key, value) in enumerate(output['past_key_values']):
            result[f'past_key_values.{index}.key'] = key
            result[f'past_key_values.{index}.value'] = value
        return result

    if case is None or case['workload'] != 'causal_lm_continuation':
        return {'forward': Workload(run=lambda: retain(model(**inputs)), collect=collect)}

    ids = inputs['input_ids']
    prefix_length = ids.shape[1] - 2

    def prepare_step(index):
        output = model(ids[:, :prefix_length])
        for previous in range(index):
            start = prefix_length + previous
            output = model(ids[:, start:start + 1], output['past_key_values'])
        state['cache'] = output['past_key_values']

    workloads = {'prefill': Workload(run=lambda: retain(model(ids[:, :prefix_length])), collect=collect)}
    for index in range(2):
        start = prefix_length + index
        workloads[f'decode_{index + 1}'] = Workload(
            run=lambda start=start: retain(model(ids[:, start:start + 1], state['cache'])),
            prepare=lambda index=index: prepare_step(index), collect=collect,
        )
    return workloads
