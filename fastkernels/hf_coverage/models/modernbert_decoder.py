"""ModernBERT causal decoder with the existing global/local attention construction."""

import torch
from ..runner import Workload
from .modernbert import build_modern, load_state_dict_into


def build_from_config(config, device, dtype):
    if not config.use_cache:
        raise ValueError('ModernBERT decoder case retains default cached continuation')
    return build_modern(config, device, dtype, causal=True)


def make_workloads(model, inputs, config, *, case=None):
    ids = inputs['input_ids']
    continuation = case is not None and case['workload'] == 'causal_lm_continuation'
    steps = 2 if continuation else 1
    prefix_length = ids.shape[1] - steps
    if prefix_length < 1:
        raise ValueError('ModernBERT continuation requires a nonempty prefix')
    prompt = ids[:, :prefix_length]
    positions = [torch.tensor([prefix_length + index], device=ids.device) for index in range(steps)]
    state = {}

    def prefill():
        logits, state['cache'] = model(prompt)
        return {'logits': logits}

    def decode(index):
        token = ids[:, prefix_length + index:prefix_length + index + 1]
        logits, state['cache'] = model(token, positions[index], state['cache'])
        return {'logits': logits}

    def prepare(index):
        prefill()
        for earlier in range(index):
            decode(earlier)

    def collect(output):
        # Native caches expose [batch, heads, sequence, width]; only reorder
        # the actual saved state, including each local layer's truncated cache.
        for index, (key, value) in enumerate(state['cache']):
            output[f'past_key_values.{index}.key'] = key.transpose(1, 2)
            output[f'past_key_values.{index}.value'] = value.transpose(1, 2)
        return output

    workloads = {'prefill': Workload(run=prefill)}
    for index in range(steps):
        name = f'decode_{index + 1}' if continuation else 'decode'
        workloads[name] = Workload(
            run=lambda index=index: decode(index),
            prepare=lambda index=index: prepare(index),
        )
    if continuation:
        for workload in workloads.values():
            workload.collect = collect
    return workloads
