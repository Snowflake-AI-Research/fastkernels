"""MVP without optional learned prompts, preserving eager query scaling."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from .bart import build_from_config as build_bart, load_state_dict_into
from .plbart import make_workloads


class EagerAttention(nn.Module):
    """Existing BMM/softmax with query or score scaling at HF's rounding boundary."""

    def __init__(self, *, prescale_query=True, divide_scores=False):
        super().__init__()
        self.matmul = BatchMatMul()
        self.softmax = Softmax(dim=-1)
        self.prescale_query = prescale_query
        self.divide_scores = divide_scores

    def forward(self, query, key, value, causal=False, attn_mask=None, return_weights=False):
        batch, length, heads, width = query.shape
        if self.prescale_query:
            query = query * (width ** -0.5)
        query = query.transpose(1, 2).reshape(batch * heads, length, width)
        key = key.transpose(1, 2).reshape(batch * heads, -1, width)
        value = value.transpose(1, 2).reshape(batch * heads, -1, width)
        scores = self.matmul(query, key.transpose(1, 2)).view(batch, heads, length, -1)
        if not self.prescale_query:
            scores = scores / (width ** 0.5) if self.divide_scores else scores * (width ** -0.5)
        if causal:
            positions = torch.arange(length, device=query.device)
            mask = torch.zeros(length, key.shape[1], device=query.device, dtype=query.dtype)
            mask.masked_fill_(torch.arange(key.shape[1], device=query.device)[None, :] > positions[:, None],
                              torch.finfo(query.dtype).min)
            scores = scores + mask
        if attn_mask is not None:
            scores = scores + attn_mask
        probabilities = self.softmax(scores).reshape(batch * heads, length, -1)
        context = self.matmul(probabilities, value).view(batch, heads, length, width).transpose(1, 2)
        if return_weights:
            return context, probabilities.view(batch, heads, length, -1)
        return context


def build_from_config(config, device, dtype):
    if config.use_prompt:
        raise ValueError("The selected ordinary MVP checkpoint disables optional prompts")
    model = build_bart(config, device, dtype)
    for layer in model.encoder.layers.layer:
        layer.attention.self.attn = EagerAttention()
    for layer in model.decoder.layers:
        layer.attention.self.attn = EagerAttention()
        layer.cross_attention.attention = EagerAttention()
    return model
