"""Pixtral vision tokens with two-dimensional RoPE and per-image attention."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.softmax import Softmax


class _Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, Linear(c.hidden_size, c.hidden_size, bias=False))
        self.bmm, self.softmax = BMM(), Softmax()

    def forward(self, hidden, positions, table, mask):
        length = hidden.shape[1]
        q, k = RotaryEmbedding.forward_native(positions, self.q_proj(hidden).reshape(length, -1),
                                               self.k_proj(hidden).reshape(length, -1), self.head_dim, table)
        q = q.reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        scores = self.bmm(q, k.transpose(-1, -2)) * self.head_dim**-0.5 + mask
        probabilities = self.softmax(scores.float()).to(q.dtype)
        context = self.bmm(probabilities, v).transpose(1, 2).reshape(1, length, -1)
        return self.o_proj(context)


class _MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = Linear(c.intermediate_size, c.hidden_size, bias=False)
        self.activation = SiluAndMul()

    def forward(self, hidden):
        joined = torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), dim=-1)
        activated = self.activation(joined) if hidden.is_cuda else self.activation.forward_native(joined)
        return self.down_proj(activated)


class _Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention_norm = RMSNormNative(c.hidden_size, 1e-5)
        self.ffn_norm = RMSNormNative(c.hidden_size, 1e-5)
        self.attention, self.feed_forward = _Attention(c), _MLP(c)

    def forward(self, hidden, positions, table, mask):
        hidden = hidden + self.attention(self.attention_norm(hidden), positions, table, mask)
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class PixtralVisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_size, self.max_width = c.patch_size, c.image_size // c.patch_size
        self.patch_conv = Conv2d(c.num_channels, c.hidden_size, c.patch_size, stride=c.patch_size, bias=False)
        self.ln_pre = RMSNormNative(c.hidden_size, 1e-5)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([_Layer(c) for _ in range(c.num_hidden_layers)])
        dim = c.hidden_size // c.num_attention_heads
        frequencies = 1. / (c.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim))
        rows = torch.arange(self.max_width).float()[:, None] * frequencies[None, ::2]
        columns = torch.arange(self.max_width).float()[:, None] * frequencies[None, 1::2]
        angles = torch.cat((rows[:, None].expand(-1, self.max_width, -1),
                            columns[None].expand(self.max_width, -1, -1)), dim=-1).reshape(-1, dim//2)
        self.register_buffer("rotary_table", torch.cat((angles.cos(), angles.sin()), dim=-1), persistent=False)

    def forward(self, pixel_values):
        patches = self.patch_conv(pixel_values)
        batch, width, h, w = patches.shape
        hidden = self.ln_pre(patches.flatten(2).transpose(1, 2).reshape(1, batch*h*w, width))
        positions = (torch.arange(h, device=hidden.device)[:, None] * self.max_width +
                     torch.arange(w, device=hidden.device)[None]).flatten().repeat(batch)
        image_ids = torch.arange(batch, device=hidden.device).repeat_interleave(h*w)
        mask = hidden.new_zeros(batch*h*w, batch*h*w)
        mask.masked_fill_(image_ids[:, None] != image_ids[None, :], torch.finfo(hidden.dtype).min)
        for layer in self.transformer.layers:
            hidden = layer(hidden, positions, self.rotary_table, mask)
        return {"last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.rope_parameters["rope_type"] != "default":
        raise ValueError("The declared Pixtral checkpoint uses SiLU and default two-dimensional RoPE")
    return PixtralVisionModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
