"""Pegasus-X global/local block attention composed without dense token attention."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from .marian import load_state_dict_into
from .mbart import PreNormConditionalGeneration
from .mvp import EagerAttention
from .plbart import make_workloads


class SinusoidalPositions(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.register_buffer("dtype_anchor", torch.empty(0), persistent=False)

    def forward(self, positions):
        # Only position metadata is involved; preserve HF's dtype rounding.
        half = self.width // 2
        frequencies = torch.exp(torch.arange(half, device=positions.device).to(self.dtype_anchor.dtype)
                                * -(math.log(10000.0) / (half - 1)))
        angles = positions[:, None] * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class GlobalLocalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.encoder_attention_heads
        self.dim = config.d_model // self.heads
        self.block = config.block_size
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(config.d_model, config.d_model, bias=False))
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, local, global_tokens, mask):
        batch, length, width = local.shape
        blocks, globals_ = length // self.block, global_tokens.shape[1]

        def project(tensor, module, scale=1.0):
            output = module(tensor)
            if scale != 1.0:
                output = output * scale
            return output.view(batch, -1, self.heads, self.dim).transpose(1, 2)

        lq = project(local, self.q_proj, self.dim ** -0.5)
        lk, lv = project(local, self.k_proj), project(local, self.v_proj)
        gq = project(global_tokens, self.q_proj, self.dim ** -0.5)
        gk, gv = project(global_tokens, self.k_proj), project(global_tokens, self.v_proj)
        global_mask = torch.cat((mask.new_zeros(batch, globals_), mask), dim=-1)
        scores = self.bmm(gq, torch.cat((gk, lk), dim=2).transpose(-1, -2))
        global_out = self.bmm(self.softmax(scores + global_mask[:, None, None]), torch.cat((gv, lv), dim=2))
        lq, lk, lv = (tensor.reshape(batch, self.heads, blocks, self.block, self.dim) for tensor in (lq, lk, lv))
        local_scores = self.bmm(lq, lk.transpose(-1, -2))
        global_scores = self.bmm(lq, gk[:, :, None].transpose(-1, -2))
        local_mask = torch.cat((mask.new_zeros(batch, blocks, globals_), mask.view(batch, blocks, self.block)), dim=-1)
        probabilities = self.softmax(torch.cat((global_scores, local_scores), dim=-1) + local_mask[:, None, :, None])
        # HF rounds the two products separately before their branch addition.
        local_out = (self.bmm(probabilities[..., :globals_], gv[:, :, None])
                     + self.bmm(probabilities[..., globals_:], lv))
        local_out = local_out.permute(0, 2, 3, 1, 4).reshape(batch, length, width)
        global_out = global_out.transpose(1, 2).reshape(batch, globals_, width)
        return self.out_proj(local_out), self.out_proj(global_out)


class EncoderLayer(nn.Module):
    def __init__(self, config, stagger):
        super().__init__()
        self.self_attn = GlobalLocalAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.d_model, promote_fp32=False)
        self.global_self_attn_layer_norm = LayerNorm(config.d_model, promote_fp32=False)
        self.final_layer_norm = LayerNorm(config.d_model, promote_fp32=False)
        self.fc1, self.fc2 = Linear(config.d_model, config.encoder_ffn_dim), Linear(config.encoder_ffn_dim, config.d_model)
        self.activation = ReLU()
        self.shift = config.block_size // 2 if stagger else 0

    def forward(self, hidden, global_tokens, mask):
        local = self.self_attn_layer_norm(hidden)
        global_normed = self.global_self_attn_layer_norm(global_tokens)
        if self.shift:
            padding = local.new_zeros(local.shape[0], self.shift, local.shape[-1])
            local = torch.cat((padding, local, padding), dim=1)
            masked = mask.new_full((mask.shape[0], self.shift), torch.finfo(mask.dtype).min)
            mask = torch.cat((masked, mask, masked), dim=1)
        local, global_update = self.self_attn(local, global_normed, mask)
        if self.shift:
            local = local[:, self.shift:-self.shift]
        hidden, global_tokens = hidden + local, global_tokens + global_update
        hidden = hidden + self.fc2(self.activation(self.fc1(self.final_layer_norm(hidden))))
        global_tokens = global_tokens + self.fc2(self.activation(self.fc1(self.final_layer_norm(global_tokens))))
        return hidden, global_tokens


class Encoder(nn.Module):
    def __init__(self, config, shared):
        super().__init__()
        self.embed_tokens = shared
        self.embed_positions = SinusoidalPositions(config.d_model)
        self.embed_global = Embedding(config.num_global_tokens, config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config, config.stagger_local_blocks and index % 2 == 1)
                                     for index in range(config.encoder_layers)])
        self.layer_norm = LayerNorm(config.d_model, promote_fp32=False)
        self.scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.block, self.globals = config.block_size, config.num_global_tokens
        self.position_offset = 0

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids) * self.scale + self.embed_positions(positions)
        batch, length, width = hidden.shape
        padding = -length % self.block
        mask = hidden.new_zeros(batch, length)
        if padding:
            hidden = torch.cat((hidden, hidden.new_zeros(batch, padding, width)), dim=1)
            mask = torch.cat((mask, mask.new_full((batch, padding), torch.finfo(mask.dtype).min)), dim=1)
        global_tokens = self.embed_global(torch.arange(self.globals, device=ids.device)[None].expand(batch, -1))
        for layer in self.layers:
            hidden, global_tokens = layer(hidden, global_tokens, mask)
        return self.layer_norm(hidden[:, :length])


class PegasusXForConditionalGeneration(PreNormConditionalGeneration):
    def __init__(self, config):
        super().__init__(config, learned_positions=False)
        del self.final_logits_bias
        self.encoder = Encoder(config, self.shared)
        self.decoder.embed_positions = SinusoidalPositions(config.d_model)
        for layer in self.decoder.layers:
            layer.attention.self.qkv.bias = None
            layer.attention.output.dense.bias = None
            layer.attention.self.attn = EagerAttention(prescale_query=False)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                getattr(layer.cross_attention, name).bias = None
            layer.cross_attention.attention = EagerAttention(prescale_query=False)

    def forward(self, input_ids, decoder_input_ids, encoder_positions=None, decoder_positions=None, *,
                encoder_hidden_states=None, past_key_values=None, attention_mask=None,
                decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("Pegasus-X coverage currently evaluates unpadded token sequences")
        memory = encoder_hidden_states
        if memory is None:
            if encoder_positions is None:
                encoder_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            memory = self.encoder(input_ids, encoder_positions)
        if decoder_positions is None:
            past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
            decoder_positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device) + past_length
        hidden, cache = self.decoder(decoder_input_ids, decoder_positions, memory, past_key_values)
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory, "past_key_values": cache}


def build_from_config(config, device, dtype):
    if config.activation_function != "relu" or not config.use_cache or not config.tie_word_embeddings:
        raise ValueError("Selected Pegasus-X checkpoint requires ReLU, tied output and caching")
    return PegasusXForConditionalGeneration(config).to(device=device, dtype=dtype).eval()
