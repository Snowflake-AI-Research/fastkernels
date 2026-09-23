"""Speech2Text convolutional GLU frontend and pre-norm encoder-decoder."""

import math
import re
from copy import copy

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.conv1d import Conv1d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.qwen3_next_attention import _gate_mul_inplace
from .mbart import PreNormStack
from .m2m_100 import sinusoidal_table
from .marian import load_state_dict_into as load_seq2seq
from .mvp import EagerAttention


class EagerEncoderAttention(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.qkv, self.proj = parent.qkv, parent.proj
        self.heads, self.width = parent.num_heads, parent.head_dim
        self.attention = EagerAttention(prescale_query=False)

    def forward(self, hidden, attn_mask=None):
        batch, length, channels = hidden.shape
        query, key, value = self.qkv(hidden).view(batch, length, 3, self.heads, self.width).unbind(2)
        return self.proj(self.attention(query, key, value, attn_mask=attn_mask).reshape(batch, length, channels))


class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            Conv1d(config.input_feat_per_channel * config.input_channels if index == 0 else config.conv_channels // 2,
                   config.conv_channels if index < config.num_conv_layers - 1 else 2 * config.d_model,
                   kernel, stride=2, padding=kernel // 2)
            for index, kernel in enumerate(config.conv_kernel_sizes)])
        self.sigmoid, self.product = Sigmoid(), ProductGate()
        self.scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.pad = config.pad_token_id
        adapted = copy(config)
        adapted.max_position_embeddings = config.max_source_positions
        carrier = PreNormStack(adapted, None, decoder=False, learned_positions=False)
        self.layers, self.layer_norm = carrier.layers, carrier.layer_norm
        self.embed_positions = Embedding(config.max_source_positions + 2, config.d_model)
        self.embed_positions.emb.weight.requires_grad_(False)
        self.embed_positions.emb.weight.data.copy_(sinusoidal_table(config.max_source_positions + 2, config.d_model, self.pad))
        for layer in self.layers:
            layer.attn = EagerEncoderAttention(layer.attn)

    def forward(self, features):
        hidden = features.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            value, gate = conv(hidden).chunk(2, dim=1)
            if value.is_cuda:
                # Existing internal kernel performs FP32 sigmoid/product and a
                # single output cast. It is not separately exposed as an L1 task.
                hidden = _gate_mul_inplace(value.contiguous(), gate.contiguous())
            else:
                # CPU construction diagnostic; the actual GPU uses the unchanged kernel.
                packed = torch.cat((value.transpose(1, 2).float(), self.sigmoid(gate.transpose(1, 2).float())), dim=-1)
                hidden = self.product(packed).to(value.dtype).transpose(1, 2).contiguous()
        hidden = hidden.transpose(1, 2).contiguous() * self.scale
        positions = torch.arange(hidden.shape[1], device=hidden.device) + self.pad + 1
        hidden = hidden + self.embed_positions(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)


class Speech2Text(nn.Module):
    generated_positions = True

    def __init__(self, config):
        super().__init__()
        self.encoder = SpeechEncoder(config)
        shared = Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        adapted = copy(config)
        adapted.max_position_embeddings = config.max_target_positions
        self.decoder = PreNormStack(adapted, shared, decoder=True, learned_positions=False)
        self.decoder.embed_positions = Embedding(config.max_target_positions + 2, config.d_model)
        self.decoder.embed_positions.emb.weight.requires_grad_(False)
        self.decoder.embed_positions.emb.weight.data.copy_(sinusoidal_table(config.max_target_positions + 2, config.d_model, config.pad_token_id))
        for layer in self.decoder.layers:
            layer.attention.self.attn = EagerAttention(prescale_query=False)
            layer.cross_attention.attention = EagerAttention(prescale_query=False)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = shared.emb.weight
        self.pad = config.pad_token_id

    def forward(self, features, decoder_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("Speech2Text coverage evaluates unpadded feature and token sequences")
        memory = self.encoder(features) if encoder_hidden_states is None else encoder_hidden_states
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        valid = decoder_ids.ne(self.pad).int()
        positions = ((valid.cumsum(dim=1).to(valid.dtype) + past_length) * valid).long() + self.pad
        hidden, cache = self.decoder(decoder_ids, positions, memory, past_key_values)
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": cache}


def build_from_config(config, device, dtype):
    if config.activation_function != "relu" or not config.tie_word_embeddings or not config.use_cache:
        raise ValueError("Speech2Text selected checkpoint requires cached tied ReLU encoder-decoder")
    return Speech2Text(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    # Only frontend paths differ from the common pre-norm seq2seq mapping.
    translated = {name.replace("model.encoder.conv.conv_layers.", "model.encoder.conv_layers."): value
                  for name, value in state_dict.items()}
    translated = {re.sub(r"\.conv_layers\.(\d+)\.", r".conv_layers.\1.conv.", name): value
                  for name, value in translated.items()}
    load_seq2seq(model, translated, config)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, encoder_input_name="input_features")

    def run():
        output = model(inputs["input_features"], inputs["decoder_input_ids"])
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
