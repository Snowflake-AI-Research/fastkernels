"""SpeechT5 ASR: waveform frontend, relative encoder and cached text decoder."""

import math

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from .m2m_100 import sinusoidal_table
from .unispeech import GroupFeatureConv
from .wav2vec2 import FeatureProjection, PositionConv


class SpeechAttention(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        self.heads = config.decoder_attention_heads if decoder else config.encoder_attention_heads
        self.width = config.hidden_size // self.heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, memory=None, relative=None, mask=None, cache=None, causal=False):
        batch, length, channels = hidden.shape
        query = (self.q_proj(hidden) * self.width**-0.5).reshape(batch, length, self.heads, self.width)
        query = query.transpose(1, 2).reshape(batch * self.heads, length, self.width)
        if memory is not None and cache is not None:
            key, value = cache
        else:
            source = hidden if memory is None else memory
            key, value = (projection(source).reshape(batch, -1, self.heads, self.width).transpose(1, 2)
                          for projection in (self.k_proj, self.v_proj))
            if cache is not None:
                key, value = (torch.cat((old, new), dim=2) for old, new in zip(cache, (key, value)))
            else:
                key, value = key.contiguous(), value.contiguous()
        source_length = key.shape[2]
        scores = self.bmm(query, key.reshape(-1, source_length, self.width).transpose(1, 2))
        if relative is not None:
            scores = scores + self.bmm(query.transpose(0, 1), relative.transpose(1, 2)).transpose(0, 1)
        scores = scores.reshape(batch, self.heads, length, source_length)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if causal:
            position = torch.arange(length, device=hidden.device) + source_length - length
            blocked = torch.arange(source_length, device=hidden.device)[None, :] > position[:, None]
            scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        output = self.bmm(self.softmax(scores).reshape(-1, length, source_length),
                          value.reshape(-1, source_length, self.width))
        output = output.reshape(batch, self.heads, length, self.width).transpose(1, 2).reshape(batch, length, channels)
        return self.out_proj(output), (key, value)


class SpeechLayer(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.attention = SpeechAttention(config, decoder)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.feed_forward = nn.Sequential(
            Linear(config.hidden_size, config.decoder_ffn_dim if decoder else config.encoder_ffn_dim),
            GELU(), Linear(config.decoder_ffn_dim if decoder else config.encoder_ffn_dim, config.hidden_size))
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        if decoder:
            self.encoder_attn = SpeechAttention(config, decoder=True)
            self.encoder_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden, memory=None, relative=None, mask=None, cache=None):
        attention, self_cache = self.attention(hidden, relative=relative,
                                               mask=None if self.decoder else mask,
                                               cache=None if cache is None else cache[0], causal=self.decoder)
        hidden = self.layer_norm(hidden + attention)
        cross_cache = None
        if self.decoder:
            attention, cross_cache = self.encoder_attn(hidden, memory=memory, mask=mask,
                                                       cache=None if cache is None else cache[1])
            hidden = self.encoder_attn_layer_norm(hidden + attention)
        hidden = self.final_layer_norm(hidden + self.feed_forward(hidden))
        return hidden, (self_cache, cross_cache)


class SpeechT5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_encoder = nn.ModuleList(GroupFeatureConv(config, i) for i in range(len(config.conv_dim)))
        self.feature_projection, self.pos_conv_embed = FeatureProjection(config), PositionConv(config)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.relative = Embedding(2 * config.encoder_max_relative_position,
                                  config.hidden_size // config.encoder_attention_heads)
        self.encoder = nn.ModuleList(SpeechLayer(config) for _ in range(config.encoder_layers))
        self.decoder = nn.ModuleList(SpeechLayer(config, decoder=True) for _ in range(config.decoder_layers))
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.emb.weight
        self.register_buffer("speech_positions", sinusoidal_table(config.max_speech_positions + config.pad_token_id + 3,
                                                                   config.hidden_size, config.pad_token_id), persistent=False)
        self.register_buffer("text_positions", sinusoidal_table(config.max_text_positions + config.pad_token_id + 3,
                                                                 config.hidden_size, config.pad_token_id), persistent=False)
        if config.mask_time_prob > 0 or config.mask_feature_prob > 0:
            self.masked_spec_embed = nn.Parameter(torch.empty(config.hidden_size))

    def feature_mask(self, attention_mask, length):
        if attention_mask is None:
            return None
        lengths = attention_mask.long().cumsum(-1)[:, -1]
        for kernel, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            lengths = (lengths - kernel) // stride + 1
        return torch.arange(length, device=attention_mask.device)[None, :] < lengths[:, None]

    def encode(self, waveform, attention_mask=None):
        hidden = waveform[:, None]
        for layer in self.feature_encoder:
            hidden = layer(hidden)
        hidden, _ = self.feature_projection(hidden.transpose(1, 2))
        hidden = hidden + self.pos_conv_embed(hidden)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        mask = self.feature_mask(attention_mask, hidden.shape[1])
        # The HF speech prenet indexes zeros with padding_idx=1, yielding
        # positions 2,3,...; its constructor uses that same padding index.
        valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long) if mask is None else mask.long()
        ids = valid.cumsum(1) * valid + self.config.pad_token_id
        hidden = self.layer_norm(hidden + self.speech_positions[ids])
        relative_ids = (positions[:, None] - positions[None, :]).clamp(
            -self.config.encoder_max_relative_position, self.config.encoder_max_relative_position - 1)
        relative = self.relative(relative_ids + self.config.encoder_max_relative_position)
        for layer in self.encoder:
            hidden, _ = layer(hidden, relative=relative, mask=mask)
        return hidden, mask

    def decode(self, ids, memory, mask=None, cache=None):
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        valid = ids.ne(self.config.pad_token_id).long()
        positions = (valid.cumsum(1) + offset) * valid + self.config.pad_token_id
        scale = math.sqrt(self.config.hidden_size) if self.config.scale_embedding else 1.0
        hidden = self.embed_tokens(ids) * scale + self.text_positions[positions]
        next_cache = []
        for index, layer in enumerate(self.decoder):
            hidden, state = layer(hidden, memory=memory, mask=mask, cache=None if cache is None else cache[index])
            next_cache.append(state)
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": tuple(next_cache)}

    def forward(self, input_values, decoder_input_ids, attention_mask=None, *,
                encoder_hidden_states=None, past_key_values=None, decoder_attention_mask=None):
        if decoder_attention_mask is not None:
            raise ValueError("SpeechT5 coverage evaluates unpadded decoder tokens")
        if encoder_hidden_states is None:
            memory, mask = self.encode(input_values, attention_mask)
        else:
            memory = encoder_hidden_states
            mask = self.feature_mask(attention_mask, memory.shape[1])
        return self.decode(decoder_input_ids, memory, mask, past_key_values)


def build_from_config(config, device, dtype):
    if (config.feat_extract_norm != "group" or config.feat_extract_activation != "gelu"
            or config.hidden_act != "gelu" or not config.use_cache):
        raise ValueError("Selected SpeechT5 ASR path requires group-norm waveform features, GELU and decoder caches")
    return SpeechT5(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    prenet, encoder, decoder = "speecht5.encoder.prenet.", "speecht5.encoder.wrapped_encoder.", "speecht5.decoder."
    for key in model.state_dict():
        if key == "pos_conv_embed.conv.weight":
            prefix = prenet + "pos_conv_embed.conv.parametrizations.weight."
            device = model.pos_conv_embed.conv.weight.device
            gain, value = (remaining.pop(prefix + name).to(device) for name in ("original0", "original1"))
            mapped[key] = torch._weight_norm(value, gain, 2)
            continue
        if key.startswith("feature_encoder."):
            source = prenet + key.replace("feature_encoder.", "feature_encoder.conv_layers.", 1)
        elif key.startswith(("feature_projection.", "pos_conv_embed.", "masked_spec_embed")):
            source = prenet + key
        elif key.startswith("layer_norm."):
            source = encoder + key
        elif key.startswith("relative."):
            source = encoder + key.replace("relative.emb.", "embed_positions.pe_k.")
        elif key.startswith(("encoder.", "decoder.")):
            is_decoder = key.startswith("decoder.")
            source = (decoder + "wrapped_decoder.layers." if is_decoder else encoder + "layers.") + key.split(".", 1)[1]
            source = source.replace("feed_forward.0.", "feed_forward.intermediate_dense.")
            source = source.replace("feed_forward.2.", "feed_forward.output_dense.")
            if is_decoder:
                source = source.replace(".attention.", ".self_attn.").replace(".layer_norm.", ".self_attn_layer_norm.")
        elif key.startswith("embed_tokens."):
            source = decoder + "prenet." + key.replace(".emb.", ".")
        elif key.startswith("lm_head."):
            source = "text_decoder_postnet." + key
        else:
            raise KeyError(key)
        mapped[key] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped SpeechT5 state: {sorted(remaining)}")
    if config.tie_word_embeddings and not torch.equal(mapped["lm_head.weight"], mapped["embed_tokens.emb.weight"]):
        raise ValueError("SpeechT5 checkpoint requires tied text embeddings")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, encoder_input_name="input_values")

    def run():
        output = model(**inputs)
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
