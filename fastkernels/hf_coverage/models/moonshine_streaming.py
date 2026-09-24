"""Moonshine Streaming frame frontend, windowed encoder and cached decoder.

The selected public transcription path compares logits and self/cross caches.
Encoder memory stays unmodified: native decoder's repeated in-place position
addition corrupts its returned encoder state but not cached cross attention.
"""
import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv1d import Conv1d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU
from ..patches.codec_top1 import CodecTop1
from ..patches.forecast_revin import ZeroSafeVarianceNormalize
from ..patches.moonshine_streaming_normalize import RoundedVarianceNormalize
from ..patches.product_gate import ProductGate
from ..patches.ratio_log import RatioLog
from ..runner import Workload, seq2seq_cache_outputs


class Asinh(nn.Module):
    """Existing pointwise composition on bounded normalized audio frames."""
    def __init__(self):
        super().__init__()
        self.relu, self.product = ReLU(), ProductGate()
        self.root, self.log, self.sign = ZeroSafeVarianceNormalize(), RatioLog(), CodecTop1()
        self.log_k = nn.Parameter(torch.empty(()))
        self.scale = None

    def forward(self, x):
        dtype = x.dtype
        x = (x * self.scale).float()
        absolute = self.relu(x) + self.relu(-x)
        variance = 1. + self.product(torch.stack((absolute, absolute), -1)).squeeze(-1)
        root = self.root(variance, torch.zeros_like(variance), variance)
        magnitude = self.log(absolute + root, torch.ones_like(root))
        sign = 1. - 2. * self.sign(torch.stack((x, torch.zeros_like(x)), -1)).to(x.dtype)
        return self.product(torch.stack((sign, magnitude), -1)).squeeze(-1).to(dtype)


class FrameEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frame_len = round(config.sample_rate * config.frame_ms / 1000.)
        self.mean, self.product = GlobalAvgPool2d(), ProductGate()
        self.normalize, self.comp = RoundedVarianceNormalize(), Asinh()
        self.linear = Linear(self.frame_len, config.hidden_size, bias=False)
        self.conv1 = Conv1d(config.hidden_size, 2 * config.hidden_size, 5, stride=2)
        self.conv2 = Conv1d(2 * config.hidden_size, config.hidden_size, 5, stride=2)
        self.activation = SiLU()

    def frame_mask(self, mask):
        count = mask.sum(-1) // self.frame_len
        return torch.arange(mask.shape[1] // self.frame_len, device=mask.device)[None] < count[:, None]

    @staticmethod
    def conv_mask(mask):
        # Supplied padding metadata: same five-tap left-padded stride-two support.
        padded = torch.cat((mask.new_zeros(mask.shape[0], 4), mask), -1)
        return padded.unfold(-1, 5, 2).any(-1)

    def forward(self, waveform, mask):
        frames = waveform.reshape(waveform.shape[0], -1, self.frame_len)
        centered = frames - self.mean(frames.unsqueeze(-2))[..., None]
        squared = self.product(torch.stack((centered, centered), -1)).squeeze(-1)
        variance = self.mean(squared.unsqueeze(-2))[..., None] + 1e-6
        hidden = self.activation(self.linear(self.comp(self.normalize(centered, variance))))
        mask = self.frame_mask(mask)
        hidden = hidden.masked_fill(~mask[..., None], 0).transpose(1, 2)
        for index, convolution in enumerate((self.conv1, self.conv2)):
            hidden = convolution(torch.cat((hidden.new_zeros(*hidden.shape[:-1], 4), hidden), -1))
            mask = self.conv_mask(mask)
            hidden = hidden.masked_fill(~mask[:, None], 0)
            if index == 0:
                hidden = self.activation(hidden)
        return hidden.transpose(1, 2), mask


class OffsetNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(width))
        self.ln = LayerNorm(width, elementwise_affine=False, promote_fp32=False)
        self.product = ProductGate()

    def forward(self, x):
        normalized = self.ln(x)
        scale = (self.gamma + 1.).expand_as(normalized)
        return self.product(torch.cat((normalized, scale), -1))


class Attention(nn.Module):
    def __init__(self, config, *, rotary=None, causal=False):
        super().__init__()
        self.heads, self.dim = config.num_attention_heads, config.head_dim
        self.rotary, self.causal = rotary, causal
        self.q_proj = Linear(config.hidden_size, self.heads * self.dim, bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, self.heads * self.dim, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, self.heads * self.dim, bias=config.attention_bias)
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, bias=config.attention_bias)
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, mask=None, *, memory=None, past=None):
        batch, length = hidden.shape[:2]
        query = self.q_proj(hidden).reshape(batch, length, self.heads, self.dim)
        if memory is not None and past is not None:
            key, value = (v.transpose(1, 2) for v in past)
        else:
            source = hidden if memory is None else memory
            key, value = (proj(source).reshape(batch, -1, self.heads, self.dim)
                          for proj in (self.k_proj, self.v_proj))
        if self.rotary is not None:
            width = self.rotary.head_dim
            offset = 0 if past is None else past[0].shape[2]
            positions = (torch.arange(length, device=hidden.device) + offset).repeat(batch)
            qrot, krot = self.rotary.forward_native_interleaved(
                positions, query[..., :width].contiguous().reshape(-1, self.heads, width),
                key[..., :width].contiguous().reshape(-1, self.heads, width), width,
                self.rotary.cos_sin_cache.to(query.dtype))
            query = torch.cat((qrot.reshape(batch, length, self.heads, width), query[..., width:]), -1)
            key = torch.cat((krot.reshape(batch, length, self.heads, width), key[..., width:]), -1)
        if past is not None and memory is None:
            key, value = (torch.cat((old.transpose(1, 2), new), 1) for old, new in zip(past, (key, value)))
        cache = tuple(v.transpose(1, 2).contiguous() for v in (key, value))
        context = self.attention(query, key, value, softmax_scale=self.dim**-.5,
                                 causal=self.causal and past is None, attn_mask=mask)
        return self.o_proj(context.reshape(batch, length, -1)), cache


class MLP(nn.Module):
    def __init__(self, config, gated=False):
        super().__init__()
        self.gated = gated
        self.fc1 = Linear(config.hidden_size, config.intermediate_size * (2 if gated else 1))
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.activation = SiLU() if gated else GELU()
        self.product = ProductGate()

    def forward(self, hidden):
        hidden = self.fc1(hidden)
        if self.gated:
            hidden, gate = hidden.chunk(2, -1)
            hidden = self.product(torch.cat((self.activation(gate), hidden), -1))
        else:
            hidden = self.activation(hidden)
        return self.fc2(hidden)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.mlp = Attention(config), MLP(config)
        self.input_layernorm, self.post_attention_layernorm = OffsetNorm(config.hidden_size), OffsetNorm(config.hidden_size)

    def forward(self, hidden, mask):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), mask)[0]
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = FrameEmbedder(config)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_norm = OffsetNorm(config.hidden_size)
        self.windows = config.sliding_windows

    def forward(self, waveform, mask):
        hidden, mask = self.embedder(waveform, mask)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        distance = positions[:, None] - positions[None, :]
        for layer, (left, right) in zip(self.layers, self.windows):
            allowed = ((distance >= 0) & (distance < left)) | ((distance < 0) & (-distance < right))
            attention_mask = torch.zeros(mask.shape[0], 1, *allowed.shape, device=hidden.device, dtype=hidden.dtype)
            attention_mask.masked_fill_(~(allowed[None, None] & mask[:, None, None, :]), torch.finfo(hidden.dtype).min)
            hidden = layer(hidden, attention_mask)
        return self.final_norm(hidden), mask


class DecoderLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.self_attn = Attention(config, rotary=rotary, causal=True)
        self.encoder_attn = Attention(config)
        self.mlp = MLP(config, gated=True)
        for name in ('input_layernorm', 'post_attention_layernorm', 'final_layernorm'):
            setattr(self, name, LayerNorm(config.hidden_size, create_offset=False, promote_fp32=False))

    def forward(self, hidden, memory, mask, past):
        previous = (None, None) if past is None else past
        residual, self_cache = self.self_attn(self.input_layernorm(hidden), past=previous[0])
        hidden = hidden + residual
        residual, cross_cache = self.encoder_attn(self.post_attention_layernorm(hidden), mask,
                                                 memory=memory, past=previous[1])
        hidden = hidden + residual
        return hidden + self.mlp(self.final_layernorm(hidden)), (self_cache, cross_cache)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.pos_emb = Embedding(config.max_position_embeddings, config.encoder_config.hidden_size)
        rope = config.rope_parameters
        rotary = RotaryEmbedding(int(config.head_dim * rope['partial_rotary_factor']),
                                 config.max_position_embeddings, rope['rope_theta'], is_neox_style=False)
        self.layers = nn.ModuleList(DecoderLayer(config, rotary) for _ in range(config.num_hidden_layers))
        self.norm = LayerNorm(config.hidden_size, create_offset=False, promote_fp32=False)
        self.proj = (Linear(config.encoder_config.hidden_size, config.hidden_size, bias=False)
                     if config.encoder_config.hidden_size != config.hidden_size else None)

    def forward(self, ids, memory, mask, past):
        memory = memory + self.pos_emb(torch.arange(memory.shape[1], device=memory.device))
        if self.proj is not None:
            memory = self.proj(memory)
        attention_mask = torch.zeros(mask.shape[0], 1, ids.shape[1], mask.shape[1], device=memory.device, dtype=memory.dtype)
        attention_mask.masked_fill_(~mask[:, None, None, :], torch.finfo(memory.dtype).min)
        hidden, cache = self.embed_tokens(ids), []
        for index, layer in enumerate(self.layers):
            hidden, current = layer(hidden, memory, attention_mask, None if past is None else past[index])
            cache.append(current)
        return self.norm(hidden), tuple(cache)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.encoder, self.model.decoder = Encoder(config.encoder_config), Decoder(config)
        self.proj_out = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.select = CodecTop1()
        if config.tie_word_embeddings:
            self.proj_out.weight = self.model.decoder.embed_tokens.emb.weight

    def forward(self, input_values, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is None or decoder_attention_mask is not None:
            raise ValueError('Selected streaming case requires audio padding metadata and unpadded decoder tokens')
        if encoder_hidden_states is None:
            memory, mask = self.model.encoder(input_values, attention_mask)
        else:
            memory = encoder_hidden_states
            embedder = self.model.encoder.embedder
            mask = embedder.conv_mask(embedder.conv_mask(embedder.frame_mask(attention_mask)))
        hidden, cache = self.model.decoder(decoder_input_ids, memory, mask, past_key_values)
        return {'logits': self.proj_out(hidden), 'encoder_last_hidden_state': memory, 'past_key_values': cache}


def build_from_config(config, device, dtype):
    encoder = config.encoder_config
    if (config.attention_bias or encoder.attention_bias or config.hidden_act != 'silu'
            or encoder.hidden_act != 'gelu' or config.num_attention_heads != config.num_key_value_heads
            or encoder.num_attention_heads != encoder.num_key_value_heads or config.pad_head_dim_to_multiple_of is not None
            or config.rope_parameters['rope_type'] != 'default' or not config.use_cache
            or len(encoder.sliding_windows) != encoder.num_hidden_layers):
        raise ValueError('Unsupported configuration outside the selected Moonshine Streaming task')
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped, consumed = {}, set()
    for name, parameter in model.state_dict().items():
        source = name.replace('.emb.weight', '.weight').replace('.conv.', '.')
        value = state_dict[source]
        if value.shape != parameter.shape:
            raise ValueError(f'Moonshine Streaming state shape mismatch: {source}')
        mapped[name] = value
        consumed.add(source)
    if consumed != set(state_dict):
        raise ValueError(f'Unmapped Moonshine Streaming state: {sorted(set(state_dict) - consumed)}')
    if config.tie_word_embeddings and not torch.equal(state_dict['proj_out.weight'], state_dict['model.decoder.embed_tokens.weight']):
        raise ValueError('Tied output and input embeddings disagree')
    model.load_state_dict(mapped, strict=True)
    # Frozen learned scalar; preserve source-dtype exp rounding before activation multiplication.
    embedder = model.model.encoder.embedder
    scale = float(embedder.comp.log_k.exp())
    if not math.isfinite(scale) or not 0 < scale * (embedder.frame_len - 1)**.5 <= 8:
        raise ValueError('Asinh composition requires the reviewed bounded normalized-frame domain')
    embedder.comp.scale = scale


def make_workloads(model, inputs, config, *, case=None):
    max_new_tokens = case['generation_kwargs']['max_new_tokens']

    def generate():
        if inputs['input_values'].shape[0] != 1:
            raise ValueError('Selected greedy transcription case has batch size one')
        ids = torch.full((1, 1), config.decoder_start_token_id, dtype=torch.long,
                         device=inputs['input_values'].device)
        sequence, output, logits = [ids], None, {}
        for step in range(max_new_tokens):
            output = model(inputs['input_values'], ids, attention_mask=inputs['attention_mask'],
                           encoder_hidden_states=None if output is None else output['encoder_last_hidden_state'],
                           past_key_values=None if output is None else output['past_key_values'])
            # Generation's raw next-token logits are float32 in the native API.
            current = output['logits'][:, -1].float()
            logits[f'logits.{step}'] = current
            ids = model.select(current).reshape(1, 1)
            sequence.append(ids)
            # Explicit host scheduling on selected integer IDs, after the selector.
            if ids.item() == config.eos_token_id:
                break
        return dict(sequences=torch.cat(sequence, -1), **logits,
                    **seq2seq_cache_outputs(output['past_key_values']))

    return {'generate': Workload(run=generate)}
