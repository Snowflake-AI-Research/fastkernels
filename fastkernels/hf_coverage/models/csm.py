"""CSM constructor text-to-speech generation, including depth decoding and Mimi.

The selected workload uses one unpadded text prompt and constructor greedy
sampling defaults. Temporal K/V persists across frames; depth K/V starts afresh
for each frame. Both unchanged native RoPE and SiLU-and-multiply callables are
used to preserve the reference's low-precision operation boundaries.
"""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.hf_coverage.models import mimi


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.width = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        for name, heads in (('q_proj', self.heads), ('k_proj', self.kv_heads), ('v_proj', self.kv_heads)):
            setattr(self, name, Linear(config.hidden_size, heads * self.width, bias=False))
        self.o_proj = Linear(self.heads * self.width, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.width, config.max_position_embeddings, config.rope_parameters['rope_theta'])
        self.matmul, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, previous, positions):
        batch, length = hidden.shape[:2]
        query, key = self.q_proj(hidden), self.k_proj(hidden)
        query, key = self.rotary.forward_native(
            positions.expand(batch, -1).reshape(-1), query.reshape(batch * length, -1),
            key.reshape(batch * length, -1), self.width, self.rotary.cos_sin_cache.to(hidden.dtype))
        query = query.reshape(batch, length, self.heads, self.width).transpose(1, 2)
        key = key.reshape(batch, length, self.kv_heads, self.width).transpose(1, 2)
        value = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.width).transpose(1, 2)
        if previous is not None:
            key, value = (torch.cat((old, new), dim=2) for old, new in zip(previous, (key, value)))
        cache = key, value
        if self.heads != self.kv_heads:
            key, value = (item.repeat_interleave(self.heads // self.kv_heads, dim=1) for item in (key, value))
        source_length = key.shape[2]
        query = query.reshape(batch * self.heads, length, self.width)
        key, value = (item.reshape(batch * self.heads, source_length, self.width) for item in (key, value))
        allowed = torch.arange(source_length, device=hidden.device)[None, :] <= positions[:, None]
        mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
        # Materialize QK in the model dtype, then scale/add the mask before
        # FP32 softmax, as in pinned eager attention. DenseAttention offers
        # fused SDPA/flash/flex paths without these rounding boundaries.
        scores = self.matmul(query, key.transpose(1, 2)) * self.width**-0.5
        weights = self.softmax((scores + mask).float()).to(hidden.dtype)
        context = self.matmul(weights, value).reshape(batch, self.heads, length, self.width)
        return self.o_proj(context.transpose(1, 2).reshape(batch, length, -1)), cache


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, previous, positions):
        update, cache = self.self_attn(self.input_layernorm(hidden), previous, positions)
        hidden = hidden + update
        normalized = self.post_attention_layernorm(hidden)
        gate = SiluAndMul.forward_native(torch.cat((self.mlp.gate_proj(normalized), self.mlp.up_proj(normalized)), -1))
        return hidden + self.mlp.down_proj(gate), cache


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, previous=None):
        start = 0 if previous is None else previous[0][0].shape[2]
        positions = torch.arange(start, start + hidden.shape[1], device=hidden.device)
        cache = []
        for index, layer in enumerate(self.layers):
            hidden, updated = layer(hidden, None if previous is None else previous[index], positions)
            cache.append(updated)
        return self.norm(hidden), tuple(cache)


class Csm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_text_tokens = Embedding(config.text_vocab_size, config.hidden_size)
        self.embed_audio_tokens = Embedding(config.num_codebooks * config.vocab_size, config.hidden_size)
        self.depth_embed_tokens = Embedding(config.num_codebooks * config.vocab_size, config.hidden_size)
        self.backbone_model = Decoder(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.depth_decoder = Decoder(config.depth_decoder_config)
        self.depth_projector = Linear(config.hidden_size, config.depth_decoder_config.hidden_size, bias=False)
        self.codebooks_head = nn.ModuleList(Linear(config.depth_decoder_config.hidden_size, config.vocab_size, bias=False)
                                           for _ in range(config.num_codebooks - 1))
        self.codec_model = mimi.build_from_config(config.codec_config, 'cpu', torch.float32)
        self.codebook_sum, self.greedy = SegmentCSR(), CodecTop1()

    def audio_embedding(self, codes):
        offsets = torch.arange(self.config.num_codebooks, device=codes.device) * self.config.vocab_size
        hidden = self.embed_audio_tokens(codes + offsets)
        rows = hidden.reshape(-1, hidden.shape[-1])
        boundaries = torch.arange(0, rows.shape[0] + 1, self.config.num_codebooks, device=codes.device)
        return self.codebook_sum(rows.float(), boundaries).to(hidden.dtype).reshape(*codes.shape[:-1], hidden.shape[-1])

    def decode_audio(self, codes):
        codec = self.codec_model
        codes = codes.transpose(1, 2)
        quantized = codec.semantic.decode(codes[:, :codec.semantic_count]) + codec.acoustic.decode(codes[:, codec.semantic_count:])
        hidden = codec.decoder_transformer(codec.upsample(quantized).transpose(1, 2)).transpose(1, 2)
        return codec.decoder(hidden)

    def forward(self, input_ids, *, max_new_tokens=3):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('CSM speech workload requires one unpadded text prompt')
        if max_new_tokens < 1:
            raise ValueError('CSM generation requires at least one audio frame')
        hidden = self.embed_text_tokens(input_ids)
        temporal_cache, frames, temporal_logits, depth_logits = None, [], [], []
        for _ in range(max_new_tokens):
            hidden, temporal_cache = self.backbone_model(hidden, temporal_cache)
            logits = self.lm_head(hidden[:, -1:])
            temporal_logits.append(logits.float())
            tokens = [self.greedy(logits[:, 0].float())]
            embedded = torch.cat((hidden[:, -1:], self.depth_embed_tokens(tokens[0][:, None])), dim=1)
            depth_cache, frame_logits = None, []
            for codebook, head in enumerate(self.codebooks_head):
                depth_hidden, depth_cache = self.depth_decoder(self.depth_projector(embedded), depth_cache)
                logits = head(depth_hidden[:, -1:])
                frame_logits.append(logits)
                token = self.greedy(logits[:, 0].float())
                tokens.append(token)
                if codebook + 1 < len(self.codebooks_head):
                    embedded = self.depth_embed_tokens((token + (codebook + 1) * self.config.vocab_size)[:, None])
            frame = torch.stack(tokens, dim=-1)
            frames.append(frame)
            depth_logits.append(torch.cat(frame_logits, dim=1))
            # Pinned HF stopping checks the first 31 codebooks; waveform crop
            # below checks all 32. Token-ID control and host transfers are timed.
            if self.frame_is_eos(frame[0, :-1]):
                break
            if len(frames) < max_new_tokens:
                hidden = self.audio_embedding(frame[:, None])
        codes = torch.stack(frames, dim=1)
        cutoff = codes.shape[1]
        for index in range(cutoff):
            if self.frame_is_eos(codes[0, index]):
                cutoff = index
                break
        result = {'sequences': codes, 'logits': torch.cat(temporal_logits, dim=1),
                  'depth_logits': torch.stack(depth_logits, dim=1),
                  'audio_values': self.decode_audio(codes[:, :cutoff])}
        for index, (key, value) in enumerate(temporal_cache):
            result[f'past_key_values.{index}.key'] = key
            result[f'past_key_values.{index}.value'] = value
        return result

    def frame_is_eos(self, codes):
        # As in Engine/CosyVoice generation, inspect selected integer IDs for
        # host loop control. The device-to-host transfer stays inside execution.
        return all(token == self.config.codebook_eos_token_id for token in codes.tolist())


def build_from_config(config, device, dtype):
    for decoder in (config, config.depth_decoder_config):
        if (decoder.hidden_act != 'silu' or decoder.attention_bias or decoder.mlp_bias
                or decoder.rope_parameters['rope_type'] != 'default' or not decoder.use_cache):
            raise ValueError('CSM requires constructor SiLU, bias-free projections, default RoPE and caching')
    if (not config.tie_codebooks_embeddings or config.num_codebooks != config.depth_decoder_config.num_codebooks
            or config.vocab_size != config.depth_decoder_config.vocab_size
            or config.hidden_size != config.depth_decoder_config.backbone_hidden_size
            or config.codec_config.num_quantizers != config.num_codebooks or config.eos_token_id is not None):
        raise ValueError('CSM requires constructor tied codebooks and aligned dependencies')
    return Csm(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    if not torch.equal(state['depth_decoder.model.embed_tokens.weight'],
                       state['backbone_model.embed_tokens.embed_audio_tokens.weight']):
        raise ValueError('CSM tied codebooks require equal backbone/depth embedding values')
    codec_state = {name.removeprefix('codec_model.'): value for name, value in state.items() if name.startswith('codec_model.')}
    mimi.load_state_dict_into(model.codec_model, codec_state, config.codec_config)
    consumed = {'codec_model.' + name for name in codec_state}
    mapped = {}
    for name, target in model.state_dict().items():
        if name.startswith('codec_model.'):
            mapped[name] = target
            continue
        source = name.replace('.emb.weight', '.weight')
        if source == 'embed_audio_tokens.weight':
            source = 'backbone_model.embed_tokens.embed_audio_tokens.weight'
        elif source == 'depth_embed_tokens.weight':
            source = 'depth_decoder.model.embed_tokens.weight'
        elif source.startswith('depth_decoder.'):
            source = source.replace('depth_decoder.', 'depth_decoder.model.', 1)
        elif source.startswith('depth_projector.'):
            source = source.replace('depth_projector.', 'depth_decoder.model.inputs_embeds_projector.', 1)
        if source.startswith('codebooks_head.'):
            index = int(source.split('.')[1])
            source = 'depth_decoder.codebooks_head.weight'
            value = state[source][index].T
        else:
            value = state[source]
        if value.shape != target.shape:
            raise ValueError(f'CSM weight mismatch: {name} <- {source}')
        mapped[name] = value
        consumed.add(source)
    if consumed != set(state):
        raise ValueError(f'CSM unmapped state: {sorted(set(state) - consumed)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    generation = (case or {}).get('generation_kwargs', {'max_new_tokens': 3, 'output_audio': True})
    if set(generation) - {'max_new_tokens', 'output_audio'} or not generation.get('output_audio', True):
        raise ValueError('CSM workload uses constructor greedy generation with waveform output')
    frames = generation.get('max_new_tokens', 3)
    return {'generate': Workload(run=lambda: model(**inputs, max_new_tokens=frames))}
