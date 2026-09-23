"""Kyutai streaming speech-to-text with the native FP32 Mimi encoder and caches."""

from copy import copy

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from .dia import AudioMLP
from .mimi import Attention, CausalConv, Mimi, load_state_dict_into as load_mimi


class StreamingConv(nn.Module):
    """Cache input padding around the unchanged convolution operation."""

    def __init__(self, parent):
        super().__init__()
        self.conv, self.total, self.replicate = parent.conv, parent.total, parent.replicate
        self.previous = None

    def forward(self, hidden):
        previous = self.previous
        if previous is None:
            previous = (hidden[..., :1].expand(*hidden.shape[:-1], self.total) if self.replicate
                        else hidden.new_zeros(*hidden.shape[:-1], self.total))
        padded = torch.cat((previous, hidden), dim=-1)
        self.previous = padded[..., -self.total:].clone() if self.total else padded[..., :0]
        return self.conv(padded)


class CachedAttention(Attention):
    def __init__(self, config, static=False):
        super().__init__(config)
        self.attention = DenseAttention(backend="sdpa")
        self.window = config.sliding_window
        self.text_decoder = static
        self.retained = self.window if static else self.window - 1
        self.key = self.value = None
        self.seen = 0

    def forward(self, hidden, mask=None):
        batch, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device) + self.seen
        query, key = self.q_proj(hidden), self.k_proj(hidden)
        # Compiled text decode fuses angle creation and rotary products before
        # rounding. The eager prefill and codec retain their input-dtype angles.
        angles = self.rotary_emb.cos_sin_cache
        if not self.text_decoder or self.seen == 0:
            angles = angles.to(hidden.dtype)
        query, key = RotaryEmbedding.forward_native(
            positions.expand(batch, -1).reshape(-1), query.reshape(batch * length, -1).float(),
            key.reshape(batch * length, -1).float(), self.head_dim, angles.float())
        query = query.to(hidden.dtype).reshape(batch, length, self.heads, self.head_dim)
        key = key.to(hidden.dtype).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.key is not None:
            key = torch.cat((self.key[:, :, -(self.window - 1):], key), 2)
            value = torch.cat((self.value[:, :, -(self.window - 1):], value), 2)
        source_positions = torch.arange(self.seen + length - key.shape[2], self.seen + length, device=hidden.device)
        allowed = (source_positions[None, :] <= positions[:, None]) & (source_positions[None, :] > positions[:, None] - self.window)
        groups = self.heads // self.kv_heads
        keys, values = (tensor.repeat_interleave(groups, dim=1).transpose(1, 2) for tensor in (key, value))
        output = self.attention(query, keys, values, attn_mask=allowed)
        self.seen += length
        self.key, self.value = key[:, :, -self.retained:].contiguous(), value[:, :, -self.retained:].contiguous()
        return self.o_proj(output.reshape(batch, length, -1))


class TextMLP(AudioMLP):
    def forward(self, hidden, decode=False):
        projected = self.gate_up_proj(hidden)
        # Native compiled decode rounds only after SiLU and its gate product;
        # eager prefill rounds the activation before the product.
        activated = self.activation(projected.float() if decode else projected)
        return self.down_proj(activated.to(hidden.dtype))


class TextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        carrier = copy(config)
        carrier.attention_bias = False
        carrier.intermediate_size = config.ffn_dim // 2
        self.self_attn = CachedAttention(carrier, static=True)
        self.input_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TextMLP(carrier)

    def forward(self, hidden):
        # Kyutai applies learned norm weights before the output cast. Running
        # the unchanged norm on FP32 inputs preserves that rounding boundary.
        decode = self.self_attn.seen > 0
        activation_dtype = self.self_attn.q_proj.weight.dtype
        # Inductor keeps the decode residual sums in FP32 across both norms
        # and subsequent layers; only inputs to the BF16 GEMMs are rounded.
        if decode:
            hidden = hidden.float()
        normed = self.input_layernorm(hidden.float()).to(activation_dtype)
        hidden = hidden + self.self_attn(normed)
        normed = self.post_attention_layernorm(hidden.float()).to(activation_dtype)
        return hidden + self.mlp(normed, decode=decode)


class Kyutai(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size + config.num_codebooks * config.codebook_vocab_size + 1,
                                      config.hidden_size, padding_idx=config.audio_pad_token_id)
        self.layers = nn.ModuleList(TextLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNormNative(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.channel_sum, self.top1 = MoeSum(), CodecTop1()
    def prepare_codec(self, device):
        config = self.config
        self.codec_model = Mimi(config.codec_config)
        for layer in self.codec_model.encoder_transformer.layers:
            layer.self_attn = CachedAttention(config.codec_config)
        self._stream_convolutions(self.codec_model.encoder)
        self.codec_model.downsample = StreamingConv(self.codec_model.downsample)
        self.codec_model.to(device=device, dtype=torch.float32)

    def _stream_convolutions(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, CausalConv):
                setattr(module, name, StreamingConv(child))
            else:
                self._stream_convolutions(child)

    def reset(self):
        for module in self.modules():
            if isinstance(module, StreamingConv):
                module.previous = None
            elif isinstance(module, CachedAttention):
                module.key = module.value = None
                module.seen = 0

    def encode_window(self, values):
        hidden = self.codec_model.encoder(values.float()).transpose(1, 2)
        for layer in self.codec_model.encoder_transformer.layers:
            hidden = layer(hidden, None)
        hidden = self.codec_model.downsample(hidden.transpose(1, 2))
        return torch.cat((self.codec_model.semantic.encode(hidden), self.codec_model.acoustic.encode(hidden)), dim=1)

    def text_step(self, text_ids, audio_ids):
        config = self.config
        ids = torch.cat((text_ids[..., None], audio_ids), dim=-1)
        offsets = torch.cat((ids.new_zeros(1), torch.arange(config.num_codebooks, device=ids.device)
                             * config.codebook_vocab_size + config.vocab_size))
        ids = torch.where(ids == config.audio_pad_token_id, ids, ids + offsets)
        embedded = self.embed_tokens(ids)
        hidden = self.channel_sum(embedded.reshape(-1, config.hidden_size), config.num_codebooks + 1)
        hidden = hidden.reshape(*ids.shape[:2], config.hidden_size)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden.float()).to(self.lm_head.weight.dtype))

    def generate(self, values, generation, steps):
        if values.shape[0] != 1 or generation["audio_window_size"] != 1:
            raise ValueError("This case preserves native one-frame streaming for one waveform")
        self.reset()
        config, logits_steps = self.config, []
        sequences = torch.full((1, 1), generation["bos_token_id"], device=values.device, dtype=torch.long)
        start = end = 0
        audio = torch.full((1, 1, config.num_codebooks), config.audio_bos_token_id, device=values.device, dtype=torch.long)
        for step in range(min(steps, values.shape[-1] // config.frame_size)):
            if step - 1 >= end:
                audio = self.encode_window(values[..., start * config.frame_size:(start + 1) * config.frame_size]).transpose(1, 2)
                start, end = end, end + 1
            logits = self.text_step(sequences[:, -1:], audio)[:, -1].float()
            logits_steps.append(logits)
            sequences = torch.cat((sequences, self.top1(logits).reshape(1, 1)), dim=1)
        output = {"sequences": sequences, **{f"logits.{index}": logits for index, logits in enumerate(logits_steps)}}
        for index, layer in enumerate(self.layers):
            output[f"past_key_values.{index}.key"] = layer.self_attn.key
            output[f"past_key_values.{index}.value"] = layer.self_attn.value
        return output


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.codec_config.pad_mode != "constant" or config.tie_word_embeddings:
        raise ValueError("Selected Kyutai checkpoint requires SiLU, constant causal codec padding and untied head")
    model = Kyutai(config)
    # Fixed rotary constants must retain FP32 information for compiled decode.
    angles = [layer.self_attn.rotary_emb.cos_sin_cache for layer in model.layers]
    model.to(device=device, dtype=dtype)
    for layer, cache in zip(model.layers, angles):
        layer.self_attn.rotary_emb.cos_sin_cache = cache.to(device=device)
    # This is a strict native loader requirement, including BF16 execution.
    model.prepare_codec(device)
    return model.eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    codec = {name.removeprefix("codec_model."): remaining.pop(name)
             for name in list(remaining) if name.startswith("codec_model.")}
    load_mimi(model.codec_model, codec, config.codec_config)
    mapped = {}
    for name in model.state_dict():
        if name.startswith("codec_model."):
            continue
        source = "lm_head.weight" if name == "lm_head.weight" else "model." + name
        source = source.replace("model.embed_tokens.emb.", "model.embed_tokens.embed_tokens.")
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            source = source.replace(f".{projection}.weight", f".{projection}.linear.weight")
        source = source.replace(".mlp.gate_up_proj.", ".mlp.fc1.").replace(".mlp.down_proj.", ".mlp.fc2.")
        mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped Kyutai state: {sorted(remaining)}")
    mapped.update({"codec_model." + name: value for name, value in model.codec_model.state_dict().items()})
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case):
    return {"generate": Workload(run=lambda: model.generate(
        inputs["input_values"], case["reference"]["generation_config"],
        case["generation_kwargs"]["max_new_tokens"]))}
