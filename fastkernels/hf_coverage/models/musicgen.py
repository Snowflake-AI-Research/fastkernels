"""MusicGen's documented text-conditioned public forward and audio-token logits."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.t5_encoder import T5Stack


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size, bias=False))
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden, memory=None, mask=None, past_key_value=None):
        batch, length, _ = hidden.shape
        source = hidden if memory is None else memory
        query = self.q_proj(hidden).reshape(batch, length, self.heads, self.width)
        if memory is not None and past_key_value is not None:
            key, value = past_key_value
        else:
            key, value = (op(source).reshape(batch, -1, self.heads, self.width).transpose(1, 2).contiguous()
                          for op in (self.k_proj, self.v_proj))
            if past_key_value is not None:
                key = torch.cat((past_key_value[0], key), dim=2)
                value = torch.cat((past_key_value[1], value), dim=2)
        causal = memory is None and past_key_value is None
        if memory is None and past_key_value is not None:
            positions = torch.arange(length, device=hidden.device) + past_key_value[0].shape[2]
            mask = (torch.arange(key.shape[2], device=hidden.device)[None, :] <= positions[:, None])[None, None]
        output = self.attention(query, key.transpose(1, 2), value.transpose(1, 2),
                                attn_mask=mask, causal=causal)
        return self.out_proj(output.reshape(batch, length, -1)), (key, value)


class Layer(nn.Module):
    def __init__(self, config, melody):
        super().__init__()
        self.self_attn = Attention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.fc1 = Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.fc2 = Linear(config.ffn_dim, config.hidden_size, bias=False)
        self.activation = GELU()
        if not melody:
            self.encoder_attn = Attention(config)
            self.encoder_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)

    def forward(self, hidden, memory, mask, past_key_value=None):
        previous_self, previous_cross = (None, None) if past_key_value is None else past_key_value
        attention, self_cache = self.self_attn(self.self_attn_layer_norm(hidden), past_key_value=previous_self)
        hidden = hidden + attention
        cross_cache = None
        if hasattr(self, "encoder_attn"):
            attention, cross_cache = self.encoder_attn(
                self.encoder_attn_layer_norm(hidden), memory, mask, previous_cross,
            )
            hidden = hidden + attention
        hidden = hidden + self.fc2(self.activation(self.fc1(self.final_layer_norm(hidden))))
        return hidden, (self_cache, cross_cache)


class Musicgen(nn.Module):
    def __init__(self, config, melody=False):
        super().__init__()
        self.config, self.melody = config, melody
        text, decoder = config.text_encoder, config.decoder
        self.shared = Embedding(text.vocab_size, text.d_model)
        self.text_encoder = T5Stack(text, self.shared)
        self.enc_to_dec_proj = Linear(text.d_model, decoder.hidden_size) if text.d_model != decoder.hidden_size else nn.Identity()
        if melody:
            self.audio_enc_to_dec_proj = Linear(config.num_chroma, decoder.hidden_size) if config.num_chroma != decoder.hidden_size else nn.Identity()
        self.embed_tokens = nn.ModuleList(Embedding(decoder.vocab_size + 1, decoder.hidden_size) for _ in range(decoder.num_codebooks))
        self.layers = nn.ModuleList(Layer(decoder, melody) for _ in range(decoder.num_hidden_layers))
        self.layer_norm = LayerNorm(decoder.hidden_size, eps=1e-5, promote_fp32=False)
        self.lm_heads = nn.ModuleList(Linear(decoder.hidden_size, decoder.vocab_size, bias=False) for _ in range(decoder.num_codebooks))
        half = decoder.hidden_size // 2
        # HF initializes this fixed table on CPU; GPU transcendental rounding
        # can change BF16 position values before the first decoder projection.
        with torch.device("cpu"):
            frequencies = torch.exp(torch.arange(half).float() * (-math.log(10000) / (half - 1)))
            angles = torch.arange(decoder.max_position_embeddings).float()[:, None] * frequencies[None, :]
            positions = torch.cat((angles.cos(), angles.sin()), -1)
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, ids, decoder_ids, attention_mask=None, encoder_hidden_states=None,
                past_key_values=None):
        memory = encoder_hidden_states
        if memory is None and ids is not None:
            memory = self.text_encoder(ids, attention_mask)
        conditioning = None if memory is None else self.enc_to_dec_proj(memory)
        if attention_mask is not None and conditioning is not None:
            # Boolean selection implements the text-padding mask without data arithmetic.
            conditioning = torch.where(attention_mask[..., None].bool(), conditioning, 0)
        if self.melody and conditioning is not None:
            chroma = conditioning.new_zeros((conditioning.shape[0], 1, self.config.num_chroma))
            chroma[..., 0] = 1
            chroma = self.audio_enc_to_dec_proj(chroma).repeat(1, self.config.chroma_length, 1)
            conditioning = torch.cat((chroma, conditioning), dim=1)
        channels = decoder_ids.reshape(-1, len(self.embed_tokens), decoder_ids.shape[-1])
        hidden = sum(embedding(channels[:, index]) for index, embedding in enumerate(self.embed_tokens))
        if self.melody and conditioning is not None:
            hidden = torch.cat((conditioning, hidden), dim=1)
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        if past_length + hidden.shape[1] > self.positions.shape[0]:
            raise ValueError("MusicGen sequence exceeds the prepared sinusoidal table")
        hidden = hidden + self.positions[past_length:past_length + hidden.shape[1]]
        output = {}
        if conditioning is not None:
            output["encoder_hidden_states" if self.melody else "encoder_last_hidden_state"] = conditioning if self.melody else memory
        mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        caches = []
        for index, layer in enumerate(self.layers):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, cache = layer(hidden, conditioning, mask, previous)
            caches.append(cache)
        hidden = self.layer_norm(hidden)
        logits = torch.stack([head(hidden) for head in self.lm_heads], 1)
        output["logits"] = logits.reshape(-1, *logits.shape[2:])
        output["past_key_values"] = caches
        return output


def build_from_config(config, device, dtype, *, melody=False):
    if (config.decoder.activation_function != "gelu" or not config.decoder.use_cache
            or config.text_encoder.is_gated_act or config.text_encoder.feed_forward_proj != "relu"):
        raise ValueError("MusicGen checkpoint requires ungated ReLU T5, GELU decoder and caches")
    return Musicgen(config, melody).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    model.inactive_state_names = sorted(name for name in remaining if name.startswith("audio_encoder."))
    for name in model.inactive_state_names:
        remaining.pop(name)
    shared = remaining.pop("text_encoder.shared.weight")
    if not torch.equal(shared, remaining.pop("text_encoder.encoder.embed_tokens.weight")):
        raise ValueError("MusicGen T5 input embeddings must be tied")
    mapped = {}
    for name in model.state_dict():
        if name in ("shared.emb.weight", "text_encoder.embed_tokens.emb.weight"):
            mapped[name] = shared
            continue
        source = name
        if name.startswith("text_encoder."):
            source = name.replace("text_encoder.", "text_encoder.encoder.", 1)
            source = source.replace("relative_attention_bias.emb.", "relative_attention_bias.")
            if ".qkv_proj." in source:
                mapped[name] = torch.cat([remaining.pop(source.replace("qkv_proj", part)) for part in ("q", "k", "v")], 0)
                continue
        elif name.startswith(("embed_tokens.", "layers.", "layer_norm.")):
            source = "decoder.model.decoder." + name.replace(".emb.", ".")
        elif name.startswith("lm_heads."):
            source = "decoder." + name
        mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped active MusicGen state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, case=None):
    ids = inputs["decoder_input_ids"]
    state = {}

    def call(start, end, previous=None):
        continuation = previous is not None
        return model(
            None if continuation else inputs["input_ids"], ids[:, start:end],
            attention_mask=None if model.melody and continuation else inputs.get("attention_mask"),
            encoder_hidden_states=(previous["encoder_last_hidden_state"]
                                   if continuation and not model.melody else None),
            past_key_values=previous["past_key_values"] if continuation else None,
        )

    def retain(output):
        state["output"] = output
        return {"logits": output["logits"]}

    def collect(_):
        output = state.pop("output")
        caches = output.pop("past_key_values")
        for index, pair in enumerate(caches):
            for kind, cache in zip(("self", "cross"), pair):
                if cache is not None:
                    for component, value in zip(("key", "value"), cache):
                        output[f"past_key_values.{index}.{kind}.{component}"] = value
        return output

    if case is None or case["workload"] == "forward":
        return {"forward": Workload(run=lambda: retain(call(0, ids.shape[1])), collect=collect)}
    prefix_length = ids.shape[1] - 2
    if prefix_length < 1:
        raise ValueError("MusicGen continuation needs a prefix and two audio-code tokens")

    def prepare(step):
        state["previous"] = call(0, prefix_length)
        if step == 1:
            state["previous"] = call(prefix_length, prefix_length + 1, state["previous"])

    workloads = {"prefill": Workload(run=lambda: retain(call(0, prefix_length)), collect=collect)}
    for step in range(2):
        workloads[f"decode_{step + 1}"] = Workload(
            run=lambda step=step: retain(call(prefix_length + step, prefix_length + step + 1, state["previous"])),
            prepare=lambda step=step: prepare(step), collect=collect,
        )
    return workloads
