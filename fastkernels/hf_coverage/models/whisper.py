"""Whisper's full encoder/decoder using existing layers and explicit HF Q scaling."""

from dataclasses import fields

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.whisper import (
    WhisperConfig, WhisperDecoder, WhisperEncoder,
)


class WhisperAttention(nn.Module):
    """Compose projections and SDPA with HF's pre-matmul query scaling."""

    def __init__(self, width, heads, causal=False, cache=False):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.causal, self.cache = causal, cache
        self.q_proj = Linear(width, width)
        self.k_proj = Linear(width, width, bias=False)
        self.v_proj = Linear(width, width)
        self.out_proj = Linear(width, width)
        self.attention = DenseAttention(backend="sdpa")
        self.last_cache = None

    def forward(self, hidden, memory=None, past_key_value=None):
        source = hidden if memory is None else memory
        batch, length = hidden.shape[:2]
        query = (self.q_proj(hidden) * self.head_dim**-0.5).view(
            batch, length, self.heads, self.head_dim,
        ).transpose(1, 2).contiguous()
        if memory is not None and past_key_value is not None:
            key, value = past_key_value
        else:
            key, value = (
                projection(source).view(batch, -1, self.heads, self.head_dim)
                .transpose(1, 2).contiguous()
                for projection in (self.k_proj, self.v_proj)
            )
        if self.cache:
            if past_key_value is None:
                # DynamicCache retains contiguous copies on its first update.
                key, value = key.clone(), value.clone()
            elif memory is None:
                if length != 1:
                    raise ValueError("Whisper continuation evaluates one new token per call")
                key, value = (torch.cat((old, new), dim=2)
                              for old, new in zip(past_key_value, (key, value)))
            self.last_cache = (key, value)
        context = self.attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            softmax_scale=1.0, causal=self.causal and past_key_value is None,
        )
        return self.out_proj(context.reshape(batch, length, -1))


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        values = {field.name: getattr(config, field.name) for field in fields(WhisperConfig)
                  if hasattr(config, field.name)}
        values.update(hidden_size=config.d_model,
                      num_hidden_layers=config.decoder_layers,
                      num_attention_heads=config.decoder_attention_heads,
                      num_key_value_heads=config.decoder_attention_heads,
                      head_dim=config.d_model // config.decoder_attention_heads)
        carrier = WhisperConfig(**values)
        self.encoder, self.decoder = WhisperEncoder(carrier), WhisperDecoder(carrier)
        for layer in self.encoder.layers:
            layer.self_attn = WhisperAttention(config.d_model, config.encoder_attention_heads)
        for layer in self.decoder.layers:
            layer.self_attn = WhisperAttention(config.d_model, config.decoder_attention_heads,
                                               causal=True, cache=True)
            layer.encoder_attn = WhisperAttention(config.d_model, config.decoder_attention_heads,
                                                  cache=True)
        self.proj_out = Linear(config.d_model, config.vocab_size, bias=False)
        self.proj_out.weight = self.decoder.embed_tokens.emb.weight

    def forward(self, input_features, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("Whisper coverage evaluates unpadded feature and token sequences")
        memory = self.encoder(input_features) if encoder_hidden_states is None else encoder_hidden_states
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device) + past_length
        hidden = self.decoder.embed_tokens(decoder_input_ids) + self.decoder.embed_positions(positions)
        cache = []
        for index, layer in enumerate(self.decoder.layers):
            previous = (None, None) if past_key_values is None else past_key_values[index]
            hidden = hidden + layer.self_attn(layer.self_attn_layer_norm(hidden),
                                              past_key_value=previous[0])
            hidden = hidden + layer.encoder_attn(layer.encoder_attn_layer_norm(hidden),
                                                 memory, previous[1])
            hidden = hidden + layer.mlp(layer.final_layer_norm(hidden))
            cache.append((layer.self_attn.last_cache, layer.encoder_attn.last_cache))
        hidden = self.decoder.layer_norm(hidden)
        return {"logits": self.proj_out(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": tuple(cache)}


def build_from_config(config, device, dtype):
    if (config.activation_function != "gelu" or config.scale_embedding
            or not config.tie_word_embeddings or not config.use_cache):
        raise ValueError("This Whisper case requires the documented GELU, unscaled, tied, cached model")
    return WhisperForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for target, parameter in model.state_dict().items():
        if target == "proj_out.weight":
            source = target
        else:
            source = "model." + target.replace(".conv.weight", ".weight").replace(
                ".conv.bias", ".bias",
            ).replace(".emb.weight", ".weight").replace(".mlp.fc", ".fc")
        value = state[source]
        if value.shape != parameter.shape:
            raise ValueError(f"Whisper weight shape mismatch at {source}")
        mapped[target] = value
        consumed.add(source)
    if consumed != set(state):
        raise ValueError(f"Whisper unmapped state: {sorted(set(state) - consumed)}")
    if not torch.equal(state["proj_out.weight"], state["model.decoder.embed_tokens.weight"]):
        raise ValueError("Whisper's loaded projection and embedding must be tied")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if inputs["input_features"].shape[-1] != 2 * config.max_source_positions:
        raise ValueError("Whisper requires twice max_source_positions input frames")
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs, encoder_input_name="input_features")

    def run():
        output = model(**inputs)
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
