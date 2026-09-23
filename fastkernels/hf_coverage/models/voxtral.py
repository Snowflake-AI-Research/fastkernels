"""Voxtral speech-conditioned greedy generation with its full audio encoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context, set_forward_context
from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from fastkernels.tasks.baseline.L4.whisper import WhisperConfig, WhisperEncoder
from . import llama
from .qwen2_precision import NativeRotaryEmbedding, configure_language
from .whisper import WhisperAttention


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        carrier = WhisperConfig(d_model=config.hidden_size, num_mel_bins=config.num_mel_bins,
                                max_source_positions=config.max_source_positions,
                                encoder_layers=config.num_hidden_layers,
                                encoder_attention_heads=config.num_attention_heads,
                                encoder_ffn_dim=config.intermediate_size)
        self.encoder = WhisperEncoder(carrier)
        for layer in self.encoder.layers:
            layer.self_attn = WhisperAttention(config.hidden_size, config.num_attention_heads)

    def forward(self, features):
        encoder = self.encoder
        hidden = encoder.gelu(encoder.conv1(features))
        hidden = encoder.gelu(encoder.conv2(hidden)).transpose(1, 2)
        # Native strict loading retains this position weight in FP32. Its
        # addition is promoted before casting the sum back to model dtype.
        hidden = (hidden + encoder.embed_positions.emb.weight).to(hidden.dtype)
        for layer in encoder.layers:
            hidden = layer(hidden)
        return encoder.layer_norm(hidden)


class Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.audio = AudioEncoder(config.audio_config)
        self.linear_1 = Linear(config.audio_config.intermediate_size, config.text_config.hidden_size, bias=False)
        self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.activation = GELU()
        self.audio_token_id = config.audio_token_id
        self.inputs = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, ids, positions):
        hidden = self.text.embed_tokens(ids)
        if get_context().is_prefill:
            audio = self.audio(self.inputs["input_features"])
            audio = audio.reshape(-1, self.linear_1.weight.shape[1])
            audio = self.linear_2(self.activation(self.linear_1(audio)))
            hidden = hidden.masked_scatter((ids == self.audio_token_id)[:, None].expand_as(hidden), audio)
        return self.text(ids, positions, inputs_embeds=hidden)


class Voxtral(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config, self.lm_head = text.config, text.lm_head
        self.model = Backbone(text.model, config)
        self.top1 = CodecTop1()


def build_language(text, dtype):
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "rms_norm_eps")
    carrier = LlamaConfig(**{name: getattr(text, name) for name in fields}, dtype=dtype,
                          rope_theta=text.rope_parameters["rope_theta"], rope_scaling_factor=1.0)
    language = LlamaForCausalLM(carrier)
    configure_language(language.model, carrier)
    language.model.rotary_emb = NativeRotaryEmbedding(carrier.head_dim, carrier.max_position_embeddings, carrier.rope_theta)
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = language.model.rotary_emb
    return language


def build_from_config(config, device, dtype):
    text, audio = config.text_config, config.audio_config
    if (config.projector_hidden_act != "gelu" or audio.activation_function != "gelu"
            or audio.intermediate_size != 4 * audio.hidden_size or text.tie_word_embeddings
            or text.hidden_act != "silu" or text.attention_bias or text.mlp_bias
            or not text.use_cache or text.rope_parameters["rope_type"] != "default"):
        raise ValueError("Voxtral requires the checkpoint's four-frame GELU audio projection and cached untied Llama")
    language = build_language(text, dtype)
    model = Voxtral(language, config).to(device=device, dtype=dtype)
    model.model.audio.encoder.embed_positions.float()
    return model.eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.removeprefix("language_model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("language_model.")}
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    encoder = model.model.audio.encoder
    mapped = {}
    for name in encoder.state_dict():
        source = name.replace(".conv.", ".").replace(".emb.", ".").replace(".mlp.fc", ".fc")
        mapped[name] = remaining.pop("audio_tower." + source)
    encoder.load_state_dict(mapped, strict=True)
    for name in ("linear_1", "linear_2"):
        getattr(model.model, name).load_state_dict({"weight": remaining.pop(f"multi_modal_projector.{name}.weight")}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped Voxtral state: {sorted(remaining)}")


def make_workloads(model, inputs, config, *, case):
    generation = case["reference"]["generation_config"]
    if generation.get("do_sample", False) or generation.get("num_beams", 1) != 1:
        raise ValueError("This declared Voxtral checkpoint uses greedy generation")
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("The speech generation development case uses one audio request")
    model.model.inputs = inputs
    prompt_length = ids.shape[1]
    steps = case["generation_kwargs"]["max_new_tokens"]
    attentions = [layer.self_attn.attn for layer in model.model.layers]
    block_size = attentions[0]._block_size
    total = prompt_length + steps
    blocks = (total + block_size - 1) // block_size
    device, dtype = ids.device, next(model.parameters()).dtype
    for attention in attentions:
        shape = (blocks, block_size, attention.num_kv_heads, attention.head_size)
        attention.k_cache = torch.empty(shape, device=device, dtype=dtype)
        attention.v_cache = torch.empty_like(attention.k_cache)
    tables = torch.arange(blocks, device=device, dtype=torch.int32)[None]
    eos = generation.get("eos_token_id")
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    eos_ids = torch.tensor(eos, device=device, dtype=torch.long)

    def run():
        sequences, outputs, length = ids, {}, prompt_length
        for step in range(steps):
            prefill = step == 0
            token_ids = sequences.reshape(-1) if prefill else sequences[:, -1]
            positions = torch.arange(prompt_length, device=device) if prefill else torch.tensor([length - 1], device=device)
            metadata = dict(slot_mapping=positions, block_tables=tables,
                            req_id_per_token=torch.zeros_like(positions, dtype=torch.int32))
            if prefill:
                cumulative = torch.tensor([0, prompt_length], device=device, dtype=torch.int32)
                metadata.update(cu_seqlens_q=cumulative, cu_seqlens_k=cumulative,
                                max_seqlen_q=prompt_length, max_seqlen_k=prompt_length)
            else:
                metadata["context_lens"] = torch.tensor([length], device=device, dtype=torch.int32)
            with set_forward_context(is_prefill=prefill, **metadata):
                hidden = model.model(token_ids, positions)
            logits = model.lm_head.linear_op(hidden[-1:], model.lm_head.embedding_op.emb.weight).float()
            outputs[f"logits.{step}"] = logits
            next_ids = model.top1(logits).reshape(1, 1)
            sequences = torch.cat((sequences, next_ids), 1)
            if eos and torch.any(next_ids == eos_ids).item():
                break
            length += 1
        cache_length = sequences.shape[1] - 1
        for index, attention in enumerate(attentions):
            for name, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
                outputs[f"past_key_values.{index}.{name}"] = cache.reshape(-1, attention.num_kv_heads, attention.head_size)[:cache_length].transpose(0, 1)[None]
        return {"sequences": sequences, **outputs}

    return {"generate": Workload(run=run)}
