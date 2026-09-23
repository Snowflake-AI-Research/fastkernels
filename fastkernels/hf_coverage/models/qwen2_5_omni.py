"""Qwen2.5 Omni's full thinker, talker and text-plus-waveform generation."""

import math
from dataclasses import fields

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.mrope import MRotaryEmbedding
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L4.qwen2_5_omni import (
    Qwen2_5OmniVisionConfig, Qwen2_5VisionTransformer,
)
from ..patches.codec_top1 import CodecTop1
from ..patches.grouped_dense_attention import GroupedDenseAttention
from .qwen_omni_sampling import OmniSampling
from ..runner import Workload


def positions_for_inputs(ids, inputs, config):
    """Position metadata for ordinary separate audio, image and video segments."""
    if ids.shape[0] != 1:
        raise ValueError("The native audio-output parent supports one sequence")
    tokens = ids[0].tolist()
    merge = config.spatial_merge_size if hasattr(config, "spatial_merge_size") else config.vision_config.spatial_merge_size
    grids = {
        config.image_token_index: iter(inputs.get("image_grid_thw", []).tolist() if "image_grid_thw" in inputs else []),
        config.video_token_index: iter(inputs.get("video_grid_thw", []).tolist() if "video_grid_thw" in inputs else []),
    }
    seconds = iter(inputs.get("video_second_per_grid", [1.0]))
    pieces, index, current = [], 0, 0
    while index < len(tokens):
        token = tokens[index]
        if token not in grids:
            pieces.append(torch.full((3, 1), current, dtype=torch.long, device=ids.device))
            index, current = index + 1, current + 1
            continue
        t, h, w = next(grids[token])
        h, w = h // merge, w // merge
        interval = config.position_id_per_seconds
        if token == config.video_token_index:
            interval *= float(next(seconds))
        temporal = (torch.arange(t, device=ids.device) * interval).long().repeat_interleave(h * w)
        height = torch.arange(h, device=ids.device).repeat_interleave(w).repeat(t)
        width = torch.arange(w, device=ids.device).repeat(h * t)
        pieces.append(torch.stack((temporal, height, width)) + current)
        current += max(int(temporal[-1]) + 1, h, w)
        count = t * h * w
        if tokens[index:index + count] != [token] * count:
            raise ValueError("Visual placeholder count does not match its grid")
        index += count
    positions = torch.cat(pieces, -1)
    return positions, current - len(tokens)


class TextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = getattr(config, "head_dim", None) or config.hidden_size // self.heads
        self.q_proj = Linear(config.hidden_size, self.heads * self.dim)
        self.k_proj = Linear(config.hidden_size, self.kv_heads * self.dim)
        self.v_proj = Linear(config.hidden_size, self.kv_heads * self.dim)
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, bias=False)
        self.attention = (
            GroupedDenseAttention() if self.heads != self.kv_heads
            else DenseAttention(backend="sdpa")
        )
        self.key = self.value = None

    def forward(self, hidden, positions, rotary):
        batch, length, _ = hidden.shape
        q = self.q_proj(hidden).reshape(length, self.heads, self.dim)
        k = self.k_proj(hidden).reshape(length, self.kv_heads, self.dim)
        v = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.dim)
        q, k = rotary.forward_native_2d(positions, q, k)
        q, k = q[None], k[None]
        fresh = self.key is None
        self.key = k if fresh else torch.cat((self.key, k), 1)
        self.value = v if fresh else torch.cat((self.value, v), 1)
        update = self.attention(q, self.key, self.value, causal=fresh and length > 1)
        return self.o_proj(update.reshape(batch, length, -1))


class TextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = SiluAndMul()

    def forward(self, hidden):
        packed = torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), -1)
        return self.down_proj(self.activation.forward_native(packed))


class TextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.mlp = TextAttention(config), TextMLP(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions, rotary):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, rotary)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TextModel(nn.Module):
    def __init__(self, config, embedding_size=None):
        super().__init__()
        if config.use_sliding_window or config.hidden_act != "silu":
            raise ValueError("The selected Omni checkpoint uses full attention and SiLU")
        self.embed_tokens = Embedding(config.vocab_size, embedding_size or config.hidden_size)
        self.layers = nn.ModuleList([TextLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        rope = config.rope_parameters
        if rope["rope_type"] != "default":
            raise ValueError("The selected Omni checkpoint uses default multimodal RoPE")
        dim = self.layers[0].self_attn.dim
        self.rotary = MRotaryEmbedding(dim, config.max_position_embeddings, rope["rope_theta"], rope["mrope_section"])

    def reset(self):
        for layer in self.layers:
            layer.self_attn.key = layer.self_attn.value = None

    def forward(self, embeddings, positions):
        hidden = embeddings
        for layer in self.layers:
            hidden = layer(hidden, positions, self.rotary)
        return self.norm(hidden)


class AudioLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.encoder_attention_heads, config.d_model // config.encoder_attention_heads
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, Linear(config.d_model, config.d_model, bias=name != "k_proj"))
        self.attention = DenseAttention(backend="sdpa")
        self.self_attn_layer_norm = LayerNorm(config.d_model)
        self.final_layer_norm = LayerNorm(config.d_model)
        self.fc1, self.fc2 = Linear(config.d_model, config.encoder_ffn_dim), Linear(config.encoder_ffn_dim, config.d_model)
        self.activation = GELU()

    def forward(self, hidden, mask):
        normalized = self.self_attn_layer_norm(hidden)
        shape = (1, hidden.shape[0], self.heads, self.dim)
        q, k, v = [getattr(self.self_attn, name)(normalized).reshape(shape) for name in ("q_proj", "k_proj", "v_proj")]
        hidden = hidden + self.self_attn.out_proj(self.attention(q, k, v, attn_mask=mask).reshape_as(hidden))
        return hidden + self.fc2(self.activation(self.fc1(self.final_layer_norm(hidden))))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = Conv1dNative(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = Conv1dNative(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.audio_bos_eos_token = Embedding(2, config.output_dim)
        self.layers = nn.ModuleList([AudioLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post, self.proj = LayerNorm(config.d_model), Linear(config.d_model, config.output_dim)
        self.pool, self.activation = AvgPool2d((1, 2), stride=(1, 2)), GELU()
        frequencies = torch.exp(-math.log(10000) / (config.d_model // 2 - 1) * torch.arange(config.d_model // 2).float())
        angles = torch.arange(config.max_source_positions)[:, None] * frequencies[None]
        self.register_buffer("positions", torch.cat((angles.sin(), angles.cos()), -1), persistent=False)

    def forward(self, features, feature_mask):
        lengths = feature_mask.sum(-1).tolist()
        chunks, chunk_lengths = [], []
        for row, length in zip(features, lengths):
            for start in range(0, length, 2 * self.config.n_window):
                chunk = row[:, start:min(start + 2 * self.config.n_window, length)]
                chunks.append(chunk)
                chunk_lengths.append(chunk.shape[-1])
        maximum = max(chunk_lengths)
        padded = features.new_zeros((len(chunks), features.shape[1], maximum))
        valid = torch.zeros((len(chunks), maximum), dtype=torch.bool, device=features.device)
        for index, chunk in enumerate(chunks):
            padded[index, :, :chunk.shape[-1]] = chunk
            valid[index, :chunk.shape[-1]] = True
        hidden = self.activation(self.conv1(padded)).masked_fill(~valid[:, None], 0)
        hidden = self.activation(self.conv2(hidden)).transpose(1, 2)
        hidden = hidden + self.positions[:hidden.shape[1]][None]
        post_lengths = [(n - 1) // 2 + 1 for n in chunk_lengths]
        hidden = torch.cat([row[:length] for row, length in zip(hidden, post_lengths)], 0)
        mask = torch.full((1, 1, hidden.shape[0], hidden.shape[0]), torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype)
        offset = 0
        for length in post_lengths:
            mask[..., offset:offset + length, offset:offset + length] = 0
            offset += length
        for layer in self.layers:
            hidden = layer(hidden, mask)
        parts, offset = [], 0
        for length in lengths:
            length = (length - 1) // 2 + 1
            audio = hidden[offset:offset + length]
            pooled = self.pool(audio.T[None, :, None, :])[0, :, 0, :].T
            parts.append(self.proj(self.ln_post(pooled)))
            offset += length
        return torch.cat(parts)


class Thinker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TextModel(config.text_config)
        self.lm_head = Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.audio_tower = AudioEncoder(config.audio_config)
        visual = Qwen2_5OmniVisionConfig(**{f.name: getattr(config.vision_config, f.name) for f in fields(Qwen2_5OmniVisionConfig) if hasattr(config.vision_config, f.name)})
        self.visual = Qwen2_5VisionTransformer(visual, norm_eps=config.text_config.rms_norm_eps)

    def embeddings(self, ids, inputs):
        hidden = self.model.embed_tokens(ids)
        if "input_features" in inputs:
            hidden[ids == self.config.audio_token_index] = self.audio_tower(inputs["input_features"], inputs["feature_attention_mask"])
        for token, pixels, grid in ((self.config.image_token_index, "pixel_values", "image_grid_thw"), (self.config.video_token_index, "pixel_values_videos", "video_grid_thw")):
            if pixels in inputs:
                hidden[ids == token] = self.visual(inputs[pixels], inputs[grid].detach().cpu())
        return hidden


class Talker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thinker_to_talker_proj = Linear(config.embedding_size, config.hidden_size)
        self.model = TextModel(config, config.embedding_size)
        self.codec_head = Linear(config.hidden_size, config.vocab_size, bias=False)


class Omni(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .qwen2_5_omni_waveform import Token2Wav

        self.config = config
        self.thinker, self.talker = Thinker(config.thinker_config), Talker(config.talker_config)
        self.token2wav = Token2Wav(config.token2wav_config)
        self.greedy = CodecTop1()
        self.sampling = OmniSampling(40, .8, .9, 1.05)
        self.speakers = None

    def generate(self, inputs, thinker_steps, talker_steps, speaker="Chelsie"):
        thinker, talker = self.thinker, self.talker
        thinker.model.reset()
        talker.model.reset()
        ids = inputs["input_ids"]
        speaker = self.speakers[speaker]
        initial_embeddings = thinker.embeddings(ids, inputs)
        positions, delta = positions_for_inputs(ids, inputs, thinker.config)
        embeddings = initial_embeddings
        history, embedding_steps, hidden_steps = ids, [], []
        for step in range(thinker_steps):
            hidden = thinker.model(embeddings, positions)
            embedding_steps.append(embeddings)
            hidden_steps.append(hidden)
            next_token = self.greedy(thinker.lm_head(hidden[:, -1]).float())[:, None]
            history = torch.cat((history, next_token), -1)
            if int(next_token[0, 0]) == thinker.config.eos_token_id:
                break
            embeddings = thinker.model.embed_tokens(next_token)
            positions = torch.full((3, 1), ids.shape[1] + step + delta, dtype=torch.long, device=ids.device)
        if len(hidden_steps) < 2:
            raise ValueError("Native Omni speech conditioning requires at least two thinker steps")
        cleaned = initial_embeddings.clone()
        for token in (thinker.config.audio_token_index, thinker.config.image_token_index, thinker.config.video_token_index):
            cleaned[ids == token] = 0
        reply = torch.cat(hidden_steps[1:], 1) + torch.cat(embedding_steps[1:], 1)
        tc = talker.config
        bos = ids.new_tensor([[speaker["bos_token"]]])
        bos_embedding = thinker.model.embed_tokens(bos)
        prefill = torch.cat((hidden_steps[0] + cleaned, bos_embedding, reply[:, :1]), 1)
        prefill[:, -2] += talker.model.embed_tokens(ids.new_tensor([tc.tts_codec_pad_token_id]))
        prefill[:, -1] += talker.model.embed_tokens(ids.new_tensor([tc.tts_codec_start_token_id]))
        reply = torch.cat((reply[:, 1:], thinker.model.embed_tokens(ids.new_tensor([[tc.tts_text_end_token_id]])), thinker.model.embed_tokens(ids.new_tensor([[tc.tts_text_pad_token_id]]))), 1)
        input_text_ids = torch.cat((ids, bos, history[:, ids.shape[1]:ids.shape[1] + 1]), -1)
        talk_positions, talk_delta = positions_for_inputs(input_text_ids, inputs, tc)
        codec_history = torch.cat((torch.full_like(ids, tc.tts_codec_mask_token_id), ids.new_tensor([[tc.tts_codec_pad_token_id, tc.tts_codec_start_token_id]])), -1)
        embeddings, codes = prefill, []
        for step in range(talker_steps):
            hidden = talker.model(talker.thinker_to_talker_proj(embeddings), talk_positions)
            logits = talker.codec_head(hidden[:, -1]).float()
            logits[:, tc.tts_codec_start_token_id] = -float("inf")
            token = self.sampling(logits, codec_history)[:, None]
            codec_history = torch.cat((codec_history, token), -1)
            codes.append(token)
            if int(token[0, 0]) in (8292, 8294):
                break
            embeddings = talker.model.embed_tokens(token) + reply[:, :1]
            if reply.shape[1] > 1:
                reply = reply[:, 1:]
            talk_positions = torch.full((3, 1), prefill.shape[1] + step + talk_delta, dtype=torch.long, device=ids.device)
        # Native parent always excludes the final talker token, including at
        # the declared max-new-token stopping boundary.
        code = torch.cat(codes, -1)[:, :-1]
        waveform = self.token2wav(code, speaker["cond"].to(ids.device).float(), speaker["ref_mel"].to(ids.device).float())
        return {"sequences": history, "waveform": waveform.float()}


def build_from_config(config, device, dtype):
    if not config.enable_audio_output:
        raise ValueError("The selected public Omni parent returns text and audio")
    model = Omni(config).to(device=device, dtype=dtype).eval()
    model.token2wav.float()
    return model


@torch.no_grad()
def load_state_dict_into(model, weights, config):
    remaining = dict(weights)
    mapped = {}
    for name, target in model.state_dict().items():
        if name.startswith("token2wav."):
            continue
        source = name.replace(".embed_tokens.emb.", ".embed_tokens.").replace(".audio_bos_eos_token.emb.", ".audio_bos_eos_token.")
        if name.startswith("thinker.visual."):
            if name.endswith("inv_freq"):
                mapped[name] = target
                continue
            source = source.replace("patch_embed.proj.conv.", "patch_embed.proj.")
            source = source.replace("merger.norm.", "merger.ln_q.").replace("merger.fc1.", "merger.mlp.0.").replace("merger.fc2.", "merger.mlp.2.")
            if ".attn.qkv." in source:
                value = torch.cat([remaining.pop(source.replace(".qkv.", f".{key}.")) for key in ("q", "k", "v")])
            elif ".mlp.gate_up_proj." in source:
                value = torch.cat([remaining.pop(source.replace("gate_up_proj", key)) for key in ("gate_proj", "up_proj")])
            else:
                value = remaining.pop(source)
        else:
            value = remaining.pop(source)
        if value.shape != target.shape:
            raise ValueError(f"Omni mapping mismatch {source}: {value.shape} != {target.shape}")
        mapped[name] = value
    model.load_state_dict(mapped, strict=False)
    from .qwen2_5_omni_waveform import load_state_dict_into as load_waveform

    waveform = {name.removeprefix("token2wav."): remaining.pop(name) for name in list(remaining) if name.startswith("token2wav.")}
    load_waveform(model.token2wav, waveform, config.token2wav_config)
    if remaining:
        raise KeyError(f"Unmapped Omni weights: {sorted(remaining)}")


def make_workloads(model, inputs, config, case):
    from huggingface_hub import hf_hub_download

    speaker = case["reference"]["speakers"]
    path = hf_hub_download(speaker["repo"], speaker["filename"], revision=speaker["revision"])
    model.speakers = torch.load(path, weights_only=True)
    steps = case["generation_kwargs"]
    return {"generate": Workload(run=lambda: model.generate(inputs, steps["thinker_max_new_tokens"], steps["talker_max_new_tokens"]))}
