"""MusicFlamingo's full audio-conditioned, cached greedy generation."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from ..patches.codec_top1 import CodecTop1
from . import audioflamingo3, qwen2, voxtral
from .qwen2_precision import NativeRotaryEmbedding, configure_language


class AudioBackbone(audioflamingo3.AudioBackbone):
    def __init__(self, text, config):
        super().__init__(text, config)
        self.time_config = config

    def features(self, features, mask):
        hidden, lengths = self.audio(features, mask)
        # These are positions derived from input token runs and feature lengths.
        token_mask = self.inputs["input_ids"] == self.audio_token_id
        diff = torch.diff(torch.nn.functional.pad(token_mask.int(), (1, 1)), dim=1)
        _, starts = torch.where(diff == 1)
        _, ends = torch.where(diff == -1)
        cumulative = torch.cat((torch.zeros(1, device=lengths.device), lengths.cumsum(0)[:-1]))
        samples = torch.searchsorted((ends - starts).cumsum(0), cumulative, right=True)
        first_rows = torch.searchsorted(samples, torch.arange(starts.numel(), device=lengths.device))
        windows = torch.arange(lengths.numel(), device=lengths.device) - first_rows[samples]
        count = hidden.shape[1]
        config = self.time_config
        timestamps = windows[:, None] * count * (config.audio_frame_step * 4)
        timestamps = timestamps + torch.arange(count, device=hidden.device, dtype=torch.float32) * (config.audio_frame_step * 4)
        dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
        frequencies = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2, device=hidden.device).float() / dim))
        time_positions = torch.arange(config.max_position_embeddings, device=hidden.device, dtype=torch.float32)
        time_positions = time_positions / config.max_position_embeddings * (2 * math.pi)
        time_angles = (time_positions[:, None] * frequencies).repeat_interleave(2, -1)[:count]
        window_positions = torch.round(timestamps[:, 0] / (config.audio_frame_step * 4 * count)) / config.max_position_embeddings
        window_angles = (window_positions[:, None] * frequencies).repeat_interleave(2, -1)
        angles = torch.cat((window_angles[:, None].expand(-1, count, -1), time_angles[None].expand(hidden.shape[0], -1, -1)), -1)
        angles = angles * (-timestamps * 2 * math.pi)[..., None]
        width = angles.shape[-1]
        cache = torch.cat((angles.cos()[..., ::2], angles.sin()[..., ::2]), -1).reshape(-1, width).double()
        prefix = hidden[..., :width].reshape(-1, width).double()
        positions = torch.arange(prefix.shape[0], device=hidden.device)
        # Reuse the parent's interleaved pointwise rotation in native FP64.
        rotated, _ = RotaryEmbedding.forward_native_interleaved(positions, prefix, prefix, width, cache)
        hidden = torch.cat((rotated.reshape(*hidden.shape[:2], width), hidden[..., width:].double()), -1).to(hidden.dtype)
        projected = self.projector(hidden)
        valid = torch.arange(count, device=hidden.device)[None] < lengths[:, None]
        return hidden, projected[valid]


def build_from_config(config, device, dtype):
    if config.text_config.use_cache or config.rope_parameters["rope_type"] != "default":
        raise ValueError("MusicFlamingo preserves the checkpoint's text config and default time RoPE")
    if config.projector_hidden_act != "gelu" or config.audio_config.activation_function != "gelu":
        raise ValueError("MusicFlamingo preserves its GELU audio encoder and projector")
    language = qwen2.build_from_config(config.text_config, device, dtype)
    configure_language(language.model, language.config)
    language.model.rotary_emb = NativeRotaryEmbedding(language.config.head_dim, language.config.max_position_embeddings, language.config.rope_theta)
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = language.model.rotary_emb
    model = audioflamingo3.AudioFlamingoModel(language, config)
    model.model = AudioBackbone(language.model, config)
    model.top1 = CodecTop1()
    return model.to(device=device, dtype=dtype).eval()


load_state_dict_into = audioflamingo3.load_state_dict_into


make_workloads = voxtral.make_workloads
