"""Qwen2.5-Omni Thinker model.

Implements the text/image/video/audio Thinker path.  Speech generation
(`talker` and `token2wav`) is intentionally outside this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from ..L1.gelu import GELU
from ..L1.mrope_input_positions import MRopeInputPositions
from ..L1.rms_norm import RMSNorm
from ..L1.silu_and_mul import SiluAndMul
from ..L1.vision_rotary_emb import VisionRotaryEmbedding
from ..L2.parallel_embedding import ParallelLMHead
from ..L2.parallel_linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from ..L2.vision_attention import VisionAttention
from ..L2.vision_patch_embed import VisionPatchEmbed
from .qwen2_vl import Qwen2Model


@dataclass
class Qwen2_5OmniVisionConfig:
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 1280
    in_channels: int = 3
    num_heads: int = 16
    intermediate_size: int = 3420
    hidden_act: str = "silu"
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: list[int] = field(
        default_factory=lambda: [7, 15, 23, 31],
    )
    out_hidden_size: int = 2048
    tokens_per_second: int = 25


@dataclass
class Qwen2_5OmniAudioConfig:
    num_mel_bins: int = 128
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    max_source_positions: int = 1500
    n_window: int = 100
    output_dim: int = 2048


@dataclass
class Qwen2_5OmniConfig:
    model_type: str = "qwen2_5_omni"
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    mrope_interleaved: bool = False
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151646
    seconds_per_chunk: int = 2
    vision: Qwen2_5OmniVisionConfig = field(
        default_factory=Qwen2_5OmniVisionConfig,
    )
    audio: Qwen2_5OmniAudioConfig = field(
        default_factory=Qwen2_5OmniAudioConfig,
    )
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen2_5OmniConfig":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        thinker = hf.thinker_config
        text = thinker.text_config
        vc = thinker.vision_config
        ac = thinker.audio_config
        rope = (
            getattr(text, "rope_scaling", None)
            or getattr(text, "rope_parameters", None)
            or {}
        )
        cfg = cls(
            hidden_size=text.hidden_size,
            intermediate_size=text.intermediate_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=getattr(
                text, "head_dim",
                text.hidden_size // text.num_attention_heads,
            ),
            vocab_size=text.vocab_size,
            max_position_embeddings=text.max_position_embeddings,
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=getattr(text, "rope_theta", 1000000.0),
            tie_word_embeddings=getattr(text, "tie_word_embeddings", False),
            mrope_section=rope.get("mrope_section", [16, 24, 24]),
            image_token_id=getattr(
                thinker, "image_token_id", thinker.image_token_index,
            ),
            video_token_id=getattr(
                thinker, "video_token_id", thinker.video_token_index,
            ),
            audio_token_id=getattr(
                thinker, "audio_token_id", thinker.audio_token_index,
            ),
            seconds_per_chunk=getattr(thinker, "seconds_per_chunk", 2),
            vision=Qwen2_5OmniVisionConfig(
                depth=vc.depth,
                embed_dim=vc.embed_dim,
                hidden_size=vc.hidden_size,
                in_channels=vc.in_channels,
                num_heads=vc.num_heads,
                intermediate_size=vc.intermediate_size,
                hidden_act=getattr(vc, "hidden_act", "silu"),
                patch_size=vc.patch_size,
                spatial_merge_size=vc.spatial_merge_size,
                temporal_patch_size=vc.temporal_patch_size,
                window_size=vc.window_size,
                fullatt_block_indexes=list(vc.fullatt_block_indexes),
                out_hidden_size=vc.out_hidden_size,
                tokens_per_second=getattr(vc, "tokens_per_second", 25),
            ),
            audio=Qwen2_5OmniAudioConfig(
                num_mel_bins=ac.num_mel_bins,
                d_model=ac.d_model,
                encoder_layers=ac.encoder_layers,
                encoder_attention_heads=ac.encoder_attention_heads,
                encoder_ffn_dim=ac.encoder_ffn_dim,
                max_source_positions=ac.max_source_positions,
                n_window=ac.n_window,
                output_dim=ac.output_dim,
            ),
        )
        cfg._model_name = model_name
        return cfg


class Qwen2_5VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=True,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_up_proj(x)))


class Qwen2_5VisionPatchMerger(nn.Module):
    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.norm = RMSNorm(context_dim, eps=eps)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        return self.fc2(self.act(self.fc1(x)))


class Qwen2_5VisionBlock(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionConfig,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=norm_eps)
        self.attn = VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = Qwen2_5VisionMLP(
            config.hidden_size, config.intermediate_size,
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary_cos: torch.Tensor, rotary_sin: torch.Tensor,
                max_seqlen: int | None = None) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x), cu_seqlens, rotary_cos, rotary_sin, max_seqlen,
        )
        x_norm, residual = self.norm2(x, x_attn)
        return residual + self.mlp(x_norm)


class Qwen2_5VisionTransformer(nn.Module):
    def __init__(self, vision_config: Qwen2_5OmniVisionConfig,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.fullatt_block_indexes = set(vision_config.fullatt_block_indexes)
        self.out_hidden_size = vision_config.out_hidden_size

        self.patch_embed = VisionPatchEmbed(
            vision_config.patch_size,
            vision_config.temporal_patch_size,
            vision_config.in_channels,
            vision_config.hidden_size,
        )
        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([
            Qwen2_5VisionBlock(vision_config, norm_eps=norm_eps)
            for _ in range(vision_config.depth)
        ])
        self.merger = Qwen2_5VisionPatchMerger(
            vision_config.out_hidden_size,
            vision_config.hidden_size,
            vision_config.spatial_merge_size,
            eps=norm_eps,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def get_window_index_thw(self, grid_t: int, grid_h: int, grid_w: int):
        vit_window = self.window_size // self.spatial_merge_size // self.patch_size
        llm_h = grid_h // self.spatial_merge_size
        llm_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_h * llm_w).reshape(
            grid_t, llm_h, llm_w,
        )
        pad_h = vit_window - llm_h % vit_window
        pad_w = vit_window - llm_w % vit_window
        num_windows_h = (llm_h + pad_h) // vit_window
        num_windows_w = (llm_w + pad_w) // vit_window
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_window, num_windows_w, vit_window,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_window, vit_window,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens = torch.unique_consecutive(
            (seqlens.cumsum(0) * self.spatial_merge_unit).to(torch.int32),
        )
        return index_new, cu_seqlens

    @lru_cache(maxsize=1024)
    def get_rope_by_thw(self, t: int, h: int, w: int):
        window_index, cu_window = self.get_window_index_thw(t, h, w)
        cos, sin = self.rotary_emb(
            [[t, h, w]], self.spatial_merge_size, self.dtype,
            self.patch_embed.proj.weight.device,
        )
        cos = cos.reshape(
            cos.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )[window_index].flatten(0, 1)
        sin = sin.reshape(
            sin.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )[window_index].flatten(0, 1)
        cu_full = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t,
        )
        return window_index, cu_window, cos, sin, cu_full

    @staticmethod
    def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
        return inv

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor | list):
        device = self.patch_embed.proj.weight.device
        dtype = self.patch_embed.proj.weight.dtype
        hidden_states = self.patch_embed(x.to(device=device, dtype=dtype))
        seq_len = hidden_states.shape[0]

        if isinstance(grid_thw, torch.Tensor):
            grid_list = grid_thw.tolist()
        else:
            grid_list = grid_thw

        rotary_cos, rotary_sin = [], []
        window_indexes = []
        cu_window_parts = [torch.tensor([0], dtype=torch.int32)]
        cu_full_parts = []
        window_offset = 0
        cu_window_last = 0
        for t, h, w in grid_list:
            t, h, w = int(t), int(h), int(w)
            window_index, cu_window, cos, sin, cu_full = self.get_rope_by_thw(
                t, h, w,
            )
            window_indexes.append(window_index + window_offset)
            window_offset += t * (h // self.spatial_merge_size) * (
                w // self.spatial_merge_size
            )
            cu_window = cu_window + cu_window_last
            cu_window_last = cu_window[-1]
            cu_window_parts.append(cu_window)
            cu_full_parts.append(cu_full)
            rotary_cos.append(cos)
            rotary_sin.append(sin)

        window_index = torch.cat(window_indexes)
        reverse_index = self.invert_permutation(window_index)
        cu_window = torch.unique_consecutive(torch.cat(cu_window_parts))
        cu_full = torch.cumsum(torch.cat(cu_full_parts), dim=0, dtype=torch.int32)
        cu_full = F.pad(cu_full, (1, 0), "constant", 0)

        rotary_cos = torch.cat(rotary_cos).to(device=device, dtype=dtype)
        rotary_sin = torch.cat(rotary_sin).to(device=device, dtype=dtype)
        window_index = window_index.to(device=device)
        reverse_index = reverse_index.to(device=device)
        cu_window = cu_window.to(device=device)
        cu_full = cu_full.to(device=device)

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index].reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        max_full = int((cu_full[1:] - cu_full[:-1]).max().item())
        max_window = int((cu_window[1:] - cu_window[:-1]).max().item())
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in self.fullatt_block_indexes:
                cu_now, max_now = cu_full, max_full
            else:
                cu_now, max_now = cu_window, max_window
            hidden_states = block(
                hidden_states, cu_now, rotary_cos, rotary_sin, max_now,
            )

        hidden_states = self.merger(hidden_states)
        return hidden_states[reverse_index]


class Qwen2_5OmniThinkerForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "attn.q.": ("attn.qkv.", "q"),
        "attn.k.": ("attn.qkv.", "k"),
        "attn.v.": ("attn.qkv.", "v"),
        "mlp.gate_proj.": ("mlp.gate_up_proj.", 0),
        "mlp.up_proj.": ("mlp.gate_up_proj.", 1),
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2_5OmniConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen2_5VisionTransformer(
            config.vision, norm_eps=config.rms_norm_eps,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoder,
        )

        hf_cfg = AutoConfig.from_pretrained(
            getattr(config, "_model_name", "Qwen/Qwen2.5-Omni-3B"),
            trust_remote_code=True,
        )
        audio_config = hf_cfg.thinker_config.audio_config
        if hasattr(audio_config, "_attn_implementation"):
            audio_config._attn_implementation = "flash_attention_2"
            audio_config._attn_implementation_autoset = True
        self.audio_tower = Qwen2_5OmniAudioEncoder(audio_config)
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._mrope_positions = MRopeInputPositions()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_mrope_input_positions(
        self, input_tokens: list[int],
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
        video_second_per_grid: list[float] | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, int]:
        if image_grid_thw is None and video_grid_thw is None:
            pos = torch.arange(len(input_tokens), dtype=torch.int64)
            return pos.unsqueeze(0).expand(3, -1), 0
        return self._mrope_positions(
            input_tokens, self.config.vision.spatial_merge_size,
            image_grid_thw, video_grid_thw, image_offsets, video_offsets,
            video_second_per_grid=video_second_per_grid,
            tokens_per_second=getattr(self.config.vision, "tokens_per_second", 25),
        )

    def forward(self, input_ids, positions, inputs_embeds=None,
                deepstack_embeds=None):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def forward_with_lm_proj(self, input_ids, positions, inputs_embeds=None):
        hidden_states = self.model(
            input_ids, positions, inputs_embeds=inputs_embeds,
        )
        return self.lm_head.project(hidden_states)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        return self.lm_head.gather_greedy(partial_logits.float())
