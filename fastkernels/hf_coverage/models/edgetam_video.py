"""EdgeTAM video with composed spatial Perceiver and partitioned memory RoPE."""

import copy
import math
import torch
from torch import nn

from fastkernels.hf_coverage.models import edgetam, sam2_video
from fastkernels.hf_coverage.models.sam import SamAttentionCore
from fastkernels.hf_coverage.patches.sam_sine_dtype import SamSineDtype
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.sam3_vit_block import _window_partition


class PerceiverAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.perceiver_resampler_num_attention_heads
        self.dim = c.perceiver_resampler_attention_head_dim
        width, inner = c.perceiver_resampler_hidden_size, self.heads * self.dim
        self.q_proj, self.k_proj, self.v_proj = (Linear(width, inner, bias=False) for _ in range(3))
        self.o_proj, self.attend = Linear(inner, width, bias=False), SamAttentionCore()

    def forward(self, query, source, position=None):
        batch, length = query.shape[:2]
        q = self.q_proj(query).view(batch, length, self.heads, self.dim)
        k = self.k_proj(source).view(batch, -1, self.heads, self.dim)
        v = self.v_proj(source).view(batch, -1, self.heads, self.dim)
        if position is not None:
            position = position.view_as(k)
            k, v = k + position, v + position
        context = self.attend(q, k, v).transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(context)


class PerceiverMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        width, inner = c.perceiver_resampler_hidden_size, c.perceiver_resampler_mlp_intermediate_size
        self.layer_norm = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.up_proj, self.down_proj = Linear(width, inner, bias=False), Linear(inner, width, bias=False)
        self.act = GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(self.layer_norm(x))))


class PerceiverLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cross_attention, self.self_attention = PerceiverAttention(c), PerceiverAttention(c)
        self.mlp, self.self_mlp = PerceiverMLP(c), PerceiverMLP(c)
        for name in ("layer_norm_input", "layer_norm_latents", "layer_norm_self"):
            setattr(self, name, LayerNorm(c.perceiver_resampler_hidden_size, eps=1e-5, promote_fp32=False))

    def forward(self, latents, source, position=None):
        latents = latents + self.cross_attention(self.layer_norm_latents(latents), self.layer_norm_input(source), position)
        latents = latents + self.mlp(latents)
        normalized = self.layer_norm_self(latents)
        latents = latents + self.self_attention(normalized, normalized)
        return latents + self.self_mlp(latents)


class Perceiver(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.latents_1d = nn.Parameter(torch.empty(c.perceiver_resampler_num_latents, c.perceiver_resampler_hidden_size))
        self.latents_2d = nn.Parameter(torch.empty(c.perceiver_resampler_num_latents_2d, c.perceiver_resampler_hidden_size))
        self.layers = nn.ModuleList([PerceiverLayer(c) for _ in range(c.perceiver_resampler_num_layers)])
        self.layer_norm = LayerNorm(c.perceiver_resampler_hidden_size, eps=1e-5, promote_fp32=False)
        self.position = SamSineDtype(c.perceiver_resampler_hidden_size)

    def forward(self, features, positions):
        batch, channels, height, _ = features.shape
        latents = self.latents_1d[None].expand(batch, -1, -1)
        source = features.permute(0, 2, 3, 1).flatten(1, 2)
        position = positions.permute(0, 2, 3, 1).flatten(1, 2)
        for layer in self.layers:
            latents = layer(latents, source, position)
        global_latents = self.layer_norm(latents)
        side = math.isqrt(self.latents_2d.shape[0])
        latents = self.latents_2d[None].expand(batch, -1, -1).reshape(-1, 1, channels)
        source, _ = _window_partition(features.permute(0, 2, 3, 1), height // side)
        for layer in self.layers:
            latents = layer(latents, source.flatten(1, 2))
        image = latents.view(batch, side, side, channels).permute(0, 3, 1, 2)
        position = self.position(image).to(features.dtype).permute(0, 2, 3, 1).flatten(1, 2)
        local_latents = self.layer_norm(image.permute(0, 2, 3, 1).flatten(1, 2))
        return (torch.cat((global_latents, local_latents), dim=1),
                torch.cat((torch.zeros_like(global_latents), position), dim=1))


class CrossAttention(sam2_video.MemoryAttention):
    def __init__(self, c):
        super().__init__(c, c.memory_encoder_output_channels)
        self.global_tokens = c.perceiver_resampler_num_latents
        self.local_tokens = c.perceiver_resampler_num_latents_2d
        key_config = copy.copy(c)
        key_config.memory_attention_rope_feat_sizes = c.memory_attention_rope_k_sizes
        self.register_buffer("key_rotary_table", self.position_table(key_config, "cpu"), persistent=False)

    def forward(self, q, k, v, num_k_exclude_rope=0):
        query, key, value = self.rotate(self.q_proj(q)), self.k_proj(k), self.v_proj(v)
        batch, length = key.shape[:2]
        groups = (length - num_k_exclude_rope) // (self.global_tokens + self.local_tokens)
        memory = key[:, :length-num_k_exclude_rope].reshape(batch, groups, self.global_tokens+self.local_tokens, -1)
        spatial = memory[:, :, self.global_tokens:].reshape(batch, -1, key.shape[-1])
        # Reuse the same interleaved rotary operation with the local spatial grid.
        spatial = self.rotate(spatial, self.key_rotary_table).reshape(batch, groups, self.local_tokens, -1)
        rotated = torch.cat((memory[:, :, :self.global_tokens], spatial), dim=2).reshape(batch, -1, key.shape[-1])
        key = torch.cat((rotated, key[:, length-num_k_exclude_rope:]), dim=1)
        values = [x.reshape(batch, -1, self.heads, self.head_dim) for x in (query, key, value)]
        return self.out_proj(self.attend(*values).reshape(batch, q.shape[1], -1))


class EdgeTamVideoModel(sam2_video.Sam2VideoModel):
    def __init__(self, config):
        c = copy.copy(config)
        c.memory_attention_feed_forward_hidden_act = c.memory_attention_mlp_hidden_act
        c.memory_attention_feed_forward_hidden_size = c.memory_attention_mlp_hidden_size
        super().__init__(c, vision_encoder=edgetam.VisionEncoder)
        if c.enable_occlusion_spatial_embedding or c.enable_temporal_pos_encoding_for_object_pointers:
            raise ValueError("The declared EdgeTAM checkpoint disables these optional embeddings")
        del self.occlusion_spatial_embedding_parameter
        self.temporal_positional_encoding_projection_layer = nn.Identity()
        self.spatial_perceiver = Perceiver(c)
        for layer in self.memory_attention.layers:
            layer.cross_attn_image = CrossAttention(c)

    def memory_state(self, memory, appearing, high):
        features, position = self.spatial_perceiver(memory["vision_features"], memory["vision_pos_enc"][0].to(high.dtype))
        return features.to(high.dtype), position.to(high.dtype)

    def frame(self, pixels, points, labels, history, index, total_frames):
        outputs = super().frame(pixels, points, labels, history, index, total_frames)
        # EdgeTAM consumes the computed high-resolution mask in its memory encoder
        # but does not retain it in the public session's per-frame state.
        del outputs["state.high_res_masks"]
        del history[-1]["high_res_masks"]
        return outputs

    def condition(self, features, positions, history, index, total_frames):
        if index == 0:
            return features[-1] + self.no_memory_embedding.reshape(1, -1, 1, 1)
        c = self.config
        selected = [(0, history[0])]
        selected.extend((offset, history[index-offset]) for offset in range(c.num_maskmem-1, 0, -1)
                        if 0 < index-offset < len(history))
        memories = [previous["maskmem_features"].transpose(0, 1) for _, previous in selected]
        positional = [previous["maskmem_pos_enc"].transpose(0, 1) + self.memory_temporal_positional_encoding[offset-1]
                      for offset, previous in selected]
        pointer_frames = [0] + [index-offset for offset in range(1, min(total_frames, c.max_object_pointers_in_encoder))
                                if index-offset > 0]
        pointers = torch.stack([history[t]["object_pointer"] for t in pointer_frames])
        splits = c.vision_config.fpn_hidden_size // c.memory_encoder_output_channels
        pointers = pointers.reshape(-1, 1, splits, c.memory_encoder_output_channels).permute(0, 2, 1, 3).flatten(0, 1)
        memories.append(pointers)
        positional.append(torch.zeros_like(pointers))
        memory, position = torch.cat(memories).transpose(0, 1), torch.cat(positional).transpose(0, 1)
        query_position = positions[-1].flatten(2).transpose(1, 2)
        query = features[-1].flatten(2).transpose(1, 2) + query_position * 0.1
        for layer in self.memory_attention.layers:
            query = layer(query, memory, pos=position, query_pos=query_position, num_k_exclude_rope=pointers.shape[0])
        return self.memory_attention.norm(query).transpose(1, 2).reshape_as(features[-1])


def build_from_config(config, device, dtype):
    model = EdgeTamVideoModel(config).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        # Preserve native SDPA rounding in decoder, memory and Perceiver attention.
        if hasattr(module, "attend") and isinstance(module.attend, SamAttentionCore):
            module.attend = DenseAttention(backend="sdpa")
        if isinstance(module, sam2_video.MemoryAttention):
            module.rotary_table = module.position_table(config, device)
        if isinstance(module, CrossAttention):
            key_config = copy.copy(config)
            key_config.memory_attention_rope_feat_sizes = config.memory_attention_rope_k_sizes
            module.key_rotary_table = module.position_table(key_config, device)
    return model


def load_state_dict_into(model, state_dict, config):
    perceiver = {k.removeprefix("spatial_perceiver."): v for k, v in state_dict.items() if k.startswith("spatial_perceiver.")}
    model.spatial_perceiver.load_state_dict(perceiver, strict=True)
    rest = {k.replace(".mlp.up_proj.", ".linear1.").replace(".mlp.down_proj.", ".linear2."): v
            for k, v in state_dict.items() if not k.startswith("spatial_perceiver.")}
    perceiver_module = model.spatial_perceiver
    del model.spatial_perceiver
    try:
        sam2_video.load_state_dict_into(model, rest, config, image_module=edgetam)
    finally:
        model.spatial_perceiver = perceiver_module


make_workloads = sam2_video.make_workloads
