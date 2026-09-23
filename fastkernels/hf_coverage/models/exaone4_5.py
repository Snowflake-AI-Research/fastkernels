"""EXAONE-4.5 image/video generation with its EXAONE-4 text carrier."""

from dataclasses import fields

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding
from fastkernels.tasks.baseline.L4.qwen2_5_omni import Qwen2_5OmniVisionConfig, Qwen2_5VisionTransformer
from ..runner import Workload
from . import exaone4, llama


class NativeRotary(nn.Module):
    """Select the existing explicit-rounding rotary callable, without fusion."""
    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary

    def forward(self, positions, query, key):
        return self.rotary.forward_native(positions, query, key, self.rotary.head_dim,
                                          self.rotary.cos_sin_cache.to(query.dtype))


class NativeGate(SiluAndMul):
    forward = staticmethod(SiluAndMul.forward_native)


class SplitProjection(nn.Module):
    """Use existing Matmul per native projection, retaining packed state."""
    def __init__(self, projection, sizes):
        super().__init__()
        self.weight, self.sizes, self.matmul = projection.weight, sizes, Matmul()

    def forward(self, x):
        return torch.cat([self.matmul(x, weight) for weight in self.weight.split(self.sizes)], -1)


class CachedDenseAttention(nn.Module):
    """DenseAttention with the caller's bounded KV buffers and native masks."""
    def __init__(self, attention, window):
        super().__init__()
        self.num_heads, self.num_kv_heads = attention.num_heads, attention.num_kv_heads
        self.head_size, self.scale = attention.head_size, attention.scale
        self.kv_layout, self._block_size, self.window = "NHD", 16, window
        self.k_cache = self.v_cache = None
        # Isolated native replay selects cuDNN SDPA. Importing the text carrier
        # disables global cuDNN SDPA availability; select the unchanged
        # explicit backend to retain the native reduction and BF16 rounding.
        self.attend = DenseAttention(backend="cudnn")

    def forward(self, q, k, v):
        context = get_context()
        n = q.shape[0]
        length = context.max_seqlen_q if context.is_prefill else context.max_context_len
        slots = context.slot_mapping
        for cache, value in ((self.k_cache, k), (self.v_cache, v)):
            cache.flatten(0, 1)[slots] = value.reshape(n, self.num_kv_heads, self.head_size)
        start = max(0, length - self.window) if self.window and not context.is_prefill else 0
        keys, values = (cache.flatten(0, 1)[start:length][None] for cache in (self.k_cache, self.v_cache))
        groups = self.num_heads // self.num_kv_heads
        keys, values = (t[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(1, length-start, self.num_heads, self.head_size)
                        for t in (keys, values))
        query_positions = torch.arange(length-n, length, device=q.device)
        key_positions = torch.arange(start, length, device=q.device)
        mask = key_positions[None] <= query_positions[:, None]
        if self.window:
            mask = mask & (key_positions[None] > query_positions[:, None] - self.window)
        # The native full-attention SDPA path delegates causality to SDPA;
        # explicit sliding-window masks remain required for local layers.
        return self.attend(q.reshape(1, n, self.num_heads, self.head_size), keys, values,
                           causal=bool(not self.window and context.is_prefill and n > 1),
                           attn_mask=mask[None, None] if self.window else None).reshape(n, -1)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_heads, config.num_key_value_heads
        self.dim = config.hidden_size // self.heads
        self.qkv = Linear(config.hidden_size, (self.heads + 2 * self.kv_heads) * self.dim)
        self.proj = Linear(config.hidden_size, config.hidden_size)
        self.attend = DenseAttention(backend="sdpa")

    def forward(self, x, cu_seqlens, cos, sin, max_seqlen=None):
        n = x.shape[0]
        q, k, v = self.qkv(x).reshape(n, -1).split(
            [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim], -1)
        # Native vision rotation promotes activations and fixed position tables.
        positions = torch.arange(n, device=x.device)
        q, k = RotaryEmbedding.forward_native(positions, q.float(), k.float(), self.dim,
                                               torch.cat((cos, sin), -1))
        q, k, v = (t.to(x.dtype).reshape(1, n, heads, self.dim)
                   for t, heads in ((q, self.heads), (k, self.kv_heads), (v, self.kv_heads)))
        groups = self.heads // self.kv_heads
        k, v = (t[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(1, n, self.heads, self.dim)
                for t in (k, v))
        boundaries = cu_seqlens.tolist()
        parts = [self.attend(q[:, a:b], k[:, a:b], v[:, a:b])
                 for a, b in zip(boundaries[:-1], boundaries[1:])]
        return self.proj(torch.cat(parts, 1).reshape_as(x).contiguous())


class Vision(Qwen2_5VisionTransformer):
    def __init__(self, config):
        adapted = Qwen2_5OmniVisionConfig(**{
            f.name: getattr(config, f.name) for f in fields(Qwen2_5OmniVisionConfig)
            if hasattr(config, f.name)})
        super().__init__(adapted)
        for block in self.blocks:
            block.attn = VisionAttention(config)

    def forward(self, x, grid_thw):
        # The parent's learned blocks/merger are unchanged. This wiring keeps
        # FP32 position metadata instead of its BF16-cast flash-attention tables.
        convolution = self.patch_embed.proj
        hidden = convolution(x.view(-1, convolution.conv.in_channels,
                                     *convolution.conv.kernel_size)).view(-1, self.patch_embed.embed_dim)
        device = hidden.device
        grids = grid_thw.tolist()
        indexes, windows, full, offset, window_offset = [], [0], [0], 0, 0
        for t, h, w in grids:
            index, ends = self.get_window_index_thw(t, h, w)
            indexes.append(index + offset)
            offset += t * h * w // self.spatial_merge_unit
            windows.extend((ends + window_offset).tolist())
            window_offset = windows[-1]
            for _ in range(t):
                full.append(full[-1] + h * w)
        index = torch.cat(indexes)
        reverse = self.invert_permutation(index).to(device)
        index = index.to(device)
        # Recreate fixed FP32 tables after module dtype conversion; no activation
        # work is cached or moved outside the measured forward.
        rotary = VisionRotaryEmbedding(self.blocks[0].attn.dim // 2,
                                       max_grid_size=max(max(g[1:]) for g in grids)).to(device)
        cos, sin = rotary(grids, self.spatial_merge_size, torch.float32, device)
        cos, sin = (t.reshape(-1, self.spatial_merge_unit, t.shape[-1])[index].flatten(0, 1)
                    for t in (cos, sin))
        hidden = hidden.reshape(-1, self.spatial_merge_unit, hidden.shape[-1])[index].flatten(0, 1).unsqueeze(1)
        windows, full = (torch.tensor(v, device=device, dtype=torch.int32) for v in (windows, full))
        for i, block in enumerate(self.blocks):
            hidden = hidden + block.attn(block.norm1(hidden), full if i in self.fullatt_block_indexes else windows, cos, sin)
            # The existing gate's explicit PyTorch path rounds SiLU before
            # multiplication like HF; its eager CUDA entry cannot accept CPU.
            mlp = block.mlp
            hidden = hidden + mlp.down_proj(mlp.act.forward_native(mlp.gate_up_proj(block.norm2(hidden))))
        return self.merger(hidden)[reverse]


class MultimodalBackbone(nn.Module):
    def __init__(self, text, vision, config):
        super().__init__()
        self.text, self.vision, self.config = text, vision, config
        self.inputs = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        hidden = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            for token, pixels, grid in ((self.config.image_token_id, "pixel_values", "image_grid_thw"),
                                         (self.config.video_token_id, "pixel_values_videos", "video_grid_thw")):
                if pixels in self.inputs:
                    hidden[input_ids == token] = self.vision(self.inputs[pixels], self.inputs[grid])
        for layer in self.text.layers:
            hidden, _ = layer(positions, hidden)
        return self.text.norm(hidden)


def build_from_config(config, device, dtype):
    model = exaone4.build_from_config(config.text_config, device, dtype)
    for layer, kind in zip(model.model.layers, config.text_config.layer_types):
        layer.self_attn.rotary_emb = NativeRotary(layer.self_attn.rotary_emb)
        layer.mlp.act_fn = NativeGate()
        attention = layer.self_attn
        attention.qkv_proj = SplitProjection(attention.qkv_proj,
                                             [attention.num_heads * attention.head_dim,
                                              attention.num_kv_heads * attention.head_dim,
                                              attention.num_kv_heads * attention.head_dim])
        layer.mlp.gate_up_proj = SplitProjection(layer.mlp.gate_up_proj,
                                                [config.text_config.intermediate_size] * 2)
        attention.attn = CachedDenseAttention(attention.attn,
                                              config.text_config.sliding_window if kind == "sliding_attention" else None)
    model.model = MultimodalBackbone(model.model, Vision(config.vision_config), config)
    # Native EXAONE rounds normalized activations before the learned scale.
    # Use the existing explicit-rounding norm, including vision residual sums.
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, RMSNorm):
                setattr(parent, name, RMSNormNative(child.hidden_size, child.eps))
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    carrier = nn.Module()
    carrier.model, carrier.lm_head = model.model.text, model.lm_head
    exaone4.load_state_dict_into(carrier, text, config.text_config)
    mapped = {}
    for name, target in model.model.vision.state_dict().items():
        source = name.replace("patch_embed.proj.conv.", "patch_embed.proj.")
        if source.startswith("merger."):
            source = source.replace(".norm.", ".ln_q.").replace(".fc1.", ".mlp.0.").replace(".fc2.", ".mlp.2.")
        if ".mlp.gate_up_proj." in source:
            value = torch.cat([remaining.pop("model.visual." + source.replace("gate_up_proj", part))
                               for part in ("gate_proj", "up_proj")])
        else:
            value = remaining.pop("model.visual." + source)
        if value.shape != target.shape:
            raise ValueError(f"EXAONE vision state shape mismatch: {source}")
        mapped[name] = value
    model.model.vision.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f"Unmapped EXAONE-4.5 state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("The EXAONE4.5 development case uses one multimodal sequence")
    model.model.inputs = inputs
    workloads = llama.make_workloads(model, inputs, model.config)
    for phase, workload in list(workloads.items()):
        def run(phase=phase, workload=workload):
            output = workload.run()
            length = inputs["input_ids"].shape[1] - (phase == "prefill")
            for i, layer in enumerate(model.model.layers):
                attention = layer.self_attn.attn
                local = config.text_config.layer_types[i] == "sliding_attention"
                start = max(0, length - config.text_config.sliding_window + 1) if local else 0
                for name, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
                    if attention.kv_layout == "HND":
                        cache = cache.transpose(1, 2)
                    output[f"past_key_values.{i}.{name}"] = cache.flatten(0, 1)[start:length].transpose(0, 1).unsqueeze(0)
            return output
        workloads[phase] = Workload(run=run, prepare=workload.prepare)
    return workloads
