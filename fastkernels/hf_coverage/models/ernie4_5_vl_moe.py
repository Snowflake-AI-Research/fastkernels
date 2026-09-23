"""ERNIE 4.5 VL: modality-isolated experts and spatial/temporal resampling."""

from copy import copy

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.mrope import MRotaryEmbedding
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding
from fastkernels.tasks.baseline.L3.vision_block import VisionBlock
from .dots1 import NormalizedExperts
from .qwen2_5_omni import TextAttention, TextMLP
from .qwen2_vl import multimodal_positions
from ..patches.grouped_topk_normalization import GroupedTopKNormalization
from ..runner import Workload


class Rotary(nn.Module):
    """Pair-coordinate and frequency layouts around unchanged FP32 M-RoPE."""

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        sections = config.rope_parameters.get("mrope_section", [22, 22, 20])
        if sections[0] != sections[1]:
            raise ValueError("Native ERNIE interleaves equal height/width sections")
        self.rotary = MRotaryEmbedding(dim, config.max_position_embeddings,
                                      config.rope_parameters["rope_theta"], sections)
        hw = sum(sections[:2])
        order = torch.tensor(list(range(0, hw, 2)) + list(range(1, hw, 2)) + list(range(hw, dim // 2)))
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("inverse", torch.argsort(order), persistent=False)
        # Native initialization rearranges frequencies before section selection.
        cos, sin = self.rotary.cos_sin_cache.chunk(2, -1)
        self.rotary.cos_sin_cache = torch.cat((cos[:, order], sin[:, order]), -1)

    def forward_native_2d(self, positions, query, key):
        dtype = query.dtype
        def pack(x):
            pairs = x.float().reshape(*x.shape[:-1], -1, 2)[..., self.order, :]
            return torch.cat((pairs[..., 0], pairs[..., 1]), -1)
        def unpack(x):
            first, second = x.chunk(2, -1)
            return torch.stack((first, second), -1)[..., self.inverse, :].flatten(-2).to(dtype)
        q, k = self.rotary.forward_native_2d(positions[[1, 2, 0]], pack(query), pack(key))
        return unpack(q), unpack(k)


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        for name, width in zip(("text_moe", "vision_moe"), config.moe_intermediate_size):
            setattr(self, name, NormalizedExperts(
                hidden_size=config.hidden_size, num_experts=config.moe_num_experts,
                top_k=config.moe_k, moe_intermediate_size=width,
                routing="softmax", keep_router_weights_fp32=False,
                normalizer=GroupedTopKNormalization(scoring_func="softmax", floor=config.moe_norm_min),
            ))
        shared = copy(config)
        shared.intermediate_size = config.moe_intermediate_size[0] * config.moe_num_shared_experts
        self.shared_experts = TextMLP(shared) if shared.intermediate_size else None

    def forward(self, hidden, modality):
        flat = hidden.reshape(-1, hidden.shape[-1])
        mask = modality.reshape(-1).bool()
        result = torch.empty_like(flat)
        for selector, expert in ((~mask, self.text_moe), (mask, self.vision_moe)):
            selected = flat[selector]
            if selected.shape[0]:
                result[selector] = expert(selected)
        result = result.reshape_as(hidden)
        if self.shared_experts is not None:
            result = result + self.shared_experts(hidden)
        return result


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.self_attn = TextAttention(config)
        for name in ("q_proj", "k_proj", "v_proj"):
            previous = getattr(self.self_attn, name)
            setattr(self.self_attn, name, Linear(config.hidden_size, previous.weight.shape[0], bias=False))
        self.mlp = Experts(config) if kind == "sparse" else TextMLP(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions, rotary, modality):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, rotary)
        normalized = self.post_attention_layernorm(hidden)
        return hidden + (self.mlp(normalized, modality) if isinstance(self.mlp, Experts) else self.mlp(normalized))


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Module()
        self.patch_embed.proj = Linear(config.in_channels * config.patch_size ** 2, config.hidden_size, bias=False)
        self.blocks = nn.ModuleList([VisionBlock(config.hidden_size, config.num_heads, config.intermediate_size,
                                                norm_eps=config.rms_norm_eps) for _ in range(config.depth)])
        self.ln = LayerNorm(config.hidden_size, eps=config.rms_norm_eps, promote_fp32=False)

    def forward(self, pixels, grids):
        values = grids.tolist()
        hidden = self.patch_embed.proj(pixels)[:, None]
        cos, sin = self.rotary(values, self.config.spatial_merge_size, torch.float32, hidden.device)
        lengths = [h * w for t, h, w in values for _ in range(t)]
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        cu = torch.tensor(cumulative, dtype=torch.int32, device=hidden.device)
        for block in self.blocks:
            hidden = block(hidden, cu, cos, sin, max(lengths))
        return self.ln(hidden[:, 0])


class Projection(nn.Module):
    def __init__(self, incoming, outgoing, eps):
        super().__init__()
        self.fc1, self.fc2 = Linear(incoming, outgoing), Linear(outgoing, outgoing)
        self.ln, self.activation = LayerNorm(outgoing, eps=eps, promote_fp32=False), GELU()

    def forward(self, hidden):
        return self.ln(self.fc2(self.activation(self.fc1(hidden))))


class Resampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision, text = config.vision_config, config.text_config
        self.merge = vision.spatial_merge_size
        self.width = vision.hidden_size * self.merge ** 2
        self.spatial_linear = Projection(self.width, self.width, vision.rms_norm_eps)
        self.temporal_linear = Projection(self.width * 2, self.width, vision.rms_norm_eps)
        self.mlp = Linear(self.width, text.hidden_size)
        self.after_norm = RMSNormNative(text.hidden_size, text.rms_norm_eps)

    def forward(self, hidden, grids):
        hidden = self.spatial_linear(hidden.reshape(-1, self.width))
        first, second, offset = [], [], 0
        for t, h, w in grids.tolist():
            count = h * w // self.merge ** 2
            if t != 1 and t % 2:
                raise ValueError("Native temporal resampler requires one image frame or even video frames")
            for even, odd in zip(range(0, t, 2), range(0 if t == 1 else 1, t, 2)):
                first.append(hidden[offset + even * count:offset + (even + 1) * count])
                second.append(hidden[offset + odd * count:offset + (odd + 1) * count])
            offset += t * count
        hidden = self.temporal_linear(torch.cat((torch.cat(first), torch.cat(second)), -1))
        return self.after_norm(self.mlp(hidden))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        text = config.text_config
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        language = self.model.language_model
        language.embed_tokens = Embedding(text.vocab_size, text.hidden_size)
        language.layers = nn.ModuleList([Layer(text, kind) for kind in text.mlp_layer_types])
        language.norm = RMSNormNative(text.hidden_size, text.rms_norm_eps)
        self.model.vision_tower = Vision(config.vision_config)
        self.model.resampler_model = Resampler(config)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        self.lm_head.weight = language.embed_tokens.emb.weight

    def reset(self):
        for layer in self.model.language_model.layers:
            layer.self_attn.key = layer.self_attn.value = None

    def forward(self, ids, positions, modality, inputs=None):
        language = self.model.language_model
        hidden = language.embed_tokens(ids)
        if inputs is not None:
            for token, pixels, grid in ((self.config.image_token_id, "pixel_values", "image_grid_thw"),
                                       (self.config.video_token_id, "pixel_values_videos", "video_grid_thw")):
                if pixels in inputs:
                    features = self.model.resampler_model(self.model.vision_tower(inputs[pixels], inputs[grid]), inputs[grid])
                    hidden[ids == token] = features
        for layer in language.layers:
            hidden = layer(hidden, positions, language.rotary, modality)
        result = {"logits": self.lm_head(language.norm(hidden))}
        for index, layer in enumerate(language.layers):
            for name in ("key", "value"):
                result[f"past_key_values.{index}.{name}"] = getattr(layer.self_attn, name).transpose(1, 2)
        return result


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if text.use_bias or text.hidden_act != "silu" or not config.tie_word_embeddings:
        raise ValueError("Selected ERNIE uses bias-free SiLU and tied text embeddings")
    if vision.hidden_act != "quick_gelu" or vision.temporal_merge_size != 2:
        raise ValueError("Selected ERNIE uses QuickGELU vision and temporal merge two")
    model = Model(config).to(device=device, dtype=dtype).eval()
    model.model.language_model.rotary = Rotary(text).to(device)
    model.model.vision_tower.rotary = VisionRotaryEmbedding(vision.hidden_size // vision.num_heads // 2).to(device)
    for layer in model.model.language_model.layers:
        if isinstance(layer.mlp, Experts):
            layer.mlp.text_moe.gate.float()
            layer.mlp.vision_moe.gate.float()
    return model


@torch.no_grad()
def load_state_dict_into(model, state, config):
    targets = {}
    for name, value in model.state_dict().items():
        source = name.replace("embed_tokens.emb.", "embed_tokens.")
        for modality in ("text_moe", "vision_moe"):
            source = source.replace(f".{modality}.w13", f".{modality}.experts.gate_up_proj")
            source = source.replace(f".{modality}.w2", f".{modality}.experts.down_proj")
            source = source.replace(f".{modality}.gate.e_score_correction_bias", f".{modality}.gate.moe_statics.e_score_correction_bias")
        targets[source] = value
    if targets.keys() != state.keys():
        raise KeyError(f"ERNIE VL state mismatch: {sorted(targets.keys() ^ state.keys())}")
    if not torch.equal(state["lm_head.weight"], state["model.language_model.embed_tokens.weight"]):
        raise ValueError("Tied ERNIE embedding and head disagree")
    for name, target in targets.items():
        source = state[name]
        if name.endswith("e_score_correction_bias"):
            source = source.reshape(-1)
        if source.shape != target.shape:
            raise ValueError(f"ERNIE VL shape mismatch: {name}: {source.shape} versus {target.shape}")
        target.copy_(source)
    for layer in model.model.language_model.layers:
        if isinstance(layer.mlp, Experts):
            for expert in (layer.mlp.text_moe, layer.mlp.vision_moe):
                expert.process_weights_after_loading()


def make_workloads(model, inputs, config, case=None):
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("The selected development case has one sequence")
    metadata = dict(inputs)
    metadata["video_grid_thw"] = inputs["video_grid_thw"].clone()
    metadata["video_grid_thw"][:, 0] //= config.vision_config.temporal_merge_size
    positions, delta = multimodal_positions(ids[0], inputs["mm_token_type_ids"][0], metadata, config)
    modality = inputs["moe_mm_token_type_ids"]
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    steps = 2 if continuation else 1
    prefix_length = ids.shape[1] - steps
    if prefix_length < 1:
        raise ValueError("ERNIE VL continuation requires a nonempty prefix")
    def prefill():
        return model(ids[:, :prefix_length], positions[:, :prefix_length], modality[:, :prefix_length], inputs)
    calls = []
    for step in range(steps):
        index = prefix_length + step
        def decode(index=index):
            return model(ids[:, index:index + 1], positions[:, index:index + 1], modality[:, index:index + 1])
        calls.append(decode)
    workloads = {"prefill": Workload(run=prefill, prepare=model.reset)}
    for step, decode in enumerate(calls):
        def prepare_decode(step=step):
            model.reset()
            prefill()
            for prior in calls[:step]:
                prior()
        name = f"decode_{step + 1}" if continuation else "decode"
        workloads[name] = Workload(run=decode, prepare=prepare_decode)
    return workloads
