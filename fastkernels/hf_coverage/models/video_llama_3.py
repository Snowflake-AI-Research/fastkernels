"""VideoLLaMA3's variable-grid vision, spatial merge and tied Qwen2 decoder."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding
from fastkernels.tasks.baseline.L3.siglip_encoder_layer import SigLIPEncoderLayer
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, qwen2
from .siglip import configure_encoder
from ..patches.ernie4_5_rope import FP32RotaryEmbedding
from ..runner import Workload


class GridAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.attention = DenseAttention(backend="sdpa")
        self.rotary = FP32RotaryEmbedding(self.head_dim, 1, 10000.0)
        self.lengths = None

    def forward(self, hidden):
        query = self.q_proj(hidden).view(-1, self.heads, self.head_dim)
        key = self.k_proj(hidden).view_as(query)
        value = self.v_proj(hidden).view_as(query)
        positions = torch.arange(hidden.shape[0], device=hidden.device)
        query, key = self.rotary(positions, query, key)
        outputs = [self.attention(q[None], k[None], v[None])[0]
                   for q, k, v in zip(query.split(self.lengths), key.split(self.lengths), value.split(self.lengths))]
        return self.out_proj(torch.cat(outputs).reshape_as(hidden))


class GridVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.patch_embedding = Conv2d(config.num_channels, config.hidden_size, self.patch_size, stride=self.patch_size)
        self.positions = VisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        self.layers = nn.ModuleList([SigLIPEncoderLayer(config.hidden_size, config.num_attention_heads,
                                                     config.intermediate_size, config.layer_norm_eps)
                                     for _ in range(config.num_hidden_layers)])
        configure_encoder(self.layers)
        for layer in self.layers:
            layer.self_attn = GridAttention(config)
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.resize = Interpolate()

    def forward(self, pixels, grids, merges):
        angles = [self.positions([grid], merge, torch.float32, pixels.device) for grid, merge in zip(grids, merges)]
        cache = torch.cat((torch.cat([v[0] for v in angles]), torch.cat([v[1] for v in angles])), dim=-1)
        lengths = [height * width for frames, height, width in grids for _ in range(frames)]
        hidden = self.patch_embedding(pixels.reshape(-1, 3, self.patch_size, self.patch_size)).flatten(1)
        for layer in self.layers:
            layer.self_attn.rotary.cos_sin_cache = cache
            layer.self_attn.lengths = lengths
            hidden = layer(hidden)
        hidden = self.post_layernorm(hidden)
        counts = [frames * height * width for frames, height, width in grids]
        merged = []
        for chunk, (frames, height, width), merge in zip(hidden.split(counts), grids, merges):
            channels = chunk.shape[-1]
            spatial = chunk.view(frames, height // merge, width // merge, merge, merge, channels)
            spatial = spatial.permute(0, 1, 3, 2, 4, 5).reshape(frames, height, width, channels).permute(0, 3, 1, 2)
            spatial = self.resize(spatial, size=(height // merge, width // merge), mode="bilinear", align_corners=False)
            merged.append(spatial.permute(0, 2, 3, 1).reshape(-1, channels))
        return torch.cat(merged)


class VideoLlama3Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision = GridVision(config.vision_config)
        self.projector = nn.Sequential(Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
                                       GELU(), Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.image_token_id, self.video_token_id = config.image_token_id, config.video_token_id
        self.inputs = None
        self.image_hidden_states = self.video_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            for kind, key, token in (("image", "pixel_values", self.image_token_id),
                                     ("video", "pixel_values_videos", self.video_token_id)):
                grids = self.inputs[kind + "_grid_thw"].tolist()
                merges = self.inputs[kind + "_merge_sizes"].tolist()
                features = self.projector(self.vision(self.inputs[key], grids, merges))
                if kind == "video" and "video_compression_mask" in self.inputs:
                    features = features[self.inputs["video_compression_mask"]]
                setattr(self, kind + "_hidden_states", features)
                embeddings = embeddings.masked_scatter((input_ids == token)[:, None].expand_as(embeddings), features)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class VideoLlama3Model(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = VideoLlama3Backbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    text, rope = config.text_config, config.text_config.rope_parameters
    if (text.model_type != "qwen2" or not text.tie_word_embeddings or text.use_sliding_window
            or any(kind != "full_attention" for kind in text.layer_types) or rope["rope_type"] != "default"
            or text.hidden_act != "silu" or config.vision_config.hidden_act != "gelu_pytorch_tanh"):
        raise ValueError("The documented VideoLLaMA3 checkpoint uses tied Qwen2 full attention and tanh GELU vision")
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "vocab_size", "max_position_embeddings", "rms_norm_eps")
    native = LlamaConfig(**{name: getattr(text, name) for name in fields},
                         head_dim=text.hidden_size // text.num_attention_heads,
                         rope_theta=rope["rope_theta"], rope_scaling_factor=1.0, dtype=dtype, qkv_bias=True)
    language = LlamaForCausalLM(native)
    language.lm_head.embedding_op.emb.weight = language.model.embed_tokens.embedding_op.emb.weight
    model = VideoLlama3Model(language, config)
    angles = model.model.vision.positions.cos_sin_cache
    model.to(device=device, dtype=dtype)
    model.model.vision.positions.cos_sin_cache = angles.to(device=device)
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    if not torch.equal(text["lm_head.weight"], text["model.embed_tokens.weight"]):
        raise ValueError("VideoLLaMA3 tied embedding weights differ")
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name.replace("patch_embedding.", "embeddings.patch_embedding.").replace("layers.", "encoder.layers.")
        mapped[name] = remaining.pop("model.vision_model." + source)
    model.model.vision.load_state_dict(mapped, strict=True)
    model.model.projector.load_state_dict({name: remaining.pop("model.projector.readout." + name)
                                          for name in model.model.projector.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped VideoLLaMA3 state: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    model.model.inputs = inputs
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config)
    prefill = workloads["prefill"]

    def run_prefill():
        return {**prefill.run(), "image_hidden_states": model.model.image_hidden_states,
                "video_hidden_states": model.model.video_hidden_states}

    workloads["prefill"] = Workload(run=run_prefill, prepare=prefill.prepare)
    return workloads
