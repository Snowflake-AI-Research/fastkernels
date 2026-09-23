"""InternVL's layer-scaled vision, pixel shuffle and dynamic-RoPE Qwen2."""

from types import SimpleNamespace
from dataclasses import replace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L2.eva_attention import EvaAttention
from fastkernels.tasks.baseline.L3.eva_block import EvaBlock
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, qwen2
from .qwen2_precision import configure_language
from ..patches.internvl_dynamic_rope import InternVLDynamicRotaryEmbedding
from ..runner import Workload


class InternAttention(EvaAttention):
    """Select cuDNN explicitly despite the language carrier's vLLM global toggle."""

    def __init__(self, width, heads, bias):
        super().__init__(width, heads, qkv_bias=bias, qkv_fused=False)
        self.attention = DenseAttention(backend="cudnn")

    def forward(self, hidden, rope=None, attn_mask=None):
        shape = (*hidden.shape[:2], self.num_heads, self.head_dim)
        query, key, value = (projection(hidden).reshape(shape)
                             for projection in (self.q_proj, self.k_proj, self.v_proj))
        attended = self.attention(query, key, value, attn_mask=attn_mask)
        return self.proj(attended.reshape_as(hidden))


class InternVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.projection = Conv2d(config.num_channels, width, config.patch_size, stride=config.patch_size)
        count = (config.image_size[0] // config.patch_size[0]) * (config.image_size[1] // config.patch_size[1])
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.position_embeddings = nn.Parameter(torch.empty(1, count + 1, width))
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = EvaBlock(width, config.num_attention_heads, config.intermediate_size / width,
                             qkv_bias=config.attention_bias, qkv_fused=False,
                             init_values=config.layer_scale_init_value)
            layer.norm1 = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
            layer.norm2 = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
            layer.attn = InternAttention(width, config.num_attention_heads, config.attention_bias)
            layer.mlp = VitEncoderMlp(width, config.intermediate_size, width, act_approximate="none")
            self.layers.append(layer)

    def forward(self, pixels):
        patches = self.projection(pixels).flatten(2).transpose(1, 2)
        hidden = torch.cat((self.cls_token.expand(pixels.shape[0], -1, -1), patches), dim=1)
        hidden = hidden + self.position_embeddings
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class InternBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text, self.vision = text, InternVision(config.vision_config)
        width = config.vision_config.hidden_size * 4
        self.projector = nn.Sequential(LayerNorm(width, eps=1e-5, promote_fp32=False),
                                       Linear(width, config.text_config.hidden_size), GELU(),
                                       Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.image_token_id = config.image_token_id
        self.pixels = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, pixels):
        hidden = self.vision(pixels)[:, 1:]
        batch, length, channels = hidden.shape
        side = int(length ** 0.5)
        hidden = hidden.reshape(batch, side, side // 2, channels * 2).permute(0, 2, 1, 3).contiguous()
        hidden = hidden.reshape(batch, side // 2, side // 2, channels * 4).permute(0, 2, 1, 3).contiguous()
        return self.projector(hidden.reshape(batch, -1, channels * 4))

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.features(self.pixels)
            embeddings = embeddings.masked_scatter((input_ids == self.image_token_id)[:, None].expand_as(embeddings),
                                                   self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class InternVLModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config, self.lm_head = text.config, text.lm_head
        self.model = InternBackbone(text.model, config)


def build_from_config(config, device, dtype):
    vision, text = config.vision_config, config.text_config
    if (vision.norm_type != "layer_norm" or vision.use_qk_norm or not vision.use_mean_pooling
            or vision.use_mask_token or not vision.use_absolute_position_embeddings
            or vision.hidden_act != "gelu" or config.downsample_ratio != 0.5
            or config.vision_feature_layer != -1 or config.vision_feature_select_strategy != "default"):
        raise ValueError("Preserve the documented InternVL3-1B visual configuration")
    rope = text.rope_parameters
    if rope["rope_type"] != "dynamic" or text.tie_word_embeddings or text.use_sliding_window:
        raise ValueError("The selected InternVL uses untied full-attention Qwen2 with dynamic RoPE")
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "vocab_size", "max_position_embeddings", "rms_norm_eps")
    native = LlamaConfig(**{name: getattr(text, name) for name in fields},
                         head_dim=text.hidden_size // text.num_attention_heads,
                         rope_theta=rope["rope_theta"], rope_scaling_factor=1.0, dtype=dtype, qkv_bias=True)
    language = LlamaForCausalLM(native)
    configure_language(language.model, native)
    language.model.rotary_emb = InternVLDynamicRotaryEmbedding(native.head_dim, text.max_position_embeddings,
                                                              rope["rope_theta"], rope["factor"])
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = language.model.rotary_emb
    return InternVLModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    mapped = {}
    for name in model.model.vision.state_dict():
        source = name
        if source.startswith("layers."):
            source = source.replace("layers.", "encoder.layer.", 1)
            for target, origin in ((".attn.proj.", ".attention.projection_layer."), (".attn.", ".attention."),
                                   (".norm1.", ".layernorm_before."), (".norm2.", ".layernorm_after."),
                                   (".gamma_1", ".lambda_1"), (".gamma_2", ".lambda_2")):
                source = source.replace(target, origin)
        else:
            source = "embeddings." + source.replace("projection.", "patch_embeddings.projection.")
        mapped[name] = remaining.pop("model.vision_tower." + source)
    model.model.vision.load_state_dict(mapped, strict=True)
    names = {"0": "layer_norm", "1": "linear_1", "3": "linear_2"}
    model.model.projector.load_state_dict({name: remaining.pop("model.multi_modal_projector." + names[name[0]] + name[1:])
                                          for name in model.model.projector.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped InternVL state: {sorted(remaining)}")


def make_workloads(model, inputs, config, *, case=None):
    model.model.pixels = inputs["pixel_values"]
    # The generic Llama runner assumes a fixed RoPE table. InternVL's table
    # grows dynamically, so size its workload buffers for the requested length.
    workload_config = replace(model.config, max_position_embeddings=max(
        model.config.max_position_embeddings, inputs["input_ids"].shape[1]))
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, workload_config, case=case)
    prefill = workloads["prefill"]
    workloads["prefill"] = Workload(run=lambda: {**prefill.run(), "image_hidden_states": model.model.image_hidden_states},
                                    prepare=prefill.prepare, collect=prefill.collect)
    return workloads
