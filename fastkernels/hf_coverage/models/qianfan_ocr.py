"""Qianfan OCR's scaled vision blocks, pixel packing and uncached Qwen3."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from . import llama, qwen3
from .deepseek_vl import DeepseekBackbone
from .dinov2 import _scaled_block
from .lfm2_vl import Projector
from .olmo2 import decoder_config
from .vit import _Embeddings
from ..runner import Workload


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = _Embeddings(config)
        local = SimpleNamespace(**(dict(config) | {"attention_probs_dropout_prob": config.attention_dropout}))
        self.layers = nn.ModuleList(
            _scaled_block(local, config.intermediate_size, config.layer_scale_init_value,
                          qkv_bias=config.attention_bias)
            for _ in range(config.num_hidden_layers)
        )

    def forward(self, pixels):
        hidden = self.embeddings(pixels)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class Backbone(DeepseekBackbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text, self.vision = text, Vision(config.vision_config)
        self.projector = Projector(SimpleNamespace(
            downsample_factor=int(1 / config.downsample_ratio),
            vision_config=config.vision_config, text_config=config.text_config,
            projector_use_layernorm=True, projector_bias=True,
            projector_hidden_size=config.text_config.hidden_size,
        ))
        self.image_token_id = config.image_token_id
        self.pixel_values = self.image_hidden_states = None

    def features(self, pixels):
        hidden = self.vision(pixels)[:, 1:]
        batch, length, width = hidden.shape
        side = int(length ** .5)
        return self.projector(hidden.reshape(batch, side, side, width)).reshape(batch, -1, self.text.norm.weight.numel())


class Model(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = Backbone(language.model, config)


def build_from_config(config, device, dtype):
    vision, text = config.vision_config, config.text_config
    if (vision.norm_type != "layer_norm" or vision.use_qk_norm or not vision.use_mean_pooling
            or vision.use_mask_token or not vision.use_absolute_position_embeddings
            or vision.hidden_act != "gelu" or config.projector_hidden_act != "gelu"
            or config.vision_feature_layer != -1 or config.vision_feature_select_strategy != "default"
            or text.use_cache or text.rope_parameters["rope_type"] != "default"):
        raise ValueError("Preserve the documented Qianfan vision settings and disabled language cache")
    language = LlamaForCausalLM(decoder_config(text, dtype))
    for layer in language.model.layers:
        layer.self_attn.q_norm = RMSNorm(text.head_dim, text.rms_norm_eps)
        layer.self_attn.k_norm = RMSNorm(text.head_dim, text.rms_norm_eps)
    return Model(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    qwen3.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, model.config)
    vision = {}
    for name in model.model.vision.embeddings.state_dict():
        source = name.replace("patch_embeddings.proj.", "patch_embeddings.projection.")
        vision["embeddings." + name] = remaining.pop("model.vision_tower.embeddings." + source)
    for index, layer in enumerate(model.model.vision.layers):
        prefix, destination = f"model.vision_tower.layers.{index}.", f"layers.{index}."
        for field in ("weight", "bias"):
            if field == "bias" and layer.attn.qkv.bias is None:
                continue
            vision[destination + "attn.qkv." + field] = torch.cat([
                remaining.pop(prefix + f"attention.{name}_proj.{field}") for name in ("q", "k", "v")
            ])
        for target, source in (("attn.proj", "attention.projection_layer"), ("norm1", "layernorm_before"),
                               ("norm2", "layernorm_after"), ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            for field in ("weight", "bias"):
                vision[destination + target + "." + field] = remaining.pop(prefix + source + "." + field)
        for number in (1, 2):
            vision[destination + f"gamma_{number}"] = remaining.pop(prefix + f"lambda_{number}")
    model.model.vision.load_state_dict(vision, strict=True)
    model.model.projector.load_state_dict({
        name: remaining.pop("model.multi_modal_projector." + name)
        for name in model.model.projector.state_dict()
    }, strict=True)
    if remaining:
        raise KeyError(f"Unmapped Qianfan OCR weights: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    model.model.pixel_values = inputs["pixel_values"]
    workload = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config,
                                    cached_decode=False)["forward"]

    def run():
        return {**workload.run(), "image_hidden_states": model.model.image_hidden_states}

    return {"forward": Workload(run=run, prepare=workload.prepare)}
