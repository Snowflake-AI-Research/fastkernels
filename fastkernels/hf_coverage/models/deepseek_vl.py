"""DeepSeek-VL's complete SigLIP, alignment MLP and Llama inference path."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionEncoder
from . import llama, llava


def make_vision(config):
    vision = SigLIPVisionEncoder(config)
    for module in vision.modules():
        if isinstance(module, LayerNorm):
            module.promote_fp32 = False
    for layer in vision.layers:
        layer.mlp.act = GELU()
    return vision


class DeepseekBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision = make_vision(config.vision_config)
        self.linear1 = Linear(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.linear2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        self.activation = GELU()
        self.image_token_id = config.image_token_id
        self.pixel_values = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, pixels):
        return self.linear2(self.activation(self.linear1(self.vision(pixels))))

    def forward(self, input_ids, positions):
        hidden = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.features(self.pixel_values)
            mask = (input_ids == self.image_token_id)[:, None].expand_as(hidden)
            hidden = hidden.masked_scatter(mask, self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=hidden)


class DeepseekVLModel(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = DeepseekBackbone(language.model, config)


def build_from_config(config, device, dtype):
    vision = config.vision_config
    if vision.vision_use_head or vision.hidden_act != "gelu" or vision.num_channels != 3:
        raise ValueError("The published DeepSeek-VL configuration uses RGB SigLIP without its pooling head")
    language = llama.build_from_config(config.text_config, device, dtype)
    return DeepseekVLModel(language, config).to(device=device, dtype=dtype).eval()


def load_vision(vision, remaining, prefix):
    mapped = {}
    for name in vision.state_dict():
        source = name.replace("layers.", "encoder.layers.", 1)
        if source.startswith("patch_embedding."):
            source = "embeddings." + source
        elif source == "position_embedding":
            source = "embeddings.position_embedding.weight"
        value = remaining.pop(prefix + source)
        mapped[name] = value.unsqueeze(0) if name == "position_embedding" else value
    vision.load_state_dict(mapped, strict=True)


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    load_vision(model.model.vision, remaining, "model.vision_model.")
    for name in ("linear1", "linear2"):
        module = getattr(model.model, name)
        module.load_state_dict({field: remaining.pop(f"model.aligner.{name}.{field}")
                                for field in module.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped DeepSeek-VL weights: {sorted(remaining)}")


make_workloads = llava.make_workloads
