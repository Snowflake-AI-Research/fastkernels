"""VipLLaVA's default multi-layer CLIP features and normalized projector."""

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from . import llama, llava
from .clip import ClipVisionModel


class VipBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision = ClipVisionModel(config.vision_config)
        for module in self.vision.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
        self.feature_layers = config.vision_feature_layers
        width = config.vision_config.hidden_size * len(self.feature_layers)
        self.projector_layernorm = LayerNorm(width, eps=config.projector_layernorm_eps, promote_fp32=False)
        self.linear_1 = Linear(width, config.text_config.hidden_size)
        self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        self.activation = GELU()
        self.image_token_id = config.image_token_index
        self.pixel_values = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            hidden = self.vision.pre_layrnorm(self.vision.embeddings(self.pixel_values))
            states = [hidden]
            for layer in self.vision.encoder.layers:
                hidden = layer(hidden)
                states.append(hidden)
            self.vision.post_layernorm(hidden[:, 0])
            selected = torch.cat([states[index][:, 1:] for index in self.feature_layers], dim=-1)
            images = self.linear_2(self.activation(self.linear_1(self.projector_layernorm(selected))))
            self.image_hidden_states = images
            mask = (input_ids == self.image_token_id)[:, None].expand_as(embeddings)
            embeddings = embeddings.masked_scatter(mask, images)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class VipLlavaModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = VipBackbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    if (config.projector_hidden_act != "gelu" or config.vision_config.hidden_act != "quick_gelu"
            or not isinstance(config.vision_feature_layers, (list, tuple))):
        raise ValueError("The documented VipLLaVA checkpoint uses multiple CLIP layers and a GELU projector")
    text = llama.build_from_config(config.text_config, device, dtype)
    return VipLlavaModel(text, config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    model.model.projector_layernorm.load_state_dict({
        field: remaining.pop("model.multi_modal_projector.projector_layernorm." + field)
        for field in ("weight", "bias")})
    llava.load_state_dict_into(model, remaining, config)


make_workloads = llava.make_workloads
