"""Documented LLaVA 1.5: CLIP features and projector into the existing Llama."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear

from . import llama
from .clip import ClipVisionModel
from ..runner import Workload


class LlavaBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision = ClipVisionModel(config.vision_config)
        for module in self.vision.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
        self.linear_1 = Linear(config.vision_config.hidden_size, config.text_config.hidden_size,
                               bias=config.multimodal_projector_bias)
        self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size,
                               bias=config.multimodal_projector_bias)
        self.activation = GELU()
        self.image_token_id = config.image_token_index
        self.pixel_values = None
        self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            vision = self.vision
            hidden = vision.pre_layrnorm(vision.embeddings(self.pixel_values))
            selected = None
            for index, layer in enumerate(vision.encoder.layers):
                hidden = layer(hidden)
                if index == len(vision.encoder.layers) - 2:
                    selected = hidden
            # Preserve the final tower layer and pooler, even though selection is -2.
            vision.post_layernorm(hidden[:, 0])
            images = self.linear_2(self.activation(self.linear_1(selected[:, 1:])))
            self.image_hidden_states = images.reshape(-1, images.shape[-1])
            mask = (input_ids == self.image_token_id)[:, None].expand_as(embeddings)
            embeddings = embeddings.masked_scatter(mask, self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class LlavaModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = LlavaBackbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    if (config.vision_feature_layer != -2 or config.vision_feature_select_strategy != "default"
            or config.projector_hidden_act != "gelu" or config.vision_config.hidden_act != "quick_gelu"
            or config.vision_config.num_hidden_layers < 2):
        raise ValueError("The selected LLaVA checkpoint requires penultimate CLIP patches and the GELU projector")
    text = llama.build_from_config(config.text_config, device, dtype)
    return LlavaModel(text, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    vision = {}
    for name in model.model.vision.state_dict():
        source = name.replace(".emb.weight", ".weight").replace("embeddings.patch_embedding.proj.", "embeddings.patch_embedding.")
        source = source.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
        source = source.replace(".mlp_fc1.", ".mlp.fc1.").replace(".mlp_fc2.", ".mlp.fc2.")
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            source = source.replace(f".{projection}.", f".self_attn.{projection}.")
        vision[name] = remaining.pop("model.vision_tower." + source)
    model.model.vision.load_state_dict(vision, strict=True)
    for name in ("linear_1", "linear_2"):
        module = getattr(model.model, name)
        module.load_state_dict({field: remaining.pop(f"model.multi_modal_projector.{name}.{field}")
                                for field in module.state_dict()})
    if remaining:
        raise KeyError(f"Unmapped LLaVA weights: {sorted(remaining)}")


def make_workloads(model, inputs, config, case=None):
    model.model.pixel_values = inputs["pixel_values"]
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config, case=case)
    prefill = workloads["prefill"]

    def run_prefill():
        output = prefill.run()
        return {**output, "image_hidden_states": model.model.image_hidden_states}

    workloads["prefill"] = Workload(run=run_prefill, prepare=prefill.prepare, collect=prefill.collect)
    return workloads
