"""Video-LLaVA's separate image/video CLIP towers and shared projector."""

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from . import llama, llava
from .clip import ClipVisionModel
from ..runner import Workload


class VideoBackbone(llava.LlavaBackbone):
    def __init__(self, text, config):
        super().__init__(text, config)
        self.video_vision = ClipVisionModel(config.vision_config)
        for module in self.video_vision.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
        self.video_token_id = config.video_token_index
        self.pixel_values_videos = self.video_hidden_states = None

    def features(self, tower, pixels, *, keep_cls):
        hidden = tower.pre_layrnorm(tower.embeddings(pixels))
        selected = None
        for index, layer in enumerate(tower.encoder.layers):
            hidden = layer(hidden)
            if index == len(tower.encoder.layers) - 2:
                selected = hidden
        tower.post_layernorm(hidden[:, 0])
        if not keep_cls:
            selected = selected[:, 1:]
        return self.linear_2(self.activation(self.linear_1(selected)))

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.features(self.vision, self.pixel_values, keep_cls=False)
            videos = self.pixel_values_videos.flatten(0, 1)
            self.video_hidden_states = self.features(self.video_vision, videos, keep_cls=True)
            for token, features in ((self.image_token_id, self.image_hidden_states),
                                    (self.video_token_id, self.video_hidden_states)):
                mask = (input_ids == token)[:, None].expand_as(embeddings)
                embeddings = embeddings.masked_scatter(mask, features)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class VideoLlavaModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = VideoBackbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    if (config.vision_feature_layer != -2 or config.vision_feature_select_strategy != "default"
            or config.projector_hidden_act != "gelu" or config.vision_config.hidden_act != "quick_gelu"):
        raise ValueError("The documented Video-LLaVA path selects penultimate CLIP features and GELU projection")
    text = llama.build_from_config(config.text_config, device, dtype)
    return VideoLlavaModel(text, config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {}
    for name in model.model.video_vision.state_dict():
        source = name.replace(".emb.weight", ".weight").replace("embeddings.patch_embedding.proj.", "embeddings.patch_embedding.")
        source = source.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
        source = source.replace(".mlp_fc1.", ".mlp.fc1.").replace(".mlp_fc2.", ".mlp.fc2.")
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            source = source.replace(f".{projection}.", f".self_attn.{projection}.")
        mapped[name] = remaining.pop("model.video_tower." + source)
    model.model.video_vision.load_state_dict(mapped, strict=True)
    remaining = {name.replace("model.image_tower.", "model.vision_tower."): value
                 for name, value in remaining.items()}
    llava.load_state_dict_into(model, remaining, config)


def make_workloads(model, inputs, config, case=None):
    model.model.pixel_values = inputs["pixel_values_images"]
    model.model.pixel_values_videos = inputs["pixel_values_videos"]
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config, case=case)
    prefill = workloads["prefill"]

    def run_prefill():
        return {**prefill.run(), "image_hidden_states": model.model.image_hidden_states,
                "video_hidden_states": model.model.video_hidden_states}

    workloads["prefill"] = Workload(run=run_prefill, prepare=prefill.prepare, collect=prefill.collect)
    return workloads
