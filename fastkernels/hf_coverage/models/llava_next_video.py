"""LLaVA-NeXT-Video with spatial video pooling and linear-scaled RoPE."""

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, llava_next
from ..patches.linear_scaled_rope import LinearScaledRotaryEmbedding
from ..runner import Workload


class NextVideoBackbone(llava_next.NextBackbone):
    def __init__(self, text, config):
        super().__init__(text, config)
        self.video_pool = AvgPool2d(config.spatial_pool_stride)
        self.video_token_id = config.video_token_index
        self.pixel_values_videos = self.video_hidden_states = None

    def video_features(self, pixels):
        pixels = pixels.flatten(0, 1)
        hidden = self.vision.pre_layrnorm(self.vision.embeddings(pixels))
        selected = None
        for index, layer in enumerate(self.vision.encoder.layers):
            hidden = layer(hidden)
            if index == len(self.vision.encoder.layers) - 2:
                selected = hidden[:, 1:]
        self.vision.post_layernorm(hidden[:, 0])
        batch, _, width = selected.shape
        spatial = selected.view(batch, self.patch_grid, self.patch_grid, width).permute(0, 3, 1, 2)
        pooled = self.video_pool(spatial).flatten(2).transpose(1, 2).contiguous()
        return self.linear_2(self.activation(self.linear_1(pooled))).flatten(0, 1)

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.image_features(self.pixel_values, self.image_sizes)
            self.video_hidden_states = self.video_features(self.pixel_values_videos)
            for token, features in ((self.image_token_id, self.image_hidden_states),
                                    (self.video_token_id, self.video_hidden_states)):
                embeddings = embeddings.masked_scatter((input_ids == token)[:, None].expand_as(embeddings), features)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class LlavaNextVideoModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = NextVideoBackbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    text, rope = config.text_config, config.text_config.rope_parameters
    if (text.model_type != "llama" or text.hidden_act != "silu" or text.tie_word_embeddings
            or text.attention_bias or text.mlp_bias or rope["rope_type"] != "linear"
            or config.spatial_pool_mode != "average" or config.vision_feature_layer != -2
            or config.vision_feature_select_strategy != "default" or config.projector_hidden_act != "gelu"
            or config.vision_config.hidden_act != "quick_gelu"):
        raise ValueError("The documented NeXT-Video checkpoint uses average pooling and linear-scaled Llama RoPE")
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "rms_norm_eps")
    native = LlamaConfig(**{name: getattr(text, name) for name in fields},
                         rope_theta=rope["rope_theta"], rope_scaling_factor=1.0, dtype=dtype)
    language = LlamaForCausalLM(native)
    rotary = LinearScaledRotaryEmbedding(text.head_dim, text.max_position_embeddings, rope["rope_theta"], rope["factor"])
    language.model.rotary_emb = rotary
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = rotary
    return LlavaNextVideoModel(language, config).to(device=device, dtype=dtype).eval()


load_state_dict_into = llava_next.load_state_dict_into


def make_workloads(model, inputs, config, case=None):
    model.model.pixel_values = inputs["pixel_values"]
    model.model.pixel_values_videos = inputs["pixel_values_videos"]
    # Feature construction converts this input metadata inside timed prefill.
    model.model.image_sizes = inputs["image_sizes"]
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config, case=case)
    prefill = workloads["prefill"]

    def run_prefill():
        return {**prefill.run(), "image_hidden_states": model.model.image_hidden_states,
                "video_hidden_states": model.model.video_hidden_states}

    workloads["prefill"] = Workload(run=run_prefill, prepare=prefill.prepare, collect=prefill.collect)
    return workloads
