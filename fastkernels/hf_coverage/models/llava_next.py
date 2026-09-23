"""LLaVA-NeXT's crop tiling, spatial unpadding and learned image newlines."""

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, llava
from ..runner import Workload


def crop_grid(size, pinpoints, crop_size):
    """Choose the native effective-area/wasted-area grid using image metadata."""
    original_h, original_w = size
    best, best_score = None, (-1, float("-inf"))
    for height, width in pinpoints:
        scale = min(width / original_w, height / original_h)
        effective = min(int(original_w * scale) * int(original_h * scale), original_h * original_w)
        score = (effective, -(height * width - effective))
        if score > best_score:
            best, best_score = (height // crop_size, width // crop_size), score
    return best


def unpad_features(features, size):
    original_h, original_w = size
    height, width = features.shape[1:]
    if original_w / original_h > width / height:
        padding = (height - int(round(original_h * width / original_w, 7))) // 2
        return features[:, padding:height - padding, :]
    padding = (width - int(round(original_w * height / original_h, 7))) // 2
    return features[:, :, padding:width - padding]


class NextBackbone(llava.LlavaBackbone):
    def __init__(self, text, config):
        super().__init__(text, config)
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.crop_size = config.vision_config.image_size
        self.patch_grid = self.crop_size // config.vision_config.patch_size
        self.grid_pinpoints = config.image_grid_pinpoints
        self.image_sizes = None

    def image_features(self, pixels, sizes):
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.tolist()
        grids = [crop_grid(size, self.grid_pinpoints, self.crop_size) for size in sizes]
        counts = [height * width + 1 for height, width in grids]
        patches = torch.cat([image[:count] for image, count in zip(pixels, counts)])
        hidden = self.vision.pre_layrnorm(self.vision.embeddings(patches))
        selected = None
        for index, layer in enumerate(self.vision.encoder.layers):
            hidden = layer(hidden)
            if index == len(self.vision.encoder.layers) - 2:
                selected = hidden[:, 1:]
        self.vision.post_layernorm(hidden[:, 0])
        projected = self.linear_2(self.activation(self.linear_1(selected)))
        outputs = []
        for features, grid, size in zip(projected.split(counts), grids, sizes):
            base, tiles = features[0], features[1:]
            height, width = grid
            tiles = tiles.view(height, width, self.patch_grid, self.patch_grid, -1)
            tiles = tiles.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
            tiles = unpad_features(tiles, size)
            newline = self.image_newline[:, None, None].expand(tiles.shape[0], tiles.shape[1], 1)
            tiles = torch.cat((tiles, newline), dim=-1).flatten(1, 2).transpose(0, 1)
            outputs.append(torch.cat((base, tiles)))
        return torch.cat(outputs)

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.image_features(self.pixel_values, self.image_sizes)
            mask = (input_ids == self.image_token_id)[:, None].expand_as(embeddings)
            embeddings = embeddings.masked_scatter(mask, self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class LlavaNextModel(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.config = text.config
        self.model = NextBackbone(text.model, config)
        self.lm_head = text.lm_head


def build_from_config(config, device, dtype):
    text = config.text_config
    if (text.model_type != "mistral" or text.sliding_window is not None or text.hidden_act != "silu"
            or text.rope_parameters["rope_type"] != "default" or text.tie_word_embeddings
            or config.vision_feature_layer != -2 or config.vision_feature_select_strategy != "default"
            or config.projector_hidden_act != "gelu" or config.vision_config.hidden_act != "quick_gelu"):
        raise ValueError("The documented LLaVA-NeXT Mistral-v0.2 checkpoint uses full causal attention and default RoPE")
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "rms_norm_eps")
    native = LlamaConfig(**{name: getattr(text, name) for name in fields},
                         rope_theta=text.rope_parameters["rope_theta"], rope_scaling_factor=1.0, dtype=dtype)
    model = LlavaNextModel(LlamaForCausalLM(native), config)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    with torch.no_grad():
        model.model.image_newline.copy_(remaining.pop("model.image_newline"))
    llava.load_state_dict_into(model, remaining, config)


def make_workloads(model, inputs, config, case=None):
    model.model.pixel_values = inputs["pixel_values"]
    # Retain input metadata; conversion and feature construction run during prefill.
    model.model.image_sizes = inputs["image_sizes"]
    workloads = llama.make_workloads(model, {"input_ids": inputs["input_ids"]}, model.config, case=case)
    prefill = workloads["prefill"]

    def run_prefill():
        return {**prefill.run(), "image_hidden_states": model.model.image_hidden_states}

    workloads["prefill"] = Workload(run=run_prefill, prepare=prefill.prepare, collect=prefill.collect)
    return workloads
