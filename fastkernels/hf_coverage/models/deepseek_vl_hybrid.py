"""DeepSeek-VL's low-resolution SigLIP and two-branch high-resolution SAM path."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.linear import Linear
from . import deepseek_vl, llama, llava
from .sam import VisionEncoder


class HybridBackbone(deepseek_vl.DeepseekBackbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text = text
        self.vision = deepseek_vl.make_vision(config.vision_config)
        high = config.high_res_vision_config
        self.high_vision = VisionEncoder(high)
        self.high_neck = nn.Sequential(
            Conv2d(high.hidden_size, high.output_channels, 1, bias=False),
            LayerNorm2d(high.output_channels),
            Conv2d(high.output_channels, high.output_channels, 3, padding=1, bias=False),
            LayerNorm2d(high.output_channels),
        )
        self.high_projection = nn.Sequential(
            Conv2d(high.output_channels, high.output_channels * 2, 3, stride=2, padding=1, bias=False),
            Conv2d(high.output_channels * 2, high.output_channels * 4, 3, stride=2, padding=1, bias=False),
        )
        self.alpha = nn.Parameter(torch.empty(1))
        self.resize = Interpolate()
        self.output_size = config.vision_config.image_size // config.vision_config.patch_size
        width = config.text_config.hidden_size
        self.vision_proj = Linear(config.vision_config.hidden_size, width // 2)
        self.high_res_vision_proj = Linear(high.output_channels * 4, width // 2)
        self.proj, self.activation = Linear(width, width), GELU()
        self.image_token_id = config.image_token_id
        self.pixel_values = self.high_res_pixel_values = self.image_hidden_states = None

    def project_high(self, hidden):
        resized = self.resize(hidden, size=(4 * self.output_size,) * 2,
                              mode="bilinear", align_corners=False)
        return self.high_projection(resized)

    def features(self, pixels):
        low = self.vision(pixels)
        final, globals_ = self.high_vision(self.high_res_pixel_values)
        first_global = self.high_neck(globals_[0].permute(0, 3, 1, 2))
        # Alpha is the fixed, loaded one-element branch scale from the checkpoint.
        high = self.project_high(final) + self.project_high(first_global) * self.alpha
        high = high.flatten(2).transpose(1, 2)
        joined = torch.cat((self.high_res_vision_proj(high), self.vision_proj(low)), dim=-1)
        return self.proj(self.activation(joined))


class HybridModel(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = HybridBackbone(language.model, config)


def build_from_config(config, device, dtype):
    high = config.high_res_vision_config
    if (config.vision_config.vision_use_head or config.vision_config.hidden_act != "gelu"
            or not high.use_rel_pos or not high.use_abs_pos
            or high.num_hidden_layers <= high.global_attn_indexes[0]):
        raise ValueError("Preserve SigLIP without pooling and both global/final SAM feature branches")
    language = llama.build_from_config(config.text_config, device, dtype)
    return HybridModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace("model.language_model.", "model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("model.language_model.")}
    text["lm_head.weight"] = remaining.pop("lm_head.weight")
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    deepseek_vl.load_vision(model.model.vision, remaining, "model.vision_model.")
    high = {}
    for name in model.model.high_vision.state_dict():
        source = name.replace(".mlp.fc1.", ".mlp.lin1.").replace(".mlp.fc2.", ".mlp.lin2.")
        for index, field in ((0, "conv1"), (1, "layer_norm1"), (2, "conv2"), (3, "layer_norm2")):
            source = source.replace(f"neck.{index}.", f"neck.{field}.")
        high[name] = remaining.pop("model.high_res_vision_model.vision_encoder." + source)
    model.model.high_vision.load_state_dict(high, strict=True)
    for index, field in ((0, "conv1"), (1, "layer_norm1"), (2, "conv2"), (3, "layer_norm2")):
        module = model.model.high_neck[index]
        module.load_state_dict({name: remaining.pop(f"model.high_res_vision_neck.{field}.{name}")
                                for name in module.state_dict()}, strict=True)
    for index, module in enumerate(model.model.high_projection):
        module.load_state_dict({name: remaining.pop(f"model.high_res_vision_proj.conv{index + 1}.{name}")
                                for name in module.state_dict()}, strict=True)
    model.model.alpha.copy_(remaining.pop("model.high_res_vision_alpha"))
    for name in ("vision_proj", "high_res_vision_proj", "proj"):
        module = getattr(model.model, name)
        module.load_state_dict({field: remaining.pop(f"model.aligner.{name}.{field}")
                                for field in module.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f"Unmapped DeepSeek-VL hybrid weights: {sorted(remaining)}")


def make_workloads(model, inputs, config):
    model.model.high_res_pixel_values = inputs["high_res_pixel_values"]
    return llava.make_workloads(model, inputs, model.config)
