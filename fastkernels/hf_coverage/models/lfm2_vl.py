"""LFM2-VL with the existing SigLIP2 tower and gated-convolution decoder."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from . import lfm2
from .siglip2 import _Vision
from ..runner import Workload


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor = config.downsample_factor
        width = config.vision_config.hidden_size * self.factor ** 2
        self.layer_norm = LayerNorm(width) if config.projector_use_layernorm else nn.Identity()
        self.linear_1 = Linear(width, config.projector_hidden_size, bias=config.projector_bias)
        self.linear_2 = Linear(config.projector_hidden_size, config.text_config.hidden_size,
                               bias=config.projector_bias)
        self.activation = GELU()

    def forward(self, hidden):
        batch, height, width, channels = hidden.shape
        factor = self.factor
        hidden = hidden.reshape(batch, height, width // factor, channels * factor).permute(0, 2, 1, 3)
        hidden = hidden.reshape(batch, width // factor, height // factor, channels * factor ** 2)
        hidden = hidden.permute(0, 2, 1, 3)
        return self.linear_2(self.activation(self.linear_1(self.layer_norm(hidden))))


class Model(nn.Module):
    def __init__(self, config, language):
        super().__init__()
        self.config, self.language = config, language
        self.vision_tower = _Vision(config.vision_config)
        self.multi_modal_projector = Projector(config)

    def forward(self, ids, positions, inputs=None):
        hidden = self.language.model.embed_tokens(ids)
        result = {}
        if inputs is not None:
            pixels = inputs["pixel_values"]
            mask = torch.zeros_like(inputs["pixel_attention_mask"], dtype=pixels.dtype)
            mask = mask.masked_fill(~inputs["pixel_attention_mask"].bool(), torch.finfo(pixels.dtype).min)
            vision = self.vision_tower(pixels, inputs["spatial_shapes"], mask[:, None, None])
            features = []
            for row, (height, width) in zip(vision, inputs["spatial_shapes"].tolist()):
                feature = row[:height * width].reshape(1, height, width, -1)
                features.append(self.multi_modal_projector(feature).reshape(-1, hidden.shape[-1]))
            images = torch.cat(features)
            hidden[ids == self.config.image_token_id] = images
            result["image_hidden_states"] = images
        for layer in self.language.model.layers:
            hidden = layer(hidden, positions)
        result["logits"] = self.language.lm_head(self.language.model.embedding_norm(hidden))
        for index, layer in enumerate(self.language.model.layers):
            prefix = f"past_key_values.{index}."
            if layer.is_attention:
                result[prefix + "key"] = layer.self_attn.keys.transpose(1, 2)
                result[prefix + "value"] = layer.self_attn.values.transpose(1, 2)
            else:
                result[prefix + "conv_states"] = layer.conv.state
        return result


def build_from_config(config, device, dtype):
    if config.projector_hidden_act != "gelu" or config.vision_config.vision_use_head:
        raise ValueError("The selected LFM2-VL checkpoint uses GELU projection and no vision pooler")
    if config.vision_config.hidden_act != "gelu_pytorch_tanh":
        raise ValueError("The selected vision tower uses tanh GELU")
    language = lfm2.build_from_config(config.text_config, device, dtype)
    return Model(config, language).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    remaining = dict(state)
    language = {name.replace("model.language_model.", "model."): remaining.pop(name)
                for name in list(remaining) if name.startswith("model.language_model.")}
    language["lm_head.weight"] = remaining.pop("lm_head.weight")
    lfm2.load_state_dict_into(model.language, language, config.text_config)
    mapped = {}
    for name in model.vision_tower.state_dict():
        source = name.replace("layers.", "encoder.layers.")
        if name in ("position_embedding", "patch_embedding.weight", "patch_embedding.bias"):
            source = "embeddings." + source
            if name == "position_embedding":
                source += ".weight"
        mapped[name] = remaining.pop("model.vision_tower." + source)
    model.vision_tower.load_state_dict(mapped, strict=True)
    model.multi_modal_projector.load_state_dict({
        name: remaining.pop("model.multi_modal_projector." + name)
        for name in model.multi_modal_projector.state_dict()
    }, strict=True)
    if remaining:
        raise KeyError(f"Unmapped LFM2-VL weights: {sorted(remaining)}")


def make_workloads(model, inputs, config, *, case=None):
    ids = inputs["input_ids"]
    continuation = case is not None and case.get("workload") == "causal_lm_continuation"
    steps = 2 if continuation else 1
    prefix_length = ids.shape[1] - steps
    if prefix_length < 1:
        raise ValueError("LFM2-VL requires a prompt and the supplied continuation tokens")
    positions = torch.arange(ids.shape[1], device=ids.device)

    def prefill():
        return model(ids[:, :prefix_length], positions[:prefix_length], inputs)

    workloads = {"prefill": Workload(run=prefill, prepare=model.language.reset)}
    for step in range(steps):
        def decode(step=step):
            position = prefix_length + step
            return model(ids[:, position:position + 1], positions[position:position + 1])

        def prepare_decode(step=step):
            model.language.reset()
            prefill()
            for prior in range(step):
                position = prefix_length + prior
                model(ids[:, position:position + 1], positions[position:position + 1])

        name = f"decode_{step + 1}" if continuation else "decode"
        workloads[name] = Workload(run=decode, prepare=prepare_decode)
    return workloads
