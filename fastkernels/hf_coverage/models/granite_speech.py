"""Granite Speech with its active, trained q/v LoRA adapter."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.linear import Linear
from . import granite_speech_plus as shared


class AudioLoRA(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.query_width = config.num_attention_heads * config.head_dim
        self.value_width = config.num_key_value_heads * config.head_dim
        # The documented adapter config specifies r=64 and alpha=32.
        self.query_a, self.query_b = Linear(config.hidden_size, 64, bias=False), Linear(64, self.query_width, bias=False)
        self.value_a, self.value_b = Linear(config.hidden_size, 64, bias=False), Linear(64, self.value_width, bias=False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        query, key, value = self.base(hidden).split((self.query_width, self.value_width, self.value_width), -1)
        query = query + self.query_b(self.query_a(hidden.to(self.query_a.weight.dtype))) * 0.5
        value = value + self.value_b(self.value_a(hidden.to(self.value_a.weight.dtype))) * 0.5
        return torch.cat((query.to(hidden.dtype), key, value.to(hidden.dtype)), -1)


class SuppressAudioToken(nn.Module):
    def __init__(self, top1, token):
        super().__init__()
        self.top1, self.token = top1, token

    def forward(self, logits):
        scores = logits.clone()
        scores[..., self.token] = -float("inf")
        return self.top1(scores)


def build_from_config(config, device, dtype):
    if not config.has_lora_adapter or not config.text_config.use_cache:
        raise ValueError("Granite Speech requires its audio-enabled LoRA adapter and cached generation")
    model = shared.build_speech_model(config, device, dtype)
    for layer in model.model.layers:
        layer.self_attn.qkv_proj = AudioLoRA(layer.self_attn.qkv_proj, model.config)
    model.top1 = SuppressAudioToken(model.top1, config.audio_token_index)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for index, layer in enumerate(model.model.layers):
        for projection, target in (("q", "query"), ("v", "value")):
            prefix = f"language_model.model.layers.{index}.self_attn.{projection}_proj."
            for letter in ("A", "B"):
                weight = remaining.pop(prefix + f"lora_{letter}.default.weight")
                module = getattr(layer.self_attn.qkv_proj, f"{target}_{letter.lower()}")
                # Native adapter loading owns its dtype, independently of the
                # base-model precision. Preserve that source tensor dtype.
                module.to(dtype=weight.dtype)
                module.load_state_dict({"weight": weight}, strict=True)
            remaining[prefix + "weight"] = remaining.pop(prefix + "base_layer.weight")
    shared.load_state_dict_into(model, remaining, config)


def make_workloads(model, inputs, config, *, case):
    adapter = case["reference"]["adapter_config"]
    if (adapter["r"] != 64 or adapter["lora_alpha"] != 32 or set(adapter["target_modules"]) != {"q_proj", "v_proj"}
            or adapter["bias"] != "none" or adapter["use_dora"] or adapter["use_rslora"]):
        raise ValueError("The selected Granite Speech adapter uses rank64 q/v low-rank residuals scaled by0.5")
    if case["reference"]["generation_config"].get("suppress_tokens") != [config.audio_token_index]:
        raise ValueError("Granite Speech preserves the checkpoint's audio-token suppression")
    if "input_features" not in inputs:
        raise ValueError("This task requires audio inputs with the adapter enabled")
    return shared.make_workloads(model, inputs, config, case=case)
