"""FLAVA image, text and joint encoders, including default intermediate states."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from .blip import BlipAttention
from .vit import _Embeddings, _encoder_block
from ..runner import Workload


class Tower(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        if kind == "image":
            self.embeddings = _Embeddings(config)
            if config.mask_token:
                self.embeddings.mask_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        elif kind == "text":
            self.embeddings = BertEmbeddings(config)
        elif config.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = _encoder_block(config)
            layer.attn = BlipAttention(config.hidden_size, config.num_attention_heads)
            layer.attn.attention.divide_scores = True
            if not config.qkv_bias:
                layer.attn.qkv.bias = None
            self.encoder.append(layer)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.pooler = nn.ModuleDict({"dense": Linear(config.hidden_size, config.hidden_size), "activation": Tanh()})

    def forward(self, inputs):
        if self.kind == "image":
            hidden = self.embeddings(inputs)
        elif self.kind == "text":
            positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
            hidden = self.embeddings(inputs, positions)
        else:
            hidden = torch.cat((self.cls_token.expand(inputs.shape[0], -1, -1), inputs), dim=1) if hasattr(self, "cls_token") else inputs
        states = [hidden]
        for layer in self.encoder:
            hidden = layer(hidden)
            states.append(hidden)
        normalized = self.layernorm(hidden)
        pooler = self.pooler["activation"](self.pooler["dense"](normalized[:, 0]))
        return normalized, pooler, states


class FlavaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        for name, kind in (("image", "image"), ("text", "text"), ("multimodal", "multimodal")):
            setattr(self, name + "_model", Tower(getattr(config, name + "_config"), kind))
        self.image_to_mm_projection = Linear(config.image_config.hidden_size, config.multimodal_config.hidden_size)
        self.text_to_mm_projection = Linear(config.text_config.hidden_size, config.multimodal_config.hidden_size)

    def forward(self, input_ids, pixel_values):
        image, image_pooler, image_states = self.image_model(pixel_values)
        text, text_pooler, text_states = self.text_model(input_ids)
        # HF's joint encoder receives the states BEFORE each tower's final norm.
        joint = torch.cat((self.image_to_mm_projection(image_states[-1]),
                           self.text_to_mm_projection(text_states[-1])), dim=1)
        multimodal, multimodal_pooler, _ = self.multimodal_model(joint)
        outputs = {"image_embeddings": image, "text_embeddings": text, "multimodal_embeddings": multimodal,
                   "image_output.last_hidden_state": image, "image_output.pooler_output": image_pooler,
                   "text_output.last_hidden_state": text, "text_output.pooler_output": text_pooler,
                   "multimodal_output.last_hidden_state": multimodal, "multimodal_output.pooler_output": multimodal_pooler}
        for tower, states in (("image", image_states), ("text", text_states)):
            outputs.update({f"{tower}_output.hidden_states.{index}": state for index, state in enumerate(states)})
        return outputs


def build_from_config(config, device, dtype):
    if any(getattr(config, name).hidden_act != "gelu" for name in ("image_config", "text_config", "multimodal_config")):
        raise ValueError("The documented FLAVA checkpoint uses GELU in all three encoders")
    return FlavaModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight").replace(".patch_embeddings.proj.", ".patch_embeddings.projection.")
        if ".encoder." in source:
            source = source.replace(".encoder.", ".encoder.layer.")
            for target, origin in ((".norm1.", ".layernorm_before."), (".norm2.", ".layernorm_after."),
                                   (".attn.proj.", ".attention.output.dense."), (".mlp.fc1.", ".intermediate.dense."),
                                   (".mlp.fc2.", ".output.dense.")):
                source = source.replace(target, origin)
        if ".attn.qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".attn.qkv.", f".attention.attention.{part}."))
                                      for part in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    # Base forward constructs but does not call these contrastive-only parameters.
    inactive = {"logit_scale", "image_projection.weight", "image_projection.bias", "text_projection.weight", "text_projection.bias"}
    if set(remaining) != inactive:
        raise KeyError(f"Unexpected FLAVA unused state: {sorted(set(remaining) ^ inactive)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
