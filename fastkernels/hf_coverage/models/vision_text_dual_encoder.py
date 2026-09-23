"""Documented ViT/BERT dual encoder, including both poolers and projections."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.bert_model import BertModel

from . import vit
from ..runner import Workload


class VisionTextDualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = vit.ViTModel(config.vision_config)
        self.text_model = BertModel(config.text_config)
        width = config.text_config.hidden_size
        self.text_pooler = Linear(width, width)
        self.pooler_activation = Tanh()
        self.visual_projection = Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        self.text_projection = Linear(width, config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer("scale", torch.empty(()), persistent=False)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values):
        vision = self.vision_model(pixel_values)
        text = self.text_model.forward_with_attention_mask(input_ids)
        pooled = self.pooler_activation(self.text_pooler(text[:, 0]))
        images = self.normalize(self.visual_projection(vision["pooler_output"]))
        texts = self.normalize(self.text_projection(pooled))
        logits = self.matmul(texts, images.t()) * self.scale
        return {"logits_per_text": logits, "logits_per_image": logits.t(),
                "text_embeds": texts, "image_embeds": images,
                "text_model_output.last_hidden_state": text,
                "text_model_output.pooler_output": pooled,
                "vision_model_output.last_hidden_state": vision["last_hidden_state"],
                "vision_model_output.pooler_output": vision["pooler_output"]}


def build_from_config(config, device, dtype):
    if config.vision_config.model_type != "vit" or config.text_config.model_type != "bert":
        raise ValueError("The documented forward example selects ViT and BERT")
    text = config.text_config
    if text.is_decoder or text.add_cross_attention or text.hidden_act != "gelu":
        raise ValueError("The selected BERT tower is a bidirectional GELU encoder")
    if config.vision_config.hidden_act != "gelu" or config.vision_config.pooler_act != "tanh":
        raise ValueError("The selected ViT tower uses GELU and its tanh pooler")
    return VisionTextDualEncoder(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    vision = {name.removeprefix("vision_model."): remaining.pop(name)
              for name in list(remaining) if name.startswith("vision_model.")}
    vit.load_state_dict_into(model.vision_model, vision, config.vision_config)
    mapped = {}
    for name in model.text_model.state_dict():
        source = "text_model." + name.replace(".emb.weight", ".weight")
        if ".qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".qkv.", f".{projection}."))
                                      for projection in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    model.text_model.load_state_dict(mapped, strict=True)
    model.text_pooler.load_state_dict({field: remaining.pop("text_model.pooler.dense." + field)
                                      for field in ("weight", "bias")})
    for name in ("visual_projection", "text_projection"):
        getattr(model, name).load_state_dict({"weight": remaining.pop(name + ".weight")})
    model.logit_scale.copy_(remaining.pop("logit_scale"))
    if remaining:
        raise KeyError(f"Unmapped dual encoder weights: {sorted(remaining)}")
    # Parameter-only preparation; the same-dtype scale is constant during inference.
    model.scale.copy_(model.logit_scale.exp())


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
