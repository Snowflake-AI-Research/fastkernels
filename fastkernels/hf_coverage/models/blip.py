"""BLIP paired base model, retaining both towers and their default poolers."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock
from . import blip_text
from .mvp import EagerAttention
from ..runner import Workload


class BlipAttention(VitEncoderAttention):
    def __init__(self, width, heads):
        super().__init__(width, heads)
        # HF BLIP scales the already-rounded scores, rather than using SDPA.
        self.attention = EagerAttention(prescale_query=False)

    def forward(self, hidden, attn_mask=None):
        batch, length, width = hidden.shape
        query, key, value = self.qkv(hidden).view(batch, length, 3, self.num_heads, self.head_dim).unbind(2)
        return self.proj(self.attention(query, key, value, attn_mask=attn_mask).reshape(batch, length, width))


class BlipVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.patch_embedding = OasisPatchEmbed(config.image_size, config.image_size, config.patch_size,
                                               config.num_channels, width)
        self.class_embedding = nn.Parameter(torch.empty(1, 1, width))
        self.position_embedding = nn.Parameter(torch.empty(1, self.patch_embedding.num_patches + 1, width))
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = VitEncoderBlock(width, config.num_attention_heads, config.intermediate_size / width,
                                    norm_eps=config.layer_norm_eps)
            layer.attn = BlipAttention(width, config.num_attention_heads)
            self.layers.append(layer)
        self.post_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values):
        hidden = torch.cat((self.class_embedding.expand(pixel_values.shape[0], -1, -1),
                            self.patch_embedding(pixel_values)), dim=1) + self.position_embedding
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.post_layernorm(hidden)
        # BLIP applies this same norm again to the already-normalized CLS token.
        return hidden, self.post_layernorm(hidden[:, 0])


class BlipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = blip_text.BlipTextModel(config.text_config)
        self.vision_model = BlipVision(config.vision_config)
        self.visual_projection = Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        self.text_projection = Linear(config.text_config.hidden_size, config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer("scale", torch.empty(()), persistent=False)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values):
        vision, vision_pooler = self.vision_model(pixel_values)
        text = self.text_model(input_ids)
        images = self.normalize(self.visual_projection(vision_pooler))
        texts = self.normalize(self.text_projection(text["pooler_output"]))
        logits = self.matmul(texts, images.t()) * self.scale
        return {"logits_per_text": logits, "logits_per_image": logits.t(),
                "text_embeds": texts, "image_embeds": images,
                "text_model_output.last_hidden_state": text["last_hidden_state"],
                "text_model_output.pooler_output": text["pooler_output"],
                "vision_model_output.last_hidden_state": vision,
                "vision_model_output.pooler_output": vision_pooler}


def build_from_config(config, device, dtype):
    if config.text_config.hidden_act != "gelu" or config.vision_config.hidden_act != "gelu":
        raise ValueError("The documented BLIP checkpoint uses GELU in both towers")
    return BlipModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.removeprefix("text_model."): remaining.pop(name)
            for name in list(remaining) if name.startswith("text_model.")}
    blip_text.load_state_dict_into(model.text_model, text, config.text_config)
    mapped = {}
    for name in model.vision_model.state_dict():
        source = name
        for target, origin in (("patch_embedding.proj.", "embeddings.patch_embedding."),
                               ("class_embedding", "embeddings.class_embedding"),
                               ("position_embedding", "embeddings.position_embedding"),
                               ("layers.", "encoder.layers."), (".norm1.", ".layer_norm1."),
                               (".norm2.", ".layer_norm2."), (".attn.qkv.", ".self_attn.qkv."),
                               (".attn.proj.", ".self_attn.projection.")):
            source = source.replace(target, origin)
        mapped[name] = remaining.pop("vision_model." + source)
    model.vision_model.load_state_dict(mapped, strict=True)
    for name in ("visual_projection", "text_projection"):
        getattr(model, name).load_state_dict({"weight": remaining.pop(name + ".weight")})
    model.logit_scale.copy_(remaining.pop("logit_scale"))
    if remaining:
        raise KeyError(f"Unmapped BLIP weights: {sorted(remaining)}")
    model.scale.copy_(model.logit_scale.exp())


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
