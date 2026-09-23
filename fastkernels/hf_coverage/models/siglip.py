"""Paired SigLIP text/image inference with existing towers and attention pooling."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L2.attention_pool import AttentionPoolLatent
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.siglip_encoder_layer import SigLIPEncoderLayer
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionEncoder

from ..runner import Workload


def configure_encoder(layers):
    for layer in layers:
        layer.layer_norm1.promote_fp32 = False
        layer.layer_norm2.promote_fp32 = False
        layer.mlp.act = GELU(approximate="tanh")


class SiglipTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("positions", torch.arange(config.max_position_embeddings)[None], persistent=False)
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(config.hidden_size, config.num_attention_heads,
                              config.intermediate_size, config.layer_norm_eps)
            for _ in range(config.num_hidden_layers)
        ])
        configure_encoder(self.layers)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.head = Linear(config.hidden_size, config.projection_size)

    def forward(self, input_ids):
        hidden = self.token_embedding(input_ids) + self.position_embedding(self.positions[:, :input_ids.shape[1]])
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_layer_norm(hidden)
        return hidden, self.head(hidden[:, -1])


class SiglipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        text, vision = config.text_config, config.vision_config
        self.text_model = SiglipTextModel(text)
        self.vision_model = SigLIPVisionEncoder(vision)
        configure_encoder(self.vision_model.layers)
        self.vision_model.post_layernorm.promote_fp32 = False
        self.pool = AttentionPoolLatent(
            vision.hidden_size, num_heads=vision.num_attention_heads,
            mlp_ratio=vision.intermediate_size / vision.hidden_size,
        )
        self.pool.norm = LayerNorm(vision.hidden_size, eps=vision.layer_norm_eps, promote_fp32=False)
        self.pool.mlp = VitEncoderMlp(vision.hidden_size, vision.intermediate_size,
                                     act_approximate="tanh", bias=True)
        self.logit_scale = nn.Parameter(torch.empty(1))
        self.logit_bias = nn.Parameter(torch.empty(1))
        self.register_buffer("scale", torch.empty(1), persistent=False)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values):
        image_hidden = self.vision_model(pixel_values)
        image_pool = self.pool(image_hidden)
        text_hidden, text_pool = self.text_model(input_ids)
        images = self.normalize(image_pool)
        texts = self.normalize(text_pool)
        logits = self.matmul(texts, images.t()) * self.scale + self.logit_bias
        return {"logits_per_text": logits, "logits_per_image": logits.t(),
                "text_embeds": texts, "image_embeds": images,
                "text_model_output.last_hidden_state": text_hidden,
                "text_model_output.pooler_output": text_pool,
                "vision_model_output.last_hidden_state": image_hidden,
                "vision_model_output.pooler_output": image_pool}


def build_from_config(config, device, dtype):
    if any(tower.hidden_act != "gelu_pytorch_tanh" for tower in (config.text_config, config.vision_config)):
        raise ValueError("SigLIP coverage uses the example checkpoint's tanh GELU")
    if config.vision_config.num_channels != 3 or not getattr(config.vision_config, "vision_use_head", True):
        raise ValueError("SigLIP coverage preserves RGB input and the attention pooling head")
    if config.text_config.projection_size != config.vision_config.hidden_size:
        raise ValueError("SigLIP text and image projection dimensions must match")
    return SiglipModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    sources = {}
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("text_model."):
            source = source.replace("text_model.token_embedding", "text_model.embeddings.token_embedding")
            source = source.replace("text_model.position_embedding", "text_model.embeddings.position_embedding")
            source = source.replace("text_model.layers.", "text_model.encoder.layers.")
        elif source.startswith("vision_model."):
            source = source.replace("vision_model.layers.", "vision_model.encoder.layers.")
            source = source.replace("vision_model.patch_embedding", "vision_model.embeddings.patch_embedding")
            source = source.replace("vision_model.position_embedding", "vision_model.embeddings.position_embedding.weight")
        elif source.startswith("pool."):
            suffix = source.removeprefix("pool.")
            if suffix.startswith(("q.", "kv.")):
                field = suffix.split(".")[1]
                source = "vision_model.head.attention.in_proj_" + field
                value = state_dict[source]
                width = config.vision_config.hidden_size
                parameter.copy_(value[:width] if suffix.startswith("q.") else value[width:])
                sources[name] = source
                continue
            suffix = {"latent": "probe"}.get(suffix, suffix)
            suffix = suffix.replace("proj.", "attention.out_proj.").replace("norm.", "layernorm.")
            source = "vision_model.head." + suffix
        value = state_dict[source]
        parameter.copy_(value.reshape(parameter.shape))
        sources[name] = source
    if set(sources.values()) != set(state_dict):
        raise KeyError(f"Unmapped SigLIP weights: {sorted(set(state_dict) - set(sources.values()))}")
    # Fixed checkpoint scalar: evaluate its exponential once in the same dtype.
    model.scale.copy_(model.logit_scale.exp())


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
