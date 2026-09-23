"""Paired SigLIP2 with rectangular/padded patches and executed position resizing."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L2.attention_pool import AttentionPoolLatent
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.siglip_encoder_layer import SigLIPEncoderLayer

from .siglip import SiglipTextModel, configure_encoder, load_state_dict_into, make_workloads
from ..patches.siglip2_interpolate import PositionTableResize


class _Text(SiglipTextModel):
    def forward(self, input_ids):
        hidden = self.token_embedding(input_ids) + self.position_embedding(self.positions[:, :input_ids.shape[1]])
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_layer_norm(hidden)
        return hidden, self.head(hidden[:, -1])


class _Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = math.isqrt(config.num_patches)
        if self.grid**2 != config.num_patches:
            raise ValueError("The learned position table must form a square source grid")
        self.patch_embedding = Linear(config.num_channels * config.patch_size**2, config.hidden_size)
        self.position_embedding = nn.Parameter(torch.empty(config.num_patches, config.hidden_size))
        self.resize = PositionTableResize()
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(config.hidden_size, config.num_attention_heads,
                              config.intermediate_size, config.layer_norm_eps)
            for _ in range(config.num_hidden_layers)
        ])
        configure_encoder(self.layers)
        for layer in self.layers:
            # DenseAttention's existing SDPA backend accepts padding masks.
            layer.self_attn.attn = DenseAttention(backend="sdpa")
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values, spatial_shapes, attention_mask):
        hidden = self.patch_embedding(pixel_values)
        table = self.position_embedding.reshape(self.grid, self.grid, -1).permute(2, 0, 1).unsqueeze(0)
        positions = torch.empty_like(hidden)
        for index, shape in enumerate(spatial_shapes.tolist()):
            height, width = shape
            count = height * width
            resized = self.resize(table, (height, width)).flatten(2).transpose(1, 2)[0]
            positions[index, :count] = resized
            positions[index, count:] = resized[:1]
        hidden = hidden + positions
        for layer in self.layers:
            normalized = layer.layer_norm1(hidden)
            attention = layer.self_attn
            batch, length, width = normalized.shape
            q, k, v = [projection(normalized).reshape(batch, length, attention.num_heads, attention.head_dim)
                       for projection in (attention.q_proj, attention.k_proj, attention.v_proj)]
            delta = attention.attn(q, k, v, causal=False, attn_mask=attention_mask,
                                   softmax_scale=attention.head_dim**-0.5)
            hidden = hidden + attention.out_proj(delta.reshape(batch, length, width))
            hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
        return self.post_layernorm(hidden)


class Siglip2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        text, vision = config.text_config, config.vision_config
        self.text_model = _Text(text)
        self.vision_model = _Vision(vision)
        self.pool = AttentionPoolLatent(vision.hidden_size, num_heads=vision.num_attention_heads,
                                       mlp_ratio=vision.intermediate_size / vision.hidden_size)
        self.pool.norm = LayerNorm(vision.hidden_size, eps=vision.layer_norm_eps, promote_fp32=False)
        self.pool.mlp = VitEncoderMlp(vision.hidden_size, vision.intermediate_size, act_approximate="tanh")
        self.logit_scale = nn.Parameter(torch.empty(1))
        self.logit_bias = nn.Parameter(torch.empty(1))
        self.register_buffer("scale", torch.empty(1), persistent=False)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values, spatial_shapes, pixel_attention_mask):
        mask = torch.zeros(pixel_attention_mask.shape, device=pixel_values.device, dtype=pixel_values.dtype)
        mask = mask.masked_fill(~pixel_attention_mask.bool(), torch.finfo(pixel_values.dtype).min)[:, None, None, :]
        hidden = self.vision_model(pixel_values, spatial_shapes, mask)
        image_pool = self.pool(hidden, attn_mask=mask)
        text_hidden, text_pool = self.text_model(input_ids)
        images = self.normalize(image_pool)
        texts = self.normalize(text_pool)
        logits = self.matmul(texts, images.t()) * self.scale + self.logit_bias
        return {"logits_per_text": logits, "logits_per_image": logits.t(),
                "text_embeds": texts, "image_embeds": images,
                "text_model_output.last_hidden_state": text_hidden,
                "text_model_output.pooler_output": text_pool,
                "vision_model_output.last_hidden_state": hidden,
                "vision_model_output.pooler_output": image_pool}


def build_from_config(config, device, dtype):
    if any(tower.hidden_act != "gelu_pytorch_tanh" for tower in (config.text_config, config.vision_config)):
        raise ValueError("Preserve default tanh GELU in both towers")
    if config.text_config.projection_size != config.vision_config.hidden_size:
        raise ValueError("Both pooled towers must have the same feature dimension")
    return Siglip2Model(config).to(device=device, dtype=dtype).eval()
