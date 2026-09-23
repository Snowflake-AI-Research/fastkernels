"""AIMv2 paired inference with learned vision positions and attention pooling."""

import math
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from ..runner import Workload


class Attention(nn.Module):
    def __init__(self, config, pooling=False):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.hidden_size
        for name in (("k_proj", "v_proj") if pooling else ("q_proj", "k_proj", "v_proj")):
            setattr(self, name, Linear(self.width, self.width, bias=config.qkv_bias))
        self.pooling = pooling
        if pooling:
            self.cls_token = nn.Parameter(torch.empty(1, 1, self.width))
            self.output_proj = Linear(self.width, self.width)
        else:
            self.out_proj = Linear(self.width, self.width, bias=config.qkv_bias)
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, hidden, mask=None):
        batch, length, _ = hidden.shape
        query = self.cls_token.expand(batch, -1, -1) if self.pooling else self.q_proj(hidden)
        query = query.reshape(batch, -1, self.heads, self.width // self.heads)
        key = self.k_proj(hidden).view(batch, length, self.heads, self.width // self.heads)
        value = self.v_proj(hidden).view(batch, length, self.heads, self.width // self.heads)
        result = self.attn(query, key, value, attn_mask=mask).reshape(batch, -1, self.width)
        return self.output_proj(result[:, 0]) if self.pooling else self.out_proj(result)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.activation = SiluAndMul()

    def forward(self, hidden):
        return self.down_proj(self.activation(torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), dim=-1)))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.ffn = MLP(config)
        self.rms_norm1 = T5LayerNorm(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = T5LayerNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, mask=None):
        hidden = hidden + self.attention(self.rms_norm1(hidden), mask)
        return hidden + self.ffn(self.rms_norm2(hidden))


class Tower(nn.Module):
    def __init__(self, config, vision=False):
        super().__init__()
        self.vision = vision
        if vision:
            self.embeddings = nn.ModuleDict({
                "patch_embed": Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size),
                "rms_norm": T5LayerNorm(config.hidden_size, config.rms_norm_eps),
                "position_embedding": Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size),
            })
            self.head = Attention(config, pooling=True)
        else:
            self.embeddings = nn.ModuleDict({
                "token_embedding": Embedding(config.vocab_size, config.hidden_size),
                "position_embedding": Embedding(config.max_position_embeddings, config.hidden_size),
            })
            self.eos_token_id = config.eos_token_id
        self.encoder = nn.ModuleDict({"layers": nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])})
        self.rms_norm = T5LayerNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, inputs, attention_mask=None):
        if self.vision:
            hidden = self.embeddings["rms_norm"](self.embeddings["patch_embed"](inputs).flatten(2).transpose(1, 2))
        else:
            hidden = self.embeddings["token_embedding"](inputs)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.embeddings["position_embedding"](positions)
        mask = None
        if attention_mask is not None:
            allowed = (positions[None, :] <= positions[:, None])[None, None] & attention_mask[:, None, None].bool()
            mask = torch.zeros(allowed.shape, dtype=hidden.dtype, device=hidden.device).masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        for layer in self.encoder["layers"]:
            hidden = layer(hidden, mask)
        hidden = self.rms_norm(hidden)
        pooler = (self.head(hidden) if self.vision else
                  hidden[torch.arange(hidden.shape[0], device=hidden.device), (inputs == self.eos_token_id).int().argmax(-1)])
        return hidden, pooler


class Aimv2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = Tower(config.vision_config, vision=True)
        self.text_model = Tower(config.text_config)
        self.visual_projection = Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        self.text_projection = Linear(config.text_config.hidden_size, config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer("scale", torch.empty(()), persistent=False)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values, attention_mask):
        vision, vision_pooler = self.vision_model(pixel_values)
        text, text_pooler = self.text_model(input_ids, attention_mask)
        images = self.normalize(self.visual_projection(vision_pooler))
        texts = self.normalize(self.text_projection(text_pooler))
        logits = self.matmul(self.scale * texts, images.t())
        return {"logits_per_text": logits, "logits_per_image": logits.t(),
                "text_embeds": texts, "image_embeds": images,
                "text_model_output.last_hidden_state": text, "text_model_output.pooler_output": text_pooler,
                "vision_model_output.last_hidden_state": vision, "vision_model_output.pooler_output": vision_pooler}


def build_from_config(config, device, dtype):
    if config.vision_config.is_native or not config.vision_config.use_head:
        raise ValueError("The documented paired checkpoint uses learned positions and attention pooling")
    if any(part.hidden_act != "silu" for part in (config.text_config, config.vision_config)):
        raise ValueError("The documented paired checkpoint uses SwiGLU")
    return Aimv2Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        mapped[name] = remaining.pop(name.replace(".emb.weight", ".weight"))
    if remaining:
        raise KeyError(f"Unmapped AIMv2 state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
    model.scale.copy_(model.logit_scale.clamp(0.0, math.log(config.max_logit_scale)).exp())


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
