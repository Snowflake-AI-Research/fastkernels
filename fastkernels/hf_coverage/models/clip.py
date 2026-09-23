"""Paired CLIP inference with the existing text encoder and CLIP vision blocks."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed
from fastkernels.tasks.baseline.L2.sam3_text_attention import Sam3TextAttentionBlock
from fastkernels.tasks.baseline.L4.clip_text_model import CLIPEncoder, CLIPTextModel

from ..runner import Workload


class ClipEncoderLayer(Sam3TextAttentionBlock):
    """Select existing CLIP-compatible children and adapt the mask argument name."""

    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_attention_heads,
                         mlp_ratio=config.intermediate_size / config.hidden_size)
        self.mlp_act = QuickGELU()
        self.attn = DenseAttention(backend="sdpa")
        self.ln_1.eps = self.ln_2.eps = config.layer_norm_eps

    def forward(self, hidden_states, attention_mask=None):
        # The default CLIP text mask is purely causal; vision supplies no mask.
        return super().forward(hidden_states, attn_mask=attention_mask)


def configure_encoder(encoder, config):
    for index in range(len(encoder.layers)):
        encoder.layers[index] = ClipEncoderLayer(config)


class ClipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = OasisPatchEmbed(
            img_height=config.image_size, img_width=config.image_size,
            patch_size=config.patch_size, in_chans=config.num_channels,
            embed_dim=config.hidden_size,
        )
        self.patch_embedding.proj.bias = None
        self.class_embedding = nn.Parameter(torch.empty(config.hidden_size))
        positions = self.patch_embedding.num_patches + 1
        self.position_embedding = Embedding(positions, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(positions)[None], persistent=False)

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values.to(self.patch_embedding.proj.weight.dtype))
        classes = self.class_embedding.expand(pixel_values.shape[0], 1, -1)
        return torch.cat((classes, patches), dim=1) + self.position_embedding(self.position_ids)


class ClipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ClipVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.encoder = CLIPEncoder(config)
        configure_encoder(self.encoder, config)
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values):
        hidden = self.encoder(self.pre_layrnorm(self.embeddings(pixel_values)))
        return hidden, self.post_layernorm(hidden[:, 0])


class ClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = CLIPTextModel(config.text_config)
        configure_encoder(self.text_model.text_model.encoder, config.text_config)
        self.vision_model = ClipVisionModel(config.vision_config)
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
        self.text_projection = Linear(config.text_config.hidden_size, config.projection_dim, bias=False)
        self.visual_projection = Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer("scale", torch.empty(()), persistent=False)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.matmul = BMM()

    def forward(self, input_ids, pixel_values):
        image_hidden, image_pool = self.vision_model(pixel_values)
        text_output = self.text_model(input_ids)
        image_pool = self.visual_projection(image_pool)
        text_pool = self.text_projection(text_output.pooler_output)
        images = self.normalize(image_pool)
        texts = self.normalize(text_pool)
        logits = self.matmul(texts, images.t()) * self.scale
        return {"logits_per_text": logits, "logits_per_image": logits.t(),
                "text_embeds": texts, "image_embeds": images,
                "text_model_output.last_hidden_state": text_output.last_hidden_state,
                "text_model_output.pooler_output": text_pool,
                "vision_model_output.last_hidden_state": image_hidden,
                "vision_model_output.pooler_output": image_pool}


def build_from_config(config, device, dtype):
    if any(tower.hidden_act != "quick_gelu" for tower in (config.text_config, config.vision_config)):
        raise ValueError("CLIP coverage preserves the example checkpoint's QuickGELU")
    if config.text_config.eos_token_id != 2:
        raise ValueError("The existing CLIP text encoder implements the checkpoint's legacy argmax pooling")
    return ClipModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, _ in model.named_parameters():
        source = name.replace("text_model.text_model.", "text_model.")
        source = source.replace(".emb.weight", ".weight")
        source = source.replace("embeddings.patch_embedding.proj.", "embeddings.patch_embedding.")
        source = source.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
        source = source.replace(".mlp_fc1.", ".mlp.fc1.").replace(".mlp_fc2.", ".mlp.fc2.")
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            source = source.replace(f".{projection}.", f".self_attn.{projection}.")
        mapped[name] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f"Unmapped CLIP weights: {sorted(set(state_dict) - used)}")
    model.load_state_dict(mapped, strict=True)
    # Fixed checkpoint scalar: the same-dtype exponential is outside timed inference.
    model.scale.copy_(model.logit_scale.exp())


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
