"""AltCLIP's projected RoBERTa sequence, CLIP vision tower and paired outputs."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.xlm_roberta_model import XLMRobertaModel
from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModelOutput

from .chinese_clip import PairedClipModel, load_state_dict_into as load_paired_state
from .clip import make_workloads


class AltCLIPTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = XLMRobertaModel(config)
        self.pre_LN = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.transformation = Linear(config.hidden_size, config.project_dim)

    def forward(self, input_ids):
        hidden = self.roberta.forward_with_attention_mask(input_ids)
        projected = self.transformation(self.pre_LN(hidden))
        return CLIPTextModelOutput(projected, projected[:, 0])


class AltCLIPModel(PairedClipModel):
    def __init__(self, config):
        super().__init__(config, AltCLIPTextModel(config.text_config))
        self.text_projection = Linear(config.text_config.project_dim, config.projection_dim, bias=False)
        # Preserve native attention rounding through both complete towers.
        for layer in self.text_model.roberta.encoder.layer:
            layer.attention.self.attn = DenseAttention(backend="cudnn")
        for layer in self.vision_model.encoder.layers:
            layer.attn = DenseAttention(backend="cudnn")

    def forward(self, input_ids, pixel_values):
        vision_hidden, vision_pool = self.vision_model(pixel_values)
        text_output = self.text_model(input_ids)
        images = self.normalize(self.visual_projection(vision_pool))
        texts = self.normalize(self.text_projection(text_output.pooler_output))
        logits = self.matmul(texts, images.t()) * self.scale
        # Unlike ChineseCLIP's feature helpers, this wrapper leaves both nested
        # pooler outputs at their original tower widths.
        return {
            "logits_per_text": logits, "logits_per_image": logits.t(),
            "text_embeds": texts, "image_embeds": images,
            "text_model_output.last_hidden_state": text_output.last_hidden_state,
            "text_model_output.pooler_output": text_output.pooler_output,
            "vision_model_output.last_hidden_state": vision_hidden,
            "vision_model_output.pooler_output": vision_pool,
        }


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if (text.hidden_act != "gelu" or vision.hidden_act != "quick_gelu"
            or text.is_decoder or text.add_cross_attention or text.chunk_size_feed_forward
            or text.position_embedding_type != "absolute"):
        raise ValueError("AltCLIP coverage preserves the bidirectional GELU text and QuickGELU vision towers")
    if any(getattr(tower, name, False) for tower in (text, vision)
           for name in ("output_hidden_states", "output_attentions")):
        raise ValueError("AltCLIP coverage returns the complete ordinary inference outputs")
    return AltCLIPModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    load_paired_state(model, {
        name.replace("text_model.roberta.encoder.layers.", "text_model.roberta.encoder.layer."): value
        for name, value in state_dict.items()
    }, config)
