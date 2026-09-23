"""ChineseCLIP's bidirectional text tower and complete paired contrastive outputs."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L3.bert_model import BertModel
from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModelOutput

from .clip import ClipVisionModel, load_state_dict_into as load_clip_state, make_workloads


class _ChineseTextModel(BertModel):
    def forward(self, input_ids):
        hidden = self.forward_with_attention_mask(input_ids)
        # The paired HF wrapper constructs this tower without a BERT pooler.
        return CLIPTextModelOutput(hidden, hidden[:, 0])


class PairedClipModel(nn.Module):
    """Shared ChineseCLIP/MetaCLIP2 projection, normalization and output wiring."""

    def __init__(self, config, text_model):
        super().__init__()
        self.text_model = text_model
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
        if self.training:
            raise RuntimeError("These paired coverage models support inference only")
        vision_hidden, vision_pool = self.vision_model(pixel_values)
        text_output = self.text_model(input_ids)
        projected_images = self.visual_projection(vision_pool)
        projected_texts = self.text_projection(text_output.pooler_output)
        images = self.normalize(projected_images)
        texts = self.normalize(projected_texts)
        logits = self.matmul(texts, images.t()) * self.scale
        return {
            "logits_per_text": logits, "logits_per_image": logits.t(),
            "text_embeds": texts, "image_embeds": images,
            "text_model_output.last_hidden_state": text_output.last_hidden_state,
            "text_model_output.pooler_output": projected_texts,
            "vision_model_output.last_hidden_state": vision_hidden,
            "vision_model_output.pooler_output": projected_images,
        }


class ChineseCLIPModel(PairedClipModel):
    def __init__(self, config):
        super().__init__(config, _ChineseTextModel(config.text_config))


def build_from_config(config, device, dtype):
    if config.text_config.hidden_act != "gelu" or config.vision_config.hidden_act != "quick_gelu":
        raise ValueError("The example checkpoint uses exact GELU text and QuickGELU vision towers")
    if config.text_config.chunk_size_feed_forward or getattr(config.text_config, "is_decoder", False):
        raise ValueError("This case preserves the ordinary bidirectional, unchunked text encoder")
    return ChineseCLIPModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = dict(state_dict)
    for name, _ in model.named_parameters():
        if ".qkv." in name:
            mapped[name] = torch.cat([
                mapped.pop(name.replace(".qkv.", f".{projection}."))
                for projection in ("query", "key", "value")
            ])
    # Reuse the stable vision mapping and same-loaded-dtype scale exponential.
    load_clip_state(model, mapped, config)
