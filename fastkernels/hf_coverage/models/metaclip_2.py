"""MetaCLIP2's causal text tower with first-EOS pooling and paired outputs."""

import torch

from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModel, CLIPTextModelOutput

from .chinese_clip import PairedClipModel
from .clip import configure_encoder, load_state_dict_into, make_workloads


class _MetaTextModel(CLIPTextModel):
    def __init__(self, config):
        super().__init__(config)
        configure_encoder(self.text_model.encoder, config)

    def forward(self, input_ids):
        transformer = self.text_model
        hidden = transformer.embeddings(input_ids)
        causal_mask = transformer._make_causal_mask(input_ids.shape, hidden.dtype, hidden.device)
        hidden = transformer.final_layer_norm(transformer.encoder(hidden, attention_mask=causal_mask))
        # Pinned MetaCLIP2 always selects the first EOS, including EOS id 2.
        positions = (input_ids.to(dtype=torch.int) == transformer.eos_token_id).int().argmax(dim=-1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return CLIPTextModelOutput(hidden, pooled)


class MetaClip2Model(PairedClipModel):
    def __init__(self, config):
        super().__init__(config, _MetaTextModel(config.text_config))


def build_from_config(config, device, dtype):
    if any(tower.hidden_act != "quick_gelu" for tower in (config.text_config, config.vision_config)):
        raise ValueError("The example checkpoint uses QuickGELU in both towers")
    if not isinstance(config.text_config.eos_token_id, int):
        raise ValueError("This case preserves the checkpoint's single configured EOS token")
    return MetaClip2Model(config).to(device=device, dtype=dtype).eval()
