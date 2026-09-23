"""ColModernVBERT retrieval projection over the unchanged multimodal construction."""

from torch import nn
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import Linear
from .modernvbert import ModernVBertModel, setup_model, mapped_base, make_workloads


class ColModernVBertForRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vlm = ModernVBertModel(config.vlm_config)
        self.embedding_proj_layer = Linear(config.vlm_config.text_config.hidden_size, config.embedding_dim)
        self.normalize = L2Norm(dim=-1, eps=0)

    def forward(self, input_ids, pixel_values=None, attention_mask=None):
        hidden, image = self.vlm(input_ids, pixel_values, attention_mask)
        embeddings = self.normalize(self.embedding_proj_layer(hidden))
        if attention_mask is not None:
            embeddings = embeddings.masked_fill(~attention_mask[:, :, None].bool(), 0)
        outputs = {'embeddings': embeddings}
        if image is not None:
            outputs['image_hidden_states'] = image
        return outputs


def build_from_config(config, device, dtype):
    return setup_model(ColModernVBertForRetrieval(config), config.vlm_config, device, dtype)


def load_state_dict_into(model, state_dict, config):
    base, used = mapped_base(model.vlm, state_dict, 'vlm.')
    mapped = {'vlm.' + name: value for name, value in base.items()}
    for field in ('weight', 'bias'):
        name = 'embedding_proj_layer.' + field
        mapped[name] = state_dict[name]
        used.add(name)
    if used != set(state_dict):
        raise KeyError(f'Unmapped ColModernVBERT weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
