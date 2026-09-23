"""Gemma4 assistant forward components, including ordered vocabulary selection."""
import torch
from torch import nn

from .gemma4 import TextModel
from ..patches.detector_topk import DetectorTopK
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR


class OrderedEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.centroids = Linear(c.text_config.hidden_size, c.num_centroids, bias=False)
        self.register_buffer('token_ordering', torch.zeros(c.text_config.vocab_size, dtype=torch.long))
        self.topk, self.mm, self.reduce = DetectorTopK(), BMM(), SegmentCSR()

    def forward(self, hidden, weight):
        c = self.c
        _, indices = self.topk(self.centroids(hidden), c.centroid_intermediate_top_k)
        selected = self.token_ordering.reshape(c.num_centroids, -1)[indices].flatten(-2)
        embeddings = weight[selected]
        logits = self.mm(hidden.unsqueeze(-2), embeddings.transpose(-1, -2)).squeeze(-2)
        flat = logits.flatten()
        minimum = self.reduce(flat, torch.tensor([0, flat.numel()], device=flat.device), 'min')[0]
        # The public implementation has this same host synchronization.
        output = hidden.new_full((*hidden.shape[:2], c.text_config.vocab_size), minimum.item()-1.0)
        return output.scatter_(-1, selected, logits)


class Assistant(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.model = TextModel(c.text_config)
        self.lm_head = Linear(c.text_config.hidden_size, c.text_config.vocab_size, bias=False)
        self.pre_projection = Linear(2*c.backbone_hidden_size, c.text_config.hidden_size, bias=False)
        self.post_projection = Linear(c.text_config.hidden_size, c.backbone_hidden_size, bias=False)
        self.masked_embedding = OrderedEmbedding(c)

    def forward(self, inputs_embeds, position_ids, shared_kv_states, attention_mask=None):
        x = self.pre_projection(inputs_embeds)
        shared = {name: tuple(t.transpose(1, 2) for t in pair) for name, pair in shared_kv_states.items()}
        masks = {}
        for name, (keys, _) in shared.items():
            length = keys.shape[1]
            valid = torch.ones(x.shape[0], length, dtype=torch.bool, device=x.device)
            if attention_mask is not None:
                valid = attention_mask[:, :length].bool() if name == 'full_attention' else attention_mask[:, -length:].bool()
            mask = valid[:, None, None, :].expand(-1, 1, x.shape[1], -1)
            if name == 'sliding_attention':
                # HF reverses the future-looking bidirectional window on KV.
                reverse_kv = torch.arange(length-1, -1, -1, device=x.device)
                query = torch.arange(x.shape[1], device=x.device)
                window = (reverse_kv[None]-query[:, None]).abs() <= self.c.text_config.sliding_window
                mask = mask & window[None, None]
            masks[name] = mask
        hidden, _ = self.model(x, None, position_ids, masks, shared=shared)
        return {'last_hidden_state': self.post_projection(hidden),
                'logits': self.masked_embedding(hidden, self.lm_head.weight)}


def build_from_config(config, device, dtype):
    return Assistant(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for key in model.state_dict():
        source = key.replace('.embed_tokens.emb.', '.embed_tokens.')
        mapped[key] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f'Unmapped assistant state: {sorted(set(state_dict)-used)}')
    model.load_state_dict(mapped, strict=True, assign=True)
