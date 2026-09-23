"""SAM3 image/text segmentation with the full CLIP text tower."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.clip_mlp import CLIPTextEmbeddings
from fastkernels.tasks.baseline.L2.sam3_text_attention import Sam3TextAttentionBlock

from .sam3_lite_text import Sam3LiteTextModel, VisionAttention, _source_key, make_workloads


class TextLayer(Sam3TextAttentionBlock):
    """Use the existing GELU block with an explicit causal/padding mask."""

    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_attention_heads,
                         config.intermediate_size / config.hidden_size)
        self.ln_1.eps = self.ln_2.eps = config.layer_norm_eps
        self.attn = DenseAttention(backend="sdpa")

    def _self_attention(self, hidden, attn_mask=None):
        batch, length, _ = hidden.shape
        shape = (batch, length, self.n_head, self.head_dim)
        q, k, v = (projection(hidden).reshape(shape)
                   for projection in (self.q_proj, self.k_proj, self.v_proj))
        output = self.attn(q, k, v, attn_mask=attn_mask)
        return self.out_proj(output.reshape(batch, length, -1).contiguous())


class FullTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_model = nn.Module()
        self.text_model.embeddings = CLIPTextEmbeddings(config)
        self.text_model.encoder = nn.Module()
        self.text_model.encoder.layers = nn.ModuleList(
            TextLayer(config) for _ in range(config.num_hidden_layers))
        self.text_model.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                                     promote_fp32=False)
        self.text_projection = Linear(config.hidden_size, config.projection_dim, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.text_model.embeddings(input_ids)
        length = input_ids.shape[1]
        blocked = torch.ones(length, length, device=hidden.device, dtype=torch.bool).triu(1)
        blocked = blocked[None, None].expand(input_ids.shape[0], 1, length, length)
        if attention_mask is not None:
            blocked = blocked | ~attention_mask[:, None, None, :].bool()
        mask = torch.zeros(blocked.shape, device=hidden.device, dtype=hidden.dtype)
        mask = mask.masked_fill(blocked, torch.finfo(hidden.dtype).min)
        for layer in self.text_model.encoder.layers:
            hidden = layer(hidden, mask)
        hidden = self.text_model.final_layer_norm(hidden)
        positions = (input_ids == self.config.eos_token_id).int().argmax(dim=-1)
        pooled = hidden[torch.arange(input_ids.shape[0], device=hidden.device), positions]
        # HF computes this projection even though SAM3 consumes the token features.
        projected = self.text_projection(pooled)
        return SimpleNamespace(last_hidden_state=hidden, text_embeds=projected)


def build_from_config(config, device, dtype):
    if config.text_config.hidden_act != "gelu":
        raise ValueError("SAM3 constructor text configuration requires exact GELU")
    model = Sam3LiteTextModel(config, text_encoder=FullTextEncoder).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if isinstance(module, VisionAttention):
            module.rotary_table = module.position_table(device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for key in model.state_dict():
        source = _source_key(key)
        if source.startswith("text_encoder."):
            source = source.replace(".emb.weight", ".weight")
            source = source.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
            source = source.replace(".mlp_fc1.", ".mlp.fc1.").replace(".mlp_fc2.", ".mlp.fc2.")
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                source = source.replace(f".{projection}.", f".self_attn.{projection}.")
        mapped[key] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f"Unmapped SAM3 state: {sorted(set(state_dict) - used)}")
    model.load_state_dict(mapped, strict=True)
