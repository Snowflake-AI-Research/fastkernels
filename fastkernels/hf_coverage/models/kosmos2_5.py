"""Kosmos2.5 patch-grid vision and segmented language composition."""

from types import SimpleNamespace
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L2.t5_dense import T5DenseGatedActDense
from ..patches.codec_top1 import CodecTop1
from .kosmos2 import TextModel, ImageProjection, make_workloads


class PatchValidity(nn.Module):
    """Execute the native sum-then-nonzero predicate through linear-work operations."""

    def __init__(self):
        super().__init__()
        self.reduce, self.relu, self.top1 = SegmentCSR(), ReLU(), CodecTop1()

    def forward(self, patches):
        width = patches.shape[-1]
        offsets = torch.arange(patches.numel() // width + 1, device=patches.device) * width
        sums = self.reduce(patches.float().flatten(), offsets, reduce='sum').to(patches.dtype)
        magnitudes = self.relu(sums) + self.relu(-sums)
        return self.top1(torch.stack((torch.zeros_like(sums), magnitudes), dim=-1)).view(patches.shape[:-1]).bool()


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.head_dim
        self.query, self.key, self.value = [Linear(config.hidden_size, self.heads * self.width, bias=False) for _ in range(3)]
        self.output = Linear(self.heads * self.width, config.hidden_size, bias=False)
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, mask):
        batch, length = hidden.shape[:2]
        query, key, value = [op(hidden).view(batch, length, self.heads, self.width) for op in (self.query, self.key, self.value)]
        return self.output(self.attention(query, key, value, attn_mask=mask).reshape(batch, length, -1))


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = VisionAttention(config)
        self.pre_attention_layer_norm = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.pre_mlp_layer_norm = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.mlp = T5DenseGatedActDense(SimpleNamespace(d_model=config.hidden_size,
            d_ff=config.intermediate_size, dense_act_fn=config.dense_act_fn))

    def forward(self, hidden, mask):
        hidden = hidden + self.attention(self.pre_attention_layer_norm(hidden), mask)
        return self.mlp(self.pre_mlp_layer_norm(hidden)) + hidden


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_projection = Linear(config.patch_embed_hidden_size, config.hidden_size)
        self.row_embedder, self.column_embedder = [Embedding(config.max_num_patches, config.hidden_size) for _ in range(2)]
        self.layers = nn.ModuleList([VisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.validity = PatchValidity()

    def forward(self, patches):
        valid = self.validity(patches)
        rows, columns = patches[..., 0].long(), patches[..., 1].long()
        hidden = self.patch_projection(patches[..., 2:]) + self.row_embedder(rows) + self.column_embedder(columns)
        mask = hidden.new_zeros((patches.shape[0], 1, patches.shape[1], patches.shape[1]))
        mask.masked_fill_(~valid[:, None, None], torch.finfo(hidden.dtype).min)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.layernorm(hidden)


class Kosmos2_5ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = VisionModel(config.vision_config)
        self.image_to_text_projection = ImageProjection(config, variant25=True)
        self.text_model = TextModel(config.text_config, variant25=True)
        self.lm_head = Linear(config.text_config.embed_dim, config.text_config.vocab_size, bias=False)
        self.normalize = L2Norm()

    def forward(self, input_ids, flattened_patches, image_embeds_position_mask):
        vision = self.vision_model(flattened_patches)
        images = self.image_to_text_projection(self.normalize(vision))
        hidden, outputs = self.text_model(input_ids, images, image_embeds_position_mask)
        return {'logits': self.lm_head(hidden), 'image_embeds': images,
                'vision_model_output.last_hidden_state': vision, **outputs}


def build_from_config(config, device, dtype):
    if config.text_config.activation_function != 'gelu' or config.vision_config.dense_act_fn != 'gelu_new':
        raise ValueError('Kosmos2.5 selected checkpoint requires GELU text and gated GELU-new vision')
    return Kosmos2_5ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, parameter in model.state_dict().items():
        source = name.replace('.emb.weight', '.weight')
        if source.startswith(('vision_model.patch_projection.', 'vision_model.row_embedder.', 'vision_model.column_embedder.')):
            source = source.replace('vision_model.', 'vision_model.embeddings.', 1)
        elif source.startswith('vision_model.layers.'):
            source = source.replace('vision_model.layers.', 'vision_model.encoder.layer.', 1)
        elif source.startswith('text_model.'):
            source = source.replace('text_model.', 'text_model.model.', 1)
        elif source.startswith('lm_head.'):
            source = 'text_model.' + source
        sources = [source.replace('wi.weight', field + '.weight') for field in ('wi_0', 'wi_1')] if source.endswith('.wi.weight') else [source]
        mapped[name] = torch.cat([state_dict[s] for s in sources]) if len(sources) > 1 else state_dict[source]
        if mapped[name].shape != parameter.shape:
            raise ValueError(f'Kosmos2.5 weight shape differs: {name}')
        used.update(sources)
    if used != set(state_dict):
        raise KeyError(f'Unmapped Kosmos2.5 weights: {sorted(set(state_dict) - used)}')
    if config.text_config.tie_word_embeddings:
        if not torch.equal(mapped['lm_head.weight'], mapped['text_model.embed_tokens.emb.weight']):
            raise ValueError('Kosmos2.5 tied language weights disagree')
        model.lm_head.weight = model.text_model.embed_tokens.emb.weight
    model.load_state_dict(mapped, strict=True)
