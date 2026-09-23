"""Kosmos2's literal pinned-HF SDPA path; projection causality remains under review."""

import math
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from .clip import ClipVisionModel
from .m2m_100 import sinusoidal_table
from ..runner import Workload


def norm(width, eps):
    return LayerNorm(width, eps=eps, promote_fp32=False)


class TextAttention(nn.Module):
    def __init__(self, config, *, inner_norm=False, prescale=False):
        super().__init__()
        self.heads, self.width = config.attention_heads, config.embed_dim // config.attention_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = [Linear(config.embed_dim, config.embed_dim) for _ in range(4)]
        self.inner_attn_ln = norm(config.embed_dim, config.layer_norm_eps) if inner_norm else None
        self.attention = DenseAttention(backend='sdpa')
        self.prescale = prescale

    def forward(self, hidden, memory=None, *, causal=True, mask=None):
        memory = hidden if memory is None else memory
        batch, length = hidden.shape[:2]
        query = self.q_proj(hidden).view(batch, length, self.heads, self.width)
        key, value = [op(memory).view(batch, memory.shape[1], self.heads, self.width) for op in (self.k_proj, self.v_proj)]
        if self.prescale:
            query = query * self.width**-0.5
        context = self.attention(query, key, value, causal=causal, attn_mask=mask,
                                 softmax_scale=1.0 if self.prescale else None).reshape(batch, length, -1)
        if self.inner_attn_ln is not None:
            context = self.inner_attn_ln(context)
        return self.out_proj(context), (key.transpose(1, 2), value.transpose(1, 2))


class TextFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1, self.fc2 = Linear(config.embed_dim, config.ffn_dim), Linear(config.ffn_dim, config.embed_dim)
        self.ffn_layernorm = norm(config.ffn_dim, config.layer_norm_eps)
        self.activation = GELU()

    def forward(self, hidden):
        return self.fc2(self.ffn_layernorm(self.activation(self.fc1(hidden))))


class TextLayer(nn.Module):
    def __init__(self, config, variant25):
        super().__init__()
        self.self_attn = TextAttention(config, inner_norm=not variant25, prescale=variant25)
        self.self_attn_layer_norm = norm(config.embed_dim, config.layer_norm_eps)
        self.final_layer_norm = norm(config.embed_dim, config.layer_norm_eps)
        self.ffn = TextFFN(config)

    def forward(self, hidden, mask):
        context, cache = self.self_attn(self.self_attn_layer_norm(hidden), causal=mask is None, mask=mask)
        hidden = hidden + context
        return hidden + self.ffn(self.final_layer_norm(hidden)), cache


class TextModel(nn.Module):
    def __init__(self, config, variant25=False):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        self.segment_emb = Embedding(2, config.embed_dim) if variant25 else None
        self.layers = nn.ModuleList([TextLayer(config, variant25) for _ in range(config.layers)])
        self.layer_norm = norm(config.embed_dim, config.layer_norm_eps)
        self.register_buffer('positions', sinusoidal_table(config.max_position_embeddings + 2, config.embed_dim,
                                                          config.pad_token_id), persistent=False)
        self.scale = math.sqrt(config.embed_dim) if config.scale_embedding else 1.0
        self.padding_idx, self.use_cache, self.variant25 = config.pad_token_id, config.use_cache, variant25

    def forward(self, input_ids, images, image_mask):
        hidden = self.embed_tokens(input_ids)
        hidden[image_mask] = images.reshape(-1, images.shape[-1])
        valid = input_ids.ne(self.padding_idx).int()
        positions = (valid.cumsum(-1).to(valid.dtype) * valid).long() + self.padding_idx
        position_values = self.positions[positions]
        if self.segment_emb is not None:
            position_values = position_values + self.segment_emb(image_mask.long())
        hidden = hidden * self.scale + position_values
        mask = None
        if not self.variant25:
            length = input_ids.shape[1]
            order = torch.arange(length, device=hidden.device)
            mask = hidden.new_zeros((1, 1, length, length))
            mask.masked_fill_(order[None] > order[:, None], torch.finfo(hidden.dtype).min)
        outputs = {}
        for index, layer in enumerate(self.layers):
            hidden, (key, value) = layer(hidden, mask)
            if self.use_cache:
                outputs[f'past_key_values.{index}.key'], outputs[f'past_key_values.{index}.value'] = key, value
        return self.layer_norm(hidden), outputs


class ImageProjection(nn.Module):
    def __init__(self, config, variant25=False):
        super().__init__()
        self.dense = Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        self.latent_query = nn.Parameter(torch.empty(config.latent_query_num, config.text_config.embed_dim))
        self.x_attn = TextAttention(config.text_config, prescale=variant25)
        self.causal = not variant25

    def forward(self, features):
        projected = self.dense(features)
        query = self.latent_query[None].expand(features.shape[0], -1, -1)
        # Kosmos2's selected native SDPA path is causal here; it differs from eager/author code.
        return self.x_attn(query, torch.cat((projected, query), dim=1), causal=self.causal)[0]


class Kosmos2ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = ClipVisionModel(config.vision_config)
        self.text_model = TextModel(config.text_config)
        self.image_to_text_projection = ImageProjection(config)
        self.lm_head = Linear(config.text_config.embed_dim, config.text_config.vocab_size, bias=False)
        self.normalize = L2Norm()
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
            if isinstance(module, DenseAttention):
                # Native SDPA selects cuDNN on the checked BF16 workload. Keep
                # that rounding when library imports disable its global flag.
                module.use_cudnn_kernel = True

    def forward(self, input_ids, pixel_values, image_embeds_position_mask):
        vision, pooled = self.vision_model(pixel_values)
        images = self.image_to_text_projection(self.normalize(self.vision_model.post_layernorm(vision)))
        hidden, outputs = self.text_model(input_ids, images, image_embeds_position_mask)
        return {'logits': self.lm_head(hidden), 'image_embeds': images,
                'vision_model_output.last_hidden_state': vision, 'vision_model_output.pooler_output': pooled, **outputs}


def build_from_config(config, device, dtype):
    if config.text_config.activation_function != 'gelu' or config.text_config.add_cross_attention:
        raise ValueError('Kosmos2 selected default requires GELU without text cross-attention')
    if config.vision_config.hidden_act != 'quick_gelu':
        raise ValueError('Kosmos2 selected checkpoint requires QuickGELU vision')
    return Kosmos2ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, parameter in model.state_dict().items():
        source = name.replace('.emb.weight', '.weight')
        if name.startswith('vision_model.'):
            source = source.replace('vision_model.', 'vision_model.model.', 1)
            source = source.replace('embeddings.patch_embedding.proj.', 'embeddings.patch_embedding.')
            source = source.replace('.ln_1.', '.layer_norm1.').replace('.ln_2.', '.layer_norm2.')
            source = source.replace('.mlp_fc1.', '.mlp.fc1.').replace('.mlp_fc2.', '.mlp.fc2.')
            for projection in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
                source = source.replace('.' + projection + '.', '.self_attn.' + projection + '.')
        elif name.startswith('text_model.'):
            source = source.replace('text_model.', 'text_model.model.', 1)
        elif name.startswith('lm_head.'):
            source = 'text_model.' + source
        mapped[name] = state_dict[source]
        if mapped[name].shape != parameter.shape:
            raise ValueError(f'Kosmos2 weight shape differs: {name}')
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped Kosmos2 weights: {sorted(set(state_dict) - used)}')
    if config.text_config.tie_word_embeddings:
        if not torch.equal(mapped['lm_head.weight'], mapped['text_model.embed_tokens.emb.weight']):
            raise ValueError('Kosmos2 tied language weights disagree')
        model.lm_head.weight = model.text_model.embed_tokens.emb.weight
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
