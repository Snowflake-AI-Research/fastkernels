"""Paired GroupViT with hard grouping and complete default tower outputs.

Assignment reuses existing audit patches CodecTop1 and grouped-topk _normalize.
These internal carriers have no standalone baseline optimization interface.
"""
import torch
import triton
from torch import nn
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import OneHot
from ..patches.codec_top1 import CodecTop1
from ..patches.grouped_topk_normalization import _normalize
from ..runner import Workload


def _norm(config):
    return LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)


class MLP(nn.Module):
    def __init__(self, config, input_size=None, intermediate_size=None, output_size=None, mixer=False):
        super().__init__()
        self.fc1 = Linear(input_size or config.hidden_size, intermediate_size or config.intermediate_size)
        self.fc2 = Linear(intermediate_size or config.intermediate_size, output_size or config.hidden_size)
        self.activation = QuickGELU() if config.hidden_act == 'quick_gelu' else GELU()
        self.mixer = mixer

    def forward(self, hidden):
        if self.mixer:
            hidden = hidden.transpose(1, 2)
        hidden = self.fc2(self.activation(self.fc1(hidden)))
        return hidden.transpose(1, 2) if self.mixer else hidden


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads, self.head_dim = config.num_attention_heads, width // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj, self.k_proj, self.v_proj = (Linear(width, width) for _ in range(3))
        self.out_proj = Linear(width, width)
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, key=None, mask=None):
        batch, length, width = hidden.shape
        source = hidden if key is None else key

        def split(x):
            return x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2).contiguous().view(
                batch * self.heads, -1, self.head_dim)

        # Preserve HF's query scaling before the BF16 score GEMM.
        query = split(self.q_proj(hidden) * self.scale)
        keys, values = split(self.k_proj(source)), split(self.v_proj(source))
        scores = self.bmm(query, keys.transpose(1, 2))
        if mask is not None:
            scores = (scores.view(batch, self.heads, length, -1) + mask).view(batch * self.heads, length, -1)
        output = self.bmm(self.softmax(scores), values)
        output = output.view(batch, self.heads, length, self.head_dim).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.mlp = Attention(config), MLP(config)
        self.layer_norm1, self.layer_norm2 = _norm(config), _norm(config)

    def forward(self, hidden, mask=None):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden), mask=mask)
        return hidden + self.mlp(self.layer_norm2(hidden))


class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn, self.mlp = Attention(config), MLP(config)
        self.norm2, self.norm_post = _norm(config), _norm(config)

    def forward(self, query, key):
        hidden = query + self.attn(query, key)
        return self.norm_post(hidden + self.mlp(self.norm2(hidden)))


class AssignAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.q_proj, self.k_proj, self.v_proj, self.proj = (Linear(width, width) for _ in range(4))
        self.scale, self.epsilon = width ** -0.5, config.assign_eps
        self.bmm, self.softmax = BatchMatMul(), Softmax(dim=-2)
        self.top1, self.one_hot = CodecTop1(), OneHot()

    def forward(self, query, key):
        values = self.v_proj(key)
        scores = self.bmm(self.q_proj(query), self.k_proj(key).transpose(-1, -2)) * self.scale
        probabilities = self.softmax(scores)
        batch, groups, tokens = scores.shape
        # Select actual rounded probabilities: raw logits can break HF ties.
        ids = self.top1(probabilities.transpose(1, 2).contiguous())
        hard = self.one_hot(ids, groups).transpose(1, 2).to(scores.dtype).contiguous()
        rows = hard.view(batch * groups, tokens)
        normalized = torch.empty_like(rows)
        # Default 196/64 tokens keep binary count+1 exact in BF16. This is an
        # unchanged existing internal patch kernel, with executed launch cost.
        _normalize[(rows.shape[0],)](rows, normalized, tokens, triton.next_power_of_2(tokens),
                                   self.epsilon, 0.0, 1.0)
        return self.proj(self.bmm(normalized.view_as(hard), values))


class TokenAssign(nn.Module):
    def __init__(self, config, groups, output_groups):
        super().__init__()
        ratio = config.assign_mlp_ratio
        ratios = ratio if isinstance(ratio, (list, tuple)) else (ratio, ratio)
        self.norm_tokens, self.norm_x = _norm(config), _norm(config)
        self.mlp_inter = MLP(config, groups, int(ratios[0] * config.hidden_size), output_groups, mixer=True)
        self.norm_post_tokens = _norm(config)
        self.pre_assign_attn = CrossAttentionLayer(config)
        self.assign = AssignAttention(config)
        self.norm_new_x = _norm(config)
        self.mlp_channels = MLP(config, intermediate_size=int(ratios[1] * config.hidden_size))

    def forward(self, image_tokens, group_tokens):
        image_tokens = self.norm_x(image_tokens)
        projected = self.norm_post_tokens(self.mlp_inter(self.norm_tokens(group_tokens)))
        projected = self.pre_assign_attn(projected, image_tokens)
        hidden = self.assign(projected, image_tokens) + projected
        return hidden + self.mlp_channels(self.norm_new_x(hidden))


class Stage(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        groups = config.num_group_tokens[index]
        previous = config.num_output_groups[index - 1] if index else 0
        self.groups = groups
        self.group_token = nn.Parameter(torch.empty(1, groups, config.hidden_size)) if groups else None
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.depths[index]))
        self.downsample = TokenAssign(config, groups, config.num_output_groups[index]) if groups else None
        self.group_projector = nn.Sequential(
            _norm(config), MLP(config, previous, config.hidden_size // 2, groups, mixer=True)
        ) if previous and groups else None

    def forward(self, hidden, previous):
        group_tokens = None
        if self.groups:
            group_tokens = self.group_token.expand(hidden.shape[0], -1, -1)
            if self.group_projector is not None:
                group_tokens = group_tokens + self.group_projector(previous)
            hidden = torch.cat((hidden, group_tokens), dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        if self.groups:
            hidden, group_tokens = hidden[:, :-self.groups], hidden[:, -self.groups:]
            hidden = self.downsample(hidden, group_tokens)
        return hidden, group_tokens


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = Conv2d(config.num_channels, config.hidden_size,
                                                 config.patch_size, stride=config.patch_size)
        self.position_embeddings = nn.Parameter(torch.empty(
            1, (config.image_size // config.patch_size) ** 2, config.hidden_size))
        self.layernorm = _norm(config)

    def forward(self, pixels):
        hidden = self.patch_embeddings.projection(pixels).flatten(2).transpose(1, 2)
        return self.layernorm(hidden) + self.position_embeddings


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList(Stage(config, index) for index in range(len(config.depths)))
        self.layernorm, self.pool = _norm(config), GlobalAvgPool2d()

    def forward(self, pixels):
        hidden, groups = self.embeddings(pixels), None
        for stage in self.encoder.stages:
            hidden, groups = stage(hidden, groups)
        hidden = self.layernorm(hidden)
        return hidden, self.pool(hidden.transpose(1, 2).unsqueeze(-1))


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eos_token_id = config.eos_token_id
        self.embeddings = nn.Module()
        self.embeddings.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_layer_norm = _norm(config)

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)[None]
        hidden = self.embeddings.token_embedding(input_ids) + self.embeddings.position_embedding(positions)
        allowed = positions[:, :, None] >= positions[:, None, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, :].bool()
        mask = torch.zeros(batch, 1, length, length, device=hidden.device, dtype=hidden.dtype)
        mask.masked_fill_(~allowed[:, None], torch.finfo(hidden.dtype).min)
        for layer in self.encoder.layers:
            hidden = layer(hidden, mask)
        hidden = self.final_layer_norm(hidden)
        # Token IDs are position metadata; EOS2 preserves highest-ID pooling.
        pool_ids = input_ids.int() if self.eos_token_id == 2 else (input_ids == self.eos_token_id).int()
        pooled = hidden[torch.arange(batch, device=hidden.device), pool_ids.argmax(-1)]
        return hidden, pooled


class GroupViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model, self.vision_model = TextModel(config.text_config), VisionModel(config.vision_config)
        for name, width in (('text_projection', config.text_config.hidden_size),
                            ('visual_projection', config.vision_config.hidden_size)):
            setattr(self, name, nn.Sequential(Linear(width, config.projection_intermediate_dim),
                    BatchNorm2d(config.projection_intermediate_dim), ReLU(),
                    Linear(config.projection_intermediate_dim, config.projection_dim)))
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.register_buffer('scale', torch.empty(()), persistent=False)
        self.normalize, self.bmm = L2Norm(eps=0), BMM()

    def forward(self, input_ids, pixel_values, attention_mask=None):
        if self.training:
            raise RuntimeError('GroupViT coverage supports inference only')
        vision_hidden, vision_pool = self.vision_model(pixel_values)
        text_hidden, text_pool = self.text_model(input_ids, attention_mask)
        images, texts = self.normalize(self.visual_projection(vision_pool)), self.normalize(self.text_projection(text_pool))
        logits = self.bmm(texts, images.t()) * self.scale
        return {'logits_per_text': logits, 'logits_per_image': logits.t(), 'text_embeds': texts,
                'image_embeds': images, 'text_model_output.last_hidden_state': text_hidden,
                'text_model_output.pooler_output': text_pool, 'vision_model_output.last_hidden_state': vision_hidden,
                'vision_model_output.pooler_output': vision_pool}


def build_from_config(config, device, dtype):
    if config.output_segmentation or config.output_attentions or config.output_hidden_states:
        raise ValueError('This case preserves the checkpoint default paired outputs')
    if config.text_config.hidden_act != 'quick_gelu' or config.vision_config.hidden_act != 'gelu':
        raise ValueError('GroupViT requires checkpoint QuickGELU text / GELU vision')
    vision = config.vision_config
    if vision.assign_eps != 1 or (vision.image_size // vision.patch_size) ** 2 > 254:
        raise ValueError('Grouping requires exact BF16 binary counts and count+1')
    return GroupViTModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped = {name: state_dict[name.replace('.emb.weight', '.weight')] for name in model.state_dict()}
    expected = {name.replace('.emb.weight', '.weight') for name in model.state_dict()}
    if expected != set(state_dict):
        raise KeyError(f'Unmapped GroupViT state: {sorted(set(state_dict) - expected)}')
    model.load_state_dict(mapped, strict=True)
    model.scale.copy_(model.logit_scale.exp())


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
