"""MobileCLIP RepMixer/transformer text encoder for SAM3 LiteText."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from ..patches.product_gate import ProductGate


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(1, 1, config.max_position_embeddings, config.hidden_size))
        self.interpolate = Interpolate()

    def forward(self, length):
        positions = self.position_embedding
        if length != positions.shape[2]:
            positions = self.interpolate(positions, size=(length, positions.shape[-1]), mode='bilinear')
        return positions.reshape(1, length, -1)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = PositionEmbedding(config)

    def forward(self, input_ids):
        hidden = self.token_embedding(input_ids)
        return hidden + self.position_embedding(input_ids.shape[1]).to(hidden.dtype)


class MobileOne(nn.Module):
    def __init__(self, width, kernel):
        super().__init__()
        self.batchnorm_skip, self.batchnorm_conv = BatchNorm2d(width), BatchNorm2d(width)
        self.conv = Conv2d(width, width, (1, kernel), padding=(0, kernel // 2), groups=width, bias=False)

    def forward(self, hidden):
        return self.batchnorm_conv(self.conv(hidden)) + self.batchnorm_skip(hidden)


class ConvMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1, self.fc2 = Conv2d(config.hidden_size, config.intermediate_size, 1), Conv2d(config.intermediate_size, config.hidden_size, 1)
        self.activation_fn = GELU()

    def forward(self, hidden):
        return self.fc2(self.activation_fn(self.fc1(hidden)))


class ConvFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.hidden_size, config.repmixer_kernel_size
        self.depthwise_conv = Conv2d(width, width, (1, kernel), padding=(0, kernel // 2), groups=width, bias=False)
        self.depthwise_batchnorm = BatchNorm2d(width)
        self.mlp = ConvMLP(config)

    def forward(self, hidden):
        return self.mlp(self.depthwise_batchnorm(self.depthwise_conv(hidden)))


class LayerScaledResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer_scale = nn.Parameter(torch.empty(width, 1, 1))
        self.multiply = ProductGate()

    def add_scaled(self, hidden, update):
        scale = self.layer_scale[None].expand_as(update)
        return hidden + self.multiply(torch.cat((scale, update), dim=-1))


class RepMixer(LayerScaledResidual):
    def __init__(self, config):
        super().__init__(config.hidden_size)
        self.reference_batchnorm = BatchNorm2d(config.hidden_size)
        self.mixer = MobileOne(config.hidden_size, config.repmixer_kernel_size)

    def forward(self, hidden):
        return self.add_scaled(hidden, self.mixer(hidden) - self.reference_batchnorm(hidden))


class RepMixerBlock(LayerScaledResidual):
    def __init__(self, config):
        super().__init__(config.hidden_size)
        self.token_mixer, self.conv_feed_forward = RepMixer(config), ConvFeedForward(config)

    def forward(self, hidden, attention_mask=None):
        hidden = self.token_mixer(hidden.transpose(1, 2).unsqueeze(2))
        return self.add_scaled(hidden, self.conv_feed_forward(hidden)).squeeze(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = [Linear(config.hidden_size, config.hidden_size) for _ in range(4)]
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, attention_mask):
        shape = (*hidden.shape[:2], self.heads, self.width)
        query, key, value = [op(hidden).reshape(shape) for op in (self.q_proj, self.k_proj, self.v_proj)]
        return self.out_proj(self.attention(query, key, value, attn_mask=attention_mask).reshape_as(hidden))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1, self.fc2 = Linear(config.hidden_size, config.intermediate_size), Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = GELU()

    def forward(self, hidden):
        return self.fc2(self.activation_fn(self.fc1(hidden)))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1, self.layer_norm2 = [LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False) for _ in range(2)]
        self.self_attn, self.mlp = Attention(config), MLP(config)

    def forward(self, hidden, attention_mask=None):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden), attention_mask)
        return hidden + self.mlp(self.layer_norm2(hidden))


class Sam3LiteTextTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.use_repmixer_blocks or config.hidden_act != 'gelu':
            raise ValueError('SAM3 LiteText selected checkpoint requires RepMixer end blocks and GELU')
        self.embeddings = Embeddings(config)
        ends = {0, config.num_hidden_layers - 1}
        self.layers = nn.ModuleList([RepMixerBlock(config) if index in ends else EncoderLayer(config)
                                     for index in range(config.num_hidden_layers)])
        self.final_layer_norm = LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        self.projection = Linear(config.hidden_size, config.projection_dim, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embeddings(input_ids)
        mask = None if attention_mask is None else attention_mask[:, None, None].bool()
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask)
        hidden = self.final_layer_norm(hidden)
        # EOT is the largest token ID in the selected tokenizer; argmax of
        # supplied integer token IDs is positional metadata, not activation math.
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), input_ids.argmax(dim=-1)]
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=self.projection(pooled))


TextEncoder = Sam3LiteTextTextModel


def load_state_dict_into(model, state_dict, config=None):
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        source = name.replace('.token_embedding.emb.weight', '.token_embedding.weight')
        mapped[name] = state_dict[source]
        if target.shape != mapped[name].shape:
            raise ValueError(f'SAM3 LiteText text weight shape mismatch: {name}')
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped SAM3 LiteText text weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
