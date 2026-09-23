"""TVP video grounding from existing ResNet, BERT, pooling and affine operations."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from .resnet import _Embeddings, _Encoder
from ..patches.input_bias_conv2d import InputBiasConv2d
from ..runner import Workload


class FramePrompt(nn.Module):
    def __init__(self, config):
        super().__init__()
        frames, size, width = config.num_frames, config.max_img_size, config.visual_prompt_size
        self.frames = frames
        self.base_size = size - 2 * width
        self.pad_up = nn.Parameter(torch.empty(1, frames, 3, width, size))
        self.pad_down = nn.Parameter(torch.empty(1, frames, 3, width, size))
        self.pad_left = nn.Parameter(torch.empty(1, frames, 3, self.base_size, width))
        self.pad_right = nn.Parameter(torch.empty(1, frames, 3, self.base_size, width))

    def forward(self, pixels):
        # HF's selected framepad/replace path multiplies by an all-one mask.
        # The actual learned border is assembled without changing image dimensions.
        base = pixels.new_zeros(1, self.frames, 3, self.base_size, self.base_size)
        prompt = torch.cat((self.pad_left, base, self.pad_right), dim=-1)
        prompt = torch.cat((self.pad_up, prompt, self.pad_down), dim=-2)
        return prompt.expand(pixels.shape[0] // self.frames, -1, -1, -1, -1).flatten(0, 1).to(pixels.dtype)


class VisualEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        self.col_position_embeddings = Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(1, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.temporal_mean = AvgPool2d((config.num_frames, 1))

    def forward(self, grid):
        batch, frames, channels, height, width = grid.shape
        temporal = grid.permute(0, 3, 4, 2, 1).reshape(-1, channels, frames, 1)
        hidden = self.temporal_mean(temporal).reshape(batch, height, width, channels)
        rows = self.row_position_embeddings(torch.arange(height, device=grid.device))
        columns = self.col_position_embeddings(torch.arange(width, device=grid.device))
        hidden = hidden + (rows[None, :, None] + columns[None, None, :])
        hidden = hidden.reshape(batch, height * width, channels)
        token_types = self.token_type_embeddings(torch.zeros(hidden.shape[:2], device=grid.device, dtype=torch.long))
        return self.layer_norm(hidden + token_types)


class TvpForVideoGrounding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_embedder = _Embeddings(config.backbone_config)
        stem = self.backbone_embedder.embedder
        stem.convolution = InputBiasConv2d(stem.convolution, FramePrompt(config))
        self.backbone_encoder = _Encoder(config.backbone_config)
        self.grid_encoder_conv = Conv2d(config.backbone_config.hidden_sizes[-1], config.hidden_size, 3, padding=1, bias=False)
        self.grid_pool, self.relu = MaxPool2d(2, stride=2), ReLU()
        self.embeddings = BertEmbeddings(config)
        self.visual_embeddings = VisualEmbedding(config)
        self.text_prompt = nn.Parameter(torch.empty(1, 10, config.hidden_size))
        self.encoder = BertEncoder(config)
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.tanh = Tanh()
        self.layer_0 = Linear(config.hidden_size, config.hidden_size * 2)
        self.layer_1 = Linear(config.hidden_size * 2, 2)
        self.sigmoid = Sigmoid()

    def forward(self, input_ids, pixel_values, attention_mask=None):
        batch, frames = pixel_values.shape[:2]
        pixels = pixel_values.flatten(0, 1)
        grid = self.backbone_encoder(self.backbone_embedder(pixels))
        grid = self.relu(self.grid_pool(self.grid_encoder_conv(grid)))
        visual = self.visual_embeddings(grid.reshape(batch, frames, *grid.shape[1:]))
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        text = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        hidden = torch.cat((self.text_prompt.expand(batch, -1, -1), text, visual), dim=1)
        mask = None
        if attention_mask is not None:
            mask = torch.cat((attention_mask.new_ones(batch, 10), attention_mask,
                              attention_mask.new_ones(batch, visual.shape[1])), dim=-1)[:, None, None].bool()
        hidden = self.encoder.forward_with_attention_mask(hidden, mask)
        pooled = self.tanh(self.pooler(hidden[:, 0]))
        return {"logits": self.sigmoid(self.layer_1(self.relu(self.layer_0(pooled))))}


def build_from_config(config, device, dtype):
    if (config.visual_prompter_type != 'framepad' or config.visual_prompter_apply != 'replace'
            or config.hidden_act != 'gelu' or config.backbone_config.layer_type != 'bottleneck'):
        raise ValueError('TVP case preserves the documented framepad/replace, GELU and ResNet bottleneck path')
    return TvpForVideoGrounding(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if source.startswith('backbone_embedder.embedder.convolution.input_bias.'):
            source = source.replace('backbone_embedder.embedder.convolution.input_bias.', 'visual_prompter.')
        elif source.startswith('backbone_embedder.'):
            source = source.replace('.convolution.convolution.', '.convolution.')
            source = source.replace('backbone_embedder.', 'vision_model.backbone.embedder.', 1)
        elif source.startswith('backbone_encoder.'):
            source = source.replace('backbone_encoder.', 'vision_model.backbone.encoder.', 1)
        elif source.startswith('grid_encoder_conv.'):
            source = 'vision_model.' + source
        elif source.startswith('embeddings.'):
            source = source.replace('.LayerNorm.', '.layer_norm.')
        elif source.startswith('pooler.'):
            source = source.replace('pooler.', 'pooler.dense.', 1)
        elif source.startswith(('layer_0.', 'layer_1.')):
            source = 'video_grounding_head.' + source
        elif source.startswith('encoder.'):
            source = source.replace('.attention.output.dense.', '.attention.dense.')
            source = source.replace('.attention.output.LayerNorm.', '.attention.layer_norm.')
            source = source.replace('.output.LayerNorm.', '.output.layer_norm.')
        if not source.startswith('video_grounding_head.'):
            source = 'model.' + source
        if '.attention.self.qkv.' in source:
            sources = [source.replace('.self.qkv.', f'.{part}.') for part in ('query', 'key', 'value')]
            mapped[name] = torch.cat([state_dict[item] for item in sources])
            used.update(sources)
        else:
            mapped[name] = state_dict[source]
            used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped TVP weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
