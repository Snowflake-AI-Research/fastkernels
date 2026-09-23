"""CLAP's unfused spectrogram and text towers assembled from existing operations."""

from types import SimpleNamespace
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.xlm_roberta_model import XLMRobertaModel
from fastkernels.tasks.baseline.L3.swinv2_block import window_partition, window_reverse
from .swin import _Stage, _WindowBlock
from ..runner import Workload


class RoundedTextAttention(nn.Module):
    """HF CLAP exposes a BF16 score boundary before its FP32 softmax."""

    def __init__(self):
        super().__init__()
        self.matmul, self.softmax = BMM(), Softmax()

    def forward(self, q, k, v, causal=False, attn_mask=None):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = self.matmul(q, k.transpose(-1, -2)) * q.shape[-1]**-0.5
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        probability = self.softmax(scores.float()).to(q.dtype)
        return self.matmul(probability, v).transpose(1, 2)


class RoundedWindowBlock(_WindowBlock):
    """Retain the native score, relative-bias, and shifted-mask rounding order."""

    def __init__(self, *args):
        super().__init__(*args)
        self.matmul, self.softmax = BMM(), Softmax()

    def forward(self, hidden):
        batch, height, width, channels = hidden.shape
        if self.shift:
            hidden = torch.roll(hidden, (-self.shift, -self.shift), (1, 2))
        windows = window_partition(hidden, (self.window, self.window)).reshape(-1, self.window**2, channels)
        attention = self.block.attn
        qkv = attention.qkv(self.block.norm1(windows)).reshape(
            windows.shape[0], windows.shape[1], 3, attention.num_heads, attention.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = self.matmul(q, k.transpose(-1, -2)) / attention.head_dim**0.5
        bias = self.relative_position_bias_table[self.relative_position_index.flatten()]
        bias = bias.reshape(self.window**2, self.window**2, -1).permute(2, 0, 1)
        scores = scores + bias[None]
        if self.shift_mask is not None:
            scores = scores.view(batch, self.shift_mask.shape[0], attention.num_heads, self.window**2, self.window**2)
            scores = scores + self.shift_mask[None, :, None]
            scores = scores.flatten(0, 1)
        context = self.matmul(self.softmax(scores), v).transpose(1, 2).reshape_as(windows)
        windows = windows + attention.proj(context)
        windows = windows + self.block.mlp(self.block.norm2(windows))
        hidden = window_reverse(windows.reshape(-1, self.window, self.window, channels),
                                (self.window, self.window), (height, width))
        return torch.roll(hidden, (self.shift, self.shift), (1, 2)) if self.shift else hidden


class AudioTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spec_size, self.freq_ratio = config.spec_size, config.spec_size // config.num_mel_bins
        self.patch_stride = config.patch_stride
        self.batch_norm = BatchNorm2d(config.num_mel_bins)
        self.resize = Interpolate()
        stride = tuple(config.patch_stride)
        padding = tuple((config.patch_size - value) // 2 for value in stride)
        self.patch_conv = Conv2d(config.patch_embed_input_channels, config.patch_embeds_hidden_size,
                                 config.patch_size, stride=stride, padding=padding)
        self.patch_norm = LayerNorm(config.patch_embeds_hidden_size, promote_fp32=False)
        swin = SimpleNamespace(embed_dim=config.patch_embeds_hidden_size, num_heads=config.num_attention_heads,
                               window_size=config.window_size, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias,
                               layer_norm_eps=config.layer_norm_eps, attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                               hidden_dropout_prob=config.hidden_dropout_prob, depths=config.depths)
        self.layers = nn.ModuleList([_Stage(swin, index, config.spec_size // stride[0] // 2**index)
                                     for index in range(len(config.depths))])
        for index, layer in enumerate(self.layers):
            resolution = config.spec_size // stride[0] // 2**index
            layer.blocks = nn.ModuleList([RoundedWindowBlock(swin, index, block, resolution)
                                          for block in range(config.depths[index])])
        self.norm = LayerNorm(config.hidden_size, promote_fp32=False)
        self.pool = GlobalAvgPool2d(keepdim=False)

    def forward(self, input_features):
        values = self.batch_norm(input_features.transpose(1, 3)).transpose(1, 3)
        batch, channels, time, frequency = values.shape
        target_time, target_frequency = self.spec_size * self.freq_ratio, self.spec_size // self.freq_ratio
        if time < target_time:
            values = self.resize(values, size=(target_time, frequency), mode='bicubic', align_corners=True)
        if frequency < target_frequency:
            values = self.resize(values, size=(time, target_frequency), mode='bicubic', align_corners=True)
        time, frequency = values.shape[-2:]
        values = values.reshape(batch, channels * self.freq_ratio, time // self.freq_ratio, frequency)
        values = values.permute(0, 1, 3, 2).reshape(batch, channels, frequency * self.freq_ratio, time // self.freq_ratio)
        hidden = self.patch_norm(self.patch_conv(values).permute(0, 2, 3, 1))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden).permute(0, 3, 1, 2).contiguous()
        batch, width, frequency, time = hidden.shape
        bins = frequency // self.freq_ratio
        hidden = hidden.reshape(batch, width, frequency // bins, bins, time)
        hidden = hidden.permute(0, 1, 3, 2, 4).reshape(batch, width, bins, -1)
        return hidden, self.pool(hidden)


class Projection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = Linear(config.hidden_size, config.projection_dim)
        self.activation = ReLU()
        self.linear2 = Linear(config.projection_dim, config.projection_dim)

    def forward(self, hidden):
        return self.linear2(self.activation(self.linear1(hidden)))


class ClapModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = XLMRobertaModel(config.text_config)
        for layer in self.text_model.encoder.layer:
            layer.attention.self.attn = RoundedTextAttention()
        self.text_pooler = Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        self.tanh = Tanh()
        self.audio_model = AudioTower(config.audio_config)
        self.text_projection, self.audio_projection = Projection(config.text_config), Projection(config.audio_config)
        self.logit_scale_a, self.logit_scale_t = nn.Parameter(torch.empty(())), nn.Parameter(torch.empty(()))
        self.register_buffer('scale_a', torch.empty(()), persistent=False)
        self.register_buffer('scale_t', torch.empty(()), persistent=False)
        self.normalize, self.matmul = L2Norm(dim=-1, eps=0), BMM()

    def forward(self, input_ids, input_features, attention_mask=None):
        audio_hidden, audio_pool = self.audio_model(input_features)
        text_hidden = self.text_model.forward_with_attention_mask(input_ids, attention_mask)
        text_pool = self.tanh(self.text_pooler(text_hidden[:, 0]))
        audio = self.normalize(self.audio_projection(audio_pool))
        text = self.normalize(self.text_projection(text_pool))
        return {'logits_per_audio': self.matmul(audio, text.t()) * self.scale_a,
                'logits_per_text': self.matmul(text, audio.t()) * self.scale_t,
                'text_embeds': text, 'audio_embeds': audio,
                'text_model_output.last_hidden_state': text_hidden, 'text_model_output.pooler_output': text_pool,
                'audio_model_output.last_hidden_state': audio_hidden, 'audio_model_output.pooler_output': audio_pool}


def build_from_config(config, device, dtype):
    audio = config.audio_config
    if (audio.enable_fusion or not audio.enable_patch_layer_norm or not audio.flatten_patch_embeds
            or audio.hidden_act != 'gelu' or config.text_config.hidden_act != 'gelu'
            or audio.projection_hidden_act != 'relu' or config.text_config.projection_hidden_act != 'relu'
            or len(audio.depths) != 4 or audio.patch_stride[0] != audio.patch_stride[1]):
        raise ValueError('CLAP case preserves the documented unfused four-stage spectrogram and GELU/ReLU towers')
    return ClapModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if source.startswith('text_pooler.'):
            source = source.replace('text_pooler.', 'text_model.pooler.dense.')
        elif source.startswith('audio_model.'):
            source = source.replace('audio_model.', 'audio_model.audio_encoder.', 1)
            source = source.replace('patch_conv.', 'patch_embed.proj.').replace('patch_norm.', 'patch_embed.norm.')
            source = source.replace('.block.norm1.', '.layernorm_before.').replace('.block.norm2.', '.layernorm_after.')
            source = source.replace('.block.attn.proj.', '.attention.output.dense.')
            source = source.replace('.block.mlp.fc1.', '.intermediate.dense.').replace('.block.mlp.fc2.', '.output.dense.')
            source = source.replace('.relative_position_bias_table', '.attention.self.relative_position_bias_table')
            source = source.replace('.relative_position_index', '.attention.self.relative_position_index')
        if '.qkv.' in source:
            source = source.replace('.block.attn.', '.attention.self.')
            sources = [source.replace('.qkv.', f'.{part}.') for part in ('query', 'key', 'value')]
            mapped[name] = torch.cat([state_dict[key] for key in sources])
            used.update(sources)
        else:
            mapped[name] = state_dict[source]
            used.add(source)
    for field, expected in [('position_ids', torch.arange(config.text_config.max_position_embeddings)[None]),
                             ('token_type_ids', torch.zeros(1, config.text_config.max_position_embeddings, dtype=torch.long))]:
        source = 'text_model.embeddings.' + field
        if not torch.equal(state_dict[source].cpu(), expected):
            raise ValueError('CLAP default text position/type buffer differs')
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped CLAP weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
    model.scale_a.copy_(model.logit_scale_a.exp())
    model.scale_t.copy_(model.logit_scale_t.exp())


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
