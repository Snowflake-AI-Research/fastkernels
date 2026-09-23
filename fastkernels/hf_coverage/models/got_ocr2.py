"""GOT-OCR2's window/global relative-position vision and tied Qwen2 decoder."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from . import llama, qwen2
from .qwen2_precision import NativeRotaryEmbedding, configure_language
from ..runner import Workload


class RelativeAttention(nn.Module):
    def __init__(self, config, window):
        super().__init__()
        self.heads, self.head_dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.side = window or config.image_size // config.patch_size
        self.qkv = Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = Linear(config.hidden_size, config.hidden_size)
        self.rel_pos_h = nn.Parameter(torch.empty(2 * self.side - 1, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.empty_like(self.rel_pos_h))
        self.bmm, self.softmax, self.resize = BMM(), Softmax(), Interpolate()
        self.register_buffer("height_positions", None, persistent=False)
        self.register_buffer("width_positions", None, persistent=False)

    def prepare_positions(self):
        indices = torch.arange(self.side, device=self.rel_pos_h.device)
        relative = indices[:, None] - indices[None] + self.side - 1
        for name, parameter in (("height_positions", self.rel_pos_h), ("width_positions", self.rel_pos_w)):
            resized = self.resize(parameter.t()[None], size=2 * self.side - 1, mode="linear")[0].t()
            setattr(self, name, resized[relative])

    def forward(self, hidden):
        batch, height, width, channels = hidden.shape
        query, key, value = self.qkv(hidden).view(batch, height * width, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).reshape(3, -1, height * width, self.head_dim).unbind(0)
        grid = query.view(-1, height, width, self.head_dim)
        by_height = grid.permute(1, 0, 2, 3).reshape(height, -1, self.head_dim)
        by_width = grid.permute(2, 0, 1, 3).reshape(width, -1, self.head_dim)
        rel_h = self.bmm(by_height, self.height_positions.transpose(1, 2)).view(height, -1, width, height).permute(1, 0, 2, 3)
        rel_w = self.bmm(by_width, self.width_positions.transpose(1, 2)).view(width, -1, height, width).permute(1, 2, 0, 3)
        bias = rel_h[..., None] + rel_w[:, :, :, None, :]
        scores = self.bmm(query * self.head_dim ** -0.5, key.transpose(-1, -2)) + bias.reshape(-1, height * width, height * width)
        probabilities = self.softmax(scores.float()).to(query.dtype)
        output = self.bmm(probabilities, value).view(batch, self.heads, height, width, self.head_dim)
        return self.proj(output.permute(0, 2, 3, 1, 4).reshape(batch, height, width, channels))


class OCRLayer(nn.Module):
    def __init__(self, config, window):
        super().__init__()
        self.window = window
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.attn = RelativeAttention(config, window)
        self.mlp = nn.ModuleDict({'lin1': Linear(config.hidden_size, config.mlp_dim),
                                  'lin2': Linear(config.mlp_dim, config.hidden_size)})
        self.activation = GELU()

    def forward(self, hidden):
        normalized = self.layer_norm1(hidden)
        batch, height, width, channels = hidden.shape
        if self.window:
            window = self.window
            ph, pw = (-height) % window, (-width) % window
            normalized = F.pad(normalized, (0, 0, 0, pw, 0, ph))
            normalized = normalized.view(batch, (height + ph) // window, window, (width + pw) // window, window, channels)
            normalized = normalized.permute(0, 1, 3, 2, 4, 5).reshape(-1, window, window, channels)
        output = self.attn(normalized)
        if self.window:
            output = output.view(batch, (height + ph) // window, (width + pw) // window, window, window, channels)
            output = output.permute(0, 1, 3, 2, 4, 5).reshape(batch, height + ph, width + pw, channels)[:, :height, :width]
        hidden = hidden + output
        return hidden + self.mlp['lin2'](self.activation(self.mlp['lin1'](self.layer_norm2(hidden))))


class OCRVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        side = config.image_size // config.patch_size
        self.pos_embed = nn.Parameter(torch.empty(1, side, side, config.hidden_size))
        self.layers = nn.ModuleList([OCRLayer(config, 0 if i in config.global_attn_indexes else config.window_size)
                                     for i in range(config.num_hidden_layers)])
        self.neck = nn.ModuleDict({'conv1': Conv2d(config.hidden_size, config.output_channels, 1, bias=False),
                                   'conv2': Conv2d(config.output_channels, config.output_channels, 3, padding=1, bias=False),
                                   'layer_norm1': LayerNorm(config.output_channels, eps=1e-6, promote_fp32=False),
                                   'layer_norm2': LayerNorm(config.output_channels, eps=1e-6, promote_fp32=False)})

    def forward(self, pixels):
        hidden = self.patch_embed(pixels).permute(0, 2, 3, 1) + self.pos_embed
        for layer in self.layers:
            hidden = layer(hidden)
        for i in (1, 2):
            hidden = self.neck[f'conv{i}'](hidden.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            hidden = self.neck[f'layer_norm{i}'](hidden)
        return hidden.permute(0, 3, 1, 2)


class OCRBackbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text, self.vision = text, OCRVision(config.vision_config)
        width, language = config.vision_config.output_channels, config.text_config.hidden_size
        self.conv_upsampler1 = Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False)
        self.conv_upsampler2 = Conv2d(2 * width, language, 3, stride=2, padding=1, bias=False)
        self.projector = Linear(language, language)
        self.image_token_id = config.image_token_index
        self.pixels = self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def features(self, pixels):
        hidden = self.conv_upsampler2(self.conv_upsampler1(self.vision(pixels)))
        return self.projector(hidden.flatten(2).transpose(1, 2))

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            self.image_hidden_states = self.features(self.pixels)
            embeddings = embeddings.masked_scatter((input_ids == self.image_token_id)[:, None].expand_as(embeddings), self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


class GotOCRModel(nn.Module):
    def __init__(self, language, config):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = OCRBackbone(language.model, config)


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if (not config.tie_word_embeddings or text.use_sliding_window or text.rope_parameters['rope_type'] != 'default'
            or vision.hidden_act != 'gelu' or not vision.use_abs_pos or not vision.use_rel_pos):
        raise ValueError('The documented GOT-OCR2 uses relative/absolute visual positions and a tied full-attention Qwen2')
    fields = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
              'vocab_size', 'max_position_embeddings', 'rms_norm_eps')
    native = LlamaConfig(**{name: getattr(text, name) for name in fields}, head_dim=text.hidden_size // text.num_attention_heads,
                         dtype=dtype, rope_theta=text.rope_parameters['rope_theta'], rope_scaling_factor=1.0, qkv_bias=True)
    language = LlamaForCausalLM(native)
    configure_language(language.model, native)
    language.model.rotary_emb = NativeRotaryEmbedding(native.head_dim, native.max_position_embeddings,
                                                      native.rope_theta)
    for layer in language.model.layers:
        layer.self_attn.rotary_emb = language.model.rotary_emb
    language.lm_head.embedding_op.emb.weight = language.model.embed_tokens.embedding_op.emb.weight
    return GotOCRModel(language, config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace('model.language_model.', 'model.'): remaining.pop(name)
            for name in list(remaining) if name.startswith('model.language_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    if not torch.equal(text['lm_head.weight'], text['model.embed_tokens.weight']):
        raise ValueError('GOT-OCR2 tied head and embeddings differ')
    qwen2.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head, config=model.config), text, config.text_config)
    model.model.vision.load_state_dict({name: remaining.pop('model.vision_tower.' + name.replace('patch_embed.', 'patch_embed.projection.'))
                                       for name in model.model.vision.state_dict()}, strict=True)
    for layer in model.model.vision.layers:
        layer.attn.prepare_positions()
    for name, source in [('conv_upsampler1','conv_upsampler1'),('conv_upsampler2','conv_upsampler2'),('projector','multimodal_projector')]:
        module = getattr(model.model, name)
        module.load_state_dict({field: remaining.pop(f'model.multi_modal_projector.{source}.{field}')
                                for field in module.state_dict()}, strict=True)
    if remaining:
        raise KeyError(f'Unmapped GOT-OCR2 state: {sorted(remaining)}')


def make_workloads(model, inputs, config, case=None):
    model.model.pixels = inputs['pixel_values']
    workloads = llama.make_workloads(model, {'input_ids': inputs['input_ids']}, model.config, case=case)
    prefill = workloads['prefill']
    workloads['prefill'] = Workload(
        run=lambda: {**prefill.run(), 'image_hidden_states': model.model.image_hidden_states},
        prepare=prefill.prepare, collect=prefill.collect,
    )
    return workloads
