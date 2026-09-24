"""Chameleon image VQ and cached text inference from existing operations."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.gemma_dense_attention import GemmaRotaryEmbedding, _apply_rotary_pos_emb
from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from ..runner import Workload
from .dac import VectorQuantizer
from .emu3 import Downsample, GatedActivation, Residual
from .qwen2_5_omni import TextMLP


class ImageAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = GroupNorm(32, width, eps=1e-6)
        for name in ('q', 'k', 'v', 'proj_out'):
            setattr(self, name, Conv2d(width, width, 1))
        self.matmul, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, hidden):
        residual = hidden
        hidden = self.norm(hidden)
        q, k, v = (getattr(self, name)(hidden).flatten(2) for name in ('q', 'k', 'v'))
        weights = self.softmax(self.matmul(q.transpose(1, 2), k) * hidden.shape[1]**-0.5)
        return residual + self.proj_out(self.matmul(v, weights.transpose(1, 2)).reshape_as(hidden))


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, resolution = config.base_channels, config.resolution
        self.conv_in = Conv2d(config.in_channels, width, 3, padding=1)
        self.down = nn.ModuleList()
        for index, multiplier in enumerate(config.channel_multiplier):
            target = config.base_channels * multiplier
            stage = nn.Module()
            stage.block, stage.attn = nn.ModuleList(), nn.ModuleList()
            for _ in range(config.num_res_blocks):
                stage.block.append(Residual(width, target))
                width = target
                if resolution in (config.attn_resolutions or []):
                    stage.attn.append(ImageAttention(width))
            if index + 1 < len(config.channel_multiplier):
                stage.downsample = Downsample(width)
                resolution //= 2
            self.down.append(stage)
        self.mid = nn.Module()
        self.mid.block_1, self.mid.block_2 = Residual(width, width), Residual(width, width)
        self.mid.attn_1 = ImageAttention(width)
        self.norm_out = GroupNorm(32, width, eps=1e-6)
        self.conv_out = Conv2d(width, config.latent_channels, 3, padding=1)
        self.activation = GatedActivation()

    def forward(self, pixels):
        hidden = self.conv_in(pixels)
        for stage in self.down:
            for index, block in enumerate(stage.block):
                hidden = block(hidden)
                if len(stage.attn):
                    hidden = stage.attn[index](hidden)
            if hasattr(stage, 'downsample'):
                hidden = stage.downsample(hidden)
        hidden = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hidden)))
        return self.conv_out(self.activation(self.norm_out(hidden)))


class Quantizer(nn.Module):
    squared_row_norm = VectorQuantizer.squared_row_norm

    def __init__(self, config):
        super().__init__()
        self.beta = config.get('beta', .25)
        self.embedding = Embedding(config.num_embeddings, config.embed_dim)
        self.product, self.reduce = ProductGate(), SegmentCSR()
        self.matmul, self.select = BMM(), CodecTop1()

    def forward(self, hidden):
        rows = hidden.permute(0, 2, 3, 1).contiguous()
        flat = rows.reshape(-1, rows.shape[-1])
        codes = self.embedding.emb.weight
        distances = self.squared_row_norm(flat) + self.squared_row_norm(codes).T - 2 * self.matmul(flat, codes.T)
        indices = self.select(-distances)
        quantized = self.embedding(indices).reshape_as(rows)
        difference = quantized - rows
        squared = self.product(torch.cat((difference, difference), -1))
        offsets = torch.tensor([0, squared.numel()], device=hidden.device)
        mean = self.reduce(squared.flatten().float(), offsets, reduce='mean').to(hidden.dtype).squeeze(0)
        loss = mean + self.beta * mean
        # Preserve the actual subtraction/addition rounding of the forward STE.
        quantized = rows + (quantized - rows)
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, indices


class VQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ImageEncoder(config)
        self.quant_conv = Conv2d(config.latent_channels, config.embed_dim, 1)
        self.quantize = Quantizer(config)
        # Native creates this parameter container but encode does not call it.
        self.post_quant_conv = Conv2d(config.embed_dim, config.latent_channels, 1)

    def encode(self, pixels):
        hidden = self.encoder(pixels)
        quantized, loss, indices = self.quantize(self.quant_conv(hidden))
        return dict(last_hidden_state=hidden, quantized_last_hidden_state=quantized,
                    embedding_loss=loss, image_tokens=indices)


class HeadLayerNorm(nn.Module):
    """Shared last-axis statistics followed by separate affine per head."""
    def __init__(self, heads, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        self.norm = LayerNorm(dim, eps=1e-5, elementwise_affine=False, promote_fp32=False)
        self.product = ProductGate()

    def forward(self, hidden):
        hidden = self.norm(hidden)
        return self.product(torch.cat((hidden, self.weight.expand_as(hidden)), -1)) + self.bias


class TextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = config.hidden_size // self.heads
        for name, heads in (('q_proj', self.heads), ('k_proj', self.kv_heads), ('v_proj', self.kv_heads)):
            setattr(self, name, Linear(config.hidden_size, heads * self.dim, bias=config.attention_bias))
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm, self.k_norm = HeadLayerNorm(self.heads, self.dim), HeadLayerNorm(self.kv_heads, self.dim)
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, cos, sin, past):
        batch, length = hidden.shape[:2]
        q = self.q_norm(self.q_proj(hidden).reshape(batch, length, self.heads, self.dim))
        k = self.k_norm(self.k_proj(hidden).reshape(batch, length, self.kv_heads, self.dim))
        v = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.dim)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        if past is not None:
            k, v = torch.cat((past[0], k), 1), torch.cat((past[1], v), 1)
        key, value = (x.repeat_interleave(self.heads // self.kv_heads, dim=2) for x in (k, v))
        output = self.attention(q, key, value, causal=past is None and length > 1)
        return self.o_proj(output.reshape_as(hidden)), (k, v)


class TextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.mlp = TextAttention(config), TextMLP(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, cos, sin, past):
        attention, state = self.self_attn(self.input_layernorm(hidden), cos, sin, past)
        hidden = hidden + attention
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), state


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(TextLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vqmodel = VQModel(config.vq_config)
        self.image_token_id = config.vocabulary_map['<image>']
        mapping = {}
        translation = str.maketrans('ABCDEFGHIJ', '0123456789')
        for name, token in config.vocabulary_map.items():
            if name.startswith('IMGIMG'):
                mapping[int(name[6:-1].translate(translation))] = token
        if set(mapping) != set(range(config.vq_config.num_embeddings)):
            raise ValueError('Chameleon requires the complete published image-code mapping')
        self.register_buffer('image_to_bpe', torch.tensor([mapping[i] for i in range(len(mapping))]), persistent=False)
        self.register_buffer('image_token_ids', torch.tensor(sorted(mapping.values())), persistent=False)
        self.rotary = GemmaRotaryEmbedding(config.hidden_size // config.num_attention_heads,
                                           config.max_position_embeddings, config.rope_parameters['rope_theta'])

    def forward(self, input_ids, pixel_values=None, past_key_values=None):
        hidden = self.embed_tokens(input_ids)
        if pixel_values is not None:
            image = self.vqmodel.encode(pixel_values)
            bpe = self.image_to_bpe[image['image_tokens']].reshape(pixel_values.shape[0], -1)
            image_features = self.embed_tokens(bpe)
            mask = (input_ids == self.image_token_id)[..., None].expand_as(hidden)
            if hidden[mask].numel() != image_features.numel():
                raise ValueError('Image placeholders must match VQ tokens')
            hidden = hidden.masked_scatter(mask, image_features)
        previous = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        if previous and input_ids.shape[1] != 1:
            raise ValueError('Selected continuation consumes one supplied token per call')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None] + previous
        cos, sin = self.rotary(hidden, positions)
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, cos, sin, None if past_key_values is None else past_key_values[index])
            states.append(state)
        logits = self.lm_head(self.norm(hidden))
        logits[:, :, self.image_token_ids] = torch.finfo(logits.dtype).min
        return {'logits': logits, 'past_key_values': tuple(states)}


def build_from_config(config, device, dtype):
    if (config.swin_norm or config.mlp_bias or config.hidden_act != 'silu' or config.tie_word_embeddings
            or config.rope_parameters['rope_type'] != 'default' or not config.use_cache
            or config.vq_config.attn_type != 'vanilla' or config.vq_config.double_latent):
        raise ValueError('Chameleon adapter selects published 7B pre-norm SiLU text and vanilla VQ encoder')
    model = Model(config).to(device=device, dtype=dtype).eval()
    with torch.device('cpu'):
        model.rotary = GemmaRotaryEmbedding(config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings, config.rope_parameters['rope_theta']).to(device=device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        source = name if name.startswith('lm_head.') else 'model.' + name
        source = source.replace('.emb.weight', '.weight')
        mapped[name] = state_dict[source].reshape(target.shape)
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped Chameleon state: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def flatten(output):
    result = {'logits': output['logits']}
    for index, (key, value) in enumerate(output['past_key_values']):
        result[f'past_key_values.{index}.key'] = key.transpose(1, 2)
        result[f'past_key_values.{index}.value'] = value.transpose(1, 2)
    return result


def make_workloads(model, inputs, config, case=None):
    if case is None or case.get('workload') != 'causal_lm_continuation':
        return {'forward': Workload(run=lambda: flatten(model(**inputs)))}
    if set(inputs) != {'input_ids', 'pixel_values'}:
        raise ValueError('Selected Chameleon continuation uses unpadded text and image inputs')
    ids, state = inputs['input_ids'], {}
    prefix = ids.shape[1] - 2
    if prefix < 1:
        raise ValueError('Continuation requires a prefix and two supplied tokens')

    def initial():
        return model(ids[:, :prefix], pixel_values=inputs['pixel_values'])

    def advance(index, previous):
        return model(ids[:, prefix + index:prefix + index + 1], past_key_values=previous)

    def prepare(index):
        previous = initial()['past_key_values']
        for step in range(index):
            previous = advance(step, previous)['past_key_values']
        state['previous'] = previous

    return {
        'prefill': Workload(run=lambda: flatten(initial())),
        'decode_1': Workload(run=lambda: flatten(advance(0, state['previous'])), prepare=lambda: prepare(0)),
        'decode_2': Workload(run=lambda: flatten(advance(1, state['previous'])), prepare=lambda: prepare(1)),
    }
