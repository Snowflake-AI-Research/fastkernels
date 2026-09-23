"""Gemma3n image/audio/text composition from existing FastKernels operations.

Preserves AltUp, Laurel, per-layer inputs, Gaussian sparse MLPs, shared KV,
MobileNetV5, and cumulative-normalized local-attention audio. The timm meta
constructor supplies architecture metadata only; no source forward executes
activations. CPU parity and GPU component checks are diagnostic; full-model
GPU acceptance is evaluated separately.
"""

from dataclasses import dataclass
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from ..patches.gemma4_rms_norm import Gemma4RMSNorm as Norm
from ..patches.gemma3n_row_stats import Gemma3nRowStats
from ..patches.product_gate import ProductGate
from ..patches.forecast_revin import ForecastNormalize
from .gemma4 import ScaledEmbedding
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from ..patches.gemma3n_vision_norm import Gemma3nVisionNorm
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative as Conv1d
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from ..patches.gemma3n_row_stats import Gemma3nRowSum, Gemma3nPrefixSum
from ..patches.forecast_revin import ZeroSafeVarianceNormalize
from ..patches.dfine_clamp import DFineClamp
from fastkernels.tasks.baseline.L1.embedding import Embedding
from ..runner import Workload


def product(x, scale):
    return ProductGate()(torch.cat((x, scale.expand_as(x)), -1))


@dataclass
class TextCache:
    layers: dict
    seen: int


class SparseMLP(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        middle = c.intermediate_size[index]
        self.gate_proj = Linear(c.hidden_size, middle, bias=False)
        self.up_proj = Linear(c.hidden_size, middle, bias=False)
        self.down_proj = Linear(middle, c.hidden_size, bias=False)
        self.sparsity = c.activation_sparsity_pattern[index]
        self.stats, self.relu, self.act = Gemma3nRowStats(), ReLU(), GELU(approximate='tanh')
        # Sparsity is configuration metadata; match HF's FP32 icdf then dtype cast.
        probability = torch.tensor(self.sparsity, dtype=torch.float32, device='cpu')
        normal = torch.distributions.Normal(torch.zeros_like(probability), torch.ones_like(probability))
        self.cutoff_scale = normal.icdf(probability) if self.sparsity else probability

    def forward(self, x):
        gate = self.gate_proj(x)
        if self.sparsity:
            mean, std = self.stats(gate)
            cutoff = mean + std * self.cutoff_scale.to(device=x.device, dtype=x.dtype)
            gate = self.relu(gate - cutoff)
        return self.down_proj(product(self.act(gate), self.up_proj(x)))


class AltUp(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        n = c.altup_num_inputs
        self.correct_output_scale = nn.Parameter(torch.empty(c.hidden_size))
        self.correction_coefs = Linear(n, n, bias=False)
        self.prediction_coefs = Linear(n, n*n, bias=False)
        self.modality_router = Linear(c.hidden_size, n, bias=False)
        self.router_norm = Norm(c.hidden_size, c.rms_norm_eps)
        self.tanh, self.mm = Tanh(), BMM()

    def modalities(self, x):
        scale = x.new_tensor(self.c.hidden_size**-1.)
        return self.tanh(self.modality_router(self.router_norm(x) * scale).float()).to(x.dtype)

    def predict(self, x):
        modalities = self.modalities(x[self.c.altup_active_idx])
        n = self.c.altup_num_inputs
        coefs = self.prediction_coefs(modalities).reshape(*modalities.shape[:-1], n, n).transpose(-1, -2)
        predicted = self.mm(x.permute(1, 2, 3, 0), coefs).permute(3, 0, 1, 2)
        return (predicted + x).contiguous()

    def correct(self, predictions, activated):
        coefs = (self.correction_coefs(self.modalities(activated)) + 1.).permute(2, 0, 1)[..., None]
        innovation = (activated - predictions[self.c.altup_active_idx])[None].expand_as(predictions)
        return (product(innovation, coefs) + predictions).contiguous()


class Laurel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear_left = Linear(c.hidden_size, c.laurel_rank, bias=False)
        self.linear_right = Linear(c.laurel_rank, c.hidden_size, bias=False)
        self.post_laurel_norm = Norm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x):
        return x + self.post_laurel_norm(self.linear_right(self.linear_left(x)))


class Attention(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.c, self.index = c, index
        self.kind = c.layer_types[index]
        first_shared = c.num_hidden_layers - c.num_kv_shared_layers
        self.shared = index >= first_shared > 0
        prior = c.layer_types[:first_shared]
        self.source = max(i for i, kind in enumerate(prior) if kind == self.kind)
        self.store_shared = not self.shared and index == self.source
        self.q_proj = Linear(c.hidden_size, c.num_attention_heads*c.head_dim, bias=c.attention_bias)
        self.q_norm = Norm(c.head_dim, c.rms_norm_eps)
        if not self.shared:
            self.k_proj = Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, bias=c.attention_bias)
            self.v_proj = Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, bias=c.attention_bias)
            self.k_norm = Norm(c.head_dim, c.rms_norm_eps)
            self.v_norm = Norm(c.head_dim, c.rms_norm_eps, False)
        self.o_proj = Linear(c.num_attention_heads*c.head_dim, c.hidden_size, bias=c.attention_bias)
        self.attention = DenseAttention(backend='sdpa')
        self.register_buffer('rope_inv_freq', None, persistent=False)

    def load_rotary_metadata(self):
        # HF initializes these nonpersistent buffers on CPU in FP32, then moves
        # them without a dtype cast. CUDA pow changes rare BF16 sine roundings.
        dim = self.c.head_dim
        theta = self.c.rope_parameters[self.kind]['rope_theta']
        exponents = torch.arange(0, dim, 2, device='cpu', dtype=torch.float32) / dim
        self.rope_inv_freq = (1. / theta**exponents).to(self.q_proj.weight.device)

    def rotary(self, x, positions):
        # All trigonometry depends only on configuration and position metadata.
        # Preserve the two BF16 products and their addition as in native HF.
        angles = positions.reshape(-1, 1).float() * self.rope_inv_freq[None]
        cos = torch.cat((angles.cos(), angles.cos()), -1).to(x.dtype).reshape(*x.shape[:2], 1, -1)
        sin = torch.cat((angles.sin(), angles.sin()), -1).to(x.dtype).reshape(*x.shape[:2], 1, -1)
        rotated = torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)
        return product(x, cos) + product(rotated, sin)

    def forward(self, x, positions, previous, shared, attention_mask):
        c = self.c
        shape = (*x.shape[:2], -1, c.head_dim)
        q = self.rotary(self.q_norm(self.q_proj(x).view(shape)), positions)
        state = None
        if self.shared:
            k, v = shared[self.source]
        else:
            k = self.rotary(self.k_norm(self.k_proj(x).view(shape)), positions)
            v = self.v_norm(self.v_proj(x).view(shape))
            if previous is not None:
                k, v = [torch.cat(pair, 1) for pair in zip(previous, (k, v))]
            if self.store_shared:
                shared[self.index] = k, v
            if self.kind == 'sliding_attention':
                state = tuple(t[:, -(c.sliding_window-1):].clone() for t in (k, v))
            else:
                state = (k, v)
        start = positions[0, -1].item() + 1 - k.shape[1]
        keys = torch.arange(start, start+k.shape[1], device=x.device)
        mask = keys[None] <= positions[..., None]
        if self.kind == 'sliding_attention':
            mask = mask & (keys[None] > positions[..., None] - c.sliding_window)
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, start:start+k.shape[1]].bool()
        repeats = c.num_attention_heads // c.num_key_value_heads
        if repeats != 1:
            k, v = (t.repeat_interleave(repeats, 2) for t in (k, v))
        # HF omits the full-attention mask for an unpadded initial sequence or
        # a single continuation query. Preserve that SDPA dispatch: an explicit
        # causal matrix selects a different CUDA kernel and changes BF16 rounding.
        unmasked_full = (self.kind == 'full_attention' and attention_mask is None
                         and (q.shape[1] == 1 or (start == 0 and q.shape[1] == k.shape[1])))
        output = self.attention(q, k, v, softmax_scale=1.,
                                causal=unmasked_full and q.shape[1] > 1,
                                attn_mask=None if unmasked_full else mask[:, None])
        return self.o_proj(output.reshape(*x.shape[:2], -1)), state


class Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.c = c
        self.self_attn, self.mlp = Attention(c, index), SparseMLP(c, index)
        self.altup, self.laurel = AltUp(c), Laurel(c)
        for name in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
                     'post_feedforward_layernorm', 'post_per_layer_input_norm'):
            setattr(self, name, Norm(c.hidden_size, c.rms_norm_eps))
        self.per_layer_input_gate = Linear(c.hidden_size, c.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = Linear(c.hidden_size_per_layer_input, c.hidden_size, bias=False)
        self.act = GELU(approximate='tanh')

    def forward(self, x, ple, positions, previous, shared, attention_mask):
        predictions = self.altup.predict(x)
        active = predictions[self.c.altup_active_idx]
        normed = self.input_layernorm(active)
        laurel = self.laurel(normed)
        attention, state = self.self_attn(normed, positions, previous, shared, attention_mask)
        hidden = (active + self.post_attention_layernorm(attention) + laurel) / 2**.5
        hidden = hidden + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        corrected = self.altup.correct(predictions, hidden)
        first = corrected[self.c.altup_active_idx].clone()
        if self.c.altup_correct_scale:
            first = product(first, self.altup.correct_output_scale)
        ple = product(self.act(self.per_layer_input_gate(first)), ple)
        corrected[1:] += self.post_per_layer_input_norm(self.per_layer_projection(ple))
        return corrected, state


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        if (c.hidden_activation != 'gelu_pytorch_tanh' or c.sliding_window < 2
                or c.num_attention_heads % c.num_key_value_heads
                or c.num_hidden_layers <= c.num_kv_shared_layers
                or any(p['rope_type'] != 'default' for p in c.rope_parameters.values())):
            raise ValueError('Text component retains default GELU, grouped attention, and RoPE semantics')
        self.c = c
        self.embed_tokens = ScaledEmbedding(c.vocab_size, c.hidden_size, c.hidden_size**.5)
        p = c.hidden_size_per_layer_input
        self.embed_tokens_per_layer = ScaledEmbedding(c.vocab_size_per_layer_input, c.num_hidden_layers*p, p**.5)
        self.per_layer_model_projection = Linear(c.hidden_size, c.num_hidden_layers*p, bias=False)
        self.per_layer_projection_norm = Norm(p, c.rms_norm_eps)
        self.altup_projections = nn.ModuleList(Linear(c.hidden_size, c.hidden_size, bias=False)
                                             for _ in range(c.altup_num_inputs-1))
        self.altup_unembed_projections = nn.ModuleList(Linear(c.hidden_size, c.hidden_size, bias=False)
                                                     for _ in range(c.altup_num_inputs-1))
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.num_hidden_layers))
        self.norm = Norm(c.hidden_size, c.rms_norm_eps)
        self.target_rms, self.current_rms = Gemma3nRowStats(rms=True), Gemma3nRowStats(rms=True, floor=1e-5)
        self.divide = ForecastNormalize(tolerance=0.)

    def match_magnitude(self, current, target_rms):
        _, rms = self.current_rms(current)
        weighted = product(current, target_rms)
        return self.divide(weighted, torch.zeros_like(rms[..., 0]), rms[..., 0])

    def forward(self, input_ids=None, inputs_embeds=None, per_layer_inputs=None,
                attention_mask=None, past_key_values=None):
        c = self.c
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.embed_tokens_per_layer(input_ids).reshape(
                *input_ids.shape, c.num_hidden_layers, c.hidden_size_per_layer_input)
        shape = (*inputs_embeds.shape[:2], c.num_hidden_layers, c.hidden_size_per_layer_input)
        projected = self.per_layer_model_projection(inputs_embeds) * inputs_embeds.new_tensor(c.hidden_size**-.5)
        ple = self.per_layer_projection_norm(projected.reshape(shape))
        if per_layer_inputs is not None:
            ple = (ple + per_layer_inputs) * inputs_embeds.new_tensor(2**-.5)
        start = 0 if past_key_values is None else past_key_values.seen
        positions = torch.arange(start, start+inputs_embeds.shape[1], device=inputs_embeds.device)[None]
        positions = positions.expand(inputs_embeds.shape[0], -1)
        _, magnitude = self.target_rms(inputs_embeds)
        x = torch.stack([inputs_embeds] + [self.match_magnitude(proj(inputs_embeds), magnitude)
                                           for proj in self.altup_projections])
        shared, states = {}, {}
        for i, layer in enumerate(self.layers):
            previous = None if past_key_values is None else past_key_values.layers.get(i)
            x, state = layer(x, ple[:, :, i], positions, previous, shared, attention_mask)
            if state is not None:
                states[i] = state
        _, magnitude = self.target_rms(x[0])
        parts = [x[0]] + [self.match_magnitude(proj(x[i+1]), magnitude)
                          for i, proj in enumerate(self.altup_unembed_projections)]
        # Four explicit additions would change BF16's native FP32 reduction;
        # use the existing average-pooling operation for the AltUp input axis.
        stacked = torch.stack(parts, -1)
        pooled = GlobalAvgPool2d()(stacked.reshape(-1, 1, 1, c.altup_num_inputs))
        hidden = self.norm(pooled.reshape_as(x[0]))
        return {'last_hidden_state': hidden, 'past_key_values': TextCache(states, start+hidden.shape[1])}


class TextForCausalLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, self.model = c, TextModel(c)
        self.lm_head = Linear(c.hidden_size, c.vocab_size, bias=False)
        self.tanh = Tanh()

    def forward(self, **inputs):
        result = self.model(**inputs)
        logits = self.lm_head(result['last_hidden_state'])
        if self.c.final_logit_softcapping is not None:
            cap = self.c.final_logit_softcapping
            logits = self.tanh(logits / cap) * cap
        return dict(result, logits=logits)


def load_text_state_dict_into(model, state_dict):
    mapped = {}
    for name in model.state_dict():
        source = name.replace('.embed_tokens.emb.', '.embed_tokens.').replace(
            '.embed_tokens_per_layer.emb.', '.embed_tokens_per_layer.')
        mapped[name] = state_dict[source]
    unused = set(state_dict) - {name.replace('.embed_tokens.emb.', '.embed_tokens.').replace(
        '.embed_tokens_per_layer.emb.', '.embed_tokens_per_layer.') for name in mapped}
    if unused:
        raise ValueError(f'Unmapped text state: {sorted(unused)}')
    model.load_state_dict(mapped, strict=True, assign=True)
    for layer in model.model.layers:
        layer.self_attn.load_rotary_metadata()


# The vision constructor is used on meta solely to obtain timm's concrete
# architecture metadata. No timm module or forward method executes activations.


class VisionConv(Conv2d):
    def __init__(self, source):
        super().__init__(source.in_channels, source.out_channels, source.kernel_size,
                         source.stride, source.padding, source.groups, source.dilation,
                         source.bias is not None)
        self.same = type(source).__name__ == 'Conv2dSame'

    def forward(self, x):
        if self.same:
            ph, pw = [max((size + stride - 1)//stride*stride - stride + (kernel-1)*dilation + 1 - size, 0)
                      for size, stride, kernel, dilation in zip(x.shape[-2:], self.stride,
                                                               self.weight.shape[-2:], self.dilation)]
            x = torch.nn.functional.pad(x, (pw//2, pw-pw//2, ph//2, ph-ph//2))
        return super().forward(x)


class VisionNorm(Gemma3nVisionNorm):
    def __init__(self, source):
        super().__init__(source.weight.numel(), source.eps)
        self.act = vision_component(source.act) if hasattr(source, 'act') else nn.Identity()

    def forward(self, x):
        return self.act(super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())


class VisionScale(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(source.gamma.shape))

    def forward(self, x):
        return product(x.permute(0, 2, 3, 1), self.gamma).permute(0, 3, 1, 2).contiguous()


class VisionWiring(nn.Module):
    def __init__(self, source, order):
        super().__init__()
        self.order, self.has_skip = order, getattr(source, 'has_skip', False)
        for name, child in source.named_children():
            self.add_module(name, vision_component(child))

    def forward(self, x):
        if getattr(self, 'conv_cpe_dw', None) is not None:
            x = x + self.conv_cpe_dw(x)
        residual = x
        for name in self.order:
            x = getattr(self, name)(x)
        return x + residual if self.has_skip else x


class VisionAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.has_query_strides or not source.fused_attn:
            raise ValueError('Gemma3n vision retains default unstrided-query SDPA')
        self.heads, self.key_dim = source.num_heads, source.key_dim
        for name in ('query', 'key', 'value', 'output'):
            self.add_module(name, vision_component(getattr(source, name)))
        self.core = DenseAttention(backend='sdpa')

    def forward(self, x):
        b, _, h, w = x.shape
        q = self.query(x).reshape(b, self.heads, self.key_dim, -1).permute(0, 3, 1, 2).contiguous()
        k, v = [layer(x).flatten(2).transpose(1, 2)[:, :, None].contiguous()
                for layer in (self.key, self.value)]
        # Native timm SDPA broadcasts its single KV head. Preserve this path
        # instead of materializing repeated K/V before attention.
        out = self.core(q, k, v)
        out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return self.output(out)


def vision_component(source):
    kind = type(source).__name__
    if isinstance(source, nn.Conv2d):
        return VisionConv(source)
    if kind in ('RmsNorm2d', 'RmsNormAct2d'):
        return VisionNorm(source)
    if isinstance(source, nn.GELU):
        return GELU(approximate=source.approximate)
    if isinstance(source, (nn.Identity, nn.Dropout)):
        return nn.Identity()
    if isinstance(source, nn.Sequential):
        from collections import OrderedDict
        return nn.Sequential(OrderedDict((name, vision_component(child)) for name, child in source.named_children()))
    if kind == 'LayerScale2d':
        return VisionScale(source)
    if kind == 'MultiQueryAttention2d':
        return VisionAttention(source)
    orders = {
        'ConvNormAct': ('conv', 'bn', 'aa'),
        'EdgeResidual': ('conv_exp', 'bn1', 'aa', 'se', 'conv_pwl', 'bn2', 'drop_path'),
        'UniversalInvertedResidual': ('dw_start', 'pw_exp', 'dw_mid', 'se', 'pw_proj', 'dw_end', 'layer_scale', 'drop_path'),
        'MobileAttention': ('norm', 'attn', 'layer_scale', 'drop_path'),
    }
    if kind in orders:
        return VisionWiring(source, tuple(name for name in orders[kind] if getattr(source, name, None) is not None))
    raise ValueError(f'Unimplemented active Gemma3n vision component: {kind}')


class VisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        import timm
        with torch.device('meta'):
            template = timm.create_model(c.architecture, pretrained=False, **(c.model_args or {}))
        if type(template).__name__ != 'MobileNetV5Encoder':
            raise ValueError('Preserve Gemma3n constructor MobileNetV5 encoder architecture')
        self.indices = template.msfa_indices
        self.resolution = template.msfa.output_resolution
        self.conv_stem = vision_component(template.conv_stem)
        self.blocks = vision_component(template.blocks)
        self.msfa = nn.Module()
        self.msfa.ffn = vision_component(template.msfa.ffn)
        self.msfa.norm = vision_component(template.msfa.norm)
        self.interpolate = Interpolate()

    def forward(self, x):
        intermediates = []
        x = self.conv_stem(x)
        if 0 in self.indices:
            intermediates.append(x)
        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            if index in self.indices:
                intermediates.append(x)
        size = intermediates[0].shape[-2:]
        aligned = [self.interpolate(x, size=size, mode='nearest') if x.shape[-2:] != size else x
                   for x in intermediates]
        x = self.msfa.ffn(torch.cat(aligned, 1))
        if size != self.resolution:
            if any(a % b for a, b in zip(size, self.resolution)):
                x = self.interpolate(x, size=self.resolution, mode='bilinear', align_corners=False)
            else:
                x = AvgPool2d(tuple(a//b for a, b in zip(size, self.resolution)))(x)
        # Preserve timm NCHW storage before the subsequent channel reduction.
        return self.msfa.norm(x).contiguous()


class AudioCumulativeNorm(nn.Module):
    def __init__(self, channels, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps
        self.sum, self.prefix, self.normalize = Gemma3nRowSum(), Gemma3nPrefixSum(), ZeroSafeVarianceNormalize()

    def forward(self, x):
        # Preserve HF's actual two-prefix formula. It accumulates each frame's
        # squared deviation from that frame's prefix mean, not E[x²]-E[x]².
        xf = x.float()
        b, t = xf.shape[:2]
        flat = xf.reshape(b, t, -1)
        count = torch.arange(1, t+1, device=x.device, dtype=torch.float32)[None, :, None] * flat.shape[-1]
        mean = self.prefix(self.sum(flat)) / count
        centered = flat - mean
        squared = product(centered, centered)
        variance = self.prefix(self.sum(squared)) / count
        normed = self.normalize(flat, mean.expand_as(flat), (variance+self.eps).expand_as(flat))
        return product(normed.reshape_as(xf), self.weight.float()).to(x.dtype)


class AudioSubsampleBlock(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        kh, kw = c.sscp_conv_kernel_size[index]
        self.padding = (1, 1, 0, kh-1)
        self.conv = Conv2d(1 if index == 0 else c.sscp_conv_channel_size[index-1],
                           c.sscp_conv_channel_size[index], (kh, kw), c.sscp_conv_stride_size[index], bias=False)
        self.norm = AudioCumulativeNorm(c.sscp_conv_channel_size[index], c.sscp_conv_group_norm_eps)
        self.activation = ReLU()

    def forward(self, x):
        x = self.conv(torch.nn.functional.pad(x, self.padding).to(self.conv.weight.dtype))
        x = self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return self.activation(x)


class AudioSubsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_0, self.conv_1 = AudioSubsampleBlock(c, 0), AudioSubsampleBlock(c, 1)
        freq = c.input_feat_size
        for kernel, stride in zip(c.sscp_conv_kernel_size, c.sscp_conv_stride_size):
            freq = (freq+2-kernel[1])//stride[1]+1
        self.input_proj_linear = Linear(freq*c.sscp_conv_channel_size[-1], c.hidden_size, bias=False)

    def forward(self, x):
        x = self.conv_1(self.conv_0(x[:, None])).permute(0, 2, 3, 1).contiguous()
        return self.input_proj_linear(x.flatten(2))


class AudioRelativePosition(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.pos_proj = Linear(c.hidden_size, c.hidden_size, bias=False)
        self.mm = BMM()

    def forward(self, q, k):
        import math
        b, u, w, n, h = q.shape
        context = k.shape[2]
        positions = torch.arange(max(0, self.c.conf_attention_context_left-1),
                                 -self.c.conf_attention_context_right-1, -1, device=q.device)[None, :, None]
        times = self.c.hidden_size//2
        inv = torch.exp(torch.arange(times, device=q.device) * (-math.log(1e4)/max(times-1, 1)))
        # The actual HF loader retains this explicitly FP32 metadata buffer.
        angles = positions.float()*inv[None, None]
        sinusoid = torch.cat((angles.sin(), angles.cos()), -1).to(q.dtype)
        f = sinusoid.shape[1]
        positional = self.pos_proj(sinusoid).reshape(f, n, h)
        queries = q.permute(0, 3, 1, 2, 4)
        content = self.mm(queries, k.permute(0, 3, 1, 4, 2))
        relative = self.mm(queries.reshape(b, n, u*w, h), positional.permute(1, 2, 0)).reshape(b, n, u, w, f)
        padded = torch.nn.functional.pad(relative, (0, context+1-f))
        shifted = padded.reshape(b, n, u, w*(context+1))[..., :w*context].reshape(b, n, u, w, context)
        return content + shifted


class AudioAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.heads, self.dim = c.conf_num_attention_heads, c.hidden_size//c.conf_num_attention_heads
        self.chunk, self.left, self.right = c.conf_attention_chunk_size, max(0, c.conf_attention_context_left-1), c.conf_attention_context_right
        self.relative_position_embedding = AudioRelativePosition(c)
        self.per_dim_scale = nn.Parameter(torch.empty(self.dim))
        for name in ('q_proj', 'k_proj', 'v_proj'):
            setattr(self, name, Linear(c.hidden_size, c.hidden_size, bias=False))
        self.tanh, self.softmax, self.mm = Tanh(), Softmax(), BMM()
        self.register_buffer('fixed_per_dim_scale', torch.empty(self.dim), persistent=False)

    def load_fixed_scale(self):
        # Fixed-weight softplus may occur once during loading. Keep the native
        # per_dim_scale parameter for exact source-state accounting.
        self.fixed_per_dim_scale.copy_(torch.nn.functional.softplus(self.per_dim_scale))

    def pad_time(self, x, left, right):
        return torch.cat((x.new_zeros(x.shape[0], left, *x.shape[2:]), x,
                          x.new_zeros(x.shape[0], right, *x.shape[2:])), 1)

    def context(self, x):
        x = self.pad_time(x, self.left, self.right+self.chunk-1)
        return x.unfold(1, self.chunk+self.left+self.right, self.chunk).movedim(-1, 2).contiguous()

    def forward(self, x, padding):
        shape = (*x.shape[:2], self.heads, self.dim)
        q, k, v = [getattr(self, name)(x).reshape(shape).contiguous() for name in ('q_proj', 'k_proj', 'v_proj')]
        # Nonpersistent scalar q_scale is initialized in FP32, then rounded by
        # HF's loader to its requested dtype. Match that fixed scalar exactly.
        scale = (torch.tensor(self.dim**-.5)/torch.nn.functional.softplus(torch.tensor(0.))).to(x)
        q = product(q * scale, self.fixed_per_dim_scale)
        b, t = x.shape[:2]
        blocks = (t+self.chunk-1)//self.chunk
        q = self.pad_time(q, 0, blocks*self.chunk-t).reshape(b, blocks, self.chunk, self.heads, self.dim)
        k, v = self.context(k), self.context(v)
        valid = self.context(~padding)
        context = self.chunk+self.left+self.right
        qi = torch.arange(self.chunk, device=x.device)[:, None]
        ki = torch.arange(context, device=x.device)[None]
        causal = (ki >= qi) & (ki <= qi+self.left+self.right)
        allowed = valid[:, None, :, None] & causal[None, None, None]
        logits = self.relative_position_embedding(q, k)
        cap = x.new_tensor(self.c.conf_attention_logit_cap)
        logits = self.tanh(logits/cap)*cap
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        probabilities = self.softmax(logits.float()).to(x.dtype)
        output = self.mm(probabilities.permute(0, 2, 1, 3, 4).reshape(-1, self.chunk, context),
                         v.permute(0, 1, 3, 2, 4).reshape(-1, context, self.dim))
        return output.reshape(b, blocks, self.heads, self.chunk, self.dim).permute(0, 1, 3, 2, 4).reshape(b, blocks*self.chunk, -1)[:, :t]


def audio_clip(x, bound):
    bound = x.new_tensor(bound)
    return DFineClamp()(x, -bound, bound)


class AudioFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.pre_layer_norm, self.post_layer_norm = Norm(c.hidden_size), Norm(c.hidden_size)
        self.ffw_layer_1 = Linear(c.hidden_size, 4*c.hidden_size, bias=False)
        self.ffw_layer_2 = Linear(4*c.hidden_size, c.hidden_size, bias=False)
        self.act = SiLU()

    def forward(self, x):
        h = self.pre_layer_norm(audio_clip(x, self.c.gradient_clipping))
        h = audio_clip(self.ffw_layer_2(self.act(self.ffw_layer_1(h))), self.c.gradient_clipping)
        return x + self.post_layer_norm(h)*self.c.conf_residual_weight


class AudioAttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.pre_attn_norm, self.post_norm = Norm(c.hidden_size), Norm(c.hidden_size)
        self.attn = AudioAttention(c)
        self.post = Linear(c.hidden_size, c.hidden_size, bias=False)

    def forward(self, x, padding):
        h = self.attn(self.pre_attn_norm(audio_clip(x, self.c.gradient_clipping)), padding)
        return x + self.post_norm(audio_clip(self.post(h), self.c.gradient_clipping))


class AudioLightConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.pre_layer_norm, self.conv_norm = Norm(c.hidden_size, c.rms_norm_eps), Norm(c.hidden_size, c.rms_norm_eps)
        self.linear_start = Linear(c.hidden_size, 2*c.hidden_size, bias=False)
        self.depthwise_conv1d = Conv1d(c.hidden_size, c.hidden_size, c.conf_conv_kernel_size,
                                     groups=c.hidden_size, bias=False)
        self.linear_end = Linear(c.hidden_size, c.hidden_size, bias=False)
        self.sigmoid, self.act = Sigmoid(), SiLU()

    def forward(self, x):
        left, right = self.linear_start(self.pre_layer_norm(x)).chunk(2, -1)
        h = product(left.float(), self.sigmoid(right.float())).to(x.dtype)
        h = self.depthwise_conv1d(torch.nn.functional.pad(h.transpose(1, 2), (self.c.conf_conv_kernel_size-1, 0)))
        h = self.conv_norm(audio_clip(h.transpose(1, 2), self.c.gradient_clipping))
        return x + self.linear_end(self.act(h))


class AudioBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.ffw_layer_start, self.ffw_layer_end = AudioFFN(c), AudioFFN(c)
        self.attention, self.lconv1d = AudioAttentionBlock(c), AudioLightConv(c)
        self.norm = Norm(c.hidden_size)

    def forward(self, x, padding):
        x = self.attention(self.ffw_layer_start(x), padding)
        x = self.lconv1d(x.masked_fill(padding[..., None], 0.))
        return self.norm(audio_clip(self.ffw_layer_end(x), self.c.gradient_clipping))


class AudioModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.subsample_conv_projection = AudioSubsample(c)
        self.conformer = nn.ModuleList(AudioBlock(c) for _ in range(c.conf_num_hidden_layers))

    def forward(self, x, padding):
        x = self.subsample_conv_projection(x)
        stride = self.c.sscp_conv_stride_size[0][0]*self.c.sscp_conv_stride_size[1][0]
        indices = (torch.arange(x.shape[1], device=x.device)*stride).clamp(max=padding.shape[1]-1)
        padding = padding[:, indices]
        for block in self.conformer:
            x = block(x, padding)
        x = x[:, ::self.c.conf_reduction_factor]
        padding = padding[:, ::self.c.conf_reduction_factor]
        return x.masked_fill(padding[..., None], 0.), padding


class MultimodalEmbedder(nn.Module):
    def __init__(self, c, text):
        super().__init__()
        self.offset, self.size = c.vocab_offset, c.vocab_size
        self.embedding = Embedding(c.vocab_size, c.hidden_size)
        self.hard_embedding_norm = Norm(c.hidden_size, c.rms_norm_eps)
        self.soft_embedding_norm = Norm(c.hidden_size, c.rms_norm_eps)
        self.embedding_projection = Linear(c.hidden_size, text.hidden_size, bias=False)
        self.embedding_post_projection_norm = Norm(text.hidden_size, c.rms_norm_eps, False)

    def forward(self, ids=None, soft=None):
        x = self.hard_embedding_norm(self.embedding(ids-self.offset)) if soft is None else self.soft_embedding_norm(soft)
        return self.embedding_post_projection_norm(self.embedding_projection(x))


class MultimodalForConditionalGeneration(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.model = nn.Module()
        self.model.language_model = TextModel(c.text_config)
        self.model.audio_tower = AudioModel(c.audio_config)
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.timm_model = VisionModel(c.vision_config)
        self.model.embed_audio = MultimodalEmbedder(c.audio_config, c.text_config)
        self.model.embed_vision = MultimodalEmbedder(c.vision_config, c.text_config)
        self.lm_head = Linear(c.text_config.hidden_size, c.text_config.vocab_size, bias=False)
        self.tanh = Tanh()

    def forward(self, input_ids, pixel_values=None, input_features=None, input_features_mask=None,
                attention_mask=None, past_key_values=None):
        c, m = self.c, self.model
        text = c.text_config
        embeds = m.language_model.embed_tokens(input_ids)
        ple_ids = input_ids.masked_fill((input_ids < 0) | (input_ids >= text.vocab_size_per_layer_input), 0)
        ple = m.language_model.embed_tokens_per_layer(ple_ids).reshape(*input_ids.shape, text.num_hidden_layers,
                                                                     text.hidden_size_per_layer_input)
        vision_mask = (input_ids >= c.vision_config.vocab_offset) & (input_ids < c.audio_config.vocab_offset)
        audio_mask = input_ids >= c.audio_config.vocab_offset
        for mask, embedder in ((vision_mask, m.embed_vision), (audio_mask, m.embed_audio)):
            ids = input_ids.masked_fill(~mask, embedder.offset+embedder.size-1)
            hard = embedder(ids=ids).to(embeds)
            embeds = torch.where(mask[..., None], hard, embeds)
        images = audio = None
        if pixel_values is not None:
            vision = m.vision_tower.timm_model(pixel_values)
            vision = vision.reshape(vision.shape[0], c.vision_config.hidden_size, c.vision_soft_tokens_per_image).transpose(1, 2)
            images = m.embed_vision(soft=vision*c.vision_config.hidden_size**.5).to(embeds)
            mask = input_ids == c.image_token_id
            if mask.sum().item() != images.shape[0]*images.shape[1]:
                raise ValueError('Image placeholder count does not match complete image features')
            embeds = embeds.masked_scatter(mask[..., None].expand_as(embeds), images)
        if input_features is not None:
            if input_features_mask is None:
                raise ValueError('Audio inputs require their valid-frame mask')
            audio_raw, padding = m.audio_tower(input_features, ~input_features_mask)
            audio = m.embed_audio(soft=audio_raw)
            padding_ids = input_ids.new_tensor([[text.vocab_size-1]])
            padding_embed = m.embed_audio(ids=padding_ids)
            audio = torch.where(padding[..., None], padding_embed, audio)
            extra = c.audio_soft_tokens_per_image-audio.shape[1]
            if extra < 0:
                raise ValueError('Audio exceeds the selected soft-token budget')
            audio = torch.cat((audio, padding_embed.expand(audio.shape[0], extra, -1)), 1).to(embeds)
            mask = input_ids == c.audio_token_id
            if mask.sum().item() != audio.shape[0]*audio.shape[1]:
                raise ValueError('Audio placeholder count does not match complete audio features')
            embeds = embeds.masked_scatter(mask[..., None].expand_as(embeds), audio)
        output = m.language_model(inputs_embeds=embeds, per_layer_inputs=ple,
                                  attention_mask=attention_mask, past_key_values=past_key_values)
        logits = self.lm_head(output['last_hidden_state'])
        if text.final_logit_softcapping is not None:
            cap = text.final_logit_softcapping
            logits = self.tanh(logits/cap)*cap
        return dict(output, logits=logits, image_hidden_states=images, audio_hidden_states=audio)


def build_from_config(config, device, dtype):
    return MultimodalForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    def source_name(name):
        return name.replace('.embed_tokens.emb.', '.embed_tokens.').replace(
            '.embed_tokens_per_layer.emb.', '.embed_tokens_per_layer.').replace('.embedding.emb.', '.embedding.')
    mapped = {name: state_dict[source_name(name)] for name in model.state_dict()}
    unused = set(state_dict)-{source_name(name) for name in mapped}
    if unused:
        raise ValueError(f'Unmapped active Gemma3n state: {sorted(unused)}')
    model.load_state_dict(mapped, strict=True, assign=True)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, AudioAttention):
                module.load_fixed_scale()
            elif isinstance(module, Attention):
                module.load_rotary_metadata()
    model.mapping_counts = {'mapped':len(mapped), 'unused':len(unused)}


def make_workloads(model, inputs, config, *, case=None):
    def flatten(output):
        result = {name: output[name] for name in ('logits', 'image_hidden_states', 'audio_hidden_states')
                  if output[name] is not None}
        for index, pair in output['past_key_values'].layers.items():
            for name, tensor in zip(('key', 'value'), pair):
                result[f'past_key_values.{index}.{name}'] = tensor.transpose(1, 2)
        return result
    if case is None or case.get('workload') != 'causal_lm_continuation':
        return {'forward':Workload(run=lambda:flatten(model(**inputs)))}
    prefix = inputs['input_ids'].shape[1]-2
    state = {}
    def initial():
        current = dict(inputs, input_ids=inputs['input_ids'][:, :prefix])
        if 'attention_mask' in current:
            current['attention_mask'] = current['attention_mask'][:, :prefix]
        return model(**current)
    def advance(index, previous):
        current = {'input_ids':inputs['input_ids'][:, prefix+index:prefix+index+1], 'past_key_values':previous}
        if 'attention_mask' in inputs:
            current['attention_mask'] = inputs['attention_mask'][:, :prefix+index+1]
        return model(**current)
    def prepare(index):
        previous = initial()['past_key_values']
        for step in range(index):
            previous = advance(step, previous)['past_key_values']
        state['previous'] = previous
    return {'prefill':Workload(run=lambda:flatten(initial())),
            'decode_1':Workload(run=lambda:flatten(advance(0,state['previous'])),prepare=lambda:prepare(0)),
            'decode_2':Workload(run=lambda:flatten(advance(1,state['previous'])),prepare=lambda:prepare(1))}
