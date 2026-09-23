"""Qwen2.5 Omni token-to-waveform composition using existing operations."""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.cosyvoice3_dit import (
    AdaLayerNormZero, AdaLayerNormZero_Final, DiTBlock, FeedForward, TimestepEmbedding,
)
from fastkernels.tasks.baseline.L3.oasis_autoencoder_kl import DiagonalGaussianDistribution
from ..patches.product_gate import ProductGate
from ..patches.qwen_omni_snake import SnakeBeta


class Multiply(nn.Module):
    def __init__(self):
        super().__init__()
        self.product = ProductGate()

    def forward(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        return self.product(torch.cat((left, right), dim=-1))


class ReflectConv(Conv1dNative):
    """Reflect-pad by copying slices, then execute the existing convolution."""

    def __init__(self, source, target, kernel=1, dilation=1):
        super().__init__(source, target, kernel, dilation=dilation)
        self.reflect = (kernel - 1) * dilation // 2

    def forward(self, hidden):
        p = self.reflect
        if p:
            hidden = torch.cat((hidden[..., 1:p + 1].flip(-1), hidden,
                                hidden[..., -p - 1:-1].flip(-1)), dim=-1)
        return super().forward(hidden)


class TDNN(nn.Module):
    def __init__(self, source, target, kernel, dilation):
        super().__init__()
        self.conv = ReflectConv(source, target, kernel, dilation)
        self.activation = ReLU()

    def forward(self, hidden):
        return self.activation(self.conv(hidden))


class Res2Net(nn.Module):
    def __init__(self, width, scale, kernel, dilation):
        super().__init__()
        self.scale = scale
        self.blocks = nn.ModuleList([TDNN(width // scale, width // scale, kernel, dilation)
                                     for _ in range(scale - 1)])

    def forward(self, hidden):
        outputs = []
        for index, part in enumerate(hidden.chunk(self.scale, dim=1)):
            if index == 0:
                output = part
            else:
                output = self.blocks[index - 1](part if index == 1 else part + output)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


class SqueezeExcitation(nn.Module):
    def __init__(self, width, squeeze):
        super().__init__()
        self.conv1, self.conv2 = ReflectConv(width, squeeze), ReflectConv(squeeze, width)
        self.relu, self.sigmoid = ReLU(), Sigmoid()
        self.mean, self.multiply = GlobalAvgPool2d(keepdim=True), Multiply()

    def forward(self, hidden):
        mean = self.mean(hidden[..., None])[..., 0]
        return self.multiply(hidden, self.sigmoid(self.conv2(self.relu(self.conv1(mean)))))


class SERes2Net(nn.Module):
    def __init__(self, source, target, scale, squeeze, kernel, dilation):
        super().__init__()
        self.tdnn1, self.tdnn2 = TDNN(source, target, 1, 1), TDNN(target, target, 1, 1)
        self.res2net_block = Res2Net(target, scale, kernel, dilation)
        self.se_block = SqueezeExcitation(target, squeeze)

    def forward(self, hidden):
        return hidden + self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(hidden))))


class WeightedStatistics(nn.Module):
    """Weighted sums and positive variance root via existing reduction/BN ops."""

    def __init__(self):
        super().__init__()
        self.multiply, self.sum = Multiply(), SegmentCSR()
        self.floor = MaxPool2d((1, 2))
        # torch.batch_norm rejects eps=0. At v>=1e-12 this positive epsilon
        # is below half an FP32 ulp, so v+eps rounds back to the same v.
        self.root = BatchNorm2d(1, eps=1e-30, affine=False).eval()
        self.root._non_persistent_buffers_set.update(('running_mean', 'running_var', 'num_batches_tracked'))

    def reduce(self, hidden):
        length = hidden.shape[-1]
        offsets = torch.arange(hidden.numel() // length + 1, device=hidden.device) * length
        return self.sum(hidden.flatten(), offsets, reduce='sum').view(hidden.shape[:-1])

    def forward(self, hidden, weights):
        mean = self.reduce(self.multiply(hidden, weights))
        centered = hidden - mean[..., None]
        variance = self.reduce(self.multiply(weights, self.multiply(centered, centered)))
        floor = torch.full_like(variance, 1e-12)
        variance = self.floor(torch.stack((variance, floor), -1).reshape(1, 1, -1, 2)).reshape_as(variance)
        flat = variance.flatten()
        self.root.running_mean, self.root.running_var = torch.zeros_like(flat), flat
        # Positive v / sqrt(v) evaluates sqrt(v) with the admitted existing BN.
        std = self.root(flat.reshape(1, -1, 1, 1)).reshape_as(variance)
        return mean, std


class AttentivePooling(nn.Module):
    def __init__(self, channels, attention_channels):
        super().__init__()
        self.tdnn = TDNN(channels * 3, attention_channels, 1, 1)
        self.tanh, self.conv = Tanh(), ReflectConv(attention_channels, channels)
        self.statistics, self.softmax = WeightedStatistics(), Softmax(dim=2)

    def forward(self, hidden):
        weights = hidden.new_full((hidden.shape[0], 1, hidden.shape[-1]), 1 / hidden.shape[-1])
        mean, std = self.statistics(hidden, weights)
        context = torch.cat((hidden, mean[..., None].expand_as(hidden), std[..., None].expand_as(hidden)), dim=1)
        weights = self.softmax(self.conv(self.tanh(self.tdnn(context))))
        mean, std = self.statistics(hidden, weights)
        return torch.cat((mean, std), dim=1)[..., None]


class SpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.enc_channels
        self.blocks = nn.ModuleList([TDNN(config.mel_dim, channels[0], config.enc_kernel_sizes[0], config.enc_dilations[0])])
        self.blocks.extend(SERes2Net(channels[i - 1], channels[i], config.enc_res2net_scale,
                                    config.enc_se_channels, config.enc_kernel_sizes[i], config.enc_dilations[i])
                           for i in range(1, len(channels) - 1))
        self.mfa = TDNN(channels[-1], channels[-1], config.enc_kernel_sizes[-1], config.enc_dilations[-1])
        self.asp = AttentivePooling(channels[-1], config.enc_attention_channels)
        self.fc = ReflectConv(channels[-1] * 2, config.enc_dim)

    def forward(self, hidden):
        hidden, outputs = hidden.transpose(1, 2), []
        for block in self.blocks:
            hidden = block(hidden)
            outputs.append(hidden)
        return self.fc(self.asp(self.mfa(torch.cat(outputs[1:], dim=1)))).squeeze(-1)


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = Linear(config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim, config.hidden_size)
        self.spk_encoder = SpeakerEncoder(config)

    def forward(self, hidden, speaker, reference, code, null_code, apply_cfg):
        if apply_cfg:
            hidden = torch.cat((hidden, hidden))
            speaker = torch.cat((speaker, torch.zeros_like(speaker)))
            reference = torch.cat((reference, torch.zeros_like(reference)))
            code = torch.cat((code, null_code))
        reference = self.spk_encoder(reference)[:, None].expand(-1, hidden.shape[1], -1)
        return self.proj(torch.cat((hidden, reference, code, speaker), dim=-1))


class CodecEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codec_embed = Embedding(config.num_embeds + 1, config.emb_dim)
        self.repeats = config.repeats

    def forward(self, code, drop=False):
        return self.codec_embed(torch.zeros_like(code) if drop else code).repeat_interleave(self.repeats, dim=1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.head_dim
        inner = self.heads * self.width
        self.to_q, self.to_k, self.to_v = [Linear(config.hidden_size, inner) for _ in range(3)]
        self.to_out = nn.Sequential(Linear(inner, config.hidden_size), nn.Identity())
        self.attention, self.multiply = DenseAttention(backend='sdpa'), Multiply()

    def forward(self, x, mask=None, rope=None):
        batch, length = x.shape[:2]
        q, k, v = [op(x).view(batch, length, self.heads, self.width).transpose(1, 2)
                   for op in (self.to_q, self.to_k, self.to_v)]
        cosine, sine = rope
        # Pinned training convention: rotate only head zero, adjacent pairs,
        # while the coefficient table concatenates its two frequency halves.
        for tensor in (q, k):
            first = tensor[:, :1]
            even, odd = first.reshape(*first.shape[:-1], -1, 2).unbind(-1)
            rotated = torch.stack((-odd, even), -1).flatten(-2)
            tensor[:, :1] = self.multiply(first, cosine) + self.multiply(rotated, sine)
        context = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return self.to_out(context.reshape(batch, length, -1))


class DecoderLayer(DiTBlock):
    """Reuse the unchanged CosyVoice DiT modulation/gating forward."""

    def __init__(self, config, index):
        nn.Module.__init__(self)
        self.attn_norm = AdaLayerNormZero(config.hidden_size)
        self.attn = Attention(config)
        self.ff_norm = LayerNorm(config.hidden_size, eps=1e-6, elementwise_affine=False, promote_fp32=False)
        self.ff = FeedForward(config.hidden_size, mult=config.ff_mult, approximate='tanh')
        self.ahead, self.backward = int(index in config.look_ahead_layers), int(index in config.look_backward_layers)


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_embed = TimestepEmbedding(config.hidden_size)
        self.text_embed, self.input_embed = CodecEmbedding(config), InputEmbedding(config)
        self.transformer_blocks = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm_out, self.proj_out = AdaLayerNormZero_Final(config.hidden_size), Linear(config.hidden_size, config.mel_dim)

    def forward(self, hidden_states, condition_vector, speaker_embedding, quantized_code, time_step, apply_cfg=True):
        if time_step.ndim == 0:
            time_step = time_step.repeat(hidden_states.shape[0])
        time = self.time_embed(time_step)
        hidden = self.input_embed(hidden_states, speaker_embedding, condition_vector,
                                  self.text_embed(quantized_code), self.text_embed(quantized_code, True) if apply_cfg else None, apply_cfg)
        # Position and block-mask metadata are independent of activation values.
        positions = torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.float32)
        width = self.config.head_dim
        frequency = 1 / (self.config.rope_parameters['rope_theta'] ** (torch.arange(0, width, 2, device=hidden.device).float() / width))
        angles = positions[:, None] * frequency[None]
        angles = torch.cat((angles, angles), -1)
        rope = (angles.cos()[None, None].to(hidden.dtype), angles.sin()[None, None].to(hidden.dtype))
        blocks = torch.arange(hidden.shape[1], device=hidden.device) // self.config.block_size
        difference = blocks[None] - blocks[:, None]
        for layer in self.transformer_blocks:
            mask = ((difference >= -layer.backward) & (difference <= layer.ahead))[None, None]
            hidden = layer(hidden, time, mask=mask, rope=rope)
        return self.proj_out(self.norm_out(hidden, time))

    def sample(self, conditioning, reference_mel, code, num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0):
        length = code.shape[1] * self.config.repeats
        if code.shape[0] != 1 or length > self.config.max_position_embeddings:
            raise ValueError('Native token2wav sampling requires batch one within the position limit')
        parameters = torch.zeros((1, length, self.config.mel_dim * 2), device=code.device, dtype=reference_mel.dtype)
        value = DiagonalGaussianDistribution(parameters, dim=-1).sample()
        speaker = conditioning[:, None].repeat(1, length, 1)
        times = torch.linspace(0, 1, num_steps, device=code.device, dtype=conditioning.dtype)
        if sway_coefficient is not None:
            times += sway_coefficient * (torch.cos(torch.pi / 2 * times) - 1 + times)

        def velocity(time, state):
            output = self(state, reference_mel, speaker, code, time, apply_cfg=guidance_scale >= 1e-5)
            if guidance_scale < 1e-5:
                return output
            guided, null = output.chunk(2)
            return guided + (guided - null) * guidance_scale

        # The supplied time grid fixes all scalar coefficients; this is the
        # native four-evaluation 3/8 Runge-Kutta integration, kept in timing.
        for start, end in zip(times[:-1], times[1:]):
            step = end - start
            k1 = velocity(start, value)
            k2 = velocity(start + step * (1 / 3), value + step * k1 * (1 / 3))
            k3 = velocity(start + step * (2 / 3), value + step * (k2 - k1 * (1 / 3)))
            k4 = velocity(end, value + step * (k1 - k2 + k3))
            value = value + (k1 + 3 * (k2 + k3) + k4) * step / 8
        return value.permute(0, 2, 1)


def kaiser_filter():
    """Prepare the fixed 12-tap, ratio-two anti-alias filter outside forward."""
    attenuation = 2.285 * 5 * math.pi * 1.2 + 7.95
    beta = 0.1102 * (attenuation - 8.7) if attenuation > 50 else 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21)
    window = torch.kaiser_window(12, beta=beta, periodic=False, dtype=torch.float32, device='cpu')
    time = torch.arange(-6, 6, dtype=torch.float32, device='cpu') + 0.5
    filt = 0.5 * window * torch.sinc(0.5 * time)
    return (filt / filt.sum()).view(1, 1, 12)


class Filter(nn.Module):
    def __init__(self, channels, up):
        super().__init__()
        self.up = up
        self.op = (ConvTranspose1d(channels, channels, 12, stride=2, groups=channels, bias=False) if up
                   else Conv1dNative(channels, channels, 12, stride=2, groups=channels, bias=False))
        del self.op._parameters['weight']
        self.op.register_buffer('weight', kaiser_filter().expand(channels, -1, -1).contiguous(), persistent=False)

    def forward(self, hidden):
        left, right = (5, 5) if self.up else (5, 6)
        padded = torch.cat((hidden[..., :1].expand(*hidden.shape[:-1], left), hidden,
                            hidden[..., -1:].expand(*hidden.shape[:-1], right)), -1)
        filtered = self.op(padded)
        return (2 * filtered)[..., 15:-15] if self.up else filtered


class Activation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample, self.downsample = Filter(channels, True), Filter(channels, False)

    def forward(self, hidden):
        return self.downsample(self.act(self.upsample(hidden)))


class AMPBlock(nn.Module):
    def __init__(self, width, kernel, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([Conv1dNative(width, width, kernel, dilation=d, padding=(kernel * d - d) // 2) for d in dilations])
        self.convs2 = nn.ModuleList([Conv1dNative(width, width, kernel, padding=(kernel - 1) // 2) for _ in dilations])
        self.activations = nn.ModuleList([Activation(width) for _ in range(2 * len(dilations))])

    def forward(self, hidden):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]):
            hidden = hidden + c2(a2(c1(a1(hidden))))
        return hidden


class BigVGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.upsample_initial_channel
        self.conv_pre = Conv1dNative(config.mel_dim, width, 7, padding=3)
        self.ups = nn.ModuleList([nn.ModuleList([ConvTranspose1d(width // 2**i, width // 2**(i + 1), k, r, (k - r) // 2)])
                                  for i, (r, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))])
        self.resblocks = nn.ModuleList([AMPBlock(width // 2**(i + 1), k, d)
                                       for i in range(len(self.ups)) for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)])
        final_width = width // 2**len(self.ups)
        self.activation_post, self.conv_post = Activation(final_width), Conv1dNative(final_width, 1, 7, padding=3, bias=False)
        self.clip_max = MaxPool2d((1, 2))

    def clip(self, value):
        # Two-element maxima preserve tiny values that ReLU-plus-offset
        # clamping would round to zero through cancellation around +/-1.
        shape = value.shape
        floor = torch.full_like(value, -1)
        lower = self.clip_max(torch.stack((value, floor), -1).reshape(1, 1, -1, 2)).reshape(shape)
        return -self.clip_max(torch.stack((-lower, floor), -1).reshape(1, 1, -1, 2)).reshape(shape)

    def process_mel_spectrogram(self, mel):
        # exp -> positive floor -> log10 -> normalization has this same
        # saturated finite-domain function. Separate operation rounding is
        # diagnosed explicitly; no claim of bitwise transcendental equality.
        return self.clip(mel * (40 / (115 * math.log(10))) + 75 / 115)

    def forward(self, mel):
        hidden = self.conv_pre(self.process_mel_spectrogram(mel))
        blocks = len(self.config.resblock_kernel_sizes)
        for i, up in enumerate(self.ups):
            hidden = up[0](hidden)
            hidden = sum(self.resblocks[i * blocks + j](hidden) for j in range(blocks)) / blocks
        return self.clip(self.conv_post(self.activation_post(hidden))).squeeze().cpu()


class Token2Wav(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.code2wav_dit_model, self.code2wav_bigvgan_model = DiT(config.dit_config), BigVGAN(config.bigvgan_config)

    def forward(self, code, conditioning, reference_mel, num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0):
        mel = self.code2wav_dit_model.sample(conditioning, reference_mel, code, num_steps, guidance_scale, sway_coefficient)
        return self.code2wav_bigvgan_model(mel)


def build_from_config(config, device, dtype=torch.float32):
    if dtype != torch.float32:
        raise ValueError('Native Qwen2.5 token2wav runs in FP32 inside the mixed-precision parent')
    return Token2Wav(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config=None):
    mapped, used = {}, set()
    for name, parameter in model.state_dict().items():
        source = name.replace('.codec_embed.emb.weight', '.codec_embed.weight')
        source = source.replace('.ff.ff.0.0.', '.ff.ff.0.').replace('.ff.ff.2.', '.ff.ff.3.')
        mapped[name] = state_dict[source]
        if parameter.shape != mapped[name].shape:
            raise ValueError(f'Waveform weight shape mismatch: {name}')
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped waveform weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
