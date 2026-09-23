"""VITS single-speaker inference with stochastic duration and latent flows.

The selected HF example retains relative-key/value text attention, reverse
rational-quadratic duration flows, additive WaveNet flows and HiFiGAN synthesis.
Numerical primitives are existing operations or explicit imported adaptations.
"""

import math
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import Exp
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L3.oasis_autoencoder_kl import DiagonalGaussianDistribution
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.gemma3n_row_stats import Gemma3nPrefixSum
from fastkernels.hf_coverage.patches.forecast_revin import ForecastNormalize, ZeroSafeVarianceNormalize
from fastkernels.hf_coverage.patches.vits_gaussian import TypedDiagonalGaussian
from fastkernels.hf_coverage.models.univnet import LeakyReLU
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.hf_coverage.runner import Workload


class DurationArithmetic(nn.Module):
    """Direct duration/spline compositions; each numerical primitive is visible."""

    def __init__(self):
        super().__init__()
        self.product = ProductGate()
        self.top1 = CodecTop1()
        self.prefix_op = Gemma3nPrefixSum()
        self.reduce = SegmentCSR()
        self.divide_op = ForecastNormalize(tolerance=-float('inf'))
        self.sqrt_op = ZeroSafeVarianceNormalize()
        self.softmax = Softmax()
        self.logsigmoid = LogSigmoid()
        self.exp = Exp()

    def mul(self, a, b):
        a, b = torch.broadcast_tensors(a, b)
        return self.product(torch.cat((a, b), dim=-1))

    def div(self, a, b):
        a, b = torch.broadcast_tensors(a, b)
        return self.divide_op(a[..., None], torch.zeros_like(b), b)[..., 0]

    def sqrt(self, x):
        # Existing supplied-variance normalization: x / sqrt(x), with 0 -> 0.
        return self.sqrt_op(x.float(), torch.zeros_like(x, dtype=torch.float32), x.float()).to(x.dtype)

    def less(self, a, b):
        a, b = torch.broadcast_tensors(a, b)
        return self.top1(torch.stack((a, b), dim=-1)).bool()

    def prefix(self, x):
        shape = x.shape
        rows = x.float().reshape(-1, shape[-1], 1)
        return self.prefix_op(rows).reshape(shape).to(x.dtype)

    def sum_last(self, x):
        width = x.shape[-1]
        offsets = torch.arange(0, x.numel() + 1, width, device=x.device)
        return self.reduce(x.float().reshape(-1), offsets, reduce='sum').reshape(x.shape[:-1]).to(x.dtype)

    def ceil(self, x):
        # Provisionally admitted direct positive-ceil composition; finite x<2^63.
        integer = x.long()
        fraction = x - integer.to(x.dtype)
        increment = self.top1(torch.stack((torch.zeros_like(fraction), fraction), dim=-1))
        return (integer + increment).to(x.dtype)

    def reverse_spline(self, inputs, widths, heights, derivatives, bound):
        low, high = torch.full_like(inputs, -bound), torch.full_like(inputs, bound)
        inside = ~self.less(inputs, low) & ~self.less(high, inputs)
        result = inputs.clone()
        x, w, h, d = inputs[inside], widths[inside], heights[inside], derivatives[inside]
        if not x.numel():
            return result
        bins = w.shape[-1]
        w = .001 + (1. - .001 * bins) * self.softmax(w)
        h = .001 + (1. - .001 * bins) * self.softmax(h)
        cw = torch.cat((w.new_zeros(*w.shape[:-1], 1), self.prefix(w)), dim=-1)
        ch = torch.cat((h.new_zeros(*h.shape[:-1], 1), self.prefix(h)), dim=-1)
        cw, ch = (2 * bound) * cw - bound, (2 * bound) * ch - bound
        cw[..., 0], cw[..., -1] = -bound, bound
        ch[..., 0], ch[..., -1] = -bound, bound
        w, h = cw[..., 1:] - cw[..., :-1], ch[..., 1:] - ch[..., :-1]
        edge = math.log(math.exp(1. - .001) - 1.)
        d = torch.cat((d.new_full((*d.shape[:-1], 1), edge), d,
                       d.new_full((*d.shape[:-1], 1), edge)), dim=-1)
        d = .001 - self.logsigmoid(-d)
        ch[..., -1] += 1e-6
        greater_equal = ~self.less(x[..., None], ch)
        index = (self.sum_last(greater_equal.float()).long() - 1)[..., None]
        def select(t):
            return t.gather(-1, index)[..., 0]
        delta = select(self.div(h, w))
        left, right = select(d), select(d[..., 1:])
        height = select(h)
        difference = x - select(ch)
        combined = left + right - 2 * delta
        product = self.mul(difference, combined)
        a = self.mul(height, delta - left) + product
        b = self.mul(height, left) - product
        c = self.mul(-delta, difference)
        discriminant = self.mul(b, b) - self.mul(4 * a, c)
        root = self.div(2 * c, -b - self.sqrt(discriminant))
        result[inside] = self.mul(root, select(w)) + select(cw)
        # HF also calculates log determinant, but inference discards that value.
        return result


class DilatedDepthSeparableConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.hidden_size, config.duration_predictor_kernel_size
        self.convs_dilated, self.convs_pointwise = nn.ModuleList(), nn.ModuleList()
        self.norms_1, self.norms_2 = nn.ModuleList(), nn.ModuleList()
        for i in range(config.depth_separable_num_layers):
            dilation = kernel ** i
            self.convs_dilated.append(Conv1dNative(width, width, kernel, groups=width,
                dilation=dilation, padding=(kernel * dilation - dilation)//2))
            self.convs_pointwise.append(Conv1dNative(width, width, 1))
            self.norms_1.append(LayerNorm(width, eps=1e-5, promote_fp32=False))
            self.norms_2.append(LayerNorm(width, eps=1e-5, promote_fp32=False))
        self.gelu, self.math = GELU(), DurationArithmetic()

    def forward(self, hidden, mask, conditioning=None):
        if conditioning is not None:
            hidden = hidden + conditioning
        for depthwise, pointwise, norm1, norm2 in zip(
                self.convs_dilated, self.convs_pointwise, self.norms_1, self.norms_2):
            branch = depthwise(self.math.mul(hidden, mask))
            branch = self.gelu(norm1(branch.transpose(1, 2)).transpose(1, 2))
            branch = pointwise(branch)
            branch = self.gelu(norm2(branch.transpose(1, 2)).transpose(1, 2))
            hidden = hidden + branch
        return self.math.mul(hidden, mask)


class ConvFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half = config.depth_separable_channels // 2
        self.width, self.bins, self.bound = config.hidden_size, config.duration_predictor_flow_bins, config.duration_predictor_tail_bound
        self.conv_pre = Conv1dNative(self.half, self.width, 1)
        self.conv_dds = DilatedDepthSeparableConv(config)
        self.conv_proj = Conv1dNative(self.width, self.half * (3*self.bins-1), 1)
        self.math = DurationArithmetic()

    def forward(self, latents, mask, conditioning):
        first, second = latents.split(self.half, dim=1)
        hidden = self.conv_dds(self.conv_pre(first), mask, conditioning)
        hidden = self.math.mul(self.conv_proj(hidden), mask)
        batch, channels, length = first.shape
        hidden = hidden.reshape(batch, channels, -1, length).permute(0, 1, 3, 2)
        second = self.math.reverse_spline(second, hidden[..., :self.bins]/math.sqrt(self.width),
            hidden[..., self.bins:2*self.bins]/math.sqrt(self.width), hidden[..., 2*self.bins:], self.bound)
        return self.math.mul(torch.cat((first, second), dim=1), mask)


class ElementwiseAffine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.translate = nn.Parameter(torch.empty(config.depth_separable_channels, 1))
        self.log_scale = nn.Parameter(torch.empty(config.depth_separable_channels, 1))
        self.math = DurationArithmetic()

    def forward(self, inputs, mask, conditioning=None):
        return self.math.mul(self.math.mul(inputs - self.translate, self.math.exp(-self.log_scale)), mask)


class StochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_pre = Conv1dNative(config.hidden_size, config.hidden_size, 1)
        self.conv_proj = Conv1dNative(config.hidden_size, config.hidden_size, 1)
        self.conv_dds = DilatedDepthSeparableConv(config)
        self.flows = nn.ModuleList([ElementwiseAffine(config)] + [ConvFlow(config) for _ in range(config.duration_predictor_num_flows)])
        self.math = DurationArithmetic()

    def forward(self, inputs, mask, noise_scale):
        hidden = self.conv_dds(self.conv_pre(inputs), mask)
        hidden = self.math.mul(self.conv_proj(hidden), mask)
        # Native duration noise is drawn on CPU in FP32, then transferred/cast.
        normal = DiagonalGaussianDistribution(torch.zeros(inputs.shape[0], 4, inputs.shape[2], device='cpu', dtype=torch.float32))
        latents = normal.sample().to(device=inputs.device, dtype=inputs.dtype) * noise_scale
        flows = list(reversed(self.flows))
        for flow in flows[:-2] + [flows[-1]]:
            latents = flow(latents.flip(1), mask, hidden)
        return latents[:, :1]


class WaveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.hidden_size
        self.in_layers, self.res_skip_layers = nn.ModuleList(), nn.ModuleList()
        count = config.prior_encoder_num_wavenet_layers
        for i in range(count):
            dilation = config.wavenet_dilation_rate**i
            kernel = config.wavenet_kernel_size
            self.in_layers.append(Conv1dNative(self.width, 2*self.width, kernel,
                dilation=dilation, padding=(kernel*dilation-dilation)//2))
            self.res_skip_layers.append(Conv1dNative(self.width, (2 if i<count-1 else 1)*self.width, 1))
        self.tanh, self.sigmoid, self.math = Tanh(), Sigmoid(), DurationArithmetic()

    def forward(self, hidden, mask):
        output = torch.zeros_like(hidden)
        for i, (convolution, projection) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            first, second = convolution(hidden).split(self.width, dim=1)
            if hidden.is_cuda:
                # HF's scripted WaveNet gate fuses after CUDA warmup: both
                # activations and their product stay FP32 until one BF16 store.
                gate = self.math.mul(self.tanh(first.float()), self.sigmoid(second.float())).to(hidden.dtype)
            else:
                gate = self.math.mul(self.tanh(first), self.sigmoid(second))
            branch = projection(gate)
            if i < len(self.in_layers)-1:
                residual, skip = branch.split(self.width, dim=1)
                hidden = self.math.mul(hidden + residual, mask)
                output = output + skip
            else:
                output = output + branch
        return self.math.mul(output, mask)


class ResidualCouplingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half = config.flow_size//2
        self.conv_pre = Conv1dNative(self.half, config.hidden_size, 1)
        self.wavenet = WaveNet(config)
        self.conv_post = Conv1dNative(config.hidden_size, self.half, 1)
        self.math = DurationArithmetic()

    def forward(self, inputs, mask):
        first, second = inputs.split(self.half, dim=1)
        hidden = self.math.mul(self.conv_pre(first), mask)
        mean = self.math.mul(self.conv_post(self.wavenet(hidden, mask)), mask)
        # Native log_stddev is identically zero in this additive coupling layer.
        second = self.math.mul(second - mean, mask)
        return torch.cat((first, second), dim=1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flows = nn.ModuleList([ResidualCouplingLayer(config) for _ in range(config.prior_encoder_num_flows)])

    def forward(self, latents, mask):
        for layer in reversed(self.flows):
            latents = layer(latents.flip(1), mask)
        return latents


def multiply(product, value, mask):
    return product(torch.cat((value, mask.to(value.dtype).expand_as(value)), dim=-1))


class VitsAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.window_size = config.window_size
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size, bias=config.use_bias))
        if self.window_size:
            shape = (1, 2 * self.window_size + 1, self.head_dim)
            self.emb_rel_k = nn.Parameter(torch.empty(shape))
            self.emb_rel_v = nn.Parameter(torch.empty(shape))
        self.bmm, self.softmax, self.pad = BatchMatMul(), Softmax(), Pad()

    def relative_embeddings(self, embeddings, length):
        padding = max(length - self.window_size - 1, 0)
        if padding:
            embeddings = self.pad(embeddings, (0, 0, padding, padding))
        start = max(self.window_size + 1 - length, 0)
        return embeddings[:, start:start + 2 * length - 1]

    def relative_to_absolute(self, value):
        batch, length, _ = value.shape
        value = self.pad(value, (0, 1)).reshape(batch, 2 * length * length)
        value = self.pad(value, (0, length - 1))
        return value.reshape(batch, length + 1, 2 * length - 1)[:, :length, length - 1:]

    def absolute_to_relative(self, value):
        batch, length, _ = value.shape
        value = self.pad(value, (0, length - 1)).reshape(batch, length * (2 * length - 1))
        value = self.pad(value, (length, 0))
        return value.reshape(batch, length, 2 * length)[:, :, 1:]

    def forward(self, hidden, attention_mask=None):
        batch, length, width = hidden.shape
        def heads(value):
            return value.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2).reshape(
                batch * self.heads, length, self.head_dim)
        query = heads(self.q_proj(hidden) * self.head_dim**-0.5)
        key, value = heads(self.k_proj(hidden)), heads(self.v_proj(hidden))
        scores = self.bmm(query, key.transpose(1, 2))
        if self.window_size is not None:
            relative = self.relative_embeddings(self.emb_rel_k, length).expand(batch * self.heads, -1, -1)
            scores = scores + self.relative_to_absolute(self.bmm(query, relative.transpose(1, 2)))
        if attention_mask is not None:
            scores = (scores.reshape(batch, self.heads, length, length) + attention_mask).reshape(
                batch * self.heads, length, length)
        probabilities = self.softmax(scores)
        output = self.bmm(probabilities, value)
        if self.window_size is not None:
            relative = self.relative_embeddings(self.emb_rel_v, length).expand(batch * self.heads, -1, -1)
            output = output + self.bmm(self.absolute_to_relative(probabilities), relative)
        output = output.reshape(batch, self.heads, length, self.head_dim).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output)


class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != 'relu':
            raise ValueError('This VITS case implements the default ReLU feed-forward path')
        kernel = config.ffn_kernel_size
        self.conv_1 = Conv1dNative(config.hidden_size, config.ffn_dim, kernel)
        self.conv_2 = Conv1dNative(config.ffn_dim, config.hidden_size, kernel)
        self.padding = ((kernel - 1) // 2, kernel // 2)
        self.product, self.activation, self.pad = ProductGate(), ReLU(), Pad()

    def forward(self, hidden, padding_mask):
        hidden, mask = hidden.transpose(1, 2), padding_mask.transpose(1, 2)
        hidden = self.conv_1(self.pad(multiply(self.product, hidden, mask), self.padding))
        hidden = self.activation(hidden)
        hidden = self.conv_2(self.pad(multiply(self.product, hidden, mask), self.padding))
        return multiply(self.product, hidden, mask).transpose(1, 2)


class VitsEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = VitsAttention(config)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.feed_forward = VitsFeedForward(config)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden, padding_mask, attention_mask):
        hidden = self.layer_norm(hidden + self.attention(hidden, attention_mask))
        return self.final_layer_norm(hidden + self.feed_forward(hidden, padding_mask))


class VitsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(VitsEncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.product = ProductGate()

    def forward(self, hidden, padding_mask, attention_mask=None):
        if attention_mask is not None:
            batch, length = attention_mask.shape
            mask = hidden.new_zeros((batch, 1, length, length))
            mask.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(hidden.dtype).min)
        else:
            mask = None
        hidden = multiply(self.product, hidden, padding_mask)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask, mask)
        return multiply(self.product, hidden, padding_mask)


class VitsTextEncoder(nn.Module):
    """Returns hidden states and the Gaussian prior mean/log-scale, all [B,T,C]."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.encoder = VitsEncoder(config)
        self.project = Conv1dNative(config.hidden_size, config.flow_size * 2, 1)
        self.product = ProductGate()

    def forward(self, input_ids, padding_mask, attention_mask=None):
        hidden = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        hidden = self.encoder(hidden, padding_mask, attention_mask)
        stats = multiply(self.product, self.project(hidden.transpose(1, 2)).transpose(1, 2), padding_mask)
        mean, log_scale = stats.split(self.config.flow_size, dim=2)
        return hidden, mean, log_scale


class HifiGanResidualBlock(nn.Module):
    def __init__(self, width, kernel, dilations, slope):
        super().__init__()
        self.convs1 = nn.ModuleList(Conv1dNative(width, width, kernel, dilation=d, padding=(kernel * d - d) // 2)
                                   for d in dilations)
        self.convs2 = nn.ModuleList(Conv1dNative(width, width, kernel, padding=(kernel - 1) // 2)
                                   for _ in dilations)
        self.activation = LeakyReLU(slope)

    def forward(self, hidden):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            hidden = hidden + conv2(self.activation(conv1(self.activation(hidden))))
        return hidden


class VitsHifiGan(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.upsample_initial_channel
        self.conv_pre = Conv1dNative(config.flow_size, width, 7, padding=3)
        self.upsampler = nn.ModuleList(ConvTranspose1d(width // 2**i, width // 2**(i + 1), kernel,
            stride=rate, padding=(kernel - rate) // 2)
            for i, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)))
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.resblocks = nn.ModuleList(HifiGanResidualBlock(width // 2**(i + 1), kernel, dilations,
            config.leaky_relu_slope) for i in range(len(self.upsampler))
            for kernel, dilations in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes))
        self.conv_post = Conv1dNative(width // 2**len(self.upsampler), 1, 7, padding=3, bias=False)
        if config.speaker_embedding_size:
            self.cond = Conv1dNative(config.speaker_embedding_size, width, 1)
        self.activation = LeakyReLU(config.leaky_relu_slope)
        # HF calls leaky_relu without a slope after the last upsampling stage.
        self.final_activation, self.tanh = LeakyReLU(0.01), Tanh()

    def forward(self, spectrogram, global_conditioning=None):
        hidden = self.conv_pre(spectrogram)
        if global_conditioning is not None:
            hidden = hidden + self.cond(global_conditioning)
        for i, upsampler in enumerate(self.upsampler):
            hidden = upsampler(self.activation(hidden))
            result = self.resblocks[i * self.num_kernels](hidden)
            for j in range(1, self.num_kernels):
                result = result + self.resblocks[i * self.num_kernels + j](hidden)
            hidden = result / self.num_kernels
        return self.tanh(self.conv_post(self.final_activation(hidden)))


class VitsModel(nn.Module):
    """Selected single-speaker, stochastic-duration inference path."""

    def __init__(self, config):
        super().__init__()
        if not config.use_stochastic_duration_prediction or config.num_speakers != 1 or config.speaker_embedding_size:
            raise ValueError('This case retains the single-speaker stochastic-duration HF example')
        self.config = config
        self.text_encoder = VitsTextEncoder(config)
        self.duration_predictor = StochasticDurationPredictor(config)
        self.flow = ResidualCouplingBlock(config)
        self.decoder = VitsHifiGan(config)
        self.math, self.bmm = DurationArithmetic(), BatchMatMul()

    def forward(self, input_ids, attention_mask=None):
        dtype = self.text_encoder.embed_tokens.emb.weight.dtype
        mask = (torch.ones_like(input_ids) if attention_mask is None else attention_mask).to(dtype)
        hidden, means, log_variances = self.text_encoder(input_ids, mask[..., None], attention_mask)
        input_mask = mask[:, None]
        log_duration = self.duration_predictor(hidden.transpose(1, 2), input_mask, self.config.noise_scale_duration)
        duration = self.math.ceil(self.math.mul(self.math.exp(log_duration), input_mask) / self.config.speaking_rate)
        totals = self.math.sum_last(duration.flatten(1))
        choices = torch.stack((totals, torch.ones_like(totals)), dim=-1)
        selected = self.math.top1(choices)[..., None]
        lengths = choices.gather(-1, selected).squeeze(-1).long()
        max_index = self.math.top1(lengths.float())
        output_length = int(lengths[max_index].item())
        indices = torch.arange(output_length, device=input_ids.device)
        output_mask = self.math.less(indices[None].float(), lengths[:, None].float())[:, None].to(dtype)
        attention_mask_4d = self.math.mul(input_mask[:, :, None, :], output_mask[..., None])
        cumulative = self.math.prefix(duration).reshape(-1, 1)
        frame_indices = torch.arange(output_length, device=input_ids.device, dtype=dtype)
        valid = self.math.less(frame_indices[None], cumulative).to(dtype)
        valid = valid.reshape(input_ids.shape[0], input_ids.shape[1], output_length)
        previous = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, :-1]), dim=1)
        attention = self.math.mul((valid - previous)[:, None].transpose(2, 3), attention_mask_4d)
        means = self.bmm(attention[:, 0], means).transpose(1, 2)
        log_variances = self.bmm(attention[:, 0], log_variances).transpose(1, 2)
        batch, channels, length = means.shape
        # Splitting along batch preserves the native B,C,T transposed layout.
        parameters = means.new_zeros((2*batch, length, channels)).transpose(1, 2)
        noise = TypedDiagonalGaussian(parameters, dim=0).sample()
        latents = means + self.math.mul(noise, self.math.exp(log_variances)) * self.config.noise_scale
        spectrogram = self.math.mul(self.flow(latents, output_mask), output_mask)
        waveform = self.decoder(spectrogram).squeeze(1)
        return {'waveform': waveform, 'sequence_lengths': lengths * math.prod(self.config.upsample_rates),
                'spectrogram': spectrogram}


def build_from_config(config, device, dtype):
    return VitsModel(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    # Inference never enters the posterior encoder or duration-training branches.
    unused_prefixes = ('posterior_encoder.', 'duration_predictor.post_')
    loaded, consumed = {}, set()
    for name, target in model.state_dict().items():
        source = name.replace('embed_tokens.emb.', 'embed_tokens.')
        if source in state_dict:
            tensor = state_dict[source]
            consumed.add(source)
        elif source.endswith('.weight'):
            stem = source[:-len('weight')] + 'parametrizations.weight.'
            norm, direction = stem+'original0', stem+'original1'
            tensor = torch._weight_norm(state_dict[direction], state_dict[norm], dim=0)
            consumed.update((norm, direction))
        else:
            raise KeyError(source)
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError(f'VITS state shape/dtype mismatch for {source}')
        loaded[name] = tensor
    unexpected = [key for key in state_dict if key not in consumed and not key.startswith(unused_prefixes)]
    if unexpected:
        raise KeyError(f'Unexpected VITS state: {unexpected}')
    model.load_state_dict(loaded, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return {'forward': Workload(lambda: model(inputs['input_ids'], inputs.get('attention_mask')))}
