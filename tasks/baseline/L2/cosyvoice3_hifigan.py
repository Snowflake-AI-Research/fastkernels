"""HiFi-GAN vocoder for CosyVoice3 mel-to-waveform conversion.

Adopted from vllm-omni CosyVoice3 code2wav_core/hifigan.py.
Implements the CausalHiFTGenerator and supporting components.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch import pow, sin
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from torch.distributions.uniform import Uniform
from torch.nn import Parameter


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    def __init__(self, channels=512, kernel_size=3, dilations=None, causal=False):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 5]
        self.causal = causal
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1,
                           dilation=dilation,
                           padding=get_padding(kernel_size, dilation))
                    if not causal
                    else CausalConv1d(channels, channels, kernel_size, 1,
                                     dilation=dilation, causal_type="left")
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1,
                           padding=get_padding(kernel_size, 1))
                    if not causal
                    else CausalConv1d(channels, channels, kernel_size, 1,
                                     dilation=1, causal_type="left")
                )
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x):
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class SineGen(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1,
                 noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    @torch.no_grad()
    def forward(self, f0):
        f0 = f0.transpose(1, 2)
        F_mat = torch.zeros(
            (f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i: i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(
            sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves.transpose(1, 2), uv.transpose(1, 2), noise


class SineGen2(nn.Module):
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003, voiced_threshold=0,
                 flag_for_pulse=False, causal=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.causal = causal
        if causal:
            self.rand_ini = torch.rand(1, 9)
            self.rand_ini[:, 0] = 0
            self.sine_waves = torch.rand(1, 300 * 24000, 9)

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        if not self.training and self.causal:
            rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
        else:
            rand_ini = torch.rand(
                f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        if not self.flag_for_pulse:
            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="nearest" if self.causal else "linear",
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        fn = torch.multiply(
            f0,
            torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device),
        )
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if not self.training and self.causal:
            noise = noise_amp * self.sine_waves[:, :sine_waves.shape[1]].to(sine_waves.device)
        else:
            noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0,
                 sinegen_type="1", causal=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        if sinegen_type == "1":
            self.l_sin_gen = SineGen(
                sampling_rate, harmonic_num, sine_amp, add_noise_std,
                voiced_threshold)
        else:
            self.l_sin_gen = SineGen2(
                sampling_rate, upsample_scale, harmonic_num, sine_amp,
                add_noise_std, voiced_threshold, causal=causal)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()
        self.causal = causal
        if causal:
            self.uv = torch.rand(1, 300 * 24000, 1)

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))
        if not self.training and self.causal:
            noise = self.uv[:, :uv.shape[1]] * self.sine_amp / 3
        else:
            noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class CausalConv1dUpsample(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size, 1, padding=0,
            dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)
        self.causal_padding = kernel_size - 1
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        x = self.upsample(x)
        input_timestep = x.shape[2]
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            x = torch.concat([cache, x], dim=2)
        x = super().forward(x)
        assert input_timestep == x.shape[2]
        return x


class CausalConv1dDownSample(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding=0,
            dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)
        self.causal_padding = stride - 1

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            x = torch.concat([cache, x], dim=2)
        x = super().forward(x)
        return x


class CausalConv1d(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 causal_type="left", device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=1, padding=0,
            dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)
        self.causal_padding = (
            int((kernel_size * dilation - dilation) / 2) * 2
            + (kernel_size + 1) % 2
        )
        self.causal_type = causal_type

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        input_timestep = x.shape[2]
        if cache.size(2) == 0:
            cache = torch.zeros(
                x.shape[0], x.shape[1], self.causal_padding).to(x)
        if self.causal_type == "left":
            x = torch.concat([cache, x], dim=2)
        else:
            x = torch.concat([x, cache], dim=2)
        x = super().forward(x)
        assert x.shape[2] == input_timestep
        return x


class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class=1, in_channels=80, cond_channels=512):
        super().__init__()
        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type="right")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x, finalize=True):
        if finalize:
            x = self.condnet[0](x)
        else:
            x = self.condnet[0](
                x[:, :, :-self.condnet[0].causal_padding],
                x[:, :, -self.condnet[0].causal_padding:])
        for i in range(1, len(self.condnet)):
            x = self.condnet[i](x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class CausalHiFTGenerator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=22050,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=None,
        upsample_kernel_sizes=None,
        istft_params=None,
        resblock_kernel_sizes=None,
        resblock_dilation_sizes=None,
        source_resblock_kernel_sizes=None,
        source_resblock_dilation_sizes=None,
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=None,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 8]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16]
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = [7, 11]
        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
            sinegen_type="1" if sampling_rate == 22050 else "2",
            causal=True,
        )
        self.upsample_rates = upsample_rates
        self.f0_upsamp = nn.Upsample(
            scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            CausalConv1d(in_channels, base_channels,
                         conv_pre_look_right + 1, 1, causal_type="right"))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(CausalConv1dUpsample(
                    base_channels // (2 ** i),
                    base_channels // (2 ** (i + 1)),
                    k, u)))

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(CausalConv1d(
                    istft_params["n_fft"] + 2,
                    base_channels // (2 ** (i + 1)), 1, 1,
                    causal_type="left"))
            else:
                self.source_downs.append(CausalConv1dDownSample(
                    istft_params["n_fft"] + 2,
                    base_channels // (2 ** (i + 1)), u * 2, u))
            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(
                    resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, causal=True))

        self.conv_post = weight_norm(CausalConv1d(
            ch, istft_params["n_fft"] + 2, 7, 1, causal_type="left"))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(
            get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.conv_pre_look_right = conv_pre_look_right
        self.f0_predictor = f0_predictor

    def _stft(self, x):
        spec = torch.stft(
            x.float(),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(device=x.device, dtype=torch.float32),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        orig_dtype = magnitude.dtype
        magnitude = torch.clip(magnitude.float(), max=1e2)
        phase = phase.float()
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        out = torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(device=magnitude.device, dtype=torch.float32),
        )
        return out.to(orig_dtype)

    def decode(self, x, s=torch.zeros(1, 1, 0), finalize=True):
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(
                x[:, :, :-self.conv_pre_look_right],
                x[:, :, -self.conv_pre_look_right:])
            s_stft_real = s_stft_real[
                :, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
            s_stft_imag = s_stft_imag[
                :, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        s_stft = s_stft.to(x.dtype)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])
        x = self._istft(magnitude, phase)
        if not finalize:
            x = x[:, :-int(
                np.prod(self.upsample_rates) * self.istft_params["hop_len"])]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat, finalize=True):
        self.f0_predictor.to("cpu")
        f0 = self.f0_predictor(speech_feat.cpu(), finalize=finalize).to(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if finalize:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            generated_speech = self.decode(
                x=speech_feat[:, :, :-self.f0_predictor.condnet[0].causal_padding],
                s=s, finalize=finalize)
        return generated_speech, s
