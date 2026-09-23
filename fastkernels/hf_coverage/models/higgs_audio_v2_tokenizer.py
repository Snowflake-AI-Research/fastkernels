"""Higgs audio tokenizer with native-rate resampling and projected codebooks."""

import math
import torch
from torch import nn
from torchaudio.functional.functional import _get_sinc_resample_kernel

from fastkernels.hf_coverage.models.encodec import Codebook
from fastkernels.hf_coverage.models.xcodec import Xcodec, load_state_dict_into as load_xcodec, make_workloads, resolve_config
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.linear import Linear


class ProjectedCodebook(Codebook):
    def __init__(self, config):
        super().__init__(config)
        self.project_in = Linear(config.hidden_size, config.codebook_dim)
        self.project_out = Linear(config.codebook_dim, config.hidden_size)

    def encode(self, hidden):
        projected = self.project_in(hidden.transpose(1, 2)).transpose(1, 2)
        return super().encode(projected)

    def decode(self, indices):
        quantized = self.embedding(indices)
        return self.project_out(quantized).transpose(1, 2)


class HiggsAudioTokenizer(Xcodec):
    def __init__(self, config):
        super().__init__(config)
        self.codebooks = nn.ModuleList(ProjectedCodebook(config) for _ in range(config.num_quantizers))
        self.source_rate, self.target_rate = config.sample_rate, config.semantic_sample_rate
        divisor = math.gcd(self.source_rate, self.target_rate)
        self.source_step, self.target_step = self.source_rate // divisor, self.target_rate // divisor
        self.width = math.ceil(6 * self.source_step / (min(self.source_step, self.target_step) * 0.99))
        self.resample = Conv1dNative(1, self.target_step, 2 * self.width + self.source_step,
                                    stride=self.source_step, bias=False)
        self.semantic_stride = int(config.hop_length / (config.sample_rate / config.semantic_sample_rate) / config.downsample_factor)

    def prepare_resample(self):
        # Coefficients depend only on sample rates and dtype, never on audio.
        weight = self.resample.weight
        kernel, width = _get_sinc_resample_kernel(self.source_rate, self.target_rate,
            math.gcd(self.source_rate, self.target_rate), device=weight.device, dtype=weight.dtype)
        if width != self.width:
            raise ValueError('Unexpected native resampling filter width')
        with torch.no_grad():
            weight.copy_(kernel)

    def semantic_features(self, input_values):
        length = input_values.shape[-1]
        padded = self.pad(input_values, (self.width, self.width + self.source_step))
        resampled = self.resample(padded).transpose(1, 2).flatten(1)
        target_length = math.ceil(self.target_step * length / self.source_step)
        states = self.semantic_model(self.pad(resampled[:, :target_length], (160, 160)))
        offsets = torch.tensor([0, states.shape[0]], device=states.device)
        semantic = self.reduce(states.flatten(1).float(), offsets, reduce='mean').to(states.dtype).reshape(states.shape[1:])
        return semantic[:, ::self.semantic_stride]


def build_from_config(config, device, dtype):
    config = resolve_config(config)
    semantic = config.semantic_model_config
    if (semantic.model_type != 'hubert' or semantic.feat_extract_norm != 'group'
            or semantic.do_stable_layer_norm or semantic.hidden_act != 'gelu'
            or (config.sample_rate, config.semantic_sample_rate) != (24000, 16000)):
        raise ValueError('Higgs case preserves the published24kHz/16kHz HuBERT/DAC computation')
    model = HiggsAudioTokenizer(config).to(device=device, dtype=dtype).eval()
    model.prepare_resample()
    return model


def load_state_dict_into(model, state, config):
    # Resampling coefficients are prepared constants, not checkpoint parameters.
    load_xcodec(model, {**state, 'resample.weight': model.resample.weight}, config)
