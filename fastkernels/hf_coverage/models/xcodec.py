"""Xcodec's default joint HuBERT/DAC encoding and acoustic reconstruction."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.hf_coverage.models.dac import CodecStack
from fastkernels.hf_coverage.models.encodec import Codebook, elu
from fastkernels.hf_coverage.models.unispeech import GroupFeatureConv, PostNormWaveformModel
from fastkernels.hf_coverage.models.unispeech import load_state_dict_into as load_hubert
from fastkernels.hf_coverage.models.wav2vec2 import PositionConv
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.tensor_ops import Pad


class HubertStates(PostNormWaveformModel):
    def forward(self, input_values):
        hidden = input_values[:, None]
        for layer in self.feature_extractor:
            hidden = layer(hidden)
        hidden, _ = self.feature_projection(hidden.transpose(1, 2))
        hidden = self.layer_norm(hidden + self.position_conv(hidden))
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        return torch.stack(states)


class SemanticResidual(nn.Module):
    def __init__(self, config, width, dilation):
        super().__init__()
        self.activation = elu()
        self.conv1 = Conv1dNative(width, width, config.unit_kernel_size,
                                  padding=((config.unit_kernel_size - 1) // 2) * dilation,
                                  dilation=dilation, bias=False)
        self.conv2 = Conv1dNative(width, width, 1, bias=False)

    def forward(self, hidden):
        return hidden + self.conv2(self.activation(self.conv1(self.activation(hidden))))


class SemanticBlock(nn.Module):
    def __init__(self, config, source, target, stride):
        super().__init__()
        self.res_units = nn.ModuleList(SemanticResidual(config, source, d) for d in config.block_dilations)
        kernel = 3 if stride == 1 else 2 * stride
        self.conv = Conv1dNative(source, target, kernel, stride=stride, padding=(kernel - 1) // 2)

    def forward(self, hidden):
        for unit in self.res_units:
            hidden = unit(hidden)
        return self.conv(hidden)


class SemanticEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.semantic_hidden_size
        self.conv = Conv1dNative(width, width, config.kernel_size, padding=config.kernel_size // 2, bias=False)
        blocks = []
        for ratio, stride in zip(config.channel_ratios, config.strides):
            target = int(config.semantic_hidden_size * ratio)
            blocks.append(SemanticBlock(config, width, target, stride))
            width = target
        self.conv_blocks = nn.ModuleList(blocks)

    def forward(self, hidden):
        hidden = self.conv(hidden)
        for block in self.conv_blocks:
            hidden = block(hidden)
        return hidden


class Xcodec(nn.Module):
    def __init__(self, config):
        super().__init__()
        acoustic, semantic = config.acoustic_model_config, config.semantic_model_config
        self.semantic_model = HubertStates(semantic,
            [GroupFeatureConv(semantic, i) for i in range(len(semantic.conv_dim))], PositionConv(semantic))
        self.acoustic_encoder, self.acoustic_decoder = CodecStack(acoustic), CodecStack(acoustic, decoder=True)
        self.acoustic_decoder.tanh = nn.Identity()
        for block in self.acoustic_decoder.block:
            block.conv_t1.output_padding = block.conv_t1.stride % 2
        self.encoder_semantic = SemanticEncoder(config)
        self.fc, self.fc2 = Linear(config.hidden_size, config.hidden_size), Linear(config.hidden_size, acoustic.hidden_size)
        self.codebooks = nn.ModuleList(Codebook(config) for _ in range(config.num_quantizers))
        self.pad, self.reduce = Pad(), SegmentCSR()
        self.padding = config.hop_length // 2

    def semantic_features(self, input_values):
        padded = self.pad(input_values, (self.padding, self.padding))
        states = self.semantic_model(padded[:, 0])
        offsets = torch.tensor([0, states.shape[0]], device=states.device)
        semantic = self.reduce(states.flatten(1).float(), offsets, reduce='mean').to(states.dtype).reshape(states.shape[1:])
        return semantic

    def forward(self, input_values):
        semantic = self.encoder_semantic(self.semantic_features(input_values).transpose(1, 2))
        padded = self.pad(input_values, (self.padding, self.padding))
        length = input_values.shape[-1]
        for block in self.acoustic_encoder.block:
            conv = block.conv1
            length = (length + 2 * conv.padding - conv.weight.shape[-1]) // conv.stride + 1
        acoustic = self.acoustic_encoder(padded if length != semantic.shape[-1] else input_values)
        residual = self.fc(torch.cat((acoustic, semantic), dim=1).transpose(1, 2)).transpose(1, 2)
        indices = []
        for codebook in self.codebooks:
            codes = codebook.encode(residual)
            residual = residual - codebook.decode(codes)
            indices.append(codes)
        quantized = torch.tensor(0.0, device=input_values.device)
        for codebook, codes in zip(self.codebooks, indices):
            quantized = quantized + codebook.decode(codes)
        acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        return {'audio_codes': torch.stack(indices, dim=1),
                'audio_values': self.acoustic_decoder(acoustic)[..., :input_values.shape[-1]]}


def resolve_config(config):
    # These HF @properties are absent from the serialized configuration.
    config = SimpleNamespace(**(dict(config) if isinstance(config, dict) else vars(config)))
    config.semantic_hidden_size = config.semantic_model_config.hidden_size
    config.hidden_size = config.acoustic_model_config.hidden_size + config.semantic_hidden_size
    config.hop_length = math.prod(config.acoustic_model_config.downsampling_ratios)
    frame_rate = math.ceil(config.sample_rate / config.hop_length)
    config.num_quantizers = int(1000 * config.target_bandwidths[-1] //
                               (frame_rate * math.ceil(math.log2(config.codebook_size))))
    return config


def build_from_config(config, device, dtype):
    config = resolve_config(config)
    semantic = config.semantic_model_config
    if (semantic.model_type != 'hubert' or semantic.feat_extract_norm != 'group'
            or semantic.do_stable_layer_norm or semantic.hidden_act != 'gelu'
            or config.acoustic_model_config.model_type != 'dac'):
        raise ValueError('Xcodec case preserves the published HuBERT-base and DAC towers')
    return Xcodec(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state, config):
    prefix = 'semantic_model.'
    load_hubert(model.semantic_model, {k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}, config.semantic_model_config)
    mapped, consumed = {}, {k for k in state if k.startswith(prefix)}
    for name, target in model.state_dict().items():
        if name.startswith(prefix):
            mapped[name] = target
            continue
        source = name
        if name.startswith('codebooks.'):
            _, index, field = name.split('.', 2)
            field = 'codebook.embed' if field == 'embedding.emb.weight' else field
            source = f'quantizer.quantizers.{index}.{field}'
        value = state[source]
        if name.endswith('.alpha'):
            value = value.reshape(-1)
        if value.shape != target.shape:
            raise ValueError(f'Xcodec weight mismatch: {source}')
        mapped[name], consumed = value, consumed | {source}
    unused = set(state) - consumed
    allowed = {k for k in state if k.startswith(('fc1.', 'decoder_semantic.'))}
    allowed |= {f'quantizer.quantizers.{i}.codebook.{field}' for i in range(len(model.codebooks))
                for field in ('inited','cluster_size','embed_avg')}
    if unused != allowed:
        raise ValueError(f'Xcodec unexpected unused state: {unused ^ allowed}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
