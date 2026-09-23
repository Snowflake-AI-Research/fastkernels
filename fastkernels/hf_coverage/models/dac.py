"""DAC's complete default encode/decode path, including inference-time losses."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.cosyvoice3_hifigan import Snake


class ResidualUnit(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.snake1, self.snake2 = Snake(width), Snake(width)
        self.conv1 = Conv1dNative(width, width, 7, dilation=dilation, padding=3 * dilation)
        self.conv2 = Conv1dNative(width, width, 1)

    def forward(self, hidden):
        output = self.conv2(self.snake2(self.conv1(self.snake1(hidden))))
        crop = (hidden.shape[-1] - output.shape[-1]) // 2
        return (hidden[..., crop:-crop] if crop else hidden) + output


class CodecBlock(nn.Module):
    def __init__(self, width, stride, decoder):
        super().__init__()
        self.decoder = decoder
        output_width = width // 2 if decoder else width * 2
        residual_width = output_width if decoder else width
        self.snake1 = Snake(width)
        conv = ConvTranspose1d if decoder else Conv1dNative
        setattr(self, 'conv_t1' if decoder else 'conv1',
                conv(width, output_width, 2 * stride, stride=stride, padding=math.ceil(stride / 2)))
        self.res_unit1 = ResidualUnit(residual_width, 1)
        self.res_unit2 = ResidualUnit(residual_width, 3)
        self.res_unit3 = ResidualUnit(residual_width, 9)

    def forward(self, hidden):
        if self.decoder:
            hidden = self.conv_t1(self.snake1(hidden))
        hidden = self.res_unit3(self.res_unit2(self.res_unit1(hidden)))
        return hidden if self.decoder else self.conv1(self.snake1(hidden))


class CodecStack(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        self.decoder = decoder
        width = config.decoder_hidden_size if decoder else config.encoder_hidden_size
        self.conv1 = Conv1dNative(config.hidden_size if decoder else 1, width, 7, padding=3)
        blocks = []
        for stride in config.upsampling_ratios if decoder else config.downsampling_ratios:
            blocks.append(CodecBlock(width, stride, decoder))
            width = width // 2 if decoder else width * 2
        self.block = nn.ModuleList(blocks)
        self.snake1 = Snake(width)
        self.conv2 = Conv1dNative(width, 1 if decoder else config.hidden_size,
                                  7 if decoder else 3, padding=3 if decoder else 1)
        self.tanh = Tanh()

    def forward(self, hidden):
        hidden = self.conv1(hidden)
        for block in self.block:
            hidden = block(hidden)
        hidden = self.conv2(self.snake1(hidden))
        return self.tanh(hidden) if self.decoder else hidden


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = Conv1dNative(config.hidden_size, config.codebook_dim, 1)
        self.out_proj = Conv1dNative(config.codebook_dim, config.hidden_size, 1)
        self.codebook = Embedding(config.codebook_size, config.codebook_dim)
        self.normalize, self.product = L2Norm(), ProductGate()
        self.reduce, self.matmul, self.select = SegmentCSR(), BatchMatMul(), CodecTop1()

    def squared_row_norm(self, rows):
        squared = self.product(torch.cat((rows, rows), dim=-1))
        offsets = torch.arange(0, squared.numel() + 1, rows.shape[-1], device=rows.device)
        return self.reduce(squared.flatten().float(), offsets).to(rows.dtype).unsqueeze(-1)

    def forward(self, hidden):
        projected = self.in_proj(hidden)
        batch, width, length = projected.shape
        rows = self.normalize(projected.transpose(1, 2).reshape(-1, width))
        codes = self.normalize(self.codebook.emb.weight)
        dot = self.matmul((2 * rows).unsqueeze(0), codes.T.unsqueeze(0))[0]
        scores = -(self.squared_row_norm(rows) - dot) + self.squared_row_norm(codes).T
        indices = self.select(scores).reshape(batch, length)
        quantized = self.codebook(indices).transpose(1, 2)
        difference = projected.float() - quantized.float()
        squared = self.product(torch.cat((difference, difference), dim=-1))
        offsets = torch.tensor([0, squared.numel()], device=hidden.device)
        loss = self.reduce(squared.flatten(), offsets, reduce='mean').to(projected.dtype)
        # Preserve the actual subtraction/addition rounding of HF's forward STE.
        quantized = projected + (quantized - projected)
        return self.out_proj(quantized), loss, indices, projected


class ResidualQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer(config) for _ in range(config.n_codebooks)])

    def forward(self, hidden):
        total, loss, codes, latents = 0, 0, [], []
        for quantizer in self.quantizers:
            value, term, indices, projected = quantizer(hidden)
            total, hidden, loss = total + value, hidden - value, loss + term
            codes.append(indices)
            latents.append(projected)
        return total, loss, torch.stack(codes, dim=1), torch.cat(latents, dim=1)


class Dac(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder, self.decoder = CodecStack(config), CodecStack(config, decoder=True)
        self.quantizer = ResidualQuantizer(config)
        self.commitment_weight, self.codebook_weight = config.commitment_loss_weight, config.codebook_loss_weight

    def forward(self, input_values):
        quantized, loss, codes, latents = self.quantizer(self.encoder(input_values))
        # Both native MSE terms have identical forward values; preserve separate scaling.
        return {'loss': (self.commitment_weight * loss + self.codebook_weight * loss).expand(input_values.shape[0]),
                'audio_values': self.decoder(quantized).squeeze(1)[..., :input_values.shape[-1]],
                'quantized_representation': quantized, 'audio_codes': codes, 'projected_latents': latents}


def build_from_config(config, device, dtype):
    return Dac(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for name, target in model.state_dict().items():
        source = name.replace('.codebook.emb.', '.codebook.')
        value = state[source]
        if name.endswith('.alpha'):
            value = value.reshape(-1)
        if value.shape != target.shape:
            raise ValueError(f'DAC weight mismatch: {source}')
        mapped[name], consumed = value, consumed | {source}
    if consumed != set(state):
        raise ValueError(f'DAC unmapped weights: {sorted(set(state) - consumed)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
