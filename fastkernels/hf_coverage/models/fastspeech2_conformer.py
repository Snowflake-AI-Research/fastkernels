"""FastSpeech2Conformer + HiFi-GAN using admitted nearest-even/count expansion.

Selected checkpoint task includes the complete waveform vocoder. FP32 is the
native executable dtype: HF's regulator allocates FP32 before the decoder.
"""
from __future__ import annotations

import math
from typing import Any
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.tensor_ops import Cat, Exp
from fastkernels.tasks.baseline.L1.pointtransformerv3_offsets import Offset2Batch
from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from ..runner import Workload
from .univnet import LeakyReLU


class NearestNonnegative(nn.Module):
    """clamp(round(x),min=0).long() for finite, materializable durations.

    Parity is activation computation expressed through admitted casts, fixed
    half scaling and required subtraction; it is not treated as metadata.
    Two CodecTop1 predicates distinguish strict and tie-inclusive comparison.
    """
    def __init__(self):
        super().__init__()
        self.relu, self.predicate = ReLU(), CodecTop1()

    def forward(self, hidden):
        hidden = self.relu(hidden)
        base = hidden.long()
        base_float = base.to(hidden.dtype)
        fraction = hidden - base_float
        half_base = (base_float * .5).long()
        odd = base - half_base - half_base
        half = torch.full_like(fraction, .5)
        greater = self.predicate(torch.stack((half, fraction), -1))
        greater_equal = 1 - self.predicate(torch.stack((fraction, half), -1))
        increment = torch.stack((greater, greater_equal), -1).gather(-1, odd[..., None]).squeeze(-1)
        return base + increment


class BatchNorm1dViaBatchNorm2d(BatchNorm2d):
    """Eval BatchNorm1d on [B,C,T] via FastKernels BatchNorm2d on [B,C,1,T]."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise RuntimeError(f"BatchNorm1dViaBatchNorm2d expects [B,C,T], got {tuple(hidden_states.shape)}")
        return super().forward(hidden_states.unsqueeze(2)).squeeze(2)


def _module_config(config: Any, name: str) -> Any:
    value = getattr(config, f"{name}_config", None)
    if value is not None:
        return value
    prefix = f"{name}_"
    return {
        "num_attention_heads": getattr(config, f"{prefix}num_attention_heads"),
        "layers": getattr(config, f"{prefix}layers"),
        "kernel_size": getattr(config, f"{prefix}kernel_size"),
        "attention_dropout_rate": getattr(config, f"{prefix}attention_dropout_rate"),
        "dropout_rate": getattr(config, f"{prefix}dropout_rate"),
        "positional_dropout_rate": getattr(config, f"{prefix}positional_dropout_rate"),
        "linear_units": getattr(config, f"{prefix}linear_units"),
        "normalize_before": getattr(config, f"{prefix}normalize_before"),
        "concat_after": getattr(config, f"{prefix}concat_after"),
    }


def _cfg_get(config: Any, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class FastSpeech2ConformerPredictorLayer(nn.Module):
    def __init__(self, input_channels: int, num_chans: int, kernel_size: int):
        super().__init__()
        self.conv = Conv1dNative(
            input_channels,
            num_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.activation = ReLU()
        self.layer_norm = LayerNorm(num_chans, promote_fp32=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.conv(hidden_states))
        hidden_states = self.layer_norm(hidden_states.transpose(1, -1)).transpose(1, -1)
        return hidden_states


class FastSpeech2ConformerDurationPredictor(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.log_domain_offset = 1.0
        layers = []
        for layer_idx in range(int(config.duration_predictor_layers)):
            input_channels = int(config.hidden_size) if layer_idx == 0 else int(config.duration_predictor_channels)
            layers.append(
                FastSpeech2ConformerPredictorLayer(
                    input_channels,
                    int(config.duration_predictor_channels),
                    int(config.duration_predictor_kernel_size),
                )
            )
        self.conv_layers = nn.ModuleList(layers)
        self.linear = Linear(int(config.duration_predictor_channels), 1)
        self.exp = Exp()
        self.round_duration = NearestNonnegative()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.linear(hidden_states.transpose(1, -1)).squeeze(-1)
        return self.round_duration(self.exp(hidden_states) - self.log_domain_offset)


class FastSpeech2ConformerVariancePredictor(nn.Module):
    def __init__(self, config: Any, num_layers: int, num_chans: int, kernel_size: int):
        super().__init__()
        layers = []
        for layer_idx in range(int(num_layers)):
            input_channels = int(config.hidden_size) if layer_idx == 0 else int(num_chans)
            layers.append(FastSpeech2ConformerPredictorLayer(input_channels, int(num_chans), int(kernel_size)))
        self.conv_layers = nn.ModuleList(layers)
        self.linear = Linear(int(num_chans), 1)

    def forward(self, encoder_hidden_states: torch.Tensor, padding_masks: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.linear(hidden_states.transpose(1, 2))
        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)
        return hidden_states


class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv = Conv1dNative(1, int(out_channels), int(kernel_size), padding=int(padding))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)


class DurationLengthRegulator(nn.Module):
    """Genuine scalar-count expansion via existing Offset2Batch; no raw scan.

    This uses one repeat operation per input token. Launch count is unoptimized,
    while storage is linear in the required output. Native all-zero fallback
    mutates duration_outputs and native output allocation remains FP32.
    """
    def __init__(self):
        super().__init__()
        self.repeat_indices = Offset2Batch()
        self.maximum = CodecTop1()

    def forward(self, hidden, durations, speaking_speed=1.0):
        if speaking_speed != 1.0:
            raise ValueError('This declared workload preserves native speaking_speed=1')
        values = durations.flatten().float()
        maximum = values[self.maximum(values)]
        positive = self.maximum(torch.stack((torch.zeros_like(maximum), maximum)))
        if not positive.item():
            durations.fill_(1)
        rows = []
        for tokens, counts in zip(hidden, durations):
            pieces = [tokens[index:index + 1].index_select(0, self.repeat_indices(count.reshape(1)))
                      for index, count in enumerate(counts)]
            rows.append(torch.cat(pieces, dim=0))
        output = hidden.new_zeros((hidden.shape[0], max(row.shape[0] for row in rows), hidden.shape[-1]), dtype=torch.float32)
        for index, row in enumerate(rows):
            output[index, :row.shape[0]] = row
        return output


class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    def __init__(self, config: Any, layer_id: int):
        super().__init__()
        in_conv_dim = int(config.num_mel_bins) if layer_id == 0 else int(config.speech_decoder_postnet_units)
        out_conv_dim = (
            int(config.num_mel_bins)
            if layer_id == int(config.speech_decoder_postnet_layers) - 1
            else int(config.speech_decoder_postnet_units)
        )
        self.conv = Conv1dNative(
            in_conv_dim,
            out_conv_dim,
            int(config.speech_decoder_postnet_kernel),
            padding=(int(config.speech_decoder_postnet_kernel) - 1) // 2,
            bias=False,
        )
        self.batch_norm = BatchNorm1dViaBatchNorm2d(out_conv_dim)
        self.activation = Tanh() if layer_id < int(config.speech_decoder_postnet_layers) - 1 else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.batch_norm(self.conv(hidden_states))
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states


class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.feat_out = Linear(int(config.hidden_size), int(config.num_mel_bins) * int(config.reduction_factor))
        self.layers = nn.ModuleList(
            [FastSpeech2ConformerBatchNormConvLayer(config, i) for i in range(int(config.speech_decoder_postnet_layers))]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs_before_postnet = self.feat_out(hidden_states).view(
            hidden_states.size(0), -1, int(self.config.num_mel_bins)
        )
        layer_output = outputs_before_postnet.transpose(1, 2)
        for layer in self.layers:
            layer_output = layer(layer_output)
        outputs_after_postnet = outputs_before_postnet + layer_output.transpose(1, 2)
        return outputs_before_postnet, outputs_after_postnet


class FastSpeech2ConformerAttention(nn.Module):
    def __init__(self, config: Any, module_config: Any):
        super().__init__()
        self.num_heads = int(_cfg_get(module_config, "num_attention_heads"))
        self.hidden_size = int(config.hidden_size)
        self.dim_key = self.hidden_size // self.num_heads
        self.head_dim = self.dim_key
        self.linear_q = Linear(self.hidden_size, self.hidden_size)
        self.linear_k = Linear(self.hidden_size, self.hidden_size)
        self.linear_v = Linear(self.hidden_size, self.hidden_size)
        self.linear_out = Linear(self.hidden_size, self.hidden_size)
        self.linear_pos = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.bmm = BatchMatMul()
        self.softmax = Softmax(dim=-1)
        self.relative_shift_cat = Cat(dim=-1)

    def _bmm4(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch, heads, rows, depth = a.shape
        cols = b.shape[-1]
        out = self.bmm(a.reshape(batch * heads, rows, depth), b.reshape(batch * heads, depth, cols))
        return out.reshape(batch, heads, rows, cols)

    def shift_relative_position_tensor(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
        pos_tensor_padded = self.relative_shift_cat((zero_pad, pos_tensor))
        pos_tensor_padded = pos_tensor_padded.view(*pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2))
        return pos_tensor_padded[:, :, 1:].view_as(pos_tensor)[:, :, :, : pos_tensor.size(-1) // 2 + 1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pos_emb: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.linear_q(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        key_states = self.linear_k(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        value_states = self.linear_v(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)

        if pos_emb is None:
            raise RuntimeError("FastSpeech2ConformerAttention requires pos_emb")
        bsz_pos = pos_emb.size(0)
        pos_encoding = self.linear_pos(pos_emb).view(bsz_pos, -1, self.num_heads, self.head_dim)

        query_with_bias_u = (query_states + self.pos_bias_u).transpose(1, 2)
        query_with_bias_v = (query_states + self.pos_bias_v).transpose(1, 2)
        key_for_scores = key_states.permute(0, 2, 3, 1)
        pos_for_scores = pos_encoding.permute(0, 2, 3, 1)
        matrix_ac = self._bmm4(query_with_bias_u, key_for_scores)
        matrix_bd = self.shift_relative_position_tensor(self._bmm4(query_with_bias_v, pos_for_scores))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.dim_key)

        if attention_mask is not None:
            expected_size = (bsz, 1, q_len)
            if tuple(attention_mask.size()) != expected_size:
                raise ValueError(f"Attention mask should be of size {expected_size}, but is {tuple(attention_mask.size())}")
            attention_mask = attention_mask.unsqueeze(1).eq(0)
            min_value = float(torch.finfo(scores.dtype).min)
            scores = scores.masked_fill(attention_mask, min_value)
            attn_weights = self.softmax(scores).masked_fill(attention_mask, 0.0)
        else:
            attn_weights = self.softmax(scores)
        attn_output = self._bmm4(attn_weights, value_states.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.linear_out(attn_output)
        return attn_output, attn_weights if output_attentions else None


class FastSpeech2ConformerConvolutionModule(nn.Module):
    def __init__(self, config: Any, module_config: Any):
        super().__init__()
        channels = int(config.hidden_size)
        kernel_size = int(_cfg_get(module_config, "kernel_size"))
        self.pointwise_conv1 = Conv1dNative(channels, 2 * channels, kernel_size=1, bias=bool(config.convolution_bias))
        self.depthwise_conv = Conv1dNative(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bool(config.convolution_bias),
        )
        self.norm = BatchNorm1dViaBatchNorm2d(channels)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1dNative(channels, channels, kernel_size=1, bias=bool(config.convolution_bias))
        self.sigmoid = Sigmoid()
        self.product = ProductGate()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        value, gate = hidden_states.chunk(2, dim=1)
        hidden_states = self.product(torch.cat((self.sigmoid(gate).transpose(1, 2), value.transpose(1, 2)), -1)).transpose(1, 2)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.activation(self.norm(hidden_states))
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    def __init__(self, config: Any, module_config: Any):
        super().__init__()
        input_channels = int(config.hidden_size)
        hidden_channels = int(_cfg_get(module_config, "linear_units"))
        kernel_size = int(config.positionwise_conv_kernel_size)
        self.conv1 = Conv1dNative(input_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.activation = ReLU()
        self.conv2 = Conv1dNative(hidden_channels, input_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(-1, 1)
        hidden_states = self.activation(self.conv1(hidden_states))
        hidden_states = self.conv2(hidden_states)
        return hidden_states.transpose(-1, 1)


class FastSpeech2ConformerEncoderLayer(nn.Module):
    def __init__(self, config: Any, module_config: Any):
        super().__init__()
        self.self_attn = FastSpeech2ConformerAttention(config, module_config)
        self.feed_forward = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
        self.macaron_style = bool(config.use_macaron_style_in_conformer)
        if self.macaron_style:
            self.feed_forward_macaron = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
            self.ff_macaron_layer_norm = LayerNorm(int(config.hidden_size), promote_fp32=False)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        self.use_cnn_module = bool(config.use_cnn_in_conformer)
        if self.use_cnn_module:
            self.conv_module = FastSpeech2ConformerConvolutionModule(config, module_config)
            self.conv_layer_norm = LayerNorm(int(config.hidden_size), promote_fp32=False)
            self.final_layer_norm = LayerNorm(int(config.hidden_size), promote_fp32=False)
        else:
            self.conv_module = None
        self.ff_layer_norm = LayerNorm(int(config.hidden_size), promote_fp32=False)
        self.self_attn_layer_norm = LayerNorm(int(config.hidden_size), promote_fp32=False)
        self.normalize_before = bool(_cfg_get(module_config, "normalize_before"))
        self.concat_after = bool(_cfg_get(module_config, "concat_after"))
        if self.concat_after:
            self.concat_linear = Linear(int(config.hidden_size) * 2, int(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.macaron_style:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)
            hidden_states = residual + self.ff_scale * self.feed_forward_macaron(hidden_states)
            if not self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        attention_output, attention_scores = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            pos_emb=pos_emb,
            output_attentions=output_attentions,
        )
        if self.concat_after:
            hidden_states = residual + self.concat_linear(torch.cat((hidden_states, attention_output), dim=-1))
        else:
            hidden_states = residual + attention_output
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.use_cnn_module:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)
            hidden_states = residual + self.conv_module(hidden_states)
            if not self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)
        hidden_states = residual + self.ff_scale * self.feed_forward(hidden_states)
        if not self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)
        if self.conv_module is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, attention_scores if output_attentions else None


class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    def __init__(self, config: Any, module_config: Any):
        super().__init__()
        del module_config
        self.embed_dim = int(config.hidden_size)
        self.input_scale = math.sqrt(self.embed_dim)
        self.max_len = 5000
        self.exp = Exp()
        self.relative_table_cat = Cat(dim=1)
        self.register_buffer(
            "pos_enc",
            self.extend_pos_enc(torch.tensor(0.0).expand(1, self.max_len)),
            persistent=False,
        )

    def extend_pos_enc(self, x: torch.Tensor, pos_enc: torch.Tensor | None = None) -> torch.Tensor:
        if pos_enc is not None and pos_enc.size(1) >= x.size(1) * 2 - 1:
            if pos_enc.dtype != x.dtype or pos_enc.device != x.device:
                pos_enc = pos_enc.to(dtype=x.dtype, device=x.device)
            return pos_enc
        pos_enc_positive = torch.zeros(x.size(1), self.embed_dim, device=x.device)
        pos_enc_negative = torch.zeros(x.size(1), self.embed_dim, device=x.device)
        position = torch.arange(0, x.size(1), dtype=torch.int64, device=x.device).float().unsqueeze(1)
        div_term = self.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.int64, device=x.device).float()
            * -(math.log(10000.0) / self.embed_dim)
        )
        pos_enc_positive[:, 0::2] = torch.sin(position * div_term)
        pos_enc_positive[:, 1::2] = torch.cos(position * div_term)
        pos_enc_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pos_enc_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pos_enc_positive = torch.flip(pos_enc_positive, [0]).unsqueeze(0)
        pos_enc_negative = pos_enc_negative[1:].unsqueeze(0)
        return self.relative_table_cat((pos_enc_positive, pos_enc_negative)).to(device=x.device, dtype=x.dtype)

    def forward(self, feature_representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.pos_enc = self.extend_pos_enc(feature_representation, self.pos_enc)
        hidden_states = feature_representation * self.input_scale
        center_idx = self.pos_enc.size(1) // 2
        pos_emb = self.pos_enc[:, center_idx - hidden_states.size(1) + 1 : center_idx + hidden_states.size(1)]
        return hidden_states, pos_emb


class FastSpeech2ConformerEncoder(nn.Module):
    def __init__(self, config: Any, module_config: Any, use_encoder_input_layer: bool = False):
        super().__init__()
        self.embed = Embedding(int(config.vocab_size), int(config.hidden_size), padding_idx=0) if use_encoder_input_layer else None
        self.pos_enc = FastSpeech2ConformerRelPositionalEncoding(config, module_config)
        self.conformer_layers = nn.ModuleList(
            [FastSpeech2ConformerEncoderLayer(config, module_config) for _ in range(int(_cfg_get(module_config, "layers")))]
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        feature_representation = self.embed(input_tensor) if self.embed is not None else input_tensor
        hidden_states, pos_emb = self.pos_enc(feature_representation)
        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        for layer in self.conformer_layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states, attention = layer(hidden_states, pos_emb, attention_mask, output_attentions)
            if all_self_attentions is not None:
                all_self_attentions.append(attention)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": tuple(all_hidden_states) if all_hidden_states is not None else None,
            "attentions": tuple(all_self_attentions) if all_self_attentions is not None else None,
        }


class FastSpeech2ConformerModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.reduction_factor = int(config.reduction_factor)
        self.stop_gradient_from_pitch_predictor = bool(config.stop_gradient_from_pitch_predictor)
        self.stop_gradient_from_energy_predictor = bool(config.stop_gradient_from_energy_predictor)
        self.encoder = FastSpeech2ConformerEncoder(config, _module_config(config, "encoder"), use_encoder_input_layer=True)
        self.duration_predictor = FastSpeech2ConformerDurationPredictor(config)
        self.pitch_predictor = FastSpeech2ConformerVariancePredictor(
            config,
            int(config.pitch_predictor_layers),
            int(config.pitch_predictor_channels),
            int(config.pitch_predictor_kernel_size),
        )
        self.pitch_embed = FastSpeech2ConformerVarianceEmbedding(
            int(config.hidden_size),
            int(config.pitch_embed_kernel_size),
            (int(config.pitch_embed_kernel_size) - 1) // 2,
        )
        self.energy_predictor = FastSpeech2ConformerVariancePredictor(
            config,
            int(config.energy_predictor_layers),
            int(config.energy_predictor_channels),
            int(config.energy_predictor_kernel_size),
        )
        self.energy_embed = FastSpeech2ConformerVarianceEmbedding(
            int(config.hidden_size),
            int(config.energy_embed_kernel_size),
            (int(config.energy_embed_kernel_size) - 1) // 2,
        )
        self.length_regulator = DurationLengthRegulator()
        self.decoder = FastSpeech2ConformerEncoder(config, _module_config(config, "decoder"), use_encoder_input_layer=False)
        self.speech_decoder_postnet = FastSpeech2ConformerSpeechDecoderPostnet(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        del return_dict
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        text_masks = attention_mask.unsqueeze(-2)
        encoder_outputs = self.encoder(
            input_ids,
            text_masks,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = encoder_outputs["last_hidden_state"]
        duration_mask = ~attention_mask.bool()
        pitch_input = hidden_states.detach() if self.stop_gradient_from_pitch_predictor else hidden_states
        energy_input = hidden_states.detach() if self.stop_gradient_from_energy_predictor else hidden_states
        pitch_predictions = self.pitch_predictor(pitch_input, duration_mask.unsqueeze(-1))
        energy_predictions = self.energy_predictor(energy_input, duration_mask.unsqueeze(-1))
        duration_predictions = self.duration_predictor(hidden_states)
        duration_predictions = duration_predictions.masked_fill(duration_mask, 0.0)
        hidden_states = hidden_states + self.energy_embed(energy_predictions) + self.pitch_embed(pitch_predictions)
        hidden_states = self.length_regulator(hidden_states, duration_predictions, float(self.config.speaking_speed))
        decoder_outputs = self.decoder(
            hidden_states,
            None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        outputs_before_postnet, outputs_after_postnet = self.speech_decoder_postnet(decoder_outputs["last_hidden_state"])
        return {
            "spectrogram": outputs_after_postnet,
            "outputs_before_postnet": outputs_before_postnet,
            "encoder_last_hidden_state": encoder_outputs["last_hidden_state"],
            "encoder_hidden_states": encoder_outputs["hidden_states"],
            "encoder_attentions": encoder_outputs["attentions"],
            "decoder_hidden_states": decoder_outputs["hidden_states"],
            "decoder_attentions": decoder_outputs["attentions"],
            "duration_outputs": duration_predictions,
            "pitch_outputs": pitch_predictions,
            "energy_outputs": energy_predictions,
        }


class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int] | tuple[int, ...], leaky_relu_slope: float):
        super().__init__()
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.activation = LeakyReLU(leaky_relu_slope)
        self.convs1 = nn.ModuleList(
            [
                Conv1dNative(
                    channels,
                    channels,
                    int(kernel_size),
                    dilation=int(dil),
                    padding=(int(kernel_size) * int(dil) - int(dil)) // 2,
                )
                for dil in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                Conv1dNative(channels, channels, int(kernel_size), padding=(int(kernel_size) - 1) // 2)
                for _ in dilation
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = conv1(self.activation(hidden_states))
            hidden_states = conv2(self.activation(hidden_states))
            hidden_states = hidden_states + residual
        return hidden_states


class FastSpeech2ConformerHifiGan(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = Conv1dNative(int(config.model_in_dim), int(config.upsample_initial_channel), kernel_size=7, padding=3)
        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                ConvTranspose1d(
                    int(config.upsample_initial_channel) // (2**i),
                    int(config.upsample_initial_channel) // (2 ** (i + 1)),
                    kernel_size=int(kernel_size),
                    stride=int(upsample_rate),
                    padding=(int(kernel_size) - int(upsample_rate)) // 2,
                )
            )
        self.resblocks = nn.ModuleList()
        channels = int(config.upsample_initial_channel)
        for i in range(len(self.upsampler)):
            channels = int(config.upsample_initial_channel) // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(
                    HifiGanResidualBlock(channels, int(kernel_size), dilation, float(config.leaky_relu_slope))
                )
        self.conv_post = Conv1dNative(channels, 1, kernel_size=7, padding=3)
        self.tanh = Tanh()
        self.activation = LeakyReLU(config.leaky_relu_slope)
        self.final_activation = LeakyReLU(.01)
        self.fixed_scales = None
        self.register_buffer("mean", torch.zeros(int(config.model_in_dim)))
        self.register_buffer("scale", torch.ones(int(config.model_in_dim)))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if bool(self.config.normalize_before):
            # Fixed-weight channel scales retain native division rounding.
            spectrogram = torch.cat([(spectrogram[..., i:i + 1] - self.mean[i]) / scale
                                     for i, scale in enumerate(self.fixed_scales)], dim=-1)
        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)
        hidden_states = spectrogram.transpose(2, 1)
        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = self.upsampler[i](self.activation(hidden_states))
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state = res_state + self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels
        hidden_states = self.conv_post(self.final_activation(hidden_states))
        hidden_states = self.tanh(hidden_states)
        if not is_batched:
            return hidden_states.squeeze(0).transpose(1, 0).view(-1)
        return hidden_states.squeeze(1)


class FastSpeech2ConformerWithHifiGan(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = FastSpeech2ConformerModel(config.model_config)
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        model_outputs = self.model(input_ids, **kwargs)
        spectrogram = model_outputs["spectrogram"]
        waveform = self.vocoder(spectrogram)
        return {**model_outputs, "waveform": waveform}



def build_from_config(config, device, dtype):
    text = config.model_config
    if dtype != torch.float32:
        raise ValueError('Pinned HF length_regulator allocates FP32; declared FastSpeech task is FP32')
    if (text.num_speakers is not None or text.num_languages is not None or text.speaker_embed_dim is not None
            or text.speaking_speed != 1.0 or text.reduction_factor != 1):
        raise ValueError('FastSpeech adapter selects native unconditioned speaking_speed1/reduction1 task')
    return FastSpeech2ConformerWithHifiGan(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        source = name.replace('.emb.weight', '.weight')
        value = state_dict[source]
        if value.shape != target.shape:
            raise ValueError(f'FastSpeech weight shape mismatch {source}: {value.shape} != {target.shape}')
        mapped[name] = value
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped FastSpeech weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
    model.vocoder.fixed_scales = tuple(float(value) for value in state_dict['vocoder.scale'].tolist())


def make_workloads(model, inputs, config, case=None):
    keys = ('spectrogram', 'encoder_last_hidden_state', 'duration_outputs', 'pitch_outputs', 'energy_outputs', 'waveform')
    return {'forward': Workload(run=lambda: {key: value for key, value in model(**inputs).items() if key in keys})}
