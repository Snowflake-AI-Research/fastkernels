"""SeamlessM4T v1 text/audio-to-speech from existing transformer and codec ops.

The selected checkpoint keeps both encoders, cached greedy text/unit decoding,
learned integer duration expansion, and every HiFiGAN upsampling stage. The
Expm1 patch preserves native duration rounding near half-integers.
"""

import math
from copy import copy
from types import SimpleNamespace

import torch
from fastkernels.hf_coverage.models.fastspeech2_conformer import NearestNonnegative
from fastkernels.hf_coverage.models.vits import DurationArithmetic, HifiGanResidualBlock
from fastkernels.hf_coverage.patches.seamless_expm1 import Expm1
from torch import nn

from fastkernels.hf_coverage.models.m2m_100 import sinusoidal_table
from fastkernels.hf_coverage.models.univnet import LeakyReLU
from fastkernels.hf_coverage.models.wav2vec2_conformer import (
    GLU,
    ConformerLayer,
    Convolution,
    FeedForward,
    RelativeAttention,
)
from fastkernels.hf_coverage.patches.codec_top1 import CodecTop1
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.pointtransformerv3_offsets import Offset2Batch
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh


class TextAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(width, width))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, memory=None, cache=None, mask=None, causal=False):
        batch, length, width = hidden.shape

        def heads(x):
            return x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        query = heads(self.q_proj(hidden) * self.head_dim**-0.5)
        if memory is not None and cache is not None:
            key, value = cache
        else:
            source = hidden if memory is None else memory
            key, value = heads(self.k_proj(source)), heads(self.v_proj(source))
            if cache is not None:
                key, value = (
                    torch.cat((old, new), dim=2)
                    for old, new in zip(cache, (key, value))
                )
            else:
                key, value = key.contiguous(), value.contiguous()
        scores = self.bmm(
            query.reshape(-1, length, self.head_dim),
            key.reshape(-1, key.shape[2], self.head_dim).transpose(1, 2),
        )
        scores = scores.reshape(batch, self.heads, length, -1)
        if causal:
            qpos = torch.arange(length, device=hidden.device) + key.shape[2] - length
            kpos = torch.arange(key.shape[2], device=hidden.device)
            scores = scores.masked_fill(
                kpos[None, :] > qpos[:, None], torch.finfo(scores.dtype).min
            )
        if mask is not None:
            additive = torch.zeros(
                (batch, 1, length, key.shape[2]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            additive.masked_fill_(
                ~mask[:, None, None, :].bool(), torch.finfo(hidden.dtype).min
            )
            scores = scores + additive
        scores = scores.reshape(-1, length, key.shape[2])
        output = self.bmm(
            self.softmax(scores), value.reshape(-1, key.shape[2], self.head_dim)
        )
        output = (
            output.reshape(batch, self.heads, length, self.head_dim)
            .transpose(1, 2)
            .reshape(batch, length, width)
        )
        return self.out_proj(output), (key, value)


class TextFeedForward(nn.Module):
    def __init__(self, width, inner):
        super().__init__()
        self.fc1, self.fc2 = Linear(width, inner), Linear(inner, width)
        self.activation = ReLU()

    def forward(self, hidden):
        return self.fc2(self.activation(self.fc1(hidden)))


class TextLayer(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        width = config.hidden_size
        self.self_attn = TextAttention(
            width,
            config.decoder_attention_heads
            if decoder
            else config.encoder_attention_heads,
        )
        self.self_attn_layer_norm = LayerNorm(width, promote_fp32=False)
        self.ffn = TextFeedForward(
            width, config.decoder_ffn_dim if decoder else config.encoder_ffn_dim
        )
        self.ffn_layer_norm = LayerNorm(width, promote_fp32=False)
        self.decoder = decoder
        if decoder:
            self.cross_attention = TextAttention(width, config.decoder_attention_heads)
            self.cross_attention_layer_norm = LayerNorm(width, promote_fp32=False)

    def forward(self, hidden, mask=None, memory=None, memory_mask=None, cache=None):
        branch, self_cache = self.self_attn(
            self.self_attn_layer_norm(hidden),
            cache=None if cache is None else cache[0],
            mask=mask,
            causal=self.decoder,
        )
        hidden = hidden + branch
        cross_cache = None
        if memory is not None:
            branch, cross_cache = self.cross_attention(
                self.cross_attention_layer_norm(hidden),
                memory,
                None if cache is None else cache[1],
                memory_mask,
            )
            hidden = hidden + branch
        hidden = hidden + self.ffn(self.ffn_layer_norm(hidden))
        return hidden, (self_cache, cross_cache)


class TextStack(nn.Module):
    def __init__(self, config, *, decoder=False, unit_encoder=False):
        super().__init__()
        self.decoder, self.unit_encoder = decoder, unit_encoder
        self.pad_id = config.pad_token_id
        self.scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        if not unit_encoder:
            self.embed_tokens = Embedding(
                config.vocab_size, config.hidden_size, padding_idx=self.pad_id
            )
            self.register_buffer(
                "positions",
                sinusoidal_table(
                    config.max_position_embeddings + 2, config.hidden_size, self.pad_id
                ),
                persistent=False,
            )
        self.layers = nn.ModuleList(
            TextLayer(config, decoder)
            for _ in range(config.decoder_layers if decoder else config.encoder_layers)
        )
        self.layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)

    def forward(
        self,
        ids=None,
        hidden=None,
        mask=None,
        memory=None,
        memory_mask=None,
        cache=None,
    ):
        if hidden is None:
            hidden = self.embed_tokens(ids) * self.scale
        if not self.unit_encoder:
            valid = ids.ne(self.pad_id).int()
            past = 0 if cache is None else cache[0][0][0].shape[2]
            positions = ((valid.cumsum(1) + past) * valid).long() + self.pad_id
            hidden = hidden + self.positions[positions]
        new_cache = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(
                hidden,
                mask,
                memory,
                memory_mask,
                None if cache is None else cache[index],
            )
            new_cache.append(state)
        return self.layer_norm(hidden), tuple(new_cache)


class UnitModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        unit = copy(config)
        for name, value in (
            config.items() if isinstance(config, dict) else vars(config).items()
        ):
            if name.startswith("t2u_"):
                setattr(unit, name[4:], value)
        self.model = nn.Module()
        self.model.encoder = TextStack(unit, unit_encoder=True)
        self.model.decoder = TextStack(unit, decoder=True)
        self.lm_head = Linear(config.hidden_size, unit.vocab_size, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.emb.weight


class VariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.unit_embed_dim, config.variance_predictor_kernel_size
        self.conv1 = Conv1dNative(width, width, kernel, padding=(kernel - 1) // 2)
        self.conv2 = Conv1dNative(width, width, kernel, padding=1)
        self.ln1, self.ln2 = (
            LayerNorm(width, promote_fp32=False),
            LayerNorm(width, promote_fp32=False),
        )
        self.proj = Linear(width, 1)
        self.activation = ReLU()

    def forward(self, hidden):
        hidden = self.ln1(
            self.activation(self.conv1(hidden.transpose(1, 2))).transpose(1, 2)
        )
        hidden = self.ln2(
            self.activation(self.conv2(hidden.transpose(1, 2))).transpose(1, 2)
        )
        return self.proj(hidden).squeeze(-1)


class HifiGan(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.upsample_initial_channel
        self.conv_pre = Conv1dNative(
            config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim,
            width,
            7,
            padding=3,
        )
        self.upsampler = nn.ModuleList(
            ConvTranspose1d(
                width // 2**i,
                width // 2 ** (i + 1),
                kernel,
                stride=rate,
                padding=(kernel - rate) // 2,
            )
            for i, (kernel, rate) in enumerate(
                zip(config.upsample_kernel_sizes, config.upsample_rates)
            )
        )
        self.resblocks = nn.ModuleList(
            HifiGanResidualBlock(
                width // 2 ** (i + 1), kernel, dilation, config.leaky_relu_slope
            )
            for i in range(len(self.upsampler))
            for kernel, dilation in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            )
        )
        self.conv_post = Conv1dNative(
            width // 2 ** len(self.upsampler), 1, 7, padding=3
        )
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.activation, self.final_activation, self.tanh = (
            LeakyReLU(config.leaky_relu_slope),
            LeakyReLU(0.01),
            Tanh(),
        )

    def forward(self, hidden):
        hidden = self.conv_pre(hidden)
        for i, upsampler in enumerate(self.upsampler):
            hidden = upsampler(self.activation(hidden))
            output = self.resblocks[i * self.num_kernels](hidden)
            for j in range(1, self.num_kernels):
                output = output + self.resblocks[i * self.num_kernels + j](hidden)
            hidden = output / self.num_kernels
        return self.tanh(self.conv_post(self.final_activation(hidden))).squeeze(1)


class CodeHifiGan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dur_predictor = VariancePredictor(config)
        self.unit_embedding = Embedding(
            config.unit_hifi_gan_vocab_size, config.unit_embed_dim
        )
        self.speaker_embedding = Embedding(
            config.vocoder_num_spkrs, config.spkr_embed_dim
        )
        self.language_embedding = Embedding(
            config.vocoder_num_langs, config.lang_embed_dim
        )
        self.hifi_gan = HifiGan(config)
        self.expm1, self.nearest = Expm1(), NearestNonnegative()
        self.offset = Offset2Batch()
        self.math = DurationArithmetic()

    def forward(self, ids, speaker, language):
        hidden = self.unit_embedding(ids).transpose(1, 2)
        counts = self.nearest(self.expm1(self.dur_predictor(hidden.transpose(1, 2))))
        choices = torch.stack((counts, torch.ones_like(counts)), dim=-1)
        counts = choices.gather(
            -1, self.nearest.predicate(choices.float())[..., None]
        ).squeeze(-1)
        # Native batch1 path expands all integer counts, including padded units.
        rows = []
        for tokens, row_counts in zip(hidden, counts):
            indices = self.offset(self.math.prefix(row_counts.float()).long())
            rows.append(tokens.index_select(-1, indices).transpose(0, 1))
        length = max(row.shape[0] for row in rows)
        expanded = hidden.new_zeros((len(rows), hidden.shape[1], length))
        for i, row in enumerate(rows):
            expanded[i, :, : row.shape[0]] = row.transpose(0, 1)
        spkr = self.speaker_embedding(speaker).transpose(1, 2).expand(-1, -1, length)
        lang = self.language_embedding(language).transpose(1, 2).expand(-1, -1, length)
        waveform = self.hifi_gan(torch.cat((lang, expanded, spkr), dim=1))
        # Preserve the native public length calculation, including its index convention.
        valid = (
            (ids != self.config.t2u_pad_token_id).sum(1).clamp(0, counts.shape[1] - 1)
        )
        cumulative = self.math.prefix(counts.float()).long()
        lengths = cumulative.gather(1, valid[:, None]).squeeze()
        for rate, kernel in zip(
            self.config.upsample_rates, self.config.upsample_kernel_sizes
        ):
            lengths = (lengths - 1) * rate - 2 * ((kernel - rate) // 2) + kernel
        return waveform, lengths


def conformer_config(config):
    return SimpleNamespace(
        hidden_size=config.hidden_size,
        intermediate_size=config.speech_encoder_intermediate_size,
        num_attention_heads=config.speech_encoder_attention_heads,
        conv_depthwise_kernel_size=config.conv_depthwise_kernel_size,
    )


class MaskedConvolution(Convolution):
    """Seamless zeros padded frames after normalization and before convolution."""

    def forward(self, hidden, mask=None):
        hidden = self.layer_norm(hidden)
        if mask is not None:
            hidden = hidden.masked_fill(~mask.bool().unsqueeze(-1), 0)
        hidden = self.glu(self.pointwise_conv1(hidden.transpose(1, 2)))
        hidden = self.activation(self.batch_norm(self.depthwise_conv(hidden)))
        return self.pointwise_conv2(hidden).transpose(1, 2)


class PlainAttention(nn.Module):
    """The temporal adapter has ordinary attention without relative positions."""

    def __init__(self, config):
        super().__init__()
        self.heads = config.speech_encoder_attention_heads
        self.width = config.hidden_size // self.heads
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, mask=None):
        batch, length, channels = hidden.shape
        q, k, v = (
            getattr(self, name)(hidden)
            .reshape(batch, length, self.heads, self.width)
            .transpose(1, 2)
            .reshape(batch * self.heads, length, self.width)
            for name in ("linear_q", "linear_k", "linear_v")
        )
        scores = self.bmm(q, k.transpose(1, 2)) / math.sqrt(self.width)
        if mask is not None:
            scores = scores.reshape(batch, self.heads, length, length)
            scores = scores.masked_fill(
                ~mask[:, None, None, :], torch.finfo(scores.dtype).min
            )
            scores = scores.reshape(batch * self.heads, length, length)
        result = self.bmm(self.softmax(scores), v).reshape(
            batch, self.heads, length, self.width
        )
        return self.linear_out(result.transpose(1, 2).reshape(batch, length, channels))


class AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.kernel_size, self.stride = (
            config.adaptor_kernel_size,
            config.adaptor_stride,
        )
        self.residual_layer_norm = LayerNorm(width, promote_fp32=False)
        self.residual_conv = Conv1dNative(
            width,
            2 * width,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn_layer_norm = LayerNorm(width, promote_fp32=False)
        self.self_attn_conv = Conv1dNative(
            width,
            2 * width,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.activation = GLU(precise_sigmoid=True)
        self.self_attn = PlainAttention(config)
        self.ffn_layer_norm = LayerNorm(width, promote_fp32=False)
        self.ffn = FeedForward(conformer_config(config))
        self.ffn.intermediate_act_fn = ReLU()

    def output_mask(self, attention_mask, length):
        # Supplied frame-mask arithmetic, not an activation-derived decision.
        sizes = attention_mask.int().sum(1)
        sizes = (
            sizes + 2 * (self.kernel_size // 2) - self.kernel_size
        ) // self.stride + 1
        return torch.arange(length, device=attention_mask.device)[None] < sizes[:, None]

    def forward(self, hidden, attention_mask=None):
        residual = self.activation(
            self.residual_conv(self.residual_layer_norm(hidden).transpose(1, 2))
        ).transpose(1, 2)
        hidden = self.activation(
            self.self_attn_conv(self.self_attn_layer_norm(hidden).transpose(1, 2))
        ).transpose(1, 2)
        mask = (
            None
            if attention_mask is None
            else self.output_mask(attention_mask, hidden.shape[1])
        )
        hidden = self.self_attn(hidden, mask) + residual
        return self.ffn(self.ffn_layer_norm(hidden)) + hidden


class SpeechEncoder(nn.Module):
    """Returns the full speech encoder's last hidden state [B,downsampled T,C]."""

    def __init__(self, config):
        super().__init__()
        if (
            config.position_embeddings_type != "relative"
            or config.speech_encoder_hidden_act not in ("swish", "silu")
        ):
            raise ValueError(
                "Selected SeamlessM4T medium speech path uses relative attention and swish"
            )
        if config.add_adapter and config.num_adapter_layers != 1:
            raise ValueError("Selected checkpoint has one temporal adapter")
        self.config = config
        self.feature_projection = nn.Module()
        self.feature_projection.layer_norm = LayerNorm(
            config.feature_projection_input_dim,
            eps=config.layer_norm_eps,
            promote_fp32=False,
        )
        self.feature_projection.projection = Linear(
            config.feature_projection_input_dim, config.hidden_size
        )
        child_config = conformer_config(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            ConformerLayer(
                child_config, attention=RelativeAttention, convolution=MaskedConvolution
            )
            for _ in range(config.speech_encoder_layers)
        )
        self.encoder.layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )
        self.intermediate_ffn = FeedForward(child_config)
        self.intermediate_ffn.intermediate_act_fn = ReLU()
        self.adapter = nn.Module() if config.add_adapter else None
        if self.adapter is not None:
            self.adapter.layers = nn.ModuleList([AdapterLayer(config)])
        self.inner_layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)
        # Native HF creates this fixed table on CPU before any model dtype conversion.
        with torch.device("cpu"):
            positions = torch.arange(
                config.max_source_positions, dtype=torch.int64
            ).float()[:, None]
            rates = torch.exp(
                torch.arange(0, config.hidden_size, 2, dtype=torch.int64).float()
                * -(math.log(10000.0) / config.hidden_size)
            )
            positive = torch.stack(
                ((positions * rates).sin(), (positions * rates).cos()), -1
            ).flatten(1)
            negative = torch.stack(
                ((-positions * rates).sin(), (-positions * rates).cos()), -1
            ).flatten(1)
        self.register_buffer(
            "position_table",
            torch.cat((positive.flip(0), negative[1:]))[None],
            persistent=False,
        )

    def forward(self, input_features, attention_mask=None):
        hidden = self.feature_projection.projection(
            self.feature_projection.layer_norm(input_features)
        )
        mask = None if attention_mask is None else attention_mask.bool()
        if mask is not None:
            hidden = hidden.masked_fill(~mask[:, :, None], 0)
        length, middle = hidden.shape[1], self.position_table.shape[1] // 2
        if length > self.config.max_source_positions:
            raise ValueError("Input exceeds the declared positional-table extent")
        positions = self.position_table[:, middle - length + 1 : middle + length].to(
            hidden.dtype
        )
        for layer in self.encoder.layers:
            hidden = layer(hidden, positions, mask)
        hidden = self.encoder.layer_norm(hidden)
        hidden = hidden + 0.5 * self.intermediate_ffn(hidden)
        if self.adapter is not None:
            hidden = self.adapter.layers[0](hidden, attention_mask)
        return self.inner_layer_norm(hidden)

    def output_mask(self, attention_mask, output_length):
        if self.adapter is None:
            return attention_mask.bool()
        return self.adapter.layers[0].output_mask(attention_mask, output_length)


class SeamlessM4T(nn.Module):
    def __init__(self, config, generation_config):
        super().__init__()
        self.config, self.generation = config, generation_config
        self.text_encoder = TextStack(config)
        self.text_decoder = TextStack(config, decoder=True)
        self.text_decoder.embed_tokens.emb.weight = (
            self.text_encoder.embed_tokens.emb.weight
        )
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.text_encoder.embed_tokens.emb.weight
        self.speech_encoder = SpeechEncoder(config)
        self.t2u_model = UnitModel(config)
        self.vocoder = CodeHifiGan(config)
        self.top1 = CodecTop1()

    def greedy(self, decoder, head, memory, mask, prefix, steps, eos, pad):
        sequences = prefix
        active = torch.ones(prefix.shape[0], device=prefix.device, dtype=torch.bool)
        cache = None
        for _ in range(steps):
            hidden, cache = decoder(
                sequences if cache is None else sequences[:, -1:],
                memory=memory,
                memory_mask=mask,
                cache=cache,
            )
            chosen = self.top1(head(hidden[:, -1]).float())
            chosen = torch.where(active, chosen, pad)
            sequences = torch.cat((sequences, chosen[:, None]), dim=1)
            active = active & chosen.ne(eos)
            if not any(active.tolist()):
                break
        return sequences

    def generate(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        *,
        tgt_lang="rus",
        spkr_id=0,
        text_max_new_tokens=4,
        speech_max_new_tokens=4,
        return_intermediate_token_ids=False,
    ):
        if input_features is not None:
            memory = self.speech_encoder(input_features, attention_mask)
            memory_mask = (
                None
                if attention_mask is None
                else self.speech_encoder.output_mask(attention_mask, memory.shape[1])
            )
        else:
            memory, _ = self.text_encoder(input_ids, mask=attention_mask)
            memory_mask = attention_mask
        batch = memory.shape[0]
        prefix = torch.tensor(
            [
                [
                    self.generation["decoder_start_token_id"],
                    self.generation["text_decoder_lang_to_code_id"][tgt_lang],
                ]
            ],
            device=memory.device,
        ).expand(batch, -1)
        sequences = self.greedy(
            self.text_decoder,
            self.lm_head,
            memory,
            memory_mask,
            prefix,
            text_max_new_tokens,
            self.config.eos_token_id,
            self.config.pad_token_id,
        )
        # Native audio-to-speech reruns its encoder before the unit stage.
        if input_features is not None:
            memory = self.speech_encoder(input_features, attention_mask)
        hidden, _ = self.text_decoder(sequences, memory=memory, memory_mask=memory_mask)
        lengths = sequences.ne(self.config.pad_token_id).int().sum(1)
        unit_mask = (
            torch.arange(sequences.shape[1], device=memory.device)[None]
            < lengths[:, None]
        )
        unit_memory, _ = self.t2u_model.model.encoder(hidden=hidden, mask=unit_mask)
        prefix = torch.tensor(
            [
                [
                    self.config.t2u_eos_token_id,
                    self.generation["t2u_lang_code_to_id"][tgt_lang],
                ]
            ],
            device=memory.device,
        ).expand(batch, -1)
        units = self.greedy(
            self.t2u_model.model.decoder,
            self.t2u_model.lm_head,
            unit_memory,
            unit_mask,
            prefix,
            speech_max_new_tokens,
            self.config.t2u_eos_token_id,
            self.config.t2u_pad_token_id,
        )
        ids = units[:, 2:].clone()
        ids.masked_fill_(
            ids == self.config.t2u_eos_token_id, self.config.t2u_pad_token_id
        )
        ids = torch.where(
            ids == self.config.t2u_pad_token_id, ids, ids - self.config.vocoder_offset
        )
        speaker = torch.full(
            (batch, 1), spkr_id, device=memory.device, dtype=torch.long
        )
        language = torch.full(
            (batch, 1),
            self.generation["vocoder_lang_code_to_id"][tgt_lang],
            device=memory.device,
            dtype=torch.long,
        )
        waveform, lengths = self.vocoder(ids, speaker, language)
        if return_intermediate_token_ids:
            return dict(
                waveform=waveform,
                waveform_lengths=lengths,
                sequences=sequences,
                unit_sequences=units,
            )
        return waveform, lengths


def build_from_config(config, device, dtype, *, generation_config):
    if (
        config.activation_function != "relu"
        or not config.tie_word_embeddings
        or not config.use_cache
    ):
        raise ValueError(
            "Selected SeamlessM4T checkpoint requires tied cached ReLU transformers"
        )
    return SeamlessM4T(config, generation_config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {
        key: state_dict[key.replace(".emb.weight", ".weight")]
        for key in model.state_dict()
    }
    used = {key.replace(".emb.weight", ".weight") for key in model.state_dict()}
    unused = set(state_dict) - used
    # HF stores a separate name for the same shared text embedding.
    if unused != {"shared.weight"}:
        raise ValueError(f"Unmapped SeamlessM4T source weights: {sorted(unused)}")
    for name in (
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
        "lm_head.weight",
    ):
        if not torch.equal(state_dict["shared.weight"], state_dict[name]):
            raise ValueError(f"Expected tied text weight: {name}")
    if not torch.equal(
        state_dict["t2u_model.lm_head.weight"],
        state_dict["t2u_model.model.decoder.embed_tokens.weight"],
    ):
        raise ValueError("Expected tied text-to-unit weights")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case):
    def run():
        waveform, lengths = model.generate(**inputs, **case["generation_kwargs"])
        return {"waveform": waveform, "waveform_lengths": lengths}

    return {"generate": Workload(run=run)}
