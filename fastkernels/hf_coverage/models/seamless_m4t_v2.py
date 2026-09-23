"""SeamlessM4T v2 text/audio-to-speech with character-conditioned unit synthesis.

Reuse v1 text transformers, greedy cache handling and HiFiGAN. V2 adds chunked
relative-key speech attention, causal convolution, fixed token-to-character
mapping and a learned-duration nonautoregressive unit decoder.
"""

import math
from copy import copy

import torch
from fastkernels.hf_coverage.models.fastspeech2_conformer import NearestNonnegative
from fastkernels.hf_coverage.models.seamless_m4t import (
    AdapterLayer,
    CodecTop1,
    CodeHifiGan,
    SeamlessM4T,
    TextAttention,
    TextStack,
    conformer_config,
    sinusoidal_table,
)
from fastkernels.hf_coverage.models.seamless_m4t import (
    load_state_dict_into as load_state_dict_into,
)
from fastkernels.hf_coverage.models.seamless_m4t import make_workloads as make_workloads
from fastkernels.hf_coverage.models.vits import DurationArithmetic
from fastkernels.hf_coverage.patches.seamless_expm1 import Expm1
from torch import nn

from fastkernels.hf_coverage.models.wav2vec2_conformer import (
    GLU,
    ConformerLayer,
    FeedForward,
)
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.pointtransformerv3_offsets import Offset2Batch
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tensor_ops import Pad


class VariancePredictor(nn.Module):
    def __init__(self, width, inner, kernel):
        super().__init__()
        self.conv1 = Conv1dNative(width, inner, kernel, padding=(kernel - 1) // 2)
        self.conv2 = Conv1dNative(inner, inner, kernel, padding=(kernel - 1) // 2)
        self.ln1, self.ln2 = (
            LayerNorm(inner, promote_fp32=False),
            LayerNorm(inner, promote_fp32=False),
        )
        self.proj, self.activation = Linear(inner, 1), ReLU()

    def forward(self, hidden, mask=None):
        if mask is not None:
            hidden = hidden.masked_fill(~mask.bool()[..., None], 0)
        hidden = self.ln1(
            self.activation(self.conv1(hidden.transpose(1, 2))).transpose(1, 2)
        )
        if mask is not None:
            hidden = hidden.masked_fill(~mask.bool()[..., None], 0)
        hidden = self.ln2(
            self.activation(self.conv2(hidden.transpose(1, 2))).transpose(1, 2)
        )
        return self.proj(hidden).squeeze(-1)


class DurationExpansion(nn.Module):
    def __init__(self):
        super().__init__()
        self.math = DurationArithmetic()
        self.offset = Offset2Batch()
        self.expm1, self.nearest = Expm1(), NearestNonnegative()

    def counts(self, logits):
        count = self.nearest(self.expm1(logits))
        choices = torch.stack((count, torch.ones_like(count)), dim=-1)
        return choices.gather(
            -1, self.nearest.predicate(choices.float())[..., None]
        ).squeeze(-1)

    def forward(self, hidden, counts):
        rows = [
            row.index_select(0, self.offset(self.math.prefix(count.float()).long()))
            for row, count in zip(hidden, counts)
        ]
        length = max(row.shape[0] for row in rows)
        result = hidden.new_zeros((len(rows), length, hidden.shape[-1]))
        for index, row in enumerate(rows):
            result[index, : row.shape[0]] = row
        return result


class UnitLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.self_attn = TextAttention(width, config.decoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(width, promote_fp32=False)
        self.conv1, self.conv2 = (
            Conv1dNative(width, width, 7, padding=3),
            Conv1dNative(width, width, 7, padding=3),
        )
        self.conv_layer_norm = LayerNorm(width, promote_fp32=False)
        self.activation = ReLU()

    def forward(self, hidden, mask):
        branch, _ = self.self_attn(hidden, mask=mask)
        hidden = self.self_attn_layer_norm(hidden + branch)
        residual = hidden
        hidden = hidden.masked_fill(~mask.bool()[..., None], 0)
        hidden = self.conv1(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.masked_fill(~mask.bool()[..., None], 0)
        hidden = self.conv2(self.activation(hidden).transpose(1, 2)).transpose(1, 2)
        return self.conv_layer_norm(residual + hidden)


class UnitDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.pad, self.scale = (
            config.pad_token_id,
            math.sqrt(width) if config.scale_embedding else 1.0,
        )
        self.embed_tokens = Embedding(config.vocab_size, width, padding_idx=self.pad)
        self.embed_char = Embedding(config.char_vocab_size, width)
        self.pos_emb_alpha_char, self.pos_emb_alpha = (
            nn.Parameter(torch.ones(1)),
            nn.Parameter(torch.ones(1)),
        )
        self.register_buffer(
            "positions",
            sinusoidal_table(config.max_position_embeddings + 2, width, self.pad),
            persistent=False,
        )
        self.duration_predictor = VariancePredictor(
            config.variance_predictor_embed_dim,
            config.variance_predictor_hidden_dim,
            config.variance_predictor_kernel_size,
        )
        self.layers = nn.ModuleList(
            UnitLayer(config) for _ in range(config.decoder_layers)
        )
        self.layer_norm = LayerNorm(width, promote_fp32=False)
        self.expand = DurationExpansion()

    def forward(self, char_ids, char_counts, memory):
        # Character counts derive from fixed token strings, not activation reductions.
        char_lengths = char_counts.sum(1)
        mask = (
            torch.arange(char_ids.shape[1], device=char_ids.device)[None]
            < char_lengths[:, None]
        )
        hidden = self.expand(memory, char_counts)
        positions = self.positions[self.pad + 1 : self.pad + 1 + hidden.shape[1]][None]
        hidden = (
            self.embed_char(char_ids) * self.scale
            + self.expand.math.mul(self.pos_emb_alpha_char, positions)
            + hidden
        )
        counts = self.expand.counts(self.duration_predictor(hidden, mask)).masked_fill(
            ~mask, 0
        )
        hidden = self.expand(hidden, counts)
        positions = self.positions[self.pad + 1 : self.pad + 1 + hidden.shape[1]][None]
        hidden = hidden + self.expand.math.mul(self.pos_emb_alpha, positions)
        # Duration-derived totals use an admitted reduction, not raw sum.
        lengths = self.expand.math.sum_last(counts.float()).long()
        # This mask depends on predicted duration, so compare with an admitted op.
        mask = self.expand.math.less(
            torch.arange(hidden.shape[1], device=hidden.device).float()[None],
            lengths.float()[:, None],
        )
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.layer_norm(hidden), mask


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
        self.model.decoder = UnitDecoder(unit)
        self.lm_head = Linear(config.hidden_size, unit.vocab_size, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.emb.weight

    def forward(self, hidden, mask, char_ids, char_counts):
        hidden, _ = self.model.encoder(hidden=hidden, mask=mask)
        hidden, mask = self.model.decoder(char_ids, char_counts, hidden)
        return self.lm_head(hidden), mask


class Vocoder(CodeHifiGan):
    def __init__(self, config):
        super().__init__(config)
        self.dur_predictor = VariancePredictor(
            config.unit_embed_dim,
            config.unit_embed_dim,
            config.variance_predictor_kernel_size,
        )


def character_inputs(ids, generation, pad=0, unknown=1):
    """Native default punctuation/space grouping, using fixed token metadata."""
    rows = ids.tolist()
    all_counts = []
    all_chars = []
    for row in rows:
        length = sum(token != pad for token in row)
        tokens = row[:length]
        words = [str(generation["id_to_text"].get(str(token))) for token in tokens]
        next_space = [
            i + 1 < len(words) and len(words[i + 1]) > 1 and words[i + 1][0] == "▁"
            for i in range(len(words))
        ]
        punctuation = [
            len(word) == 1
            and not word.isalpha()
            and not word.isnumeric()
            and word != "▁"
            for word in words
        ]
        counts = [0] * len(row)
        characters = []
        for i, (token, word) in enumerate(zip(tokens, words)):
            if token == pad:
                break
            count = 1 if token == unknown else len(word)
            if token != unknown:
                if punctuation[i] and next_space[i]:
                    count += 1
                elif i > 0 and punctuation[i - 1] and next_space[i - 1]:
                    count -= 1
            counts[i] = count
        for token, word in zip(tokens, words):
            characters.extend(
                [unknown]
                if token == unknown
                else [generation["char_to_id"].get(char, unknown) for char in word]
            )
        all_counts.append([0, *counts, 0])
        all_chars.append(characters)
    counts = ids.new_tensor(all_counts)
    width = max(sum(row) for row in all_counts)
    characters = ids.new_full((len(rows), width), pad)
    for index, row in enumerate(all_chars):
        characters[index, : len(row)] = ids.new_tensor(row)
    return characters, counts


class RelativeKeyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        self.left = config.left_max_position_embeddings
        self.right = config.right_max_position_embeddings
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.distance_embedding = Embedding(self.left + self.right + 1, self.width)
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, attention_mask=None):
        batch, length, channels = hidden.shape
        q, k, v = (
            getattr(self, name)(hidden)
            .reshape(batch, length, self.heads, self.width)
            .transpose(1, 2)
            .reshape(batch * self.heads, length, self.width)
            for name in ("linear_q", "linear_k", "linear_v")
        )
        scores = self.bmm(q, k.transpose(1, 2)) / math.sqrt(self.width)
        positions = torch.arange(length, device=hidden.device)
        distance = (positions[None] - positions[:, None]).clamp(-self.left, self.right)
        relative = self.distance_embedding(distance + self.left).to(q.dtype)
        # HF contracts each query with the positional vectors for its key row.
        relative_scores = self.bmm(
            q.transpose(0, 1), relative.transpose(1, 2)
        ).transpose(0, 1)
        scores = scores + relative_scores / math.sqrt(self.width)
        if attention_mask is not None:
            scores = (
                scores.reshape(batch, self.heads, length, length) + attention_mask
            ).reshape(batch * self.heads, length, length)
        hidden = self.bmm(self.softmax(scores), v).reshape(
            batch, self.heads, length, self.width
        )
        return self.linear_out(hidden.transpose(1, 2).reshape(batch, length, channels))


class CausalConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config.hidden_size, config.conv_depthwise_kernel_size
        self.left_padding = kernel - 1
        self.layer_norm = LayerNorm(width, promote_fp32=False)
        self.pointwise_conv1 = Conv1dNative(width, 2 * width, 1, bias=False)
        self.glu = GLU(precise_sigmoid=True)
        self.depthwise_conv = Conv1dNative(
            width, width, kernel, groups=width, bias=False
        )
        self.depthwise_layer_norm = LayerNorm(width, promote_fp32=False)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1dNative(width, width, 1, bias=False)
        self.pad = Pad()

    def forward(self, hidden, mask=None):
        hidden = self.layer_norm(hidden)
        if mask is not None:
            hidden = hidden.masked_fill(~mask.bool().unsqueeze(-1), 0)
        hidden = self.glu(self.pointwise_conv1(hidden.transpose(1, 2)))
        hidden = self.depthwise_conv(self.pad(hidden, (self.left_padding, 0)))
        hidden = self.depthwise_layer_norm(hidden.transpose(1, 2)).transpose(1, 2)
        return self.pointwise_conv2(self.activation(hidden)).transpose(1, 2)


class SpeechLayer(ConformerLayer):
    def __init__(self, config):
        super().__init__(
            config, attention=RelativeKeyAttention, convolution=CausalConvolution
        )

    def forward(self, hidden, attention_mask, conv_mask):
        hidden = self.ffn1(self.ffn1_layer_norm(hidden)) * 0.5 + hidden
        hidden = (
            self.self_attn(self.self_attn_layer_norm(hidden), attention_mask) + hidden
        )
        hidden = hidden + self.conv_module(hidden, conv_mask)
        hidden = self.ffn2(self.ffn2_layer_norm(hidden)) * 0.5 + hidden
        return self.final_layer_norm(hidden)


class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (
            config.position_embeddings_type != "relative_key"
            or config.speech_encoder_hidden_act not in ("swish", "silu")
        ):
            raise ValueError(
                "Selected v2 speech encoder uses relative-key attention and swish"
            )
        if config.add_adapter and config.num_adapter_layers != 1:
            raise ValueError("Selected checkpoint has one temporal adapter")
        self.config = config
        child = conformer_config(config)
        child.left_max_position_embeddings = config.left_max_position_embeddings
        child.right_max_position_embeddings = config.right_max_position_embeddings
        self.feature_projection = nn.Module()
        self.feature_projection.layer_norm = LayerNorm(
            config.feature_projection_input_dim,
            eps=config.layer_norm_eps,
            promote_fp32=False,
        )
        self.feature_projection.projection = Linear(
            config.feature_projection_input_dim, config.hidden_size
        )
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            SpeechLayer(child) for _ in range(config.speech_encoder_layers)
        )
        self.encoder.layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )
        self.intermediate_ffn = FeedForward(child)
        self.intermediate_ffn.intermediate_act_fn = ReLU()
        self.adapter = nn.Module() if config.add_adapter else None
        if self.adapter is not None:
            self.adapter.layers = nn.ModuleList([AdapterLayer(config)])
        self.inner_layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)

    def attention_bias(self, hidden, mask):
        length = hidden.shape[1]
        blocked = None if mask is None else ~mask[:, None, None, :].bool()
        chunk_size = self.config.speech_encoder_chunk_size
        if chunk_size is not None:
            positions = torch.arange(length, device=hidden.device)
            chunk = positions // chunk_size
            end = ((chunk + 1) * chunk_size).clamp(max=length)
            if self.config.speech_encoder_left_chunk_num >= 0:
                start = (chunk - self.config.speech_encoder_left_chunk_num).clamp(
                    min=0
                ) * chunk_size
            else:
                start = torch.zeros_like(chunk)
            chunks = (positions[None] < start[:, None]) | (
                positions[None] >= end[:, None]
            )
            blocked = (
                chunks[None, None]
                if blocked is None
                else (blocked | chunks[None, None])
            )
        if blocked is None:
            return None
        return hidden.new_zeros(blocked.shape).masked_fill(
            blocked, torch.finfo(hidden.dtype).min
        )

    def forward(self, input_features, attention_mask=None):
        # Native v2 explicitly converts input features to the learned norm dtype.
        input_features = input_features.to(
            self.feature_projection.layer_norm.weight.dtype
        )
        hidden = self.feature_projection.projection(
            self.feature_projection.layer_norm(input_features)
        )
        if attention_mask is not None:
            hidden = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        bias = self.attention_bias(hidden, attention_mask)
        for layer in self.encoder.layers:
            hidden = layer(hidden, bias, attention_mask)
        hidden = self.encoder.layer_norm(hidden)
        hidden = hidden + 0.5 * self.intermediate_ffn(hidden)
        if self.adapter is not None:
            hidden = self.adapter.layers[0](hidden, attention_mask)
        return self.inner_layer_norm(hidden)

    def output_mask(self, attention_mask, output_length):
        if self.adapter is None:
            return attention_mask.bool()
        return self.adapter.layers[0].output_mask(attention_mask, output_length)


class SeamlessM4Tv2(SeamlessM4T):
    def __init__(self, config, generation_config):
        nn.Module.__init__(self)
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
        self.vocoder = Vocoder(config)
        self.top1 = CodecTop1()

    def generate(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        *,
        tgt_lang="rus",
        speaker_id=0,
        text_max_new_tokens=4,
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
        if input_features is not None:
            memory = self.speech_encoder(input_features, attention_mask)
        # Native v2 leaves the final prediction out of character synthesis.
        hidden, _ = self.text_decoder(
            sequences[:, :-1], memory=memory, memory_mask=memory_mask
        )
        lengths = sequences[:, :-1].ne(self.config.pad_token_id).int().sum(1)
        mask = (
            torch.arange(hidden.shape[1], device=memory.device)[None] < lengths[:, None]
        )
        ids = sequences[:, 2:-1].masked_fill(
            sequences[:, 2:-1] == self.config.eos_token_id, self.config.pad_token_id
        )
        chars, counts = character_inputs(ids, self.generation, self.config.pad_token_id)
        logits, padding = self.t2u_model(hidden, mask, chars, counts)
        units = self.top1(logits.float())
        ids = units.masked_fill(
            (units == self.config.t2u_eos_token_id) | ~padding,
            self.config.t2u_pad_token_id,
        )
        ids = torch.where(
            ids == self.config.t2u_pad_token_id, ids, ids - self.config.vocoder_offset
        )
        speakers = torch.full(
            (batch, 1), speaker_id, device=memory.device, dtype=torch.long
        )
        language = torch.full(
            (batch, 1),
            self.generation["vocoder_lang_code_to_id"][tgt_lang],
            device=memory.device,
            dtype=torch.long,
        )
        waveform, lengths = self.vocoder(ids, speakers, language)
        if return_intermediate_token_ids:
            return dict(
                waveform=waveform,
                waveform_lengths=lengths,
                sequences=sequences,
                unit_sequences=units,
            )
        return waveform, lengths


def build_from_config(config, device, dtype, *, generation_config_source):
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    if (
        config.activation_function != "relu"
        or not config.tie_word_embeddings
        or not config.use_cache
    ):
        raise ValueError(
            "Selected SeamlessM4T v2 checkpoint requires tied cached ReLU transformers"
        )
    # Fixed vocabulary/character metadata is loaded before measured execution.
    path = hf_hub_download(
        generation_config_source["repo"],
        "generation_config.json",
        revision=generation_config_source["revision"],
    )
    generation = json.loads(Path(path).read_text())
    return SeamlessM4Tv2(config, generation).to(device=device, dtype=dtype).eval()
