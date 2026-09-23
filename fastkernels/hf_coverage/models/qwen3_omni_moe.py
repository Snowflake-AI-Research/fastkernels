"""Qwen3 Omni's multimodal thinker, sampled speech codes, and waveform decoder."""

import math
from dataclasses import fields
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.mrope import MRotaryEmbedding
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.qwen3_vl import Qwen3VLVisionConfig, Qwen3VisionTransformer

from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from ..patches.qwen_omni_snake import SnakeBeta
from ..runner import Workload, config_values
from .mimi import CausalConv, LayerScale
from .qwen_omni_sampling import OmniSampling
from .qwen2_5_omni import positions_for_inputs as _omni_positions


def positions_for_inputs(ids, inputs, config):
    # Qwen2.5 serializes *_index aliases; Qwen3 retains *_id fields.
    values = config.to_dict() if hasattr(config, "to_dict") else vars(config).copy()
    values.update(image_token_index=config.image_token_id, video_token_index=config.video_token_id)
    if not hasattr(config, "spatial_merge_size"):
        values["spatial_merge_size"] = config.vision_config.spatial_merge_size
    return _omni_positions(ids, inputs, SimpleNamespace(**values))


class SwiGLU(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate_proj = Linear(width, intermediate, bias=False)
        self.up_proj = Linear(width, intermediate, bias=False)
        self.down_proj = Linear(intermediate, width, bias=False)
        self.act = SiluAndMul()

    def forward(self, hidden):
        return self.down_proj(self.act.forward_native(torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), -1)))


class Attention(nn.Module):
    """Existing projection, rotary and attention operations with a dynamic KV cache."""

    def __init__(self, config, *, qk_norm=True, rotary=True, bias=None):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.heads
        bias = config.attention_bias if bias is None else bias
        for name, heads in (("q", self.heads), ("k", self.kv_heads), ("v", self.kv_heads)):
            setattr(self, name + "_proj", Linear(config.hidden_size, heads * self.head_dim, bias=bias))
        self.o_proj = Linear(self.heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = RMSNormNative(self.head_dim, config.rms_norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNormNative(self.head_dim, config.rms_norm_eps) if qk_norm else nn.Identity()
        self.attention = DenseAttention(backend="sdpa")
        if rotary:
            rope = config.rope_parameters
            section = rope.get("mrope_section", [self.head_dim // 2, 0, 0])
            self.rotary_emb = MRotaryEmbedding(self.head_dim, config.max_position_embeddings,
                                               rope["rope_theta"], section,
                                               mrope_interleaved="mrope_section" in rope)
        else:
            self.rotary_emb = None
        self.window = getattr(config, "sliding_window", None)

    def forward(self, hidden, positions=None, cache=None, *, causal=True):
        batch, length, _ = hidden.shape
        query = self.q_norm(self.q_proj(hidden).view(batch, length, self.heads, self.head_dim))
        key = self.k_norm(self.k_proj(hidden).view(batch, length, self.kv_heads, self.head_dim))
        value = self.v_proj(hidden).view(batch, length, self.kv_heads, self.head_dim)
        previous = 0 if cache is None else cache[0].shape[1]
        if self.rotary_emb is not None:
            if positions is None:
                positions = torch.arange(previous, previous + length, device=hidden.device)[None].expand(3, -1)
            query, key = self.rotary_emb.forward_native_2d(
                positions, query.reshape(batch * length, self.heads, self.head_dim),
                key.reshape(batch * length, self.kv_heads, self.head_dim))
            query = query.reshape(batch, length, self.heads, self.head_dim)
            key = key.reshape(batch, length, self.kv_heads, self.head_dim)
        if cache is not None:
            key, value = torch.cat((cache[0], key), 1), torch.cat((cache[1], value), 1)
        new_cache = key, value
        if self.heads != self.kv_heads:
            key = key.repeat_interleave(self.heads // self.kv_heads, dim=2)
            value = value.repeat_interleave(self.heads // self.kv_heads, dim=2)
        mask = None
        if causal and ((previous and length > 1) or self.window):
            qpos = torch.arange(previous, previous + length, device=hidden.device)
            kpos = torch.arange(key.shape[1], device=hidden.device)
            mask = kpos[None] <= qpos[:, None]
            if self.window:
                mask &= kpos[None] > qpos[:, None] - self.window
        output = self.attention(query, key, value, causal=causal and mask is None and length > 1,
                                attn_mask=mask, softmax_scale=self.head_dim ** -0.5)
        return self.o_proj(output.reshape(batch, length, -1)), new_cache


class DecoderLayer(nn.Module):
    def __init__(self, config, *, moe=False, shared=False, scaled=False, qk_norm=True):
        super().__init__()
        self.self_attn = Attention(config, qk_norm=qk_norm)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        if moe:
            experts = getattr(config, "num_experts", getattr(config, "num_local_experts", None))
            self.mlp = SharedExpertMoE(hidden_size=config.hidden_size, num_experts=experts,
                top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob, keep_router_weights_fp32=False,
                shared_expert_intermediate_size=config.shared_expert_intermediate_size if shared else 0,
                shared_expert_gate=shared)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.self_attn_layer_scale = LayerScale(config.hidden_size) if scaled else nn.Identity()
        self.mlp_layer_scale = LayerScale(config.hidden_size) if scaled else nn.Identity()

    def forward(self, hidden, positions=None, cache=None):
        output, cache = self.self_attn(self.input_layernorm(hidden), positions, cache)
        hidden = hidden + self.self_attn_layer_scale(output)
        return hidden + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(hidden))), cache


class Decoder(nn.Module):
    def __init__(self, config, **options):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(config, **options) for _ in range(config.num_hidden_layers))
        self.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions=None, cache=None, deepstack=None, visual_mask=None):
        states, next_cache = [hidden], []
        for index, layer in enumerate(self.layers):
            hidden, layer_cache = layer(hidden, positions, None if cache is None else cache[index])
            next_cache.append(layer_cache)
            # HF's output recorder captures the decoder-layer output before
            # the outer model injects DeepStack features. Talker consumes it.
            states.append(hidden)
            if deepstack is not None and index < len(deepstack):
                hidden = hidden.clone()
                hidden[visual_mask] = hidden[visual_mask] + deepstack[index]
        hidden = self.norm(hidden)
        states[-1] = hidden
        return hidden, next_cache, tuple(states)


class TransConv(nn.Module):
    """Qwen's transposed-convolution crop differs from Mimi's right-only crop."""

    def __init__(self, source, target, kernel, stride):
        super().__init__()
        self.conv = ConvTranspose1d(source, target, kernel, stride=stride)
        self.crop = kernel - stride

    def forward(self, hidden):
        output = self.conv(hidden)
        return output[..., self.crop:output.shape[-1] - self.crop].contiguous()


class ConvNext(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dwconv = CausalConv(width, width, 7, groups=width)
        self.norm = LayerNorm(width, eps=1e-6, promote_fp32=False)
        self.pwconv1, self.pwconv2 = Linear(width, 4 * width), Linear(4 * width, width)
        self.act, self.product = GELU(), ProductGate()
        self.gamma = nn.Parameter(torch.empty(width))

    def forward(self, hidden):
        output = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(hidden).transpose(1, 2)))))
        output = self.product(torch.cat((output, self.gamma.expand_as(output)), -1))
        return hidden + output.transpose(1, 2)


class WaveResidual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.act1, self.act2 = SnakeBeta(width), SnakeBeta(width)
        self.conv1 = CausalConv(width, width, 7, dilation=dilation)
        self.conv2 = CausalConv(width, width, 1)

    def forward(self, hidden):
        return hidden + self.conv2(self.act2(self.conv1(self.act1(hidden))))


class WaveBlock(nn.Module):
    def __init__(self, width, rate):
        super().__init__()
        self.block = nn.Sequential(SnakeBeta(width), TransConv(width, width // 2, 2 * rate, rate),
                                  *(WaveResidual(width // 2, dilation) for dilation in (1, 3, 9)))

    def forward(self, hidden):
        return self.block(hidden)


class Code2Wav(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_transformer = Decoder(config, scaled=True, qk_norm=False)
        self.code_embedding = Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        self.register_buffer("code_offset", torch.arange(config.num_quantizers)[None, :, None] * config.codebook_size,
                             persistent=False)
        self.upsample = nn.ModuleList(nn.ModuleList((TransConv(config.hidden_size, config.hidden_size, rate, rate),
                                                    ConvNext(config.hidden_size))) for rate in config.upsampling_ratios)
        layers = [CausalConv(config.hidden_size, config.decoder_dim, 7)]
        for index, rate in enumerate(config.upsample_rates):
            layers.append(WaveBlock(config.decoder_dim // 2 ** index, rate))
        width = config.decoder_dim // 2 ** len(config.upsample_rates)
        self.decoder = nn.Sequential(*layers, SnakeBeta(width), CausalConv(width, 1, 7))
        self.reduce, self.maximum = SegmentCSR(), MaxPool2d((1, 2))
        self.total_upsample = math.prod(config.upsample_rates + config.upsampling_ratios)

    def forward(self, codes):
        embedded = self.code_embedding(codes + self.code_offset)
        # One ordinary reduction across the actual code groups, with linear storage.
        hidden = self.reduce(embedded.permute(1, 0, 2, 3).contiguous().float(),
                             torch.tensor([0, codes.shape[1]], device=codes.device), "mean")[0].to(embedded.dtype)
        hidden = self.pre_transformer(hidden)[0].transpose(1, 2)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = self.decoder(hidden)
        shape = wav.shape
        # MaxPool exposes the two bounds of clamp without a new pointwise primitive.
        rows = wav.reshape(-1, 1, 1, 1)
        rows = self.maximum(torch.cat((rows, torch.full_like(rows, -1)), -1))
        rows = -self.maximum(torch.cat((-rows, torch.full_like(rows, -1)), -1))
        return rows.reshape(shape)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        chunks = []
        for start in range(0, codes.shape[-1], chunk_size):
            context = min(start, left_context_size)
            wav = self(codes[..., start - context:start + chunk_size])
            chunks.append(wav[..., context * self.total_upsample:])
        return torch.cat(chunks, -1)


class AudioLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = SimpleNamespace(hidden_size=config.d_model, num_attention_heads=config.encoder_attention_heads,
                            num_key_value_heads=config.encoder_attention_heads, attention_bias=True)
        self.self_attn = Attention(c, rotary=False, qk_norm=False)
        self.self_attn.out_proj = self.self_attn.o_proj
        del self.self_attn.o_proj
        self.self_attn_layer_norm = LayerNorm(config.d_model, promote_fp32=False)
        self.final_layer_norm = LayerNorm(config.d_model, promote_fp32=False)
        self.fc1, self.fc2 = Linear(config.d_model, config.encoder_ffn_dim), Linear(config.encoder_ffn_dim, config.d_model)
        self.act = GELU()

    def forward(self, hidden):
        # Native eager/SDPA audio receives no mask despite supplied cu_seqlens.
        attention = self.self_attn
        x = self.self_attn_layer_norm(hidden)
        shape = (*x.shape[:2], attention.heads, attention.head_dim)
        q, k, v = [getattr(attention, name + "_proj")(x).view(shape) for name in ("q", "k", "v")]
        hidden = hidden + attention.out_proj(attention.attention(q, k, v).reshape_as(x))
        return hidden + self.fc2(self.act(self.fc1(self.final_layer_norm(hidden))))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        small, width = config.downsample_hidden_size, config.d_model
        self.conv2d1 = Conv2d(1, small, 3, stride=2, padding=1)
        self.conv2d2, self.conv2d3 = (Conv2d(small, small, 3, stride=2, padding=1) for _ in range(2))
        freq = (config.num_mel_bins + 7) // 8
        self.conv_out = Linear(small * freq, width, bias=False)
        self.layers = nn.ModuleList(AudioLayer(config) for _ in range(config.encoder_layers))
        self.ln_post = LayerNorm(width, promote_fp32=False)
        self.proj1, self.proj2 = Linear(width, width), Linear(width, config.output_dim)
        self.act, self.pad = GELU(), Pad()
        inverse = torch.exp(-math.log(10000) / (width // 2 - 1) * torch.arange(width // 2).float())
        angles = torch.arange(config.max_source_positions)[:, None] * inverse[None]
        self.register_buffer("positions", torch.cat((angles.sin(), angles.cos()), -1), persistent=False)

    def forward(self, features, lengths):
        chunk_lengths = []
        for length in lengths.tolist():
            chunk_lengths.extend([min(self.config.n_window * 2, length - start)
                                  for start in range(0, length, self.config.n_window * 2)])
        chunks = features.T.split(chunk_lengths)
        width = max(chunk_lengths)
        padded = torch.stack([self.pad(chunk.T, (0, width - chunk.shape[0])) for chunk in chunks])[:, None]
        outputs = []
        for chunk in padded.split(self.config.conv_chunksize):
            outputs.append(self.act(self.conv2d3(self.act(self.conv2d2(self.act(self.conv2d1(chunk)))))))
        hidden = torch.cat(outputs)
        b, c, f, t = hidden.shape
        hidden = self.conv_out(hidden.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        hidden = hidden + self.positions[:t].to(hidden.dtype)[None]
        hidden = torch.cat([value[:(length + 7) // 8] for value, length in zip(hidden, chunk_lengths)])[None]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.proj2(self.act(self.proj1(self.ln_post(hidden))))[0]


class Thinker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio_config)
        vision_config = Qwen3VLVisionConfig(**{field.name: getattr(config.vision_config, field.name)
                                             for field in fields(Qwen3VLVisionConfig)})
        self.visual = Qwen3VisionTransformer(vision_config)
        self.model = Decoder(config.text_config, moe=True)
        self.model.embed_tokens = Embedding(config.text_config.vocab_size, config.text_config.hidden_size)
        self.lm_head = Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def prefill_embeddings(self, inputs):
        ids = inputs["input_ids"]
        hidden = self.model.embed_tokens(ids)
        if "input_features" in inputs:
            lengths = inputs["feature_attention_mask"].sum(-1)
            features = inputs["input_features"].transpose(1, 2)[inputs["feature_attention_mask"].bool()].T
            hidden[ids == self.config.audio_token_id] = self.audio_tower(features, lengths)
        deepstack, visual_mask = None, torch.zeros_like(ids, dtype=torch.bool)
        for token, pixels, grid in ((self.config.image_token_id, "pixel_values", "image_grid_thw"),
                                    (self.config.video_token_id, "pixel_values_videos", "video_grid_thw")):
            if pixels not in inputs:
                continue
            features = self.visual(inputs[pixels], inputs[grid].cpu())
            mask = ids == token
            parts = features.split(hidden.shape[-1], -1)
            hidden[mask] = parts[0]
            if deepstack is None:
                deepstack = [torch.zeros_like(hidden) for _ in parts[1:]]
            for target, source in zip(deepstack, parts[1:]):
                target[mask] = source
            visual_mask |= mask
        if deepstack is not None:
            deepstack = [value[visual_mask] for value in deepstack]
        return hidden, deepstack, visual_mask


class ResizeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = Linear(config.thinker_hidden_size, config.text_config.intermediate_size)
        self.linear_fc2 = Linear(config.text_config.intermediate_size, config.text_config.hidden_size)
        self.act_fn = SiLU()

    def forward(self, hidden):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden)))


class CodePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Decoder(config)
        self.model.codec_embedding = nn.ModuleList(Embedding(config.vocab_size, config.hidden_size)
                                                   for _ in range(config.num_code_groups - 1))
        self.lm_head = nn.ModuleList(Linear(config.hidden_size, config.vocab_size, bias=False)
                                    for _ in range(config.num_code_groups - 1))
        self.sampler = OmniSampling(50, 0.8, 1.0, 1.0)

    def generate(self, hidden):
        codes, embeddings, cache = [], [], None
        for index, head in enumerate(self.lm_head):
            output, cache, _ = self.model(hidden, cache=cache)
            history = torch.cat(codes, -1) if codes else torch.empty((1, 0), device=hidden.device, dtype=torch.long)
            code = self.sampler(head(output[:, -1]), history)[:, None]
            codes.append(code)
            hidden = self.model.codec_embedding[index](code)
            embeddings.append(hidden)
        return torch.cat(codes, -1), torch.cat(embeddings, 1)


class Talker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Decoder(config.text_config, moe=True, shared=True)
        self.model.codec_embedding = Embedding(config.text_config.vocab_size, config.text_config.hidden_size)
        self.text_projection, self.hidden_projection = ResizeMLP(config), ResizeMLP(config)
        self.codec_head = Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.code_predictor = CodePredictor(config.code_predictor_config)
        self.sampler, self.reduce = OmniSampling(50, 1.0, 0.9, 1.05), SegmentCSR()

    def generate(self, hidden, position_ids, trailing_text, pad_embed, steps, delta):
        history = torch.empty((1, 0), device=hidden.device, dtype=torch.long)
        cache, codes = None, []
        initial_length = hidden.shape[1]
        for step in range(steps):
            output, cache, _ = self.model(hidden, position_ids, cache)
            logits = self.codec_head(output[:, -1]).float()
            first_special = self.config.text_config.vocab_size - 1024
            eos = self.config.codec_eos_token_id
            logits[:, first_special:eos] = -float("inf")
            logits[:, eos + 1:] = -float("inf")
            token = self.sampler(logits, history)[:, None]
            history = torch.cat((history, token), -1)
            if int(token[0, 0]) == eos or step + 1 == steps:
                break
            embedding = self.model.codec_embedding(token)
            residual_codes, residual_embeddings = self.code_predictor.generate(torch.cat((output[:, -1:], embedding), 1))
            codes.append(torch.cat((token, residual_codes), -1))
            codec_embeddings = torch.cat((embedding, residual_embeddings), 1)
            hidden = self.reduce(codec_embeddings.transpose(0, 1).contiguous().float(),
                                 torch.tensor([0, codec_embeddings.shape[1]], device=hidden.device), "sum")
            hidden = hidden.to(codec_embeddings.dtype).transpose(0, 1)
            hidden = hidden + (trailing_text[:, step:step + 1] if step < trailing_text.shape[1] else pad_embed)
            position_ids = torch.full((3, 1), initial_length + step + delta, dtype=torch.long, device=hidden.device)
        if not codes:
            raise RuntimeError("The native parent has no waveform codes if talker stops at its first token")
        return torch.stack(codes, dim=-1)


class Omni(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thinker, self.talker = Thinker(config.thinker_config), Talker(config.talker_config)
        self.code2wav, self.argmax = Code2Wav(config.code2wav_config), CodecTop1()

    def generate(self, inputs, thinker_steps, talker_steps, thinker_eos=151645):
        config, ids = self.config, inputs["input_ids"]
        if ids.shape[0] != 1:
            raise ValueError("Native Qwen3 Omni audio generation supports batch one")
        history = ids
        hidden, deepstack, visual_mask = self.thinker.prefill_embeddings(inputs)
        positions, delta = positions_for_inputs(ids, inputs, config.thinker_config)
        cache, embeddings, intermediate = None, [], []
        for step in range(thinker_steps):
            output, cache, states = self.thinker.model(hidden, positions, cache, deepstack, visual_mask)
            embeddings.append(states[0])
            intermediate.append(states[config.talker_config.accept_hidden_layer])
            token = self.argmax(self.thinker.lm_head(output[:, -1]).float())[:, None]
            history = torch.cat((history, token), -1)
            if int(token[0, 0]) == thinker_eos or step + 1 == thinker_steps:
                break
            hidden = self.thinker.model.embed_tokens(token)
            positions = torch.full((3, 1), ids.shape[1] + step + delta, device=ids.device, dtype=torch.long)
            deepstack, visual_mask = None, None
        thinker_embed, thinker_hidden = torch.cat(embeddings, 1), torch.cat(intermediate, 1)
        special = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], device=ids.device)
        bos, eos, pad = self.talker.text_projection(self.thinker.model.embed_tokens(special)).chunk(3, 1)
        starts = (ids[0] == config.im_start_token_id).nonzero().flatten().tolist() + [history.shape[1]]
        multimodal = ((history == config.thinker_config.audio_token_id) |
                      (history == config.thinker_config.image_token_id) |
                      (history == config.thinker_config.video_token_id))
        parts, id_parts, trailing = [], [], None
        for index, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            role = int(ids[0, start + 1])
            if role == config.system_token_id:
                continue
            if role == config.user_token_id:
                mask = multimodal[:, start:end]
                part = thinker_embed.new_empty((1, end - start, config.talker_config.text_config.hidden_size))
                if mask.any():
                    part[mask] = self.talker.hidden_projection(thinker_hidden[:, start:end][mask])
                part[~mask] = self.talker.text_projection(thinker_embed[:, start:end][~mask])
                parts.append(part)
                id_parts.append(history[:, start:end])
            elif role == config.assistant_token_id and index == len(starts) - 2:
                assistant = self.talker.text_projection(thinker_embed[:, start:end])
                text = torch.cat((assistant[:, :3], pad.expand(-1, 4, -1), bos, assistant[:, 3:4]), 1)
                tc = config.talker_config
                codec_ids = torch.tensor([[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id,
                                          tc.speaker_id["ethan"], tc.codec_pad_id, tc.codec_bos_id]], device=ids.device)
                codec = torch.cat((torch.zeros_like(text[:, :3]), self.talker.model.codec_embedding(codec_ids)), 1)
                parts.append(text + codec)
                id_parts.append(torch.full(text.shape[:2], config.tts_pad_token_id, device=ids.device, dtype=torch.long))
                trailing = torch.cat((assistant[:, 4:], eos), 1)
            elif role != config.assistant_token_id:
                raise ValueError("Expected ChatML system, user, or assistant role")
        if trailing is None:
            raise ValueError("Audio generation requires a final assistant prefix")
        talker_ids = torch.cat(id_parts, 1)
        talk_positions, talk_delta = positions_for_inputs(talker_ids, inputs, config.talker_config)
        codes = self.talker.generate(torch.cat(parts, 1), talk_positions, trailing, pad, talker_steps, talk_delta)
        return {"sequences": history, "waveform": self.code2wav.chunked_decode(codes).float()}


def build_from_config(config, device, dtype):
    config = config_values(config.to_dict())
    if not config.enable_audio_output:
        raise ValueError("Qwen3 Omni coverage preserves the default text-plus-audio parent")
    model = Omni(config).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if isinstance(module, LayerNorm):
            module.promote_fp32 = False
    return model


@torch.no_grad()
def load_state_dict_into(model, weights, config):
    remaining, mapped = dict(weights), {}
    for name, target in model.state_dict().items():
        source = name.replace(".emb.weight", ".weight")
        if name.endswith("inv_freq"):
            mapped[name] = target
            continue
        if ".mlp.w13" in source:
            source = source.replace(".mlp.w13", ".mlp.experts.gate_up_proj")
        if ".mlp.w2" in source:
            source = source.replace(".mlp.w2", ".mlp.experts.down_proj")
        if ".shared_expert.gate_up_proj." in source:
            value = torch.cat([remaining.pop(source.replace("gate_up_proj", part)) for part in ("gate_proj", "up_proj")])
        else:
            if name.startswith("thinker.visual."):
                source = source.replace("deepstack_merger_list.", "merger_list.")
                source = source.replace("pos_embed_interp._embed.weight", "pos_embed.weight")
                source = source.replace("patch_embed.proj.conv.", "patch_embed.proj.")
                source = source.replace(".mlp.fc1.", ".mlp.linear_fc1.").replace(".mlp.fc2.", ".mlp.linear_fc2.")
                if "merger" in source:
                    source = source.replace(".norm.", ".ln_q.")
                    source = source.replace(".fc1.", ".mlp.0.").replace(".fc2.", ".mlp.2.")
            value = remaining.pop(source)
        if value.shape != target.shape:
            raise ValueError(f"Qwen3 Omni state shape mismatch at {source}: {value.shape} != {target.shape}")
        mapped[name] = value
    if remaining:
        raise KeyError(f"Unmapped Qwen3 Omni state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
    for module in model.modules():
        if isinstance(module, SharedExpertMoE):
            module.process_weights_after_loading()


def make_workloads(model, inputs, config, case):
    generation = case["generation_kwargs"]
    return {"generate": Workload(run=lambda: model.generate(inputs, generation["thinker_max_new_tokens"],
        generation["talker_max_new_tokens"], generation.get("thinker_eos_token_id", 151645)))}
