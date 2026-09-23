"""CLVP sampled speech-code generation and both contrastive encoders."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L2.geglu import GEGLU
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from ..runner import Workload
from .qwen_omni_sampling import OmniSampling


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size, bias=config.use_attention_bias))
        self.out_proj = Linear(config.hidden_size, config.hidden_size)
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, rotary=None, mask=None, previous=None):
        batch, length, width = hidden.shape
        q, k, v = (getattr(self, name)(hidden).reshape(batch, length, self.heads, self.width).transpose(1, 2)
                   for name in ("q_proj", "k_proj", "v_proj"))
        q = q * self.width**-0.5
        if previous is not None:
            k, v = torch.cat((previous[0], k), 2), torch.cat((previous[1], v), 2)
        cache = (k, v)
        if rotary is not None:
            rotary_width = rotary.shape[-1]
            positions = torch.arange(length, device=hidden.device).repeat(batch)
            flat = [x.transpose(1, 2)[..., :rotary_width].reshape(batch * length, -1) for x in (q, k, v)]
            q_rot, k_rot = RotaryEmbedding.forward_native(positions, flat[0], flat[1], rotary_width, rotary)
            v_rot, _ = RotaryEmbedding.forward_native(positions, flat[2], flat[2], rotary_width, rotary)
            q, k, v = (torch.cat((rot.reshape(batch, length, self.heads, rotary_width).transpose(1, 2), original[..., rotary_width:]), -1)
                       for rot, original in zip((q_rot, k_rot, v_rot), (q, k, v)))
        scores = self.bmm(q.reshape(-1, length, self.width), k.reshape(-1, k.shape[2], self.width).transpose(1, 2))
        scores = scores.reshape(batch, self.heads, length, -1)
        if mask is not None:
            scores = scores + mask
        output = self.bmm(self.softmax(scores).reshape(batch * self.heads, length, -1), v.reshape(batch * self.heads, -1, self.width))
        output = output.reshape(batch, self.heads, length, self.width).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output), cache


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_rmsnorm = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attention_rmsnorm = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = nn.Module()
        self.mlp.fc1 = GEGLU(config.hidden_size, config.intermediate_size)
        self.mlp.fc2 = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden, rotary, mask):
        hidden = hidden + self.self_attn(self.input_rmsnorm(hidden), rotary, mask)[0]
        return hidden + self.mlp.fc2(self.mlp.fc1(self.post_attention_rmsnorm(hidden)))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.projection = Linear(config.hidden_size, config.projection_dim, bias=False)
        self.pool = GlobalAvgPool2d()
        self.rotary_pos_emb = nn.Module()
        dim = max(config.projection_dim // (config.num_attention_heads * 2), 32)
        self.rotary_pos_emb.register_buffer("inv_freq", 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, ids, attention_mask=None):
        hidden = self.token_embedding(ids)
        frequencies = torch.arange(ids.shape[1], device=ids.device).to(self.rotary_pos_emb.inv_freq)[:, None] * self.rotary_pos_emb.inv_freq
        rotary = torch.cat((frequencies.cos(), frequencies.sin()), -1)
        mask = None if attention_mask is None else torch.zeros((ids.shape[0], 1, ids.shape[1], ids.shape[1]), device=ids.device, dtype=hidden.dtype).masked_fill(~attention_mask[:, None, None].bool(), torch.finfo(hidden.dtype).min)
        for layer in self.layers:
            hidden = layer(hidden, rotary, mask)
        hidden = self.final_layer_norm(hidden)
        pooled = self.pool(hidden.transpose(1, 2).unsqueeze(-1))
        return self.projection(pooled), pooled


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.attn = Attention(config)
        self.mlp = nn.Module()
        inner = config.n_inner or 4 * config.hidden_size
        self.mlp.c_fc, self.mlp.c_proj = Linear(config.hidden_size, inner), Linear(inner, config.hidden_size)
        self.activation = NewGELUActivation()

    def forward(self, hidden, mask, previous):
        attention, cache = self.attn(self.input_layernorm(hidden), mask=mask, previous=previous)
        hidden = hidden + attention
        return hidden + self.mlp.c_proj(self.activation(self.mlp.c_fc(self.post_attention_layernorm(hidden)))), cache


class SpeechDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        decoder = nn.Module()
        self.model.decoder = decoder
        decoder.input_embeds_layer = Embedding(config.vocab_size, config.hidden_size)
        decoder.position_embeds_layer = Embedding(config.max_position_embeddings, config.hidden_size)
        decoder.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        decoder.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, promote_fp32=False)
        self.final_norm = LayerNorm(config.hidden_size, promote_fp32=False)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden, positions, previous=None, valid_length=None):
        decoder = self.model.decoder
        hidden = hidden + decoder.position_embeds_layer(positions)
        length = hidden.shape[1]
        offset = 0 if previous is None else previous[0][0].shape[2]
        queries = torch.arange(length, device=hidden.device) + offset
        keys = torch.arange(length + offset, device=hidden.device)
        mask = torch.zeros(length, length + offset, device=hidden.device, dtype=hidden.dtype).masked_fill(keys[None] > queries[:, None], torch.finfo(hidden.dtype).min)[None, None]
        if valid_length is not None:
            mask.masked_fill_(keys[None, None, None] >= valid_length, torch.finfo(hidden.dtype).min)
        caches = []
        for index, layer in enumerate(decoder.layers):
            hidden, cache = layer(hidden, mask, None if previous is None else previous[index])
            caches.append(cache)
        return self.lm_head(self.final_norm(decoder.layer_norm(hidden))), caches


def pad_tokens(ids, mask, bos, eos, add_bos=True):
    if add_bos:
        ids = torch.nn.functional.pad(ids, (1, 0), value=bos)
        mask = torch.nn.functional.pad(mask, (1, 0), value=1)
    rows = []
    for row in ids:
        pads = torch.where(row == 0)[0]
        position = int(pads[0]) if pads.numel() else row.shape[0]
        rows.append(torch.cat((row[:position], row.new_tensor([eos]), row[position:])))
    return torch.stack(rows), torch.nn.functional.pad(mask, (1, 0), value=1)


class ConditioningEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder = config.decoder_config
        self.config = config
        self.text_token_embedding = Embedding(config.text_config.vocab_size, decoder.hidden_size)
        self.text_position_embedding = Embedding(decoder.max_text_tokens, decoder.hidden_size)
        self.mel_conv = Conv1dNative(decoder.feature_size, decoder.hidden_size, 1)
        groups = 8 if decoder.hidden_size <= 16 else 16 if decoder.hidden_size <= 64 else 32
        while decoder.hidden_size % groups:
            groups //= 2
        if groups <= 2:
            raise ValueError("CLVP conditioning requires more than two normalization groups")
        self.group_norms = nn.ModuleList(GroupNorm(groups, decoder.hidden_size, eps=1e-5) for _ in range(decoder.num_mel_attn_blocks))
        self.mel_attn_blocks = nn.ModuleList(Attention(decoder) for _ in range(decoder.num_mel_attn_blocks))

    def forward(self, ids, mask, features):
        text = self.config.text_config
        ids, mask = pad_tokens(ids, mask, text.bos_token_id, text.eos_token_id)
        hidden = self.text_token_embedding(ids) + self.text_position_embedding(mask.cumsum(-1) - 1)
        mel = self.mel_conv(features)
        for norm, attention in zip(self.group_norms, self.mel_attn_blocks):
            mel = (attention(norm(mel).transpose(1, 2))[0] + mel.transpose(1, 2)).transpose(1, 2)
        mel = mel[:, :, 0][:, None]
        return torch.cat((mel, hidden), 1)


class Clvp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conditioning_encoder = ConditioningEncoder(config)
        self.speech_decoder_model = SpeechDecoder(config.decoder_config)
        self.text_encoder_model, self.speech_encoder_model = Encoder(config.text_config), Encoder(config.speech_config)
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.normalize, self.bmm = L2Norm(eps=0), BatchMatMul()


def build_from_config(config, device, dtype):
    for encoder in (config.text_config, config.speech_config):
        if not encoder.use_rotary_embedding or encoder.summary_type != "mean" or encoder.hidden_act != "gelu":
            raise ValueError("CLVP preserves rotary Q/K/V encoders with mean pooling and GEGLU")
    if config.decoder_config.activation_function != "gelu_new":
        raise ValueError("CLVP preserves its decoder's separately rounded GELU")
    return Clvp(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        value = remaining.pop(source)
        if name.endswith((".mlp.c_fc.weight", ".mlp.c_proj.weight")):
            value = value.t()
        mapped[name] = value
    if remaining:
        raise KeyError(f"Unmapped CLVP state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
    model.fixed_logit_scale = model.logit_scale.detach().exp()


def make_workloads(model, inputs, config, *, case):
    generation = case["reference"]["generation_config"]
    if not generation.get("do_sample") or generation.get("num_beams", 1) != 1:
        raise ValueError("CLVP preserves its checkpoint's sampled single-beam generation")
    sampler = OmniSampling(generation.get("top_k", 50), generation.get("top_p", 1.0), generation.get("temperature", 1.0), generation.get("repetition_penalty", 1.0))
    steps = case["generation_kwargs"]["max_new_tokens"]
    if inputs["input_ids"].shape[0] != 1 or inputs["input_features"].shape[0] != 1:
        raise ValueError("The CLVP development workload uses one text/audio pair")

    def run():
        text = config.text_config
        ids, mask = pad_tokens(inputs["input_ids"], inputs["attention_mask"], text.bos_token_id, text.eos_token_id, False)
        hidden = model.conditioning_encoder(ids, mask, inputs["input_features"])
        decoder = model.speech_decoder_model.model.decoder
        sequences = ids.new_full((1, 1), config.decoder_config.bos_token_id)
        start = decoder.input_embeds_layer(sequences) + decoder.position_embeds_layer(torch.zeros_like(sequences))
        hidden = torch.cat((hidden, start), 1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
        # Native conditioning preparation subtracts arange positions, while
        # GenerationMixin supplies a single initial position zero to broadcast.
        hidden = hidden - decoder.position_embeds_layer(positions)
        positions = ids.new_zeros((1, 1))
        previous = None
        for step in range(steps):
            logits, previous = model.speech_decoder_model(hidden, positions, previous, sequences.shape[1])
            next_ids = sampler(logits[:, -1].float(), sequences).reshape(1, 1)
            sequences = torch.cat((sequences, next_ids), 1)
            if int(next_ids.item()) == config.decoder_config.eos_token_id:
                break
            positions = ids.new_tensor([sequences.shape[1]])
            hidden = decoder.input_embeds_layer(next_ids)
        speech_ids = sequences[:, 1:].clone()
        codes = config.decoder_config.decoder_fixing_codes
        for row in speech_ids:
            stops = torch.where(row == config.decoder_config.eos_token_id)[0]
            if stops.numel():
                row[int(stops[0]):] = codes[0]
                row[-3:] = row.new_tensor(codes[1:])
        speech, speech_pool = model.speech_encoder_model(speech_ids)
        text_embed, text_pool = model.text_encoder_model(ids, mask)
        speech, text_embed = model.normalize(speech), model.normalize(text_embed)
        similarity = model.bmm(text_embed[None], speech.t()[None])[0] * model.fixed_logit_scale
        return {"speech_ids": speech_ids, "logits_per_speech": similarity.t(), "logits_per_text": similarity,
                "text_embeds": text_embed, "speech_embeds": speech,
                "text_model_output": text_pool, "speech_model_output": speech_pool}

    return {"generate": Workload(run=run)}
