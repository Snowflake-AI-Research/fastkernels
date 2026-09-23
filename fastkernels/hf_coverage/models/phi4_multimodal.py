"""Phi4's image and single-clip speech generation with the active modality LoRA.

The two author examples are separate workloads of this common model. The
reference adapter loader supplies the selected trained adapter in common state.
Padded multi-audio is excluded: pinned native and author masking disagree there.
"""

import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from ..patches.codec_top1 import CodecTop1
from ..patches.phi4_audio_affine import BiasedAudioSiluAndMul, NormalizedAudioConv2d
from ..runner import Workload


def norm(width, epsilon=1e-5):
    return LayerNorm(width, eps=epsilon, promote_fp32=False)


class LoRALinear(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict({'default': Linear(a.shape[1], a.shape[0], bias=False)})
        self.lora_B = nn.ModuleDict({'default': Linear(b.shape[1], b.shape[0], bias=False)})
        self.lora_A.to(device=base.weight.device, dtype=a.dtype)
        self.lora_B.to(device=base.weight.device, dtype=b.dtype)

    def forward(self, hidden):
        result = self.base_layer(hidden)
        update = self.lora_B.default(self.lora_A.default(hidden.to(self.lora_A.default.weight.dtype)))
        # Both author adapters have alpha/r = 2. PEFT casts back after addition.
        return (result + update * 2.0).to(result.dtype)


class MLP(nn.Module):
    def __init__(self, width, intermediate, approximate='none'):
        super().__init__()
        self.fc1, self.fc2 = Linear(width, intermediate), Linear(intermediate, width)
        self.act = GELU(approximate)

    def forward(self, hidden):
        return self.fc2(self.act(self.fc1(hidden)))


class Attention(nn.Module):
    def __init__(self, width, heads, output_name):
        super().__init__()
        self.heads, self.dim, self.output_name = heads, width // heads, output_name
        self.q_proj, self.k_proj, self.v_proj = (Linear(width, width) for _ in range(3))
        setattr(self, output_name, Linear(width, width))
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, mask):
        shape = (*hidden.shape[:2], self.heads, self.dim)
        q, k, v = (getattr(self, name)(hidden).reshape(shape) for name in ('q_proj', 'k_proj', 'v_proj'))
        return getattr(self, self.output_name)(self.attention(
            q, k, v, attn_mask=mask, causal=mask is None and hidden.shape[1] > 1,
        ).reshape_as(hidden))


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = norm(config.hidden_size, config.layer_norm_eps)
        self.layer_norm2 = norm(config.hidden_size, config.layer_norm_eps)
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads, 'out_proj')
        self.mlp = MLP(config.hidden_size, config.intermediate_size, 'tanh')

    def forward(self, hidden, mask):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden), mask)
        return hidden + self.mlp(self.layer_norm2(hidden))


class VisionPool(nn.Module):
    """Native MHA's query scaling, score BMM, softmax and value BMM."""

    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.probe = nn.Parameter(torch.empty(1, 1, width))
        self.q_proj, self.kv_proj, self.out_proj = Linear(width, width), Linear(width, 2 * width), Linear(width, width)
        self.heads, self.dim = config.num_attention_heads, width // config.num_attention_heads
        self.layernorm = norm(width, config.layer_norm_eps)
        self.mlp = MLP(width, config.intermediate_size, 'tanh')
        self.bmm, self.softmax = BMM(), Softmax(-1)
        self.average_weights = AvgPool2d((self.heads, 1), (1, 1))

    def forward(self, hidden, mask):
        batch, length, width = hidden.shape
        q = self.q_proj(self.probe.expand(batch, -1, -1)).reshape(batch, 1, self.heads, self.dim).transpose(1, 2)
        k, v = self.kv_proj(hidden).chunk(2, -1)
        k, v = (x.reshape(batch, length, self.heads, self.dim).transpose(1, 2) for x in (k, v))
        scores = self.bmm(q * (self.dim ** -0.5), k.transpose(-1, -2))
        if mask is not None:
            scores = scores + mask
        weights = self.softmax(scores)
        pooled = self.out_proj(self.bmm(weights, v).transpose(1, 2).reshape(batch, 1, width))
        # nn.MultiheadAttention also returns averaged weights by default.
        self.average_weights(weights.reshape(batch, 1, self.heads, length))
        return (pooled + self.mlp(self.layernorm(pooled)))[:, 0]


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Module()
        self.embeddings.patch_embedding = Conv2d(config.num_channels, config.hidden_size, config.patch_size, config.patch_size)
        self.embeddings.position_embedding = Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(VisionLayer(config) for _ in range(config.num_hidden_layers))
        self.post_layernorm = norm(config.hidden_size, config.layer_norm_eps)
        self.head = VisionPool(config)

    def forward(self, pixels, patch_mask):
        batch, height, width = patch_mask.shape
        side = self.config.image_size // self.config.patch_size
        boundaries = torch.arange(1 / side, 1., 1 / side, device=pixels.device)
        # Position arithmetic uses only supplied image-mask metadata.
        hcoord = torch.arange(height, device=pixels.device).float()[None] * (1.0 / patch_mask[:, :, 0].sum(1))[:, None]
        wcoord = torch.arange(width, device=pixels.device).float()[None] * (1.0 / patch_mask[:, 0, :].sum(1))[:, None]
        hids = torch.bucketize(hcoord.clamp(max=1.-1e-6).to(pixels.dtype), boundaries, right=True)
        wids = torch.bucketize(wcoord.clamp(max=1.-1e-6).to(pixels.dtype), boundaries, right=True)
        ids = (hids[:, :, None] * side + wids[:, None, :]).masked_fill(~patch_mask, 0).flatten(1)
        hidden = self.embeddings.patch_embedding(pixels).flatten(2).transpose(1, 2)
        hidden = hidden + self.embeddings.position_embedding(ids)
        mask = hidden.new_zeros(batch, 1, 1, height * width).masked_fill(~patch_mask.flatten(1)[:, None, None], -float('inf'))
        # Pinned native SDPA elides an all-valid mask and then infers causal
        # attention from its vision module flag. Preserve native execution;
        # this differs from author SigLIP and is recorded as a source issue.
        if bool(patch_mask.all()):
            mask = None
        states = [hidden]
        for layer in self.encoder.layers:
            hidden = layer(hidden, mask)
            states.append(hidden)
        # Native still executes these default outputs before selecting -2.
        last = self.post_layernorm(hidden)
        self.head(last, mask)
        return states[self.config.feature_layer]


class ImageEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.vision_config
        if (c.image_size // c.patch_size) % 2:
            raise ValueError('Selected Phi4 checkpoint has an even patch grid')
        self.img_processor = Vision(c)
        self.image_token_compression = AvgPool2d(2, 2)
        self.img_projection_up, self.img_projection_down = Linear(c.hidden_size, config.hidden_size), Linear(config.hidden_size, config.hidden_size)
        self.global_img_feature_extensor = nn.Parameter(torch.empty(1, 1, c.hidden_size))
        self.sub_img_feature_extensor = nn.Parameter(torch.empty(1, 1, 1, c.hidden_size))
        self.act = GELU()

    def forward(self, pixels, sizes, masks):
        c = self.config.vision_config
        features = self.img_processor(pixels.flatten(0, 1), masks.flatten(0, 1).bool())
        side = math.isqrt(features.shape[1])
        features = self.image_token_compression(features.reshape(-1, side, side, c.hidden_size).permute(0, 3, 1, 2))
        side = features.shape[-1]
        features = features.permute(0, 2, 3, 1).reshape(pixels.shape[0], -1, side * side, c.hidden_size)
        output = []
        for index, size in enumerate(sizes):
            rows, cols = (int(value) // c.crop_size for value in size)
            global_image = features[index, :1].reshape(1, side, side, c.hidden_size)
            global_image = torch.cat((global_image, self.sub_img_feature_extensor.expand(1, side, 1, -1)), 2).reshape(1, -1, c.hidden_size)
            sub = features[index, 1:1 + rows * cols].reshape(rows, cols, side, side, c.hidden_size).transpose(1, 2).reshape(1, rows * side, cols * side, c.hidden_size)
            submask = masks[index, 1:1 + rows * cols, ::2, ::2].reshape(rows, cols, side, side).transpose(1, 2).reshape(1, rows * side, cols * side)
            height, width = int(submask[0, :, 0].sum()), int(submask[0, 0, :].sum())
            sub = sub[:, :height, :width]
            sub = torch.cat((sub, self.sub_img_feature_extensor.expand(1, height, 1, -1)), 2).reshape(1, -1, c.hidden_size)
            merged = torch.cat((sub, self.global_img_feature_extensor, global_image), 1)
            output.append(self.img_projection_down(self.act(self.img_projection_up(merged))))
        return torch.cat(output, 1).squeeze(0)


class AudioMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = norm(config.hidden_size)
        self.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden):
        up, gate = self.gate_up_proj(self.layer_norm(hidden)).chunk(2, -1)
        return self.down_proj(SiluAndMul.forward_native(torch.cat((gate, up), -1)))


class AudioConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        if config.ext_pw_out_channel != width or config.depthwise_separable_out_channel != width:
            raise ValueError('Selected Phi4 audio convolutions preserve the hidden width')
        self.kernel = config.kernel_size
        self.layer_norm = norm(width)
        self.glu = nn.Module()
        self.glu.ext_pw_conv_1d = Conv1dNative(width, width * 2, 1)
        self.glu.b1, self.glu.b2 = nn.Parameter(torch.empty(1, width, 1)), nn.Parameter(torch.empty(1, width, 1))
        self.glu_act = BiasedAudioSiluAndMul()
        self.dw_sep_conv_1d = nn.Module()
        self.dw_sep_conv_1d.dw_conv = Conv1dNative(width, width * config.depthwise_multiplier, self.kernel, padding=self.kernel-1, groups=width)
        self.dw_sep_conv_1d.pw_conv = Conv1dNative(width * config.depthwise_multiplier, width, 1)
        self.ext_pw_conv_1d = Conv1dNative(width, width, 1)
        self.act = SiLU()

    def forward(self, hidden):
        up, gate = self.glu.ext_pw_conv_1d(self.layer_norm(hidden).transpose(1, 2)).chunk(2, 1)
        packed = torch.cat((gate, up), 1).transpose(1, 2)
        bias = torch.cat((self.glu.b2, self.glu.b1), 1).transpose(1, 2)
        hidden = self.glu_act(packed, bias).transpose(1, 2)
        hidden = self.dw_sep_conv_1d.pw_conv(self.dw_sep_conv_1d.dw_conv(hidden))
        if self.kernel > 1:
            hidden = hidden[:, :, :-(self.kernel-1)]
        return self.ext_pw_conv_1d(self.act(hidden)).transpose(1, 2)


class AudioLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward_in, self.feed_forward_out = AudioMLP(config), AudioMLP(config)
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads, 'o_proj')
        self.conv = AudioConv(config)
        self.layer_norm_att, self.layer_norm = norm(config.hidden_size), norm(config.hidden_size)

    def forward(self, hidden, mask):
        hidden = hidden + self.feed_forward_in(hidden) * 0.5
        hidden = hidden + self.self_attn(self.layer_norm_att(hidden), mask)
        hidden = hidden + self.conv(hidden)
        hidden = hidden + self.feed_forward_out(hidden) * 0.5
        return self.layer_norm(hidden)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.chunk_size != -1 or config.downsample_rate != 1 or config.time_reduction != 8:
            raise ValueError('Preserve the selected nonstreaming factor-eight audio frontend')
        self.encoder_embedding = nn.Module()
        self.encoder_embedding.register_buffer('global_mean', torch.empty(config.input_size))
        self.encoder_embedding.register_buffer('global_invstd', torch.empty(config.input_size))
        width = config.nemo_conv_channels
        conv = [NormalizedAudioConv2d(1, width, 3, 2, 1), ReLU()]
        for _ in range(2):
            conv.extend((Conv2d(width, width, 3, 2, 1, groups=width), Conv2d(width, width, 1), ReLU()))
        self.embed = nn.Module()
        self.embed.conv = nn.ModuleList(conv)
        self.embed.out = Linear(width * config.nemo_final_size, config.hidden_size)
        self.relative_attention_bias_layer = nn.Module()
        self.relative_attention_bias_layer.bias_values = Embedding(config.bias_max_distance * (1 if config.bias_symmetric else 2), config.num_attention_heads)
        self.encoders = nn.ModuleList(AudioLayer(config) for _ in range(config.num_blocks))

    def forward(self, features):
        affine = self.encoder_embedding
        hidden = self.embed.conv[0](features, affine.global_mean, affine.global_invstd)
        for operation in self.embed.conv[1:]:
            hidden = operation(hidden)
        batch, _, length, _ = hidden.shape
        hidden = self.embed.out(hidden.transpose(1, 2).reshape(batch, length, -1))
        # The author and native single-clip branches unfold into independent
        # 500-frame chunks, including zero-padded final chunk frames.
        if length > 500:
            padding = (-length) % 500
            if padding:
                hidden = torch.cat((hidden, hidden.new_zeros(batch, padding, hidden.shape[-1])), 1)
            hidden = hidden.reshape(-1, 500, hidden.shape[-1])
        size = hidden.shape[1]
        position = torch.arange(size, device=hidden.device)
        relative = (position[None, :] - position[:, None]).clamp(-self.config.bias_max_distance, self.config.bias_max_distance-1)
        indices = relative.abs() if self.config.bias_symmetric else relative + self.config.bias_max_distance
        # Native adds its all-true nonstreaming mask as a constant score bias.
        mask = self.relative_attention_bias_layer.bias_values(indices).permute(2, 0, 1).unsqueeze(0) + 1
        for layer in self.encoders:
            hidden = layer(hidden, mask)
        return hidden.reshape(batch, -1, hidden.shape[-1])[:, :length]


class AudioEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AudioEncoder(config.audio_config)
        for modality in ('speech', 'vision_speech'):
            setattr(self, 'up_proj_for_' + modality, Linear(config.audio_config.hidden_size, config.hidden_size))
            setattr(self, 'down_proj_for_' + modality, Linear(config.hidden_size, config.hidden_size))
        self.act = GELU()

    def forward(self, features, sizes):
        hidden = self.down_proj_for_speech(self.act(self.up_proj_for_speech(self.encoder(features))))
        return torch.cat([hidden[index, :int(size)] for index, size in enumerate(sizes)], 0)


class TextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = config.hidden_size // self.heads
        self.rotary_dim = int(self.dim * config.rope_parameters['partial_rotary_factor'])
        self.qkv_proj = Linear(config.hidden_size, (self.heads + 2 * self.kv_heads) * self.dim, bias=False)
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attention = DenseAttention(backend='sdpa')
        self.key = self.value = None

    def forward(self, hidden, positions, rotary):
        batch, length, _ = hidden.shape
        q, k, v = self.qkv_proj(hidden).split((self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim), -1)
        q, k, v = (x.reshape(batch, length, -1, self.dim) for x in (q, k, v))
        qr, kr = RotaryEmbedding.forward_native(positions.repeat(batch), q[..., :self.rotary_dim].reshape(batch*length, -1), k[..., :self.rotary_dim].reshape(batch*length, -1), self.rotary_dim, rotary)
        q = torch.cat((qr.reshape(batch, length, self.heads, self.rotary_dim), q[..., self.rotary_dim:]), -1)
        k = torch.cat((kr.reshape(batch, length, self.kv_heads, self.rotary_dim), k[..., self.rotary_dim:]), -1)
        self.key = k if self.key is None else torch.cat((self.key, k), 1)
        self.value = v if self.value is None else torch.cat((self.value, v), 1)
        keys = torch.arange(self.key.shape[1], device=hidden.device)
        allowed = keys[None, :] <= positions[:, None]
        if self.config.sliding_window is not None:
            allowed &= keys[None, :] > positions[:, None] - self.config.sliding_window
        k = self.key.repeat_interleave(self.heads // self.kv_heads, 2)
        v = self.value.repeat_interleave(self.heads // self.kv_heads, 2)
        output = self.attention(q, k, v, attn_mask=allowed[None, None])
        return self.o_proj(output.reshape(batch, length, -1))


class TextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = TextAttention(config)
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.mlp.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, positions, rotary):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, rotary)
        return hidden + self.mlp.down_proj(SiluAndMul.forward_native(self.mlp.gate_up_proj(self.post_attention_layernorm(hidden))))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.model.embed_tokens_extend = nn.Module()
        self.model.embed_tokens_extend.image_embed = ImageEmbedding(config)
        self.model.embed_tokens_extend.audio_embed = AudioEmbedding(config)
        self.model.layers = nn.ModuleList(TextLayer(config) for _ in range(config.num_hidden_layers))
        self.model.norm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.select = CodecTop1()

    def reset(self):
        for layer in self.model.layers:
            layer.self_attn.key = layer.self_attn.value = None

    def rotary(self, length, dtype, device):
        # LongRoPE frequencies and positions are configuration metadata. The
        # unchanged rotary parent performs all arithmetic on activation data.
        c, rope = self.config, self.config.rope_parameters
        dim = int((c.hidden_size // c.num_attention_heads) * rope['partial_rotary_factor'])
        original = rope['original_max_position_embeddings']
        factors = torch.tensor(rope['long_factor' if length > original else 'short_factor'], dtype=torch.float32, device=device)
        inverse = 1.0 / (factors * rope['rope_theta'] ** (torch.arange(0, dim, 2, device=device).float() / dim))
        ratio = rope.get('factor', c.max_position_embeddings / original)
        scale = rope.get('attention_factor', 1.0 if ratio <= 1 else math.sqrt(1 + math.log(ratio) / math.log(original)))
        phase = torch.arange(length, device=device).float()[:, None] * inverse[None]
        return torch.cat((phase.cos() * scale, phase.sin() * scale), -1).to(dtype)

    def forward(self, ids, start, inputs):
        hidden = self.model.embed_tokens(ids)
        extensions = self.model.embed_tokens_extend
        if start == 0:
            if 'image_pixel_values' in inputs:
                features = extensions.image_embed(inputs['image_pixel_values'], inputs['image_sizes'], inputs['image_attention_mask'])
                hidden[ids == self.config.vision_config.image_token_id] = features.to(hidden.dtype)
            else:
                features = extensions.audio_embed(inputs['audio_input_features'], inputs['audio_embed_sizes'])
                hidden[ids == self.config.audio_config.audio_token_id] = features.to(hidden.dtype)
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)
        rotary = self.rotary(start + ids.shape[1], hidden.dtype, hidden.device)
        for layer in self.model.layers:
            hidden = layer(hidden, positions, rotary)
        return self.lm_head(self.model.norm(hidden)[:, -1:])

    def generate(self, input_ids, max_new_tokens, eos_token_id, **inputs):
        self.reset()
        ids = input_ids.clone()
        outputs = {}
        original = self.config.rope_parameters['original_max_position_embeddings']
        eos = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        for step in range(max_new_tokens):
            refresh = step == 0 or ids.shape[1] == original + 1
            if refresh:
                self.reset()
            current, start = (ids, 0) if refresh else (ids[:, -1:], ids.shape[1]-1)
            # GenerationMixin exposes a copied FP32 last-token logit tensor
            # for each step, before token selection or logits processing.
            logits = self.forward(current, start, inputs)[:, -1].to(dtype=torch.float32, copy=True)
            outputs[f'logits.{step}'] = logits
            token = self.select(logits).reshape(1, 1)
            ids = torch.cat((ids, token), 1)
            if int(token[0, 0]) in eos:
                break
        outputs['sequences'] = ids
        # Like native generation, the final cache contains the last evaluated
        # prompt, excluding the final token that was selected from its logits.
        for index, layer in enumerate(self.model.layers):
            outputs[f'past_key_values.{index}.key'] = layer.self_attn.key.transpose(1, 2)
            outputs[f'past_key_values.{index}.value'] = layer.self_attn.value.transpose(1, 2)
        return outputs


def build_from_config(config, device, dtype):
    if config.hidden_act != 'silu' or config.rope_parameters['rope_type'] != 'longrope' or not config.use_cache:
        raise ValueError('Selected Phi4 task retains SiLU, partial LongRoPE and cached generation')
    if any(getattr(config.audio_config, name) != 'swish' for name in ('activation', 'conv_activation', 'conv_glu_type')):
        raise ValueError('Selected Phi4 audio activation is swish')
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for index, layer in enumerate(model.model.layers):
        for branch, name in (('self_attn', 'qkv_proj'), ('self_attn', 'o_proj'), ('mlp', 'gate_up_proj'), ('mlp', 'down_proj')):
            prefix = f'model.layers.{index}.{branch}.{name}.'
            a, b = (remaining[prefix + f'lora_{letter}.default.weight'] for letter in ('A', 'B'))
            owner = getattr(layer, branch)
            setattr(owner, name, LoRALinear(getattr(owner, name), a, b))
    prefix = 'model.embed_tokens_extend.image_embed.img_processor.head.'
    weight, bias = remaining.pop(prefix + 'attention.in_proj_weight'), remaining.pop(prefix + 'attention.in_proj_bias')
    width = config.vision_config.hidden_size
    for field, source in (('weight', weight), ('bias', bias)):
        remaining[prefix + 'q_proj.' + field] = source[:width]
        remaining[prefix + 'kv_proj.' + field] = source[width:]
        remaining[prefix + 'out_proj.' + field] = remaining.pop(prefix + 'attention.out_proj.' + field)
    mapped = {name: remaining.pop(name.replace('.emb.weight', '.weight')) for name in model.state_dict()}
    if remaining:
        raise KeyError(f'Unmapped Phi4 weights: {sorted(remaining)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case):
    image, audio = 'image_pixel_values' in inputs, 'audio_input_features' in inputs
    if image == audio or inputs['input_ids'].shape[0] != 1:
        raise ValueError('Use one literal image or single-audio author workload per adapter job')
    if inputs.get('audio_attention_mask') is not None:
        raise ValueError('Padded multi-audio is outside the source-equivalent author workload')
    if 'attention_mask' in inputs and not bool(inputs['attention_mask'].all()):
        raise ValueError('The selected one-prompt author examples have no text padding')
    adapter = case['reference']['adapter_config']
    rank = 256 if image else 320
    if adapter['r'] != rank or adapter['lora_alpha'] != 2 * rank or adapter.get('use_dora', False) or adapter.get('use_rslora', False):
        raise ValueError('Retain the active author vision or speech LoRA rank and scaling')
    options = dict(case['generation_kwargs'])
    if options.pop('do_sample', False) or options.pop('num_beams', 1) != 1 or not options.pop('use_cache', True):
        raise ValueError('The author task uses greedy cached generation')
    return {'generate': Workload(run=lambda: model.generate(**inputs, **options))}
