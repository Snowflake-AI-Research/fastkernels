"""Gemma4 E2B image/text composition from existing FastKernels operations."""
import torch
from torch import nn

from ..patches.gemma4_rms_norm import Gemma4RMSNorm as Norm
from ..patches.dfine_clamp import DFineClamp
from ..patches.product_gate import ProductGate
from ..patches.codec_top1 import CodecTop1
from ..patches.detector_topk import DetectorTopK
from ..patches.qwen_omni_sampling import sample_nucleus
from ..runner import Workload
from fastkernels.tasks.baseline.L1.linear import Linear, BMM
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.tanh import Tanh


class ScaledEmbedding(Embedding):
    def __init__(self, size, width, scale):
        super().__init__(size, width)
        self.scale = scale

    def forward(self, ids):
        x = super().forward(ids)
        return x * torch.tensor(self.scale, device=x.device, dtype=x.dtype)


class ClippedLinear(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.linear = Linear(source, target, bias=False)
        for name, value in [('input_min', -float('inf')), ('input_max', float('inf')),
                            ('output_min', -float('inf')), ('output_max', float('inf'))]:
            self.register_buffer(name, torch.tensor(value))
        self.clamp = DFineClamp()

    def forward(self, x):
        return self.clamp(self.linear(self.clamp(x, self.input_min, self.input_max)),
                          self.output_min, self.output_max)


class MLP(nn.Module):
    def __init__(self, c, middle=None, clipped=False):
        super().__init__()
        middle = middle or c.intermediate_size
        linear = ClippedLinear if clipped else lambda a, b: Linear(a, b, bias=False)
        self.gate_proj = linear(c.hidden_size, middle)
        self.up_proj = linear(c.hidden_size, middle)
        self.down_proj = linear(middle, c.hidden_size)
        self.act, self.product = GELU(approximate='tanh'), ProductGate()

    def forward(self, x):
        return self.down_proj(self.product(torch.cat((self.act(self.gate_proj(x)), self.up_proj(x)), -1)))


def rope(x, positions, theta, proportion=1.0):
    # Angles depend only on positions/configuration metadata. The two actual
    # activation products use ProductGate with the native rounding boundaries.
    dim = x.shape[-1]
    n = int(proportion * dim // 2)
    inv = torch.cat((1 / theta ** (torch.arange(0, 2*n, 2, device=x.device).float()/dim),
                     torch.zeros(dim//2-n, device=x.device)))
    angles = positions.reshape(-1, 1).float() * inv[None]
    cos = torch.cat((angles.cos(), angles.cos()), -1).to(x.dtype).reshape(*x.shape[:2], 1, dim).expand_as(x)
    sin = torch.cat((angles.sin(), angles.sin()), -1).to(x.dtype).reshape(*x.shape[:2], 1, dim).expand_as(x)
    rotated = torch.cat((-x[..., dim//2:], x[..., :dim//2]), -1)
    product = ProductGate()
    return product(torch.cat((x, cos), -1)) + product(torch.cat((rotated, sin), -1))


class Attention(nn.Module):
    def __init__(self, c, index, vision=False):
        super().__init__()
        self.c, self.index, self.vision = c, index, vision
        self.kind = 'vision' if vision else c.layer_types[index]
        self.dim = c.head_dim if vision or self.kind == 'sliding_attention' else c.global_head_dim
        first = c.num_hidden_layers if vision else c.num_hidden_layers-c.num_kv_shared_layers
        self.shared = not vision and index >= first
        self.save_shared = not vision and not self.shared and index == max(
            j for j in range(first) if c.layer_types[j] == self.kind)
        linear = ClippedLinear if vision else lambda a, b: Linear(a, b, bias=c.attention_bias)
        self.q_proj = linear(c.hidden_size, c.num_attention_heads*self.dim)
        self.q_norm = Norm(self.dim, c.rms_norm_eps)
        if not self.shared:
            self.k_proj = linear(c.hidden_size, c.num_key_value_heads*self.dim)
            self.v_proj = linear(c.hidden_size, c.num_key_value_heads*self.dim)
            self.k_norm = Norm(self.dim, c.rms_norm_eps)
            self.v_norm = Norm(self.dim, c.rms_norm_eps, False)
        self.o_proj = linear(c.num_attention_heads*self.dim, c.hidden_size)
        self.core = DenseAttention(backend='sdpa')

    def rotate(self, x, positions):
        if self.vision:
            return torch.cat([rope(part, positions[..., i], self.c.rope_parameters['rope_theta'])
                              for i, part in enumerate(x.chunk(2, -1))], -1)
        p = self.c.rope_parameters[self.kind]
        return rope(x, positions, p['rope_theta'], p.get('partial_rotary_factor', 1.0))

    def forward(self, x, positions, mask, shared, cache=None):
        shape = (*x.shape[:2], -1, self.dim)
        q = self.rotate(self.q_norm(self.q_proj(x).view(shape)), positions)
        if self.shared:
            k, v = shared[self.kind]
        else:
            k = self.rotate(self.k_norm(self.k_proj(x).view(shape)), positions)
            v = self.v_norm(self.v_proj(x).view(shape))
            if cache is not None:
                if isinstance(cache, StaticKV):
                    k, v = cache.update(self.index, k, v, self.kind)
                elif self.index in cache:
                    k, v = [torch.cat(pair, 1) for pair in zip(cache[self.index], (k, v))]
                if not isinstance(cache, StaticKV):
                    cache[self.index] = (k, v)
            if self.save_shared:
                shared[self.kind] = k, v
        groups = self.c.num_attention_heads // self.c.num_key_value_heads
        if groups != 1:
            k, v = [t.repeat_interleave(groups, 2) for t in (k, v)]
        output = self.core(q, k, v, softmax_scale=1.0, attn_mask=mask)
        return self.o_proj(output.reshape(*x.shape[:2], -1))


class Layer(nn.Module):
    def __init__(self, c, index, vision=False):
        super().__init__()
        self.self_attn = Attention(c, index, vision)
        shared = not vision and index >= c.num_hidden_layers-c.num_kv_shared_layers > 0
        middle = c.intermediate_size * (2 if shared and c.use_double_wide_mlp else 1)
        self.mlp = MLP(c, middle, clipped=vision)
        for name in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
                     'post_feedforward_layernorm'):
            setattr(self, name, Norm(c.hidden_size, c.rms_norm_eps))
        self.ple = 0 if vision else c.hidden_size_per_layer_input
        if not vision:
            self.register_buffer('layer_scalar', torch.ones(1))
        if self.ple:
            self.per_layer_input_gate = Linear(c.hidden_size, self.ple, bias=False)
            self.per_layer_projection = Linear(self.ple, c.hidden_size, bias=False)
            self.post_per_layer_input_norm = Norm(c.hidden_size, c.rms_norm_eps)
            self.act, self.product = GELU(approximate='tanh'), ProductGate()

    def forward(self, x, positions, mask, shared, cache=None, ple=None):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), positions, mask, shared, cache))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        if self.ple:
            x = x + self.post_per_layer_input_norm(self.per_layer_projection(
                self.product(torch.cat((self.act(self.per_layer_input_gate(x)), ple), -1))))
        return x * self.layer_scalar if hasattr(self, 'layer_scalar') else x


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed_tokens = ScaledEmbedding(c.vocab_size, c.hidden_size, c.hidden_size**.5)
        self.layers = nn.ModuleList([Layer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = Norm(c.hidden_size, c.rms_norm_eps)
        if c.hidden_size_per_layer_input:
            p = c.hidden_size_per_layer_input
            self.embed_tokens_per_layer = ScaledEmbedding(c.vocab_size_per_layer_input, c.num_hidden_layers*p, p**.5)
            self.per_layer_model_projection = Linear(c.hidden_size, c.num_hidden_layers*p, bias=False)
            self.per_layer_projection_norm = Norm(p, c.rms_norm_eps)

    def forward(self, x, ids, positions, masks, cache=None, shared=None):
        c = self.c
        ple = None
        if c.hidden_size_per_layer_input:
            shape = (*x.shape[:2], c.num_hidden_layers, c.hidden_size_per_layer_input)
            token = self.embed_tokens_per_layer(ids).reshape(shape)
            projected = self.per_layer_model_projection(x) * c.hidden_size**-.5
            ple = (self.per_layer_projection_norm(projected.reshape(shape)) + token) * 2**-.5
        shared = {} if shared is None else shared
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, masks[c.layer_types[i]], shared, cache,
                      ple[:, :, i] if ple is not None else None)
        return self.norm(x), shared


class Vision(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.patch_embedder = nn.Module()
        self.patch_embedder.input_proj = Linear(3*c.patch_size**2, c.hidden_size, bias=False)
        self.patch_embedder.position_embedding_table = nn.Parameter(torch.empty(2, c.position_embedding_size, c.hidden_size))
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Layer(c, i, vision=True) for i in range(c.num_hidden_layers)])
        self.mm = BMM()

    def forward(self, pixels, positions):
        c = self.c
        padding = (positions == -1).all(-1)
        x = self.patch_embedder.input_proj((2*(pixels-.5)).to(self.patch_embedder.input_proj.weight.dtype))
        indices = positions.clamp(min=0)
        onehot = torch.nn.functional.one_hot(indices, c.position_embedding_size).permute(0, 2, 1, 3)
        embeds = self.mm(onehot.to(x.dtype), self.patch_embedder.position_embedding_table)
        pos_embed = embeds[:, 0] + embeds[:, 1]
        x = x + pos_embed.masked_fill(padding[..., None], 0)
        # HF's SDPA mask preparation omits an all-valid vision mask. Keeping
        # that redundant mask changes CUDA BF16 kernel dispatch and rounding.
        # Padding depends only on the integer patch-position metadata.
        mask = (~padding)[:, None, None, :] if padding.any() else None
        for layer in self.encoder.layers:
            x = layer(x, positions, mask, {})
        x = x.masked_fill(padding[..., None], 0)
        k = c.pooling_kernel_size
        length = x.shape[1] // k**2
        idx = indices // k
        group = idx[..., 0] + ((indices[..., 0].max(-1, keepdim=True).values+1)//k)*idx[..., 1]
        weights = torch.nn.functional.one_hot(group, length).float() / k**2
        pooled = self.mm(weights.transpose(1, 2), x.float()).to(x.dtype)
        valid = ~(weights == 0).all(1)
        return (pooled * c.hidden_size**.5)[valid]


class StaticKV:
    """Static full/sliding cache storage; its copies and rolls stay in timing."""
    def __init__(self, maximum, window):
        self.maximum, self.window = maximum, window
        self.seen, self.layers = 0, {}

    def sizes(self, kind, length):
        if kind == 'full_attention':
            return self.maximum, 0
        width = min(self.maximum, self.window)
        offset = max(self.seen-width+1, 0)
        length = width+length-1 if self.seen >= width else max(width, self.seen+length)
        return length, offset

    def update(self, index, k, v, kind):
        width = self.maximum if kind == 'full_attention' else min(self.maximum, self.window)
        if index not in self.layers:
            self.layers[index] = tuple(t.new_zeros(t.shape[0], width, t.shape[2], t.shape[3]) for t in (k, v))
        stored = self.layers[index]
        if self.seen+k.shape[1] <= width:
            for target, source in zip(stored, (k, v)):
                target[:, self.seen:self.seen+k.shape[1]] = source
            return stored
        if kind != 'sliding_attention':
            raise ValueError('Static cache capacity exceeded')
        if self.seen == 0:
            full = (k, v)
        elif self.seen >= width:
            full = tuple(torch.cat((old[:, 1:], new), 1) for old, new in zip(stored, (k, v)))
        else:
            full = tuple(torch.cat((old[:, :self.seen], new), 1) for old, new in zip(stored, (k, v)))
        for target, source in zip(stored, full):
            target.copy_(source[:, -width:])
        return stored if self.seen >= width and k.shape[1] == 1 else full


class ImageTextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.model = nn.Module()
        self.model.language_model = TextModel(c.text_config)
        self.model.vision_tower = Vision(c.vision_config)
        self.model.embed_vision = nn.Module()
        self.model.embed_vision.embedding_pre_projection_norm = Norm(c.vision_config.hidden_size, c.vision_config.rms_norm_eps, False)
        self.model.embed_vision.embedding_projection = Linear(c.vision_config.hidden_size, c.text_config.hidden_size, bias=False)
        self.lm_head = Linear(c.text_config.hidden_size, c.text_config.vocab_size, bias=False)
        self.tanh = Tanh()
        self.topk, self.compare = DetectorTopK(), CodecTop1()

    def forward(self, input_ids, pixel_values=None, image_position_ids=None, attention_mask=None, cache=None):
        c = self.c.text_config
        start = cache.seen if isinstance(cache, StaticKV) else (0 if not cache else cache[0][0].shape[1])
        positions = torch.arange(start, start+input_ids.shape[1], device=input_ids.device)[None].expand(input_ids.shape[0], -1)
        image_mask = input_ids == self.c.image_token_id
        ids = input_ids.masked_fill(image_mask, c.pad_token_id)
        x = self.model.language_model.embed_tokens(ids)
        images = None
        if pixel_values is not None:
            images = self.model.vision_tower(pixel_values, image_position_ids)
            images = self.model.embed_vision.embedding_projection(self.model.embed_vision.embedding_pre_projection_norm(images))
            x = x.masked_scatter(image_mask[..., None].expand_as(x), images)
        masks = {}
        for kind in ('full_attention', 'sliding_attention'):
            length, offset = cache.sizes(kind, input_ids.shape[1]) if isinstance(cache, StaticKV) else (start+input_ids.shape[1], 0)
            kv_positions = torch.arange(offset, offset+length, device=x.device)
            mask = (kv_positions[None, :] <= positions[0, :, None])[None, None].expand(x.shape[0], -1, -1, -1)
            if kind == 'sliding_attention':
                mask = mask & (kv_positions[None, :] > positions[0, :, None]-c.sliding_window)[None, None]
            if attention_mask is not None:
                valid = torch.zeros(x.shape[0], offset+length, device=x.device, dtype=torch.bool)
                available = min(valid.shape[1], attention_mask.shape[1])
                valid[:, :available] = attention_mask[:, :available].bool()
                mask = mask & valid[:, None, None, offset:]
            masks[kind] = mask
        hidden, shared = self.model.language_model(x, ids, positions, masks, cache)
        if isinstance(cache, StaticKV):
            cache.seen += input_ids.shape[1]
        logits = self.lm_head(hidden)
        if c.final_logit_softcapping is not None:
            logits = self.tanh(logits/c.final_logit_softcapping)*c.final_logit_softcapping
        return {'logits': logits, 'image_hidden_states': images, 'shared_kv_states': shared}

    def generate(self, inputs, steps, generation):
        history = inputs['input_ids'].clone()
        mask = inputs['attention_mask'].clone()
        cache = StaticKV(history.shape[1]+steps-1, self.c.text_config.sliding_window)
        outputs = {}
        stopped = torch.zeros(history.shape[0], dtype=torch.bool, device=history.device)
        current = dict(inputs)
        for step in range(steps):
            result = self.forward(**current, cache=cache)
            logits = result['logits'][:, -1].float().clone()
            outputs[f'logits.{step}'] = logits
            filtered = logits / generation['temperature']
            top, _ = self.topk(filtered, min(generation['top_k'], filtered.shape[-1]))
            threshold = top[:, -1:].expand_as(filtered)
            remove = self.compare(torch.stack((filtered, threshold), -1)).bool()
            filtered = filtered.masked_fill(remove, -float('inf'))
            token = sample_nucleus(filtered, generation['top_p']).squeeze(-1)
            token = torch.where(stopped, generation['pad_token_id'], token)
            history = torch.cat((history, token[:, None]), -1)
            eos = torch.tensor(generation['eos_token_id'], device=history.device)
            stopped |= torch.isin(token, eos)
            if stopped.all():
                break
            mask = torch.cat((mask, torch.ones_like(token[:, None])), -1)
            current = {'input_ids': token[:, None], 'attention_mask': mask}
        outputs['sequences'] = history
        for index, values in cache.layers.items():
            for name, tensor in zip(('key', 'value'), values):
                outputs[f'past_key_values.{index}.{name}'] = tensor.transpose(1, 2)
        return outputs


def build_from_config(config, device, dtype):
    return ImageTextModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    # Audio is not invoked by the documented image-text task. Retain and report
    # its native state separately; do not claim audio execution coverage.
    expected = model.state_dict()
    mapped, consumed = {}, set()
    for key in expected:
        source = key.replace('.embed_tokens.emb.', '.embed_tokens.').replace('.embed_tokens_per_layer.emb.', '.embed_tokens_per_layer.')
        if source not in state_dict:
            raise KeyError(source)
        mapped[key] = state_dict[source]
        consumed.add(source)
    unused = set(state_dict)-consumed
    if any(not key.startswith(('model.audio_tower.', 'model.embed_audio.')) for key in unused):
        raise ValueError(f'Unmapped active weights: {sorted(unused)}')
    model.load_state_dict(mapped, strict=True, assign=True)
    model.unexecuted_audio_state = {key: state_dict[key] for key in unused}
    model.mapping_counts = {'mapped': len(consumed), 'inactive_audio': len(unused)}


def make_workloads(model, inputs, config, case):
    return {'generate': Workload(run=lambda: model.generate(inputs, case['generation_kwargs']['max_new_tokens'],
                                                            case['reference']['generation_config']))}
