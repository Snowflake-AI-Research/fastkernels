"""Gemma3 image/text inference composed from existing vision and decoder ops."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionEncoder
from ..runner import Workload
from .siglip import configure_encoder


class NativeNorm(GemmaRMSNorm):
    """Select the unchanged native callable and its explicit HF cast order."""

    forward = GemmaRMSNorm.forward_native


class NativeGeluAndMul(GeluAndMul):
    def __init__(self, approximate='tanh'):
        nn.Module.__init__(self)
        self.approximate = approximate

    forward = GeluAndMul.forward_native


class PositionRotary(nn.Module):
    """Position-only angles followed by the unchanged native rotary operation."""

    def __init__(self, config, kind):
        super().__init__()
        self.dim = config.head_dim
        parameters = config.rope_parameters[kind]
        if parameters['rope_type'] not in ('default', 'linear'):
            raise ValueError('Gemma3 coverage requires default or linear rotary frequencies')
        # HF initializes these FP32 constants on CPU before loading weights.
        # Keep them outside dtype-converted model buffers; their device copy
        # and position-dependent evaluation remain in the measured forward.
        dims = torch.arange(0, self.dim, 2, device='cpu', dtype=torch.float32)
        self.inverse = 1.0 / (parameters['rope_theta'] ** (dims / self.dim))
        self.inverse = self.inverse / parameters.get('factor', 1.0)

    def forward(self, query, key, positions):
        inverse = self.inverse.to(positions.device)
        angles = positions.float().reshape(-1, 1) * inverse[None]
        table = torch.cat((angles.cos(), angles.sin()), -1).to(query.dtype)
        indices = torch.arange(positions.numel(), device=positions.device)
        q, k = RotaryEmbedding.forward_native(
            indices, query.reshape(positions.numel(), -1),
            key.reshape(positions.numel(), -1), self.dim, table,
        )
        return q.reshape_as(query), k.reshape_as(key)


class Attention(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.heads, self.kv_heads, self.dim = (
            config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        self.window = config.sliding_window if kind == 'sliding_attention' else None
        self.scale = config.query_pre_attn_scalar ** -0.5
        self.q_proj = Linear(config.hidden_size, self.heads*self.dim, config.attention_bias)
        self.k_proj = Linear(config.hidden_size, self.kv_heads*self.dim, config.attention_bias)
        self.v_proj = Linear(config.hidden_size, self.kv_heads*self.dim, config.attention_bias)
        self.o_proj = Linear(self.heads*self.dim, config.hidden_size, config.attention_bias)
        self.q_norm = NativeNorm(self.dim, config.rms_norm_eps)
        self.k_norm = NativeNorm(self.dim, config.rms_norm_eps)
        self.rotary = PositionRotary(config, kind)
        self.core = DenseAttention(backend='cudnn')

    def forward(self, hidden, positions, mask, previous):
        shape = (*hidden.shape[:2], -1, self.dim)
        query = self.q_norm(self.q_proj(hidden).reshape(shape))
        key = self.k_norm(self.k_proj(hidden).reshape(shape))
        value = self.v_proj(hidden).reshape(shape)
        query, key = self.rotary(query, key, positions)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        if previous is not None:
            key, value = (torch.cat((old, new), dim=2)
                          for old, new in zip(previous, (key, value)))
        # Native DynamicSlidingWindowLayer returns full states to attention,
        # but retains only window-1 entries for the next token.
        retained = ((key[:, :, -self.window+1:], value[:, :, -self.window+1:])
                    if self.window is not None else (key, value))
        groups = self.heads // self.kv_heads
        key, value = (x.transpose(1, 2).repeat_interleave(groups, dim=2) for x in (key, value))
        output = self.core(query, key, value, softmax_scale=self.scale, attn_mask=mask)
        return self.o_proj(output.reshape(*hidden.shape[:2], -1)), retained


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = NativeGeluAndMul(approximate='tanh')

    def forward(self, hidden):
        packed = torch.cat((self.gate_proj(hidden), self.up_proj(hidden)), dim=-1)
        return self.down_proj(self.act(packed))


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.self_attn, self.mlp = Attention(config, kind), MLP(config)
        for name in ('input_layernorm', 'post_attention_layernorm',
                     'pre_feedforward_layernorm', 'post_feedforward_layernorm'):
            setattr(self, name, NativeNorm(config.hidden_size, config.rms_norm_eps))

    def forward(self, hidden, positions, mask, previous):
        attention, cache = self.self_attn(self.input_layernorm(hidden), positions, mask, previous)
        hidden = hidden + self.post_attention_layernorm(attention)
        hidden = hidden + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        return hidden, cache


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision = config.vision_config
        self.side = vision.image_size // vision.patch_size
        kernel = self.side // int(config.mm_tokens_per_image**0.5)
        self.pool = AvgPool2d(kernel)
        self.mm_soft_emb_norm = NativeNorm(vision.hidden_size, vision.layer_norm_eps)
        self.mm_input_projection_weight = nn.Parameter(torch.empty(vision.hidden_size, config.text_config.hidden_size))
        self.matmul = BMM()

    def forward(self, hidden):
        spatial = hidden.transpose(1, 2).reshape(hidden.shape[0], -1, self.side, self.side).contiguous()
        hidden = self.pool(spatial).flatten(2).transpose(1, 2)
        return self.matmul(self.mm_soft_emb_norm(hidden), self.mm_input_projection_weight)


class Gemma3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        text = config.text_config
        self.model = nn.Module()
        self.model.vision_tower = SigLIPVisionEncoder(config.vision_config)
        configure_encoder(self.model.vision_tower.layers)
        self.model.vision_tower.post_layernorm.promote_fp32 = False
        self.model.multi_modal_projector = Projector(config)
        language = self.model.language_model = nn.Module()
        language.embed_tokens = Embedding(text.vocab_size, text.hidden_size, text.pad_token_id)
        language.layers = nn.ModuleList(Layer(text, kind) for kind in text.layer_types)
        language.norm = NativeNorm(text.hidden_size, text.rms_norm_eps)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        self.lm_head.weight = language.embed_tokens.emb.weight

    def forward(self, input_ids, pixel_values=None, token_type_ids=None,
                attention_mask=None, cache=None):
        config, text = self.config, self.config.text_config
        seen = 0 if cache is None else cache['seen']
        positions = torch.arange(seen, seen+input_ids.shape[1], device=input_ids.device)
        positions = positions[None].expand(input_ids.shape[0], -1)
        image_mask = input_ids == config.image_token_index
        safe_ids = input_ids.masked_fill(image_mask, 0) if config.image_token_index >= text.vocab_size else input_ids
        hidden = self.model.language_model.embed_tokens(safe_ids)
        hidden = hidden * torch.tensor(text.hidden_size**0.5, device=hidden.device, dtype=hidden.dtype)
        image_hidden = None
        if pixel_values is not None:
            image_hidden = self.model.multi_modal_projector(self.model.vision_tower(pixel_values))
            hidden = hidden.masked_scatter(image_mask[..., None].expand_as(hidden), image_hidden)

        image_blocks = None
        if token_type_ids is not None:
            images = token_type_ids == 1
            starts = images & ~torch.cat((torch.zeros_like(images[:, :1]), images[:, :-1]), dim=1)
            image_blocks = (starts.int().cumsum(1)-1).masked_fill(~images, -1)
        masks = {}
        for kind in set(text.layer_types):
            offset = max(seen-text.sliding_window+1, 0) if kind == 'sliding_attention' else 0
            keys = torch.arange(offset, seen+input_ids.shape[1], device=input_ids.device)
            mask = keys[None, :] <= positions[0, :, None]
            if kind == 'sliding_attention':
                mask = mask & (positions[0, :, None]-keys[None, :] < text.sliding_window)
            mask = mask[None, None].expand(input_ids.shape[0], 1, -1, -1)
            if image_blocks is not None:
                if seen:
                    raise ValueError('Gemma3 continuation expects text-only tokens without token_type_ids')
                same_image = (image_blocks[:, :, None] == image_blocks[:, None, :]) & (image_blocks[:, :, None] >= 0)
                mask = mask | same_image[:, None]
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, None, offset:].bool()
            masks[kind] = mask
        states = []
        for index, layer in enumerate(self.model.language_model.layers):
            previous = None if cache is None else cache['layers'][index]
            hidden, state = layer(hidden, positions, masks[text.layer_types[index]], previous)
            states.append(state)
        logits = self.lm_head(self.model.language_model.norm(hidden))
        return {'logits': logits, 'image_hidden_states': image_hidden,
                'cache': {'seen': seen+input_ids.shape[1], 'layers': states}}


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if (text.hidden_activation != 'gelu_pytorch_tanh' or text.attn_logit_softcapping is not None
            or text.use_bidirectional_attention or not config.tie_word_embeddings
            or vision.vision_use_head or vision.num_channels != 3):
        raise ValueError('Gemma3 coverage preserves the selected 4B conversion recipe')
    return Gemma3(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    expected = model.state_dict()
    for name in expected:
        source = name.replace('.embed_tokens.emb.', '.embed_tokens.')
        prefix = 'model.vision_tower.'
        if name.startswith(prefix):
            suffix = name[len(prefix):].replace('layers.', 'encoder.layers.')
            if suffix.startswith('patch_embedding.'):
                suffix = 'embeddings.' + suffix
            elif suffix == 'position_embedding':
                suffix = 'embeddings.position_embedding.weight'
            source = prefix + suffix
        mapped[name] = state_dict[source].reshape(expected[name].shape)
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f'Unmapped Gemma3 weights: {sorted(set(state_dict)-used)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, case=None):
    ids = inputs['input_ids']
    prefix = ids.shape[1]-2
    state = {}

    def initial():
        kwargs = dict(inputs, input_ids=ids[:, :prefix])
        for name in ('attention_mask', 'token_type_ids'):
            if name in kwargs:
                kwargs[name] = kwargs[name][:, :prefix]
        return model(**kwargs)

    def advance(index):
        mask = inputs.get('attention_mask')
        output = model(ids[:, prefix+index:prefix+index+1], cache=state['cache'],
                       attention_mask=None if mask is None else mask[:, :prefix+index+1])
        state['cache'] = output['cache']
        return output

    def prepare(index):
        state['cache'] = initial()['cache']
        for previous in range(index):
            advance(previous)

    def retain(output):
        state['output'] = output
        return {'logits': output['logits']}

    def collect(_):
        output = state.pop('output')
        result = {'logits': output['logits']}
        if output['image_hidden_states'] is not None:
            result['image_hidden_states'] = output['image_hidden_states']
        for index, (key, value) in enumerate(output['cache']['layers']):
            result[f'past_key_values.{index}.key'] = key
            result[f'past_key_values.{index}.value'] = value
        return result

    return {'prefill': Workload(run=lambda: retain(initial()), collect=collect),
            **{f'decode_{index+1}': Workload(run=lambda index=index: retain(advance(index)),
                                           prepare=lambda index=index: prepare(index), collect=collect)
               for index in range(2)}}
