"""PaliGemma's coherent bare-constructor Gemma-v1 image/text computation.

This is a separately declared constructor workload, not the gated PaliGemma2
checkpoint or the inconsistent stock-component example in the config docstring.
"""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.gemma_dense_attention import GemmaRotaryEmbedding
from fastkernels.tasks.baseline.L3.gemma_dense_decoder_layer import GemmaDenseDecoderLayer

from .modernvbert import VisionTower, base_source
from ..runner import Workload


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        text = config.text_config
        self.image_token_id = config.image_token_index
        self.vocab_size = text.vocab_size
        self.vision = VisionTower(config.vision_config)
        for layer in self.vision.encoder.layers:
            layer.self_attn.attn = DenseAttention(backend='sdpa')
        self.image_projection = Linear(config.vision_config.hidden_size, config.projection_dim)
        self.embeddings = Embedding(text.vocab_size, text.hidden_size, padding_idx=text.pad_token_id)
        self.register_buffer('embed_scale', torch.tensor(text.hidden_size**0.5, dtype=torch.float32), persistent=False)
        self.layers = nn.ModuleList([GemmaDenseDecoderLayer(text) for _ in range(text.num_hidden_layers)])
        for layer in self.layers:
            layer.input_layernorm = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
            layer.post_attention_layernorm = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
            layer.mlp.act_fn = GeluAndMul(approximate='tanh')
            layer.self_attn.attn = DenseAttention(backend='sdpa')
        self.norm = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
        with torch.device('meta'):
            self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.emb.weight
        self.rotary = GemmaRotaryEmbedding(text.head_dim, text.max_position_embeddings,
                                           text.rope_parameters['rope_theta'])

    def forward(self, input_ids, pixel_values=None, attention_mask=None, token_type_ids=None, past_key_values=None):
        if past_key_values is None and token_type_ids is None:
            raise ValueError('PaliGemma workload requires processor prefix token_type_ids on prefill')
        image_tokens = input_ids == self.image_token_id
        ids = input_ids.masked_fill(image_tokens, 0) if self.image_token_id >= self.vocab_size else input_ids
        hidden = self.embeddings(ids) * self.embed_scale.to(self.embeddings.emb.weight.dtype)
        outputs = {}
        if pixel_values is not None:
            image = self.image_projection(self.vision(pixel_values))
            if hidden[image_tokens].numel() != image.numel():
                raise ValueError('Image placeholder count must match the projected SigLIP patch count')
            hidden = hidden.masked_scatter(image_tokens[..., None].expand_as(hidden), image)
            outputs['image_hidden_states'] = image
        previous_length = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None] + previous_length + 1
        cos, sin = self.rotary(hidden, positions)
        key_positions = torch.arange(previous_length + input_ids.shape[1], device=input_ids.device) + 1
        mask = key_positions[None, None, :] <= positions[:, :, None]
        if token_type_ids is not None and (past_key_values is None or pixel_values is not None):
            prefix = token_type_ids == 0
            mask = mask | (prefix[:, :, None] & prefix[:, None, :])
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, :].bool()
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, cos, sin, attention_mask=mask[:, None],
                                  kv_cache=None if past_key_values is None else past_key_values[index])
            states.append(state)
        outputs['logits'] = self.lm_head(self.norm(hidden))
        outputs['past_key_values'] = tuple(states)
        return outputs


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if (text.model_type != 'gemma' or text.hidden_act != 'gelu_pytorch_tanh' or text.attention_bias
            or text.rope_parameters['rope_type'] != 'default' or not text.use_cache
            or not config.tie_word_embeddings or vision.vision_use_head
            or vision.hidden_act != 'gelu_pytorch_tanh' or config.projection_dim != text.hidden_size):
        raise ValueError('PaliGemma constructor case requires coherent Gemma-v1 projection, tied head and unpooled SigLIP')
    model = PaliGemmaForConditionalGeneration(config).to(device=device, dtype=dtype).eval()
    # Native metadata frequencies remain FP32 even when model weights are BF16.
    with torch.device('cpu'):
        model.rotary = GemmaRotaryEmbedding(text.head_dim, text.max_position_embeddings,
                                            text.rope_parameters['rope_theta']).to(device=device)
    return model


def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict['lm_head.weight'], state_dict['model.language_model.embed_tokens.weight']):
        raise ValueError('PaliGemma tied input and output embedding weights disagree')
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        if name.startswith('vision.'):
            source = 'model.vision_tower.' + base_source(name.replace('vision.', 'vision_model.', 1)).removeprefix('vision_model.')
        elif name.startswith('image_projection.'):
            source = name.replace('image_projection.', 'model.multi_modal_projector.linear.', 1)
        elif name.startswith('embeddings.emb.'):
            source = name.replace('embeddings.emb.', 'model.language_model.embed_tokens.', 1)
        elif name.startswith(('layers.', 'norm.')):
            source = 'model.language_model.' + name
        else:
            source = name
        if '.gate_up_proj.' in source:
            sources = [source.replace('.gate_up_proj.', '.' + part + '.') for part in ('gate_proj', 'up_proj')]
            mapped[name] = torch.cat([state_dict[item] for item in sources])
            used.update(sources)
        else:
            mapped[name] = state_dict[source].reshape(target.shape)
            used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped PaliGemma weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def flatten(output):
    result = {name: value for name, value in output.items() if name != 'past_key_values'}
    for index, (key, value) in enumerate(output['past_key_values']):
        result[f'past_key_values.{index}.key'] = key.transpose(1, 2)
        result[f'past_key_values.{index}.value'] = value.transpose(1, 2)
    return result


def make_workloads(model, inputs, config, case=None):
    if case is None or case.get('workload') != 'causal_lm_continuation':
        return {'forward': Workload(run=lambda: flatten(model(**inputs)))}
    if set(inputs) != {'input_ids', 'pixel_values', 'token_type_ids'}:
        raise ValueError('PaliGemma continuation selects unpadded image+token inputs with prefix token types')
    ids = inputs['input_ids']
    prefix = ids.shape[1] - 2
    if prefix < 1:
        raise ValueError('Continuation requires a prefix and two supplied tokens')
    state = {}

    def initial():
        return model(ids[:, :prefix], pixel_values=inputs['pixel_values'],
                     token_type_ids=inputs['token_type_ids'][:, :prefix])

    def advance(index, previous):
        return model(ids[:, prefix + index:prefix + index + 1], past_key_values=previous)

    def prepare(index):
        previous = initial()['past_key_values']
        for step in range(index):
            previous = advance(step, previous)['past_key_values']
        state['previous'] = previous

    return {
        'prefill': Workload(run=lambda: flatten(initial())),
        'decode_1': Workload(run=lambda: flatten(advance(0, state['previous'])), prepare=lambda: prepare(0)),
        'decode_2': Workload(run=lambda: flatten(advance(1, state['previous'])), prepare=lambda: prepare(1)),
    }
