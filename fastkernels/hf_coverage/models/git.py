"""GIT captioning with a CLIP vision tower and image-prefix BERT decoder."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer
from .clip import ClipVisionModel
from .mvp import EagerAttention
from ..runner import Workload


class VisionSoftmax(Softmax):
    def forward(self, scores):
        return super().forward(scores.float()).to(scores.dtype)


class GitAttention(EagerAttention):
    """Reuse BMM/softmax while retaining HF's intermediate score rounding."""

    def __init__(self, *, vision=False):
        super().__init__(prescale_query=False, divide_scores=not vision)
        if vision:
            self.softmax = VisionSoftmax(dim=-1)

    def forward(self, query, key, value, causal=False, attn_mask=None):
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            additive = torch.zeros(attn_mask.shape, dtype=query.dtype, device=query.device)
            additive.masked_fill_(~attn_mask, torch.finfo(query.dtype).min)
            attn_mask = additive
        return super().forward(query, key, value, causal=causal, attn_mask=attn_mask)


class VisionModel(ClipVisionModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.encoder.layers:
            layer.attn = GitAttention(vision=True)

    def forward(self, pixels):
        return self.post_layernorm(self.encoder(self.pre_layrnorm(self.embeddings(pixels))))


class DecoderLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention.self.attn = GitAttention()

    def forward(self, hidden, mask, past_key_value=None):
        a = self.attention.self
        shape = (*hidden.shape[:2], a.num_attention_heads, a.attention_head_size)
        query, key, value = (part.view(shape) for part in a._project_qkv(hidden))
        if past_key_value is not None:
            key = torch.cat((past_key_value[0].transpose(1, 2), key), dim=1)
            value = torch.cat((past_key_value[1].transpose(1, 2), value), dim=1)
        context = a.attn(query, key, value, attn_mask=mask).reshape_as(hidden)
        hidden = self.attention.output(context, hidden)
        hidden = self.output(self.intermediate(hidden), hidden)
        return hidden, key.transpose(1, 2), value.transpose(1, 2)


class GitForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.image_encoder = VisionModel(config.vision_config)
        self.visual_projection = Linear(config.vision_config.hidden_size, config.hidden_size)
        self.visual_norm = LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.output = Linear(config.hidden_size, config.vocab_size)
        self.use_cache = config.use_cache

    def forward(self, input_ids, pixel_values=None, attention_mask=None,
                position_ids=None, past_key_values=None):
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_key_values is not None:
            if pixel_values is not None or input_ids.shape[1] != 1:
                raise ValueError("GIT continuation requires one text token without repeating the image")
            if position_ids is None or attention_mask is None:
                raise ValueError("Pinned GIT continuation requires text positions and the full text mask")
            # Preserve pinned GitModel's image-inclusive offset, including its
            # addition to the text position supplied by generation preparation.
            position_ids = position_ids + past_length
        positions = (torch.arange(past_length, past_length + input_ids.shape[1],
                                  device=input_ids.device)[None]
                     if position_ids is None else position_ids)
        text = self.embedding_norm(self.word_embeddings(input_ids) + self.position_embeddings(positions))
        image_length = 0
        hidden = text
        if pixel_values is not None:
            image = self.visual_norm(self.visual_projection(self.image_encoder(pixel_values)))
            image_length = image.shape[1]
            hidden = torch.cat((image, text), dim=1)
        queries = torch.arange(hidden.shape[1], device=hidden.device) + past_length
        keys = torch.arange(past_length + hidden.shape[1], device=hidden.device)
        mask = (queries[:, None] >= keys[None, :])[None, None]
        if image_length:
            mask[..., :image_length, :image_length] = True
        if attention_mask is not None:
            prefix = (image_length if past_key_values is None
                      else past_length - attention_mask.shape[1] + 1)
            padding = torch.cat((attention_mask.new_ones((input_ids.shape[0], prefix)),
                                 attention_mask), dim=1)
            mask = mask & padding[:, None, None].bool()
        outputs = {}
        for index, layer in enumerate(self.layers):
            hidden, key, value = layer(hidden, mask, None if past_key_values is None else past_key_values[index])
            if self.use_cache:
                outputs[f'past_key_values.{index}.key'] = key
                outputs[f'past_key_values.{index}.value'] = value
        outputs['logits'] = self.output(hidden)
        return outputs


def build_from_config(config, device, dtype):
    if (config.num_image_with_embedding is not None or config.hidden_act != 'gelu'
            or config.vision_config.hidden_act != 'quick_gelu' or config.tie_word_embeddings):
        raise ValueError('GIT case preserves the first image-captioning checkpoint with GELU and CLIP QuickGELU')
    model = GitForCausalLM(config)
    for layer in model.modules():
        if isinstance(layer, LayerNorm):
            layer.promote_fp32 = False
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name in model.state_dict():
        source = name.replace('.emb.weight', '.weight')
        if source.startswith(('word_embeddings.', 'position_embeddings.')):
            source = 'git.embeddings.' + source
        elif source.startswith('embedding_norm.'):
            source = source.replace('embedding_norm.', 'git.embeddings.LayerNorm.', 1)
        elif source.startswith('image_encoder.'):
            source = source.replace('image_encoder.', 'git.image_encoder.vision_model.', 1)
            source = source.replace('patch_embedding.proj.', 'patch_embedding.')
            source = source.replace('.ln_1.', '.layer_norm1.').replace('.ln_2.', '.layer_norm2.')
            source = source.replace('.mlp_fc1.', '.mlp.fc1.').replace('.mlp_fc2.', '.mlp.fc2.')
            for projection in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
                source = source.replace('.' + projection + '.', '.self_attn.' + projection + '.')
        elif source.startswith('visual_projection.'):
            source = source.replace('visual_projection.', 'git.visual_projection.visual_projection.0.', 1)
        elif source.startswith('visual_norm.'):
            source = source.replace('visual_norm.', 'git.visual_projection.visual_projection.1.', 1)
        elif source.startswith('layers.'):
            source = source.replace('layers.', 'git.encoder.layer.', 1)
        if '.self.qkv.' in source:
            sources = [source.replace('.self.qkv.', '.self.' + part + '.') for part in ('query', 'key', 'value')]
            mapped[name] = torch.cat([state_dict[item] for item in sources])
            used.update(sources)
        else:
            mapped[name] = state_dict[source]
            used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped GIT weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    if case is None or case.get('workload') != 'causal_lm_continuation':
        return {'forward': Workload(run=lambda: model(**inputs))}
    ids = inputs['input_ids']
    if not config.use_cache or ids.ndim != 2 or ids.shape[1] < 3:
        raise ValueError('GIT continuation requires caching, a prefix, and two supplied tokens')
    prefix_length = ids.shape[1] - 2
    attention_mask = inputs.get('attention_mask', torch.ones_like(ids))
    state = {}

    def initial():
        return model(ids[:, :prefix_length], pixel_values=inputs['pixel_values'],
                     attention_mask=attention_mask[:, :prefix_length])

    def cache(output):
        return tuple((output[f'past_key_values.{layer}.key'],
                      output[f'past_key_values.{layer}.value'])
                     for layer in range(config.num_hidden_layers))

    def advance(index, previous):
        position = prefix_length + index
        return model(ids[:, position:position + 1], past_key_values=previous,
                     attention_mask=attention_mask[:, :position + 1],
                     position_ids=torch.full((1, 1), position, device=ids.device, dtype=torch.long))

    def prepare(index):
        previous = cache(initial())
        for step in range(index):
            previous = cache(advance(step, previous))
        state['previous'] = previous

    return {
        'prefill': Workload(run=initial),
        'decode_1': Workload(run=lambda: advance(0, state['previous']), prepare=lambda: prepare(0)),
        'decode_2': Workload(run=lambda: advance(1, state['previous']), prepare=lambda: prepare(1)),
    }
