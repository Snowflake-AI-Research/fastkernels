"""ColPali document embeddings with SigLIP, Gemma and retrieval normalization."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.gemma_dense_attention import GemmaRotaryEmbedding
from fastkernels.tasks.baseline.L3.gemma_dense_decoder_layer import GemmaDenseDecoderLayer
from .modernvbert import VisionTower, base_source, make_workloads


class ColPaliForRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        vlm, text = config.vlm_config, config.vlm_config.text_config
        self.image_token_id = vlm.image_token_index
        self.vocab_size = text.vocab_size
        self.vision = VisionTower(vlm.vision_config)
        for layer in self.vision.encoder.layers:
            layer.self_attn.attn = DenseAttention(backend='sdpa')
        self.image_projection = Linear(vlm.vision_config.hidden_size, vlm.projection_dim)
        self.embeddings = Embedding(text.vocab_size, text.hidden_size, padding_idx=text.pad_token_id)
        self.register_buffer('embed_scale', torch.tensor(text.hidden_size**0.5, dtype=torch.float32), persistent=False)
        self.layers = nn.ModuleList([GemmaDenseDecoderLayer(text) for _ in range(text.num_hidden_layers)])
        for layer in self.layers:
            layer.input_layernorm = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
            layer.post_attention_layernorm = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
            layer.mlp.act_fn = GeluAndMul(approximate='tanh')
            layer.self_attn.attn = DenseAttention(backend='sdpa')
        self.norm = GemmaRMSNorm(text.hidden_size, text.rms_norm_eps)
        self.embedding_proj_layer = Linear(text.hidden_size, config.embedding_dim)
        self.normalize = L2Norm(dim=-1, eps=0)
        self.rotary = GemmaRotaryEmbedding(text.head_dim, text.max_position_embeddings, text.rope_parameters['rope_theta'])
        self.use_cache = text.use_cache

    def forward(self, input_ids, pixel_values=None, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            raise ValueError('ColPali retrieval inputs require processor token_type_ids')
        image_tokens = input_ids == self.image_token_id
        ids = input_ids.masked_fill(image_tokens, 0) if self.image_token_id >= self.vocab_size else input_ids
        hidden = self.embeddings(ids) * self.embed_scale.to(self.embeddings.emb.weight.dtype)
        outputs = {}
        if pixel_values is not None:
            image = self.image_projection(self.vision(pixel_values))
            hidden[image_tokens] = image.reshape(-1, image.shape[-1])
            outputs['image_hidden_states'] = image
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None] + 1
        cos, sin = self.rotary(hidden, positions)
        causal = positions[:, None, :] <= positions[:, :, None]
        prefix = token_type_ids == 0
        mask = causal | (prefix[:, :, None] & prefix[:, None, :])
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, :].bool()
        for index, layer in enumerate(self.layers):
            hidden, (key, value) = layer(hidden, cos, sin, attention_mask=mask[:, None])
            if self.use_cache:
                outputs[f'past_key_values.{index}.key'] = key.transpose(1, 2)
                outputs[f'past_key_values.{index}.value'] = value.transpose(1, 2)
        embeddings = self.normalize(self.embedding_proj_layer(self.norm(hidden)))
        if attention_mask is not None:
            embeddings = embeddings.masked_fill(~attention_mask[:, :, None].bool(), 0)
        outputs['embeddings'] = embeddings
        return outputs


def build_from_config(config, device, dtype):
    text, vision = config.vlm_config.text_config, config.vlm_config.vision_config
    if (text.hidden_act != 'gelu_pytorch_tanh' or text.attention_bias
            or text.rope_parameters['rope_type'] != 'default' or vision.vision_use_head):
        raise ValueError('ColPali case preserves Gemma v1 and the documented unpooled SigLIP tower')
    model = ColPaliForRetrieval(config).to(device=device, dtype=dtype).eval()
    model.rotary = GemmaRotaryEmbedding(text.head_dim, text.max_position_embeddings,
                                        text.rope_parameters['rope_theta']).to(device=device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name in model.state_dict():
        if name.startswith('vision.'):
            source = 'vlm.vision_tower.' + base_source(name.replace('vision.', 'vision_model.', 1)).removeprefix('vision_model.')
        elif name.startswith('image_projection.'):
            source = name.replace('image_projection.', 'vlm.multi_modal_projector.linear.', 1)
        elif name.startswith('embeddings.emb.'):
            source = name.replace('embeddings.emb.', 'vlm.language_model.embed_tokens.', 1)
        elif name.startswith(('layers.', 'norm.')):
            source = 'vlm.language_model.' + name
        else:
            source = name
        if '.gate_up_proj.' in source:
            sources = [source.replace('.gate_up_proj.', '.' + part + '.') for part in ('gate_proj', 'up_proj')]
            mapped[name] = torch.cat([state_dict[item] for item in sources])
            used.update(sources)
        else:
            mapped[name] = state_dict[source].reshape(model.state_dict()[name].shape)
            used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped ColPali weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
