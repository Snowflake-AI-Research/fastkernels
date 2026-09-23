"""PI0 action sampling through the existing vision, Gemma and flow pipeline."""

from dataclasses import fields

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu_and_mul import GeluAndMul
from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sinusoidal_embed import SinusoidalEmbed
from fastkernels.tasks.baseline.L2.gemma_dense_attention import GemmaRotaryEmbedding
from fastkernels.tasks.baseline.L2.pi0_action_embed import Pi0ActionTimeEmbedding
from fastkernels.tasks.baseline.L4.pi0 import (
    GemmaConfig, Pi0Config, Pi0Model, Pi0Pipeline, Pi0SamplingParams, SigLIPVisionConfig,
)


class Model(Pi0Model):
    def embed_prefix(self, input_ids, pixel_values, pixel_attention_mask):
        features = self.encode_images(pixel_values.flatten(0, 1))
        features = features.reshape(*pixel_attention_mask.shape, *features.shape[1:])
        features = features[pixel_attention_mask]
        image_mask = input_ids == self.config.image_token_id
        ids = input_ids.masked_fill(image_mask, 0)
        embeddings = self.vlm.embed_tokens(ids)
        scale = embeddings.new_tensor(self.config.vlm_text_config.hidden_size ** 0.5)
        embeddings = embeddings * scale
        return embeddings.masked_scatter(image_mask[..., None].expand_as(embeddings), features)


class Pipeline(Pi0Pipeline):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config, self.model = config, Model(config)
        self.embed_action_time = Pi0ActionTimeEmbedding(
            config.dit_config.hidden_size, config.max_action_dim, config.max_state_dim,
            config.min_period, config.max_period,
        )
        self.action_out_proj = Linear(config.dit_config.hidden_size, config.max_action_dim)


def build_from_config(config, device, dtype):
    def gemma(source):
        values = {field.name: getattr(source, field.name) for field in fields(GemmaConfig)
                  if field.name != 'rope_theta'}
        if source.rope_parameters['rope_type'] != 'default' or source.hidden_act != 'gelu_pytorch_tanh':
            raise ValueError('PI0 case requires default Gemma rotary positions and tanh GELU')
        return GemmaConfig(**values, rope_theta=source.rope_parameters['rope_theta'])

    vision = config.vlm_config.vision_config
    fk = Pi0Config(
        vlm_text_config=gemma(config.vlm_config.text_config), dit_config=gemma(config.dit_config),
        vlm_vision_config=SigLIPVisionConfig(**{field.name: getattr(vision, field.name)
                                             for field in fields(SigLIPVisionConfig)}),
        projection_dim=config.vlm_config.projection_dim,
        image_token_id=config.vlm_config.image_token_index,
        **{name: getattr(config, name) for name in ('chunk_size', 'max_state_dim', 'max_action_dim',
                                                   'num_inference_steps', 'min_period', 'max_period')},
    )
    model = Pipeline(fk)
    for tower, settings in ((model.model.vlm, fk.vlm_text_config), (model.model.dit, fk.dit_config)):
        tower.norm = GemmaRMSNorm(settings.hidden_size, settings.rms_norm_eps)
        for layer in tower.layers:
            layer.input_layernorm = GemmaRMSNorm(settings.hidden_size, settings.rms_norm_eps)
            layer.post_attention_layernorm = GemmaRMSNorm(settings.hidden_size, settings.rms_norm_eps)
            layer.mlp.act_fn = GeluAndMul(approximate='tanh')
            layer.self_attn.attn = DenseAttention(backend='sdpa')
    for layer in model.model.vision_tower.layers:
        layer.self_attn.attn = DenseAttention(backend='sdpa')
    model.to(device=device, dtype=dtype).eval()
    # HF initializes these fixed frequencies on CPU and retains FP32 metadata.
    with torch.device('cpu'):
        for tower, settings in ((model.model.vlm, fk.vlm_text_config), (model.model.dit, fk.dit_config)):
            tower.rotary_emb = GemmaRotaryEmbedding(settings.head_dim, settings.max_position_embeddings,
                                                    settings.rope_theta).to(device=device)
        model.embed_action_time.sinusoid_embeds = SinusoidalEmbed(
            fk.dit_config.hidden_size, fk.min_period, fk.max_period,
        ).to(device=device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        source = name
        if source.startswith('model.vision_tower.'):
            source = source.replace('model.vision_tower.', 'model.vlm.vision_tower.', 1)
            source = source.replace('.layers.', '.encoder.layers.')
            source = source.replace('.patch_embedding.', '.embeddings.patch_embedding.')
            if source.endswith('.position_embedding'):
                source = source.removesuffix('.position_embedding') + '.embeddings.position_embedding.weight'
        elif source.startswith('model.multi_modal_projector.'):
            source = source.replace('model.multi_modal_projector.', 'model.vlm.multi_modal_projector.linear.', 1)
        elif source.startswith('model.vlm.'):
            source = source.replace('model.vlm.', 'model.vlm.language_model.', 1).replace('.embed_tokens.emb.', '.embed_tokens.')
        if '.gate_up_proj.' in source:
            names = [source.replace('.gate_up_proj.', f'.{part}.') for part in ('gate_proj', 'up_proj')]
            mapped[name] = torch.cat([state_dict[key] for key in names])
            used.update(names)
        else:
            mapped[name] = state_dict[source].reshape(target.shape)
            used.add(source)
    # The action expert always receives inputs_embeds; its token table is never called.
    if set(state_dict) - used != {'model.dit.embed_tokens.weight'}:
        raise KeyError(f'Unexpected PI0 unmapped weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, workload, case=None):
    params = Pi0SamplingParams(num_inference_steps=model.config.num_inference_steps)
    return {'sample_actions': Workload(run=lambda: {'actions': model(**inputs, params=params).actions})}
