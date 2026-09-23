"""Mistral3's Pixtral image features, spatial merger and dense language model."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from ..runner import Workload
from . import llama
from .olmo2 import decoder_config
from .pixtral import PixtralVisionModel
from .lighton_ocr import PixtralSDPA
from .qwen2_precision import NativeRotaryEmbedding


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, text_width = config.vision_config.hidden_size, config.text_config.hidden_size
        self.merge_size = config.spatial_merge_size
        self.norm = RMSNorm(width, config.text_config.rms_norm_eps)
        self.patch_merger = nn.Module()
        self.patch_merger.merging_layer = Linear(width * self.merge_size**2, width, bias=False)
        self.linear_1 = Linear(width, text_width, bias=config.multimodal_projector_bias)
        self.linear_2 = Linear(text_width, text_width, bias=config.multimodal_projector_bias)
        self.activation = GELU()

    def forward(self, hidden, batch, height, width):
        hidden = self.norm(hidden)
        channels, merge = hidden.shape[-1], self.merge_size
        # Non-overlapping unfold: channel, local row, local column within each patch.
        hidden = hidden.reshape(batch, height // merge, merge, width // merge, merge, channels)
        hidden = hidden.permute(0, 1, 3, 5, 2, 4).reshape(-1, channels * merge**2)
        hidden = self.patch_merger.merging_layer(hidden)
        return self.linear_2(self.activation(self.linear_1(hidden)))


class Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text = text
        self.vision_tower = PixtralVisionModel(config.vision_config)
        for layer in self.vision_tower.transformer.layers:
            layer.attention = PixtralSDPA(config.vision_config, backend="cudnn")
        self.multi_modal_projector = Projector(config)
        self.image_token_id = config.image_token_index
        self.patch_size = config.vision_config.patch_size
        self.pixel_values = None
        self.image_hidden_states = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        embeddings = self.text.embed_tokens(input_ids)
        if get_context().is_prefill:
            pixels = self.pixel_values
            hidden = self.vision_tower(pixels)['last_hidden_state']
            self.image_hidden_states = self.multi_modal_projector(
                hidden, pixels.shape[0], pixels.shape[-2] // self.patch_size,
                pixels.shape[-1] // self.patch_size,
            )
            mask = (input_ids == self.image_token_id)[:, None].expand_as(embeddings)
            embeddings = embeddings.masked_scatter(mask, self.image_hidden_states)
        return self.text(input_ids, positions, inputs_embeds=embeddings)


def build_from_config(config, device, dtype):
    if (config.vision_feature_layer != -1 or config.projector_hidden_act != 'gelu'
            or config.text_config.sliding_window is not None
            or config.text_config.rope_parameters['rope_type'] != 'default'
            or config.vision_config.hidden_act != 'silu'):
        raise ValueError('The selected Mistral3 checkpoint uses final Pixtral features and non-windowed Mistral text attention')
    model = LlamaForCausalLM(decoder_config(config.text_config, dtype))
    text = config.text_config
    model.model.rotary_emb = NativeRotaryEmbedding(
        text.head_dim, text.max_position_embeddings, text.rope_parameters['rope_theta'],
    )
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = model.model.rotary_emb
    if config.tie_word_embeddings:
        model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    model.model = Backbone(model.model, config)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace('model.language_model.', 'model.'): remaining.pop(name)
            for name in list(remaining) if name.startswith('model.language_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    if config.tie_word_embeddings and not torch.equal(text['lm_head.weight'], text['model.embed_tokens.weight']):
        raise ValueError('Mistral3 tied embedding and output weights differ')
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    for name in ('vision_tower', 'multi_modal_projector'):
        module = getattr(model.model, name)
        weights = {key: remaining.pop('model.' + name + '.' + key) for key in module.state_dict()}
        module.load_state_dict(weights, strict=True)
    if remaining:
        raise KeyError(f'Unmapped Mistral3 weights: {sorted(remaining)}')


def make_workloads(model, inputs, config):
    pixels = inputs['pixel_values']
    sizes = inputs['image_sizes']
    if any(tuple(size) != tuple(pixels.shape[-2:]) for size in sizes.tolist()):
        raise ValueError('This workload uses equally sized, unpadded images')
    model.model.pixel_values = pixels
    workloads = llama.make_workloads(model, {'input_ids': inputs['input_ids']}, model.config)
    batch, total_length = inputs['input_ids'].shape

    def run_phase(work, length, include_image):
        outputs = work.run()
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn.attn
            for name, cache in (('key', attention.k_cache), ('value', attention.v_cache)):
                if attention.kv_layout == 'HND':
                    cache = cache.transpose(1, 2)
                cache = cache.reshape(batch, -1, attention.num_kv_heads, attention.head_size)
                outputs[f'past_key_values.{index}.{name}'] = cache[:, :length].transpose(1, 2)
        if include_image:
            outputs['image_hidden_states'] = model.model.image_hidden_states
        return outputs

    for name, length in (('prefill', total_length - 1), ('decode', total_length)):
        work = workloads[name]
        workloads[name] = Workload(
            run=lambda work=work, length=length, image=name == 'prefill': run_phase(work, length, image),
            prepare=work.prepare,
        )
    return workloads
