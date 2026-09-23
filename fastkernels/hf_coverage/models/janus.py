"""Janus's default image-conditioned text generation; optional image mode excluded."""

from types import SimpleNamespace
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from ..patches.codec_top1 import CodecTop1
from ..runner import Workload
from . import deepseek_vl
from .smolvlm import TextDecoder, Model as SmolModel


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.use_qk_norm or config.hidden_act != 'gelu' or not config.attention_bias:
            raise ValueError('Selected Janus vision uses biased attention without QK normalization and exact GELU')
        values = dict(config)
        values['intermediate_size'] = int(config.hidden_size * config.mlp_ratio)
        self.core = deepseek_vl.make_vision(SimpleNamespace(**values))

    def forward(self, pixels):
        hidden = self.core(pixels)
        # Native computes its pooler even though the aligner overwrites it.
        self.core.post_layernorm(hidden[:, 0])
        return hidden


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.projection_dim)
        self.hidden_layers = nn.ModuleList(Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.depth))
        self.activation = GELU()

    def forward(self, hidden):
        hidden = self.fc1(hidden)
        for layer in self.hidden_layers:
            hidden = layer(self.activation(hidden))
        return hidden


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model, self.aligner = Vision(config.vision_config), Aligner(config.vision_config)
        self.text_model = TextDecoder(config.text_config)
        self.lm_head = Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.select = CodecTop1()
        self.inactive_state = nn.Module()

    def features(self, pixels, pixel_attention_mask=None):
        if pixel_attention_mask is not None:
            raise ValueError('Native Janus vision does not accept a pixel attention mask')
        return self.aligner(self.vision_model(pixels))

    def generate(self, input_ids, pixel_values, max_new_tokens=4, **kwargs):
        return SmolModel.generate(self, input_ids, pixel_values, max_new_tokens=max_new_tokens,
                                  eos_token_id=self.config.text_config.eos_token_id,
                                  pad_token_id=self.config.text_config.pad_token_id, **kwargs)


def build_from_config(config, device, dtype):
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    # Janus names the ordinary attention output projection differently.
    vision = {name.replace('.self_attn.projection_layer.', '.self_attn.out_proj.'): remaining.pop(name)
              for name in list(remaining) if name.startswith('model.vision_model.')}
    deepseek_vl.load_vision(model.vision_model.core, vision, 'model.vision_model.')
    if vision:
        raise KeyError(f'Unmapped Janus vision state: {sorted(vision)}')
    model.aligner.load_state_dict({name: remaining.pop('model.aligner.' + name)
                                  for name in model.aligner.state_dict()}, strict=True)
    model.text_model.load_state_dict({name: remaining.pop('model.language_model.' + name.replace('.emb.weight', '.weight'))
                                     for name in model.text_model.state_dict()}, strict=True)
    model.lm_head.load_state_dict({'weight': remaining.pop('lm_head.weight')}, strict=True)
    # These native namespaces belong exclusively to explicit generation_mode=image
    # or decode_image_tokens. Preserve their serialization, with no execution
    # claim for that optional branch. All active weights above map strictly.
    inactive = ('model.vqmodel.', 'model.generation_embeddings.', 'model.generation_aligner.', 'model.generation_head.')
    unknown = [name for name in remaining if not name.startswith(inactive)]
    if unknown:
        raise KeyError(f'Unmapped Janus state: {unknown}')
    model.inactive_state = nn.Module()
    for name, value in remaining.items():
        model.inactive_state.register_buffer(name.replace('.', '__'), value.clone())


def make_workloads(model, inputs, config, case=None):
    options = {} if case is None else dict(case['generation_kwargs'])
    if options.pop('generation_mode', 'text') != 'text' or options.pop('do_sample', False):
        raise ValueError('Selected public workload uses greedy text generation')
    return {'generate': Workload(run=lambda: model.generate(**inputs, **options))}
