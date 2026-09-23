"""Idefics3: padded-image filtering, SigLIP, pixel packing and Llama3."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM
from ..patches.codec_top1 import CodecTop1
from . import deepseek_vl, llama, llava
from .olmo2 import decoder_config


class RealImages(nn.Module):
    """Detect nonzero finite images using existing reductions and argmax.

    Packed positive/negative pixels give max(abs(pixel)); first-index argmax
    of [0, maximum] returns zero exactly for an all-zero padding image.
    Both packing and the reduction stay in the measured forward.
    """

    def __init__(self, channels, image_size):
        super().__init__()
        self.maximum = MaxPool2d((2 * channels * image_size, image_size))
        self.select = CodecTop1()

    def forward(self, pixels):
        pixels = pixels.flatten(0, 1)
        packed = torch.cat((pixels, -pixels), dim=1).flatten(1, 2).unsqueeze(1)
        maximum = self.maximum(packed).flatten()
        present = self.select(torch.stack((torch.zeros_like(maximum), maximum), dim=-1)).bool()
        return pixels[present].contiguous()


def vision_model(config):
    model = deepseek_vl.make_vision(config)
    if config.hidden_act != 'gelu_pytorch_tanh':
        raise ValueError('Selected Idefics vision uses tanh GELU')
    for layer in model.layers:
        layer.mlp.act = GELU('tanh')
    return model


def square_position_ids(config, dtype, device):
    """Native position metadata for the selected fully valid square image."""
    side = config.image_size // config.patch_size
    boundaries = torch.arange(1 / side, 1.0, 1 / side, device=device, dtype=torch.float32)
    step = 1.0 / torch.tensor(side, device=device)
    coordinates = (torch.arange(side, device=device, dtype=torch.float32) * step).clamp(max=1.0 - 1e-6)
    buckets = torch.bucketize(coordinates.to(dtype), boundaries, right=True)
    return (buckets[:, None] * side + buckets[None, :]).flatten()


class Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.scale_factor
        self.proj = Linear(config.vision_config.hidden_size * self.scale ** 2,
                           config.text_config.hidden_size, bias=False)

    def forward(self, hidden):
        batch, sequence, width = hidden.shape
        side, scale = int(sequence ** 0.5), self.scale
        hidden = hidden.view(batch, side, side // scale, width * scale).permute(0, 2, 1, 3)
        hidden = hidden.reshape(batch, side // scale, side // scale, width * scale ** 2)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, sequence // scale ** 2, width * scale ** 2)
        return self.proj(hidden)


class Backbone(deepseek_vl.DeepseekBackbone):
    def __init__(self, text, config):
        nn.Module.__init__(self)
        self.text = text
        self.real_images = RealImages(3, config.vision_config.image_size)
        self.vision = vision_model(config.vision_config)
        self.connector = Connector(config)
        self.image_token_id = config.image_token_id
        self.pixel_values = self.image_hidden_states = None

    def features(self, pixels):
        return self.connector(self.vision(self.real_images(pixels)))


class Model(nn.Module):
    def __init__(self, language, config, backbone=Backbone):
        super().__init__()
        self.config, self.lm_head = language.config, language.lm_head
        self.model = backbone(language.model, config)


def build_from_config(config, device, dtype):
    text = config.text_config
    rope = text.rope_parameters
    if (rope['rope_type'] != 'llama3' or text.hidden_act != 'silu'
            or text.tie_word_embeddings or text.attention_bias or text.mlp_bias):
        raise ValueError('Preserve the documented untied Llama3 decoder and its scaled RoPE')
    native = decoder_config(text, dtype)
    native.rope_scaling_factor = rope['factor']
    native.rope_low_freq_factor = rope['low_freq_factor']
    native.rope_high_freq_factor = rope['high_freq_factor']
    native.rope_original_max_position_embeddings = rope['original_max_position_embeddings']
    return Model(LlamaForCausalLM(native), config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    text = {name.replace('model.text_model.', 'model.'): remaining.pop(name)
            for name in list(remaining) if name.startswith('model.text_model.')}
    text['lm_head.weight'] = remaining.pop('lm_head.weight')
    llama.load_state_dict_into(SimpleNamespace(model=model.model.text, lm_head=model.lm_head), text, config.text_config)
    deepseek_vl.load_vision(model.model.vision, remaining, 'model.vision_model.')
    # Native embeddings round fractional coordinates through pixel dtype even
    # for a fully valid square. Reorder these fixed parameter rows at loading;
    # the existing SigLIP forward then executes the native position addition.
    positions = model.model.vision.position_embedding
    device = 'cpu' if positions.is_meta else positions.device
    indices = square_position_ids(config.vision_config, positions.dtype, device).to(positions.device)
    positions.copy_(positions.index_select(1, indices))
    model.model.connector.proj.weight.copy_(remaining.pop('model.connector.modality_projection.proj.weight'))
    if remaining:
        raise KeyError(f'Unmapped Idefics3 weights: {sorted(remaining)}')


def make_workloads(model, inputs, config, case=None):
    expected = (config.vision_config.image_size, config.vision_config.image_size)
    if tuple(inputs['pixel_values'].shape[-2:]) != expected:
        raise ValueError('This Idefics3 workload requires its selected full square image geometry')
    if 'pixel_attention_mask' in inputs and not inputs['pixel_attention_mask'].bool().all():
        raise ValueError('This Idefics3 workload requires fully valid image patches')
    return llava.make_workloads(model, inputs, config, case=case)
