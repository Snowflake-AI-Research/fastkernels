"""AyaVision's SigLIP features, normalized pixel shuffle and Cohere2 decoder."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from . import cohere2
from .modernvbert import VisionTower, base_source
from .cohere2_vision import make_workloads


class AyaVision(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.image_token_id = config.image_token_index
        self.factor = config.downsample_factor
        self.vision = VisionTower(config.vision_config).to(device=device, dtype=dtype)
        width = config.vision_config.hidden_size * self.factor**2
        self.layernorm = LayerNorm(width, eps=config.adapter_layer_norm_eps, promote_fp32=False)
        self.linear_1 = Linear(width, config.alignment_intermediate_size)
        self.linear_2 = Linear(config.alignment_intermediate_size // 2, config.text_config.hidden_size)
        self.layernorm.to(device=device, dtype=dtype)
        self.linear_1.to(device=device, dtype=dtype)
        self.linear_2.to(device=device, dtype=dtype)
        self.language_model = cohere2.build_from_config(config.text_config, device, dtype)

    def image_features(self, pixels):
        encoder = self.vision.encoder
        hidden = encoder.patch_embedding(pixels).flatten(2).transpose(1, 2)
        hidden = hidden + encoder.position_embedding
        for layer in encoder.layers:
            hidden = layer(hidden)
        # HF executes the whole tower, but hidden_states[-1] is the last encoder
        # output BEFORE post_layernorm. Its normalized last_hidden_state is unused.
        encoder.post_layernorm(hidden)
        batch, length, width = hidden.shape
        side, factor = int(length**0.5), self.factor
        if side * side != length or side % factor:
            raise ValueError('AyaVision pixel shuffle requires a divisible square patch grid')
        image = hidden.reshape(batch, side, side // factor, width * factor)
        image = image.permute(0, 2, 1, 3).reshape(batch, side // factor, side // factor, -1)
        image = image.permute(0, 2, 1, 3)
        value, gate = self.linear_1(self.layernorm(image)).chunk(2, dim=-1)
        return self.linear_2(SiluAndMul.forward_native(torch.cat((gate, value), dim=-1)))

    def forward(self, input_ids, pixel_values=None, past_key_values=None):
        if self.training:
            raise RuntimeError('AyaVision coverage supports inference only')
        text = self.language_model
        hidden = text.embed_tokens(input_ids)
        output = {}
        if pixel_values is not None:
            image = self.image_features(pixel_values).to(hidden.dtype)
            mask = (input_ids == self.image_token_id)[..., None].expand_as(hidden)
            if hidden[mask].numel() != image.numel():
                raise ValueError('Image placeholders must match projected feature count')
            hidden = hidden.masked_scatter(mask, image)
            output['image_hidden_states'] = image
        start = 0 if past_key_values is None else past_key_values.seen_tokens
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + start
        states = []
        for index, layer in enumerate(text.layers):
            hidden, state = layer(hidden, positions, text.rotary.cos_sin_cache,
                                  None if past_key_values is None else past_key_values.layers[index])
            states.append(state)
        # Pinned AyaVision calls Cohere2Model, not Cohere2ForCausalLM: its own
        # tied head has no multiplication by the child's configured logit_scale.
        output['logits'] = text.lm_head(text.norm(hidden))
        output['past_key_values'] = cohere2.Cache(tuple(states), start + input_ids.shape[1])
        return output


def build_from_config(config, device, dtype):
    vision = config.vision_config
    if (vision.model_type != 'siglip_vision_model' or config.text_config.model_type != 'cohere2'
            or not config.tie_word_embeddings or getattr(vision, 'vision_use_head', True)
            or vision.hidden_act != 'gelu_pytorch_tanh' or vision.num_channels != 3
            or config.vision_feature_layer != -1 or config.vision_feature_select_strategy != 'full'
            or config.downsample_factor != 2 or config.alignment_intermediate_size % 2
            or not 0 <= config.image_token_index < config.text_config.vocab_size):
        raise ValueError('AyaVision selected source requires full final SigLIP features, no pool, factor2 and tied Cohere2')
    return AyaVision(config, device, dtype).eval()


def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict['lm_head.weight'], state_dict['model.language_model.embed_tokens.weight']):
        raise ValueError('AyaVision tied input and output weights disagree')
    mapped, used = {}, set()
    for name, target in model.state_dict().items():
        if name.startswith('vision.'):
            suffix = base_source(name.replace('vision.', 'vision_model.', 1)).removeprefix('vision_model.')
            source = 'model.vision_tower.' + suffix
        elif name.startswith(('linear_1.', 'linear_2.', 'layernorm.')):
            source = 'model.multi_modal_projector.' + name
        elif name == 'language_model.lm_head.weight':
            source = 'lm_head.weight'
        else:
            source = 'model.' + name.replace('.emb.weight', '.weight')
        mapped[name] = state_dict[source].reshape(target.shape)
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped AyaVision weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)
    for module in model.modules():
        if isinstance(module, LayerNorm):
            module._cast_done = False
