"""ModernVBERT image merging and masked language modeling from existing towers."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.segment_csr import SegmentCSR
from fastkernels.tasks.baseline.L2.attention_pool import AttentionPoolLatent
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionEncoder
from .modernbert import ModernLayer, normalization
from .siglip import configure_encoder
from ..patches.codec_top1 import CodecTop1
from ..runner import Workload


class ImagePaddingFilter(nn.Module):
    """Keep nonzero images with linear-work operations, retaining HF's empty fallback."""

    def __init__(self):
        super().__init__()
        self.relu, self.reduce, self.top1 = ReLU(), SegmentCSR(), CodecTop1()

    def forward(self, images):
        values = self.relu(images) + self.relu(-images)
        count = images.shape[0]
        offsets = torch.arange(count + 1, device=images.device) * images[0].numel()
        maxima = self.reduce(values.flatten(), offsets, reduce='max')
        nonzero = self.top1(torch.stack((torch.zeros_like(maxima), maxima), dim=-1))
        maximum = self.reduce(nonzero.float(), offsets.new_tensor([0, count]), reduce='max')
        any_image = self.top1(torch.stack((torch.zeros_like(maximum), maximum), dim=-1)).bool()
        keep = nonzero.bool()
        keep[0] |= ~any_image[0]
        return images[keep].contiguous()


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SigLIPVisionEncoder(config)
        configure_encoder(self.encoder.layers)
        self.encoder.post_layernorm.promote_fp32 = False
        self.pool = None
        if getattr(config, 'vision_use_head', True):
            self.pool = AttentionPoolLatent(config.hidden_size, num_heads=config.num_attention_heads,
                                            mlp_ratio=config.intermediate_size / config.hidden_size)
            self.pool.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
            self.pool.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, act_approximate='tanh', bias=True)

    def forward(self, pixels):
        hidden = self.encoder(pixels)
        if self.pool is not None:
            # The native tower computes its head even though ModernVBERT replaces its output.
            self.pool(hidden)
        return hidden


class ModernVBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        text, vision = config.text_config, config.vision_config
        self.image_token_id, self.factor = config.image_token_id, config.pixel_shuffle_factor
        self.padding_filter = ImagePaddingFilter()
        self.vision_model = VisionTower(vision)
        self.connector = Linear(vision.hidden_size * self.factor**2, text.hidden_size, bias=False)
        self.embeddings = Embedding(text.vocab_size, text.hidden_size, padding_idx=text.pad_token_id)
        self.embedding_norm = normalization(text)
        self.layers = nn.ModuleList([ModernLayer(text, index, causal=False) for index in range(text.num_hidden_layers)])
        self.final_norm = normalization(text)

    def forward(self, input_ids, pixel_values=None, attention_mask=None):
        hidden = self.embeddings(input_ids)
        image = None
        if pixel_values is not None:
            pixels = self.padding_filter(pixel_values.flatten(0, 1))
            image = self.vision_model(pixels)
            batch, length, width = image.shape
            side, factor = int(length**0.5), self.factor
            image = image.view(batch, side, side, width).view(batch, side, side // factor, width * factor)
            image = image.permute(0, 2, 1, 3).reshape(batch, side // factor, side // factor, width * factor**2)
            image = image.permute(0, 2, 1, 3).reshape(batch, length // factor**2, width * factor**2)
            image = self.connector(image)
            # Integer token positions identify image blocks; no activation reductions are hidden here.
            hidden[input_ids == self.image_token_id] = image.reshape(-1, image.shape[-1])
        hidden = self.embedding_norm(hidden)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for layer in self.layers:
            hidden, _ = layer(hidden, positions, attention_mask=attention_mask)
        return self.final_norm(hidden), image


class ModernVBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ModernVBertModel(config)
        text = config.text_config
        self.dense = Linear(text.hidden_size, text.hidden_size, bias=text.classifier_bias)
        self.activation = GELU()
        self.norm = normalization(text)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=text.decoder_bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.emb.weight

    def forward(self, input_ids, pixel_values):
        hidden, image = self.model(input_ids, pixel_values)
        return {'logits': self.lm_head(self.norm(self.activation(self.dense(hidden)))), 'image_hidden_states': image}


def setup_model(model, config, device, dtype):
    if (config.text_config.hidden_activation != 'gelu' or config.text_config.mlp_bias
            or config.vision_config.hidden_act != 'gelu_pytorch_tanh'):
        raise ValueError('ModernVBERT case preserves gated GELU text and tanh GELU vision')
    model.to(device=device, dtype=dtype).eval()
    base = model.model if hasattr(model, 'model') else model.vlm
    for index, layer in enumerate(base.layers):
        rope = config.text_config.rope_parameters[config.text_config.layer_types[index]]
        if rope['rope_type'] != 'default':
            raise ValueError('ModernVBERT case uses default rotary coefficients')
        layer.rotary = RotaryEmbedding(layer.width, config.text_config.max_position_embeddings,
                                       rope['rope_theta']).to(device=device)
    return model


def build_from_config(config, device, dtype):
    return setup_model(ModernVBertForMaskedLM(config), config, device, dtype)


def base_source(name):
    if name.startswith('vision_model.encoder.'):
        suffix = name.removeprefix('vision_model.encoder.')
        suffix = suffix.replace('layers.', 'encoder.layers.').replace('patch_embedding.', 'embeddings.patch_embedding.')
        suffix = suffix.replace('position_embedding', 'embeddings.position_embedding.weight')
        return 'vision_model.' + suffix
    if name.startswith('vision_model.pool.'):
        suffix = name.removeprefix('vision_model.pool.')
        suffix = {'latent': 'probe'}.get(suffix, suffix)
        suffix = suffix.replace('proj.', 'attention.out_proj.').replace('norm.', 'layernorm.')
        return 'vision_model.head.' + suffix
    if name.startswith('layers.'):
        name = 'text_model.' + name
        return (name.replace('.qkv.', '.attn.Wqkv.').replace('.out_proj.', '.attn.Wo.')
                .replace('.mlp.gate_up_proj.', '.mlp.Wi.').replace('.mlp.down_proj.', '.mlp.Wo.'))
    for target, source in [('embeddings.emb.', 'text_model.embeddings.tok_embeddings.'),
                           ('embedding_norm.', 'text_model.embeddings.norm.'),
                           ('final_norm.', 'text_model.final_norm.'),
                           ('connector.', 'connector.modality_projection.')]:
        if name.startswith(target):
            return name.replace(target, source, 1)
    raise KeyError(name)


def mapped_base(model, state, prefix):
    mapped, used = {}, set()
    for name, parameter in model.state_dict().items():
        if name.startswith(('vision_model.pool.q.', 'vision_model.pool.kv.')):
            field = name.rsplit('.', 1)[-1]
            source = prefix + 'vision_model.head.attention.in_proj_' + field
            width = model.vision_model.pool.q.weight.shape[0]
            mapped[name] = state[source][:width] if '.q.' in name else state[source][width:]
        else:
            source = prefix + base_source(name)
            mapped[name] = state[source].reshape(parameter.shape)
        used.add(source)
    return mapped, used


def load_state_dict_into(model, state_dict, config):
    base, used = mapped_base(model.model, state_dict, 'model.')
    mapped = {'model.' + name: value for name, value in base.items()}
    for name in ('dense.weight', 'dense.bias', 'norm.weight', 'norm.bias', 'lm_head.weight', 'lm_head.bias'):
        if name not in model.state_dict():
            continue
        source = name if name.startswith('lm_head.') else 'projection_head.' + name
        mapped[name] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise KeyError(f'Unmapped ModernVBERT weights: {sorted(set(state_dict) - used)}')
    if config.tie_word_embeddings and not torch.equal(mapped['lm_head.weight'], mapped['model.embeddings.emb.weight']):
        raise ValueError('Tied ModernVBERT embeddings disagree')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
