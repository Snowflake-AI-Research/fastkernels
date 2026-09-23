"""PerceptionLM's PE vision tower, pooling projector and image/video text generation."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.eva_block import EvaBlock
from ..patches.codec_top1 import CodecTop1
from ..runner import Workload
from .smolvlm import TextDecoder


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.architecture != 'vit_pe_core_large_patch14_336':
            raise ValueError('Selected PerceptionLM uses the PE large patch14 EVA graph')
        args = config.model_args
        width, heads = args['embed_dim'], args.get('num_heads', 16)
        h, w = args['img_size']
        gh, gw = h // 14, w // 14
        self.patch_embed = nn.Module()
        self.patch_embed.proj = Conv2d(3, width, 14, stride=14, bias=False)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.pos_embed = nn.Parameter(torch.empty(1, 1 + gh * gw, width))
        self.norm_pre = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.blocks = nn.ModuleList()
        for _ in range(args['depth']):
            block = EvaBlock(width, heads, qkv_bias=True, rotate_half=False, init_values=args['init_values'])
            block.norm1 = LayerNorm(width, eps=1e-5, promote_fp32=False)
            block.norm2 = LayerNorm(width, eps=1e-5, promote_fp32=False)
            block.mlp = VitEncoderMlp(width, width * 4, act_approximate='none')
            self.blocks.append(block)
        if args['global_pool'] != '' or args['use_post_transformer_norm']:
            raise ValueError('Selected tower has identity final normalization and pooling/head')
        # Fixed XY rotary grid only; activation rotation is the unchanged
        # EvaAttention parent, including interleaved rotation and dtype stores.
        ref_h, ref_w = args['ref_feat_shape']
        bands = 1. / (10000. ** (torch.arange(width // heads // 4).float() / (width // heads // 4)))
        y = (torch.arange(gh).float() + 1.) / gh * ref_h
        x = (torch.arange(gw).float() + 1.) / gw * ref_w
        phase = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)[..., None] * bands
        sine = phase.sin().reshape(gh * gw, -1).repeat_interleave(2, -1)
        cosine = phase.cos().reshape(gh * gw, -1).repeat_interleave(2, -1)
        self.register_buffer('rope', torch.cat((sine, cosine), -1), persistent=False)

    def forward(self, pixels):
        hidden = self.patch_embed.proj(pixels).flatten(2).transpose(1, 2)
        hidden = torch.cat((self.cls_token.expand(hidden.shape[0], -1, -1), hidden), dim=1)
        hidden = self.norm_pre(hidden + self.pos_embed)
        for block in self.blocks:
            hidden = block(hidden, rope=self.rope)
        # Native forward_head is entirely identities for global_pool=''.
        return hidden


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.text_config.hidden_size
        self.linear_1 = Linear(config.vision_config.model_args['embed_dim'], width)
        self.linear_2 = Linear(width, width)
        self.gelu = GELU()
        self.ratio = config.projector_pooling_ratio
        self.pooling = AvgPool2d(self.ratio, stride=self.ratio)

    def forward(self, hidden):
        hidden = self.linear_2(self.gelu(self.linear_1(hidden.transpose(0, 1)))).transpose(0, 1)
        batch, length, width = hidden.shape
        side = int(length ** .5)
        if side % self.ratio:
            raise ValueError('Selected image grid must divide the native pooling ratio')
        return self.pooling(hidden.transpose(1, 2).reshape(batch, width, side, side)).flatten(2).transpose(1, 2)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = Vision(config.vision_config)
        self.multi_modal_projector = Projector(config)
        self.text_model = TextDecoder(config.text_config)
        self.lm_head = Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.text_model.embed_tokens.emb.weight
        self.select = CodecTop1()

    def features(self, pixels):
        hidden = self.vision_tower(pixels.flatten(0, 1))
        return self.multi_modal_projector(hidden[:, 1:] if self.config.vision_use_cls_token else hidden)

    def generate(self, input_ids, pixel_values=None, pixel_values_videos=None, max_new_tokens=4):
        if input_ids.shape[0] != 1:
            raise ValueError('Selected generation uses one prompt with image and video inputs')
        self.text_model.reset()
        ids = input_ids.clone()
        outputs = {}
        eos = self.config.text_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        for step in range(max_new_tokens):
            current = ids if step == 0 else ids[:, -1:]
            hidden = self.text_model.embed_tokens(current)
            if step == 0:
                for pixels, token in ((pixel_values, self.config.image_token_id),
                                      (pixel_values_videos, self.config.video_token_id)):
                    if pixels is not None:
                        hidden = hidden.masked_scatter((current == token)[..., None].expand_as(hidden), self.features(pixels))
            start = 0 if step == 0 else ids.shape[1] - 1
            hidden = self.text_model(hidden, torch.arange(start, ids.shape[1], device=ids.device))
            logits = self.lm_head(hidden[:, -1:])[:, -1].float()
            outputs[f'logits.{step}'] = logits
            token = self.select(logits).reshape(1, 1)
            ids = torch.cat((ids, token), dim=1)
            if int(token[0, 0]) in eos:
                break
        outputs['sequences'] = ids
        for index, layer in enumerate(self.text_model.layers):
            outputs[f'past_key_values.{index}.key'] = layer.self_attn.key.transpose(1, 2)
            outputs[f'past_key_values.{index}.value'] = layer.self_attn.value.transpose(1, 2)
        return outputs


def build_from_config(config, device, dtype):
    return Model(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for name, module, prefix in [('vision', model.vision_tower, 'model.vision_tower.timm_model.'),
                                ('projector', model.multi_modal_projector, 'model.multi_modal_projector.'),
                                ('text', model.text_model, 'model.language_model.')]:
        values = {key: remaining.pop(prefix + key.replace('.emb.weight', '.weight')) for key in module.state_dict()}
        module.load_state_dict(values, strict=True)
    head = remaining.pop('lm_head.weight')
    if config.text_config.tie_word_embeddings and not torch.equal(head, model.text_model.embed_tokens.emb.weight):
        raise ValueError('Native tied head and embeddings disagree')
    model.lm_head.load_state_dict({'weight': head}, strict=True)
    if remaining:
        raise KeyError(f'Unmapped PerceptionLM state: {sorted(remaining)}')


def make_workloads(model, inputs, config, case=None):
    options = {} if case is None else dict(case['generation_kwargs'])
    if options.pop('do_sample', False):
        raise ValueError('Selected public workload uses greedy generation')
    return {'generate': Workload(run=lambda: model.generate(**inputs, **options))}
