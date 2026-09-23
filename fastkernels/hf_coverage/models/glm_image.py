"""GLM-Image image-editing prefix: patch vision, VQ/loss and image-token LM."""

import torch
from torch import nn

from fastkernels.infra.context import get_context
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from ..patches.codec_top1 import CodecTop1
from ..patches.product_gate import ProductGate
from . import glm4v


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        self.attn = VitEncoderAttention(config.hidden_size, config.num_heads, config.attention_bias, config.attention_bias)
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size)

    def forward(self, hidden, lengths):
        attention = torch.cat([self.attn(chunk[None])[0] for chunk in self.norm1(hidden).split(lengths)])
        hidden = hidden + attention
        return hidden + self.mlp(self.norm2(hidden))


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = Conv2d(config.in_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        self.position_embedding = nn.Parameter(torch.empty((config.image_size // config.patch_size) ** 2, config.hidden_size))
        self.interpolate = Interpolate()
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.depth))

    def forward(self, pixels, grid):
        config, grids = self.config, grid.tolist()
        hidden = self.patch_embed(pixels.reshape(-1, config.in_channels, config.patch_size, config.patch_size)).flatten(1)
        side = config.image_size // config.patch_size
        weight = self.position_embedding.float().reshape(side, side, -1).permute(2, 0, 1)[None]
        positions = [self.interpolate(weight, size=(h, w), mode='bilinear', align_corners=False)[0]
                     .permute(1, 2, 0).reshape(h * w, -1).repeat(t, 1) for t, h, w in grids]
        hidden = hidden + torch.cat(positions).to(hidden.dtype)
        lengths = [h * w for t, h, w in grids for _ in range(t)]
        for block in self.blocks:
            hidden = block(hidden, lengths)
        return hidden


class Quantizer(nn.Module):
    """Linear-storage composition retaining duplicate norms and both loss terms."""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(config.num_embeddings, config.embed_dim))
        self.normalize, self.product = L2Norm(), ProductGate()
        self.average, self.matmul, self.select = GlobalAvgPool2d(), Matmul(), CodecTop1()
        self.beta = getattr(config, 'beta', 0.25)

    def square(self, values):
        return self.product(torch.cat((values, values), dim=-1))

    def sum_last(self, values):
        # Sum's FP32 accumulation and final source-dtype store are retained.
        means = self.average(values.float().reshape(-1, 1, 1, values.shape[-1]))
        return (means * values.shape[-1]).to(values.dtype)

    def loss_term(self, difference):
        return self.average(self.square(difference).reshape(1, 1, 1, -1)).reshape(())

    def forward(self, hidden):
        hidden = hidden.permute(0, 2, 3, 1).contiguous()
        flattened = self.normalize(hidden.reshape(-1, hidden.shape[-1]))
        hidden = self.normalize(hidden)
        embedding = self.normalize(self.embedding)
        distances = self.sum_last(self.square(flattened)) + self.sum_last(self.square(embedding)).transpose(0, 1)
        distances = distances - 2 * self.matmul(flattened, embedding)
        indices = self.select(-distances)
        quantized = embedding[indices].reshape_as(hidden)
        loss = self.loss_term(quantized - hidden) + self.beta * self.loss_term(quantized - hidden)
        quantized = hidden + (quantized - hidden)
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, indices


class VQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantize = Quantizer(config)
        self.quant_conv = Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = Conv2d(config.embed_dim, config.latent_channels, 1)

    def forward(self, features, grid):
        outputs, losses, quantized = [], [], []
        grids = grid.tolist()
        for value, (t, h, w) in zip(features.split([t * h * w for t, h, w in grids]), grids):
            value = value.reshape(t, h, w, -1).permute(0, 3, 1, 2).contiguous()
            quant, loss, indices = self.quantize(self.quant_conv(value))
            outputs.append(indices)
            losses.append(loss)
            quantized.append(quant)
        return torch.cat(outputs), losses, quantized


class Backbone(nn.Module):
    def __init__(self, text, config):
        super().__init__()
        self.text, self.config = text, config
        self.vision, self.vqmodel = Vision(config.vision_config), VQ(config.vq_config)
        self.inputs = self.rope_delta = self.decode_positions = None
        self.prefill_length = None

    @property
    def layers(self):
        return self.text.layers

    def forward(self, input_ids, positions):
        if not get_context().is_prefill:
            decode_offsets = positions - self.prefill_length
            return self.text(input_ids, self.decode_positions[:, decode_offsets])
        self.prefill_length = input_ids.numel()
        grids = self.inputs['image_grid_thw']
        source_grids = grids[:-1]
        features = self.vision(self.inputs['pixel_values'], source_grids)
        tokens, _, _ = self.vqmodel(features, source_grids)
        ids = input_ids.clone()
        ids[input_ids == self.config.image_token_id] = tokens
        # Metadata uses complete source-image boundary markers and the final target grid.
        starts = (input_ids == self.config.image_start_token_id).nonzero().flatten().tolist()
        ends = (input_ids == self.config.image_end_token_id).nonzero().flatten().tolist()
        pieces, current, previous = [], 0, 0
        for start, end, (t, h, w) in zip(starts, ends, source_grids.tolist()):
            count = start + 1 - previous
            pieces.append(torch.arange(count, device=ids.device)[None].expand(3, -1) + current)
            current += count
            spatial = torch.stack((torch.arange(t, device=ids.device).repeat_interleave(h * w),
                torch.arange(h, device=ids.device).repeat_interleave(w).repeat(t),
                torch.arange(w, device=ids.device).repeat(h * t)))
            pieces.append(spatial + current)
            current += max(h, w)
            previous = end
        count = input_ids.numel() - previous
        pieces.append(torch.arange(count, device=ids.device)[None].expand(3, -1) + current)
        current += count
        positions = torch.cat(pieces, dim=-1)
        _, h, w = grids[-1].tolist()
        self.decode_positions = torch.stack((torch.full((h * w,), current, device=ids.device),
            torch.arange(h, device=ids.device).repeat_interleave(w) + current,
            torch.arange(w, device=ids.device).repeat(h) + current))
        self.rope_delta = torch.zeros((1, 1), device=ids.device, dtype=torch.long)
        return self.text(ids, positions)


def build_from_config(config, device, dtype):
    if config.vision_config.spatial_merge_size != 1 or config.vision_config.hidden_act != 'gelu':
        raise ValueError('Documented GLM-Image uses unmerged patches and exact-GELU vision')
    language = glm4v.make_text(config.text_config, dtype, native_attention=True)
    language.lm_head = ParallelLMHead(config.text_config.vision_vocab_size, config.text_config.hidden_size)
    model = nn.Module()
    model.model, model.lm_head, model.config = Backbone(language.model, config), language.lm_head, language.config
    model.to(device=device, dtype=dtype).eval()
    rotary = glm4v.MultimodalRotary(config.text_config, interleaved=False).to(device=device)
    model.model.text.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    glm4v.load_text(model, remaining, config)
    mapped = {}
    for name, value in model.model.vision.state_dict().items():
        source = 'embeddings.position_embedding.weight' if name == 'position_embedding' else name.replace('patch_embed.', 'patch_embed.proj.')
        supplied = remaining.pop('model.visual.' + source)
        if value.shape != supplied.shape:
            raise ValueError(f'GLM-Image vision shape mismatch: {source}')
        mapped[name] = supplied
    model.model.vision.load_state_dict(mapped, strict=True)
    mapped = {name: remaining.pop('model.vqmodel.' + name.replace('quantize.embedding', 'quantize.embedding.weight'))
              for name in model.model.vqmodel.state_dict()}
    model.model.vqmodel.load_state_dict(mapped, strict=True)
    if remaining:
        raise KeyError(f'Unmapped GLM-Image weights: {sorted(remaining)}')


def make_workloads(model, inputs, config, *, case=None):
    # Input embedding vocabulary differs from the image-token output vocabulary.
    model.config.vocab_size = config.text_config.vision_vocab_size
    return glm4v.make_workloads(model, inputs, config, case=case)
