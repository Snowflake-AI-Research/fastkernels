"""PE video/text model composed from existing EVA, pooling, and audio components."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L2.eva_attention import EvaAttention
from fastkernels.tasks.baseline.L2.attention_pool import AttentionPoolLatent, _PoolMlp
from . import pe_audio


def norm(width):
    return LayerNorm(width, eps=1e-5, promote_fp32=False)


class VisionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = norm(width), norm(width)
        self.attn = EvaAttention(width, heads, qkv_bias=True, rotate_half=False)
        self.mlp = _PoolMlp(width, width * 4, width)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class PeVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.architecture != 'vit_pe_core_large_patch14_336':
            raise ValueError('Only the declared PE core large vision architecture is composed')
        args = config.model_args
        args = args or {}
        unsupported = set(args) - {'embed_dim', 'depth', 'num_heads', 'img_size'}
        if unsupported:
            raise ValueError(f'Uncomposed timm model arguments: {unsupported}')
        width, heads = args.get('embed_dim', 1024), args.get('num_heads', 16)
        size = args.get('img_size', 336)
        self.image_size = (size, size) if isinstance(size, int) else tuple(size)
        h, w = (s // 14 for s in self.image_size)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = Conv2d(3, width, 14, stride=14, bias=False)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.pos_embed = nn.Parameter(torch.empty(1, h * w + 1, width))
        self.norm_pre, self.norm = norm(width), norm(width)
        self.blocks = nn.ModuleList([VisionBlock(width, heads) for _ in range(args.get('depth', 24))])
        self.attn_pool = AttentionPoolLatent(width, num_heads=8)
        self.attn_pool.norm = norm(width)
        self.head = Linear(width, config.num_classes)
        # Static positional metadata: native timm XY grid, offset 1, reference 24x24.
        dim = width // heads
        bands = 1. / (10000. ** (torch.arange(dim // 4).float() / (dim // 4)))
        xs, ys = (torch.arange(w).float() + 1) / w * 24, (torch.arange(h).float() + 1) / h * 24
        grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), -1)
        angles = grid[..., None] * bands
        sin = angles.sin().reshape(h * w, -1).repeat_interleave(2, -1)
        cos = angles.cos().reshape(h * w, -1).repeat_interleave(2, -1)
        self.register_buffer('rope', torch.cat((sin, cos), -1), persistent=False)

    def forward(self, pixels):
        if tuple(pixels.shape[-2:]) != self.image_size:
            raise ValueError('PE vision requires its configured input image size')
        x = self.patch_embed.proj(pixels).flatten(2).transpose(1, 2)
        x = self.norm_pre(torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1) + self.pos_embed)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.head(self.attn_pool(self.norm(x)))


class VideoEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.timm_model = PeVision(config.vision_config)
        self.proj = Linear(config.vision_config.num_classes, config.hidden_size, bias=False)
        self.data_proj = Linear(config.hidden_size, config.hidden_size)
        self.normalize = L2Norm()

    def forward(self, pixels, padding_mask=None):
        b, t = pixels.shape[:2]
        x = self.vision_model.timm_model(pixels.reshape(b * t, *pixels.shape[2:]))
        return self.data_proj(self.proj(self.normalize(x.reshape(b, t, -1)))), padding_mask


class PeVideo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = pe_audio.TextEncoder(config.text_config)
        self.video_encoder = VideoEncoder(config.video_config, VideoEmbedder(config.video_config))
        self.text_video_head = pe_audio.ContrastiveHead(config.text_config.hidden_size, config.text_config.hidden_size)
        self.video_head = pe_audio.ContrastiveHead(config.video_config.hidden_size, config.text_config.hidden_size)
        self.text_video_logit_scale = nn.Parameter(torch.empty(1))
        self.text_video_logit_bias = nn.Parameter(torch.empty(1))
        self.matmul = Matmul()

    def forward(self, input_ids, pixel_values_videos, attention_mask=None, padding_mask_videos=None):
        video = self.video_encoder(pixel_values_videos, padding_mask_videos)
        text = self.text_model(input_ids, attention_mask)
        v = self.video_head(video['pooler_output'])
        t = self.text_video_head(text['hidden_states'][-1][:, 0])
        logits = self.matmul(v, t) * self.text_video_logit_scale + self.text_video_logit_bias
        return {'logits_video_text': logits, 'text_video_embeds': t, 'video_embeds': v,
                'text_outputs': text, 'video_outputs': video}


class VideoEncoder(pe_audio.TemporalEncoder):
    def forward(self, values, padding_mask=None):
        output = super().forward(values, padding_mask)
        del output['output_mask']
        return output


def build_from_config(config, device, dtype):
    return PeVideo(config).to(device=device, dtype=dtype).eval()


load_state_dict_into = pe_audio.load_state_dict_into
make_workloads = pe_audio.make_workloads
