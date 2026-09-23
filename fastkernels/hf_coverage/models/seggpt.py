"""SegGPT prompt painting with default masking, branch merge, and mask decoder."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.vitmatte import SpatialRelativeAttention
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.linear import Linear


class _Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        for name in ("mask_token", "segment_token_input", "segment_token_prompt", "type_token_semantic", "type_token_instance"):
            self.register_parameter(name, nn.Parameter(torch.empty(1, 1, 1, c.hidden_size)))
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = Conv2d(c.num_channels, c.hidden_size, c.patch_size, stride=c.patch_size)
        self.pretrain_grid = c.pretrain_image_size // c.patch_size
        self.position_embeddings = nn.Parameter(torch.empty(1, self.pretrain_grid**2+1, c.hidden_size))
        self.resize = Interpolate()

    def forward(self, pixels, prompt_pixels, prompt_masks):
        images = torch.cat((prompt_pixels, pixels), dim=2)
        masks = torch.cat((prompt_masks, prompt_masks), dim=2)
        image_tokens = self.patch_embeddings.projection(images).permute(0, 2, 3, 1)
        mask_tokens = self.patch_embeddings.projection(masks).permute(0, 2, 3, 1)
        batch, h, w, width = image_tokens.shape
        # Default bool_masked_pos is fixed: only the target (lower) half is hidden.
        mask_tokens = torch.cat((mask_tokens[:, :h//2], self.mask_token.expand(batch, h-h//2, w, width)), dim=1)
        positions = self.position_embeddings[:, 1:].reshape(1, self.pretrain_grid, self.pretrain_grid, width).permute(0, 3, 1, 2)
        positions = self.resize(positions, size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        image_tokens = image_tokens + self.segment_token_input + positions + self.type_token_instance
        mask_tokens = mask_tokens + self.segment_token_prompt + positions + self.type_token_instance
        return torch.cat((image_tokens, mask_tokens), dim=0)


class _MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lin1, self.lin2, self.act = Linear(c.hidden_size, c.mlp_dim), Linear(c.mlp_dim, c.hidden_size), GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class _Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        shape = tuple(length//c.patch_size for length in c.image_size)
        self.attention = SpatialRelativeAttention(c.hidden_size, c.num_attention_heads, shape, c.qkv_bias, fp32_softmax=True)
        self.layernorm_before = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.layernorm_after = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.mlp = _MLP(c)

    def forward(self, hidden):
        hidden = hidden + self.attention(self.layernorm_before(hidden))
        return hidden + self.mlp(self.layernorm_after(hidden))


class _DecoderHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = Conv2d(c.decoder_hidden_size, c.decoder_hidden_size, 3, padding=1)
        self.layernorm = LayerNorm2d(c.decoder_hidden_size, eps=c.layer_norm_eps)
        self.act_fct = GELU()
        self.head = Conv2d(c.decoder_hidden_size, 3, 1)

    def forward(self, x):
        return self.head(self.act_fct(self.layernorm(self.conv(x))))


class SegGptForImageSegmentation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.model = nn.Module()
        self.model.embeddings = _Embeddings(c)
        self.model.encoder = nn.Module()
        self.model.encoder.layers = nn.ModuleList([_Layer(c) for _ in range(c.num_hidden_layers)])
        self.model.encoder.layernorm = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.decoder = nn.Module()
        self.decoder.decoder_embed = Linear(c.hidden_size*len(c.intermediate_hidden_state_indices), c.patch_size**2*c.decoder_hidden_size)
        self.decoder.decoder_pred = _DecoderHead(c)
        self.config = c

    def forward(self, pixel_values, prompt_pixel_values, prompt_masks):
        c = self.config
        hidden = self.model.embeddings(pixel_values, prompt_pixel_values, prompt_masks)
        intermediate = []
        for index, layer in enumerate(self.model.encoder.layers):
            hidden = layer(hidden)
            if index == c.merge_index:
                image_branch, mask_branch = hidden.chunk(2, dim=0)
                hidden = (image_branch + mask_branch) * 0.5
            if index in c.intermediate_hidden_state_indices:
                intermediate.append(self.model.encoder.layernorm(hidden))
        decoded = self.decoder.decoder_embed(torch.cat(intermediate, dim=-1))
        batch, h, w, _ = decoded.shape
        decoded = decoded.reshape(batch, h, w, c.patch_size, c.patch_size, c.decoder_hidden_size)
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).reshape(batch, c.decoder_hidden_size, h*c.patch_size, w*c.patch_size)
        return {"pred_masks": self.decoder.decoder_pred(decoded)}


def build_from_config(config, device, dtype):
    if not config.use_relative_position_embeddings or config.hidden_act != "gelu":
        raise ValueError("The declared SegGPT checkpoint uses relative spatial attention and GELU")
    return SegGptForImageSegmentation(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
