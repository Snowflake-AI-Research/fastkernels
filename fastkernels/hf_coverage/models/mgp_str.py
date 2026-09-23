"""MGP-STR image encoder and all three learned token-selection heads."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch = config.patch_size
        height, width = config.image_size
        self.proj = Conv2d(config.num_channels, config.hidden_size, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.empty(1, height // patch * (width // patch) + 1, config.hidden_size))

    def forward(self, pixels):
        patches = self.proj(pixels).flatten(2).transpose(1, 2)
        return torch.cat((self.cls_token.expand(pixels.shape[0], -1, -1), patches), dim=1) + self.pos_embed


class TokenSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.token_norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.tokenLearner = nn.Sequential(Conv2d(width, width, 1, groups=8, bias=False),
                                          Conv2d(width, config.max_token_length, 1, bias=False))
        self.feat = Conv2d(width, width, 1, groups=8, bias=False)
        self.norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.softmax, self.matmul = Softmax(), BatchMatMul()

    def forward(self, hidden):
        hidden = self.token_norm(hidden).transpose(1, 2).unsqueeze(-1)
        attention = self.softmax(self.tokenLearner(hidden).flatten(2))
        features = self.feat(hidden).flatten(2).transpose(1, 2)
        return self.norm(self.matmul(attention, features))


class Mgpstr(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mgp_str = nn.Module()
        self.mgp_str.embeddings = Embeddings(config)
        self.mgp_str.encoder = nn.Module()
        self.mgp_str.encoder.blocks = nn.ModuleList([
            VitEncoderBlock(config.hidden_size, config.num_attention_heads, mlp_ratio=config.mlp_ratio,
                            qkv_bias=config.qkv_bias, norm_eps=config.layer_norm_eps)
            for _ in range(config.num_hidden_layers)
        ])
        for name, count in (("char", config.num_character_labels), ("bpe", config.num_bpe_labels),
                            ("wp", config.num_wordpiece_labels)):
            setattr(self, name + "_a3_module", TokenSelector(config))
            setattr(self, name + "_head", Linear(config.hidden_size, count))

    def forward(self, pixel_values):
        hidden = self.mgp_str.embeddings(pixel_values)
        for block in self.mgp_str.encoder.blocks:
            hidden = block(hidden)
        return {f"logits.{index}": getattr(self, name + "_head")(
            getattr(self, name + "_a3_module")(hidden)) for index, name in enumerate(("char", "bpe", "wp"))}


def build_from_config(config, device, dtype):
    if config.distilled:
        raise ValueError("The selected MGP-STR task has no distillation token")
    return Mgpstr(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
