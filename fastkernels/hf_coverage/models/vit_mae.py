"""Masked-image encoder/decoder and default reconstruction loss using existing ops."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from ..patches.product_gate import ProductGate
from ..runner import Workload


class ViTMAEForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size, self.image_size = config.patch_size, config.image_size
        self.patch_count = (config.image_size // config.patch_size) ** 2
        self.keep_count = int(self.patch_count * (1 - config.mask_ratio))
        width, decoder_width = config.hidden_size, config.decoder_hidden_size
        self.projection = Conv2d(config.num_channels, width, config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.position_embeddings = nn.Parameter(torch.empty(1, self.patch_count + 1, width), requires_grad=False)
        self.encoder = nn.ModuleList([
            VitEncoderBlock(width, config.num_attention_heads, config.intermediate_size / width,
                            qkv_bias=config.qkv_bias, norm_eps=config.layer_norm_eps)
            for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.decoder_embed = Linear(width, decoder_width)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_width))
        self.decoder_pos_embed = nn.Parameter(torch.empty(1, self.patch_count + 1, decoder_width), requires_grad=False)
        self.decoder = nn.ModuleList([
            VitEncoderBlock(decoder_width, config.decoder_num_attention_heads,
                            config.decoder_intermediate_size / decoder_width,
                            qkv_bias=config.qkv_bias, norm_eps=config.layer_norm_eps)
            for _ in range(config.decoder_num_hidden_layers)
        ])
        self.decoder_norm = LayerNorm(decoder_width, eps=config.layer_norm_eps, promote_fp32=False)
        self.decoder_pred = Linear(decoder_width, config.patch_size ** 2 * config.num_channels)
        self.product = ProductGate()
        self.mean = GlobalAvgPool2d()

    def forward(self, pixel_values, noise):
        batch, channels, height, width = pixel_values.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError("ViT-MAE image dimensions must match the configuration")
        hidden = self.projection(pixel_values).flatten(2).transpose(1, 2)
        hidden = hidden + self.position_embeddings[:, 1:]
        # Supplied randomness is masking metadata, shared with HF before execution.
        shuffle = noise.argsort(dim=1)
        restore = shuffle.argsort(dim=1)
        hidden = torch.gather(hidden, 1, shuffle[:, :self.keep_count, None].expand(-1, -1, hidden.shape[-1]))
        mask = torch.ones(batch, self.patch_count, device=hidden.device)
        mask[:, :self.keep_count] = 0
        mask = mask.gather(1, restore)
        cls = (self.cls_token + self.position_embeddings[:, :1]).expand(batch, -1, -1)
        hidden = torch.cat((cls, hidden), dim=1)
        for block in self.encoder:
            hidden = block(hidden)
        hidden = self.decoder_embed(self.layernorm(hidden))
        padding = self.mask_token.expand(batch, self.patch_count - self.keep_count, -1)
        patches = torch.cat((hidden[:, 1:], padding), dim=1)
        patches = torch.gather(patches, 1, restore.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        hidden = torch.cat((hidden[:, :1], patches), dim=1) + self.decoder_pos_embed
        for block in self.decoder:
            hidden = block(hidden)
        logits = self.decoder_pred(self.decoder_norm(hidden))[:, 1:]
        patch = self.patch_size
        target = pixel_values.reshape(batch, channels, height // patch, patch, width // patch, patch)
        target = target.permute(0, 2, 4, 3, 5, 1).reshape(batch, self.patch_count, patch * patch * channels)
        difference = logits - target
        square = self.product(torch.cat((difference, difference), dim=-1))
        per_patch = self.mean(square.unsqueeze(-1))
        # HF promotes per-patch errors to the FP32 mask dtype before its final sum.
        masked = self.product(torch.cat((per_patch.to(mask.dtype), mask), dim=-1))
        loss = self.mean(masked.reshape(1, 1, batch, self.patch_count)).reshape(())
        loss = loss * (self.patch_count / (self.patch_count - self.keep_count))
        return {"loss": loss, "logits": logits, "mask": mask, "ids_restore": restore}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.norm_pix_loss:
        raise ValueError("Preserve default exact GELU and unnormalized pixel reconstruction loss")
    if not 0 < config.mask_ratio < 1:
        raise ValueError("The selected workload must include kept and removed patches")
    return ViTMAEForPreTraining(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    def copy(dst, src):
        mapped[dst] = remaining.pop(src)
    for name in ("cls_token", "position_embeddings"):
        copy(name, "vit.embeddings." + name)
    for name in ("mask_token", "decoder_pos_embed"):
        copy(name, "decoder." + name)
    for dst, src in (("projection", "vit.embeddings.patch_embeddings.projection"),
                     ("layernorm", "vit.layernorm"), ("decoder_embed", "decoder.decoder_embed"),
                     ("decoder_norm", "decoder.decoder_norm"), ("decoder_pred", "decoder.decoder_pred")):
        for field in ("weight", "bias"):
            copy(dst + "." + field, src + "." + field)
    for target_stack, source_stack in (("encoder", "vit.encoder.layer"), ("decoder", "decoder.decoder_layers")):
        for i, block in enumerate(getattr(model, target_stack)):
            dst, src = f"{target_stack}.{i}.", f"{source_stack}.{i}."
            for field in (("weight", "bias") if config.qkv_bias else ("weight",)):
                mapped[dst + "attn.qkv." + field] = torch.cat([
                    remaining.pop(src + f"attention.attention.{name}.{field}") for name in ("query", "key", "value")
                ])
            for target, source in (("attn.proj", "attention.output.dense"), ("norm1", "layernorm_before"),
                                   ("norm2", "layernorm_after"), ("mlp.fc1", "intermediate.dense"),
                                   ("mlp.fc2", "output.dense")):
                for field in ("weight", "bias"):
                    copy(dst + target + "." + field, src + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped ViT-MAE state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
