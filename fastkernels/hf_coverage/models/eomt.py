"""EoMT's complete encoder and late mask-guided query refinement."""

import torch
from torch import nn

from .depth_anything import NonoverlappingTransposeConv
from .vit_msn import make_workloads
from ..patches.codec_top1 import CodecTop1
from ..patches.dinov3_rope import HFDINOv3RoPE
from ..patches.product_gate import ProductGate
from ..patches.linear import PostBiasLinear
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L2.eva_attention import EvaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp


class _Embeddings(nn.Module):
    def __init__(self, config, rotary=False):
        super().__init__()
        width = config.hidden_size
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, width))
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = Conv2d(config.num_channels, width, config.patch_size, stride=config.patch_size)
        if not rotary:
            self.position_embeddings = Embedding((config.image_size // config.patch_size) ** 2, width)
        self.num_prefix_tokens = 1 + config.num_register_tokens

    def forward(self, pixels):
        hidden = self.patch_embeddings.projection(pixels).flatten(2).transpose(1, 2)
        if hasattr(self, "position_embeddings"):
            hidden = hidden + self.position_embeddings(torch.arange(hidden.shape[1], device=pixels.device))[None]
        return torch.cat((self.cls_token.expand(pixels.shape[0], -1, -1),
                          self.register_tokens.expand(pixels.shape[0], -1, -1), hidden), dim=1)


class _LayerScale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.empty(width))
        self.product = ProductGate()

    def forward(self, hidden):
        return self.product(torch.cat((hidden, self.lambda1.expand_as(hidden)), dim=-1))


class _Layer(nn.Module):
    def __init__(self, config, rotary=False, prefix_tokens=1):
        super().__init__()
        width = config.hidden_size
        self.norm1, self.norm2 = (LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False) for _ in range(2))
        self.attention = EvaAttention(width, config.num_attention_heads, qkv_bias=True, qkv_fused=False,
                                      num_prefix_tokens=prefix_tokens)
        if rotary:
            for name, bias in (("q_proj", config.query_bias), ("k_proj", config.key_bias),
                               ("v_proj", config.value_bias), ("proj", config.proj_bias)):
                setattr(self.attention, name, Linear(width, width, bias=bias))
        self.layer_scale1, self.layer_scale2 = _LayerScale(width), _LayerScale(width)
        intermediate = config.intermediate_size if rotary else int(width * config.mlp_ratio)
        self.mlp = VitEncoderMlp(width, intermediate, width,
                                 bias=config.mlp_bias if rotary else True, act_approximate="none")

    def forward(self, hidden, mask=None, rope=None):
        hidden = hidden + self.layer_scale1(self.attention(self.norm1(hidden), attn_mask=mask, rope=rope))
        return hidden + self.layer_scale2(self.mlp(self.norm2(hidden)))


class _ScaleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.conv1 = NonoverlappingTransposeConv(width, width, 2)
        # CUDA's transpose convolution rounds its product before adding bias.
        self.conv1.projection = PostBiasLinear(width, width * 4)
        self.activation = GELU(approximate="none")
        self.conv2 = Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.layernorm2d = LayerNorm(width, eps=1e-6, promote_fp32=False)

    def forward(self, hidden):
        hidden = self.conv2(self.activation(self.conv1(hidden)))
        return self.layernorm2d(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _MaskHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = (Linear(width, width) for _ in range(3))
        self.activation = GELU(approximate="none")

    def forward(self, hidden):
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(hidden)))))


class _Eomt(nn.Module):
    def __init__(self, config, rotary=False):
        super().__init__()
        self.config, self.rotary = config, rotary
        self.embeddings = _Embeddings(config, rotary)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.query = Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([
            _Layer(config, rotary, 1 + config.num_register_tokens +
                   (config.num_queries if index >= config.num_hidden_layers - config.num_blocks else 0))
            for index in range(config.num_hidden_layers)])
        if rotary:
            self.rope = HFDINOv3RoPE(config.hidden_size // config.num_attention_heads,
                                     config.rope_parameters["rope_theta"])
        self.upscale_block = nn.Module()
        self.upscale_block.block = nn.ModuleList([_ScaleLayer(config) for _ in range(config.num_upscale_blocks)])
        self.mask_head, self.class_predictor = _MaskHead(config.hidden_size), Linear(config.hidden_size, len(config.id2label) + 1)
        self.register_buffer("attn_mask_probs", torch.ones(config.num_blocks))
        # This fixed loss weight is serialized by HF but unused in no-label inference.
        self.criterion = nn.Module()
        self.criterion.register_buffer("empty_weight", torch.empty(len(config.id2label) + 1))
        self.bmm, self.interpolate, self.compare = BMM(), Interpolate(), CodecTop1()
        self.grid_size = (config.image_size // config.patch_size,) * 2

    def predict(self, hidden):
        queries = hidden[:, :self.config.num_queries]
        classes = self.class_predictor(queries)
        patches = hidden[:, self.config.num_queries + self.embeddings.num_prefix_tokens:]
        patches = patches.transpose(1, 2).reshape(hidden.shape[0], -1, *self.grid_size)
        for block in self.upscale_block.block:
            patches = block(patches)
        masks = self.bmm(self.mask_head(queries), patches.flatten(2))
        return masks.reshape(hidden.shape[0], self.config.num_queries, *patches.shape[-2:]), classes

    def forward(self, pixel_values):
        hidden, attention_mask = self.embeddings(pixel_values), None
        rope = self.rope.get_embed(self.grid_size).to(hidden.dtype) if self.rotary else None
        start = self.config.num_hidden_layers - self.config.num_blocks
        queries = self.config.num_queries
        for index, layer in enumerate(self.layers):
            if index == start:
                hidden = torch.cat((self.query.emb.weight[None].expand(hidden.shape[0], -1, -1), hidden), dim=1)
            if index >= start and self.attn_mask_probs[index - start] > 0:
                masks, _ = self.predict(self.layernorm(hidden))
                logits = self.interpolate(masks, size=self.grid_size, mode="bilinear", align_corners=False).flatten(2)
                # First-index argmax yields 1 exactly when a mask logit is > 0.
                selected = self.compare(torch.stack((torch.zeros_like(logits), logits), dim=-1)).bool()
                attention_mask = torch.ones(hidden.shape[:2] + (hidden.shape[1],), device=hidden.device, dtype=torch.bool)
                prefix = queries + self.embeddings.num_prefix_tokens
                attention_mask[:, :queries, prefix:] = selected
                probability = self.attn_mask_probs[index - start]
                if probability < 1:
                    disabled = torch.rand(hidden.shape[0], queries, device=hidden.device) > probability
                    attention_mask[:, :queries, prefix:][disabled] = True
                attention_mask = attention_mask[:, None].expand(-1, self.config.num_attention_heads, -1, -1)
                # Both native variants use 1 on allowed positions, not zero.
                mask_dtype = hidden.dtype if self.rotary else torch.float32
                masked_value = torch.finfo(hidden.dtype).min if self.rotary else -1e9
                attention_mask = attention_mask.to(mask_dtype).masked_fill(~attention_mask, masked_value)
            hidden = layer(hidden, attention_mask, rope)
        hidden = self.layernorm(hidden)
        masks, classes = self.predict(hidden)
        return {"masks_queries_logits": masks, "class_queries_logits": classes, "last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.use_swiglu_ffn:
        raise ValueError("This case preserves the published GELU EoMT path")
    return _Eomt(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for key, value in state_dict.items():
        key = key.replace("query.weight", "query.emb.weight").replace("position_embeddings.weight", "position_embeddings.emb.weight")
        key = key.replace(".attention.out_proj.", ".attention.proj.")
        if model.rotary:
            key = key.replace(".attention.o_proj.", ".attention.proj.")
            key = key.replace(".mlp.up_proj.", ".mlp.fc1.").replace(".mlp.down_proj.", ".mlp.fc2.")
            key = key.replace("embeddings.patch_embeddings.", "embeddings.patch_embeddings.projection.")
        if ".conv1.weight" in key:
            value = value.permute(1, 2, 3, 0).reshape(-1, value.shape[0]).contiguous()
            key = key.replace(".conv1.weight", ".conv1.projection.weight")
        elif ".conv1.bias" in key:
            value = value.repeat_interleave(4)
            key = key.replace(".conv1.bias", ".conv1.projection.bias")
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
