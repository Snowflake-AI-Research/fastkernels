"""SAM image/point mask prediction using existing SAM3 decoder components."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.vitmatte import SpatialRelativeAttention
from fastkernels.hf_coverage.patches.sam_position_dtype import SamPositionDtype
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.linear import BMM
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.sam3_prompt_encoder import Sam3PromptEncoder
from fastkernels.tasks.baseline.L2.sam3_vit_mlp import Sam3ViTMLP
from fastkernels.tasks.baseline.L3.sam3_mask_decoder import Sam3MaskDecoder, TwoWayTransformer
from fastkernels.tasks.baseline.L3.sam3_vit_block import _window_partition, _window_unpartition
from fastkernels.tasks.baseline.L3.sam3_rope_attention import Sam3Attention


class SamAttentionCore(nn.Module):
    """Native eager attention with explicit FP32 softmax and BF16 storage."""

    def __init__(self):
        super().__init__()
        self.matmul, self.softmax = BMM(), Softmax()

    def forward(self, query, key, value):
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        scores = self.matmul(query, key.transpose(-1, -2)) * query.shape[-1] ** -0.5
        probabilities = self.softmax(scores.float()).to(query.dtype)
        return self.matmul(probabilities, value).transpose(1, 2)


class DecoderAttention(Sam3Attention):
    """Existing projections and BMM/softmax retain native eager BF16 rounding."""

    def __init__(self, width, heads, downsample_rate=1):
        super().__init__(width, heads, downsample_rate=downsample_rate)
        self.attend = SamAttentionCore()

    def forward(self, q, k, v):
        batch = q.shape[0]
        width = self.internal_dim // self.num_heads
        query = self.q_proj(q).reshape(batch, -1, self.num_heads, width)
        key = self.k_proj(k).reshape(batch, -1, self.num_heads, width)
        value = self.v_proj(v).reshape(batch, -1, self.num_heads, width)
        return self.out_proj(self.attend(query, key, value).reshape(batch, q.shape[1], -1))


def retain_eager_attention(transformer):
    for module in transformer.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, Sam3Attention):
                setattr(module, name, DecoderAttention(
                    child.embedding_dim, child.num_heads, child.embedding_dim // child.internal_dim,
                ))


class SamRelativeAttention(SpatialRelativeAttention):
    """Existing relative projections with native eager or SDPA rounding."""

    def forward(self, x):
        batch, h, w, width = x.shape
        qkv = self.qkv(x).reshape(batch, h*w, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, batch*self.heads, h*w, self.head_dim).unbind(0)
        query = q.reshape(batch*self.heads, h, w, self.head_dim)
        rh, rw = self.relative_table(self.rel_pos_h, h), self.relative_table(self.rel_pos_w, w)
        rel_h = self.bmm(query.permute(1, 0, 2, 3).reshape(h, -1, self.head_dim), rh.transpose(1, 2))
        rel_h = rel_h.reshape(h, batch*self.heads, w, h).permute(1, 0, 2, 3)
        rel_w = self.bmm(query.permute(2, 0, 1, 3).reshape(w, -1, self.head_dim), rw.transpose(1, 2))
        rel_w = rel_w.reshape(w, batch*self.heads, h, w).permute(1, 2, 0, 3)
        relative = rel_h[..., None] + rel_w[..., None, :]
        if hasattr(self, "dense"):
            query, key, value = (
                tensor.reshape(batch, self.heads, h*w, self.head_dim).transpose(1, 2)
                for tensor in (q, k, v)
            )
            context = self.dense(query, key, value,
                                 attn_mask=relative.reshape(batch, self.heads, h*w, h*w))
            return self.proj(context.reshape(batch, h, w, width))
        scores = self.bmm(q * self.head_dim**-0.5, k.transpose(-1, -2))
        probabilities = self.softmax((scores + relative.reshape_as(scores)).float()).to(q.dtype)
        context = self.bmm(probabilities, v).reshape(batch, self.heads, h, w, self.head_dim)
        return self.proj(context.permute(0, 2, 3, 1, 4).reshape(batch, h, w, width))


class VisionLayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.window = 0 if index in c.global_attn_indexes else c.window_size
        side = self.window or c.image_size//c.patch_size
        self.layer_norm1 = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.layer_norm2 = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.attn = SamRelativeAttention(c.hidden_size, c.num_attention_heads, (side, side), c.qkv_bias)
        self.mlp = Sam3ViTMLP(c.hidden_size, c.mlp_dim)

    def forward(self, x):
        normalized = self.layer_norm1(x)
        if self.window:
            windows, padded = _window_partition(normalized, self.window)
            attended = _window_unpartition(self.attn(windows), self.window, padded, x.shape[1:3])
        else:
            attended = self.attn(normalized)
        x = x + attended
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.projection = Conv2d(c.num_channels, c.hidden_size, c.patch_size, stride=c.patch_size)
        side = c.image_size//c.patch_size
        self.pos_embed = nn.Parameter(torch.empty(1, side, side, c.hidden_size))
        self.layers = nn.ModuleList([VisionLayer(c, i) for i in range(c.num_hidden_layers)])
        self.global_indexes = c.global_attn_indexes
        self.neck = nn.Sequential(Conv2d(c.hidden_size, c.output_channels, 1, bias=False),
                                  LayerNorm2d(c.output_channels),
                                  Conv2d(c.output_channels, c.output_channels, 3, padding=1, bias=False),
                                  LayerNorm2d(c.output_channels))

    def forward(self, pixels):
        hidden = self.patch_embed.projection(pixels).permute(0, 2, 3, 1) + self.pos_embed
        global_features = []
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index in self.global_indexes:
                global_features.append(hidden)
        return self.neck(hidden.permute(0, 3, 1, 2)), global_features


def prompt_encoder(c):
    p = c.prompt_encoder_config
    encoder = Sam3PromptEncoder(p.hidden_size, (p.image_embedding_size,)*2, (p.image_size,)*2,
                               p.mask_input_channels)
    encoder.pe_layer = SamPositionDtype(p.hidden_size//2)
    encoder.mask_downscaling[1] = LayerNorm2d(p.mask_input_channels//4, eps=p.layer_norm_eps)
    encoder.mask_downscaling[4] = LayerNorm2d(p.mask_input_channels, eps=p.layer_norm_eps)
    return encoder


def mask_decoder(c):
    d = c.mask_decoder_config
    transformer = TwoWayTransformer(d.num_hidden_layers, d.hidden_size, d.num_attention_heads,
                                    d.mlp_dim, attention_downsample_rate=d.attention_downsample_rate)
    retain_eager_attention(transformer)
    for module in transformer.modules():
        if isinstance(module, nn.LayerNorm):
            module.eps = d.layer_norm_eps
    decoder = Sam3MaskDecoder(transformer_dim=d.hidden_size, transformer=transformer,
                             num_multimask_outputs=d.num_multimask_outputs,
                             iou_head_depth=d.iou_head_depth, iou_head_hidden_dim=d.iou_head_hidden_dim)
    decoder.output_upscaling[1] = LayerNorm2d(d.hidden_size//4)
    return decoder


class SamModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.vision_encoder = VisionEncoder(c.vision_config)
        self.prompt_encoder = prompt_encoder(c)
        self.mask_decoder = mask_decoder(c)

    def forward(self, pixel_values, input_points, input_labels=None):
        image, _ = self.vision_encoder(pixel_values)
        batch, point_batch, count, _ = input_points.shape
        if input_labels is None:
            input_labels = torch.ones((batch, point_batch, count), device=image.device, dtype=torch.long)
        points = input_points.reshape(batch*point_batch, count, 2)
        labels = input_labels.reshape(batch*point_batch, count)
        sparse, dense = self.prompt_encoder(points=(points, labels), boxes=None, masks=None)
        # The reused prompt assembler concatenates a default-FP32 empty tensor.
        # Its actual embeddings already have native BF16 values; restore dtype.
        sparse = sparse.to(image.dtype)
        masks, scores, _, _ = self.mask_decoder(image.repeat_interleave(point_batch, dim=0),
                                               self.prompt_encoder.get_dense_pe(), sparse, dense,
                                               multimask_output=True, repeat_image=False)
        return {"pred_masks": masks.reshape(batch, point_batch, *masks.shape[1:]),
                "iou_scores": scores.reshape(batch, point_batch, -1)}


def build_from_config(config, device, dtype):
    if not config.vision_config.use_rel_pos or not config.vision_config.use_abs_pos:
        raise ValueError("The declared SAM example enables both relative and absolute positions")
    model = SamModel(config)
    # The selected image task uses native SDPA. Restrict this configuration to
    # the image entry point; shared decoder defaults also serve video models.
    for layer in model.vision_encoder.layers:
        layer.attn.dense = DenseAttention(backend="sdpa")
    for module in model.mask_decoder.modules():
        if hasattr(module, "attend") and isinstance(module.attend, SamAttentionCore):
            module.attend = DenseAttention(backend="sdpa")
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name, value in model.state_dict().items():
        source = name
        if source.startswith("vision_encoder."):
            source = source.replace(".mlp.fc1.", ".mlp.lin1.").replace(".mlp.fc2.", ".mlp.lin2.")
            for index, native in ((0, "conv1"), (1, "layer_norm1"), (2, "conv2"), (3, "layer_norm2")):
                source = source.replace(f".neck.{index}.", f".neck.{native}.")
        elif source.startswith("prompt_encoder."):
            source = source.replace("point_embeddings.", "point_embed.")
            source = source.replace("pe_layer.positional_encoding_gaussian_matrix", "shared_embedding.positional_embedding")
            for index, native in ((0, "conv1"), (1, "layer_norm1"), (3, "conv2"), (4, "layer_norm2"), (6, "conv3")):
                source = source.replace(f"mask_downscaling.{index}.", f"mask_embed.{native}.")
        else:
            for index, native in ((0, "upscale_conv1"), (1, "upscale_layer_norm"), (3, "upscale_conv2")):
                source = source.replace(f"output_upscaling.{index}.", native+".")
            source = source.replace(".norm_final_attn.", ".layer_norm_final_attn.")
            for index in range(1, 5):
                source = source.replace(f".norm{index}.", f".layer_norm{index}.")
            if ".output_hypernetworks_mlps." in source or ".iou_prediction_head." in source:
                count = config.mask_decoder_config.iou_head_depth if ".iou_prediction_head." in source else 3
                for index in range(count):
                    native = "proj_in" if index == 0 else "proj_out" if index == count-1 else f"layers.{index-1}"
                    source = source.replace(f".layers.{index}.", f".{native}.")
        mapped[name], used = state_dict[source], used | {source}
    shared = "shared_image_embedding.positional_embedding"
    if shared in state_dict:
        if not torch.equal(state_dict[shared], state_dict["prompt_encoder.shared_embedding.positional_embedding"]):
            raise ValueError("Native SAM aliases must carry the same shared position matrix")
        used.add(shared)
    if used != set(state_dict):
        raise ValueError(f"Unmapped SAM state: {sorted(set(state_dict)-used)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
