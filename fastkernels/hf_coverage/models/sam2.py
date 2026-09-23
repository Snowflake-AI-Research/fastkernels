"""SAM2 image prediction with Hiera pooling and existing SAM3 mask components."""

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.hf_coverage.models.sam import SamAttentionCore, prompt_encoder, retain_eager_attention
from fastkernels.hf_coverage.patches.sam_sine_dtype import SamSineDtype
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.sam3_prompt_encoder import PositionEmbeddingRandom
from fastkernels.tasks.baseline.L3.sam3_mask_decoder import Sam3MaskDecoder, TwoWayTransformer
from fastkernels.tasks.baseline.L3.sam3_vit_block import _window_partition, _window_unpartition


class Sam2PointPrompt(nn.Module):
    """Reuse prompt modules while retaining native coordinate metadata precision."""

    def __init__(self, c):
        super().__init__()
        assembled = prompt_encoder(c)
        self.pe_layer = PositionEmbeddingRandom(c.prompt_encoder_config.hidden_size // 2)
        self.point_embeddings = assembled.point_embeddings
        self.not_a_point_embed = assembled.not_a_point_embed
        self.no_mask_embed = assembled.no_mask_embed
        self.mask_downscaling = assembled.mask_downscaling
        self.image_embedding_size = assembled.image_embedding_size
        self.image_size = c.prompt_encoder_config.image_size

    def forward(self, points, labels):
        coords = F.pad(points + 0.5, (0, 0, 0, 1)) / self.image_size
        labels = F.pad(labels, (0, 1), value=-1)
        encoded = self.pe_layer._pe_encoding(coords)
        encoded = torch.where((labels == -1)[..., None], self.not_a_point_embed.weight, encoded)
        encoded = encoded.masked_fill((labels == -10)[..., None], 0)
        for label, embedding in enumerate(self.point_embeddings):
            encoded = torch.where((labels == label)[..., None], encoded + embedding.weight, encoded)
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return encoded, dense.expand(points.shape[0], -1, *self.image_embedding_size)


def dense_position(encoder, size):
    """Fixed image-grid metadata in the native position buffer's dtype."""
    matrix = encoder.positional_encoding_gaussian_matrix
    h, w = size
    y = (torch.arange(1, h + 1, device=matrix.device, dtype=matrix.dtype) - 0.5) / h
    x = (torch.arange(1, w + 1, device=matrix.device, dtype=matrix.dtype) - 0.5) / w
    coords = torch.stack((x[None, :].expand(h, w), y[:, None].expand(h, w)), dim=-1)
    return encoder._pe_encoding(coords).permute(2, 0, 1)[None]


class HieraAttention(nn.Module):
    def __init__(self, inputs, outputs, heads, stride):
        super().__init__()
        self.heads, self.width, self.stride = heads, outputs//heads, stride
        self.qkv, self.proj = Linear(inputs, outputs*3), Linear(outputs, outputs)
        self.attend = SamAttentionCore()
        self.pool = MaxPool2d(tuple(stride)) if stride else nn.Identity()
        self.bmm, self.softmax = BMM(), Softmax()

    def forward(self, x):
        batch, h, w, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, h*w, 3, self.heads, self.width).unbind(2)
        # Pinned HF also computes and discards this across-head attention. Retain
        # its actual work while comparing the same native execution path.
        self.softmax(self.bmm(q*self.width**-0.5, k.transpose(-1, -2)).float()).to(q.dtype)
        if self.stride:
            q = self.pool(q.reshape(batch, h, w, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            h, w = q.shape[1:3]
            q = q.reshape(batch, h*w, self.heads, self.width)
        return self.proj(self.attend(q, k, v).reshape(batch, h, w, -1))


class HieraBlock(nn.Module):
    def __init__(self, c, stage, block, total):
        super().__init__()
        inputs = c.embed_dim_per_stage[stage-1] if stage>0 and block==0 else c.embed_dim_per_stage[stage]
        width = c.embed_dim_per_stage[stage]
        window_stage = stage-1 if stage>0 and block==0 else stage
        self.window = 0 if total in c.global_attention_blocks else c.window_size_per_stage[window_stage]
        self.stride = c.query_stride if 0<stage<=c.num_query_pool_stages and block==0 else None
        self.layer_norm1 = LayerNorm(inputs, eps=c.layer_norm_eps, promote_fp32=False)
        self.layer_norm2 = LayerNorm(width, eps=c.layer_norm_eps, promote_fp32=False)
        self.attn = HieraAttention(inputs, width, c.num_attention_heads_per_stage[stage], self.stride)
        self.proj = Linear(inputs, width) if inputs != width else nn.Identity()
        self.pool = MaxPool2d(tuple(self.stride)) if self.stride else nn.Identity()
        self.mlp = nn.Module()
        self.mlp.proj_in, self.mlp.proj_out = Linear(width, int(width*c.mlp_ratio)), Linear(int(width*c.mlp_ratio), width)
        self.activation = GELU()

    def forward(self, x):
        normalized = self.layer_norm1(x)
        residual = x
        if self.stride:
            residual = self.pool(self.proj(normalized).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.window:
            windows, padded = _window_partition(normalized, self.window)
            attended = self.attn(windows)
            window = self.window//self.stride[0] if self.stride else self.window
            h, w = residual.shape[1:3]
            if self.stride:
                padded = (h+(-h)%window, w+(-w)%window)
            attended = _window_unpartition(attended, window, padded, (h,w))
        else:
            attended = self.attn(normalized)
        x = residual + attended
        return x + self.mlp.proj_out(self.activation(self.mlp.proj_in(self.layer_norm2(x))))


class HieraBackbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.projection = Conv2d(c.num_channels, c.hidden_size, tuple(c.patch_kernel_size),
                                            stride=tuple(c.patch_stride), padding=tuple(c.patch_padding))
        self.pos_embed = nn.Parameter(torch.empty(1, c.hidden_size, *c.window_positional_embedding_background_size))
        self.pos_embed_window = nn.Parameter(torch.empty(1,c.hidden_size,c.window_size_per_stage[0],c.window_size_per_stage[0]))
        self.blocks, self.stage_ends = nn.ModuleList(), []
        for stage, count in enumerate(c.blocks_per_stage):
            for block in range(count):
                self.blocks.append(HieraBlock(c, stage, block, len(self.blocks)))
            self.stage_ends.append(len(self.blocks)-1)
        self.resize = Interpolate()

    def forward(self, pixels):
        x = self.patch_embed.projection(pixels).permute(0,2,3,1)
        position = self.resize(self.pos_embed, size=x.shape[1:3], mode="bicubic", align_corners=False)
        tiled = self.pos_embed_window.tile([a//b for a,b in zip(position.shape,self.pos_embed_window.shape)])
        x = x + (position+tiled).permute(0,2,3,1)
        stages = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in self.stage_ends:
                stages.append(x)
        return stages


class VisionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.backbone = HieraBackbone(c.backbone_config)
        self.neck = nn.Module()
        self.neck.convs = nn.ModuleList([Conv2d(width,c.fpn_hidden_size,c.fpn_kernel_size,
                                              stride=c.fpn_stride,padding=c.fpn_padding)
                                        for width in c.backbone_channel_list])
        self.position, self.resize = SamSineDtype(c.fpn_hidden_size), Interpolate()
        self.top_down, self.levels = c.fpn_top_down_levels, c.num_feature_levels

    def forward(self, pixels):
        stages, features, positions = self.backbone(pixels), [], []
        for i in range(len(stages)-1,-1,-1):
            lateral = self.neck.convs[len(stages)-1-i](stages[i].permute(0,3,1,2))
            value = lateral
            if i in self.top_down and i != len(stages)-1:
                value = lateral + self.resize(features[-1].float(),scale_factor=2.,mode="nearest").to(lateral.dtype)
            features.append(value)
            positions.append(self.position(value).to(value.dtype))
        return features[-self.levels:][::-1], positions[-self.levels:][::-1]


class Sam2Model(nn.Module):
    def __init__(self, c, vision_encoder=VisionEncoder):
        super().__init__()
        self.config = c
        self.vision_encoder = vision_encoder(c.vision_config)
        p = c.prompt_encoder_config
        p.image_embedding_size = p.image_size//p.patch_size
        self.prompt_encoder = Sam2PointPrompt(c)
        self.shared_image_embedding = PositionEmbeddingRandom(p.hidden_size//2)
        d = c.mask_decoder_config
        transformer = TwoWayTransformer(d.num_hidden_layers,d.hidden_size,d.num_attention_heads,d.mlp_dim,
                                        activation=ReLU,attention_downsample_rate=d.attention_downsample_rate)
        retain_eager_attention(transformer)
        self.mask_decoder = Sam3MaskDecoder(transformer_dim=d.hidden_size,transformer=transformer,
                                            num_multimask_outputs=d.num_multimask_outputs,
                                            iou_head_depth=d.iou_head_depth,iou_head_hidden_dim=d.iou_head_hidden_dim,
                                            use_high_res_features=True,iou_prediction_use_sigmoid=True,
                                            pred_obj_scores=True,pred_obj_scores_mlp=True,use_multimask_token_for_obj_ptr=True)
        self.mask_decoder.output_upscaling[1] = LayerNorm2d(d.hidden_size//4)
        self.no_memory_embedding = nn.Parameter(torch.empty(1,1,c.vision_config.fpn_hidden_size))

    def forward(self, pixel_values, input_points, input_labels=None):
        features, _ = self.vision_encoder(pixel_values)
        features[0] = self.mask_decoder.conv_s0(features[0])
        features[1] = self.mask_decoder.conv_s1(features[1])
        features[-1] = features[-1] + self.no_memory_embedding.reshape(1, -1, 1, 1)
        batch, point_batch, count, _ = input_points.shape
        labels = input_labels
        if labels is None:
            labels = torch.ones(input_points.shape[:-1], device=input_points.device, dtype=torch.long)
        sparse, dense = self.prompt_encoder(input_points.reshape(-1, count, 2), labels.reshape(-1, count))
        pe = dense_position(self.shared_image_embedding, self.prompt_encoder.image_embedding_size)
        masks, scores, _, object_scores = self.mask_decoder(
            features[-1].repeat_interleave(point_batch, 0), pe, sparse, dense,
            multimask_output=True, repeat_image=False,
            high_res_features=[x.repeat_interleave(point_batch, 0) for x in features[:-1]],
        )
        outputs = {
            "pred_masks": masks.reshape(batch, point_batch, *masks.shape[1:]),
            "iou_scores": scores.reshape(batch, point_batch, -1),
            "object_score_logits": object_scores.reshape(batch, point_batch, -1),
        }
        outputs.update({f"image_embeddings.{i}": value for i, value in enumerate(features)})
        return outputs


def build_from_config(config, device, dtype):
    model = Sam2Model(config)
    # Match the selected native image task's SDPA rounding. Other task builders
    # select their attention backend independently.
    for module in model.modules():
        if hasattr(module, "attend") and isinstance(module.attend, SamAttentionCore):
            module.attend = DenseAttention(backend="sdpa")
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for name in model.state_dict():
        source = name
        if source.startswith("shared_image_embedding."):
            source = "shared_image_embedding.positional_embedding"
        elif source.startswith("prompt_encoder."):
            if ".point_embeddings." in source:
                index = int(source.split(".point_embeddings.")[1].split('.')[0])
                source = "prompt_encoder.point_embed.weight"
                mapped[name] = state_dict[source][index:index + 1]
                used.add(source)
                continue
            source = source.replace("pe_layer.positional_encoding_gaussian_matrix", "shared_embedding.positional_embedding")
            for index, native in ((0, "conv1"), (1, "layer_norm1"), (3, "conv2"), (4, "layer_norm2"), (6, "conv3")):
                source = source.replace(f"mask_downscaling.{index}.", f"mask_embed.{native}.")
        elif source.startswith("mask_decoder."):
            source = source.replace('.out_proj.', '.o_proj.')
            for index, native in ((0, "upscale_conv1"), (1, "upscale_layer_norm"), (3, "upscale_conv2")):
                source = source.replace(f"output_upscaling.{index}.", native + ".")
            source = source.replace(".norm_final_attn.", ".layer_norm_final_attn.")
            for index in range(1, 5):
                source = source.replace(f".norm{index}.", f".layer_norm{index}.")
            source = source.replace('.mlp.lin1.', '.mlp.proj_in.').replace('.mlp.lin2.', '.mlp.proj_out.')
            if any(x in source for x in (".output_hypernetworks_mlps.", ".iou_prediction_head.", ".pred_obj_score_head.")):
                count = config.mask_decoder_config.iou_head_depth if ".iou_prediction_head." in source else 3
                for index in range(count):
                    native = "proj_in" if index == 0 else "proj_out" if index == count - 1 else f"layers.{index - 1}"
                    source = source.replace(f".layers.{index}.", f".{native}.")
        mapped[name] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f"Unmapped SAM2 state: {sorted(set(state_dict) - used)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
