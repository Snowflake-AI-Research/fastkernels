"""SAM3 LiteText ordinary text-prompt segmentation from SAM3 components."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.models.sam3_lite_text_encoder import Sam3LiteTextTextModel
from fastkernels.hf_coverage.patches.sam3_decoder_dtype import Sam3DecoderDtype, Sam3ScoringDtype
from fastkernels.hf_coverage.patches.sam_sine_dtype import SamSineDtype
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.sam3_cross_attention import Sam3CrossAttention
from fastkernels.tasks.baseline.L2.sam3_fpn_conv import Sam3FPNConvStage
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.sam3_encoder_layer import Sam3EncoderLayer
from fastkernels.tasks.baseline.L3.sam3_vit_block import _window_partition, _window_unpartition
from fastkernels.tasks.baseline.L4.sam3 import Sam3Config, Sam3MLP, Sam3SegmentationHead


class VisionAttention(nn.Module):
    def __init__(self, c, side):
        super().__init__()
        self.heads, self.dim = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (Linear(c.hidden_size, c.hidden_size) for _ in range(4))
        self.attend = DenseAttention(backend="sdpa")
        self.side, self.scale, self.theta = side, c.window_size / side, c.rope_theta
        self.register_buffer("rotary_table", self.position_table("cpu"), persistent=False)

    def position_table(self, device):
        frequency = 1.0 / (self.theta ** (torch.arange(0, self.dim, 4, device="cpu").float() / self.dim))
        indices = torch.arange(self.side * self.side, device="cpu")
        x = (indices % self.side).float() * self.scale
        y = torch.div(indices, self.side, rounding_mode="floor").float() * self.scale
        angles = torch.cat((x[:, None] * frequency, y[:, None] * frequency), dim=-1)
        return torch.cat((angles.cos(), angles.sin()), dim=-1).to(device)

    def rotate(self, x):
        positions = torch.arange(self.side * self.side, device=x.device).repeat(x.shape[0])
        flat = x.float().reshape(positions.numel(), -1)
        rotated, _ = RotaryEmbedding.forward_native_interleaved(positions, flat, flat, self.dim, self.rotary_table)
        return rotated.reshape_as(x).to(x.dtype)

    def forward(self, x):
        batch, height, width, _ = x.shape
        shape = (batch, height * width, self.heads, self.dim)
        q, k = self.rotate(self.q_proj(x)).reshape(shape), self.rotate(self.k_proj(x)).reshape(shape)
        v = self.v_proj(x).reshape(shape)
        return self.o_proj(self.attend(q, k, v).reshape(batch, height, width, -1).contiguous())


class VisionLayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.window = 0 if index in c.global_attn_indexes else c.window_size
        self.layer_norm1, self.layer_norm2 = (LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False) for _ in range(2))
        self.attention = VisionAttention(c, self.window or c.image_size // c.patch_size)
        self.mlp = VitEncoderMlp(c.hidden_size, c.intermediate_size, c.hidden_size, act_approximate="none")

    def forward(self, x):
        residual, x = x, self.layer_norm1(x)
        if self.window:
            height, width = x.shape[1:3]
            x, padded = _window_partition(x, self.window)
        x = self.attention(x)
        if self.window:
            x = _window_unpartition(x, self.window, padded, (height, width))
        x = residual + x
        return x + self.mlp(self.layer_norm2(x))


class VisionEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        side = c.pretrain_image_size // c.patch_size
        self.position_embeddings = nn.Parameter(torch.empty(1, side * side, c.hidden_size))
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = Conv2d(c.num_channels, c.hidden_size, c.patch_size, stride=c.patch_size, bias=False)

    def forward(self, x):
        x = self.patch_embeddings.projection(x).permute(0, 2, 3, 1)
        height, width = x.shape[1:3]
        side = math.isqrt(self.position_embeddings.shape[1])
        position = self.position_embeddings.reshape(1, side, side, -1)
        position = position.repeat(1, height // side + 1, width // side + 1, 1)[:, :height, :width]
        return x + position


class VisionBackbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = VisionEmbeddings(c)
        self.layer_norm = LayerNorm(c.hidden_size, eps=c.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList([VisionLayer(c, index) for index in range(c.num_hidden_layers)])

    def forward(self, x):
        x = self.layer_norm(self.embeddings(x))
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class FPNLayer(nn.Module):
    def __init__(self, c, scale):
        super().__init__()
        children = list(Sam3FPNConvStage(c.backbone_config.hidden_size, c.fpn_hidden_size, scale).conv.children())
        self.scale_layers = nn.ModuleList(children[:-2])
        self.proj1, self.proj2 = children[-2:]

    def forward(self, x):
        for layer in self.scale_layers:
            x = layer(x)
        return self.proj2(self.proj1(x))


class VisionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.backbone = VisionBackbone(c.backbone_config)
        self.neck = nn.Module()
        self.neck.fpn_layers = nn.ModuleList([FPNLayer(c, scale) for scale in c.scale_factors])
        self.neck.position_encoding = SamSineDtype(c.fpn_hidden_size)

    def forward(self, pixels):
        features, positions = [], []
        hidden = self.backbone(pixels)
        for layer in self.neck.fpn_layers:
            feature = layer(hidden)
            features.append(feature)
            positions.append(self.neck.position_encoding(feature))
        return features, positions


class GeometryState(nn.Module):
    """Complete serialized box-prompt state; inactive for the declared text task."""

    def __init__(self, c):
        super().__init__()
        width = c.hidden_size
        self.label_embed, self.cls_embed = Embedding(2, width), Embedding(1, width)
        self.boxes_direct_project = Linear(4, width)
        self.boxes_pool_project = Conv2d(width, width, c.roi_size)
        self.boxes_pos_enc_project = Linear(width + 2, width)
        self.final_proj = Linear(width, width)
        for name in ("vision_layer_norm", "prompt_layer_norm", "output_layer_norm"):
            setattr(self, name, LayerNorm(width, promote_fp32=False))
        self.layers = nn.ModuleList([Sam3EncoderLayer(width, c.num_attention_heads, c.intermediate_size) for _ in range(c.num_layers)])


class NCHWConv(Conv2d):
    def forward(self, x):
        # The existing pixel decoder adds the skip first; native HF adds its
        # resized NCHW map first. Match the resulting native convolution layout
        # explicitly, including the copy in forward timing.
        return super().forward(x.contiguous())


class Sam3LiteTextModel(nn.Module):
    def __init__(self, c, text_encoder=Sam3LiteTextTextModel):
        super().__init__()
        self.config = c
        self.vision_encoder = VisionEncoder(c.vision_config)
        self.text_encoder = text_encoder(c.text_config)
        self.text_projection = Linear(c.text_config.hidden_size, c.detr_encoder_config.hidden_size)
        self.geometry_encoder = GeometryState(c.geometry_encoder_config)
        e, d, m = c.detr_encoder_config, c.detr_decoder_config, c.mask_decoder_config
        self.detr_encoder = nn.Module()
        self.detr_encoder.layers = nn.ModuleList([Sam3EncoderLayer(e.hidden_size, e.num_attention_heads, e.intermediate_size) for _ in range(e.num_layers)])
        shared = Sam3Config(d_model=d.hidden_size, decoder_layers=d.num_layers,
                           decoder_dim_feedforward=d.intermediate_size, decoder_n_head=d.num_attention_heads,
                           num_queries=d.num_queries, encoder_n_head=m.num_attention_heads,
                           pixel_decoder_upsampling_stages=m.num_upsampling_stages)
        self.detr_decoder = Sam3DecoderDtype(shared)
        self.mask_decoder = Sam3SegmentationHead(shared)
        self.mask_decoder.pixel_decoder.conv_layers = nn.ModuleList([
            NCHWConv(m.hidden_size, m.hidden_size, 3, padding=1)
            for _ in range(m.num_upsampling_stages)])
        prompt_mlp = Sam3MLP(d.hidden_size, d.intermediate_size, d.hidden_size, 2,
                            residual=True, out_norm=LayerNorm(d.hidden_size, promote_fp32=False))
        self.dot_product_scoring = Sam3ScoringDtype(d.hidden_size, d.hidden_size, prompt_mlp=prompt_mlp)
        self.sigmoid = Sigmoid()
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False
            if isinstance(module, Sam3CrossAttention):
                module.attn = DenseAttention(backend="sdpa")

    def forward(self, pixel_values, input_ids, attention_mask=None):
        all_features, all_positions = self.vision_encoder(pixel_values)
        features, positions = all_features[:-1], all_positions[:-1]
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text = self.text_projection(text_outputs.last_hidden_state)
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        padding = ~valid
        vision = torch.cat([features[-1].flatten(2).transpose(1, 2)], dim=1)
        position = torch.cat([positions[-1].flatten(2).transpose(1, 2)], dim=1)
        for layer in self.detr_encoder.layers:
            vision = layer(vision, text, query_pos=position, memory_key_padding_mask=padding)
        hs, references, presence, _ = self.detr_decoder(
            vision.transpose(0, 1), position.transpose(0, 1), features[-1].shape[-2:],
            memory_text=text.transpose(0, 1), text_attention_mask=padding)
        offsets = self.detr_decoder.bbox_embed(hs)
        boxes = self.sigmoid(self.detr_decoder._inverse_sigmoid(references) + offsets)
        boxes = self.detr_decoder._box_cxcywh_to_xyxy(boxes)
        logits = self.dot_product_scoring(hs, text.transpose(0, 1), padding)
        masks = self.mask_decoder(features, hs[-1], vision.transpose(0, 1),
                                  prompt=text.transpose(0, 1), prompt_mask=padding)
        return {"pred_masks": masks["pred_masks"], "pred_boxes": boxes[-1],
                "pred_logits": logits[-1].squeeze(-1), "presence_logits": presence[-1],
                "semantic_seg": masks["semantic_seg"], "decoder_reference_boxes": references}


def build_from_config(config, device, dtype):
    model = Sam3LiteTextModel(config).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if isinstance(module, VisionAttention):
            module.rotary_table = module.position_table(device)
    return model


def _source_key(key):
    if key.startswith("detr_encoder.") or key.startswith("geometry_encoder.layers."):
        for number in (1, 2, 3):
            key = key.replace(f".norm{number}.", f".layer_norm{number}.")
        key = key.replace(".linear1.", ".mlp.fc1.").replace(".linear2.", ".mlp.fc2.")
        return key.replace(".out_proj.", ".o_proj.")
    if key.startswith("geometry_encoder."):
        return key.replace(".emb.weight", ".weight")
    if key.startswith("detr_decoder."):
        key = key.replace(".bbox_embed.", ".box_head.").replace(".boxRPB_embed_", ".box_rpb_embed_")
        key = key.replace(".presence_token_head.", ".presence_head.").replace(".presence_token_out_norm.", ".presence_layer_norm.")
        key = key.replace(".norm.", ".output_layer_norm.")
        if ".layers." in key and key.split(".layers.")[1].split(".")[0].isdigit():
            # Decoder layer names; MLP children below are handled afterwards.
            for a, b in (("ca_text", "text_cross_attn"), ("cross_attn", "vision_cross_attn"),
                         ("catext_norm", "text_cross_attn_layer_norm"), ("norm1", "vision_cross_attn_layer_norm"),
                         ("norm2", "self_attn_layer_norm"), ("norm3", "mlp_layer_norm"),
                         ("linear1", "mlp.fc1"), ("linear2", "mlp.fc2")):
                key = key.replace(f".{a}.", f".{b}.")
        key = key.replace(".out_proj.", ".o_proj.")
        for head in ("box_head", "ref_point_head", "presence_head", "box_rpb_embed_x", "box_rpb_embed_y"):
            for index in range(3):
                key = key.replace(f".{head}.layers.{index}.", f".{head}.layer{index+1}.")
        return key
    if key.startswith("mask_decoder."):
        for a, b in (("mask_predictor", "mask_embedder"), ("instance_seg_head", "instance_projection"),
                     ("semantic_seg_head", "semantic_projection"), ("cross_attend_prompt", "prompt_cross_attn"),
                     ("cross_attn_norm", "prompt_cross_attn_norm")):
            key = key.replace(f".{a}.", f".{b}.")
        return key.replace(".out_proj.", ".o_proj.")
    if key.startswith("dot_product_scoring."):
        key = key.replace(".prompt_mlp.out_norm.", ".text_mlp_out_norm.")
        for index in range(2):
            key = key.replace(f".prompt_mlp.layers.{index}.", f".text_mlp.layer{index+1}.")
        return key.replace(".prompt_proj.", ".text_proj.").replace(".hs_proj.", ".query_proj.")
    return key


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    for key in model.state_dict():
        source = _source_key(key)
        # Text encoder embeddings may use the public Embedding operation wrapper.
        if source.startswith("text_encoder."):
            source = source.replace(".emb.weight", ".weight")
        mapped[key] = state_dict[source]
        used.add(source)
    if used != set(state_dict):
        raise ValueError(f"Unmapped SAM3 LiteText state: {sorted(set(state_dict)-used)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
