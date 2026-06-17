"""SAM3 image segmentation pipeline (L4 pipeline).

Contains:
- Sam3Config: model configuration dataclass.
- Sam3Model: full SAM3 image segmentation model wiring VL backbone, fusion
  encoder, detection decoder, geometry encoder, and segmentation head.

L4 wiring/configuration; computation lives in L1-L3 tasks.

Reference: sam3/model/sam3_image.py Sam3Image
           sam3/model_builder.py build_sam3_image_model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.sam3_position_encoding import Sam3PositionEncoding
from ..L2.sam3_cross_attention import Sam3CrossAttention
from ..L2.sam3_mask_predictor import Sam3MaskPredictor
from ..L3.sam3_decoder_layer import Sam3DecoderLayer
from ..L3.sam3_encoder_layer import Sam3EncoderLayer
from ..L3.sam3_neck import Sam3Neck
from ..L3.sam3_pixel_decoder import Sam3PixelDecoder
from ..L3.sam3_text_encoder import Sam3TextEncoder
from ..L3.sam3_vit import Sam3ViT


@dataclass
class Sam3Config:
    """Configuration for SAM3 image segmentation model."""

    # ViT backbone
    img_size: int = 1008
    patch_size: int = 14
    embed_dim: int = 1024
    vit_depth: int = 32
    vit_num_heads: int = 16
    vit_mlp_ratio: float = 4.625
    window_size: int = 24
    global_att_blocks: tuple = (7, 15, 23, 31)
    use_rope: bool = True
    use_tiled_rope: bool = True
    pretrain_img_size: int = 336
    retain_cls_token: bool = False
    ln_pre: bool = True
    bias_patch_embed: bool = False

    # Neck / FPN
    d_model: int = 256
    scale_factors: tuple = (4.0, 2.0, 1.0, 0.5)
    scalp: int = 1

    # Text encoder
    text_width: int = 1024
    text_heads: int = 16
    text_layers: int = 24
    text_context_length: int = 32
    text_vocab_size: int = 49408

    # Fusion encoder
    encoder_layers: int = 6
    encoder_dim_feedforward: int = 2048
    encoder_n_head: int = 8

    # Detection decoder
    decoder_layers: int = 6
    decoder_dim_feedforward: int = 2048
    decoder_n_head: int = 8
    num_queries: int = 200
    text_cross_attention: bool = True

    # Segmentation head
    pixel_decoder_upsampling_stages: int = 3
    presence_head: bool = False

    # Decoder features matching reference TransformerDecoder
    boxRPB: str = "log"
    presence_token: bool = True
    clamp_presence_logits: bool = True
    clamp_presence_logit_max_val: float = 10.0
    use_normed_output_consistently: bool = True

    # DotProductScoring
    use_dot_prod_scoring: bool = True
    dot_prod_d_proj: int = 256
    dot_prod_clamp_logits: bool = True
    dot_prod_clamp_max_val: float = 12.0

    # Geometry encoder
    geo_encoder_layers: int = 3
    geo_encoder_dim_feedforward: int = 2048
    geo_encoder_n_head: int = 8

    # Box head
    box_head_hidden: int = 256

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Sam3Config":
        """Create default config for known SAM3 models."""
        return cls()


class Sam3MLP(nn.Module):
    """Multi-layer perceptron matching reference sam3 MLP exactly.

    Structure: [input_dim] -> [hidden_dim]*(num_layers-1) -> output_dim.
    Hidden layers use ReLU + optional dropout; last layer is linear only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


class Sam3DotProductScoring(nn.Module):
    """Dot-product scoring matching reference DotProductScoring exactly.

    Mean-pools text features (masking padding), projects both text and decoder
    hidden states, and computes scaled dot-product scores.
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        prompt_mlp: Optional[nn.Module] = None,
        clamp_logits: bool = True,
        clamp_max_val: float = 12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        self.prompt_mlp = prompt_mlp
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.hs_proj = nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)
        proj_hs = self.hs_proj(hs)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)
        return scores


class Sam3GeometryEncoderLayer(nn.Module):
    """Encoder layer for geometry encoder (pre-norm, post-cross-attn keys with pos).

    Unlike the fusion encoder layer, this does NOT add pos to self-attn q/k
    and DOES add image pos encoding to cross-attn keys.

    Reference: sam3/model/encoder.py TransformerEncoderLayer (pre_norm,
               pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True)
    """

    def __init__(self, d_model: int, n_head: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.self_attn = Sam3CrossAttention(d_model, n_head)
        self.cross_attn = Sam3CrossAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.linear1 = Linear(d_model, dim_feedforward, bias=True)
        self.linear2 = Linear(dim_feedforward, d_model, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        key = memory + pos if pos is not None else memory
        tgt2 = self.cross_attn(tgt2, key, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class Sam3GeometryEncoder(nn.Module):
    """Geometry encoder for SAM3 (handles dummy prompt for grounding).

    For grounding (no geometric inputs), produces a single CLS token that
    cross-attends to image features through multiple encoder layers.

    Reference: sam3/model/geometry_encoder.py SequenceGeometryEncoder
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        d = config.d_model
        self.d_model = d
        self.cls_embed = nn.Embedding(1, d)
        self.final_proj = Linear(d, d, bias=True)
        self.norm = LayerNorm(d)
        self.encode = nn.ModuleList([
            Sam3GeometryEncoderLayer(d, config.geo_encoder_n_head, config.geo_encoder_dim_feedforward)
            for _ in range(config.geo_encoder_layers)
        ])
        self.encode_norm = LayerNorm(d)

    def forward(
        self,
        img_feat: torch.Tensor,
        img_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce geometry embedding for dummy prompt.

        Args:
            img_feat: (B, HW, D) flattened image features.
            img_pos: (B, HW, D) image position encoding.

        Returns:
            geo_feat: (1, B, D) geometry feature (seq-first).
            geo_mask: (B, 1) mask (all False = valid).
        """
        B = img_feat.shape[0]
        cls = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        cls = self.norm(self.final_proj(cls))
        for layer in self.encode:
            cls = layer(tgt=cls, memory=img_feat, pos=img_pos)
        cls = self.encode_norm(cls)
        geo_feat = cls.transpose(0, 1)  # (1, B, D) seq-first
        geo_mask = torch.zeros(B, 1, dtype=torch.bool, device=img_feat.device)
        return geo_feat, geo_mask


class Sam3FusionEncoder(nn.Module):
    """Multi-layer fusion encoder that fuses text and image features.

    Uses only the last feature level (matching reference num_feature_levels=1)
    and does NOT add pooled text to image features (matching reference
    add_pooled_text_to_img_feat=False).
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.num_feature_levels = 1
        self.layers = nn.ModuleList([
            Sam3EncoderLayer(
                config.d_model,
                config.encoder_n_head,
                config.encoder_dim_feedforward,
            )
            for _ in range(config.encoder_layers)
        ])
        self.text_pooling_proj = Linear(config.d_model, config.d_model, bias=True)
        self.level_embed = None

    def forward(
        self,
        src: List[torch.Tensor],
        src_pos: List[torch.Tensor],
        prompt: torch.Tensor,
        prompt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run fusion encoder.

        Args:
            src: List of (B, C, H, W) image feature maps (last level used).
            src_pos: Corresponding position encodings.
            prompt: (seq, B, D) text features.
            prompt_key_padding_mask: (B, seq) padding mask.

        Returns:
            Dict with 'memory', 'memory_text', 'spatial_shapes', etc.
        """
        feat = src[-1]
        pos = src_pos[-1]

        b, c, h, w = feat.shape
        spatial_shapes = [(h, w)]
        src_flat = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        pos_flat = pos.flatten(2).transpose(1, 2)

        prompt_bf = prompt.transpose(0, 1)  # (B, seq, D)

        output = src_flat
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=prompt_bf,
                query_pos=pos_flat,
                memory_key_padding_mask=prompt_key_padding_mask,
            )

        return {
            "memory": output.transpose(0, 1),  # (HW, B, D)
            "memory_text": prompt,
            "pos_embed": pos_flat.transpose(0, 1),
            "spatial_shapes": spatial_shapes,
        }


class Sam3Decoder(nn.Module):
    """Multi-layer detection decoder with learnable queries and box heads.

    Matches reference TransformerDecoder including boxRPB, presence_token,
    stacked intermediate outputs, and box refinement with normed output.

    Reference: sam3/model/decoder.py TransformerDecoder
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_queries = config.num_queries
        self.num_layers = config.decoder_layers
        self.n_head = config.decoder_n_head
        self.use_normed_output_consistently = config.use_normed_output_consistently

        self.query_embed = nn.Embedding(config.num_queries, config.d_model)
        self.reference_points = nn.Embedding(config.num_queries, 4)

        self.layers = nn.ModuleList([
            Sam3DecoderLayer(
                config.d_model,
                config.decoder_n_head,
                config.decoder_dim_feedforward,
                text_cross_attention=config.text_cross_attention,
            )
            for _ in range(config.decoder_layers)
        ])
        self.norm = LayerNorm(config.d_model)

        self.bbox_embed = Sam3MLP(config.d_model, config.d_model, 4, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.ref_point_head = Sam3MLP(2 * config.d_model, config.d_model, config.d_model, 2)

        # boxRPB
        self.boxRPB_mode = config.boxRPB
        if config.boxRPB != "none":
            n_input = 4 if config.boxRPB == "both" else 2
            self.boxRPB_embed_x = Sam3MLP(n_input, config.d_model, config.decoder_n_head, 2)
            self.boxRPB_embed_y = Sam3MLP(n_input, config.d_model, config.decoder_n_head, 2)

        # presence_token
        self.has_presence_token = config.presence_token
        self.clamp_presence_logits = config.clamp_presence_logits
        self.clamp_presence_logit_max_val = config.clamp_presence_logit_max_val
        if config.presence_token:
            self.presence_token = nn.Embedding(1, config.d_model)
            self.presence_token_head = Sam3MLP(config.d_model, config.d_model, 1, 3)
            self.presence_token_out_norm = nn.LayerNorm(config.d_model)

    def _get_rpb_matrix(
        self,
        reference_boxes: torch.Tensor,
        feat_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Compute box-relative position bias matrix.

        Args:
            reference_boxes: (B, Q, 4) cxcywh in sigmoid [0,1].
            feat_size: (H, W).

        Returns:
            (B, n_heads, Q, H*W) additive attention bias.
        """
        H, W = feat_size
        bs, num_queries, _ = reference_boxes.shape

        boxes_xyxy = self._box_cxcywh_to_xyxy(reference_boxes)

        coords_h = torch.arange(H, device=reference_boxes.device, dtype=torch.float32) / H
        coords_w = torch.arange(W, device=reference_boxes.device, dtype=torch.float32) / W

        # deltas_y: (bs, nq, H, 2) — distance from each grid row to y1, y2
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        # deltas_x: (bs, nq, W, 2) — distance from each grid col to x1, x2
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)

        if self.boxRPB_mode in ["log", "both"]:
            deltas_x_log = deltas_x * 8
            deltas_x_log = (
                torch.sign(deltas_x_log)
                * torch.log2(torch.abs(deltas_x_log) + 1.0)
                / np.log2(8)
            )
            deltas_y_log = deltas_y * 8
            deltas_y_log = (
                torch.sign(deltas_y_log)
                * torch.log2(torch.abs(deltas_y_log) + 1.0)
                / np.log2(8)
            )
            if self.boxRPB_mode == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

        deltas_x = self.boxRPB_embed_x(deltas_x)  # (bs, nq, W, n_heads)
        deltas_y = self.boxRPB_embed_y(deltas_y)  # (bs, nq, H, n_heads)

        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)  # (bs, nq, H, W, n_heads)
        B = B.flatten(2, 3)  # (bs, nq, H*W, n_heads)
        B = B.permute(0, 3, 1, 2).contiguous()  # (bs, n_heads, nq, H*W)
        return B

    def forward(
        self,
        memory: torch.Tensor,
        pos_embed: torch.Tensor,
        spatial_shapes: Tuple[int, int],
        memory_text: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run detection decoder.

        Args:
            memory: (L, B, D) encoder memory.
            pos_embed: (L, B, D) position embeddings.
            spatial_shapes: (H, W) of the single feature level.
            memory_text: (S, B, D) text features.
            text_attention_mask: (B, S) padding mask.

        Returns:
            (hs, reference_boxes, presence_logits, presence_feats):
                hs: (num_layers, B, Q, D) stacked intermediate hidden states.
                reference_boxes: (num_layers+1, B, Q, 4) stacked ref boxes.
                presence_logits: (num_layers, B, 1) or None.
                presence_feats: (B, 1, D) last layer presence features or None.
        """
        B = memory.shape[1]
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        memory_bf = memory.transpose(0, 1)  # (B, L, D)
        memory_pos_bf = pos_embed.transpose(0, 1)  # (B, L, D)
        text_bf = memory_text.transpose(0, 1) if memory_text is not None else None

        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(B, -1, -1).sigmoid()

        intermediate_hs = []
        intermediate_ref_boxes = [reference_boxes]
        intermediate_presence_logits = []
        presence_feats = None

        presence_out = None
        if self.has_presence_token:
            presence_out = self.presence_token.weight[None].expand(B, -1, -1)  # (B, 1, D)

        for layer_idx, layer in enumerate(self.layers):
            query_sine_embed = self._gen_sineembed(reference_boxes)  # (B, Q, 2*d_model)
            query_pos = self.ref_point_head(query_sine_embed)  # (B, Q, d_model)

            # Compute boxRPB cross-attention mask
            cross_attn_mask = None
            if self.boxRPB_mode != "none":
                cross_attn_mask = self._get_rpb_matrix(reference_boxes, spatial_shapes)
                cross_attn_mask = cross_attn_mask.flatten(0, 1)  # (B*n_heads, Q, HW)

            tgt, presence_out = layer(
                tgt=tgt,
                memory=memory_bf,
                tgt_query_pos=query_pos,
                memory_pos=memory_pos_bf,
                memory_text=text_bf,
                text_attention_mask=text_attention_mask,
                cross_attn_mask=cross_attn_mask,
                presence_token=presence_out,
            )

            # Box refinement using normed output
            reference_before_sigmoid = self._inverse_sigmoid(reference_boxes)
            if self.use_normed_output_consistently:
                delta_unsig = self.bbox_embed(self.norm(tgt))
            else:
                delta_unsig = self.bbox_embed(tgt)
            new_ref = (reference_before_sigmoid + delta_unsig).sigmoid()
            reference_boxes = new_ref.detach()
            if layer_idx != self.num_layers - 1:
                intermediate_ref_boxes.append(new_ref)
            intermediate_hs.append(self.norm(tgt))

            if self.has_presence_token:
                pres_logit = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)  # (B, 1)
                if self.clamp_presence_logits:
                    pres_logit = pres_logit.clamp(
                        min=-self.clamp_presence_logit_max_val,
                        max=self.clamp_presence_logit_max_val,
                    )
                intermediate_presence_logits.append(pres_logit)
                presence_feats = presence_out.clone()

        stacked_hs = torch.stack(intermediate_hs)  # (num_layers, B, Q, D)
        stacked_ref = torch.stack(intermediate_ref_boxes)  # (num_layers+1, B, Q, 4)
        stacked_presence = (
            torch.stack(intermediate_presence_logits) if self.has_presence_token else None
        )  # (num_layers, B, 1)

        return stacked_hs, stacked_ref, stacked_presence, presence_feats

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    @staticmethod
    def _gen_sineembed(pos: torch.Tensor, num_feats: int = 256) -> torch.Tensor:
        """Generate sine positional embedding from normalized coordinates.

        Matches reference gen_sineembed_for_position exactly.
        """
        half = num_feats // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half, dtype=torch.float32, device=pos.device)
        dim_t = 10000.0 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / half)

        x_embed = pos[..., 0] * scale
        y_embed = pos[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        w_embed = pos[..., 2] * scale
        h_embed = pos[..., 3] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_h = h_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)

        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1).to(pos.dtype)


class Sam3SegmentationHead(nn.Module):
    """Segmentation head combining pixel decoder, mask predictor,
    and prompt cross-attention (matching reference UniversalSegmentationHead)."""

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.d_model = config.d_model
        self.pixel_decoder = Sam3PixelDecoder(
            config.d_model, config.pixel_decoder_upsampling_stages,
        )
        self.mask_predictor = Sam3MaskPredictor(config.d_model, config.d_model)
        self.instance_seg_head = nn.Conv2d(config.d_model, config.d_model, kernel_size=1)
        self.semantic_seg_head = nn.Conv2d(config.d_model, 1, kernel_size=1)

        self.cross_attend_prompt = Sam3CrossAttention(config.d_model, config.encoder_n_head)
        self.cross_attn_norm = LayerNorm(config.d_model)

        if config.presence_head:
            self.presence_head = nn.Sequential(
                nn.Identity(),
                nn.Identity(),
                Linear(config.d_model, 1, bias=True),
            )
        else:
            self.presence_head = None

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Predict segmentation masks.

        Args:
            backbone_feats: FPN feature maps.
            obj_queries: (B, Q, D) decoder hidden states.
            encoder_hidden_states: (L, B, D) encoder memory (seq-first).
            prompt: (S, B, D) text/prompt features (seq-first).
            prompt_mask: (B, S) padding mask.

        Returns:
            Dict with 'pred_masks', 'semantic_seg', 'presence_logit'.
        """
        if prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2_bf = tgt2.transpose(0, 1)  # (B, L, D)
            prompt_bf = prompt.transpose(0, 1)  # (B, S, D)
            tgt2 = self.cross_attend_prompt(
                tgt2_bf, prompt_bf, prompt_bf,
                key_padding_mask=prompt_mask,
            )
            encoder_hidden_states = tgt2.transpose(0, 1) + encoder_hidden_states

        B = obj_queries.shape[0]

        spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
        enc_vis = encoder_hidden_states.permute(1, 2, 0)  # (B, D, L)
        enc_vis = enc_vis[..., :spatial_dim].reshape(-1, *backbone_feats[-1].shape[1:])

        vis_feats = list(backbone_feats)
        vis_feats[-1] = enc_vis

        pixel_embed = self.pixel_decoder(vis_feats)
        instance_embeds = self.instance_seg_head(pixel_embed)

        mask_preds = self.mask_predictor(obj_queries, instance_embeds)

        presence_logit = None
        if self.presence_head is not None:
            pooled = encoder_hidden_states.mean(0)  # (B, D)
            presence_logit = self.presence_head(pooled)

        return {
            "pred_masks": mask_preds,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }


def load_sam3_checkpoint(model: "Sam3Model", checkpoint_path: str) -> Tuple[list, list]:
    """Load a reference SAM3 checkpoint into a kb-nano Sam3Model.

    Handles the key remapping between the reference Sam3Image module hierarchy
    and the kb-nano Sam3Model hierarchy, including:
    - Prefix remapping (backbone.vision_backbone -> neck, etc.)
    - FPN conv name remapping (named submodules -> sequential indices)
    - Fused in_proj_weight/bias splitting into separate q/k/v projections
    - Text encoder layer renaming
    - Encoder/decoder attention key splitting

    Args:
        model: kb-nano Sam3Model instance.
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict.
    """
    import re

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    det_keys = {
        k.replace("detector.", ""): v
        for k, v in ckpt.items()
        if "detector" in k
    }

    # --- FPN conv key mapping ---
    # Reference: convs.0.dconv_2x2_0 / .dconv_2x2_1 / .conv_1x1 / .conv_3x3
    # kb-nano:   convs.0.conv.{0,1,2,...}  (via nn.Sequential)
    #
    # Scale 4.0 (convs.0): dconv_2x2_0->0, GELU->1(skipped), dconv_2x2_1->2, conv_1x1->3, conv_3x3->4
    # Scale 2.0 (convs.1): dconv_2x2->0, conv_1x1->1, conv_3x3->2
    # Scale 1.0 (convs.2): conv_1x1->0, conv_3x3->1
    # Scale 0.5 (convs.3): MaxPool2d->0(no params), conv_1x1->1, conv_3x3->2
    fpn_remap = {
        "convs.0.dconv_2x2_0": "convs.0.conv.0",
        "convs.0.dconv_2x2_1": "convs.0.conv.2",
        "convs.0.conv_1x1": "convs.0.conv.3",
        "convs.0.conv_3x3": "convs.0.conv.4",
        "convs.1.dconv_2x2": "convs.1.conv.0",
        "convs.1.conv_1x1": "convs.1.conv.1",
        "convs.1.conv_3x3": "convs.1.conv.2",
        "convs.2.conv_1x1": "convs.2.conv.0",
        "convs.2.conv_3x3": "convs.2.conv.1",
        "convs.3.conv_1x1": "convs.3.conv.1",
        "convs.3.conv_3x3": "convs.3.conv.2",
    }

    remapped = {}
    skipped_prefixes = (
        "inst_interactive_predictor.",
        "transformer.decoder.instance_",
        "transformer.encoder.text_pooling_proj.",
    )

    for ref_key, val in det_keys.items():
        if any(ref_key.startswith(p) for p in skipped_prefixes):
            continue

        new_key = ref_key

        # --- Top-level prefix remapping ---
        if new_key.startswith("backbone.vision_backbone."):
            rest = new_key[len("backbone.vision_backbone."):]
            # FPN conv remapping
            matched = False
            for ref_conv, kb_conv in fpn_remap.items():
                if rest.startswith(ref_conv):
                    suffix = rest[len(ref_conv):]
                    new_key = f"neck.{kb_conv}{suffix}"
                    matched = True
                    break
            if not matched:
                new_key = f"neck.{rest}"

        elif new_key.startswith("backbone.language_backbone.encoder."):
            rest = new_key[len("backbone.language_backbone.encoder."):]
            new_key = f"text_encoder.{rest}"

        elif new_key.startswith("backbone.language_backbone.resizer."):
            rest = new_key[len("backbone.language_backbone.resizer."):]
            new_key = f"text_encoder.resizer.{rest}"

        elif new_key.startswith("transformer.encoder."):
            rest = new_key[len("transformer.encoder."):]
            new_key = f"encoder.{rest}"

        elif new_key.startswith("transformer.decoder."):
            rest = new_key[len("transformer.decoder."):]
            new_key = f"decoder.{rest}"

        elif new_key.startswith("segmentation_head."):
            rest = new_key[len("segmentation_head."):]
            new_key = f"seg_head.{rest}"

        # --- Text encoder transformer layer remapping ---
        # Ref:  text_encoder.transformer.resblocks.{i}.attn.in_proj_weight
        # KB:   text_encoder.transformer_layers.{i}.block.q_proj.weight etc.
        m = re.match(r"text_encoder\.transformer\.resblocks\.(\d+)\.(.*)", new_key)
        if m:
            layer_idx = m.group(1)
            layer_rest = m.group(2)

            # Fused attention: in_proj_weight -> split q/k/v
            if layer_rest == "attn.in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.q_proj.weight"] = val[:d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.k_proj.weight"] = val[d:2*d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.v_proj.weight"] = val[2*d:]
                continue
            elif layer_rest == "attn.in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.q_proj.bias"] = val[:d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.k_proj.bias"] = val[d:2*d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.v_proj.bias"] = val[2*d:]
                continue
            elif layer_rest.startswith("attn.out_proj."):
                suffix = layer_rest[len("attn.out_proj."):]
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.out_proj.{suffix}"
            elif layer_rest.startswith("mlp.c_fc."):
                suffix = layer_rest[len("mlp.c_fc."):]
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.mlp_fc1.{suffix}"
            elif layer_rest.startswith("mlp.c_proj."):
                suffix = layer_rest[len("mlp.c_proj."):]
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.mlp_fc2.{suffix}"
            elif layer_rest.startswith("ln_1.") or layer_rest.startswith("ln_2."):
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.{layer_rest}"
            else:
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.{layer_rest}"

        # --- text_encoder token_embedding ---
        if new_key == "text_encoder.token_embedding.weight":
            new_key = "text_encoder.token_embedding.emb.weight"

        # --- text_encoder text_projection (not used in kb-nano, skip) ---
        if new_key == "text_encoder.text_projection":
            continue

        # --- Encoder layer: fused in_proj -> split q/k/v ---
        em = re.match(r"encoder\.layers\.(\d+)\.(self_attn|cross_attn_image)\.(.*)", new_key)
        if em:
            layer_idx = em.group(1)
            attn_name = "self_attn" if em.group(2) == "self_attn" else "cross_attn"
            attn_rest = em.group(3)

            if attn_rest == "in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.q_proj.weight"] = val[:d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.k_proj.weight"] = val[d:2*d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.v_proj.weight"] = val[2*d:]
                continue
            elif attn_rest == "in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.q_proj.bias"] = val[:d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.k_proj.bias"] = val[d:2*d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.v_proj.bias"] = val[2*d:]
                continue
            elif attn_rest.startswith("out_proj."):
                new_key = f"encoder.layers.{layer_idx}.{attn_name}.{attn_rest}"

        # --- Decoder layer: fused in_proj -> split q/k/v ---
        # Reference names: self_attn, cross_attn (image), ca_text
        # kb-nano names: self_attn, cross_attn (image), ca_text (same now)
        dm = re.match(r"decoder\.layers\.(\d+)\.(self_attn|cross_attn|ca_text)\.(.*)", new_key)
        if dm:
            layer_idx = dm.group(1)
            kb_attn = dm.group(2)
            attn_rest = dm.group(3)

            if attn_rest == "in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.q_proj.weight"] = val[:d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.k_proj.weight"] = val[d:2*d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.v_proj.weight"] = val[2*d:]
                continue
            elif attn_rest == "in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.q_proj.bias"] = val[:d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.k_proj.bias"] = val[d:2*d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.v_proj.bias"] = val[2*d:]
                continue
            elif attn_rest.startswith("out_proj."):
                new_key = f"decoder.layers.{layer_idx}.{kb_attn}.{attn_rest}"

        # bbox_embed and ref_point_head now use Sam3MLP with layers.{i}
        # which matches reference MLP naming directly — no remapping needed.

        # --- Segmentation head cross_attend_prompt: fused in_proj -> split q/k/v ---
        if new_key == "seg_head.cross_attend_prompt.in_proj_weight":
            d = val.shape[0] // 3
            remapped["seg_head.cross_attend_prompt.q_proj.weight"] = val[:d]
            remapped["seg_head.cross_attend_prompt.k_proj.weight"] = val[d:2*d]
            remapped["seg_head.cross_attend_prompt.v_proj.weight"] = val[2*d:]
            continue
        if new_key == "seg_head.cross_attend_prompt.in_proj_bias":
            d = val.shape[0] // 3
            remapped["seg_head.cross_attend_prompt.q_proj.bias"] = val[:d]
            remapped["seg_head.cross_attend_prompt.k_proj.bias"] = val[d:2*d]
            remapped["seg_head.cross_attend_prompt.v_proj.bias"] = val[2*d:]
            continue

        # --- Segmentation head mask_predictor: mask_embed.layers -> layers ---
        sm = re.match(r"seg_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.(.*)", new_key)
        if sm:
            new_key = f"seg_head.mask_predictor.layers.{sm.group(1)}.{sm.group(2)}"

        # --- Geometry encoder: encode.{i} attn remapping ---
        gm = re.match(r"geometry_encoder\.encode\.(\d+)\.(self_attn|cross_attn_image)\.(.*)", new_key)
        if gm:
            layer_idx = gm.group(1)
            attn_name = "self_attn" if gm.group(2) == "self_attn" else "cross_attn"
            attn_rest = gm.group(3)

            if attn_rest == "in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"geometry_encoder.encode.{layer_idx}.{attn_name}.q_proj.weight"] = val[:d]
                remapped[f"geometry_encoder.encode.{layer_idx}.{attn_name}.k_proj.weight"] = val[d:2*d]
                remapped[f"geometry_encoder.encode.{layer_idx}.{attn_name}.v_proj.weight"] = val[2*d:]
                continue
            elif attn_rest == "in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"geometry_encoder.encode.{layer_idx}.{attn_name}.q_proj.bias"] = val[:d]
                remapped[f"geometry_encoder.encode.{layer_idx}.{attn_name}.k_proj.bias"] = val[d:2*d]
                remapped[f"geometry_encoder.encode.{layer_idx}.{attn_name}.v_proj.bias"] = val[2*d:]
                continue
            elif attn_rest.startswith("out_proj."):
                new_key = f"geometry_encoder.encode.{layer_idx}.{attn_name}.{attn_rest}"

        # --- Geometry encoder: skip unused modules (points/boxes/masks) ---
        geo_skip = (
            "geometry_encoder.label_embed.",
            "geometry_encoder.points_",
            "geometry_encoder.boxes_",
            "geometry_encoder.img_pre_norm.",
            "geometry_encoder.mask_",
        )
        if any(new_key.startswith(p) for p in geo_skip):
            continue

        # --- ViT block attention freqs_cis: trunk.blocks.{i}.attn.freqs_cis ---
        if ".attn.freqs_cis" in new_key:
            new_key = new_key.replace(".attn.freqs_cis", ".attn.rope.freqs_cis")

        # --- ViT MLP: mlp.fc1/fc2 -> mlp.fc1/fc2 (same in kb-nano) ---
        # (already matches)

        remapped[new_key] = val

    # Handle kb-nano Linear wrapper: weight is stored directly, not inside .matmul
    # Our Linear has .weight and .bias directly as nn.Parameter, same as nn.Linear
    # But our L1 Linear wraps Matmul - check if state dict keys need adjustment
    # Actually, our Linear stores .weight and .bias as nn.Parameter + .matmul as Matmul
    # The state dict key is "xxx.weight" not "xxx.matmul.weight", so it should work.

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return missing, unexpected


def load_sam3_tracker_checkpoint(
    tracker: "Sam3TrackerPredictor",
    checkpoint_path: str,
) -> Tuple[list, list]:
    """Load tracker weights from a full SAM3 video model checkpoint.

    Handles remapping from reference tracker.xxx keys to kb-nano Sam3TrackerPredictor.

    The reference checkpoint stores tracker weights under:
    - tracker.sam_prompt_encoder.xxx -> sam_prompt_encoder.xxx
    - tracker.sam_mask_decoder.xxx -> sam_mask_decoder.xxx
    - tracker.maskmem_backbone.xxx -> maskmem_backbone.xxx
    - tracker.transformer.encoder.xxx -> memory_attention.xxx
    - tracker.obj_ptr_proj.xxx -> obj_ptr_proj.xxx
    - tracker.maskmem_tpos_enc -> maskmem_tpos_enc
    etc.

    Reference: sam3/model_builder.py build_sam3_video_model checkpoint loading
    """
    import re

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    tracker_keys = {
        k.replace("tracker.", ""): v
        for k, v in ckpt.items()
        if k.startswith("tracker.")
    }

    remapped = {}

    for ref_key, val in tracker_keys.items():
        new_key = ref_key

        # --- Memory attention: transformer.encoder.xxx -> memory_attention.xxx ---
        if new_key.startswith("transformer.encoder."):
            rest = new_key[len("transformer.encoder."):]
            new_key = f"memory_attention.{rest}"

        # --- SAM mask decoder: split fused attention weights ---
        # TwoWayTransformer layers use Sam3Attention with q/k/v/out_proj
        # Reference may store them as q_proj/k_proj/v_proj/out_proj already
        # (since sam3/sam/transformer.py Attention uses separate projections)
        # So no splitting needed.

        # --- Memory attention layers: RoPEAttention uses q/k/v/out_proj ---
        # Reference sam3/sam/transformer.py RoPEAttention also uses separate
        # projections, so naming should match directly.

        # --- freqs_cis buffers in RoPEAttention ---
        # These are computed from params, not stored, so skip if present
        if "freqs_cis" in new_key:
            continue

        # --- Skip backbone weights (loaded separately) ---
        if new_key.startswith("backbone."):
            continue

        remapped[new_key] = val

    missing, unexpected = tracker.load_state_dict(remapped, strict=False)
    return missing, unexpected


class Sam3Model(nn.Module):
    """Full SAM3 image segmentation model.

    Assembles VL backbone (ViT + neck + text encoder), fusion encoder,
    detection decoder, and segmentation head.

    Args:
        config: Sam3Config with all hyperparameters.
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.config = config

        vit = Sam3ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.vit_depth,
            num_heads=config.vit_num_heads,
            mlp_ratio=config.vit_mlp_ratio,
            window_size=config.window_size,
            global_att_blocks=config.global_att_blocks,
            use_rope=config.use_rope,
            use_tiled_rope=config.use_tiled_rope,
            pretrain_img_size=config.pretrain_img_size,
            retain_cls_token=config.retain_cls_token,
            ln_pre=config.ln_pre,
            bias_patch_embed=config.bias_patch_embed,
            return_interm_layers=False,
        )

        self.neck = Sam3Neck(
            trunk=vit,
            d_model=config.d_model,
            scale_factors=config.scale_factors,
        )

        self.text_encoder = Sam3TextEncoder(
            d_model=config.d_model,
            width=config.text_width,
            heads=config.text_heads,
            layers=config.text_layers,
            context_length=config.text_context_length,
            vocab_size=config.text_vocab_size,
        )

        self.encoder = Sam3FusionEncoder(config)
        self.geometry_encoder = Sam3GeometryEncoder(config)
        self.decoder = Sam3Decoder(config)
        self.seg_head = Sam3SegmentationHead(config)

        if config.use_dot_prod_scoring:
            prompt_mlp = Sam3MLP(
                input_dim=config.d_model,
                hidden_dim=2048,
                output_dim=config.d_model,
                num_layers=2,
                dropout=0.1,
                residual=True,
                out_norm=nn.LayerNorm(config.d_model),
            )
            self.dot_prod_scoring = Sam3DotProductScoring(
                d_model=config.d_model,
                d_proj=config.dot_prod_d_proj,
                prompt_mlp=prompt_mlp,
                clamp_logits=config.dot_prod_clamp_logits,
                clamp_max_val=config.dot_prod_clamp_max_val,
            )
        else:
            self.dot_prod_scoring = None

        self.scalp = config.scalp
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal (std=0.02) for stability."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, Linear)):
                w = m.weight if hasattr(m, "weight") else None
                if w is not None:
                    nn.init.trunc_normal_(w, std=0.02)
                b = m.bias if hasattr(m, "bias") else None
                if b is not None:
                    nn.init.zeros_(b)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        tokenized_text: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run full SAM3 forward pass.

        Matches reference Sam3Image.forward_grounding + _run_decoder +
        _update_scores_and_boxes flow.

        Args:
            images: (B, 3, H, W) input images, normalized.
            tokenized_text: (B, seq_len) tokenized text prompts.

        Returns:
            Dict with:
                'pred_boxes': (B, Q, 4) predicted bounding boxes.
                'pred_masks': (B, Q, H, W) predicted segmentation masks.
                'pred_logits': (num_layers, B, Q, 1) classification logits.
        """
        sam3_feats, sam3_pos, _, _ = self.neck(images)

        if self.scalp > 0:
            sam3_feats = sam3_feats[:-self.scalp]
            sam3_pos = sam3_pos[:-self.scalp]

        text_mask, text_memory, text_embeds = self.text_encoder(tokenized_text)

        encoder_out = self.encoder(
            src=sam3_feats,
            src_pos=sam3_pos,
            prompt=text_memory,
            prompt_key_padding_mask=text_mask,
        )

        spatial_shapes = tuple(sam3_feats[-1].shape[-2:])  # (H, W)

        feat_flat = sam3_feats[-1].flatten(2).transpose(1, 2)  # (B, HW, D)
        pos_flat = sam3_pos[-1].flatten(2).transpose(1, 2)  # (B, HW, D)
        geo_feat, geo_mask = self.geometry_encoder(feat_flat, pos_flat)

        prompt = torch.cat([encoder_out["memory_text"], geo_feat], dim=0)
        prompt_mask = torch.cat([text_mask, geo_mask], dim=1)

        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.decoder(
            memory=encoder_out["memory"],
            pos_embed=encoder_out["pos_embed"],
            spatial_shapes=spatial_shapes,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
        )

        if self.dot_prod_scoring is not None:
            pred_logits = self.dot_prod_scoring(
                hs, prompt, prompt_mask,
            )
        else:
            pred_logits = None

        # Box prediction: bbox_embed on normed hs + reference_boxes
        # reference_boxes: (num_layers, B, Q, 4) matches hs: (num_layers, B, Q, D)
        anchor_box_offsets = self.decoder.bbox_embed(hs)  # (num_layers, B, Q, 4)
        outputs_coord = (
            Sam3Decoder._inverse_sigmoid(reference_boxes) + anchor_box_offsets
        ).sigmoid()

        # Use last-layer outputs for final predictions
        pred_boxes = outputs_coord[-1]  # (B, Q, 4)
        final_hs = hs[-1]  # (B, Q, D)

        seg_out = self.seg_head(
            backbone_feats=sam3_feats,
            obj_queries=final_hs,
            encoder_hidden_states=encoder_out["memory"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        out_logits = pred_logits[-1] if pred_logits is not None else None

        return {
            "pred_boxes": pred_boxes,
            "pred_masks": seg_out["pred_masks"],
            "pred_logits": out_logits,
            "semantic_seg": seg_out.get("semantic_seg"),
        }
