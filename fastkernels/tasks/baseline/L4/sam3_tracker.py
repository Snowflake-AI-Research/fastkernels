"""SAM3 Tracker: tracker base + predictor for video segmentation.

Sam3TrackerBase: Core tracker that handles frame-by-frame tracking with
  memory conditioning, SAM prompt encoding, mask decoding, and memory encoding.

Sam3TrackerPredictor: Extends base with init_state, add_new_mask, propagate_in_video
  for managing inference state across frames with multi-object tracking.

Reference: sam3/model/sam3_tracker_base.py Sam3TrackerBase
           sam3/model/sam3_tracking_predictor.py Sam3TrackerPredictor
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L2.sam3_prompt_encoder import Sam3PromptEncoder
from ..L2.sam3_memory_encoder import Sam3MemoryEncoder, SimpleMaskDownSampler, CXBlock, SimpleFuser, PositionEmbeddingSine
from ..L3.sam3_mask_decoder import Sam3MaskDecoder, TwoWayTransformer, MLP, LayerNorm2d
from ..L3.sam3_memory_attention import Sam3MemoryAttention, Sam3MemoryAttentionLayer
from ..L1.sam3_rope_attention import Sam3RoPEAttention

NO_OBJ_SCORE = -1024.0

logger = logging.getLogger(__name__)


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: int = 10000) -> torch.Tensor:
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False):
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}

    assert max_cond_frame_num >= 2
    selected_outputs = {}
    if keep_first_cond_frame:
        idx_first = min((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_first is None:
            idx_first = max((t for t in cond_frame_outputs if t > frame_idx), default=None)
        if idx_first is not None:
            selected_outputs[idx_first] = cond_frame_outputs[idx_first]

    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected_outputs[idx_before] = cond_frame_outputs[idx_before]

    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected_outputs[idx_after] = cond_frame_outputs[idx_after]

    num_remain = max_cond_frame_num - len(selected_outputs)
    inds_remain = sorted(
        (t for t in cond_frame_outputs if t not in selected_outputs),
        key=lambda x: abs(x - frame_idx),
    )[:num_remain]
    selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
    unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
    return selected_outputs, unselected_outputs


def fill_holes_in_mask_scores(mask, max_area=0):
    """Placeholder for post-processing fill holes. In fastkernels, we skip connected components."""
    return mask


class Sam3TrackerBase(nn.Module):
    """Core frame-by-frame tracker with memory bank.

    Reference: sam3/model/sam3_tracker_base.py Sam3TrackerBase
    """

    def __init__(
        self,
        backbone,
        memory_attention: nn.Module,
        maskmem_backbone: nn.Module,
        num_maskmem: int = 7,
        image_size: int = 1008,
        backbone_stride: int = 14,
        max_cond_frames_in_attn: int = -1,
        keep_first_cond_frame: bool = False,
        multimask_output_in_sam: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        multimask_output_for_tracking: bool = False,
        memory_temporal_stride_for_eval: int = 1,
        max_obj_ptrs_in_encoder: int = 16,
        sam_mask_decoder_extra_args: dict | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        self.memory_attention = memory_attention
        self.hidden_dim = 256  # d_model

        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = 64  # out_dim of memory encoder
        self.num_maskmem = num_maskmem

        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.no_mem_embed, std=0.02)
        nn.init.trunc_normal_(self.no_mem_pos_enc, std=0.02)

        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval

        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking

        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args

        self.no_obj_ptr = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.trunc_normal_(self.no_obj_ptr, std=0.02)
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, self.mem_dim))
        nn.init.trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_sam_heads(self):
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        self.sam_prompt_encoder = Sam3PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = Sam3MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None):
        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = torch.tensor(rel_pos_list, device=device, dtype=torch.float32) / t_diff_max
        tpos_dim = self.hidden_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        return self.obj_ptr_tpos_proj(pos_enc)

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        B = backbone_features.size(0)
        device = backbone_features.device

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False, mode="bilinear", antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        image_pe = self.sam_prompt_encoder.get_dense_pe()

        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // self.backbone_stride * 4,
                  high_res_masks.size(-1) // self.backbone_stride * 4),
            align_corners=False, mode="bilinear", antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
        )
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def forward_image(self, img_batch):
        backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
        feat_sizes, output_dict, num_frames, track_in_reverse=False,
    ):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        if not is_init_cond_frame:
            to_cat_prompt, to_cat_prompt_pos_embed = [], []

            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )
            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]

            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    prev_frame_idx = frame_idx - t_rel if not track_in_reverse else frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_frame_idx = ((frame_idx - 2) // r) * r - (t_rel - 2) * r
                    else:
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r + (t_rel - 2) * r

                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device)
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                to_cat_prompt_pos_embed.append(maskmem_enc)

            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            ptr_cond_outputs = {
                t: out for t, out in selected_cond_outputs.items()
                if (t >= frame_idx if track_in_reverse else t <= frame_idx)
            }
            pos_and_ptrs = [
                ((frame_idx - t) * tpos_sign_mul, out["obj_ptr"], True)
                for t, out in ptr_cond_outputs.items()
            ]

            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))

            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, _ = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                obj_pos = self._get_tpos_enc(pos_list, max_abs_pos=max_obj_ptrs_in_encoder, device=device)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]

            if len(to_cat_prompt) == 0:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
        else:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)

        encoder_out = self.memory_attention(
            src=current_vision_feats[-1:],
            src_pos=current_vision_pos_embeds[-1:],
            src_key_padding_mask=[None],
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=None,
            feat_sizes=feat_sizes[-1:],
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_out = self.maskmem_backbone(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc

    def track_step(
        self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
        feat_sizes, point_inputs, mask_inputs, output_dict, num_frames,
        track_in_reverse=False, run_mem_encoder=True, prev_sam_mask_logits=None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if mask_inputs is not None:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:], output_dict=output_dict,
                num_frames=num_frames, track_in_reverse=track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits

            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        _, _, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )


class Sam3TrackerPredictor(Sam3TrackerBase):
    """High-level tracker API with inference state management.

    Reference: sam3/model/sam3_tracking_predictor.py Sam3TrackerPredictor
    """

    def __init__(
        self,
        fill_hole_area: int = 0,
        always_start_from_first_ann_frame: bool = False,
        non_overlap_masks_for_output: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.non_overlap_masks_for_output = non_overlap_masks_for_output
        self.add_all_frames_to_correct_as_cond = True

    @torch.inference_mode()
    def init_state(self, video_height, video_width, num_frames):
        inference_state = {}
        inference_state["device"] = self.device
        inference_state["storage_device"] = torch.device("cuda")
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["num_frames"] = num_frames
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["first_ann_frame_idx"] = None
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        else:
            raise RuntimeError(f"Cannot add new object id {obj_id} after tracking starts.")

    @torch.inference_mode()
    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        """Add a mask prompt for an object on a specific frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = frame_idx

        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = mask
        return obj_idx

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        images,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate tracking through video frames.

        Args:
            inference_state: state dict from init_state
            images: list of [C,H,W] tensors or [T,C,H,W] tensor
            start_frame_idx: frame to start propagation from
            max_frame_num_to_track: max frames to track
            reverse: track in reverse order

        Yields:
            (frame_idx, obj_ids, pred_masks_per_obj) tuples
        """
        inference_state["tracking_has_started"] = True
        num_frames = inference_state["num_frames"]

        if start_frame_idx is None:
            start_frame_idx = inference_state.get("first_ann_frame_idx", 0) or 0
        if self.always_start_from_first_ann_frame:
            start_frame_idx = inference_state.get("first_ann_frame_idx", 0) or 0

        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        # Consolidate temp outputs into main output_dict
        self._consolidate_temp_outputs(inference_state)

        if reverse:
            frame_indices = range(start_frame_idx, max(start_frame_idx - max_frame_num_to_track, -1), -1)
        else:
            frame_indices = range(start_frame_idx, min(start_frame_idx + max_frame_num_to_track, num_frames))

        obj_ids = inference_state["obj_ids"]
        output_dict = inference_state["output_dict"]

        for frame_idx in frame_indices:
            # Get image
            if isinstance(images, torch.Tensor):
                img = images[frame_idx:frame_idx+1]
            else:
                img = images[frame_idx].unsqueeze(0)
            img = img.to(self.device)

            # Forward backbone
            backbone_out = self.forward_image(img)
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

            is_init_cond_frame = (frame_idx in output_dict["cond_frame_outputs"])

            # Get prompts for this frame
            point_inputs = None
            mask_inputs = None
            for obj_idx in range(len(obj_ids)):
                if frame_idx in inference_state["mask_inputs_per_obj"].get(obj_idx, {}):
                    mask_inputs = inference_state["mask_inputs_per_obj"][obj_idx][frame_idx]
                    if mask_inputs.ndim == 2:
                        mask_inputs = mask_inputs[None, None]
                    elif mask_inputs.ndim == 3:
                        mask_inputs = mask_inputs.unsqueeze(0)
                    mask_inputs = mask_inputs.to(self.device).float()
                    is_init_cond_frame = True

            current_out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=reverse,
            )

            if is_init_cond_frame:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out

            pred_masks = current_out["pred_masks_high_res"]

            if self.fill_hole_area > 0:
                pred_masks = fill_holes_in_mask_scores(pred_masks, max_area=self.fill_hole_area)

            yield frame_idx, obj_ids, pred_masks

    def _consolidate_temp_outputs(self, inference_state):
        """Merge temp outputs into main output_dict."""
        for obj_idx in inference_state["temp_output_dict_per_obj"]:
            temp_out = inference_state["temp_output_dict_per_obj"][obj_idx]
            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for fidx, out in temp_out[key].items():
                    inference_state["output_dict"][key][fidx] = out
            temp_out["cond_frame_outputs"].clear()
            temp_out["non_cond_frame_outputs"].clear()


def build_tracker_components():
    """Build the memory attention transformer and memory encoder components.

    Returns configuration matching the reference model_builder.py:
    - memory_attention: TransformerEncoderCrossAttention with 4 layers
    - maskmem_backbone: SimpleMaskEncoder with 64-D output
    """
    # Self attention for memory attention layers
    self_attention = Sam3RoPEAttention(
        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1,
        rope_theta=10000.0, feat_sizes=(72, 72),
    )
    # Cross attention (kv_in_dim=64 for memory tokens)
    cross_attention = Sam3RoPEAttention(
        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1,
        kv_in_dim=64, rope_theta=10000.0, feat_sizes=(72, 72), rope_k_repeat=True,
    )
    # Memory attention layer
    mem_attn_layer = Sam3MemoryAttentionLayer(
        activation="relu", d_model=256, dim_feedforward=2048, dropout=0.1,
        num_heads=1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False, self_attention=self_attention,
        cross_attention=cross_attention, cross_attention_first=False,
    )
    # Memory attention transformer (4 layers)
    memory_attention = Sam3MemoryAttention(
        d_model=256, pos_enc_at_input=True, layer=mem_attn_layer,
        num_layers=4, batch_first=True,
    )

    # Memory encoder
    position_encoding = PositionEmbeddingSine(num_pos_feats=64, normalize=True, temperature=10000)
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152],
    )
    cx_block = CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True)
    fuser = SimpleFuser(layer=cx_block, num_layers=2)
    maskmem_backbone = Sam3MemoryEncoder(
        out_dim=64, mask_downsampler=mask_downsampler,
        fuser=fuser, position_encoding=position_encoding,
    )

    return memory_attention, maskmem_backbone
