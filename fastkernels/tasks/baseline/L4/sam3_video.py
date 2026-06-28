"""SAM3 Video Pipeline: detection + tracking + association.

Orchestrates frame-by-frame video segmentation by:
1. Running detector to get per-frame bounding boxes and masks
2. Running tracker to propagate object masks across frames
3. Associating detections with tracked objects (matching, hotstart, reconditioning)

Reference: sam3/model/sam3_video_base.py Sam3VideoBase
           sam3/model/sam3_video_inference.py Sam3VideoInferenceWithInstanceInteractivity
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam3_tracker import Sam3TrackerPredictor, fill_holes_in_mask_scores

logger = logging.getLogger(__name__)


class Sam3VideoModel(nn.Module):
    """Video segmentation model combining detector + tracker with association heuristics.

    Reference: sam3/model/sam3_video_base.py Sam3VideoBase
               sam3/model/sam3_video_inference.py Sam3VideoInferenceWithInstanceInteractivity
    """

    def __init__(
        self,
        detector: nn.Module,
        tracker: Sam3TrackerPredictor,
        score_threshold_detection: float = 0.5,
        det_nms_thresh: float = 0.1,
        assoc_iou_thresh: float = 0.1,
        trk_assoc_iou_thresh: float = 0.5,
        new_det_thresh: float = 0.7,
        hotstart_delay: int = 15,
        hotstart_unmatch_thresh: int = 8,
        hotstart_dup_thresh: int = 8,
        fill_hole_area: int = 16,
        image_size: int = 1008,
        image_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.fill_hole_area = fill_hole_area
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def process_video(
        self,
        images: torch.Tensor,
        text_prompt: Optional[str] = None,
        captions: Optional[List[str]] = None,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """Process a video clip through detection + tracking pipeline.

        This is the simplified single-GPU version of _det_track_one_frame.

        Args:
            images: [T, C, H, W] tensor of video frames
            text_prompt: text prompt for detection
            captions: per-frame captions (not used in basic pipeline)

        Returns:
            Dict mapping frame_idx -> {obj_id -> mask_tensor}
        """
        T = images.shape[0]
        all_frame_outputs = {}
        tracker_state = None
        tracked_objects = {}  # obj_id -> info
        next_obj_id = 0

        for frame_idx in range(T):
            frame = images[frame_idx:frame_idx+1].to(self.device)

            # Step 1: Run detector
            det_boxes, det_scores, det_masks = self._run_detection(frame, text_prompt)

            # Step 2: Run tracker propagation (if we have tracked objects)
            trk_masks = None
            if tracker_state is not None and len(tracked_objects) > 0:
                trk_masks = self._run_tracker_propagation(
                    frame, frame_idx, tracker_state, tracked_objects, T,
                )

            # Step 3: Associate detections with tracked objects
            new_det_inds, matched_pairs = self._associate_det_trk(
                det_masks, det_scores, trk_masks, tracked_objects,
            )

            # Step 4: Update tracker with new detections and matched updates
            tracker_state, tracked_objects, next_obj_id = self._update_tracker(
                frame, frame_idx, tracker_state, tracked_objects,
                det_masks, det_scores, new_det_inds, matched_pairs,
                next_obj_id, T,
            )

            # Step 5: Build output masks for this frame
            frame_output = self._build_frame_output(
                frame_idx, tracker_state, tracked_objects, trk_masks,
                det_masks, new_det_inds, matched_pairs,
            )
            all_frame_outputs[frame_idx] = frame_output

        return all_frame_outputs

    def _run_detection(self, frame, text_prompt=None):
        """Run detector on a single frame.

        Returns:
            det_boxes: [N, 4] detected boxes
            det_scores: [N] detection scores
            det_masks: [N, 1, H, W] detection masks (or None)
        """
        with torch.no_grad():
            det_out = self.detector(frame, captions=[text_prompt] if text_prompt else None)

        if det_out is None:
            return None, None, None

        det_boxes = det_out.get("pred_boxes", None)
        det_scores = det_out.get("scores", None)
        det_masks = det_out.get("pred_masks", None)

        if det_scores is not None:
            keep = det_scores >= self.score_threshold_detection
            if keep.any():
                det_boxes = det_boxes[keep] if det_boxes is not None else None
                det_scores = det_scores[keep]
                det_masks = det_masks[keep] if det_masks is not None else None
            else:
                return None, None, None

        return det_boxes, det_scores, det_masks

    def _run_tracker_propagation(self, frame, frame_idx, tracker_state, tracked_objects, num_frames):
        """Propagate tracker to current frame and return predicted masks."""
        # Use the tracker's track_step to predict masks
        backbone_out = self.tracker.forward_image(frame)
        _, vision_feats, vision_pos_embeds, feat_sizes = self.tracker._prepare_backbone_features(backbone_out)

        output_dict = tracker_state["output_dict"]
        current_out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=num_frames,
        )

        output_dict["non_cond_frame_outputs"][frame_idx] = current_out
        return current_out.get("pred_masks_high_res", None)

    def _associate_det_trk(self, det_masks, det_scores, trk_masks, tracked_objects):
        """Associate detections with tracked objects via mask IoU.

        Returns:
            new_det_inds: indices of new detections (not matched to any track)
            matched_pairs: list of (det_idx, obj_id) for matched pairs
        """
        new_det_inds = []
        matched_pairs = []

        if det_masks is None or det_scores is None:
            return new_det_inds, matched_pairs

        N_det = det_masks.shape[0]

        if trk_masks is None or len(tracked_objects) == 0:
            keep = det_scores >= self.new_det_thresh
            new_det_inds = torch.where(keep)[0].tolist() if isinstance(keep, torch.Tensor) else list(range(N_det))
            return new_det_inds, matched_pairs

        det_binary = (det_masks > 0).flatten(1).float()
        trk_binary = (trk_masks > 0).flatten(1).float()

        intersection = det_binary @ trk_binary.T
        det_area = det_binary.sum(dim=1, keepdim=True)
        trk_area = trk_binary.sum(dim=1, keepdim=True)
        union = det_area + trk_area.T - intersection
        iou = intersection / union.clamp(min=1.0)

        obj_ids = list(tracked_objects.keys())
        for d in range(N_det):
            max_iou, max_idx = iou[d].max(dim=0)
            if max_iou >= self.assoc_iou_thresh and max_idx < len(obj_ids):
                matched_pairs.append((d, obj_ids[max_idx.item()]))
            elif det_scores[d] >= self.new_det_thresh:
                new_det_inds.append(d)

        return new_det_inds, matched_pairs

    def _update_tracker(
        self, frame, frame_idx, tracker_state, tracked_objects,
        det_masks, det_scores, new_det_inds, matched_pairs,
        next_obj_id, num_frames,
    ):
        """Update tracker state with new and matched detections."""
        if tracker_state is None:
            tracker_state = self.tracker.init_state(
                video_height=frame.shape[-2],
                video_width=frame.shape[-1],
                num_frames=num_frames,
            )

        # Add new detections as new tracked objects
        for det_idx in new_det_inds:
            if det_masks is not None:
                mask = det_masks[det_idx]
                if mask.ndim == 3:
                    mask = mask[0]
                self.tracker.add_new_mask(tracker_state, frame_idx, next_obj_id, mask)
                tracked_objects[next_obj_id] = {
                    "start_frame": frame_idx,
                    "last_seen": frame_idx,
                }
                next_obj_id += 1

        # Update last_seen for matched objects
        for det_idx, obj_id in matched_pairs:
            if obj_id in tracked_objects:
                tracked_objects[obj_id]["last_seen"] = frame_idx

        return tracker_state, tracked_objects, next_obj_id

    def _build_frame_output(
        self, frame_idx, tracker_state, tracked_objects,
        trk_masks, det_masks, new_det_inds, matched_pairs,
    ):
        """Build output masks for a single frame."""
        obj_id_to_mask = {}

        if trk_masks is not None:
            for i, obj_id in enumerate(tracked_objects.keys()):
                if i < trk_masks.shape[0]:
                    mask = trk_masks[i]
                    if self.fill_hole_area > 0:
                        mask = fill_holes_in_mask_scores(mask.unsqueeze(0), max_area=self.fill_hole_area).squeeze(0)
                    obj_id_to_mask[obj_id] = mask

        return obj_id_to_mask
