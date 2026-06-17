"""Repo-native RTDetrV2 object detector (L4 wiring)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import RTDetrV2Config

from ..L3.rtdetrv2_model import RTDetrV2Model


def _boxes_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


class RTDetrV2ForObjectDetection(nn.Module):
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.config = config
        self.model = RTDetrV2Model(config)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "RTDetrV2ForObjectDetection":
        config = RTDetrV2Config.from_pretrained(model_name)
        model = cls(config)
        model_dir = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.bin", "*.json"])
        state_dict = {}
        safetensor_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            state_dict.update(load_file(safetensor_path))
        elif os.path.exists(bin_path):
            loaded = torch.load(bin_path, map_location="cpu")
            state_dict.update(loaded.get("state_dict", loaded))
        else:
            raise FileNotFoundError(f"No RTDetrV2 checkpoint found in {model_dir}")

        # L1 Embedding wraps nn.Embedding as self.emb, so remap keys
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            for emb_name in ("denoising_class_embed", "weight_embedding"):
                prefix = f"model.{emb_name}."
                if k.startswith(prefix) and ".emb." not in k:
                    new_k = k.replace(prefix, f"{prefix}emb.")
                    break
            remapped[new_k] = v
        state_dict = remapped

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if not (k.startswith("class_embed.") or k.startswith("bbox_embed."))]
        if missing or unexpected:
            raise RuntimeError(f"RTDetrV2 weight load mismatch: missing={missing}, unexpected={unexpected}")
        return model.to(device=device, dtype=dtype).eval()

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        logits = outputs.intermediate_logits[:, -1]
        pred_boxes = outputs.intermediate_reference_points[:, -1]
        return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)

    def predict(self, pixel_values: torch.Tensor, image_size: int, max_detections: int = 100):
        outputs = self.forward(pixel_values)
        logits = outputs.logits.float().sigmoid()
        pred_boxes = outputs.pred_boxes.float()
        scores, labels = logits.max(dim=-1)
        boxes = _boxes_cxcywh_to_xyxy(pred_boxes) * float(image_size)
        topk = min(max_detections, scores.shape[1])
        top_scores, top_indices = scores.topk(topk, dim=1)
        top_labels = labels.gather(1, top_indices)
        top_boxes = boxes.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 4))

        batch = scores.shape[0]
        padded_boxes = torch.zeros(batch, max_detections, 4, device=boxes.device)
        padded_scores = torch.zeros(batch, max_detections, device=boxes.device)
        padded_labels = torch.full((batch, max_detections), -1, device=boxes.device, dtype=torch.long)
        padded_boxes[:, :topk] = top_boxes
        padded_scores[:, :topk] = top_scores
        padded_labels[:, :topk] = top_labels
        return {"boxes": padded_boxes, "scores": padded_scores, "labels": padded_labels}


RTDetrV2ForObjectDetectionWrapper = RTDetrV2ForObjectDetection
