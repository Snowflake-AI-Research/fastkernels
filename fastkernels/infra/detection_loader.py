"""Loader helpers for object detection baselines and repo-native models."""

from __future__ import annotations

import torch


def is_yolov10_model(model_name: str) -> bool:
    return "yolov10" in model_name.lower()


def is_rtdetrv2_model(model_name: str) -> bool:
    name = model_name.lower()
    return "rtdetr_v2" in name or "rt-detr_v2" in name or "rtdetrv2" in name


def infer_image_size(model_name: str) -> int:
    if is_yolov10_model(model_name) or is_rtdetrv2_model(model_name):
        return 640
    raise ValueError(f"Unsupported detection model: {model_name}")


def _hf_name_to_ultralytics_pt(model_name: str) -> str:
    """Map a HuggingFace model id like 'jameslahm/yolov10n' to a temp path."""
    import tempfile
    from pathlib import Path

    base = model_name.split("/")[-1]
    if not base.endswith(".pt"):
        base += ".pt"
    cache_dir = Path(tempfile.gettempdir()) / "fastkernels" / "ultralytics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / base)


def _ultralytics_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        parts = device.split(":")
        if len(parts) == 2 and parts[1]:
            return parts[1]
        return "0"
    return device


def _boxes_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _standardize_yolov10_results(results, max_detections: int, image_size: int) -> dict:
    boxes_list = []
    scores_list = []
    labels_list = []
    device = None
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.xyxy.numel() == 0:
            device = device or torch.device("cpu")
            boxes_list.append(torch.zeros(max_detections, 4, device=device))
            scores_list.append(torch.zeros(max_detections, device=device))
            labels_list.append(torch.full((max_detections,), -1, device=device, dtype=torch.long))
            continue

        xyxy = boxes.xyxy.float()
        scores = boxes.conf.float()
        labels = boxes.cls.long()
        device = xyxy.device

        count = min(max_detections, xyxy.shape[0])
        padded_boxes = torch.zeros(max_detections, 4, device=device)
        padded_scores = torch.zeros(max_detections, device=device)
        padded_labels = torch.full((max_detections,), -1, device=device, dtype=torch.long)
        padded_boxes[:count] = xyxy[:count]
        padded_scores[:count] = scores[:count]
        padded_labels[:count] = labels[:count]
        boxes_list.append(padded_boxes)
        scores_list.append(padded_scores)
        labels_list.append(padded_labels)

    return {
        "boxes": torch.stack(boxes_list),
        "scores": torch.stack(scores_list),
        "labels": torch.stack(labels_list),
    }


def _standardize_yolov10_export(
    outputs: torch.Tensor,
    max_detections: int,
    conf_threshold: float = 0.25,
) -> dict:
    boxes = outputs[..., :4].float()
    scores = outputs[..., 4].float()
    labels = outputs[..., 5].long()

    batch = scores.shape[0]
    padded_boxes = torch.zeros(batch, max_detections, 4, device=boxes.device)
    padded_scores = torch.zeros(batch, max_detections, device=boxes.device)
    padded_labels = torch.full((batch, max_detections), -1, device=boxes.device, dtype=torch.long)
    for i in range(batch):
        keep = scores[i] > conf_threshold
        count = min(int(keep.sum().item()), max_detections)
        if count == 0:
            continue
        padded_boxes[i, :count] = boxes[i, keep][:count]
        padded_scores[i, :count] = scores[i, keep][:count]
        padded_labels[i, :count] = labels[i, keep][:count]

    return {
        "boxes": padded_boxes,
        "scores": padded_scores,
        "labels": padded_labels,
    }


def _standardize_rtdetrv2_outputs(outputs, image_size: int, max_detections: int) -> dict:
    logits = outputs.logits.float().sigmoid()
    pred_boxes = outputs.pred_boxes.float()
    scores, labels = logits.max(dim=-1)
    boxes = _boxes_cxcywh_to_xyxy(pred_boxes)
    boxes = boxes * float(image_size)

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
    return {
        "boxes": padded_boxes,
        "scores": padded_scores,
        "labels": padded_labels,
    }


def load_reference_detector(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    if is_yolov10_model(model_name):
        from ultralytics import YOLO

        pt_name = _hf_name_to_ultralytics_pt(model_name)
        model = YOLO(pt_name)
        core = model.model.float().eval().fuse(verbose=False)
        detect_head = core.model[-1]
        detect_head.export = True
        detect_head.format = ""
        core = core.to(device=device, dtype=dtype).eval()
        return core, "yolov10"

    if is_rtdetrv2_model(model_name):
        from transformers import RTDetrV2ForObjectDetection

        model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        model = model.to(device=device, dtype=dtype).eval()
        return model, "transformers"

    raise ValueError(f"Unsupported detection model: {model_name}")


def load_ours_detector(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    if is_yolov10_model(model_name):
        from fastkernels.tasks.baseline.L4.yolov10 import YOLOv10ForObjectDetection

        return YOLOv10ForObjectDetection.from_pretrained(
            model_name, device=device, dtype=dtype
        )

    if is_rtdetrv2_model(model_name):
        from fastkernels.tasks.baseline.L4.rtdetrv2 import RTDetrV2ForObjectDetection

        return RTDetrV2ForObjectDetection.from_pretrained(
            model_name, device=device, dtype=dtype
        )

    raise ValueError(f"Unsupported detection model: {model_name}")


def run_reference_detector(
    detector,
    model_name: str,
    pixel_values: torch.Tensor,
    image_size: int,
    max_detections: int = 100,
) -> dict:
    if is_yolov10_model(model_name):
        outputs = detector(pixel_values)
        return _standardize_yolov10_export(outputs, max_detections=max_detections, conf_threshold=0.25)

    if is_rtdetrv2_model(model_name):
        outputs = detector(pixel_values=pixel_values)
        return _standardize_rtdetrv2_outputs(outputs, image_size=image_size, max_detections=max_detections)

    raise ValueError(f"Unsupported detection model: {model_name}")


def run_ours_detector(
    detector,
    model_name: str,
    pixel_values: torch.Tensor,
    image_size: int,
    max_detections: int = 100,
) -> dict:
    if hasattr(detector, "predict"):
        return detector.predict(pixel_values=pixel_values, image_size=image_size, max_detections=max_detections)
    raise ValueError(f"Unsupported detection model: {model_name}")
