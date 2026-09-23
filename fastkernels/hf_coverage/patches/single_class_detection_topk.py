"""Singleton-class specialization of L3.yolov10_head.v10postprocess.

Keep the parent's first score reduction, top-k, and box/score gathers. With
one class, its second top-k selects the same entries again but can permute
exact ties. Return the first selection instead; its order follows PyTorch's
same-device top-k behavior, not a stable or device-independent tie rule.
"""

import torch


def v10postprocess_single_class(preds: torch.Tensor, max_det: int, nc: int = 1):
    if nc != 1:
        raise ValueError("This detection-selector specialization requires nc=1")
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))
    return boxes, scores.squeeze(-1), torch.zeros_like(index.squeeze(-1))
