"""Loader helpers for image classification baselines and repo-native models."""

from __future__ import annotations

from typing import Any

import torch


def _strip_timm_prefix(model_name: str) -> str:
    return model_name.split("/", 1)[1] if model_name.startswith("timm/") else model_name


def is_convnextv2_model(model_name: str) -> bool:
    return "convnextv2" in model_name.lower()


def is_efficientnetv2_model(model_name: str) -> bool:
    return "efficientnetv2" in model_name.lower()


def infer_image_size(model_name: str) -> int:
    if is_convnextv2_model(model_name):
        return 384
    if is_efficientnetv2_model(model_name):
        import timm

        ref = timm.create_model(_strip_timm_prefix(model_name), pretrained=False)
        return int(ref.default_cfg["input_size"][1])
    raise ValueError(f"Unsupported image classification model: {model_name}")


def infer_image_mean_std(model_name: str) -> tuple[list[float], list[float]]:
    if is_convnextv2_model(model_name):
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = [float(x) for x in processor.image_mean]
        std = [float(x) for x in processor.image_std]
        return mean, std
    if is_efficientnetv2_model(model_name):
        import timm

        ref = timm.create_model(_strip_timm_prefix(model_name), pretrained=False)
        mean = [float(x) for x in ref.default_cfg["mean"]]
        std = [float(x) for x in ref.default_cfg["std"]]
        return mean, std
    raise ValueError(f"Unsupported image classification model: {model_name}")


def load_reference_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    if is_convnextv2_model(model_name):
        from transformers import ConvNextV2ForImageClassification

        model = ConvNextV2ForImageClassification.from_pretrained(model_name)
        model = model.to(device=device, dtype=dtype).eval()
        return model, "transformers"
    if is_efficientnetv2_model(model_name):
        import timm

        model = timm.create_model(_strip_timm_prefix(model_name), pretrained=True)
        model = model.to(device=device, dtype=dtype).eval()
        return model, "timm"
    raise ValueError(f"Unsupported image classification model: {model_name}")


def _extract_efficientnetv2_stage_specs(reference_model) -> list[list[dict[str, Any]]]:
    stage_specs: list[list[dict[str, Any]]] = []
    for stage in reference_model.blocks:
        specs = []
        for block in stage:
            if hasattr(block, "conv_exp"):
                spec = {
                    "kind": "edge",
                    "in_chs": block.conv_exp.weight.shape[1],
                    "exp_chs": block.conv_exp.weight.shape[0],
                    "out_chs": block.conv_pwl.weight.shape[0],
                    "stride": block.conv_exp.stride[0],
                    "has_skip": bool(block.has_skip),
                }
            else:
                se_reduce_chs = block.se.conv_reduce.weight.shape[0] if hasattr(block.se, "conv_reduce") else 0
                spec = {
                    "kind": "inverted",
                    "in_chs": block.conv_pw.weight.shape[1],
                    "exp_chs": block.conv_pw.weight.shape[0],
                    "out_chs": block.conv_pwl.weight.shape[0],
                    "stride": block.conv_dw.stride[0],
                    "se_reduce_chs": se_reduce_chs,
                    "has_skip": bool(block.has_skip),
                }
            specs.append(spec)
        stage_specs.append(specs)
    return stage_specs


def load_ours_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    if is_convnextv2_model(model_name):
        from transformers import ConvNextV2Config, ConvNextV2ForImageClassification as HFConvNextV2ForImageClassification
        try:
            from kb_nano.tasks.baseline.L4.convnextv2 import ConvNextV2ForImageClassification
        except ModuleNotFoundError:
            from tasks.baseline.L4.convnextv2 import ConvNextV2ForImageClassification

        config = ConvNextV2Config.from_pretrained(model_name)
        model = ConvNextV2ForImageClassification(config)
        reference = HFConvNextV2ForImageClassification.from_pretrained(model_name)
        model.load_state_dict(reference.state_dict(), strict=True)
        model = model.to(device=device, dtype=dtype).eval()
        return model

    if is_efficientnetv2_model(model_name):
        import timm
        try:
            from kb_nano.tasks.baseline.L4.efficientnetv2 import EfficientNetV2ForImageClassification
        except ModuleNotFoundError:
            from tasks.baseline.L4.efficientnetv2 import EfficientNetV2ForImageClassification

        reference = timm.create_model(_strip_timm_prefix(model_name), pretrained=True)
        stage_specs = _extract_efficientnetv2_stage_specs(reference)
        model = EfficientNetV2ForImageClassification(
            stem_out=reference.conv_stem.weight.shape[0],
            stage_specs=stage_specs,
            head_out=reference.conv_head.weight.shape[0],
            num_classes=reference.classifier.weight.shape[0],
        )
        model.load_state_dict(reference.state_dict(), strict=True)
        model = model.to(device=device, dtype=dtype).eval()
        return model

    raise ValueError(f"Unsupported image classification model: {model_name}")
