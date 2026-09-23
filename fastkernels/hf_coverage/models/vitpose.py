"""VitPose's documented single-expert backbone and simple heatmap head."""

from torch import nn

from fastkernels.hf_coverage.models import vitpose_backbone
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.relu import ReLU


class VitPoseForPoseEstimation(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.backbone = vitpose_backbone.build_from_config(config.backbone_config, device, dtype)
        self.head = nn.Module()
        self.head.conv = Conv2d(config.backbone_config.hidden_size, len(config.id2label), 3, padding=1)
        self.activation, self.interpolate = ReLU(), Interpolate()
        self.scale_factor = config.scale_factor
        self.grid = tuple(image // patch for image, patch in zip(config.backbone_config.image_size,
                                                                config.backbone_config.patch_size))

    def forward(self, pixel_values):
        features = list(self.backbone(pixel_values).values())[-1]
        features = features.transpose(1, 2).reshape(pixel_values.shape[0], -1, *self.grid).contiguous()
        enlarged = self.interpolate(self.activation(features), scale_factor=self.scale_factor,
                                    mode="bilinear", align_corners=False)
        return {"heatmaps": self.head.conv(enlarged)}


def build_from_config(config, device, dtype):
    if not config.use_simple_decoder:
        raise ValueError("The documented VitPose checkpoint uses the simple decoder")
    return VitPoseForPoseEstimation(config, device, dtype).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    backbone = {name.removeprefix("backbone."): value for name, value in state_dict.items() if name.startswith("backbone.")}
    vitpose_backbone.load_state_dict_into(model.backbone, backbone, config.backbone_config)
    head = {name.removeprefix("head."): value for name, value in state_dict.items() if name.startswith("head.")}
    if len(backbone) + len(head) != len(state_dict):
        raise ValueError("Unmapped VitPose state outside backbone/head")
    model.head.load_state_dict(head, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
