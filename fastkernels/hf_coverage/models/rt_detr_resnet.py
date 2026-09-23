"""Default RT-DETR ResNet backbone using the unchanged FastKernels backbone."""

from ..runner import Workload
from fastkernels.tasks.baseline.L3.rtdetrv2_backbone import RTDetrV2ResNetBackbone


def build_from_config(config, device, dtype):
    if config.layer_type != "bottleneck" or config.hidden_act != "relu":
        raise ValueError("RT-DETR ResNet coverage uses the default ReLU bottleneck path")
    return RTDetrV2ResNetBackbone(config, frozen_batch_norm=False).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    def forward():
        maps = model(inputs["pixel_values"]).feature_maps
        return {f"feature_maps.{index}": value for index, value in enumerate(maps)}

    return {"forward": Workload(run=forward)}
