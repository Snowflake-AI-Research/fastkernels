"""The public HF wrapper example's ResNet50, including its default pooler."""

from .timm_backbone import resnet50, load_resnet50
from ..runner import Workload


def build_from_config(config, device, dtype):
    if config.architecture != "resnet50" or not config.do_pooling or config.model_args:
        raise ValueError("Preserve the public task example's plain ResNet50 and default pooling")
    return resnet50(device, dtype)


def load_state_dict_into(model, state_dict, config):
    load_resnet50(model, state_dict, "timm_model.")


def make_workloads(model, inputs, config):
    def forward():
        output = model(**inputs)
        output["pooler_output"] = output["pooler_output"].flatten(1)
        return output
    return {"forward": Workload(run=forward)}
