"""Donut's Swin encoder retains pooling but omits Swin's final normalization."""

from torch import nn

from .swin import SwinModel, _check_config, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    _check_config(config)
    model = SwinModel(config)
    model.norm = nn.Identity()
    return model.to(device=device, dtype=dtype).eval()
