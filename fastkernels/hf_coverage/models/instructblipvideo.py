"""The documented four-frame InstructBLIP extension using the same carriers."""

from .instructblip import build, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    return build(config, device, dtype, video=True)
