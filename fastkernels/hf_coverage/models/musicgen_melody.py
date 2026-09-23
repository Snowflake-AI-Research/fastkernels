"""MusicGen Melody's full text/chroma-prefix public logits forward."""

from .musicgen import build_from_config as _build, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    return _build(config, device, dtype, melody=True)
