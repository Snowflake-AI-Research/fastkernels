"""Granite shared MoE adds the existing unconditionally evaluated shared SwiGLU."""

from fastkernels.hf_coverage.models.granitemoe import build_from_config as build_granite
from fastkernels.hf_coverage.models.granitemoe import load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    model = build_granite(config, device, dtype)
    for layer in model.model.layers:
        # HF materializes the first projection before SiLU. The existing
        # separate expert backend preserves this boundary; TRT fuses it.
        # Output weighting/reduction still differ and require measured checks.
        layer.mlp.use_trtllm = False
    return model
