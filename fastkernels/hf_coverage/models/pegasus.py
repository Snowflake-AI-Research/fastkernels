"""Pegasus conditional generation with frozen sinusoidal state and ReLU blocks."""

from fastkernels.hf_coverage.models.mbart import build_pre_norm, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    return build_pre_norm(config, device, dtype, learned_positions=False)
