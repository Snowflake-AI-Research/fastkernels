"""VaultGemma constructor-derived SDPA graph, with no post-branch norms.

This source scope has alternating sliding/full layers and an 8192-token context;
it does not stand in for the model-documentation all-full 1024-token checkpoint.
"""
from .gemma2 import build_from_config as build_gemma2
from .gemma2 import load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    return build_gemma2(config, device, dtype, post_branch_norms=False)
