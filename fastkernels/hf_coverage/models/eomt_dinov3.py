"""EoMT-DINOv3's constructor-defined rotary encoder and mask-guided queries."""

from .eomt import _Eomt, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.use_gated_mlp or config.rope_parameters["rope_type"] != "default":
        raise ValueError("This case preserves the native GELU MLP and default two-dimensional RoPE")
    model = _Eomt(config, rotary=True)
    inv_freq = model.rope.inv_freq
    model.to(device=device, dtype=dtype)
    model.rope.inv_freq = inv_freq.to(device=device)
    return model.eval()
