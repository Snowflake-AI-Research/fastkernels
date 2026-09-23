"""Table Transformer's default ResNet18 and pre-normalized detection transformer."""

import re

from .detr import _ObjectDetection, _check_config, make_workloads


def build_from_config(config, device, dtype):
    _check_config(config)
    return _ObjectDetection(config, prenorm=True, backend="eager").to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace(".backbone.conv_encoder.", ".backbone.")
        name = name.replace(".query_position_embeddings.weight", ".query_position_embeddings.emb.weight")
        name = re.sub(r"(\.layers\.\d+)\.(fc[12])\.", r"\1.mlp.\2.", name)
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
