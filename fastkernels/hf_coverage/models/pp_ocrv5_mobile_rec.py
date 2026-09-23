"""PP-OCRv5 mobile backbone with the shared native SVTR recognition head."""

from .pp_lcnet import divisible
from .pp_lcnet_v3 import build_from_config as build_backbone
from .pp_ocrv5_server_rec import Recognition, load_state_dict_into, make_workloads


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu":
        raise ValueError("The selected OCR recognition configuration uses SiLU")
    backbone_config = config.backbone_config
    channels = divisible(backbone_config.block_configs[-1][-1][2] * backbone_config.scale, backbone_config.divisor)
    backbone = build_backbone(backbone_config, device, dtype)
    return Recognition(config, backbone, channels).to(device=device, dtype=dtype).eval()
