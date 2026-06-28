"""Loader helpers for PointTransformerV3 benchmarking."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

from fastkernels import PROJECT_ROOT


DEFAULT_PTV3_CHECKPOINT_REPO = "Pointcept/PointTransformerV3"
DEFAULT_PTV3_CHECKPOINT_FILE = "scannet-semseg-pt-v3m1-0-base/model/model_best.pth"


def is_pointtransv3_model(model_name: str) -> bool:
    name = model_name.lower()
    return "pointtransformerv3" in name or "pointtransv3" in name or "ptv3" in name


def _project_root() -> Path:
    # Repo root, where external checkouts under third_party/ live.
    return PROJECT_ROOT


def _ptv3_repo_root() -> Path:
    return _project_root() / "third_party" / "PointTransformerV3"


def _install_torch_scatter_shim() -> None:
    if "torch_scatter" in sys.modules:
        return
    mod = types.ModuleType("torch_scatter")

    def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        reduce_map = {"sum": "sum", "mean": "mean", "min": "amin", "max": "amax"}
        return torch.segment_reduce(src, reduce=reduce_map[reduce], offsets=indptr, axis=0)

    mod.segment_csr = segment_csr
    sys.modules["torch_scatter"] = mod


def _load_official_ptv3_class():
    _install_torch_scatter_shim()
    repo_root = _ptv3_repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Missing PointTransformerV3 repo at {repo_root}. Clone https://github.com/Pointcept/PointTransformerV3 there."
        )
    package_name = "ptv3_ref"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(repo_root)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.model"
    if module_name in sys.modules:
        return sys.modules[module_name].PointTransformerV3
    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_root / "model.py",
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PointTransformerV3


def default_ptv3_kwargs(enable_flash: bool = False) -> dict:
    patch = (128, 128, 128, 128, 128)
    dec_patch = (128, 128, 128, 128)
    return {
        "enable_flash": enable_flash,
        "enc_patch_size": patch,
        "dec_patch_size": dec_patch,
        "shuffle_orders": False,
    }


def load_reference_point_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16, **model_kwargs):
    if not is_pointtransv3_model(model_name):
        raise ValueError(f"Unsupported point model: {model_name}")
    model_cls = _load_official_ptv3_class()
    model = model_cls(**model_kwargs).to(device=device, dtype=dtype).eval()
    return model


def load_ours_point_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16, **model_kwargs):
    if not is_pointtransv3_model(model_name):
        raise ValueError(f"Unsupported point model: {model_name}")
    from tasks.baseline.L4.pointtransformerv3 import PointTransformerV3

    model = PointTransformerV3(**model_kwargs).to(device=device, dtype=dtype).eval()
    return model


def resolve_point_checkpoint(checkpoint_file: str) -> str:
    local_path = Path(checkpoint_file).expanduser()
    if local_path.is_file():
        return str(local_path)

    if local_path.is_absolute():
        raise FileNotFoundError(f"PointTransformerV3 checkpoint not found: {checkpoint_file}")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        DEFAULT_PTV3_CHECKPOINT_REPO,
        checkpoint_file,
        repo_type="model",
    )


def load_point_backbone_checkpoint(model: torch.nn.Module, checkpoint_file: str) -> dict:
    checkpoint_path = resolve_point_checkpoint(checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported PointTransformerV3 checkpoint format: {type(state_dict)}")

    model_keys = set(model.state_dict().keys())
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("module.backbone."):
            backbone_state[key.removeprefix("module.backbone.")] = value
        elif key.startswith("backbone."):
            backbone_state[key.removeprefix("backbone.")] = value

    if not backbone_state:
        stripped_state = {key.removeprefix("module."): value for key, value in state_dict.items()}
        backbone_state = {key: value for key, value in stripped_state.items() if key in model_keys}

    if not backbone_state:
        raise ValueError(f"No PointTransformerV3 backbone weights found in checkpoint: {checkpoint_file}")

    missing, unexpected = model.load_state_dict(backbone_state, strict=True)
    return {
        "checkpoint_file": str(checkpoint_path),
        "loaded_tensors": len(backbone_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
