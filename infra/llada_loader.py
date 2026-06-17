"""Loader for repo-native LLaDA models."""

from __future__ import annotations

import os
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..tasks.baseline.L4.llada import LLaDAConfig, LLaDAModelLM


def download_model(model_name: str) -> str:
    return snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])


def load_weights(model: torch.nn.Module, model_path: str) -> None:
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    print(f"  Loading LLaDA weights from {len(safetensor_files)} safetensors file(s)...")
    loaded = 0
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                mapped_name = weight_name
                if mapped_name == "model.transformer.wte.weight":
                    mapped_name = "model.transformer.wte.emb.weight"
                mapped_name = mapped_name.replace(".q_proj.", ".attention.q_proj.")
                mapped_name = mapped_name.replace(".k_proj.", ".attention.k_proj.")
                mapped_name = mapped_name.replace(".v_proj.", ".attention.v_proj.")
                mapped_name = mapped_name.replace(".attn_out.", ".attention.attn_out.")
                mapped_name = mapped_name.replace(".ff_proj.", ".mlp.ff_proj.")
                mapped_name = mapped_name.replace(".up_proj.", ".mlp.up_proj.")
                mapped_name = mapped_name.replace(".ff_out.", ".mlp.ff_out.")
                mapped_name = mapped_name.replace(
                    "model.transformer.mlp.ff_out.",
                    "model.transformer.ff_out.",
                )
                try:
                    param = model.get_parameter(mapped_name)
                except AttributeError:
                    continue
                loaded_weight = f.get_tensor(weight_name).to(dtype=param.dtype)
                weight_loader_fn = getattr(param, "weight_loader", None)
                if callable(weight_loader_fn):
                    weight_loader_fn(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight.to(device=param.device))
                loaded += 1
    print(f"  Loaded {loaded} tensors")


def load_llada_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    tensor_parallel_size: int = 1,
):
    if tensor_parallel_size != 1:
        raise ValueError("LLaDA first-pass loader currently supports tensor_parallel_size=1 only.")

    config = LLaDAConfig.from_pretrained(model_name)
    config.dtype = dtype
    model = LLaDAModelLM(config).to(device=device, dtype=dtype)
    model.model.rebuild_rope_cache_fp32()
    model_path = download_model(model_name)
    load_weights(model, model_path)
    model.eval()
    return model
