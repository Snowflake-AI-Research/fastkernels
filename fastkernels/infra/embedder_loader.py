"""Weight loader for encoder-only embedding models."""

from __future__ import annotations

import os
import re
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

try:
    from ..tasks.baseline.L4.bge_m3 import BGEM3Config, BgeM3EmbeddingModel
    from ..tasks.baseline.L4.colbertv2 import ColBERTModel, ColBERTv2Config
except ImportError as exc:
    if "attempted relative import beyond top-level package" not in str(exc):
        raise
    from tasks.baseline.L4.bge_m3 import BGEM3Config, BgeM3EmbeddingModel
    from tasks.baseline.L4.colbertv2 import ColBERTModel, ColBERTv2Config


_EMBEDDING_WEIGHT_RE = re.compile(
    r"(.+\.(?:word_embeddings|position_embeddings|token_type_embeddings))\.weight$",
)


def download_embedder_model(model_name: str) -> str:
    return snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json", "*.metadata"],
    )


def _load_backbone_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        state: dict[str, torch.Tensor] = {}
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    state[key] = handle.get_tensor(key)
        return state

    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No backbone weight file found in {model_path}")


def _remap_encoder_embedding_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = _EMBEDDING_WEIGHT_RE.match(key)
        if match:
            remapped[f"{match.group(1)}.emb.weight"] = value
        else:
            remapped[key] = value
    return remapped


def _prefix_model_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in ("bert.", "roberta."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        remapped[key if key.startswith("model.") else f"model.{key}"] = value
    return remapped


def load_bge_m3_weights(model: BgeM3EmbeddingModel, model_path: str) -> None:
    backbone_state = _prefix_model_keys(
        _remap_encoder_embedding_keys(_load_backbone_state_dict(model_path)),
    )
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)

    sparse_path = os.path.join(model_path, "sparse_linear.pt")
    if not os.path.exists(sparse_path):
        raise FileNotFoundError(f"Missing sparse head weights: {sparse_path}")
    sparse_state = torch.load(sparse_path, map_location="cpu", weights_only=True)
    model.sparse_linear.load_state_dict(sparse_state)

    colbert_path = os.path.join(model_path, "colbert_linear.pt")
    if not os.path.exists(colbert_path):
        raise FileNotFoundError(f"Missing ColBERT head weights: {colbert_path}")
    colbert_state = torch.load(colbert_path, map_location="cpu", weights_only=True)
    model.colbert_linear.load_state_dict(colbert_state)

    missing = [
        key for key in missing
        if key not in {
            "model.embeddings.position_ids",
            "model.embeddings.token_type_ids",
            "sparse_linear.weight",
            "sparse_linear.bias",
            "colbert_linear.weight",
            "colbert_linear.bias",
        }
    ]
    if missing:
        raise RuntimeError(f"Unexpected missing BGE-M3 weights: {missing}")
    unexpected = [
        key for key in unexpected
        if key not in {
            "model.pooler.dense.weight",
            "model.pooler.dense.bias",
        }
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected BGE-M3 checkpoint keys: {unexpected}")


def load_bge_m3_model(
    model_name: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[BgeM3EmbeddingModel, BGEM3Config]:
    model_path = download_embedder_model(model_name)
    config = BGEM3Config.from_pretrained(model_name)
    config.dtype = dtype
    model = BgeM3EmbeddingModel(config)
    load_bge_m3_weights(model, model_path)
    model = model.to(device=device, dtype=dtype)
    model.sparse_linear.to(device=device, dtype=torch.float32)
    model.colbert_linear.to(device=device, dtype=torch.float32)
    model.eval()
    return model, config


def load_colbertv2_weights(model: ColBERTModel, model_path: str) -> None:
    state = _prefix_model_keys(
        _remap_encoder_embedding_keys(_load_backbone_state_dict(model_path)),
    )
    if "linear.weight" in state:
        state["colbert_linear.weight"] = state.pop("linear.weight")
    if "model.linear.weight" in state:
        state["colbert_linear.weight"] = state.pop("model.linear.weight")
    if "model.colbert_linear.weight" in state:
        state["colbert_linear.weight"] = state.pop("model.colbert_linear.weight")
    missing, unexpected = model.load_state_dict(state, strict=False)

    missing = [
        key for key in missing
        if key not in {
            "model.embeddings.position_ids",
            "colbert_linear.weight",
        }
    ]
    if missing:
        raise RuntimeError(f"Unexpected missing ColBERTv2 weights: {missing}")

    unexpected = [
        key for key in unexpected
        if key not in {
            "model.embeddings.position_ids",
            "model.pooler.dense.weight",
            "model.pooler.dense.bias",
        }
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected ColBERTv2 checkpoint keys: {unexpected}")


def load_colbertv2_model(
    model_name: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ColBERTModel, ColBERTv2Config]:
    model_path = download_embedder_model(model_name)
    config = ColBERTv2Config.from_pretrained(model_name)
    config.dtype = dtype
    model = ColBERTModel(config)
    load_colbertv2_weights(model, model_path)
    model = model.to(device=device, dtype=dtype)
    model.colbert_linear.to(device=device, dtype=torch.float32)
    model.eval()
    return model, config
