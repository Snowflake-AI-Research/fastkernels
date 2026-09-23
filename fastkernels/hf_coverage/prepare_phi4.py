"""Prepare shared random weights and synthetic inputs for Phi4 speech on CPU.

The fixed seeds reproduce the reviewed speech workload. No pretrained weights,
processor assets, or prior audit files are needed. Vision remains outside this
preparer because its pinned HF position indexing requires separate review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare(directory: Path) -> dict:
    import torch

    from .cases import CASES
    from .reference import load_reference_model, merge_config_values, verify_reference_pin
    from .runner import digest, symbol

    package = Path(__file__).resolve().parent
    corpus = json.loads((package / "corpus.json").read_text())
    pin = verify_reference_pin(corpus["transformers_revision"])
    case = CASES["phi4_multimodal"]["variants"]["speech"]
    reference = case["reference"]
    source = reference["source"]
    config_class = symbol(reference["config_class"])
    raw, _ = config_class.get_config_dict(source["checkpoint"], revision=source["revision"])
    converted = symbol(source["config_converter"])(raw).to_dict()
    config = config_class(**merge_config_values(converted, case["dimension_overrides"]))
    model_class = symbol(reference["model_class"])

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(43)
    model = model_class(config).eval()
    # Retain the reviewed BF16-rounded random base for both BF16 execution and
    # subsequent FP32 diagnosis. These are not checkpoint weights.
    weights = {name: value.detach().to(torch.bfloat16).clone()
               for name, value in model.state_dict().items()}
    del model
    projections = ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                   "mlp.gate_up_proj.weight", "mlp.down_proj.weight")
    rank = reference["adapter_config"]["r"]
    for name in list(weights):
        if name.startswith("model.layers.") and name.endswith(projections):
            weight = weights.pop(name)
            prefix = name.removesuffix("weight")
            weights[prefix + "base_layer.weight"] = weight
            weights[prefix + "lora_A.default.weight"] = torch.empty(rank, weight.shape[1])
            weights[prefix + "lora_B.default.weight"] = torch.empty(weight.shape[0], rank)
    generator = torch.Generator().manual_seed(3)
    for name in sorted(weights):
        if ".lora_" in name:
            shape = weights[name].shape
            weights[name] = (torch.randn(shape, generator=generator)
                             * shape[1] ** -0.5).to(torch.bfloat16)

    # Use the existing native loading checks, including complete adapter loading.
    model, loading = load_reference_model(
        model_class, config, weights, torch.bfloat16,
        reference_backend=case["reference_backend"],
        adapter_config=reference["adapter_config"],
        generation_config=reference["generation_config"],
    )
    weights = dict(model.state_dict())
    torch.save(weights, directory / "weights.pt")
    del model, weights

    # The native encoder unfolds after 500 subsampled frames. One extra frame
    # exercises a second chunk and its padding without changing that boundary.
    audio = config.audio_config
    if audio.downsample_rate != 1:
        raise ValueError("The selected single-clip speech path requires downsample_rate=1")
    feature_count = 501
    frames = feature_count * audio.time_reduction
    # Ordinary synthetic text tokens surround the required audio placeholders.
    ids = torch.tensor([[config.bos_token_id, 37]
                        + [audio.audio_token_id] * feature_count + [39, 40]])
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "audio_input_features": torch.randn(
            1, frames, audio.input_size, generator=torch.Generator().manual_seed(1),
        ).to(torch.bfloat16),
        "audio_embed_sizes": torch.tensor([feature_count]),
    }
    torch.save(inputs, directory / "inputs.pt")
    record = {
        "model": "phi4_multimodal", "variant": "speech", "reference_pin": pin,
        "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        "configuration_source": source, "dimension_overrides": case["dimension_overrides"],
        "seeds": {"base": 43, "adapters": 3, "inputs": 1},
        "initialization": "Native FP32 random base rounded to BF16; sorted LoRA A/B matrices use independent normal values with standard deviation 1/sqrt(input width), rounded to BF16",
        "adapter_config": reference["adapter_config"], "loading_info": loading,
        "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in inputs.items()},
        "weights_sha256": digest(directory / "weights.pt"),
        "inputs_sha256": digest(directory / "inputs.pt"),
        "scope": "CPU preparation only; not numerical acceptance or a GPU result",
    }
    (directory / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new or empty directory in your storage")
    parser.add_argument("--hf-source", default=os.environ.get("HF_COVERAGE_REFERENCE_SOURCE"),
                        help="pinned Transformers checkout or its source directory")
    args = parser.parse_args()
    directory = Path(args.output_dir).expanduser().resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("--output-dir must be a new or empty directory")
    if args.hf_source:
        source = Path(args.hf_source).expanduser().resolve()
        source = source / "src" if (source / "src" / "transformers").is_dir() else source
        if not (source / "transformers" / "__init__.py").is_file():
            parser.error("--hf-source must contain the pinned Transformers source")
        sys.path.insert(0, str(source))
    directory.mkdir(parents=True, exist_ok=True)
    prepare(directory)
    print(f"Prepared Phi4 speech weights.pt, inputs.pt, and preparation.json in {directory}")


if __name__ == "__main__":
    main()
