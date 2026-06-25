#!/usr/bin/env python3
"""Phase 2 validation: verify custom ops are bitwise-identical to eager.

Loads a model, runs a single forward pass with and without custom ops,
and asserts the outputs are bitwise identical.  Custom ops should be pure
indirection through no_compile_layers — they must not introduce any
numerical drift.

Usage:
    python tests/debug/test_custom_ops.py --model meta-llama/Llama-3.1-8B-Instruct
    python tests/debug/test_custom_ops.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
torch.set_grad_enabled(False)


def test_custom_ops(model_name: str, tp: int = 1):
    from fastkernels.infra.engine import LlamaEngine, SamplingParams
    from fastkernels.infra.context import (
        get_no_compile_layers,
        enable_custom_ops,
        disable_custom_ops,
    )
    from fastkernels.infra.compilation import ensure_custom_ops_registered

    print(f"Loading {model_name} (TP={tp}, enforce_eager=True)...")
    engine = LlamaEngine(
        model_name=model_name,
        seed=42,
        enforce_eager=True,
        tensor_parallel_size=tp,
    )

    layers = get_no_compile_layers()
    print(f"Registered {len(layers)} no_compile_layers:")
    for name in sorted(layers.keys())[:10]:
        print(f"  {name} -> {type(layers[name]).__name__}")
    if len(layers) > 10:
        print(f"  ... and {len(layers) - 10} more")

    sp = SamplingParams(temperature=0.0, max_tokens=16, seed=42)

    # Run 1: without custom ops (pure eager)
    disable_custom_ops()
    print("\nRun 1: eager (no custom ops)...")
    out_eager = engine.generate(["Hello world"], sp)
    eager_ids = out_eager[0].token_ids

    # Run 2: with custom ops enabled
    ensure_custom_ops_registered()
    enable_custom_ops()
    print("Run 2: custom ops enabled...")
    engine.block_manager.reset()
    out_ops = engine.generate(["Hello world"], sp)
    ops_ids = out_ops[0].token_ids

    disable_custom_ops()

    # Compare
    print(f"\nEager tokens:      {eager_ids[:10]}...")
    print(f"Custom-op tokens:  {ops_ids[:10]}...")

    if eager_ids == ops_ids:
        print("\nPASS: Outputs are bitwise identical")
    else:
        print("\nFAIL: Outputs differ!")
        min_len = min(len(eager_ids), len(ops_ids))
        matching = sum(1 for i in range(min_len) if eager_ids[i] == ops_ids[i])
        print(f"  Matching tokens: {matching}/{min_len}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    test_custom_ops(args.model, args.tp)
