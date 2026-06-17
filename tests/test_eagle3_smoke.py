#!/usr/bin/env python3
"""Smoke test: load EAGLE-3 engine and generate a few tokens."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kb_nano.infra.eagle3_engine import LlamaEagle3Engine, Eagle3SamplingParams


def main():
    engine = LlamaEagle3Engine(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        draft_repo="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
        max_model_len=2048,
        max_num_seqs=4,
        spec_steps=3,
        spec_topk=4,
        num_draft_tokens=16,
        gpu_memory_utilization=0.85,
    )
    print("\n[smoke] Engine ready.\n")

    prompts = [
        "The capital of France is",
        "Once upon a time, in a small village,",
    ]
    sp = Eagle3SamplingParams(max_tokens=16)
    outs = engine.generate(prompts, sp, use_tqdm=False)

    for p, o in zip(prompts, outs):
        print(f"PROMPT: {p!r}")
        print(f"OUTPUT: {o.generated_text!r}")
        print(f"  num_gen_tokens = {len(o.token_ids)}")
        print()


if __name__ == "__main__":
    main()
