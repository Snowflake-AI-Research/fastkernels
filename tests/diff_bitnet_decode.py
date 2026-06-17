"""Token-by-token decode comparison: kb-nano vs SOTA on a single prompt.

For one fixed prompt, generate N tokens greedily with each engine and
report (a) where the first divergence occurs and (b) how many tokens
match overall.  This complements ``diff_bitnet_layers.py`` which only
covers prefill.

Usage::

    BITNET_REPO=/home/yak/vllm_repo/BitNet \
        python tests/diff_bitnet_decode.py [--prompt-len 256] [--out-len 100]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
os.environ.setdefault("KB_NANO_DISABLE_FASTSAFETENSORS", "1")
sys.path.insert(0, str(_PROJECT_ROOT))


def run_kb_nano(prompt_ids: list[int], out_len: int) -> list[int]:
    from kb_nano.infra import weight_loader as _wl
    _wl._HAS_FASTSAFETENSORS = False
    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    print(f"[kb] loading engine...", flush=True)
    engine = LlamaEngine(
        model_name="microsoft/bitnet-b1.58-2B-4T",
        seed=0,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=4096,
    )
    print(f"[kb] generating...", flush=True)
    sps = [SamplingParams(temperature=0.0, max_tokens=out_len, ignore_eos=True)]
    outs = engine.generate([prompt_ids], sps, use_tqdm=False)
    return list(outs[0].token_ids)


def run_sota(prompt_ids: list[int], out_len: int, bitnet_repo: str) -> list[int]:
    sota_gpu = Path(bitnet_repo) / "gpu"
    sys.path.insert(0, str(sota_gpu))
    cwd_save = os.getcwd()
    os.chdir(sota_gpu)
    try:
        import generate as _bitnet_generate  # type: ignore

        in_len = len(prompt_ids)
        args = _bitnet_generate.GenArgs(
            prompt_length=in_len, gen_length=out_len, gen_bsz=1,
        )
        print(f"[sota] building FastGen for prompt_len={in_len}, out_len={out_len}...", flush=True)
        g = _bitnet_generate.FastGen.build("./checkpoints/", args, "cuda:0")
        g.tokenizer.eot_id = -1
        torch.cuda.synchronize()
        print(f"[sota] generating...", flush=True)
        _stats, answers = g.generate_all([prompt_ids], use_cuda_graphs=True, use_sampling=False)
        torch.cuda.synchronize()
        return list(answers[0])
    finally:
        os.chdir(cwd_save)
        sys.path.remove(str(sota_gpu))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitnet-repo", default=os.environ.get("BITNET_REPO"),
                    required=os.environ.get("BITNET_REPO") is None)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--out-len", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prompt_ids = [rng.randint(2, 50000) for _ in range(args.prompt_len)]

    sota_tokens = run_sota(prompt_ids, args.out_len, args.bitnet_repo)
    kb_tokens = run_kb_nano(prompt_ids, args.out_len)

    print(f"\nPrompt[:8] = {prompt_ids[:8]}")
    print(f"SOTA out  ({len(sota_tokens)}): {sota_tokens[:30]}")
    print(f"KB   out  ({len(kb_tokens)}): {kb_tokens[:30]}")

    # Find first divergence
    n = min(len(sota_tokens), len(kb_tokens))
    first_div = -1
    matches = 0
    for i in range(n):
        if sota_tokens[i] == kb_tokens[i]:
            matches += 1
        else:
            if first_div < 0:
                first_div = i
    print(f"\nfirst divergence at index: {first_div}")
    print(f"total matches: {matches}/{n}")
    if first_div >= 0:
        ctx_lo = max(0, first_div - 3)
        print(f"\nContext around divergence (idx {first_div}):")
        print(f"  SOTA: {sota_tokens[ctx_lo:first_div+5]}")
        print(f"  KB:   {kb_tokens[ctx_lo:first_div+5]}")


if __name__ == "__main__":
    main()
