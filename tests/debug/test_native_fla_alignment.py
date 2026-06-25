#!/usr/bin/env python3
"""
Correctness test: our task implementation vs native FLA reference.

Modes:
    --ref-only   : Run only the FLA reference and print output.
    (default)    : Run both FLA reference and our model, compare token-for-token.

Usage:
    # Full alignment test
    python tests/debug/test_native_fla_alignment.py \
        --model fla-hub/gla-2.7B-100B --max-tokens 32 --seed 42

    # Reference only
    python tests/debug/test_native_fla_alignment.py \
        --model fla-hub/gla-2.7B-100B --ref-only --max-tokens 32 --seed 42
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' into French, German, and Japanese.",
    (
        "Explain the difference between a stack and a queue in computer "
        "science. Give a real-world analogy for each."
    ),
    (
        "Write a Python function that computes the factorial of a number "
        "using recursion. Include a docstring."
    ),
]

SUPPORTED_NATIVE_FLA_MODELS = [
    "fla-hub/gla-2.7B-100B",
    "fla-hub/rwkv7-2.9B-g1",
    "fla-hub/rwkv7-2.9B-world",
    "fla-hub/retnet-2.7B-100B",
]


# ---------------------------------------------------------------------------
# FLA reference worker (runs in subprocess)
# ---------------------------------------------------------------------------
NATIVE_FLA_WORKER = r'''
import json
import sys
import torch


def _resolve_dtype(name: str):
    name = (name or "bfloat16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_native_model(model_name: str, trust_remote_code: bool, dtype_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = _resolve_dtype(dtype_name)
    if model_name == "fla-hub/gla-2.7B-100B":
        from fla.models.gla import GLAForCausalLM
        model = GLAForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    elif model_name in ("fla-hub/rwkv7-2.9B-g1", "fla-hub/rwkv7-2.9B-world"):
        from fla.models.rwkv7 import RWKV7ForCausalLM
        model = RWKV7ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    elif model_name == "fla-hub/retnet-2.7B-100B":
        from fla.models.retnet import RetNetForCausalLM
        model = RetNetForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(
            f"Unsupported model for native FLA probe: {model_name}. "
            "Supported: fla-hub/gla-2.7B-100B, "
            "fla-hub/rwkv7-2.9B-g1, "
            "fla-hub/rwkv7-2.9B-world, "
            "fla-hub/retnet-2.7B-100B"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    cfg = json.loads(sys.argv[1])
    model_name = cfg["model"]
    seed = int(cfg["seed"])
    max_tokens = int(cfg["max_tokens"])
    prompts = cfg["prompts"]
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    dtype_name = cfg.get("dtype", "bfloat16")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer, device = _load_native_model(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        dtype_name=dtype_name,
    )

    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output[0][input_ids.shape[1]:].tolist()
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append({
            "text": text,
            "token_ids": generated_ids,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "ok": True,
            "model_class": type(model).__name__,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "results": results,
        }, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Standalone (our implementation) worker (runs in subprocess)
# ---------------------------------------------------------------------------
STANDALONE_WORKER = r'''
import json
import os
import sys
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer


def _resolve_dtype(name: str):
    name = (name or "bfloat16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    cfg = json.loads(sys.argv[1])
    model_name = cfg["model"]
    seed = int(cfg["seed"])
    max_tokens = int(cfg["max_tokens"])
    prompts = cfg["prompts"]
    project_root = cfg["project_root"]
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    dtype_name = cfg.get("dtype", "bfloat16")
    dtype = _resolve_dtype(dtype_name)

    sys.path.insert(0, project_root)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Download model and load config
    model_path = snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )

    # Import our model implementation
    if model_name == "fla-hub/gla-2.7B-100B":
        from fastkernels.tasks.baseline.L4.gla import GLAConfig, GLAForCausalLM
        config = GLAConfig.from_pretrained(model_path)
        model = GLAForCausalLM(config)
    elif model_name == "fla-hub/retnet-2.7B-100B":
        from fastkernels.tasks.baseline.L4.retnet import RetNetConfig, RetNetForCausalLM
        config = RetNetConfig.from_pretrained(model_path)
        model = RetNetForCausalLM(config)
    elif model_name in ("fla-hub/rwkv7-2.9B-g1", "fla-hub/rwkv7-2.9B-world"):
        from fastkernels.tasks.baseline.L4.rwkv7 import RWKV7Config, RWKV7ForCausalLM
        config = RWKV7Config.from_pretrained(model_path)
        model = RWKV7ForCausalLM(config)
    else:
        raise ValueError(f"Unsupported model for standalone worker: {model_name}")

    # FLA checkpoints store the token embedding at ``model.embeddings.weight``;
    # our L1 Embedding nests ``nn.Embedding`` as ``self.emb``, so remap.
    def _remap(name):
        if name == "model.embeddings.weight":
            return "model.embeddings.emb.weight"
        return name

    sf_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    loaded = 0
    for sf in sf_files:
        with safe_open(sf, "pt", "cpu") as f:
            for name in f.keys():
                mapped = _remap(name)
                try:
                    param = model.get_parameter(mapped)
                    param.data.copy_(f.get_tensor(name))
                    loaded += 1
                except AttributeError:
                    pass
    print(f"  [Ours] Loaded {loaded} weights", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        context = input_ids[0].tolist()

        with torch.no_grad():
            for _ in range(max_tokens):
                ids_tensor = torch.tensor([context], device=device)
                output = model(input_ids=ids_tensor, use_cache=False)
                logits = output.logits[:, -1, :]
                next_id = int(logits.argmax(dim=-1).item())
                context.append(next_id)
                if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                    break

        generated_ids = context[input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append({
            "text": text,
            "token_ids": generated_ids,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def run_worker(script: str, config: dict, label: str) -> dict | None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
        f.write(script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    config["output_file"] = output_path
    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")
        result = subprocess.run(
            [sys.executable, script_path, json.dumps(config)],
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed with exit code {result.returncode}")
            return None
        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_correctness(
    ref_data: dict, standalone_data: dict, prompts: list[str],
) -> int:
    """Print correctness comparison. Returns number of mismatches."""
    print(f"\n{'=' * 70}")
    print("  CORRECTNESS")
    print(f"{'=' * 70}")

    mismatches = 0
    for i, (rr, sr) in enumerate(
        zip(ref_data["results"], standalone_data["results"])
    ):
        r_ids = rr["token_ids"]
        s_ids = sr["token_ids"]
        match = r_ids == s_ids

        prompt_preview = prompts[i][:55] + ("..." if len(prompts[i]) > 55 else "")
        ntoks = len(r_ids)

        if match:
            print(f"  #{i} MATCH   ({ntoks:>3} tokens) | {prompt_preview}")
        else:
            mismatches += 1
            min_len = min(len(r_ids), len(s_ids))
            div = next(
                (j for j in range(min_len) if r_ids[j] != s_ids[j]), min_len
            )
            print(f"  #{i} MISMATCH at token {div:>3} | {prompt_preview}")
            print(f"       FLA  : {rr['text'][:70]!r}...")
            print(f"       Ours : {sr['text'][:70]!r}...")

    total = len(prompts)
    print(f"\n  Result: {total - mismatches}/{total} exact matches")
    return mismatches


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_ref_model(model_name: str, args) -> bool:
    """Run FLA reference only and print full generated text."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'#' * 70}")
    print(f"  MODEL: {model_name}")
    print(f"  max_tokens={args.max_tokens}  seed={args.seed}  dtype={args.dtype}")
    print(f"{'#' * 70}")

    config = {
        "model": model_name,
        "seed": args.seed,
        "prompts": PROMPTS,
        "max_tokens": args.max_tokens,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
    }
    ref_data = run_worker(
        NATIVE_FLA_WORKER, dict(config),
        f"Native FLA  [{short_name}] (load+generate)",
    )
    if ref_data is None:
        print(f"\n  FAIL: Native FLA run failed for {short_name}")
        return False

    print(f"\n{'=' * 70}")
    print(f"  NATIVE FLA OUTPUT — {model_name}")
    print(f"{'=' * 70}")
    print(f"  Model class         : {ref_data.get('model_class', 'unknown')}")
    print(f"  Device              : {ref_data.get('device', 'unknown')}")
    print(f"  DType               : {ref_data.get('dtype', 'unknown')}")

    for i, r in enumerate(ref_data["results"]):
        prompt_preview = PROMPTS[i][:60] + ("..." if len(PROMPTS[i]) > 60 else "")
        print(f"\n  Prompt #{i}: {prompt_preview}")
        print(f"  Tokens   : {len(r['token_ids'])}")
        print("  Output   :")
        for line in r["text"].splitlines():
            print(f"    {line}")
    print(f"\n{'=' * 70}")
    return True


def run_model_test(model_name: str, args, project_root: str) -> int:
    """Run correctness test for a single model. Returns number of mismatches."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'#' * 70}")
    print(f"  MODEL: {model_name}")
    print(f"  max_tokens={args.max_tokens}  seed={args.seed}  dtype={args.dtype}")
    print(f"{'#' * 70}")

    config = {
        "model": model_name,
        "seed": args.seed,
        "prompts": PROMPTS,
        "max_tokens": args.max_tokens,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "project_root": project_root,
    }

    ref_data = run_worker(
        NATIVE_FLA_WORKER, dict(config),
        f"FLA reference  [{short_name}]",
    )
    standalone_data = run_worker(
        STANDALONE_WORKER, dict(config),
        f"Ours  [{short_name}]",
    )

    if ref_data is None or standalone_data is None:
        print(f"\n  ERROR: One or both sides failed for {short_name}.")
        return len(PROMPTS)

    mismatches = report_correctness(ref_data, standalone_data, PROMPTS)

    if mismatches == 0:
        print(f"\n  PASS [{short_name}]: All outputs are token-identical.")
    else:
        print(f"\n  WARN [{short_name}]: {mismatches}/{len(PROMPTS)} prompts had mismatches.")

    return mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Correctness test: our task implementation vs native FLA reference",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=[SUPPORTED_NATIVE_FLA_MODELS[0]],
        help=f"Model(s) to test. Supported now: {', '.join(SUPPORTED_NATIVE_FLA_MODELS)}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per prompt (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True when loading tokenizer/model (default: True)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code",
    )
    parser.add_argument(
        "--ref-only",
        action="store_true",
        default=False,
        help="Only run the FLA reference (no comparison)",
    )
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(this_dir))

    models = args.model
    mode = "Native FLA Reference Output" if args.ref_only else "Ours vs Native FLA — Correctness Test"
    print("=" * 70)
    print(f"  {mode}")
    print("=" * 70)
    print(f"  Models     : {', '.join(models)}")
    print(f"  Max tokens : {args.max_tokens}")
    print(f"  Seed       : {args.seed}")
    print(f"  DType      : {args.dtype}")
    print(f"  Prompts    : {len(PROMPTS)}")
    print(f"  Trust RC   : {args.trust_remote_code}")
    print("=" * 70)

    if args.ref_only:
        results = {}
        for model_name in models:
            ok = run_ref_model(model_name, args)
            results[model_name] = ok

        print(f"\n{'=' * 70}")
        print("  FINAL SUMMARY")
        print(f"{'=' * 70}")
        any_fail = False
        for model_name, ok in results.items():
            short = model_name.split("/")[-1]
            status = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True
            print(f"  {short:<45} {status}")
        print("=" * 70)

        sys.exit(1 if any_fail else 0)

    # Default: full alignment test
    results = {}
    for model_name in models:
        mismatches = run_model_test(model_name, args, project_root)
        results[model_name] = mismatches

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    any_fail = False
    for model_name, mismatches in results.items():
        short = model_name.split("/")[-1]
        status = "PASS" if mismatches == 0 else f"WARN ({mismatches}/{len(PROMPTS)} mismatches)"
        if mismatches == len(PROMPTS):
            status = f"FAIL ({mismatches}/{len(PROMPTS)} mismatches)"
            any_fail = True
        print(f"  {short:<45} {status}")
    print("=" * 70)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
