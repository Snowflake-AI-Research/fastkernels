"""Deterministically (re)build the two real-prompt LLM E2E datasets.

Two frozen artifacts, both written to ``~/.fastkernels/datasets/<name>/`` as
``data.jsonl`` + ``data.parquet`` + ``meta.json``:

* ``wildchat-mixed-1k``  -- Scenario A (bulk throughput). 1000 English, single-turn
  WildChat-1M requests (deduplicated), natural short/long prompt+decode mix.
* ``longbench-longctx``  -- Scenario B (long context). 64 LongBench-v2 documents
  truncated into 8K..128K prefill buckets, each with its real MC question.

Determinism: everything is pinned -- the source dataset *revisions*, a fixed
seed, deterministic filtering/selection, and a fixed reference tokenizer. Re-running
this script reproduces byte-identical artifacts. Prompts are stored as raw text
and are chat-templated + tokenized per *target* model at load time (see
``fastkernels.workloads.load_real_prompt_workload``); the reference token counts
recorded here are advisory (Llama-3.1-8B).

Usage:
    python -m fastkernels.build_datasets --which all
    python -m fastkernels.build_datasets --which mixed --push
    python -m fastkernels.build_datasets --which longctx --out /some/dir

The Hub repo IDs and the mixed decode cap come from
``fastkernels.workloads`` so this stays in sync with the loader.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import numpy as np

from .workloads import DEFAULT_DECODE_CAPS, DEFAULT_WORKLOAD_DATASETS

# --- frozen, canonical build parameters (do not change without a re-push) ----
SEED = 42
REF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Source dataset revisions, pinned so upstream edits can never silently change
# the artifacts. Override with --wildchat-revision / --longbench-revision.
WILDCHAT_REPO = "allenai/WildChat-1M"
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
LONGBENCH_REPO = "THUDM/LongBench-v2"
LONGBENCH_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"

# Scenario A (mixed)
MIXED_N = 1000
MIXED_POOL = 40000
MIXED_DECODE_CAP = DEFAULT_DECODE_CAPS["mixed"]  # 1024

# Scenario B (long context): (target prefill length, count), largest first.
LONGCTX_BUCKETS = [(131072, 4), (65536, 4), (32768, 8), (16384, 16), (8192, 32)]
LONGCTX_DECODE = 256
LONGCTX_INSTR = (
    "You are given a long document followed by a multiple-choice question. "
    "Read the document carefully and answer.\n\n"
)


def _datasets_root() -> str:
    return os.path.expanduser("~/.fastkernels/datasets")


def _write_artifact(out_dir: str, rows: list[dict], meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(out_dir, "data.parquet"))
    assert os.path.getsize(os.path.join(out_dir, "data.parquet")) > 0, "empty parquet"
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _stats(name: str, a) -> None:
    a = np.asarray(a)
    q = [0, 25, 50, 75, 90, 99, 100]
    v = np.percentile(a, q).astype(int)
    print(f"    {name}: mean={int(a.mean()):6d} sum={int(a.sum()):>10,}  "
          + " ".join(f"p{x}={y}" for x, y in zip(q, v)))


# ---------------------------------------------------------------------------
# Scenario A -- wildchat-mixed-1k
# ---------------------------------------------------------------------------
def build_mixed(out_dir: str, revision: str) -> str:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[mixed] streaming {WILDCHAT_REPO}@{revision[:12]} "
          f"(English, 1-turn, dedup) ...", flush=True)
    ds = load_dataset(WILDCHAT_REPO, split="train", streaming=True, revision=revision)
    seen: set[str] = set()
    pool: list[dict] = []
    for row in ds:
        if row.get("turn") != 1 or row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        u = next((t["content"] for t in conv if t.get("role") == "user"), None)
        a = next((t["content"] for t in conv if t.get("role") == "assistant"), None)
        if not u or not a:
            continue
        key = u.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        pool.append({"user": u, "assistant": a})
        if len(pool) >= MIXED_POOL:
            break
    if len(pool) < MIXED_N:
        raise RuntimeError(f"pool={len(pool)} < N={MIXED_N}; source may have changed")
    print(f"      pool={len(pool)}; sampling {MIXED_N} (seed={SEED})", flush=True)

    sample = random.Random(SEED).sample(pool, MIXED_N)

    tok = AutoTokenizer.from_pretrained(REF_MODEL)
    rows = []
    for r in sample:
        p = tok.apply_chat_template(
            [{"role": "user", "content": r["user"]}], tokenize=True, add_generation_prompt=True)
        d = tok.encode(r["assistant"], add_special_tokens=False)
        rows.append({
            "user": r["user"],
            "assistant": r["assistant"],
            "ref_prompt_tokens": len(p),
            "ref_response_tokens": len(d),
        })

    meta = {
        "source": WILDCHAT_REPO,
        "source_revision": revision,
        "filter": "language=English, turn=1, non-empty, dedup(user)",
        "seed": SEED,
        "n": len(rows),
        "pool": len(pool),
        "decode_cap": MIXED_DECODE_CAP,
        "ref_model": REF_MODEL,
        "format": "rows of {user, assistant}; tokenize + decode-cap per target model at load",
    }
    _write_artifact(out_dir, rows, meta)

    pre = [r["ref_prompt_tokens"] for r in rows]
    dec = [max(1, min(r["ref_response_tokens"], MIXED_DECODE_CAP)) for r in rows]
    print(f"[mixed] wrote {len(rows)} rows -> {out_dir}")
    _stats("prefill      ", pre)
    _stats(f"decode(cap{MIXED_DECODE_CAP})", dec)
    print(f"    unique prompts={len(set(r['user'] for r in rows))}/{len(rows)}  "
          f"total tokens={int(np.sum(pre) + np.sum(dec)):,}")
    return out_dir


# ---------------------------------------------------------------------------
# Scenario B -- longbench-longctx
# ---------------------------------------------------------------------------
def _question_block(row: dict) -> str:
    qb = "\n\nQuestion: " + (row.get("question") or "").strip() + "\n"
    for letter in ("A", "B", "C", "D"):
        choice = row.get(f"choice_{letter}")
        if choice:
            qb += f"({letter}) {choice}\n"
    return qb + "\nThe correct answer is:"


def build_longctx(out_dir: str, revision: str) -> str:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REF_MODEL)
    print(f"[longctx] loading {LONGBENCH_REPO}@{revision[:12]} + measuring contexts ...",
          flush=True)
    ds = load_dataset(LONGBENCH_REPO, split="train", revision=revision)
    docs = []
    for row in ds:
        ctx = row.get("context") or ""
        if ctx:
            docs.append({"row": row, "clen": len(tok.encode(ctx, add_special_tokens=False))})
    print(f"      {len(docs)} docs; longest={max(d['clen'] for d in docs):,} tokens", flush=True)

    used: set[int] = set()

    def pick(count: int, budget: int):
        elig = sorted((i for i, d in enumerate(docs) if i not in used and d["clen"] >= budget),
                      key=lambda i: docs[i]["clen"], reverse=True)
        out = [("single", [i]) for i in elig[:count]]
        for _, idxs in out:
            used.add(idxs[0])
        while len(out) < count:  # fallback: concat real docs to reach budget
            remaining = sorted((i for i in range(len(docs)) if i not in used),
                               key=lambda i: docs[i]["clen"], reverse=True)
            if not remaining:
                break
            combo, tot = [], 0
            for i in remaining:
                combo.append(i)
                used.add(i)
                tot += docs[i]["clen"]
                if tot >= budget:
                    break
            out.append(("concat", combo))
        return out

    rows = []
    for L, count in LONGCTX_BUCKETS:
        picks = pick(count, budget=L - 512)
        for kind, idxs in picks:
            head = docs[idxs[0]]["row"]
            qb = _question_block(head)
            overhead = (len(tok.encode(LONGCTX_INSTR, add_special_tokens=False))
                        + len(tok.encode(qb, add_special_tokens=False)) + 8)
            ctx_budget = max(1, L - overhead)
            if kind == "single":
                ctx_ids = tok.encode(head["context"], add_special_tokens=False)[:ctx_budget]
            else:
                sep = tok.encode("\n\n[Next Document]\n\n", add_special_tokens=False)
                merged = []
                for j, i in enumerate(idxs):
                    if j:
                        merged += sep
                    merged += tok.encode(docs[i]["row"]["context"], add_special_tokens=False)
                ctx_ids = merged[:ctx_budget]
            user_text = LONGCTX_INSTR + tok.decode(ctx_ids) + qb
            templated = tok.apply_chat_template(
                [{"role": "user", "content": user_text}], tokenize=True, add_generation_prompt=True)
            rows.append({
                "user": user_text,
                "assistant": "",
                "ref_prompt_tokens": len(templated),
                "output_len": LONGCTX_DECODE,
                "bucket": L,
                "source": kind,
            })

    random.Random(SEED).shuffle(rows)

    meta = {
        "source": LONGBENCH_REPO,
        "source_revision": revision,
        "filter": "non-empty context; real multiple-choice question appended; "
                  "single-doc prefix per bucket, multi-doc concat only as fallback",
        "seed": SEED,
        "buckets": {str(L): c for L, c in LONGCTX_BUCKETS},
        "decode_len": LONGCTX_DECODE,
        "n": len(rows),
        "ref_model": REF_MODEL,
        "format": "rows of {user, assistant}; tokenize + truncate-to-max_model_len per target model at load",
    }
    _write_artifact(out_dir, rows, meta)

    pre = np.array([r["ref_prompt_tokens"] for r in rows])
    print(f"[longctx] wrote {len(rows)} rows -> {out_dir}  sources={dict(Counter(r['source'] for r in rows))}")
    for L, _ in LONGCTX_BUCKETS:
        b = pre[[r["bucket"] == L for r in rows]]
        print(f"    ~{L // 1024:>3}K: {len(b):>2} reqs  actual mean={int(b.mean()):>7,}")
    print(f"    total prefill={int(pre.sum()):,}  total decode={len(rows) * LONGCTX_DECODE:,}")
    return out_dir


# ---------------------------------------------------------------------------
def _push(out_dir: str, repo_id: str, private: bool) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    ds = load_dataset("json", data_files=os.path.join(out_dir, "data.jsonl"), split="train")
    ds.push_to_hub(repo_id, private=private)
    HfApi().upload_file(
        path_or_fileobj=os.path.join(out_dir, "meta.json"),
        path_in_repo="meta.json", repo_id=repo_id, repo_type="dataset",
    )
    vis = "private" if private else "public"
    print(f"[push] {out_dir} -> https://huggingface.co/datasets/{repo_id} ({vis})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["all", "mixed", "longctx"], default="all")
    ap.add_argument("--out", default=None, help="datasets root (default: ~/.fastkernels/datasets)")
    ap.add_argument("--push", action="store_true", help="push rebuilt artifact(s) to the Hub")
    ap.add_argument("--public", action="store_true", help="push as public (default: private)")
    ap.add_argument("--wildchat-revision", default=WILDCHAT_REVISION)
    ap.add_argument("--longbench-revision", default=LONGBENCH_REVISION)
    args = ap.parse_args()

    random.seed(SEED)
    root = args.out or _datasets_root()
    mixed_repo = DEFAULT_WORKLOAD_DATASETS["mixed"]
    longctx_repo = DEFAULT_WORKLOAD_DATASETS["long-context"]

    if args.which in ("all", "mixed"):
        out = build_mixed(os.path.join(root, mixed_repo.split("/")[-1]), args.wildchat_revision)
        if args.push:
            _push(out, mixed_repo, private=not args.public)
    if args.which in ("all", "longctx"):
        out = build_longctx(os.path.join(root, longctx_repo.split("/")[-1]), args.longbench_revision)
        if args.push:
            _push(out, longctx_repo, private=not args.public)
    print("[done]")


if __name__ == "__main__":
    main()
