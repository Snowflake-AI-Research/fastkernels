"""OpenFold3 (AlphaFold3-style structure prediction) adapter for ``fastkernels e2e``.

Runs the fastkernels ``tasks.baseline.L4.openfold3.OpenFold3Model`` in-process -- the
production path of ``validate/bench_openfold3.py`` (eager PyTorch, bs=1, sequential
queries; the harness only uses ``torch.compile`` behind a non-default flag), so any
patched-in candidate kernels are exercised. Reused from the harness: the model config
(48 PairFormer / 4 MSA blocks, 10 diffusion rollout steps, 1 recycle, <=512 MSA rows),
the OpenProteinSet chain catalog, the featurizer (``_FEATURIZE_FN``), the global seeding
+ deterministic-algorithm settings, ``manual_seed(seed + qi)`` before every query, and the
per-query pass criteria (``TARGETS``).

Workloads (``StructurePrediction``):

* ``short`` / ``medium`` / ``long`` / ``extra-long`` (throughput): ``num_queries`` chains
  drawn from the workload's own length bucket of ``CHAIN_CATALOG`` (seeded permutation;
  ``max_requests`` keeps a prefix). After ``extra["warmup_passes"]`` (default 1) untimed
  forwards over every distinct input shape of the workload (so lazily compiled / autotuned
  kernels never compile inside the timer), timed like the harness -- sync, then per query H2D ->
  ``manual_seed(seed + qi)`` -> forward -> pull the (compact) outputs to CPU, sync; value =
  residues (tokens) / wall second (``tok/s``) -- except that featurization (MSA parsing +
  synthetic atom features on CPU) happens before the timer: in the harness loop it is
  I/O-bound on a cold volume (5 s vs 0.03 s warm for two short queries) and swamps
  the model time.
* ``single-*`` (latency): the bucket's first chain, featurized outside the timer;
  ``num_warmup`` untimed + ``num_iters`` timed forwards (``max_requests`` caps the
  iterations); value = median seconds.

Correctness (``outputs.pt``, ``kind="structures"``): the first ``correctness_samples``
throughput queries in workload order (then one per latency workload if room), keyed by
``workload/qi/chain``; per sample the predicted positions of the real (masked-in) atoms,
their pLDDT logits, the PAE logits strided down to <= 64 x 64 tokens (fp16) and the trunk
single representation (bf16). ``compare``'s per-query d uses the quantities that are stable
under benign numerics (trunk single rep, pLDDT, structure size); see ``_TOL`` for why the
harness's structure/PAE criteria are reported but not used.

Data (provision once on CPU: ``python -m fastkernels.e2e.adapters.openfold3 --provision``):
MSAs from the public OpenProteinSet bucket (``s3://openfold/pdb/<pdb>_<chain>/a3m``) under
``$FASTKERNELS_OF3_DATA_DIR`` (default ``~/.fastkernels/datasets/openfold3``), and the
public OpenFold3 checkpoint ``s3://openfold/openfold3_params/of3_ft3_v1.pt`` (the harness's
``OpenFold/OpenFold3:checkpoints/of3-p2-155k.pt`` lives in a gated HF repo; point
``FASTKERNELS_OF3_CHECKPOINT`` at it to use it instead). Missing data is fetched on demand.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

from .base import Adapter, RunSpec, Timing

HF_NAME = "OpenFold/OpenFold3"
S3_HTTP = "https://openfold.s3.amazonaws.com"
PUBLIC_CKPT_KEY = "openfold3_params/of3_ft3_v1.pt"
PLDDT_BINS = 50

# Tolerances of the per-query discrepancy terms (see ``OpenFold3Adapter.metric``), set from a
# B200 sensitivity probe: benign numerics (cuBLAS bf16 reduced-precision reduction off, 0.1%
# weight noise, an fp32 model) give 1-cos(s_trunk) 0.7e-5..1.4e-4, pLDDT MAE 0.1..0.6
# points and |ln Rg ratio| <= 0.26 -- but Kabsch RMSD 4..17 A and PAE-logit cosine 0.5..0.8:
# on the harness's synthetic features the trunk pair track and the diffusion trajectory are
# chaotic, so the harness criteria (reported in the summary) only pass bitwise-equal runs.
_TOL = {"s_trunk": 1e-3, "plddt_mae": 2.0, "log_rg": 1.0}


def _data_dir() -> Path:
    return Path(os.environ.get("FASTKERNELS_OF3_DATA_DIR",
                               str(Path.home() / ".fastkernels" / "datasets" / "openfold3")))


def _bench():
    from fastkernels.validate import bench_openfold3
    return bench_openfold3


# --------------------------------------------------------------------------- data
def _catalog(bucket: str) -> list[str]:
    seen: dict[str, None] = {}
    for pdb, chain, _, _ in _bench().CHAIN_CATALOG[bucket]:
        seen.setdefault(f"{pdb}_{chain}", None)
    return list(seen)


def _bucket_queries(bucket: str, n: int, seed: int) -> list[str]:
    """``n`` chain keys from ``bucket``: a seeded permutation of the bucket (wrapping if
    ``n`` exceeds it). Independent of other workloads, so any subset/cap is a prefix."""
    import numpy as np
    chains = _catalog(bucket)
    order = np.random.RandomState(seed).permutation(len(chains))
    return [chains[order[i % len(chains)]] for i in range(n)]


def _plan(scenario, spec: RunSpec) -> list[dict]:
    """The workloads to run, in scenario order, with their chains / iteration counts."""
    from fastkernels.workloads import (STRUCTURE_PREDICTION_LATENCY_WORKLOADS,
                                       STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS)
    tput = {w.name: w for w in STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS}
    lat = {w.name: w for w in STRUCTURE_PREDICTION_LATENCY_WORKLOADS}
    names = [getattr(w, "value", str(w)) for w in scenario.workloads]
    if spec.workloads:
        unknown = [w for w in spec.workloads if w not in names]
        if unknown:
            raise ValueError(f"unknown workloads {unknown}; scenario has {names}")
        names = [n for n in names if n in spec.workloads]
    cap = spec.max_requests
    plan = []
    for name in names:
        if name in tput:
            n = tput[name].num_queries if cap is None else min(cap, tput[name].num_queries)
            plan.append({"name": name, "kind": "throughput",
                         "chains": _bucket_queries(name, n, spec.seed)})
        elif name in lat:
            w = lat[name]
            iters = w.num_iters if cap is None else max(1, min(cap, w.num_iters))
            plan.append({"name": name, "kind": "latency", "num_warmup": w.num_warmup,
                         "num_iters": iters, "chains": _bucket_queries(w.length_bucket, 1, spec.seed)})
        else:
            raise ValueError(f"{name}: not a StructurePrediction workload")
    return plan


def _http_get(url: str, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.rename(dst)


def _ensure_chain(key: str) -> Path:
    """``<data>/alignments/<pdb>_<chain>/*.a3m`` from the public OpenProteinSet bucket."""
    import re
    d = _data_dir() / "alignments" / key
    if (d / ".complete").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    listing = urllib.request.urlopen(f"{S3_HTTP}/?list-type=2&prefix=pdb/{key}/a3m/",
                                     timeout=60).read().decode()
    keys = [k for k in re.findall(r"<Key>([^<]+)</Key>", listing) if k.endswith((".a3m", ".sto"))]
    if not keys:
        raise FileNotFoundError(f"OpenProteinSet has no alignments for {key}")
    for k in keys:
        _http_get(f"{S3_HTTP}/{k}", d / Path(k).name)
    (d / ".complete").write_text("\n".join(keys))
    return d


def _checkpoint() -> tuple[str, str]:
    """(path, label) of the OpenFold3 weights; downloads the public one if missing."""
    env = os.environ.get("FASTKERNELS_OF3_CHECKPOINT")
    if env:
        return env, Path(env).name
    dst = _data_dir() / Path(PUBLIC_CKPT_KEY).name
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[openfold3] downloading {S3_HTTP}/{PUBLIC_CKPT_KEY} -> {dst}", flush=True)
        _http_get(f"{S3_HTTP}/{PUBLIC_CKPT_KEY}", dst)
    return str(dst), dst.name


def _load_weights(model, path: str) -> None:
    """``load_openfold3_checkpoint``, plus the two layout changes of the released
    ``of3_ft3_v1`` checkpoint vs the harness's ``of3-p2-155k``, remapped exactly:

    * atom transformers carry one z LayerNorm (weight only) per block instead of one shared
      norm; the pair-bias input is the same for every block, so each block's norm weight
      is folded into that block's ``linear_z`` columns and the shared norm set to ones;
    * the Fourier time embedding ``w`` is a ``Linear(1, c)`` weight ``[c, 1]`` -> ``[c]``.
    """
    import re

    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # as load_openfold3_checkpoint
    sd = ckpt.get("state_dict") or (ckpt["model"] if isinstance(ckpt.get("model"), dict) else ckpt)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    pat = re.compile(r"(.*\.atom_transformer)\.blocks\.(\d+)\.attention_pair_bias\.layer_norm_z\.weight$")
    for k in [k for k in sd if pat.match(k)]:
        prefix, i = pat.match(k).groups()
        w = sd.pop(k)
        lin = f"{prefix}.blocks.{i}.attention_pair_bias.linear_z.weight"
        sd[lin] = sd[lin] * w[None, :].to(sd[lin].dtype)
        sd[f"{prefix}.layer_norm_z.weight"] = torch.ones_like(w)
    for k in [k for k in sd if k.endswith("fourier_emb.w") and sd[k].dim() == 2]:
        sd[k] = sd[k].reshape(-1)
    # Every model *parameter* must be loaded (strict, as in the harness), but candidate
    # modules may legitimately (a) drop dead submodules -- e.g. AuxiliaryHeads' unused
    # ``pairformer_embedding`` -> unexpected checkpoint keys; (b) alias a parameter under a
    # second name (``self._src_w = self.weight``) -> missing keys that share storage with a
    # loaded one; (c) register their own buffers (caches, tables) -> missing buffer keys.
    # These are tolerated; a dropped *live* module fails correctness instead.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    msd = model.state_dict(keep_vars=True)
    loaded = {msd[k].data_ptr() for k in msd if k in sd}
    bad = [k for k in missing
           if isinstance(msd[k], torch.nn.Parameter) and msd[k].data_ptr() not in loaded]
    if bad:
        raise RuntimeError(f"checkpoint lacks {len(bad)} model parameters, e.g. {bad[:8]}")
    for what, keys in (("unused checkpoint", unexpected), ("aliased/buffer model", missing)):
        if keys:
            roots = sorted({k.rsplit(".", 1)[-1] for k in keys})
            print(f"[openfold3] tolerated {len(keys)} {what} tensors ({roots[:6]})", flush=True)


def provision(seed: int = 42, all_chains: bool = False) -> None:
    """Fetch the checkpoint and the chains the default scenario uses (CPU only)."""
    from concurrent.futures import ThreadPoolExecutor

    from fastkernels.workloads import STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS
    print(f"[openfold3] data dir {_data_dir()}", flush=True)
    keys: dict[str, None] = {}
    for w in STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS:
        for k in (_catalog(w.name) if all_chains else _bucket_queries(w.name, w.num_queries, seed)):
            keys.setdefault(k, None)
    with ThreadPoolExecutor(16) as ex:
        futs = {k: ex.submit(_ensure_chain, k) for k in keys}
        ckpt = _checkpoint()
        for k, f in futs.items():
            f.result()
    print(f"[openfold3] {len(keys)} chains + checkpoint {ckpt[0]} ready", flush=True)


# --------------------------------------------------------------------------- run
def _chain_seed(key: str) -> int:
    # The harness seeds each chain's synthetic atom features from sha256(alignment dir),
    # which changes with the data location; hash the chain id instead (location-free).
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31)


def _compact(outputs: dict, aux: dict, batch: dict) -> dict:
    """CPU copies of what ``compare`` needs, restricted to the real (unmasked) atoms."""
    import torch
    mask = batch["atom_mask"][0] > 0.5
    pos = outputs["atom_positions_predicted"].reshape(-1, mask.shape[0], 3)[0][mask]
    plddt = outputs["plddt_logits"][0].reshape(mask.shape[0], PLDDT_BINS)[mask]
    pae = outputs["pae_logits"][0]
    stride = -(-pae.shape[0] // 64)  # <= 64 x 64 (the harness uses n // 64: up to 127 x 127)
    pae = pae[::stride, ::stride, :]
    return {"atom_positions": pos.float().cpu(), "plddt_logits": plddt.to(torch.float16).cpu(),
            "pae_logits": pae.to(torch.float16).cpu(),
            "s_trunk": aux["s_trunk"][0].to(torch.bfloat16).cpu()}  # bf16 = model dtype, lossless


class OpenFold3Adapter(Adapter):
    name = "openfold3"
    metric = ("per query: d = max_k e_k/(e_k+tol_k) over e = 1-cos(trunk single rep s) "
              "[tol 1e-3], mean |delta per-atom pLDDT| in points [tol 2], |ln(Rg_cand/Rg_ref)| of "
              "the predicted structure [tol 1]; d = 1 if missing/non-finite. Structure/PAE "
              "agreement is not in d: it is chaotic under benign numerics (see _TOL); the harness "
              "criteria are reported as harness_pass_rate")

    @classmethod
    def handles(cls, scenario) -> bool:
        return scenario.hf_name == HF_NAME

    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import copy
        import random

        import numpy as np
        import torch

        b = _bench()
        seed = spec.seed
        # Harness worker preamble: global seeds + deterministic algorithms.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

        ns: dict = {}
        exec(compile(b._FEATURIZE_FN, "bench_openfold3._FEATURIZE_FN", "exec"), ns)
        featurize, to_device = ns["load_and_featurize_chain"], ns["batch_to_device"]

        plan = _plan(scenario, spec)
        print(f"[openfold3] plan: " + "; ".join(
            f"{p['name']}({p['kind']}, {len(p['chains']) if p['kind'] == 'throughput' else p['num_iters']})"
            for p in plan), flush=True)
        dirs = {k: str(_ensure_chain(k)) for p in plan for k in p["chains"]}
        ckpt_path, ckpt_label = _checkpoint()

        # Late import: candidate classes are already patched into the fastkernels modules.
        from fastkernels.tasks.baseline.L4.openfold3 import OpenFold3Config, OpenFold3Model
        device, dtype = "cuda", getattr(torch, scenario.dtype)
        cfg = OpenFold3Config(pairformer_no_blocks=b.PF_BLOCKS, msa_no_blocks=b.MSA_BLOCKS,
                              no_rollout_steps=b.NO_ROLLOUT_STEPS, num_recycles=b.NUM_RECYCLES)
        t0 = time.time()
        model = OpenFold3Model(cfg)
        _load_weights(model, ckpt_path)
        model = model.to(device=device, dtype=dtype).eval()
        # enforce_eager: nothing to disable -- the production path is eager (no compile /
        # CUDA graphs).
        print(f"[openfold3] model ready ({sum(p.numel() for p in model.parameters()):,} params, "
              f"{ckpt_label}, {time.time() - t0:.0f}s)", flush=True)

        def feats(key: str, max_msa: int = b.MAX_MSA_SEQS):  # CPU batch, n_tokens
            batch = featurize({"alignment_dir": dirs[key]}, max_msa_seqs=max_msa, c_m=cfg.c_m,
                              c_token=cfg.c_token_embedder, seed=_chain_seed(key))
            if batch is None:
                raise RuntimeError(f"{key}: featurization produced no MSA")
            return batch, batch.pop("n_tokens")

        # Harness warmup: one forward on the first query with a 16-row MSA.
        wb, _ = feats(plan[0]["chains"][0], max_msa=16)
        with torch.no_grad():
            model(to_device(wb, device, dtype))
        torch.cuda.synchronize()
        del wb

        timings: dict[str, Timing] = {}
        samples: list[dict] = []
        want = spec.correctness_samples
        warmup_passes = int(spec.extra.get("warmup_passes", 1))

        for p in (p for p in plan if p["kind"] == "throughput"):
            name, total_tok = p["name"], 0
            # Featurize (MSA file parsing + synthetic atom features, CPU) before the timer: the
            # harness times it too, but on a cold network volume it is I/O-bound (seconds per
            # query) and would swamp the model time. H2D copies stay inside the timed loop.
            tf = time.perf_counter()
            queries = [(key, *feats(key)) for key in p["chains"]]
            t_feat = time.perf_counter() - tf
            # Untimed pass(es) over every distinct input shape, so lazily JIT-compiled /
            # autotuned kernels (Triton, candidates) never compile inside the timed loop.
            shapes = {(n, tuple(cb["msa"].shape)): cb for _, cb, n in queries}
            tw = time.perf_counter()
            for _ in range(warmup_passes):
                for cb in shapes.values():
                    with torch.no_grad():
                        model(to_device(cb, device, dtype))
            torch.cuda.synchronize()
            t_warm = time.perf_counter() - tw
            start = time.perf_counter()
            for qi, (key, cpu_batch, n_tok) in enumerate(queries):
                batch = to_device(cpu_batch, device, dtype)
                total_tok += n_tok
                torch.manual_seed(seed + qi)
                torch.cuda.manual_seed_all(seed + qi)
                with torch.no_grad():
                    outputs, aux = model(batch)
                out = _compact(outputs, aux, batch)  # D2H (syncs), as the harness extracts per query
                if len(samples) < want:
                    samples.append({"key": f"{name}/{qi}/{key}", "n_tokens": n_tok,
                                    "seed": seed + qi, **out})
                del outputs, aux, batch, out
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            del queries
            timings[name] = {"kind": "throughput", "value": total_tok / elapsed, "unit": "tok/s"}
            print(f"[openfold3] {name}: {len(p['chains'])} queries, {total_tok} tokens, "
                  f"{elapsed:.2f}s, {total_tok / elapsed:.1f} tok/s (untimed: {t_feat:.1f}s featurize, "
                  f"{t_warm:.1f}s warmup over {len(shapes)} shapes x {warmup_passes})", flush=True)

        for p in (p for p in plan if p["kind"] == "latency"):
            name, key = p["name"], p["chains"][0]
            batch, n_tok = feats(key)
            batch = to_device(batch, device, dtype)
            for _ in range(p["num_warmup"]):
                with torch.no_grad():
                    model(copy.deepcopy(batch))
                torch.cuda.synchronize()
            lats = []
            for it in range(p["num_iters"]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs, aux = model(copy.deepcopy(batch))
                torch.cuda.synchronize()
                lats.append(time.perf_counter() - t0)
                if it == 0 and len(samples) < want:
                    samples.append({"key": f"{name}/0/{key}", "n_tokens": n_tok, "seed": seed,
                                    **_compact(outputs, aux, batch)})
                del outputs, aux
            med = float(np.median(lats))
            timings[name] = {"kind": "latency", "value": med, "unit": "s"}
            print(f"[openfold3] {name}: {key} ({n_tok} tok), median {med:.4f}s over "
                  f"{len(lats)} iters {[round(x, 4) for x in lats]}", flush=True)
            del batch

        print(f"[openfold3] peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        torch.save({"kind": "structures", "checkpoint": ckpt_label, "seed": seed,
                    "samples": samples}, os.path.join(spec.out_dir, "outputs.pt"))
        return timings

    # ----------------------------------------------------------------------- compare
    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        import numpy as np
        b = _bench()

        def arr(x):
            return np.asarray(x.float().numpy() if hasattr(x, "float") else x, dtype=np.float64)

        def plddt(logits):  # expected per-atom pLDDT from the binned logits
            l = arr(logits)
            p = np.exp(l - l.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            return p @ ((np.arange(l.shape[-1]) + 0.5) / l.shape[-1])

        def cos(x, y):
            x, y = x.ravel(), y.ravel()
            den = np.linalg.norm(x) * np.linalg.norm(y)
            return float(x @ y / den) if den > 0 else (0.0 if den == 0 else float("nan"))

        def rg(x):
            return float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean()))

        by_key = {s["key"]: s for s in cand.get("samples", [])}
        per, rows = [], []
        np_err = np.seterr(all="ignore")
        for r in ref.get("samples", []):
            c = by_key.get(r["key"])
            if c is None and by_key:  # probe runs produce fewer samples: compare what ran
                continue
            row = None
            try:
                pc = arr(c["atom_positions"]) if c is not None else None
                if pc is not None and pc.shape == tuple(r["atom_positions"].shape) \
                        and np.isfinite(pc).all():
                    pr = arr(r["atom_positions"])
                    lr, lc = plddt(r["plddt_logits"]), plddt(c["plddt_logits"])
                    row = {"s_cos": cos(arr(c["s_trunk"]), arr(r["s_trunk"])),
                           "plddt_mae": float(np.abs(lc - lr).mean() * 100),
                           "log_rg": abs(float(np.log(rg(pc) / rg(pr)))),
                           "atom_cos": b.cosine_sim(pc, pr), "rmsd": b.kabsch_rmsd(pc, pr),
                           "plddt_pearson": b.pearson_corr(lc, lr),
                           "pae_cos": b.cosine_sim(arr(c["pae_logits"]), arr(r["pae_logits"]))}
                    if not all(np.isfinite(v) for v in row.values()):
                        row = None
            except Exception:  # noqa: BLE001 -- malformed candidate output = maximal discrepancy
                row = None
            if row is None:
                per.append(1.0)
                rows.append(None)
                continue
            err = {"s_trunk": max(1 - row["s_cos"], 0.0), "plddt_mae": row["plddt_mae"],
                   "log_rg": row["log_rg"]}
            per.append(float(max(e / (e + _TOL[k]) for k, e in err.items())))
            row["harness_pass"] = (row["atom_cos"] >= b.TARGETS["atom_pos_cosine_min"]
                                   and row["rmsd"] < b.TARGETS["atom_rmsd_kabsch_mean"]
                                   and row["plddt_pearson"] >= b.TARGETS["plddt_pearson_mean"]
                                   and row["pae_cos"] >= b.TARGETS["pae_cosine_mean"])
            rows.append(row)

        np.seterr(**np_err)
        ok = [r for r in rows if r is not None]
        mean = lambda k: float(np.mean([r[k] for r in ok])) if ok else None  # noqa: E731
        worst = lambda k, f: float(f(r[k] for r in ok)) if ok else None  # noqa: E731
        summary = {
            "n": len(per),
            "invalid": len(per) - len(ok),
            "mean_d": float(np.mean(per)) if per else None,
            "frac_d_below_0.5": float(np.mean([d < 0.5 for d in per])) if per else None,
            "mean_s_trunk_cosine": mean("s_cos"),
            "min_s_trunk_cosine": worst("s_cos", min),
            "mean_plddt_mae": mean("plddt_mae"),
            "max_abs_log_rg": worst("log_rg", max),
            # validate/bench_openfold3 per-query criteria (met only by bitwise-equal numerics)
            "harness_pass_rate": (sum(bool(r["harness_pass"]) for r in ok) / len(per)) if per else None,
            "mean_atom_cosine": mean("atom_cos"),
            "mean_rmsd": mean("rmsd"),
            "max_rmsd": worst("rmsd", max),
            "mean_plddt_pearson": mean("plddt_pearson"),
            "mean_pae_cosine": mean("pae_cos"),
        }
        if ref.get("checkpoint") != cand.get("checkpoint"):
            summary["checkpoint_mismatch"] = [ref.get("checkpoint"), cand.get("checkpoint")]
        return {"per_sample": per, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m fastkernels.e2e.adapters.openfold3")
    ap.add_argument("--provision", action="store_true", help="download checkpoint + MSAs (CPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-chains", action="store_true", help="every catalog chain, not just the seed's")
    args = ap.parse_args(argv)
    if args.provision:
        provision(args.seed, args.all_chains)
    return 0


if __name__ == "__main__":
    sys.exit(main())
