"""Object detection (YOLOv10, RT-DETRv2) adapter for ``fastkernels e2e``.

In-process port of the fastkernels side of ``validate/bench_detection.py``:

* model: ``infra.detection_loader.load_ours_detector`` (the production path; the L4 module
  is imported lazily, after the runner has patched candidate classes in), run through
  ``run_ours_detector`` (= ``model.predict``: forward + score threshold + padding to
  ``max_detections``, per-image host syncs included, exactly as the harness times it);
* images: COCO val2017 (``detection-datasets/coco``, split ``val``), shuffled with
  ``random.Random(seed)``, Resize + CenterCrop to 640, stored fp16 on the host -- the
  harness's selection and preprocessing (decoded with ``datasets.Image``). The preprocessed
  set is built once and cached (raw uint8) under ``~/.fastkernels/e2e_data/coco``
  (override: ``FASTKERNELS_E2E_DATA_DIR``); ``python -m fastkernels.e2e.adapters.detection
  --prepare`` builds it ahead of time (CPU only);
* timing (as the harness): one bs=1 warmup forward; throughput workloads tile
  ``num_images`` over the unique images in batches (host gather + H2D copy inside the
  timed loop), 3 measured passes, images/s from the median pass. Warmup before them: one
  batch of every distinct shape plus ``spec.extra["warmup_passes"]`` untimed full passes
  (default 1; the harness uses 3), so lazy JIT/autotuning never lands in a timed pass; latency
  workloads time ``num_iters`` forwards of the first ``batch_size`` images after
  ``num_warmup``, median seconds. ``spec.max_requests`` caps the throughput image count
  (and so the unique images loaded); latency probes keep their iteration counts.

Correctness: a separate untimed pass over the first ``spec.correctness_samples`` images (in
throughput-sized batches) saves the padded ``predict`` outputs (boxes xyxy px, scores,
labels; label -1 = empty slot). ``compare`` matches detections per image (see ``metric``),
which is invariant to the rank order of near-tied detections.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import ClassVar

from .base import Adapter, RunSpec, Timing

_COCO_REPO = "detection-datasets/coco"
_COCO_SPLIT_PREFIX = "data/val-"
_MAX_DETECTIONS = 100
_MATCH_IOU = 0.5
_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


def _is_yolov10(name: str) -> bool:  # mirrors infra.detection_loader (no torch import)
    return "yolov10" in name.lower()


def _is_rtdetrv2(name: str) -> bool:
    n = name.lower()
    return "rtdetr_v2" in n or "rt-detr_v2" in n or "rtdetrv2" in n


# ---------------------------------------------------------------------------
# COCO val2017 images (harness selection + preprocessing, cached as uint8)
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    return Path(os.environ.get("FASTKERNELS_E2E_DATA_DIR",
                               str(Path.home() / ".fastkernels" / "e2e_data"))) / "coco"


def _cache_path(image_size: int, seed: int) -> Path:
    return _data_dir() / f"coco_val2017_{image_size}_seed{seed}.u8"


def prepare_coco(image_size: int = 640, seed: int = 42, workers: int = 8) -> Path:
    """Build (once) all COCO val2017 images, in the harness's seeded-shuffle order,
    preprocessed to uint8 (3, S, S) each, as raw bytes in ``<path>``; ``<path>.json`` (written
    last) holds the image count. ``x.float() / 255`` equals the harness's ``ToTensor()``.
    Plain sequential file I/O (no mmap), so the cache can live on a network volume."""
    import json
    import random
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    path = _cache_path(image_size, seed)
    meta = path.with_suffix(".json")
    if path.is_file() and meta.is_file():
        return path

    import pyarrow.parquet as pq
    from datasets import Image as HFImage
    from huggingface_hub import HfApi, hf_hub_download
    from torchvision import transforms

    t0 = time.time()
    files = sorted(f for f in HfApi().list_repo_files(_COCO_REPO, repo_type="dataset")
                   if f.startswith(_COCO_SPLIT_PREFIX) and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"no {_COCO_SPLIT_PREFIX}*.parquet files in {_COCO_REPO}")
    blobs: list = []
    for f in files:  # same order as streaming load_dataset(..., split="val")
        col = pq.read_table(hf_hub_download(_COCO_REPO, f, repo_type="dataset"),
                            columns=["image"]).column("image")
        for chunk in col.chunks:
            blobs += [{"bytes": b, "path": p} for b, p in
                      zip(chunk.field("bytes").to_pylist(), chunk.field("path").to_pylist())]
    print(f"  [detection] COCO val: {len(blobs)} images from {len(files)} parquet files "
          f"({time.time() - t0:.0f}s)", flush=True)

    # bench_detection: items = list(ds); random.Random(seed).shuffle(items). shuffle()
    # only depends on the length, so shuffling indices gives the same permutation.
    order = list(range(len(blobs)))
    random.Random(seed).shuffle(order)
    feature = HFImage()
    tf = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])

    def prep(i: int):
        try:
            img = feature.decode_example(blobs[i]).convert("RGB")
            return np.ascontiguousarray(np.asarray(tf(img), dtype=np.uint8).transpose(2, 0, 1))
        except Exception:  # noqa: BLE001 -- the harness skips undecodable items too
            return None

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    n = 0
    with open(tmp, "wb") as fh, ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(order), 256):
            for arr in pool.map(prep, order[start:start + 256]):
                if arr is not None:
                    fh.write(arr.tobytes())
                    n += 1
    os.replace(tmp, path)  # atomic publish; parallel builders just duplicate work
    meta.write_text(json.dumps({"count": n, "image_size": image_size, "seed": seed,
                                "dtype": "uint8", "shape": [3, image_size, image_size],
                                "source": [f"{_COCO_REPO}/{f}" for f in files]}))
    print(f"  [detection] cached {n} preprocessed images -> {path} ({time.time() - t0:.0f}s)",
          flush=True)
    return path


def load_coco(n: int, image_size: int, seed: int, dtype):
    """First ``n`` images of the seeded order as a host tensor (N, 3, S, S), rounded through
    fp16 like the harness's tensor cache."""
    import json

    import numpy as np
    import torch

    path = prepare_coco(image_size, seed)
    n = min(n, int(json.loads(path.with_suffix(".json").read_text())["count"]))
    per = 3 * image_size * image_size
    out = torch.empty((n, 3, image_size, image_size), dtype=dtype)
    with open(path, "rb") as fh:
        for s in range(0, n, 256):
            k = min(256, n - s)
            buf = np.frombuffer(fh.read(k * per), dtype=np.uint8).reshape(k, 3, image_size, image_size)
            out[s:s + k] = torch.from_numpy(buf.copy()).float().div_(255).half().to(dtype)
    return out


# ---------------------------------------------------------------------------
# Correctness helpers (CPU, numpy)
# ---------------------------------------------------------------------------

def _np(x, dtype):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        x = x.float() if dtype is float else x.long()
        x = x.numpy()
    return np.asarray(x, dtype=np.float64 if dtype is float else np.int64)


def _cos(a, b) -> float:
    import numpy as np
    a, b = np.nan_to_num(a.reshape(-1)), np.nan_to_num(b.reshape(-1))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(a @ b / max(na * nb, 1e-12))


def _iou(a, b):
    import numpy as np
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return np.nan_to_num(iou, nan=0.0)


def _match_image(rb, rs, rl, cb, cs, cl) -> dict:
    """Greedy (reference score order) same-label IoU>=0.5 matching of one image's valid
    detections. Returns the score-weighted F1, matched IoUs and |score| diffs."""
    import numpy as np
    rw, cw = np.clip(np.nan_to_num(rs), 0, 1), np.clip(np.nan_to_num(cs), 0, 1)
    if len(rs) == 0 and len(cs) == 0:
        return {"f1": 1.0, "ious": [], "dscore": [], "matched": 0}
    ious, dscore, mw = [], [], 0.0
    if len(rs) and len(cs):
        iou = _iou(np.nan_to_num(rb), np.nan_to_num(cb))
        iou[rl[:, None] != cl[None, :]] = 0.0
        used = np.zeros(len(cs), dtype=bool)
        for r in np.argsort(-rw, kind="stable"):
            cand = np.where(used, -1.0, iou[r])
            c = int(np.argmax(cand))
            if cand[c] >= _MATCH_IOU:
                used[c] = True
                ious.append(float(cand[c]))
                dscore.append(abs(float(rw[r] - cw[c])))
                mw += rw[r] + cw[c]
    total = rw.sum() + cw.sum()
    if total > 0:
        f1 = float(mw / total)
    else:  # degenerate zero scores: count-based F1
        f1 = 2.0 * len(ious) / (len(rs) + len(cs))
    return {"f1": f1, "ious": ious, "dscore": dscore, "matched": len(ious)}


class DetectionAdapter(Adapter):
    name: ClassVar[str] = "detection"
    metric: ClassVar[str] = (
        "per image: valid detections (label >= 0) of reference and candidate are matched "
        "greedily in reference-score order (same label, IoU >= 0.5); "
        "d = max(1 - F1_w, 1 - mean matched IoU, mean matched |score diff|), where F1_w is "
        "the score-weighted F1 = sum over matched pairs (s_ref + s_cand) / (sum of all "
        "reference and candidate scores), so a low-confidence detection crossing the "
        "threshold costs little and a missed/spurious confident one costs a lot; "
        "d = 0 when neither side detects anything, 1 when nothing matches")

    @classmethod
    def handles(cls, scenario) -> bool:
        from fastkernels.workloads import Detection
        return (any(isinstance(w, Detection) for w in scenario.workloads)
                and (_is_yolov10(scenario.hf_name) or _is_rtdetrv2(scenario.hf_name)))

    # ------------------------------------------------------------------ run

    def run(self, scenario, spec: RunSpec) -> dict[str, Timing]:
        import torch

        from fastkernels.infra.detection_loader import (
            infer_image_size, load_ours_detector, run_ours_detector,
        )
        from fastkernels.workloads import Purpose, spec_for

        model_name = scenario.hf_name
        dtype = getattr(torch, _DTYPES.get(scenario.dtype, "float16"))
        device = "cuda"
        torch.manual_seed(spec.seed)

        wanted = set(spec.workloads or [])
        workloads = [spec_for(w) for w in scenario.workloads
                     if not wanted or wanted & {w.value, w.name, f"{type(w).__name__}.{w.name}"}]
        if not workloads:
            raise ValueError(f"no workloads of {model_name} match {spec.workloads}")
        image_size = next((int(ws.params.image_size) for ws in workloads
                           if getattr(ws.params, "image_size", None)), infer_image_size(model_name))

        def n_thr(ws) -> int:
            n = int(ws.params.num_images)
            return min(n, int(spec.max_requests)) if spec.max_requests else n

        thr = [ws for ws in workloads if ws.purpose is Purpose.THROUGHPUT]
        lat = [ws for ws in workloads if ws.purpose is Purpose.LATENCY]
        corr_bs = max([int(ws.params.batch_size) for ws in thr] or [32])
        n_unique = max([n_thr(ws) for ws in thr] + [int(ws.params.batch_size) for ws in lat]
                       + [spec.correctness_samples, 1])

        t0 = time.time()
        images = load_coco(n_unique, image_size, spec.seed, dtype)
        available = images.shape[0]
        print(f"  [detection] {available} COCO images {tuple(images.shape)} {dtype} "
              f"({time.time() - t0:.0f}s)", flush=True)

        model = load_ours_detector(model_name, device=device, dtype=dtype)
        if spec.enforce_eager and hasattr(model, "_use_hopper_cudagraph"):
            model._use_hopper_cudagraph = False  # the only graph path (Hopper-only)

        def predict(batch):
            return run_ours_detector(model, model_name, batch, image_size,
                                     max_detections=_MAX_DETECTIONS)

        warmup_passes = int(spec.extra.get("warmup_passes", 1))
        timings: dict[str, Timing] = {}
        with torch.no_grad():
            predict(images[:1].to(device=device))
            torch.cuda.synchronize()

            for ws in thr:
                num, bs = n_thr(ws), int(ws.params.batch_size)
                cycle = torch.arange(num) % available

                def one_pass():
                    for s in range(0, num, bs):
                        predict(images[cycle[s:s + bs]].to(device=device))

                # Untimed warmup so lazy JIT / autotuning never lands in a timed pass: one
                # batch of every distinct shape (full + remainder), then ``warmup_passes``
                # full passes (default 1; the orchestrator sets 0 for cheap probes).
                for s in sorted({0, (num - 1) // bs * bs}):
                    predict(images[cycle[s:s + bs]].to(device=device))
                for _ in range(warmup_passes):
                    one_pass()
                torch.cuda.synchronize()
                runs = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    one_pass()
                    torch.cuda.synchronize()
                    runs.append(time.perf_counter() - t)
                med = sorted(runs)[len(runs) // 2]
                timings[ws.name] = {"kind": "throughput", "value": num / med, "unit": "images/s",
                                    "num_images": num, "batch_size": bs,
                                    "elapsed_runs": [round(r, 5) for r in runs]}
                print(f"  [detection] {ws.name}: {num / med:.1f} img/s ({num} images, bs={bs})",
                      flush=True)

            for ws in lat:
                bs = int(ws.params.batch_size)
                warm, iters = int(getattr(ws.params, "num_warmup", 3)), int(getattr(ws.params, "num_iters", 20))
                batch = images[:bs].to(device=device)
                for _ in range(warm):
                    predict(batch)
                torch.cuda.synchronize()
                lats = []
                for _ in range(iters):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    predict(batch)
                    torch.cuda.synchronize()
                    lats.append(time.perf_counter() - t)
                med = sorted(lats)[len(lats) // 2]
                timings[ws.name] = {"kind": "latency", "value": med, "unit": "s",
                                    "batch_size": bs, "num_iters": iters}
                print(f"  [detection] {ws.name}: median {med * 1e3:.3f} ms (bs={bs})", flush=True)

            # Correctness: first N images, throughput-sized batches, untimed.
            n_corr = min(spec.correctness_samples, available)
            boxes, scores, labels = [], [], []
            for s in range(0, n_corr, corr_bs):
                det = predict(images[s:min(s + corr_bs, n_corr)].to(device=device))
                boxes.append(det["boxes"].float().cpu())
                scores.append(det["scores"].float().cpu())
                labels.append(det["labels"].to(torch.int16).cpu())
        torch.save({
            "kind": "detections", "model": model_name, "dtype": scenario.dtype,
            "image_size": image_size, "max_detections": _MAX_DETECTIONS,
            "conf_threshold": getattr(model, "conf_threshold", None),
            "dataset": f"{_COCO_REPO}:val seeded-shuffle(seed={spec.seed})[:{n_corr}]",
            "batch_size": corr_bs,
            "boxes": torch.cat(boxes) if boxes else torch.zeros(0, _MAX_DETECTIONS, 4),
            "scores": torch.cat(scores) if scores else torch.zeros(0, _MAX_DETECTIONS),
            "labels": torch.cat(labels) if labels else torch.zeros(0, _MAX_DETECTIONS, dtype=torch.int16),
        }, os.path.join(spec.out_dir, "outputs.pt"))
        return timings

    # -------------------------------------------------------------- compare

    @classmethod
    def compare(cls, ref: dict, cand: dict) -> dict:
        import numpy as np

        rb, rs, rl = _np(ref["boxes"], float), _np(ref["scores"], float), _np(ref["labels"], int)
        cb, cs, cl = _np(cand["boxes"], float), _np(cand["scores"], float), _np(cand["labels"], int)
        n_ref, n = rb.shape[0], min(rb.shape[0], cb.shape[0])
        per_sample, f1s, ious, dscores = [], [], [], []
        n_rdet = n_cdet = 0
        for i in range(n):
            rv, cv = rl[i] >= 0, cl[i] >= 0
            n_rdet += int(rv.sum())
            n_cdet += int(cv.sum())
            m = _match_image(rb[i][rv], rs[i][rv], rl[i][rv], cb[i][cv], cs[i][cv], cl[i][cv])
            miou = float(np.mean(m["ious"])) if m["ious"] else 1.0
            mds = float(np.mean(m["dscore"])) if m["dscore"] else 0.0
            d = max(1.0 - m["f1"], 1.0 - miou, mds)
            per_sample.append(float(min(max(d, 0.0), 1.0)))
            f1s.append(m["f1"])
            ious += m["ious"]
            dscores += m["dscore"]
        per_sample += [1.0] * (n_ref - n)  # candidate produced fewer images

        # Slot-wise aggregates as reported by validate/bench_detection.
        k = min(rb.shape[1], cb.shape[1]) if n else 0
        slot = {}
        if n and k:
            slot = {
                "labels_match_rate": float(np.mean(rl[:n, :k] == cl[:n, :k])),
                "boxes_cosine": _cos(rb[:n, :k], cb[:n, :k]),
                "scores_cosine": _cos(rs[:n, :k], cs[:n, :k]),
                "boxes_mae": float(np.mean(np.abs(np.nan_to_num(rb[:n, :k] - cb[:n, :k])))),
                "scores_mae": float(np.mean(np.abs(np.nan_to_num(rs[:n, :k] - cs[:n, :k])))),
            }
        summary = {
            "num_images": n_ref,
            "num_compared": n,
            **slot,
            "det_f1_weighted": float(np.mean(f1s)) if f1s else None,
            "matched_iou_mean": float(np.mean(ious)) if ious else None,
            "matched_score_absdiff_mean": float(np.mean(dscores)) if dscores else None,
            "ref_detections": n_rdet,
            "cand_detections": n_cdet,
            "matched_detections": len(ious),
            "d_mean": float(np.mean(per_sample)) if per_sample else None,
            "d_max": float(np.max(per_sample)) if per_sample else None,
        }
        return {"per_sample": per_sample, "summary": summary}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the cached COCO val2017 image set (CPU).")
    ap.add_argument("--prepare", action="store_true", required=True)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    print(prepare_coco(a.image_size, a.seed, a.workers))
