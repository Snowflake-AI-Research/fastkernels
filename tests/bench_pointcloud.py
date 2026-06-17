"""Benchmark PointTransformerV3 on ScanObjectNN."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_KB_ROOT = Path(__file__).resolve().parents[1]
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))

import torch

from infra.pointcloud_loader import DEFAULT_PTV3_CHECKPOINT_FILE
from bench.utils.worker import run_worker

POINTCLOUD_WORKER = r'''
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

from infra.pointcloud_loader import (
    default_ptv3_kwargs,
    load_point_backbone_checkpoint,
    load_ours_point_model,
    load_reference_point_model,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resample_points(pos: torch.Tensor, num_points: int) -> torch.Tensor:
    if pos.shape[0] == num_points:
        return pos
    if pos.shape[0] > num_points:
        index = torch.linspace(0, pos.shape[0] - 1, steps=num_points).round().long()
        return pos[index]
    repeat = (num_points + pos.shape[0] - 1) // pos.shape[0]
    return pos.repeat((repeat, 1))[:num_points]


def _voxel_unique(pos: torch.Tensor, grid_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.div(pos - pos.min(dim=0).values, grid_size, rounding_mode="trunc").int()
    grid_unique, inverse, counts = torch.unique(
        grid,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    coord = torch.zeros((grid_unique.shape[0], pos.shape[1]), dtype=pos.dtype)
    coord.index_add_(0, inverse, pos)
    coord = coord / counts.unsqueeze(1)
    return coord, grid_unique


def load_scanobjectnn(dataset_root: str, split: str, points_per_sample: int, max_samples: int, grid_size: float):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "ScanObjectNN benchmark requires the Hugging Face datasets package."
        ) from exc

    dataset = load_dataset(
        "jxie/scanobjectnn",
        split=f"{split}[:{max_samples}]",
        cache_dir=dataset_root,
    )
    samples = []
    for idx, row in enumerate(dataset):
        pos = torch.tensor(row["inputs"], dtype=torch.float32)
        pos = _resample_points(pos, points_per_sample)
        pos = pos - pos.mean(dim=0, keepdim=True)
        scale = pos.norm(dim=1).amax().clamp(min=1e-6)
        pos = pos / scale
        pos, grid_coord = _voxel_unique(pos, grid_size=grid_size)
        samples.append(
            {
                "pos": pos,
                "normal": torch.zeros_like(pos),
                "grid_coord": grid_coord,
                "sample_id": f"{split}:{idx}",
            }
        )
    return samples


def _to_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _sample_get(sample, key, default=None):
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _sample_point_count(sample) -> int:
    pos = _sample_get(sample, "pos")
    return int(pos.shape[0])


def _normalize_feat_dim(feat: torch.Tensor, target_dim: int = 6) -> torch.Tensor:
    feat = feat.float()
    if feat.shape[1] < target_dim:
        pad = torch.zeros(feat.shape[0], target_dim - feat.shape[1], dtype=feat.dtype)
        feat = torch.cat([feat, pad], dim=1)
    elif feat.shape[1] > target_dim:
        feat = feat[:, :target_dim]
    return feat


def build_batches(
    samples,
    dataset_name: str,
    batch_size: int,
    grid_size: float,
    feat_dim: int,
    device: str,
    dtype: torch.dtype,
    drop_last: bool = True,
):
    batches = []
    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        if drop_last and len(batch_samples) < batch_size:
            break

        coords = []
        feats = []
        grid_coords = []
        counts = []
        sample_ids = []
        for local_idx, sample in enumerate(batch_samples):
            coord = _to_tensor(_sample_get(sample, "pos")).float()
            normal = _to_tensor(_sample_get(sample, "normal"))
            grid_coord = _to_tensor(_sample_get(sample, "grid_coord"))
            if normal is None:
                normal = torch.zeros_like(coord)
            feat = torch.cat([coord, normal.float()], dim=1)
            feat = _normalize_feat_dim(feat, feat_dim)
            coords.append(coord)
            feats.append(feat)
            if grid_coord is not None:
                grid_coords.append(grid_coord.long())
            counts.append(coord.shape[0])
            sample_ids.append(str(_sample_get(sample, "sample_id", f"{dataset_name}:{batch_start + local_idx}")))

        coord = torch.cat(coords, dim=0).to(device=device, dtype=dtype)
        feat = torch.cat(feats, dim=0).to(device=device, dtype=dtype)
        offset = torch.as_tensor(counts, dtype=torch.long, device=device).cumsum(dim=0)
        batch = {
            "coord": coord,
            "feat": feat,
            "grid_size": float(grid_size),
            "offset": offset,
            "sample_ids": sample_ids,
        }
        if grid_coords:
            batch["grid_coord"] = torch.cat(grid_coords, dim=0).to(device=device, dtype=torch.int32)
        batches.append(batch)
    return batches


def forward_model(model, batch):
    out = model(batch)
    return out.feat if hasattr(out, "feat") else out["feat"]


def measure(model, batches, warmup_iters: int, measure_iters: int):
    with torch.inference_mode():
        for _ in range(warmup_iters):
            for batch in batches:
                forward_model(model, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        total_points = 0
        for _ in range(measure_iters):
            for batch in batches:
                forward_model(model, batch)
                total_points += int(batch["coord"].shape[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return total_points / elapsed


def alignment_stats(ours, ref, align_batches):
    ours_feats = []
    ref_feats = []
    sample_ids = []
    with torch.inference_mode():
        for batch in align_batches:
            ours_feats.append(forward_model(ours, batch).float())
            ref_feats.append(forward_model(ref, batch).float())
            sample_ids.extend(batch["sample_ids"])
    if not ours_feats:
        raise RuntimeError("No alignment batches could be built from dataset samples")
    ours_feat = torch.cat(ours_feats, dim=0)
    ref_feat = torch.cat(ref_feats, dim=0)
    feat_cos = torch.nn.functional.cosine_similarity(
        ours_feat.reshape(1, -1), ref_feat.reshape(1, -1)
    ).item()
    feat_mae = torch.mean(torch.abs(ours_feat - ref_feat)).item()
    return {
        "feat_cosine": feat_cos,
        "feat_mae": feat_mae,
        "feat_shape": list(ours_feat.shape),
        "alignment_num_batches": len(align_batches),
        "alignment_num_samples": len(sample_ids),
        "alignment_sample_ids": sample_ids,
    }


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    device = cfg["device"]
    dtype = torch.float16 if cfg["use_fp16"] else torch.float32
    enable_flash = bool(cfg.get("enable_flash", False))
    model_kwargs = default_ptv3_kwargs(enable_flash=enable_flash)

    if cfg["dataset"] != "scanobjectnn":
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
    samples = load_scanobjectnn(
        cfg["dataset_root"],
        cfg["split"],
        cfg["points_per_sample"],
        cfg["max_samples"],
        cfg["grid_size"],
    )
    if len(samples) < cfg["batch_size"]:
        raise RuntimeError(
            f"Need at least batch_size={cfg['batch_size']} samples, found {len(samples)} under {cfg['dataset_root']}"
        )
    align_count = max(cfg["alignment_samples"], cfg["batch_size"])

    throughput_batches = build_batches(
        samples=samples,
        dataset_name=cfg["dataset"],
        batch_size=cfg["batch_size"],
        grid_size=cfg["grid_size"],
        feat_dim=cfg["feat_dim"],
        device=device,
        dtype=dtype,
        drop_last=True,
    )
    align_batches = build_batches(
        samples=samples[:align_count],
        dataset_name=cfg["dataset"],
        batch_size=min(cfg["batch_size"], align_count),
        grid_size=cfg["grid_size"],
        feat_dim=cfg["feat_dim"],
        device=device,
        dtype=dtype,
        drop_last=False,
    )
    if not throughput_batches:
        raise RuntimeError("No full batches could be built from dataset samples")

    set_seed(cfg["seed"])
    ours = load_ours_point_model(cfg["model"], device=device, dtype=dtype, **model_kwargs)
    checkpoint_info = None
    if cfg.get("checkpoint_file"):
        checkpoint_info = load_point_backbone_checkpoint(ours, cfg["checkpoint_file"])
    set_seed(cfg["seed"])
    ref = None
    if not cfg.get("skip_reference", False):
        ref = load_reference_point_model(cfg["model"], device=device, dtype=dtype, **model_kwargs)
        missing, unexpected = ref.load_state_dict(ours.state_dict(), strict=True)
        if missing or unexpected:
            raise RuntimeError(f"PTv3 state mismatch missing={missing} unexpected={unexpected}")

    ours_tps = measure(ours, throughput_batches, cfg["warmup_iters"], cfg["measure_iters"])

    result = {
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "dataset_root": cfg["dataset_root"],
        "split": cfg["split"],
        "num_batches": len(throughput_batches),
        "num_samples": len(samples),
        "batch_size": cfg["batch_size"],
        "points_per_sample": cfg["points_per_sample"],
        "avg_points_after_voxelization": sum(_sample_point_count(sample) for sample in samples) / len(samples),
        "enable_flash": enable_flash,
        "weight_source": "checkpoint" if checkpoint_info is not None else "synchronized_random_init",
        "checkpoint_file": checkpoint_info["checkpoint_file"] if checkpoint_info is not None else None,
        "checkpoint_loaded_tensors": checkpoint_info["loaded_tensors"] if checkpoint_info is not None else 0,
        "correctness_metric": "final_feature_alignment",
        "ours": {"baseline_name": "kb-nano", "points_per_second": ours_tps},
    }

    if ref is not None:
        ref_tps = measure(ref, throughput_batches, cfg["warmup_iters"], cfg["measure_iters"])
        result["reference"] = {"baseline_name": "official-detached", "points_per_second": ref_tps}
        result["comparison"] = alignment_stats(ours, ref, align_batches) | {
            "throughput_ratio": ours_tps / ref_tps if ref_tps > 0 else float("inf"),
        }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PointTransformerV3 on ScanObjectNN")
    parser.add_argument("--model", default="Pointcept/PointTransformerV3")
    parser.add_argument("--dataset", default="scanobjectnn", choices=["scanobjectnn"])
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--split",
        default="nobg_test",
        choices=[
            "bg_test",
            "bg_train",
            "hardest_test",
            "hardest_train",
            "nobg_test",
            "nobg_train",
        ],
    )
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--alignment-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--points-per-sample", type=int, default=2048)
    parser.add_argument("--feat-dim", type=int, default=6)
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-file", default=DEFAULT_PTV3_CHECKPOINT_FILE)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/pointtransv3_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = Path(args.dataset_root) if args.dataset_root else (_KB_ROOT / "data" / args.dataset)
    cfg = {
        "kb_root": str(_KB_ROOT),
        "model": args.model,
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "device": device,
        "max_samples": args.max_samples,
        "alignment_samples": args.alignment_samples,
        "batch_size": args.batch_size,
        "points_per_sample": args.points_per_sample,
        "feat_dim": args.feat_dim,
        "grid_size": args.grid_size,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "seed": args.seed,
        "checkpoint_file": args.checkpoint_file,
        "use_fp16": bool(args.use_fp16 and device == "cuda"),
        "enable_flash": args.enable_flash,
        "skip_reference": args.skip_reference,
    }
    data = run_worker(POINTCLOUD_WORKER, cfg, "PointTransformerV3 benchmark", timeout=7200)
    if data is None:
        raise SystemExit(1)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
