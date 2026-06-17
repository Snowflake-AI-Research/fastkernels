#!/usr/bin/env python3
"""Throughput, latency, and correctness benchmark: kb-nano DP3 vs reference DP3.

Compares kb-nano's L1-L4 DP3 implementation against the official 3D Diffusion
Policy reference (https://github.com/YanjieZe/3D-Diffusion-Policy) on real
3-D point-cloud robotics data.

Inputs (default — real data):
  - **Real point clouds + robot state + actions** from
    ``rishabhrj11/gym-xarm-pointcloud`` on HuggingFace Hub:
    18374 frames across 50 episodes of a gym-xarm push-block task with
    point clouds extracted from depth cameras (512 points x 6 channels
    XYZRGB; we slice to XYZ for ``use_pc_color=False``), 7-D joint state,
    and 4-D end-effector actions.
  - **Shared deterministic noise** (saved to disk and read by both engines)
    for byte-equal flow initialisation.

Weights:
  - DP3 publishes no pretrained checkpoints. By default we initialise a
    kb-nano DP3 model with a fixed seed and save it in the reference's
    checkpoint format (``cfg`` + ``state_dicts['model']``); both engines
    load this shared checkpoint. Random-init weights are sufficient for
    kernel/math-equivalence validation: the U-Net runs the same code path
    regardless of weight values.
  - Pass ``--checkpoint /path/to/latest.ckpt`` to use a real trained
    checkpoint produced by the reference's ``train.py``.

What this validates:
  - End-to-end correctness: kb-nano vs reference action chunks per frame
    (max abs diff + cosine similarity) on real point cloud inputs.
  - Speed parity: throughput (inferences/sec) and latency (P50/P99
    per-inference ms) on identical inputs and weights.

What this does NOT validate (out of scope):
  - Real-task accuracy (would require setting up MuJoCo + scripted-policy
    demo generation + multi-hour training).

Usage:
    python tests/bench_dp3.py                              # real xarm data
    python tests/bench_dp3.py --skip-reference             # kb-nano only
    python tests/bench_dp3.py --variant dp3                # full DP3 (heavier U-Net)
    python tests/bench_dp3.py --num-requests 50
    python tests/bench_dp3.py --synthetic-only             # debug only
    python tests/bench_dp3.py --checkpoint /path/to/dp3.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.worker import run_worker
from kb_nano.bench.utils.workloads import (
    DP3_CONFIG,
    DP3_THROUGHPUT_WORKLOADS,
    DP3_LATENCY_WORKLOADS,
)


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Shared checkpoint + dataset materialization (parent-process side)
# ---------------------------------------------------------------------------

def materialize_shared_checkpoint(
    ckpt_dir: str, variant: str, seed: int,
    action_dim: int, state_dim: int, num_points: int,
    dp3_repo: str,
) -> str:
    """Build a shared DP3 checkpoint from a reference-DP3 random-init policy.

    We build the **reference** ``DP3`` policy (which uses ``nn.Linear`` /
    ``nn.Conv1d`` with proper Kaiming-uniform init) and save its state_dict.
    Both kb-nano and reference workers then load identical
    properly-initialized weights from this single file. This avoids
    kb-nano's L1 ``Linear`` having uninitialized weights (it uses
    ``torch.empty()`` and relies on a checkpoint to fill values — fine in
    production, but breaks random-init benchmarks).

    Args:
        action_dim / state_dim / num_points: dimensions taken from the real
            dataset so kb-nano and reference share the same architecture.
        dp3_repo: path to the reference 3D-Diffusion-Policy clone (must be
            on PYTHONPATH for the reference module imports).

    Returns the path to the saved checkpoint.
    """
    import sys as _sys
    import types as _types
    import torch
    import pickle
    from omegaconf import OmegaConf

    # Stub pytorch3d (only used in env wrappers) and inject reference repo.
    _sys.modules.setdefault("pytorch3d", _types.ModuleType("pytorch3d"))
    _sys.modules.setdefault("pytorch3d.ops", _types.ModuleType("pytorch3d.ops"))
    if dp3_repo not in _sys.path:
        _sys.path.insert(0, dp3_repo)

    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusion_policy_3d.policy.dp3 import DP3 as RefDP3
    from diffusion_policy_3d.model.common.normalizer import (
        LinearNormalizer as RefNorm,
    )

    if variant == "simple_dp3":
        down_dims = [128, 256, 384]
    else:
        down_dims = [512, 1024, 2048]

    diffusion_step_embed_dim = 128
    encoder_output_dim = 64
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8

    # Build the OmegaConf cfg both engines (and the reference's __init__) need.
    sched = OmegaConf.create({
        "_target_": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
        "num_train_timesteps": 100,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "set_alpha_to_one": True,
        "steps_offset": 0,
        "prediction_type": "sample",
    })
    pcfg = OmegaConf.create({
        "in_channels": 3,
        "out_channels": encoder_output_dim,
        "use_layernorm": True,
        "final_norm": "layernorm",
        "normal_channel": False,
    })
    policy_cfg = OmegaConf.create({
        "_target_": (
            "diffusion_policy_3d.policy.simple_dp3.SimpleDP3"
            if variant == "simple_dp3"
            else "diffusion_policy_3d.policy.dp3.DP3"
        ),
        "use_point_crop": True,
        "condition_type": "film",
        "use_down_condition": True,
        "use_mid_condition": True,
        "use_up_condition": True,
        "diffusion_step_embed_dim": diffusion_step_embed_dim,
        "down_dims": down_dims,
        "crop_shape": [80, 80],
        "encoder_output_dim": encoder_output_dim,
        "horizon": horizon,
        "kernel_size": 5,
        "n_action_steps": n_action_steps,
        "n_groups": 8,
        "n_obs_steps": n_obs_steps,
        "noise_scheduler": sched,
        "num_inference_steps": 10,
        "obs_as_global_cond": True,
        "use_pc_color": False,
        "pointnet_type": "pointnet",
        "pointcloud_encoder_cfg": pcfg,
    })
    shape_meta = OmegaConf.create({
        "obs": {
            "point_cloud": {"shape": [num_points, 3], "type": "point_cloud"},
            "agent_pos":   {"shape": [state_dim], "type": "low_dim"},
        },
        "action": {"shape": [action_dim]},
    })
    full_cfg = OmegaConf.create({
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "shape_meta": shape_meta,
        "policy": policy_cfg,
        "task_name": "synthetic_init",
    })
    full_cfg.policy.shape_meta = shape_meta

    # Build the reference DP3 (proper Kaiming init via nn.Linear / nn.Conv1d).
    torch.manual_seed(seed)
    ddim = DDIMScheduler(
        num_train_timesteps=100, beta_start=1e-4, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        set_alpha_to_one=True, steps_offset=0, prediction_type="sample",
    )
    ref_policy = RefDP3(
        shape_meta=shape_meta, noise_scheduler=ddim,
        horizon=horizon, n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
        num_inference_steps=10, obs_as_global_cond=True,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=tuple(down_dims), kernel_size=5, n_groups=8,
        condition_type="film",
        use_down_condition=True, use_mid_condition=True, use_up_condition=True,
        encoder_output_dim=encoder_output_dim, use_pc_color=False,
        pointnet_type="pointnet", pointcloud_encoder_cfg=pcfg,
    )
    # Fit a *real* per-channel normalizer to the real dataset's first frames
    # — exercises the normalize/unnormalize math (scale/offset broadcast,
    # forward/inverse direction) instead of an identity no-op. Both engines
    # consume the resulting scale/offset from the saved state_dict.
    fit_data = _load_normalizer_fit_data(num_points, state_dim, action_dim)
    norm = RefNorm()
    norm.fit(data=fit_data, last_n_dims=1, mode="limits")
    ref_policy.set_normalizer(norm)

    state_dict = ref_policy.state_dict()

    payload = {
        "cfg": full_cfg,
        "state_dicts": {"model": state_dict},
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "latest.ckpt")
    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"  Shared checkpoint saved: {ckpt_path}  "
        f"({len(state_dict)} state-dict entries from reference DP3)",
        flush=True,
    )
    return ckpt_path


def _load_normalizer_fit_data(num_points: int, state_dim: int, action_dim: int):
    """Load ~1000 frames of real xarm data to fit per-channel normalizer
    statistics. Falls back to standard-normal samples if the dataset can't
    be reached (e.g. offline run); the normalizer math is exercised either
    way."""
    import numpy as _np
    import torch as _torch
    try:
        pc, st, ac, _ = _load_xarm_pointclouds(
            "rishabhrj11/gym-xarm-pointcloud", 1000, num_points, seed=0,
        )
    except Exception as e:
        print(
            f"  WARNING: normalizer fit fallback to standard normal ({e}).",
            file=sys.stderr,
        )
        rng = _np.random.default_rng(0)
        pc = rng.standard_normal((1000, num_points, 3)).astype(_np.float32)
        st = rng.standard_normal((1000, state_dim)).astype(_np.float32)
        ac = rng.standard_normal((1000, action_dim)).astype(_np.float32)

    # The reference's normalizer.fit handles per-key dicts; pass torch
    # tensors with the same dtype (float32) the policy uses at inference.
    return {
        "point_cloud": _torch.from_numpy(pc[..., :3]).float(),
        "agent_pos":   _torch.from_numpy(st).float(),
        "action":      _torch.from_numpy(ac).float(),
    }


def materialize_dataset(
    cache_dir: str,
    num_frames: int,
    num_points: int,
    state_dim: int,
    action_dim: int,
    n_obs_steps: int,
    horizon: int,
    seed: int,
    dataset_name: str,
    use_real_data: bool,
) -> tuple[dict, "np.ndarray | None"]:
    """Materialize point clouds + state + shared noise tensors.

    Saves to ``cache_dir`` and returns ``(paths, gt_actions)``. ``gt_actions``
    is ``(num_frames, action_dim)`` ground-truth actions when real data is
    used (otherwise ``None``); the bench reports it for transparency but
    does not use it as a correctness target — kb-nano vs reference must
    match each other, not the GT actions.
    """
    import torch
    import numpy as _np

    os.makedirs(cache_dir, exist_ok=True)

    pc_arr: "_np.ndarray | None" = None
    state_arr: "_np.ndarray | None" = None
    action_arr: "_np.ndarray | None" = None
    episode_arr: "_np.ndarray | None" = None

    if use_real_data:
        try:
            pc_arr, state_arr, action_arr, episode_arr = _load_xarm_pointclouds(
                dataset_name, num_frames, num_points, seed,
            )
        except Exception as e:
            print(
                f"  WARNING: real-data load failed ({e}); "
                "falling back to synthetic.", file=sys.stderr,
            )

    rng = _np.random.default_rng(seed)
    torch_g = torch.Generator(device="cpu").manual_seed(seed)

    if pc_arr is None:
        pc_arr = rng.standard_normal(
            (num_frames, num_points, 3),
        ).astype(_np.float32)
    if state_arr is None:
        # Synthetic state: low-frequency sinusoid per dim.
        t = _np.arange(num_frames)[:, None]
        freqs = rng.uniform(0.05, 0.3, size=(state_dim,))
        phases = rng.uniform(0, 2 * _np.pi, size=(state_dim,))
        state_arr = _np.sin(
            t * freqs[None, :] + phases[None, :]
        ).astype(_np.float32)
    if action_arr is None:
        action_arr = _np.zeros((num_frames, action_dim), dtype=_np.float32)
    if episode_arr is None:
        episode_arr = _np.zeros(num_frames, dtype=_np.int64)

    # Build To-step rolling windows. Where frames cross episode boundaries,
    # we pad with the first frame of the new episode so windows stay
    # within-episode (matches how DP3's SequenceSampler pads with pad_before).
    pc_windows = _np.empty(
        (num_frames, n_obs_steps, num_points, pc_arr.shape[-1]),
        dtype=_np.float32,
    )
    state_windows = _np.empty(
        (num_frames, n_obs_steps, state_dim), dtype=_np.float32,
    )
    for i in range(num_frames):
        for k in range(n_obs_steps):
            j = i - (n_obs_steps - 1 - k)
            if j < 0 or episode_arr[j] != episode_arr[i]:
                j = i
            pc_windows[i, k] = pc_arr[j]
            state_windows[i, k] = state_arr[j]

    # Per-frame shared noise tensors.
    noise = torch.randn(
        num_frames, horizon, action_dim, generator=torch_g, dtype=torch.float32,
    ).numpy()

    paths = {
        "point_cloud": os.path.join(cache_dir, "point_cloud.npy"),
        "agent_pos":   os.path.join(cache_dir, "agent_pos.npy"),
        "noise":       os.path.join(cache_dir, "noise.npy"),
        "gt_actions":  os.path.join(cache_dir, "gt_actions.npy"),
    }
    _np.save(paths["point_cloud"], pc_windows)
    _np.save(paths["agent_pos"],   state_windows)
    _np.save(paths["noise"],       noise)
    _np.save(paths["gt_actions"],  action_arr)
    print(
        f"  Dataset materialised: PC{tuple(pc_windows.shape)}  "
        f"state{tuple(state_windows.shape)}  noise{tuple(noise.shape)}  "
        f"action{tuple(action_arr.shape)}",
        flush=True,
    )
    return paths, action_arr


def _load_xarm_pointclouds(
    dataset_name: str, num_frames: int, num_points: int, seed: int,
):
    """Load real point clouds + state + actions from gym-xarm-pointcloud.

    The xarm dataset stores per-frame:
      - ``observation.environment_state``: (512, 6) XYZRGB point cloud
        (we slice to the first 3 channels for ``use_pc_color=False``).
      - ``observation.state``: (7,) joint angles.
      - ``action``: (4,) end-effector control (x, y, z, gripper).
      - ``episode_index``: episode id (used to keep To-step windows inside
        a single episode).

    Strategy: download the first parquet shard (~9 MB), slice the first
    ``num_frames`` rows. This avoids the full ~80 MB download for small
    bench runs. Frames are kept in their original order so the rolling-window
    construction respects episode boundaries.
    """
    import numpy as _np
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print(f"  Loading {num_frames} frames from {dataset_name} ...", flush=True)

    cols = [
        "observation.environment_state",
        "observation.state",
        "action",
        "episode_index",
    ]
    # The dataset is sharded into multiple parquet files; we may need to
    # walk shards if num_frames exceeds the first shard's row count.
    shard_idx = 0
    pc_rows: list[_np.ndarray] = []
    st_rows: list[_np.ndarray] = []
    ac_rows: list[_np.ndarray] = []
    ep_rows: list[_np.ndarray] = []
    while sum(len(r) for r in pc_rows) < num_frames:
        try:
            path = hf_hub_download(
                dataset_name,
                f"data/chunk-000/file-{shard_idx:03d}.parquet",
                repo_type="dataset",
            )
        except Exception as e:
            if shard_idx == 0:
                raise
            print(f"  Stopping at shard {shard_idx} (no more): {e}",
                  file=sys.stderr)
            break
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()

        # observation.environment_state is a list-of-arrays (one per point);
        # stack to (rows, 512, 6).
        pc = _np.stack([_np.stack(list(row), axis=0) for row in df[cols[0]]])
        st = _np.stack([_np.asarray(x, dtype=_np.float32) for x in df[cols[1]]])
        ac = _np.stack([_np.asarray(x, dtype=_np.float32) for x in df[cols[2]]])
        ep = _np.asarray(df[cols[3]].astype(_np.int64))

        pc_rows.append(pc.astype(_np.float32))
        st_rows.append(st)
        ac_rows.append(ac)
        ep_rows.append(ep)
        shard_idx += 1

    pc_full = _np.concatenate(pc_rows, axis=0)[:num_frames]
    st_full = _np.concatenate(st_rows, axis=0)[:num_frames]
    ac_full = _np.concatenate(ac_rows, axis=0)[:num_frames]
    ep_full = _np.concatenate(ep_rows, axis=0)[:num_frames]

    # Keep XYZ only (DP3 use_pc_color=False).
    pc_xyz = pc_full[..., :3]
    # If the dataset has != num_points per frame, sub-sample / pad.
    if pc_xyz.shape[1] != num_points:
        rng = _np.random.default_rng(seed)
        if pc_xyz.shape[1] >= num_points:
            idx = rng.choice(pc_xyz.shape[1], size=num_points, replace=False)
            pc_xyz = pc_xyz[:, idx]
        else:
            pad = _np.repeat(
                pc_xyz[:, :1], num_points - pc_xyz.shape[1], axis=1,
            )
            pc_xyz = _np.concatenate([pc_xyz, pad], axis=1)

    print(
        f"  Loaded xarm: PC{tuple(pc_xyz.shape)}  state{tuple(st_full.shape)}  "
        f"action{tuple(ac_full.shape)}  episodes={len(_np.unique(ep_full))}",
        flush=True,
    )
    return pc_xyz, st_full, ac_full, ep_full


# ---------------------------------------------------------------------------
# kb-nano DP3 worker (runs in subprocess)
# ---------------------------------------------------------------------------

KB_NANO_DP3_WORKER = r'''
import json, os, sys, time
with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

import numpy as np
import torch

from kb_nano.infra.dp3_engine import DP3Engine
from kb_nano.tasks.baseline.L4.dp3 import DP3SamplingParams

def main():
    engine = DP3Engine(
        checkpoint_path=cfg["checkpoint"],
        seed=cfg["seed"],
        dtype=torch.float32,
        device="cuda",
        enforce_eager=cfg.get("enforce_eager", False),
    )
    engine.warmup(num_steps=cfg.get("num_inference_steps", 10))

    pc      = np.load(cfg["data"]["point_cloud"])      # (N, To, P, 3)
    state   = np.load(cfg["data"]["agent_pos"])        # (N, To, state_dim)
    noise   = np.load(cfg["data"]["noise"])            # (N, horizon, action_dim)
    N = pc.shape[0]

    actions_dir = cfg.get("actions_dir")
    if actions_dir:
        os.makedirs(actions_dir, exist_ok=True)

    results = {"throughput": [], "latency": []}

    for sc in cfg.get("scenarios", []):
        bsz = sc["batch_size"]
        n_req = sc["num_requests"]
        pc_t   = torch.from_numpy(pc[:n_req]).cuda()
        st_t   = torch.from_numpy(state[:n_req]).cuda()
        noi_t  = torch.from_numpy(noise[:n_req]).cuda()
        # Per-scenario warmup at the matching batch size — absorbs any
        # compile / first-call overhead so the timed phase is apples-to-apples.
        for _ in range(2):
            _ = engine.generate(
                pc_t[:bsz], st_t[:bsz],
                params=DP3SamplingParams(num_inference_steps=cfg.get("num_inference_steps")),
                noise=noi_t[:bsz],
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n_req, bsz):
            j = min(i + bsz, n_req)
            out = engine.generate(
                pc_t[i:j], st_t[i:j],
                params=DP3SamplingParams(
                    num_inference_steps=cfg.get("num_inference_steps"),
                ),
                noise=noi_t[i:j],
            )
            if actions_dir:
                act = out.actions.detach().cpu().numpy()
                np.save(os.path.join(actions_dir, f"{sc['name']}_{i:05d}.npy"), act)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        results["throughput"].append({
            "name": sc["name"], "batch_size": bsz, "num_requests": n_req,
            "elapsed_s": elapsed, "throughput_req_s": n_req / elapsed,
        })
        print(f"  [kb-nano] {sc['name']}: {n_req} reqs in {elapsed:.3f}s "
              f"= {n_req/elapsed:.2f} req/s", flush=True)

    for sc in cfg.get("latency_scenarios", []):
        bsz = sc["batch_size"]
        # warmup
        for _ in range(sc["num_warmup"]):
            _ = engine.generate(
                torch.from_numpy(pc[:bsz]).cuda(),
                torch.from_numpy(state[:bsz]).cuda(),
                noise=torch.from_numpy(noise[:bsz]).cuda(),
            )
        torch.cuda.synchronize()

        latencies = []
        for k in range(sc["num_iters"]):
            i = k % max(1, (pc.shape[0] - bsz))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = engine.generate(
                torch.from_numpy(pc[i:i+bsz]).cuda(),
                torch.from_numpy(state[i:i+bsz]).cuda(),
                noise=torch.from_numpy(noise[i:i+bsz]).cuda(),
            )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latencies.sort()
        results["latency"].append({
            "name": sc["name"], "batch_size": bsz,
            "p50_ms": latencies[len(latencies)//2],
            "p99_ms": latencies[max(0, int(len(latencies)*0.99) - 1)],
            "mean_ms": float(sum(latencies)/len(latencies)),
            "min_ms": min(latencies), "max_ms": max(latencies),
            "all_ms": latencies,
        })
        print(f"  [kb-nano] {sc['name']} bsz={bsz}: "
              f"P50={results['latency'][-1]['p50_ms']:.2f}ms "
              f"P99={results['latency'][-1]['p99_ms']:.2f}ms", flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

main()
'''


# ---------------------------------------------------------------------------
# Reference DP3 worker (runs in same venv with pytorch3d stubbed)
# ---------------------------------------------------------------------------

REFERENCE_DP3_WORKER = r'''
import json, os, sys, time, types
with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

# Stub pytorch3d (only used in env wrappers which we don't touch).
sys.modules.setdefault("pytorch3d", types.ModuleType("pytorch3d"))
sys.modules.setdefault("pytorch3d.ops", types.ModuleType("pytorch3d.ops"))
sys.path.insert(0, cfg["dp3_repo"])

import pickle
import numpy as np
import torch

from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy_3d.policy.dp3 import DP3 as RefDP3

def _build_ref_policy(payload):
    fcfg = payload["cfg"]
    sched = fcfg.policy.noise_scheduler
    ddim = DDIMScheduler(
        num_train_timesteps=int(sched.num_train_timesteps),
        beta_start=float(sched.beta_start), beta_end=float(sched.beta_end),
        beta_schedule=str(sched.beta_schedule),
        clip_sample=bool(sched.clip_sample),
        set_alpha_to_one=bool(sched.set_alpha_to_one),
        steps_offset=int(sched.steps_offset),
        prediction_type=str(sched.prediction_type),
    )
    pol = fcfg.policy
    return RefDP3(
        shape_meta=fcfg.shape_meta,
        noise_scheduler=ddim,
        horizon=int(fcfg.horizon),
        n_action_steps=int(fcfg.n_action_steps),
        n_obs_steps=int(fcfg.n_obs_steps),
        num_inference_steps=int(pol.num_inference_steps),
        obs_as_global_cond=bool(pol.obs_as_global_cond),
        diffusion_step_embed_dim=int(pol.diffusion_step_embed_dim),
        down_dims=list(pol.down_dims),
        kernel_size=int(pol.kernel_size),
        n_groups=int(pol.n_groups),
        condition_type=str(pol.condition_type),
        use_down_condition=bool(pol.use_down_condition),
        use_mid_condition=bool(pol.use_mid_condition),
        use_up_condition=bool(pol.use_up_condition),
        encoder_output_dim=int(pol.encoder_output_dim),
        use_pc_color=bool(pol.use_pc_color),
        pointnet_type=str(pol.pointnet_type),
        pointcloud_encoder_cfg=pol.pointcloud_encoder_cfg,
    )

def main():
    with open(cfg["checkpoint"], "rb") as f:
        payload = pickle.load(f)
    # Build on CPU first; ``DictOfTensorMixin._load_from_state_dict``
    # *reassigns* ``params_dict`` to a fresh ParameterDict built from the
    # state-dict tensors, which would put the normalizer back on CPU even
    # after a pre-load .cuda() — so we move to GPU AFTER load_state_dict.
    policy = _build_ref_policy(payload)

    sd = payload["state_dicts"]["model"]
    # Drop kb-nano-only marker / ref-only buffers that aren't in sd.
    sd = dict(sd)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    benign_missing = ("noise_scheduler_pc.", "mask_generator.", "_dummy_variable")
    real_missing = [m for m in missing if not m.startswith(benign_missing)]
    if real_missing:
        print(f"  WARNING: ref missing keys: {real_missing[:10]}", file=sys.stderr)
    if unexpected:
        print(f"  WARNING: ref unexpected keys: {unexpected[:10]}", file=sys.stderr)
    # Sanity: normalizer keys must be populated after load.
    assert "point_cloud" in policy.normalizer.params_dict, (
        "normalizer.params_dict missing 'point_cloud' after load — "
        "DictOfTensorMixin._load_from_state_dict did not run."
    )
    policy = policy.cuda()
    # Pre-move scheduler's alphas_cumprod to GPU so the per-step
    # scheduler.step() arithmetic runs on GPU — matches kb-nano's
    # DP3Pipeline.conditional_sample, which does the same. Without this,
    # diffusers lazily transfers per call and accumulates 1-ULP-per-step
    # rounding drift over 10 DDIM steps.
    policy.noise_scheduler.alphas_cumprod = (
        policy.noise_scheduler.alphas_cumprod.cuda()
    )
    policy.eval()

    pc      = np.load(cfg["data"]["point_cloud"])
    state   = np.load(cfg["data"]["agent_pos"])
    noise   = np.load(cfg["data"]["noise"])

    actions_dir = cfg.get("actions_dir")
    if actions_dir:
        os.makedirs(actions_dir, exist_ok=True)

    results = {"throughput": [], "latency": []}

    # We pre-stage tensors to GPU once.
    pc_all = torch.from_numpy(pc).cuda()
    state_all = torch.from_numpy(state).cuda()
    noise_all = torch.from_numpy(noise).cuda()

    # Patch torch.randn so the trajectory init in conditional_sample uses
    # our shared noise tensor (matches kb-nano injection).
    orig_randn = torch.randn
    g = {"noise": None}
    def my_randn(*a, **kw):
        sz = a[0] if a else kw.get("size")
        if sz is not None and g["noise"] is not None and tuple(sz) == tuple(g["noise"].shape):
            return g["noise"].clone()
        return orig_randn(*a, **kw)

    # warmup
    torch.randn = my_randn
    g["noise"] = noise_all[:1]
    with torch.no_grad():
        for _ in range(2):
            _ = policy.predict_action({"point_cloud": pc_all[:1], "agent_pos": state_all[:1]})
    torch.cuda.synchronize()
    torch.randn = orig_randn

    for sc in cfg.get("scenarios", []):
        bsz = sc["batch_size"]
        n_req = sc["num_requests"]
        torch.randn = my_randn
        # Per-scenario warmup at matching bsz — apples-to-apples vs kb-nano.
        for _ in range(2):
            g["noise"] = noise_all[:bsz]
            with torch.no_grad():
                _ = policy.predict_action({
                    "point_cloud": pc_all[:bsz], "agent_pos": state_all[:bsz],
                })
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n_req, bsz):
            j = min(i + bsz, n_req)
            g["noise"] = noise_all[i:j]
            with torch.no_grad():
                out = policy.predict_action({
                    "point_cloud": pc_all[i:j],
                    "agent_pos":   state_all[i:j],
                })
            if actions_dir:
                act = out["action"].detach().cpu().numpy()
                np.save(os.path.join(actions_dir, f"{sc['name']}_{i:05d}.npy"), act)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        torch.randn = orig_randn
        results["throughput"].append({
            "name": sc["name"], "batch_size": bsz, "num_requests": n_req,
            "elapsed_s": elapsed, "throughput_req_s": n_req / elapsed,
        })
        print(f"  [reference] {sc['name']}: {n_req} reqs in {elapsed:.3f}s "
              f"= {n_req/elapsed:.2f} req/s", flush=True)

    for sc in cfg.get("latency_scenarios", []):
        bsz = sc["batch_size"]
        torch.randn = my_randn
        # warmup
        for _ in range(sc["num_warmup"]):
            g["noise"] = noise_all[:bsz]
            with torch.no_grad():
                _ = policy.predict_action({"point_cloud": pc_all[:bsz], "agent_pos": state_all[:bsz]})
        torch.cuda.synchronize()
        latencies = []
        for k in range(sc["num_iters"]):
            i = k % max(1, (pc.shape[0] - bsz))
            g["noise"] = noise_all[i:i+bsz]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = policy.predict_action({
                    "point_cloud": pc_all[i:i+bsz],
                    "agent_pos":   state_all[i:i+bsz],
                })
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        torch.randn = orig_randn
        latencies.sort()
        results["latency"].append({
            "name": sc["name"], "batch_size": bsz,
            "p50_ms": latencies[len(latencies)//2],
            "p99_ms": latencies[max(0, int(len(latencies)*0.99) - 1)],
            "mean_ms": float(sum(latencies)/len(latencies)),
            "min_ms": min(latencies), "max_ms": max(latencies),
            "all_ms": latencies,
        })
        print(f"  [reference] {sc['name']} bsz={bsz}: "
              f"P50={results['latency'][-1]['p50_ms']:.2f}ms "
              f"P99={results['latency'][-1]['p99_ms']:.2f}ms", flush=True)

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

main()
'''


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_throughput_comparison(kb, ref):
    print(f"\n{'Scenario':<22}{'kb-nano req/s':>16}{'reference req/s':>20}{'speedup':>12}")
    print("─" * 70)
    rd = {r["name"]: r for r in (ref or [])}
    for k in kb:
        r = rd.get(k["name"])
        if r:
            spd = k["throughput_req_s"] / r["throughput_req_s"]
            print(f"{k['name']:<22}{k['throughput_req_s']:>16.2f}"
                  f"{r['throughput_req_s']:>20.2f}{spd:>11.2f}x")
        else:
            print(f"{k['name']:<22}{k['throughput_req_s']:>16.2f}{'—':>20}{'—':>12}")
    print()


def _print_latency_comparison(kb, ref):
    print(f"{'Scenario':<22}{'kb P50/P99 ms':>22}{'ref P50/P99 ms':>22}{'speedup(P50)':>16}")
    print("─" * 82)
    rd = {r["name"]: r for r in (ref or [])}
    for k in kb:
        r = rd.get(k["name"])
        if r:
            spd = r["p50_ms"] / k["p50_ms"]
            print(f"{k['name']:<22}"
                  f"{f'{k['p50_ms']:.2f}/{k['p99_ms']:.2f}':>22}"
                  f"{f'{r['p50_ms']:.2f}/{r['p99_ms']:.2f}':>22}"
                  f"{spd:>15.2f}x")
        else:
            print(f"{k['name']:<22}"
                  f"{f'{k['p50_ms']:.2f}/{k['p99_ms']:.2f}':>22}"
                  f"{'—':>22}{'—':>16}")
    print()


def _compare_actions(kb_dir: str, ref_dir: str) -> dict:
    """Compute per-scenario MSE + cosine sim between saved kb / ref actions."""
    out: dict = {}
    if not (os.path.isdir(kb_dir) and os.path.isdir(ref_dir)):
        return out
    kb_files = sorted(os.listdir(kb_dir))
    for name in kb_files:
        kb_p = os.path.join(kb_dir, name)
        ref_p = os.path.join(ref_dir, name)
        if not os.path.isfile(ref_p):
            continue
        kb_a = np.load(kb_p).astype(np.float64)
        ref_a = np.load(ref_p).astype(np.float64)
        if kb_a.shape != ref_a.shape:
            continue
        scenario = name.split("_")[0]
        d = out.setdefault(scenario, {"mses": [], "coss": [], "max_abs": 0.0})
        d["mses"].append(float(np.mean((kb_a - ref_a) ** 2)))
        kb_f, ref_f = kb_a.flatten(), ref_a.flatten()
        denom = (np.linalg.norm(kb_f) * np.linalg.norm(ref_f)) or 1.0
        d["coss"].append(float(np.dot(kb_f, ref_f) / denom))
        d["max_abs"] = max(d["max_abs"], float(np.abs(kb_a - ref_a).max()))
    summary = {}
    for sc, d in out.items():
        summary[sc] = {
            "mean_mse": float(np.mean(d["mses"])),
            "mean_cos": float(np.mean(d["coss"])),
            "max_abs": d["max_abs"],
            "num_samples": len(d["mses"]),
        }
    return summary


def _print_correctness(c: dict):
    print(f"{'Scenario':<22}{'samples':>10}{'mean MSE':>14}{'mean cos':>12}{'max abs':>12}{'verdict':>10}")
    print("─" * 80)
    for sc, s in c.items():
        cos = s["mean_cos"]
        verdict = "PASS" if cos > 0.999 else ("WARN" if cos > 0.95 else "FAIL")
        print(f"{sc:<22}{s['num_samples']:>10}{s['mean_mse']:>14.6e}"
              f"{cos:>12.6f}{s['max_abs']:>12.6e}{verdict:>10}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DP3 benchmark: kb-nano vs reference 3D-Diffusion-Policy",
    )
    parser.add_argument("--variant", choices=("simple_dp3", "dp3"),
                        default="simple_dp3",
                        help="Model variant — Simple-DP3 (smaller, ~25 FPS on A40) or full DP3.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a real reference DP3 ckpt (latest.ckpt). "
                             "If unset, a kb-nano random-init checkpoint is created.")
    parser.add_argument("--dp3-repo", type=str,
                        default="/raid/user_data/olu/3D-Diffusion-Policy/3D-Diffusion-Policy",
                        help="Path to the cloned 3D-Diffusion-Policy repo (must be "
                             "on PYTHONPATH for the reference worker).")
    parser.add_argument("--dataset", type=str,
                        default="rishabhrj11/gym-xarm-pointcloud",
                        help="HF point-cloud robotics dataset (or 'synthetic').")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=DP3_CONFIG.num_inference_steps,
                        help="DDIM inference steps (default 10 — matches reference).")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of frames per throughput scenario.")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Skip torch.compile on kb-nano (default — apples-to-apples vs ref).")
    parser.add_argument("--torch-compile", dest="enforce_eager", action="store_false",
                        help="Enable torch.compile(reduce-overhead) on kb-nano U-Net.")
    parser.add_argument("--skip-reference", action="store_true",
                        help="kb-nano only.")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Use synthetic Gaussian point clouds (debug).")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()
    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu_name / f"dp3_{args.variant}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nBenchmark: DP3 ({args.variant}) on {gpu_name}")
    print(f"Dataset:    {args.dataset}")
    print(f"Seed: {args.seed}  |  Steps: {args.num_steps}  |  Requests: {args.num_requests}")
    print(f"Output dir: {args.output_dir}")

    # --- 1. Materialize point cloud + state + noise data first (so we know
    # the real action / state dims from the dataset).
    cache_dir = os.path.join(args.output_dir, "data_cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    paths, gt_actions = materialize_dataset(
        cache_dir=cache_dir,
        num_frames=args.num_requests,
        num_points=DP3_CONFIG.num_points,
        state_dim=DP3_CONFIG.state_dim,
        action_dim=DP3_CONFIG.action_dim,
        n_obs_steps=DP3_CONFIG.n_obs_steps,
        horizon=DP3_CONFIG.horizon,
        seed=args.seed,
        dataset_name=args.dataset,
        use_real_data=not args.synthetic_only and args.dataset != "synthetic",
    )

    # Detect the real dataset's dims from the saved arrays (drives the
    # shared checkpoint). Falls back to DP3_CONFIG defaults if synthetic.
    pc_real = np.load(paths["point_cloud"])      # (N, To, P, 3)
    st_real = np.load(paths["agent_pos"])        # (N, To, state_dim)
    ac_real = np.load(paths["gt_actions"])       # (N, action_dim)
    real_action_dim = int(ac_real.shape[-1])
    real_state_dim  = int(st_real.shape[-1])
    real_num_points = int(pc_real.shape[2])
    print(
        f"  Using DP3Config: action_dim={real_action_dim}  "
        f"state_dim={real_state_dim}  num_points={real_num_points}",
        flush=True,
    )

    # --- 2. Materialize / locate the shared checkpoint sized to the data.
    if args.checkpoint:
        ckpt_path = args.checkpoint
        print(f"  Using user-supplied checkpoint: {ckpt_path}")
    else:
        if not os.path.isdir(args.dp3_repo):
            print(
                f"ERROR: --dp3-repo {args.dp3_repo} not found; cannot build "
                "shared checkpoint without the reference DP3 source.",
                file=sys.stderr,
            )
            return 1
        ckpt_dir = os.path.join(args.output_dir, "shared_ckpt")
        ckpt_path = materialize_shared_checkpoint(
            ckpt_dir, args.variant, args.seed,
            action_dim=real_action_dim,
            state_dim=real_state_dim,
            num_points=real_num_points,
            dp3_repo=args.dp3_repo,
        )

    scenarios = [
        {"name": w.name, "batch_size": w.batch_size, "num_requests": min(args.num_requests, w.num_requests)}
        for w in DP3_THROUGHPUT_WORKLOADS
    ] if not args.skip_throughput else []
    latency_scenarios = [
        {"name": w.name, "batch_size": w.batch_size,
         "num_warmup": w.num_warmup, "num_iters": w.num_iters}
        for w in DP3_LATENCY_WORKLOADS
    ] if not args.skip_latency else []

    # --- 3. Run kb-nano worker ---
    actions_root = os.path.join(args.output_dir, "actions")
    if os.path.isdir(actions_root):
        shutil.rmtree(actions_root)
    kb_actions = os.path.join(actions_root, "kb_nano")
    ref_actions = os.path.join(actions_root, "reference")

    kb_cfg = {
        "project_root": str(_PROJECT_ROOT),
        "checkpoint": ckpt_path,
        "data": paths,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "actions_dir": kb_actions,
        "seed": args.seed,
        "num_inference_steps": args.num_steps,
        "enforce_eager": args.enforce_eager,
    }
    print("\n--- kb-nano DP3 ---")
    kb_data = run_worker(KB_NANO_DP3_WORKER, kb_cfg, "kb-nano DP3", timeout=3600)
    if kb_data is None:
        print("ERROR: kb-nano worker failed.", file=sys.stderr)
        return 1

    # --- 4. Run reference worker ---
    ref_data = None
    if not args.skip_reference:
        if not os.path.isdir(args.dp3_repo):
            print(
                f"WARNING: --dp3-repo {args.dp3_repo} not found; skipping reference.",
                file=sys.stderr,
            )
        else:
            ref_cfg = {
                "project_root": str(_PROJECT_ROOT),
                "dp3_repo": args.dp3_repo,
                "checkpoint": ckpt_path,
                "data": paths,
                "scenarios": scenarios,
                "latency_scenarios": latency_scenarios,
                "actions_dir": ref_actions,
                "seed": args.seed,
                "num_inference_steps": args.num_steps,
            }
            print("\n--- Reference DP3 ---")
            ref_data = run_worker(
                REFERENCE_DP3_WORKER, ref_cfg, "reference DP3", timeout=3600,
            )
            if ref_data is None:
                print("ERROR: reference worker failed.", file=sys.stderr)

    # --- 5. Compare ---
    if kb_data.get("throughput"):
        _print_throughput_comparison(
            kb_data["throughput"], (ref_data or {}).get("throughput"),
        )
    if kb_data.get("latency"):
        _print_latency_comparison(
            kb_data["latency"], (ref_data or {}).get("latency"),
        )
    correctness = None
    if ref_data is not None:
        correctness = _compare_actions(kb_actions, ref_actions)
        if correctness:
            _print_correctness(correctness)

    summary = {
        "gpu": gpu_name,
        "variant": args.variant,
        "checkpoint": ckpt_path,
        "dataset": args.dataset,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "num_requests": args.num_requests,
        "kb_nano": kb_data,
        "reference": ref_data,
        "correctness": correctness,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {os.path.join(args.output_dir, 'results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
