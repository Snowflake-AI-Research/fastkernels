#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: fastkernels Pi0 vs a reference (OpenPI or HF).

Runs standardized robotics VLA workloads and compares:
  - Throughput: inferences/sec for action generation
  - Latency: per-inference latency with percentile stats
  - Correctness: per-sample action MSE between both engines

**Default (authoritative) runs** use real **ALOHA pen_uncap** data via LeRobot.
Use ``--synthetic-only`` only for quick debugging; do not report those numbers
as the main benchmark.

**Like-with-like comparison:** both fastkernels and OpenPI load the **same
fine-tuned Pi0 checkpoint** (``pi0_aloha_pen_uncap``, converted to PyTorch)
with ``action_horizon=50``, ``action_dim=32``. Both sides apply matching
ALOHA input transforms (joint flip, gripper encoding, z-score normalization)
and output transforms (un-normalize, AbsoluteActions, AlohaOutputs truncation
to 14 dims). Pre-generated shared noise ensures identical flow-matching
initialisation. Correctness is measured on the 14-dim robot-space actions.

**OpenPI** reference uses ``create_trained_policy`` + ``Policy.infer`` (see
Physical-Intelligence/openpi). **Default is the PyTorch OpenPI path**
(``--openpi-backend pytorch``): the checkpoint directory must contain
``model.safetensors``. Use ``openpi/examples/convert_jax_model_to_pytorch.py
--config-name pi0_aloha_pen_uncap`` to convert the JAX checkpoint. There is
**no automatic JAX fallback**.

Each engine runs in a subprocess. Install OpenPI via a **clone + uv sync**
(see upstream README) and pass ``--reference-python /path/to/openpi/.venv/bin/python``.

Usage:
    python tests/bench_openpi.py \\
        --reference-python /raid/user_data/olu/openpi/.venv/bin/python
    python tests/bench_openpi.py --skip-reference       # fastkernels only
    python tests/bench_openpi.py --reference hf         # HF Transformers Pi0
    python tests/bench_openpi.py --synthetic-only       # dev/debug only
    python tests/bench_openpi.py --num-requests 50 --num-steps 10
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

from fastkernels.bench.utils.worker import run_worker
from fastkernels.bench.utils.workloads import PI0_CONFIG


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
# Dataset loading helper (inlined for subprocess workers)
# ---------------------------------------------------------------------------

_DATASET_PRELOAD = r'''
import torch, random, os, io, sys

_IMAGE_TOKEN_ID = 257152
_PAD_TOKEN_ID = 0
_BOS_TOKEN_ID = 2
_OPENPI_TOKENIZER_PATH = os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model")


def _load_paligemma_tokenizer():
    """Load PaLiGemma sentencepiece tokenizer, preferring OpenPI's cached .model file."""
    if os.path.isfile(_OPENPI_TOKENIZER_PATH):
        try:
            import sentencepiece as _spm
            _sp = _spm.SentencePieceProcessor(model_file=_OPENPI_TOKENIZER_PATH)
            class _SPWrapper:
                def encode(self, text, add_special_tokens=False):
                    return _sp.encode(text)
            return _SPWrapper()
        except Exception:
            pass
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/paligemma2-3b-mix-224")
    except Exception:
        return None


_TEXT_MAX_LEN = 48  # text-only portion; total seq = total_img_tokens + 1 + _TEXT_MAX_LEN
_NEWLINE_TOKEN_ID = 108  # PaLiGemma sentencepiece "\n" = "start of answer" token


def _tokenize_instruction(text, tokenizer, num_image_tokens, num_cameras, text_max_length=_TEXT_MAX_LEN):
    """Build input_ids matching OpenPI's prefix order: IMG_TOKENs first, then BOS + text + newline + padding.

    Sequence: [IMG×(num_cameras*num_image_tokens), BOS, text_tokens, NL, PAD×...]
    Total length = total_image_tokens + 1 + text_max_length.
    Matches OpenPI's PaligemmaTokenizer: images at 0..N-1, BOS at N, text at N+1..., newline after text.
    """
    total_image_tokens = num_image_tokens * num_cameras
    # Clean text the same way OpenPI's PaligemmaTokenizer does
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    text_enc = tokenizer.encode(cleaned, add_special_tokens=False)[:text_max_length - 1]  # reserve 1 for \n
    text_enc = text_enc + [_NEWLINE_TOKEN_ID]
    seq_len = total_image_tokens + 1 + text_max_length
    ids = [_IMAGE_TOKEN_ID] * total_image_tokens + [_BOS_TOKEN_ID] + text_enc
    if len(ids) < seq_len:
        ids = ids + [_PAD_TOKEN_ID] * (seq_len - len(ids))
    return ids


def load_aloha_dataset(num_samples, num_cameras, image_size=224,
                       max_state_dim=32, max_action_dim=32,
                       device="cuda", dtype=torch.bfloat16, seed=42):
    """Load real ALOHA pen_uncap observations via HuggingFace datasets.

    Each returned element is a dict with the same keys as make_synthetic_batch(),
    plus ``task_text`` and ``gt_action``.
    Falls back to ``None`` (caller should use synthetic data) on failure.
    """
    from torchvision.transforms.functional import resize

    rng = random.Random(seed)
    num_image_tokens = (image_size // 14) ** 2

    tokenizer = _load_paligemma_tokenizer()

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("WARNING: datasets library not installed, falling back to synthetic data.", file=sys.stderr)
        return None

    all_camera_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    dataset_cameras = len(all_camera_keys)

    try:
        ds = hf_load_dataset(
            "physical-intelligence/aloha_pen_uncap_diverse",
            split="train",
            streaming=False,
        )
    except Exception as e:
        print(f"WARNING: Could not load ALOHA dataset: {e}", file=sys.stderr)
        return None

    frame_indices = list(range(len(ds)))
    rng.shuffle(frame_indices)
    frame_indices = frame_indices[:num_samples]

    task_text = "uncap the pen"
    batches = []
    for idx in frame_indices:
        try:
            row = ds[idx]

            cam_tensors = []
            for cam_key in all_camera_keys[:num_cameras]:
                pil_img = row[cam_key]
                img = torch.from_numpy(
                    __import__("numpy").array(pil_img, dtype="float32")
                ).permute(2, 0, 1) / 255.0
                img = resize(img, [image_size, image_size], antialias=True)
                cam_tensors.append(img)

            if num_cameras > dataset_cameras:
                for _ in range(num_cameras - dataset_cameras):
                    cam_tensors.append(cam_tensors[-1].clone())

            images = torch.stack(cam_tensors, dim=0)

            raw_state = row["observation.state"]
            state_vec = torch.tensor(raw_state, dtype=torch.float32)
            if state_vec.numel() < max_state_dim:
                state_vec = torch.nn.functional.pad(state_vec, (0, max_state_dim - state_vec.numel()))
            else:
                state_vec = state_vec[:max_state_dim]

            if tokenizer is not None:
                ids = _tokenize_instruction(task_text, tokenizer, num_image_tokens, num_cameras)
            else:
                total_img_tok = num_image_tokens * num_cameras
                seq_len = total_img_tok + 1 + _TEXT_MAX_LEN
                ids = ([_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID]
                       + [_PAD_TOKEN_ID] * _TEXT_MAX_LEN)[:seq_len]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()

            raw_action = row.get("action")
            if raw_action is not None:
                gt_action = torch.tensor(raw_action, dtype=torch.float32)
            else:
                gt_action = None

            batches.append({
                "state": state_vec.unsqueeze(0).to(device=device, dtype=dtype),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).to(device=device, dtype=dtype),
                "pixel_attention_mask": torch.ones(1, num_cameras, device=device, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": num_cameras,
                "task_text": task_text,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped frame {idx}: {e}", file=sys.stderr)
            continue

    if len(batches) == 0:
        print("WARNING: No samples loaded, falling back to synthetic data.", file=sys.stderr)
        return None

    print(f"Loaded {len(batches)} ALOHA samples ({num_cameras} cameras, {image_size}px).", file=sys.stderr)
    return batches


def make_synthetic_batch(batch_size, num_cameras, image_size=224, max_state_dim=32,
                         max_action_dim=32, tokenizer_max_length=_TEXT_MAX_LEN, device="cuda",
                         dtype=torch.bfloat16, seed=42):
    """Create a synthetic batch for benchmarking when real data is unavailable.

    tokenizer_max_length is the TEXT-ONLY length; total input_ids length =
    1 + num_cameras*(image_size//14)**2 + tokenizer_max_length.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    state = torch.randn(batch_size, max_state_dim, generator=gen).to(device=device, dtype=dtype)
    pixel_values = torch.randn(
        batch_size, num_cameras, 3, image_size, image_size, generator=gen,
    ).to(device=device, dtype=dtype)
    pixel_attention_mask = torch.ones(
        batch_size, num_cameras, device=device, dtype=torch.bool,
    )

    num_image_tokens = (image_size // 14) ** 2
    total_image_tokens = num_image_tokens * num_cameras
    seq_len = total_image_tokens + 1 + tokenizer_max_length  # IMG × N, BOS, text

    input_ids_list = []
    for _ in range(batch_size):
        ids = [_IMAGE_TOKEN_ID] * total_image_tokens + [_BOS_TOKEN_ID] + [1] * tokenizer_max_length
        input_ids_list.append(ids[:seq_len])
    input_ids = torch.tensor(input_ids_list, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return {
        "state": state,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "attention_mask": attention_mask,
    }


def make_pi0_flow_noise(batch_size, chunk_size, max_action_dim, noise_seed, device, dtype):
    """Deterministic flow-matching noise; must match fastkernels and HF workers."""
    g = torch.Generator(device=device)
    g.manual_seed(int(noise_seed))
    return torch.randn(
        batch_size, chunk_size, max_action_dim,
        generator=g, dtype=dtype, device=device,
    )


# ---- ALOHA transforms (replicate OpenPI AlohaInputs/Outputs) ----

import numpy as _np

_JOINT_FLIP_MASK = _np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=_np.float32)

_DELTA_ACTION_MASK = _np.array(
    [True]*6 + [False] + [True]*6 + [False] + [False]*18,
    dtype=bool,
)


def _aloha_normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _aloha_unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    """Aloha linear gripper → PI angular space (input transform)."""
    value = _aloha_unnormalize(value, min_val=0.01844, max_val=0.05800)
    def linear_to_radian(lp, arm_length, horn_radius):
        v = (horn_radius**2 + lp**2 - arm_length**2) / (2 * horn_radius * lp)
        return _np.arcsin(_np.clip(v, -1.0, 1.0))
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _aloha_normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    """PI angular space → Aloha linear gripper (output transform)."""
    value = value + 0.5476
    return _aloha_normalize(value, min_val=-0.6213, max_val=1.4910)


def _load_norm_stats(checkpoint_dir):
    """Load norm_stats.json from an OpenPI / pi0 checkpoint directory.

    Some checkpoints (e.g. pi0_libero) nest norm_stats under
    ``assets/<owner>/<dataset>/norm_stats.json``; others (aloha, droid) keep
    them at ``assets/<name>/norm_stats.json``. Walk recursively to cover both.
    """
    import json as _json, glob as _glob
    candidates = (_glob.glob(os.path.join(checkpoint_dir, "assets", "*", "norm_stats.json"))
                  + _glob.glob(os.path.join(checkpoint_dir, "assets", "**", "norm_stats.json"),
                               recursive=True))
    # de-duplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    if not candidates:
        raise FileNotFoundError(f"No norm_stats.json found under {checkpoint_dir}/assets/")
    with open(candidates[0]) as f:
        raw = _json.load(f)
    ns = raw.get("norm_stats", raw)
    out = {}
    for key in ("state", "actions"):
        if key in ns:
            out[key] = {
                "mean": _np.array(ns[key]["mean"], dtype=_np.float32),
                "std": _np.array(ns[key]["std"], dtype=_np.float32),
            }
    return out


def aloha_preprocess_state(raw_state_14, norm_stats):
    """AlohaInputs._decode_state + z-score normalize + pad to 32."""
    s = raw_state_14.copy().astype(_np.float32)
    s *= _JOINT_FLIP_MASK
    s[[6, 13]] = _gripper_to_angular(s[[6, 13]])
    mean = norm_stats["state"]["mean"][:14]
    std = norm_stats["state"]["std"][:14]
    s = (s - mean) / (std + 1e-6)
    out = _np.zeros(32, dtype=_np.float32)
    out[:14] = s
    return out


def aloha_postprocess_actions(actions_np, normalized_state_32, norm_stats):
    """Unnormalize + AbsoluteActions + AlohaOutputs (matches OpenPI output chain).

    Args:
        actions_np: (horizon, 32) raw model output in normalised space.
        normalized_state_32: (32,) state after aloha_preprocess_state (normalised+padded).
        norm_stats: dict with 'state' and 'actions' mean/std arrays (32-D).
    Returns:
        (horizon, 14) actions in ALOHA robot space.
    """
    a_mean = norm_stats["actions"]["mean"]
    a_std = norm_stats["actions"]["std"]
    s_mean = norm_stats["state"]["mean"]
    s_std = norm_stats["state"]["std"]

    actions = actions_np.copy().astype(_np.float32)
    actions = actions * (a_std + 1e-6) + a_mean

    state_unnorm = normalized_state_32.astype(_np.float32) * (s_std + 1e-6) + s_mean

    mask14 = _DELTA_ACTION_MASK[:14]
    actions[:, :14] += _np.where(mask14, state_unnorm[:14], 0.0)[None, :]

    actions = actions[:, :14]
    actions = actions * _JOINT_FLIP_MASK[None, :]
    actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions
'''


# ---------------------------------------------------------------------------
# DROID + LIBERO dataset loaders and transforms (injected into workers)
# ---------------------------------------------------------------------------

_DROID_LIBERO_PRELOAD = r'''
import numpy as _dl_np  # noqa: F811 — redefine in worker scope is fine


def load_droid_dataset(num_samples, image_size=224, max_state_dim=32, max_action_dim=32,
                       device="cuda", dtype=torch.bfloat16, seed=42):
    """Load DROID observations from lerobot/droid via LeRobotDataset."""
    import numpy as np
    from torchvision.transforms.functional import resize as _tv_resize
    rng = random.Random(seed)
    num_cameras = 2
    num_image_tokens = (image_size // 14) ** 2

    tokenizer = _load_paligemma_tokenizer()

    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("lerobot/droid_100", image_transforms=None, video_backend="pyav")
    except Exception as e:
        print(f"WARNING: Could not load DROID dataset: {e}", file=sys.stderr)
        return None

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    task_text_default = "pick and place"
    batches = []
    for idx in indices:
        try:
            row = ds[idx]

            def _to_chw_float(x):
                t = torch.as_tensor(np.asarray(x, dtype="float32"))
                if t.ndim == 3 and t.shape[-1] == 3:
                    t = t.permute(2, 0, 1) / 255.0
                elif t.max() > 1.0:
                    t = t / 255.0
                if t.shape[-2] != image_size or t.shape[-1] != image_size:
                    t = _tv_resize(t, [image_size, image_size], antialias=True)
                return t

            ext_img = _to_chw_float(row["observation.images.exterior_image_1_left"])
            wrist_img = _to_chw_float(row["observation.images.wrist_image_left"])
            images = torch.stack([ext_img, wrist_img], dim=0)

            raw_state = np.asarray(row["observation.state"], dtype="float32")
            state_vec = torch.from_numpy(raw_state)
            if state_vec.numel() < max_state_dim:
                state_vec = torch.nn.functional.pad(state_vec, (0, max_state_dim - state_vec.numel()))
            else:
                state_vec = state_vec[:max_state_dim]

            task = row.get("task", task_text_default)
            if isinstance(task, (bytes, bytearray)):
                task = task.decode("utf-8", errors="replace")
            if not isinstance(task, str) or not task.strip():
                task = task_text_default

            if tokenizer is not None:
                ids = _tokenize_instruction(task, tokenizer, num_image_tokens, num_cameras)
            else:
                total_img_tok = num_image_tokens * num_cameras
                seq_len = total_img_tok + 1 + _TEXT_MAX_LEN
                ids = ([_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID]
                       + [_PAD_TOKEN_ID] * _TEXT_MAX_LEN)[:seq_len]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()
            raw_action = row.get("action")
            gt_action = torch.tensor(np.asarray(raw_action, dtype="float32")) if raw_action is not None else None

            dev = torch.device(device)
            batches.append({
                "state": state_vec.unsqueeze(0).to(device=dev, dtype=dtype),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).to(device=dev, dtype=dtype),
                "pixel_attention_mask": torch.ones(1, num_cameras, device=dev, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": num_cameras,
                "task_text": task,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped DROID frame {idx}: {e}", file=sys.stderr)

    if not batches:
        print("WARNING: No DROID samples loaded, falling back to synthetic data.", file=sys.stderr)
        return None
    print(f"Loaded {len(batches)} DROID samples ({image_size}px).", file=sys.stderr)
    return batches


def load_libero_dataset(num_samples, image_size=224, max_state_dim=32, max_action_dim=32,
                        device="cuda", dtype=torch.bfloat16, seed=42):
    """Load LIBERO observations from lerobot/libero_90 via LeRobotDataset."""
    import numpy as np
    from torchvision.transforms.functional import resize as _tv_resize
    rng = random.Random(seed)
    num_cameras = 2
    num_image_tokens = (image_size // 14) ** 2

    tokenizer = _load_paligemma_tokenizer()

    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("lerobot/libero_10_image", image_transforms=None)
    except Exception as e:
        print(f"WARNING: Could not load LIBERO dataset: {e}", file=sys.stderr)
        return None

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    task_text_default = "pick up the object"
    batches = []
    for idx in indices:
        try:
            row = ds[idx]

            def _to_chw_float(x):
                t = torch.as_tensor(np.asarray(x, dtype="float32"))
                if t.ndim == 3 and t.shape[-1] == 3:
                    t = t.permute(2, 0, 1) / 255.0
                elif t.max() > 1.0:
                    t = t / 255.0
                if t.shape[-2] != image_size or t.shape[-1] != image_size:
                    t = _tv_resize(t, [image_size, image_size], antialias=True)
                return t

            base_img = _to_chw_float(row["observation.images.image"])
            wrist_img = _to_chw_float(row["observation.images.wrist_image"])
            images = torch.stack([base_img, wrist_img], dim=0)

            raw_state = np.asarray(row["observation.state"], dtype="float32")
            state_vec = torch.from_numpy(raw_state)
            if state_vec.numel() < max_state_dim:
                state_vec = torch.nn.functional.pad(state_vec, (0, max_state_dim - state_vec.numel()))
            else:
                state_vec = state_vec[:max_state_dim]

            task = row.get("task", task_text_default)
            if isinstance(task, (bytes, bytearray)):
                task = task.decode("utf-8", errors="replace")
            if not isinstance(task, str) or not task.strip():
                task = task_text_default

            if tokenizer is not None:
                ids = _tokenize_instruction(task, tokenizer, num_image_tokens, num_cameras)
            else:
                total_img_tok = num_image_tokens * num_cameras
                seq_len = total_img_tok + 1 + _TEXT_MAX_LEN
                ids = ([_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID]
                       + [_PAD_TOKEN_ID] * _TEXT_MAX_LEN)[:seq_len]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()
            raw_action = row.get("action")
            gt_action = torch.tensor(np.asarray(raw_action, dtype="float32")) if raw_action is not None else None

            dev = torch.device(device)
            batches.append({
                "state": state_vec.unsqueeze(0).to(device=dev, dtype=dtype),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).to(device=dev, dtype=dtype),
                "pixel_attention_mask": torch.ones(1, num_cameras, device=dev, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": num_cameras,
                "task_text": task,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped LIBERO frame {idx}: {e}", file=sys.stderr)

    if not batches:
        print("WARNING: No LIBERO samples loaded, falling back to synthetic data.", file=sys.stderr)
        return None
    print(f"Loaded {len(batches)} LIBERO samples ({image_size}px).", file=sys.stderr)
    return batches


def _load_dataset_batches(dataset_name, num_samples, num_cameras_aloha, image_size, seed,
                          device, dtype):
    if dataset_name == "aloha":
        return load_aloha_dataset(num_samples, num_cameras_aloha, image_size=image_size,
                                  seed=seed, device=device, dtype=dtype)
    elif dataset_name == "droid":
        return load_droid_dataset(num_samples, image_size=image_size, seed=seed,
                                  device=device, dtype=dtype)
    elif dataset_name == "libero":
        return load_libero_dataset(num_samples, image_size=image_size, seed=seed,
                                   device=device, dtype=dtype)
    return None


def _synth_cameras(dataset_name):
    return {"aloha": 3, "droid": 2, "libero": 2}.get(dataset_name, 2)


# ---- DROID per-sample transforms ----

def _droid_preprocess_state(raw_state_8, ns):
    import numpy as np
    s = np.asarray(raw_state_8[:8], dtype=np.float32).copy()
    if ns is not None and "state" in ns:
        s = (s - ns["state"]["mean"][:8]) / (ns["state"]["std"][:8] + 1e-6)
    out = np.zeros(32, dtype=np.float32)
    out[:8] = s
    return out


def _droid_postprocess_actions(act, ns):
    import numpy as np
    act = np.asarray(act, dtype=np.float32).copy()
    if ns is not None and "actions" in ns:
        act = act * (ns["actions"]["std"] + 1e-6) + ns["actions"]["mean"]
    return act[:, :8]


# ---- LIBERO per-sample transforms ----

def _libero_preprocess_state(raw_state_8, ns):
    import numpy as np
    s = np.asarray(raw_state_8[:8], dtype=np.float32).copy()
    if ns is not None and "state" in ns:
        s = (s - ns["state"]["mean"][:8]) / (ns["state"]["std"][:8] + 1e-6)
    out = np.zeros(32, dtype=np.float32)
    out[:8] = s
    return out


def _libero_postprocess_actions(act, ns, raw_state_8=None):
    """Unnormalize LIBERO model output, then convert delta→absolute for joints.

    OpenPI's pi0_libero training config uses ``extra_delta_transform=True`` which
    applies ``DeltaActions([T]*6+[F])`` on input and the matching ``AbsoluteActions``
    mask on output. At inference, after unnormalize, the model's first 6 action dims
    are deltas relative to ``state`` (joint positions). The 7th dim (gripper) is
    absolute. See openpi/src/openpi/training/config.py:LeRobotLiberoDataConfig and
    openpi/src/openpi/transforms.py:AbsoluteActions.
    """
    import numpy as np
    act = np.asarray(act, dtype=np.float32).copy()
    if ns is not None and "actions" in ns:
        act = act * (ns["actions"]["std"] + 1e-6) + ns["actions"]["mean"]
    act = act[:, :7]
    if raw_state_8 is not None:
        state = np.asarray(raw_state_8, dtype=np.float32).reshape(-1)
        # AbsoluteActions: add state[:6] to first 6 action dims; leave gripper (dim 6) untouched.
        act[:, :6] = act[:, :6] + state[:6]
    return act


# ---- OpenPI observation converters ----

def kb_batch_to_openpi_droid(batch):
    """Map fastkernels batch -> OpenPI DroidInputs observation dict."""
    pv = batch["pixel_values"][0]  # (2, 3, H, W) float [0,1]
    st = batch["state"][0].detach().cpu().float().numpy()

    def to_hwc_uint8(t):
        arr = t.detach().float().cpu().clamp(0, 1).numpy() * 255.0
        arr = arr.astype(np.uint8)
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr

    prompt = batch.get("task_text", "pick and place")
    if not isinstance(prompt, str):
        prompt = "pick and place"

    return {
        "observation/exterior_image_1_left": to_hwc_uint8(pv[0]),
        "observation/wrist_image_left": to_hwc_uint8(pv[1]),
        "observation/joint_position": st[:7].astype(np.float64),
        "observation/gripper_position": st[7:8].astype(np.float64),
        "prompt": prompt,
    }


def kb_batch_to_openpi_libero(batch):
    """Map fastkernels batch -> OpenPI LiberoInputs observation dict."""
    pv = batch["pixel_values"][0]  # (2, 3, H, W) float [0,1]
    st = batch["state"][0].detach().cpu().float().numpy()

    def to_hwc_uint8(t):
        arr = t.detach().float().cpu().clamp(0, 1).numpy() * 255.0
        arr = arr.astype(np.uint8)
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr

    prompt = batch.get("task_text", "pick up the object")
    if not isinstance(prompt, str):
        prompt = "pick up the object"

    return {
        "observation/image": to_hwc_uint8(pv[0]),
        "observation/wrist_image": to_hwc_uint8(pv[1]),
        "observation/state": st[:8].astype(np.float64),
        "prompt": prompt,
    }
'''


# ---------------------------------------------------------------------------
# Hugging Face Transformers Pi0 reference (subprocess)
# ---------------------------------------------------------------------------

HF_PI0_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

''' + _DATASET_PRELOAD + r'''

def _get_batch(dataset_batches, idx, scenario_cameras, image_size, seed, strict_data):
    if dataset_batches is not None and idx < len(dataset_batches):
        return dataset_batches[idx]
    if strict_data:
        print("ERROR: Real ALOHA data required but batch %s is missing." % idx, file=sys.stderr)
        sys.exit(1)
    return make_synthetic_batch(1, scenario_cameras, image_size=image_size, seed=seed + idx)


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from transformers import PI0ForConditionalGeneration

    model_name = cfg["model"]
    seed = cfg["seed"]
    num_steps = cfg["num_steps"]

    model = PI0ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    image_size = cfg.get("image_size", 224)
    chunk_size = cfg.get("chunk_size", 50)
    max_action_dim = cfg.get("max_action_dim", 32)
    use_real_data = cfg.get("use_real_data", True)
    strict_data = cfg.get("strict_data", False)

    # Warmup with 1-camera synthetic batch
    warmup_batch = make_synthetic_batch(1, 1, image_size=image_size, seed=seed)
    for _ in range(2):
        torch.cuda.synchronize()
        with torch.inference_mode():
            model.sample_actions(
                state=warmup_batch["state"], input_ids=warmup_batch["input_ids"],
                pixel_values=warmup_batch["pixel_values"],
                pixel_attention_mask=warmup_batch["pixel_attention_mask"],
                attention_mask=warmup_batch["attention_mask"],
                num_steps=2,
            )
        torch.cuda.synchronize()

    throughput_results = []
    sample_offset = 0
    for scenario in cfg.get("scenarios", []):
        num_cameras = scenario["num_cameras"]
        num_requests = scenario["num_requests"]
        total_elapsed = 0.0
        all_actions = []

        dataset_batches = None
        if use_real_data:
            dataset_batches = load_aloha_dataset(
                num_samples=num_requests, num_cameras=num_cameras,
                image_size=image_size, seed=seed,
            )
            if strict_data and (dataset_batches is None or len(dataset_batches) < num_requests):
                print("ERROR: strict data mode requires at least %d samples; got %s." % (
                    num_requests, len(dataset_batches) if dataset_batches else 0), file=sys.stderr)
                sys.exit(1)

        pbar = tqdm(range(num_requests), desc=f"hf-ref {scenario['name']}", file=sys.stderr)
        for req_idx in pbar:
            batch = _get_batch(dataset_batches, req_idx,
                               num_cameras, image_size, seed + sample_offset, strict_data)
            dev = batch["state"].device
            dt = batch["state"].dtype
            noise_seed = seed + 10000 * (sample_offset + req_idx) + 424242
            noise = make_pi0_flow_noise(1, chunk_size, max_action_dim, noise_seed, dev, dt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                actions = model.sample_actions(
                    state=batch["state"], input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    attention_mask=batch["attention_mask"],
                    num_steps=num_steps,
                    noise=noise,
                )
            torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0
            all_actions.append(actions.cpu())
            pbar.set_postfix(ips=f"{(req_idx + 1) / total_elapsed:.2f}")

        sample_offset += num_requests
        throughput_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_requests": num_requests,
            "inferences_per_second": num_requests / total_elapsed,
            "data_source": "aloha" if dataset_batches else "synthetic",
        })

        actions_dir = cfg.get("actions_dir")
        if actions_dir:
            os.makedirs(actions_dir, exist_ok=True)
            torch.save(
                torch.cat(all_actions, dim=0),
                os.path.join(actions_dir, f"{scenario['name']}_actions.pt"),
            )

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        nc = ls["num_cameras"]
        nlat = ls.get("num_warmup", 3) + ls.get("num_iters", 10)
        lat_ds = None
        if use_real_data:
            lat_ds = load_aloha_dataset(
                num_samples=max(nlat, 4), num_cameras=nc,
                image_size=image_size, seed=seed + 777,
            )
            if strict_data and (lat_ds is None or len(lat_ds) < 1):
                print("ERROR: strict data mode failed to load latency dataset.", file=sys.stderr)
                sys.exit(1)
        def _lat_batch(i):
            if lat_ds is not None and len(lat_ds) > 0:
                return _get_batch(lat_ds, i % len(lat_ds), nc, image_size, seed + 888 + i, strict_data)
            return make_synthetic_batch(1, nc, image_size=image_size, seed=seed + 888 + i)

        for wi in tqdm(range(ls.get("num_warmup", 3)), desc=f"hf-ref warmup {ls['name']}", file=sys.stderr):
            batch = _lat_batch(wi)
            torch.cuda.synchronize()
            with torch.inference_mode():
                model.sample_actions(
                    state=batch["state"], input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    attention_mask=batch["attention_mask"],
                    num_steps=num_steps,
                )
            torch.cuda.synchronize()

        latencies = []
        for ii in tqdm(range(ls.get("num_iters", 10)), desc=f"hf-ref latency {ls['name']}", file=sys.stderr):
            batch = _lat_batch(ii + 1000)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.sample_actions(
                    state=batch["state"], input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    attention_mask=batch["attention_mask"],
                    num_steps=num_steps,
                )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_cameras": ls["num_cameras"],
            "batch_size": ls.get("batch_size", 1),
            "num_iters": ls.get("num_iters", 10),
            "latencies": latencies,
        })

    del model
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": throughput_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# OpenPI package reference: Policy.infer (PyTorch or JAX checkpoint)
# ---------------------------------------------------------------------------

OPENPI_POLICY_WORKER = r'''
import json, os, sys, time
import numpy as np
import torch
from tqdm import tqdm

''' + _DATASET_PRELOAD + _DROID_LIBERO_PRELOAD + r'''

def _chw_float_to_chw_uint8(t):
    x = t.detach().float().cpu().clamp(0, 1) * 255.0
    return x.byte().contiguous().numpy()


def kb_batch_to_openpi_aloha(batch, num_cameras):
    """Map fastkernels batch dict -> OpenPI AlohaInputs keys (numpy CHW uint8)."""
    pv = batch["pixel_values"][0]
    st = batch["state"][0].detach().cpu().float().numpy()
    state_14 = np.zeros(14, dtype=np.float32)
    n = min(14, st.shape[0])
    state_14[:n] = st[:n]
    prompt = batch.get("task_text", "uncap the pen")
    if not isinstance(prompt, str):
        prompt = "uncap the pen"
    images = {"cam_high": _chw_float_to_chw_uint8(pv[0])}
    if num_cameras >= 2:
        images["cam_left_wrist"] = _chw_float_to_chw_uint8(pv[1])
    if num_cameras >= 3:
        images["cam_right_wrist"] = _chw_float_to_chw_uint8(pv[2])
    return {
        "state": state_14,
        "images": images,
        "prompt": prompt,
    }


def _get_batch(dataset_batches, idx, scenario_cameras, image_size, seed, strict_data):
    if dataset_batches is not None and idx < len(dataset_batches):
        return dataset_batches[idx]
    if strict_data:
        print("ERROR: Real ALOHA data required but batch %s is missing." % idx, file=sys.stderr)
        sys.exit(1)
    return make_synthetic_batch(1, scenario_cameras, image_size=image_size, seed=seed + idx)


def _kb_batch_cuda_bf16(batch):
    """Match fastkernels Pi0 dtype/device for cached CPU batches from the parent process."""
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k in ("state", "pixel_values"):
                out[k] = v.to(device=dev, dtype=dtype)
            elif k in ("input_ids", "attention_mask", "pixel_attention_mask"):
                out[k] = v.to(device=dev)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _stub_lerobot_dataset_module_if_missing():
    """OpenPI imports training.data_loader -> lerobot.common.datasets.lerobot_dataset.

    Inference-only policy loading does not need LeRobot at runtime. Some reference
    venvs lack this module path entirely. Stub missing parents + leaf so
    ``policy_config`` loads (training data APIs are not used for ``Policy.infer``).
    """
    import importlib
    import types

    key = "lerobot.common.datasets.lerobot_dataset"
    if key in sys.modules:
        return
    try:
        importlib.import_module(key)
        return
    except ImportError:
        pass

    chain = ("lerobot", "lerobot.common", "lerobot.common.datasets", key)
    for full in chain:
        if full in sys.modules:
            continue
        try:
            importlib.import_module(full)
        except ImportError:
            m = types.ModuleType(full)
            if full == "lerobot":
                m.__path__ = []
            parent = full.rsplit(".", 1)[0] if "." in full else ""
            leaf = full.rsplit(".", 1)[-1]
            if parent and parent in sys.modules:
                setattr(sys.modules[parent], leaf, m)
            sys.modules[full] = m


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    _stub_lerobot_dataset_module_if_missing()

    from openpi.shared import download as openpi_download
    from openpi.training import config as openpi_train_config
    from openpi.policies import policy_config

    ckpt = openpi_download.maybe_download(cfg["openpi_checkpoint"])
    safetensors_path = os.path.join(ckpt, "model.safetensors")
    params_path = os.path.join(ckpt, "params")
    backend = cfg.get("openpi_backend", "pytorch")

    if backend == "pytorch":
        if not os.path.isfile(safetensors_path):
            print("ERROR: PyTorch OpenPI requires model.safetensors in the checkpoint directory:", ckpt, file=sys.stderr)
            print("  This benchmark does not fall back to JAX — fix the checkpoint or export PyTorch weights.", file=sys.stderr)
            print("  Set --openpi-checkpoint to a tree that contains model.safetensors (and matching assets/).", file=sys.stderr)
            print("  JAX is opt-in only: --openpi-backend jax (not used when PyTorch is unavailable).", file=sys.stderr)
            sys.exit(1)
    else:
        if os.path.isfile(safetensors_path):
            print("ERROR: --openpi-backend jax but checkpoint contains model.safetensors.", file=sys.stderr)
            print("  OpenPI loads PyTorch whenever safetensors is present. Use a params-only directory", file=sys.stderr)
            print("  or --openpi-backend pytorch.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(params_path):
            print("ERROR: JAX reference requested but no params/ in checkpoint:", ckpt, file=sys.stderr)
            sys.exit(1)

    _DATASET_TO_TRAIN_CONFIG = {
        "aloha": "pi0_aloha_pen_uncap",
        "droid": "pi0_droid",
        "libero": "pi0_libero",
    }
    dataset_name_pre = cfg.get("dataset_name", "aloha")
    train_config_name = cfg.get("openpi_train_config") or _DATASET_TO_TRAIN_CONFIG.get(dataset_name_pre, "pi0_aloha_pen_uncap")
    train_cfg = openpi_train_config.get_config(train_config_name)
    # OpenPI's Pi0Config sets pytorch_compile_mode="max-autotune" by default, which
    # fails on DROID (action_horizon=10) with a dynamo shape-broadcast error. Disable
    # compile unless the caller opts back in.
    import dataclasses as _dc
    if cfg.get("openpi_compile_mode") is None:
        try:
            new_model = _dc.replace(train_cfg.model, pytorch_compile_mode=None)
            train_cfg = _dc.replace(train_cfg, model=new_model)
        except Exception as _e:
            print(f"WARNING: could not disable pytorch_compile_mode: {_e}", file=sys.stderr)
    policy = policy_config.create_trained_policy(
        train_cfg,
        ckpt,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": cfg["num_steps"]},
    )

    seed = cfg["seed"]
    num_steps = cfg["num_steps"]
    image_size = cfg.get("image_size", 224)
    use_real_data = cfg.get("use_real_data", True)
    strict_data = cfg.get("strict_data", False)
    dataset_name = cfg.get("dataset_name", "aloha")

    loaded_pytorch = getattr(policy, "_is_pytorch_model", False)
    meta = {
        "openpi_checkpoint": str(ckpt),
        "requested_openpi_backend": backend,
        "loaded_pytorch": loaded_pytorch,
        "dataset_name": dataset_name,
    }
    batch_cache = cfg.get(f"{dataset_name}_batch_cache") or cfg.get("aloha_batch_cache")

    def _to_openpi_obs(batch, nc):
        if dataset_name == "droid":
            return kb_batch_to_openpi_droid(batch)
        elif dataset_name == "libero":
            return kb_batch_to_openpi_libero(batch)
        return kb_batch_to_openpi_aloha(batch, nc)

    warmup_nc = _synth_cameras(dataset_name)
    warmup_batch = make_synthetic_batch(1, warmup_nc, image_size=image_size, seed=seed)
    obs_w = _to_openpi_obs(warmup_batch, warmup_nc)
    for _ in range(2):
        torch.cuda.synchronize()
        policy.infer(obs_w)
        torch.cuda.synchronize()

    throughput_results = []
    sample_offset = 0
    for scenario in cfg.get("scenarios", []):
        num_cameras = scenario["num_cameras"]
        num_requests = scenario["num_requests"]
        total_elapsed = 0.0
        all_actions = []

        dataset_batches = None
        if use_real_data:
            if batch_cache and scenario["name"] in batch_cache.get("throughput", {}):
                raw = torch.load(batch_cache["throughput"][scenario["name"]], map_location="cpu", weights_only=False)
                dataset_batches = [_kb_batch_cuda_bf16(b) for b in raw]
            else:
                dataset_batches = _load_dataset_batches(
                    dataset_name, num_requests, num_cameras, image_size, seed,
                    device="cuda", dtype=torch.bfloat16,
                )
            if strict_data and (dataset_batches is None or len(dataset_batches) < num_requests):
                print("ERROR: strict data mode requires %d samples; got %s." % (
                    num_requests, len(dataset_batches) if dataset_batches else 0), file=sys.stderr)
                sys.exit(1)

        noise_dir = cfg.get("noise_dir")
        scenario_noise = None
        if noise_dir:
            nf = os.path.join(noise_dir, f"{scenario['name']}_noise.pt")
            if os.path.isfile(nf):
                scenario_noise = torch.load(nf, map_location="cpu", weights_only=True)

        pbar = tqdm(range(num_requests), desc=f"openpi {scenario['name']}", file=sys.stderr)
        for req_idx in pbar:
            synth_nc = _synth_cameras(dataset_name)
            batch = _get_batch(dataset_batches, req_idx, synth_nc, image_size,
                               seed + sample_offset, strict_data)
            obs = _to_openpi_obs(batch, num_cameras)
            noise_np = None
            if scenario_noise is not None and req_idx < scenario_noise.shape[0]:
                noise_np = scenario_noise[req_idx].numpy()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = policy.infer(obs, noise=noise_np)
            torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0
            act = out.get("actions")
            if act is None:
                print("ERROR: policy.infer returned no actions", file=sys.stderr)
                sys.exit(1)
            all_actions.append(torch.from_numpy(np.asarray(act)).float().cpu())
            pbar.set_postfix(ips=f"{(req_idx + 1) / total_elapsed:.2f}")

        sample_offset += num_requests
        throughput_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_requests": num_requests,
            "inferences_per_second": num_requests / total_elapsed,
            "data_source": dataset_name if dataset_batches else "synthetic",
        })

        actions_dir = cfg.get("actions_dir")
        if actions_dir:
            os.makedirs(actions_dir, exist_ok=True)
            torch.save(
                torch.stack(all_actions, dim=0),
                os.path.join(actions_dir, f"{scenario['name']}_actions.pt"),
            )

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        nc = ls["num_cameras"]
        nlat = ls.get("num_warmup", 3) + ls.get("num_iters", 10)
        lat_ds = None
        if use_real_data:
            if batch_cache and ls["name"] in batch_cache.get("latency", {}):
                raw = torch.load(batch_cache["latency"][ls["name"]], map_location="cpu", weights_only=False)
                lat_ds = [_kb_batch_cuda_bf16(b) for b in raw]
            else:
                lat_ds = _load_dataset_batches(
                    dataset_name, max(nlat, 4), nc, image_size, seed + 777,
                    device="cuda", dtype=torch.bfloat16,
                )
            if strict_data and (lat_ds is None or len(lat_ds) < 1):
                print("ERROR: strict data mode failed to load latency dataset.", file=sys.stderr)
                sys.exit(1)

        synth_nc = _synth_cameras(dataset_name)

        def _lat_batch(i):
            if lat_ds is not None and len(lat_ds) > 0:
                return _get_batch(lat_ds, i % len(lat_ds), synth_nc, image_size, seed + 888 + i, strict_data)
            return make_synthetic_batch(1, synth_nc, image_size=image_size, seed=seed + 888 + i)

        for wi in tqdm(range(ls.get("num_warmup", 3)), desc=f"openpi warmup {ls['name']}", file=sys.stderr):
            batch = _lat_batch(wi)
            obs = _to_openpi_obs(batch, nc)
            torch.cuda.synchronize()
            policy.infer(obs)
            torch.cuda.synchronize()

        latencies = []
        for ii in tqdm(range(ls.get("num_iters", 10)), desc=f"openpi latency {ls['name']}", file=sys.stderr):
            batch = _lat_batch(ii + 1000)
            obs = _to_openpi_obs(batch, nc)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.infer(obs)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_cameras": ls["num_cameras"],
            "batch_size": ls.get("batch_size", 1),
            "num_iters": ls.get("num_iters", 10),
            "latencies": latencies,
        })

    del policy
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "throughput": throughput_results,
            "latency": latency_results,
            "meta": meta,
        }, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# fastkernels subprocess worker
# ---------------------------------------------------------------------------

FASTKERNELS_PI0_WORKER = r'''
import json, os, sys, time, torch, glob
import numpy as np
from tqdm import tqdm

''' + _DATASET_PRELOAD + _DROID_LIBERO_PRELOAD + r'''

# ---- Inline ALOHA transforms (match OpenPI AlohaInputs/Outputs pipeline) ----

_JOINT_FLIP = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
_DELTA_MASK = np.array([True]*6 + [False] + [True]*6 + [False] + [False]*18, dtype=bool)


def _aloha_norm(x, lo, hi):
    return (x - lo) / (hi - lo)


def _aloha_unnorm(x, lo, hi):
    return x * (hi - lo) + lo


def _grip_to_ang(v):
    v = _aloha_unnorm(v, 0.01844, 0.05800)
    r = (0.022**2 + v**2 - 0.036**2) / (2 * 0.022 * v)
    v = np.arcsin(np.clip(r, -1.0, 1.0))
    return _aloha_norm(v, 0.5476, 1.6296)


def _grip_from_ang(v):
    return _aloha_norm(v + 0.5476, -0.6213, 1.4910)


def _load_ns(ckpt_dir):
    hits = (glob.glob(os.path.join(ckpt_dir, "assets", "*", "norm_stats.json"))
            + glob.glob(os.path.join(ckpt_dir, "assets", "**", "norm_stats.json"),
                        recursive=True))
    seen = set()
    hits = [h for h in hits if not (h in seen or seen.add(h))]
    if not hits:
        return None
    with open(hits[0]) as f:
        raw = json.load(f)
    ns = raw.get("norm_stats", raw)
    out = {}
    for k in ("state", "actions"):
        if k in ns:
            out[k] = {"mean": np.array(ns[k]["mean"], dtype=np.float32),
                       "std": np.array(ns[k]["std"], dtype=np.float32)}
    return out


def _preprocess_state(raw_14, ns):
    s = raw_14.copy().astype(np.float32)
    s *= _JOINT_FLIP
    s[[6, 13]] = _grip_to_ang(s[[6, 13]])
    m = ns["state"]["mean"][:14]
    sd = ns["state"]["std"][:14]
    s = (s - m) / (sd + 1e-6)
    out = np.zeros(32, dtype=np.float32)
    out[:14] = s
    return out


def _postprocess_actions(act, norm_state_32, ns):
    a_m = ns["actions"]["mean"]
    a_s = ns["actions"]["std"]
    s_m = ns["state"]["mean"]
    s_s = ns["state"]["std"]
    act = act.copy().astype(np.float32)
    act = act * (a_s + 1e-6) + a_m
    su = norm_state_32.astype(np.float32) * (s_s + 1e-6) + s_m
    m14 = _DELTA_MASK[:14]
    act[:, :14] += np.where(m14, su[:14], 0.0)[None, :]
    act = act[:, :14]
    act = act * _JOINT_FLIP[None, :]
    act[:, [6, 13]] = _grip_from_ang(act[:, [6, 13]])
    return act


def _get_batch(dataset_batches, idx, scenario_cameras, image_size, seed, strict_data):
    if dataset_batches is not None and idx < len(dataset_batches):
        return dataset_batches[idx]
    if strict_data:
        print("ERROR: Real ALOHA data required but batch %s is missing." % idx, file=sys.stderr)
        sys.exit(1)
    return make_synthetic_batch(1, scenario_cameras, image_size=image_size, seed=seed + idx)


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    eng_mod = __import__(f"{pkg}.infra.pi0_engine", fromlist=["Pi0Engine"])
    pi0_mod = __import__(f"{pkg}.tasks.baseline.L4.pi0", fromlist=["Pi0SamplingParams"])
    Pi0Engine = eng_mod.Pi0Engine
    Pi0SamplingParams = pi0_mod.Pi0SamplingParams

    seed = cfg["seed"]
    num_steps = cfg["num_steps"]
    image_size = cfg.get("image_size", 224)
    chunk_size = cfg.get("chunk_size", 50)
    max_action_dim = cfg.get("max_action_dim", 32)
    use_real_data = cfg.get("use_real_data", True)
    strict_data = cfg.get("strict_data", False)
    dataset_name = cfg.get("dataset_name", "aloha")

    norm_stats = _load_ns(cfg["model"])
    if norm_stats is None:
        print(f"WARNING: No norm_stats.json found under {cfg['model']}; skipping normalization.", file=sys.stderr)

    engine = Pi0Engine(
        model_name=cfg["model"],
        seed=seed,
        enforce_eager=cfg.get("enforce_eager", False),
    )

    warmup_nc = _synth_cameras(dataset_name)
    warmup_batch = make_synthetic_batch(1, warmup_nc, image_size=image_size, seed=seed)
    params = Pi0SamplingParams(num_inference_steps=2)
    for _ in range(2):
        torch.cuda.synchronize()
        engine.generate(
            state=warmup_batch["state"], input_ids=warmup_batch["input_ids"],
            pixel_values=warmup_batch["pixel_values"],
            pixel_attention_mask=warmup_batch["pixel_attention_mask"],
            attention_mask=warmup_batch["attention_mask"],
            params=params,
        )
        torch.cuda.synchronize()

    throughput_results = []
    params = Pi0SamplingParams(num_inference_steps=num_steps, seed=seed)
    sample_offset = 0

    for scenario in cfg.get("scenarios", []):
        num_cameras = scenario["num_cameras"]
        num_requests = scenario["num_requests"]
        total_elapsed = 0.0
        all_actions = []

        dataset_batches = None
        if use_real_data:
            dataset_batches = _load_dataset_batches(
                dataset_name, num_requests, num_cameras, image_size, seed,
                device="cuda", dtype=torch.bfloat16,
            )
            if strict_data and (dataset_batches is None or len(dataset_batches) < num_requests):
                print("ERROR: strict data mode requires %d samples; got %s." % (
                    num_requests, len(dataset_batches) if dataset_batches else 0), file=sys.stderr)
                sys.exit(1)

        noise_dir = cfg.get("noise_dir")
        scenario_noise = None
        if noise_dir:
            nf = os.path.join(noise_dir, f"{scenario['name']}_noise.pt")
            if os.path.isfile(nf):
                scenario_noise = torch.load(nf, map_location="cpu", weights_only=True)

        pbar = tqdm(range(num_requests), desc=f"fastkernels {scenario['name']}", file=sys.stderr)
        for req_idx in pbar:
            synth_nc = _synth_cameras(dataset_name)
            batch = _get_batch(dataset_batches, req_idx, synth_nc, image_size,
                               seed + sample_offset, strict_data)
            dev = batch["state"].device
            dt = batch["state"].dtype

            raw_state_np = batch["state"][0].detach().cpu().float().numpy()
            norm_state = None
            if norm_stats is not None:
                if dataset_name == "aloha":
                    norm_state = _preprocess_state(raw_state_np[:14], norm_stats)
                elif dataset_name == "droid":
                    norm_state = _droid_preprocess_state(raw_state_np, norm_stats)
                elif dataset_name == "libero":
                    norm_state = _libero_preprocess_state(raw_state_np, norm_stats)
            if norm_state is not None:
                batch["state"] = torch.from_numpy(norm_state).unsqueeze(0).to(device=dev, dtype=dt)

            batch["pixel_values"] = batch["pixel_values"] * 2.0 - 1.0

            if scenario_noise is not None and req_idx < scenario_noise.shape[0]:
                noise = scenario_noise[req_idx:req_idx+1].to(device=dev, dtype=dt)
            else:
                noise_seed = seed + 10000 * (sample_offset + req_idx) + 424242
                noise = make_pi0_flow_noise(1, chunk_size, max_action_dim, noise_seed, dev, dt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = engine.generate(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                params=params,
                noise=noise,
            )
            torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0

            raw_act = output.actions[0].detach().cpu().float().numpy()
            if norm_stats is not None and norm_state is not None:
                if dataset_name == "aloha":
                    post_act = _postprocess_actions(raw_act, norm_state, norm_stats)
                elif dataset_name == "droid":
                    post_act = _droid_postprocess_actions(raw_act, norm_stats)
                elif dataset_name == "libero":
                    post_act = _libero_postprocess_actions(raw_act, norm_stats, raw_state_np[:8])
                else:
                    post_act = raw_act
                all_actions.append(torch.from_numpy(post_act).unsqueeze(0))
            else:
                all_actions.append(output.actions.cpu())
            pbar.set_postfix(ips=f"{(req_idx + 1) / total_elapsed:.2f}")

        sample_offset += num_requests
        throughput_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_requests": num_requests,
            "inferences_per_second": num_requests / total_elapsed,
            "data_source": dataset_name if dataset_batches else "synthetic",
        })

        actions_dir = cfg.get("actions_dir")
        if actions_dir:
            os.makedirs(actions_dir, exist_ok=True)
            torch.save(
                torch.cat(all_actions, dim=0),
                os.path.join(actions_dir, f"{scenario['name']}_actions.pt"),
            )

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        nc = ls["num_cameras"]
        nlat = ls.get("num_warmup", 3) + ls.get("num_iters", 10)
        lat_ds = None
        if use_real_data:
            lat_ds = _load_dataset_batches(
                dataset_name, max(nlat, 4), nc, image_size, seed + 777,
                device="cuda", dtype=torch.bfloat16,
            )
            if strict_data and (lat_ds is None or len(lat_ds) < 1):
                print("ERROR: strict data mode failed to load latency dataset.", file=sys.stderr)
                sys.exit(1)
        latency_params = Pi0SamplingParams(num_inference_steps=num_steps, seed=seed)

        def _lat_batch(i):
            synth_nc = _synth_cameras(dataset_name)
            if lat_ds is not None and len(lat_ds) > 0:
                return _get_batch(lat_ds, i % len(lat_ds), synth_nc, image_size, seed + 888 + i, strict_data)
            return make_synthetic_batch(1, synth_nc, image_size=image_size, seed=seed + 888 + i)

        for wi in tqdm(range(ls.get("num_warmup", 3)), desc=f"fastkernels warmup {ls['name']}", file=sys.stderr):
            batch = _lat_batch(wi)
            torch.cuda.synchronize()
            engine.generate(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                params=latency_params,
            )
            torch.cuda.synchronize()

        latencies = []
        for ii in tqdm(range(ls.get("num_iters", 10)), desc=f"fastkernels latency {ls['name']}", file=sys.stderr):
            batch = _lat_batch(ii + 1000)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                params=latency_params,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_cameras": ls["num_cameras"],
            "batch_size": ls.get("batch_size", 1),
            "num_iters": ls.get("num_iters", 10),
            "latencies": latencies,
        })

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": throughput_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Parent-side dataset materialization (runs in fastkernels interpreter)
# ---------------------------------------------------------------------------

def _batches_to_cpu_float32(batches: list[dict]) -> list[dict]:
    """Detach tensors to CPU float32 for cross-process serialisation."""
    out = []
    for b in batches:
        nb = {}
        for k, v in b.items():
            if isinstance(v, __import__("torch").Tensor):
                nb[k] = v.detach().cpu().float()
            else:
                nb[k] = v
        out.append(nb)
    return out


def _materialize_cache(
    load_fn,
    cache_dir: str,
    scenarios: list[dict],
    latency_scenarios: list[dict],
    *,
    seed: int,
    image_size: int,
    cache_key: str,
) -> "dict | None":
    """Generic materialization helper used by all three datasets."""
    import os, shutil, torch as _torch
    os.makedirs(cache_dir, exist_ok=True)
    cache: dict = {"throughput": {}, "latency": {}}

    for scenario in scenarios:
        name = scenario["name"]
        nc = scenario.get("num_cameras", 2)
        nreq = scenario["num_requests"]
        batches = load_fn(num_samples=nreq, num_cameras=nc, image_size=image_size,
                          seed=seed, device="cpu", dtype=_torch.float32)
        if not batches or len(batches) < nreq:
            print(f"ERROR: Could not materialize {cache_key} for scenario {name} (need {nreq}).",
                  flush=True)
            return None
        path = os.path.join(cache_dir, f"throughput_{name}.pt")
        _torch.save(_batches_to_cpu_float32(batches[:nreq]), path)
        cache["throughput"][name] = path

    for ls in latency_scenarios:
        name = ls["name"]
        nc = ls.get("num_cameras", 2)
        nneed = max(ls.get("num_warmup", 3) + ls.get("num_iters", 10), 4)
        batches = load_fn(num_samples=nneed, num_cameras=nc, image_size=image_size,
                          seed=seed + 777, device="cpu", dtype=_torch.float32)
        if not batches:
            print(f"ERROR: Could not materialize {cache_key} latency scenario {name}.", flush=True)
            return None
        path = os.path.join(cache_dir, f"latency_{name}.pt")
        _torch.save(_batches_to_cpu_float32(batches[:nneed]), path)
        cache["latency"][name] = path

    return cache


def _load_paligemma_tokenizer_parent():
    """Load PaLiGemma sentencepiece tokenizer for parent-process use."""
    import os
    _path = os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model")
    if os.path.isfile(_path):
        try:
            import sentencepiece as _spm
            _sp = _spm.SentencePieceProcessor(model_file=_path)
            class _SPWrapper:
                def encode(self, text, add_special_tokens=False):
                    return _sp.encode(text)
            return _SPWrapper()
        except Exception:
            pass
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/paligemma2-3b-mix-224")
    except Exception:
        return None


def _load_aloha_for_cache(num_samples, num_cameras, image_size, seed, device, dtype):
    import sys, torch, random
    from torchvision.transforms.functional import resize

    _IMAGE_TOKEN_ID = 257152
    _PAD_TOKEN_ID = 0
    _BOS_TOKEN_ID = 2
    _NEWLINE_TOKEN_ID = 108
    _TEXT_MAX_LEN = 48

    rng = random.Random(seed)
    num_image_tokens = (image_size // 14) ** 2
    tokenizer = _load_paligemma_tokenizer_parent()
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        return None
    all_camera_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    try:
        ds = hf_load_dataset("physical-intelligence/aloha_pen_uncap_diverse", split="train", streaming=False)
    except Exception as e:
        print(f"WARNING: Could not load ALOHA dataset: {e}", file=sys.stderr)
        return None
    import numpy as np
    frame_indices = list(range(len(ds)))
    rng.shuffle(frame_indices)
    frame_indices = frame_indices[:num_samples]
    task_text = "uncap the pen"
    batches = []
    for idx in frame_indices:
        try:
            row = ds[idx]
            cam_tensors = []
            for cam_key in all_camera_keys[:num_cameras]:
                pil_img = row[cam_key]
                img = torch.from_numpy(np.array(pil_img, dtype="float32")).permute(2, 0, 1) / 255.0
                img = resize(img, [image_size, image_size], antialias=True)
                cam_tensors.append(img)
            images = torch.stack(cam_tensors, dim=0)
            raw_state = row["observation.state"]
            state_vec = torch.tensor(raw_state, dtype=torch.float32)
            if state_vec.numel() < 32:
                state_vec = torch.nn.functional.pad(state_vec, (0, 32 - state_vec.numel()))
            else:
                state_vec = state_vec[:32]
            total_img_tok = num_image_tokens * num_cameras
            seq_len = total_img_tok + 1 + _TEXT_MAX_LEN
            if tokenizer is not None:
                _cleaned = task_text.strip().replace("_", " ").replace("\n", " ")
                text_enc = tokenizer.encode(_cleaned, add_special_tokens=False)[:_TEXT_MAX_LEN - 1]
                text_enc = text_enc + [_NEWLINE_TOKEN_ID]
                ids = [_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID] + text_enc
                if len(ids) < seq_len:
                    ids += [_PAD_TOKEN_ID] * (seq_len - len(ids))
            else:
                ids = ([_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID]
                       + [_PAD_TOKEN_ID] * _TEXT_MAX_LEN)[:seq_len]
            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()
            raw_action = row.get("action")
            gt_action = torch.tensor(raw_action, dtype=torch.float32) if raw_action is not None else None
            batches.append({
                "state": state_vec.unsqueeze(0).float(),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).float(),
                "pixel_attention_mask": torch.ones(1, num_cameras, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": num_cameras,
                "task_text": task_text,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped ALOHA frame {idx}: {e}", file=sys.stderr)
    if not batches:
        return None
    print(f"Loaded {len(batches)} ALOHA samples ({num_cameras} cameras, {image_size}px).", flush=True)
    return batches


def _load_droid_for_cache(num_samples, num_cameras=2, image_size=224, seed=42,
                          device="cpu", dtype=None):
    import sys, random, torch, numpy as np
    from torchvision.transforms.functional import resize as tv_resize
    rng = random.Random(seed)
    _IMAGE_TOKEN_ID = 257152
    _PAD_TOKEN_ID = 0
    _BOS_TOKEN_ID = 2
    _NEWLINE_TOKEN_ID = 108
    _TEXT_MAX_LEN = 48
    num_image_tokens = (image_size // 14) ** 2
    tokenizer = _load_paligemma_tokenizer_parent()
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("lerobot/droid_100", image_transforms=None, video_backend="pyav")
    except Exception as e:
        print(f"WARNING: Could not load DROID dataset: {e}", file=sys.stderr)
        return None
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_samples]
    task_text_default = "pick and place"
    batches = []
    for idx in indices:
        try:
            row = ds[idx]
            def _chw(x):
                t = torch.as_tensor(np.asarray(x, dtype="float32"))
                if t.ndim == 3 and t.shape[-1] == 3:
                    t = t.permute(2, 0, 1) / 255.0
                elif t.max() > 1.0:
                    t = t / 255.0
                if t.shape[-2] != image_size or t.shape[-1] != image_size:
                    t = tv_resize(t, [image_size, image_size], antialias=True)
                return t
            ext_img = _chw(row["observation.images.exterior_image_1_left"])
            wrist_img = _chw(row["observation.images.wrist_image_left"])
            images = torch.stack([ext_img, wrist_img], dim=0)
            raw_state = np.asarray(row["observation.state"], dtype="float32")
            state_vec = torch.from_numpy(raw_state)
            if state_vec.numel() < 32:
                state_vec = torch.nn.functional.pad(state_vec, (0, 32 - state_vec.numel()))
            else:
                state_vec = state_vec[:32]
            task = row.get("task", task_text_default)
            if isinstance(task, (bytes, bytearray)):
                task = task.decode("utf-8", errors="replace")
            if not isinstance(task, str) or not task.strip():
                task = task_text_default
            total_img_tok = num_image_tokens * 2
            seq_len = total_img_tok + 1 + _TEXT_MAX_LEN
            if tokenizer is not None:
                _cleaned = task.strip().replace("_", " ").replace("\n", " ")
                text_enc = tokenizer.encode(_cleaned, add_special_tokens=False)[:_TEXT_MAX_LEN - 1]
                text_enc = text_enc + [_NEWLINE_TOKEN_ID]
                ids = [_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID] + text_enc
                if len(ids) < seq_len:
                    ids += [_PAD_TOKEN_ID] * (seq_len - len(ids))
            else:
                ids = ([_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID]
                       + [_PAD_TOKEN_ID] * _TEXT_MAX_LEN)[:seq_len]
            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()
            raw_action = row.get("action")
            gt_action = torch.tensor(np.asarray(raw_action, dtype="float32")) if raw_action is not None else None
            batches.append({
                "state": state_vec.unsqueeze(0).float(),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).float(),
                "pixel_attention_mask": torch.ones(1, 2, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": 2,
                "task_text": task,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped DROID frame {idx}: {e}", file=sys.stderr)
    if not batches:
        return None
    print(f"Loaded {len(batches)} DROID samples ({image_size}px).", flush=True)
    return batches


def _load_libero_for_cache(num_samples, num_cameras=2, image_size=224, seed=42,
                           device="cpu", dtype=None):
    import sys, random, torch, numpy as np
    from torchvision.transforms.functional import resize as tv_resize
    rng = random.Random(seed)
    _IMAGE_TOKEN_ID = 257152
    _PAD_TOKEN_ID = 0
    _BOS_TOKEN_ID = 2
    _NEWLINE_TOKEN_ID = 108
    _TEXT_MAX_LEN = 48
    num_image_tokens = (image_size // 14) ** 2
    tokenizer = _load_paligemma_tokenizer_parent()
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("lerobot/libero_10_image", image_transforms=None)
    except Exception as e:
        print(f"WARNING: Could not load LIBERO dataset: {e}", file=sys.stderr)
        return None
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_samples]
    task_text_default = "pick up the object"
    batches = []
    for idx in indices:
        try:
            row = ds[idx]
            def _chw(x):
                t = torch.as_tensor(np.asarray(x, dtype="float32"))
                if t.ndim == 3 and t.shape[-1] == 3:
                    t = t.permute(2, 0, 1) / 255.0
                elif t.max() > 1.0:
                    t = t / 255.0
                if t.shape[-2] != image_size or t.shape[-1] != image_size:
                    t = tv_resize(t, [image_size, image_size], antialias=True)
                return t
            base_img = _chw(row["observation.images.image"])
            wrist_img = _chw(row["observation.images.wrist_image"])
            images = torch.stack([base_img, wrist_img], dim=0)
            raw_state = np.asarray(row["observation.state"], dtype="float32")
            state_vec = torch.from_numpy(raw_state)
            if state_vec.numel() < 32:
                state_vec = torch.nn.functional.pad(state_vec, (0, 32 - state_vec.numel()))
            else:
                state_vec = state_vec[:32]
            task = row.get("task", task_text_default)
            if isinstance(task, (bytes, bytearray)):
                task = task.decode("utf-8", errors="replace")
            if not isinstance(task, str) or not task.strip():
                task = task_text_default
            total_img_tok = num_image_tokens * 2
            seq_len = total_img_tok + 1 + _TEXT_MAX_LEN
            if tokenizer is not None:
                _cleaned = task.strip().replace("_", " ").replace("\n", " ")
                text_enc = tokenizer.encode(_cleaned, add_special_tokens=False)[:_TEXT_MAX_LEN - 1]
                text_enc = text_enc + [_NEWLINE_TOKEN_ID]
                ids = [_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID] + text_enc
                if len(ids) < seq_len:
                    ids += [_PAD_TOKEN_ID] * (seq_len - len(ids))
            else:
                ids = ([_IMAGE_TOKEN_ID] * total_img_tok + [_BOS_TOKEN_ID]
                       + [_PAD_TOKEN_ID] * _TEXT_MAX_LEN)[:seq_len]
            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()
            raw_action = row.get("action")
            gt_action = torch.tensor(np.asarray(raw_action, dtype="float32")) if raw_action is not None else None
            batches.append({
                "state": state_vec.unsqueeze(0).float(),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).float(),
                "pixel_attention_mask": torch.ones(1, 2, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": 2,
                "task_text": task,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped LIBERO frame {idx}: {e}", file=sys.stderr)
    if not batches:
        return None
    print(f"Loaded {len(batches)} LIBERO samples ({image_size}px).", flush=True)
    return batches


def materialize_aloha_cache_for_openpi(cache_dir, scenarios, latency_scenarios, *, seed, image_size):
    return _materialize_cache(_load_aloha_for_cache, cache_dir, scenarios, latency_scenarios,
                              seed=seed, image_size=image_size, cache_key="aloha")


def materialize_droid_cache_for_openpi(cache_dir, scenarios, latency_scenarios, *, seed, image_size):
    return _materialize_cache(_load_droid_for_cache, cache_dir, scenarios, latency_scenarios,
                              seed=seed, image_size=image_size, cache_key="droid")


def materialize_libero_cache_for_openpi(cache_dir, scenarios, latency_scenarios, *, seed, image_size):
    return _materialize_cache(_load_libero_for_cache, cache_dir, scenarios, latency_scenarios,
                              seed=seed, image_size=image_size, cache_key="libero")


# ---------------------------------------------------------------------------
# Result comparison helpers
# ---------------------------------------------------------------------------

def _build_scenarios(cfg: dict) -> list[dict]:
    """ALOHA throughput scenarios (3-camera and 1-camera)."""
    num_requests = cfg.get("num_requests", 100)
    return [
        {"name": "aloha-3cam", "num_cameras": 3, "num_requests": num_requests},
        {"name": "aloha-1cam", "num_cameras": 1, "num_requests": num_requests},
    ]


def _build_latency_scenarios() -> list[dict]:
    """ALOHA latency scenarios."""
    return [
        {"name": "aloha-single-3cam", "num_cameras": 3, "batch_size": 1,
         "num_warmup": 3, "num_iters": 10},
        {"name": "aloha-single-1cam", "num_cameras": 1, "batch_size": 1,
         "num_warmup": 3, "num_iters": 10},
    ]


def _build_scenarios_for_dataset(dataset_name: str, num_requests: int) -> list[dict]:
    if dataset_name == "aloha":
        return [
            {"name": "aloha-3cam", "num_cameras": 3, "num_requests": num_requests},
            {"name": "aloha-1cam", "num_cameras": 1, "num_requests": num_requests},
        ]
    elif dataset_name == "droid":
        return [
            {"name": "droid-2cam", "num_cameras": 2, "num_requests": num_requests},
        ]
    elif dataset_name == "libero":
        return [
            {"name": "libero-2cam", "num_cameras": 2, "num_requests": num_requests},
        ]
    return []


def _build_latency_scenarios_for_dataset(dataset_name: str) -> list[dict]:
    if dataset_name == "aloha":
        return [
            {"name": "aloha-single-3cam", "num_cameras": 3, "batch_size": 1,
             "num_warmup": 3, "num_iters": 10},
            {"name": "aloha-single-1cam", "num_cameras": 1, "batch_size": 1,
             "num_warmup": 3, "num_iters": 10},
        ]
    elif dataset_name == "droid":
        return [
            {"name": "droid-single", "num_cameras": 2, "batch_size": 1,
             "num_warmup": 3, "num_iters": 10},
        ]
    elif dataset_name == "libero":
        return [
            {"name": "libero-single", "num_cameras": 2, "batch_size": 1,
             "num_warmup": 3, "num_iters": 10},
        ]
    return []


def _print_throughput_comparison(kb_results, ref_results=None, ref_label: str = "reference"):
    data_src = kb_results[0].get("data_source", "synthetic") if kb_results else "synthetic"
    print("\n" + "=" * 90)
    print(f"  THROUGHPUT COMPARISON (inferences/sec) — data: {data_src}")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Requests':>9} {'fastkernels':>12}"
    if ref_results:
        header += f" {ref_label:>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb['num_requests']:>9} {kb['inferences_per_second']:>12.2f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                speedup = kb["inferences_per_second"] / ref["inferences_per_second"]
                line += f" {ref['inferences_per_second']:>12.2f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(kb_results, ref_results=None, ref_label: str = "reference"):
    rl = f"{ref_label} p50"
    print("\n" + "=" * 80)
    print("  LATENCY COMPARISON (seconds)")
    print("=" * 80)
    header = f"  {'Scenario':<25} {'fastkernels p50':>12}"
    if ref_results:
        header += f" {rl:>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 60)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<25} {kb_p50:>12.3f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                ref_lats = np.array(ref["latencies"])
                ref_p50 = np.percentile(ref_lats, 50)
                speedup = ref_p50 / kb_p50
                line += f" {ref_p50:>12.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _compare_actions(kb_actions_dir, ref_actions_dir):
    """Compare action outputs between engines."""
    import torch

    kb_files = sorted(
        f for f in os.listdir(kb_actions_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_actions_dir) else []
    ref_files = sorted(
        f for f in os.listdir(ref_actions_dir) if f.endswith(".pt")
    ) if os.path.isdir(ref_actions_dir) else []

    common = sorted(set(kb_files) & set(ref_files))
    if not common:
        return {}

    results = {}
    for fname in common:
        kb_act = torch.load(
            os.path.join(kb_actions_dir, fname), map_location="cpu", weights_only=True,
        ).float()
        ref_act = torch.load(
            os.path.join(ref_actions_dir, fname), map_location="cpu", weights_only=True,
        ).float()

        n = min(kb_act.shape[0], ref_act.shape[0])
        kb_act = kb_act[:n]
        ref_act = ref_act[:n]
        # Align horizon and action dims (e.g. fastkernels 32-dim vs OpenPI ALOHA 14-dim).
        if kb_act.ndim >= 3 and ref_act.ndim >= 3:
            h = min(kb_act.shape[1], ref_act.shape[1])
            d = min(kb_act.shape[2], ref_act.shape[2])
            kb_act = kb_act[:, :h, :d]
            ref_act = ref_act[:, :h, :d]
        else:
            d = min(kb_act.shape[-1], ref_act.shape[-1])
            kb_act = kb_act[..., :d]
            ref_act = ref_act[..., :d]

        mse = float((kb_act - ref_act).pow(2).mean())
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                kb_act.flatten(1), ref_act.flatten(1), dim=1,
            ).mean()
        )

        scenario = fname.replace("_actions.pt", "")
        results[scenario] = {
            "num_samples": n,
            "mean_mse": mse,
            "mean_cosine_sim": cos_sim,
        }
    return results


def _print_correctness(correctness):
    print("\n" + "=" * 80)
    print("  CORRECTNESS COMPARISON (action space)")
    print("=" * 80)
    print(f"  {'Scenario':<25} {'Samples':>8} {'Mean MSE':>12} {'CosSim':>10} {'Result':>8}")
    print("  " + "-" * 65)

    for scenario, stats in correctness.items():
        cos = stats["mean_cosine_sim"]
        verdict = "PASS" if cos > 0.95 else ("WARN" if cos > 0.0 else "FAIL")
        print(
            f"  {scenario:<25} {stats['num_samples']:>8} "
            f"{stats['mean_mse']:>12.6f} {cos:>10.6f} {verdict:>8}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_dataset_benchmark(
    dataset_name: str,
    args,
    gpu_name: str,
    use_real_data: bool,
    strict_data: bool,
    run_reference: bool,
    ref_py: "str | None",
    output_dir: str,
) -> dict:
    """Run full benchmark (throughput + latency + correctness) for one dataset."""
    import torch as _torch

    # Per-dataset model/checkpoint paths (fall back to global --model/--openpi-checkpoint).
    # The OpenPI checkpoint defaults to the fastkernels model path so passing only --droid-model /
    # --libero-model is enough for both sides to use the matching domain checkpoint.
    model_path = getattr(args, f"{dataset_name}_model", None) or args.model
    openpi_ckpt = (getattr(args, f"{dataset_name}_openpi_checkpoint", None)
                   or getattr(args, f"{dataset_name}_model", None)
                   or args.openpi_checkpoint)

    # Resolve action_horizon from the checkpoint's own config.json. OpenPI's
    # conversion script writes ``action_horizon`` (e.g. 10 for pi0_droid,
    # 50 for pi0_aloha/libero). fastkernels's Pi0Config reads ``chunk_size`` but
    # now falls back to ``action_horizon`` if only that is present. We use
    # this value to size the shared flow-matching noise; size mismatches
    # between noise and the model's config.action_horizon crash OpenPI with
    # a suffix_att_2d_masks broadcast error (pi0_pytorch.py ln 308).
    ckpt_chunk_size = PI0_CONFIG.chunk_size
    try:
        with open(os.path.join(model_path, "config.json")) as _cf:
            _cfg_json = json.load(_cf)
        ckpt_chunk_size = int(_cfg_json.get("chunk_size",
                                            _cfg_json.get("action_horizon", PI0_CONFIG.chunk_size)))
    except Exception as _e:
        print(f"WARNING: could not read {model_path}/config.json ({_e}); "
              f"falling back to chunk_size={ckpt_chunk_size}", file=sys.stderr)

    scenarios = _build_scenarios_for_dataset(dataset_name, args.num_requests) if not args.skip_throughput else []
    latency_scenarios = _build_latency_scenarios_for_dataset(dataset_name) if not args.skip_latency else []

    noise_dir = os.path.join(output_dir, "shared_noise")
    if os.path.isdir(noise_dir):
        shutil.rmtree(noise_dir)
    os.makedirs(noise_dir, exist_ok=True)
    for scenario in scenarios:
        n_req = scenario["num_requests"]
        noise_list = []
        for ri in range(n_req):
            ns = args.seed + 10000 * ri + 424242
            g = _torch.Generator(device="cpu")
            g.manual_seed(ns)
            noise_list.append(_torch.randn(
                1, ckpt_chunk_size, PI0_CONFIG.max_action_dim,
                generator=g, dtype=_torch.float32,
            ))
        _torch.save(_torch.cat(noise_list, 0),
                    os.path.join(noise_dir, f"{scenario['name']}_noise.pt"))

    base_config = {
        "model": model_path,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "fastkernels",
        "image_size": PI0_CONFIG.image_resolution[0],
        "chunk_size": ckpt_chunk_size,
        "max_action_dim": PI0_CONFIG.max_action_dim,
        "use_real_data": use_real_data,
        "strict_data": strict_data,
        "noise_dir": noise_dir,
        "dataset_name": dataset_name,
    }

    save_actions = run_reference
    if save_actions:
        kb_actions_dir = os.path.join(output_dir, "actions", "fastkernels")
        ref_actions_dir = os.path.join(output_dir, "actions", "openpi")
        actions_root = os.path.join(output_dir, "actions")
        if os.path.isdir(actions_root):
            shutil.rmtree(actions_root)
        os.makedirs(kb_actions_dir, exist_ok=True)
        os.makedirs(ref_actions_dir, exist_ok=True)
    else:
        kb_actions_dir = ref_actions_dir = None

    kb_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
    if kb_actions_dir:
        kb_config["actions_dir"] = kb_actions_dir

    print(f"\n--- {dataset_name.upper()} | fastkernels ---", flush=True)
    kb_data = run_worker(FASTKERNELS_PI0_WORKER, kb_config,
                         f"fastkernels Pi0 [{dataset_name}]", timeout=36000)
    if kb_data is None:
        print(f"ERROR: fastkernels [{dataset_name}] failed.", file=sys.stderr)
        return {"error": "fastkernels failed"}

    ref_data = None
    if run_reference and args.reference == "openpi":
        materialize_fns = {
            "aloha": materialize_aloha_cache_for_openpi,
            "droid": materialize_droid_cache_for_openpi,
            "libero": materialize_libero_cache_for_openpi,
        }
        batch_cache = None
        if use_real_data:
            cache_dir = os.path.join(output_dir, f"{dataset_name}_batch_cache")
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
            mat_fn = materialize_fns[dataset_name]
            batch_cache = mat_fn(cache_dir, scenarios, latency_scenarios,
                                 seed=args.seed, image_size=PI0_CONFIG.image_resolution[0])
            if batch_cache is None:
                print(f"ERROR: Failed to materialize {dataset_name} batches.", file=sys.stderr)
                return {"error": "materialization failed", "fastkernels": kb_data}
            print(f"Materialized {dataset_name} batch cache: {cache_dir}", flush=True)

        ref_config = {
            **base_config,
            "model": openpi_ckpt,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
            "openpi_checkpoint": openpi_ckpt,
            "openpi_backend": args.openpi_backend,
            # train config: auto-selected per dataset unless overridden
        }
        if args.openpi_train_config != "pi0_aloha_pen_uncap":
            ref_config["openpi_train_config"] = args.openpi_train_config
        if ref_actions_dir:
            ref_config["actions_dir"] = ref_actions_dir
        if batch_cache is not None:
            ref_config[f"{dataset_name}_batch_cache"] = batch_cache

        print(f"\n--- {dataset_name.upper()} | OpenPI ---", flush=True)
        ref_data = run_worker(OPENPI_POLICY_WORKER, ref_config,
                              f"OpenPI [{dataset_name}]", timeout=36000,
                              python_executable=ref_py)
        if ref_data is None:
            print(f"\nERROR: OpenPI [{dataset_name}] did not complete. "
                  "Ensure --reference-python points at the OpenPI venv and the "
                  "checkpoint contains model.safetensors.", file=sys.stderr)

    elif run_reference and args.reference == "hf":
        ref_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
        if ref_actions_dir:
            ref_config["actions_dir"] = ref_actions_dir
        ref_data = run_worker(HF_PI0_WORKER, ref_config,
                              f"HF Pi0 [{dataset_name}]", timeout=36000)

    ref_label = "OpenPI" if args.reference == "openpi" else "HF"
    kb_tp = kb_data.get("throughput", [])
    kb_lat = kb_data.get("latency", [])
    ref_tp = ref_data.get("throughput", []) if ref_data else None
    ref_lat = ref_data.get("latency", []) if ref_data else None

    if kb_tp:
        _print_throughput_comparison(kb_tp, ref_tp, ref_label=ref_label)
    if kb_lat:
        _print_latency_comparison(kb_lat, ref_lat, ref_label=ref_label)

    correctness = None
    if save_actions and kb_actions_dir and ref_actions_dir and ref_data is not None:
        correctness = _compare_actions(kb_actions_dir, ref_actions_dir)
        if correctness:
            _print_correctness(correctness)

    return {"fastkernels": kb_data, "reference": ref_data, "correctness": correctness}


def main():
    parser = argparse.ArgumentParser(
        description="Pi0 benchmark: fastkernels vs OpenPI — ALOHA, DROID, and LIBERO workloads",
    )
    parser.add_argument("--model", type=str, default="/raid/user_data/olu/pi0_aloha_pen_uncap_pytorch",
                        help="Default Pi0 checkpoint for all datasets")
    parser.add_argument("--droid-model", type=str, default=None,
                        help="Checkpoint for DROID (overrides --model; use pi0_droid converted checkpoint)")
    parser.add_argument("--libero-model", type=str, default=None,
                        help="Checkpoint for LIBERO (overrides --model; use pi0_libero converted checkpoint)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=["aloha", "droid", "libero"],
                        default=["aloha", "droid", "libero"],
                        help="Datasets to benchmark (default: all three)")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip reference engine; fastkernels only")
    parser.add_argument("--skip-openpi", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Synthetic tensors only (debug). Do not report these numbers.")
    parser.add_argument("--reference", choices=("openpi", "hf"), default="openpi",
                        help="Reference: openpi (paper comparison) or hf (debug only)")
    parser.add_argument("--openpi-backend", choices=("pytorch", "jax"), default="pytorch")
    parser.add_argument("--openpi-checkpoint", default="/raid/user_data/olu/pi0_aloha_pen_uncap_pytorch",
                        help="Default OpenPI checkpoint (model.safetensors + assets/)")
    parser.add_argument("--droid-openpi-checkpoint", type=str, default=None,
                        help="OpenPI checkpoint for DROID (overrides --openpi-checkpoint)")
    parser.add_argument("--libero-openpi-checkpoint", type=str, default=None,
                        help="OpenPI checkpoint for LIBERO (overrides --openpi-checkpoint)")
    parser.add_argument("--openpi-train-config", default="pi0_aloha_pen_uncap",
                        help="Override OpenPI train config for ALL datasets (auto-selected per dataset by default)")
    parser.add_argument("--reference-python", default=None,
                        help="Python executable for OpenPI subprocess. Overrides env OPENPI_PYTHON.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    skip_reference = args.skip_reference or args.skip_openpi
    ref_py = args.reference_python or os.environ.get("OPENPI_PYTHON")
    gpu_name = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu_name / short)

    use_real_data = not args.synthetic_only
    strict_data = use_real_data

    print(f"\nBenchmark: Pi0 on {gpu_name}")
    print(f"Datasets:  {args.datasets}")
    print(f"Model:     {args.model}")
    print(f"Seed: {args.seed}  |  Steps: {args.num_steps}  |  Requests: {args.num_requests}")
    print(f"Data: {'real' if use_real_data else 'synthetic (debug)'}")
    print(f"Reference: {args.reference}" + (f" ({args.openpi_backend})" if args.reference == "openpi" else ""))
    if ref_py:
        print(f"Reference Python: {ref_py}")
    print(f"Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for dataset_name in args.datasets:
        ds_out = os.path.join(args.output_dir, dataset_name)
        os.makedirs(ds_out, exist_ok=True)
        result = _run_dataset_benchmark(
            dataset_name=dataset_name,
            args=args,
            gpu_name=gpu_name,
            use_real_data=use_real_data,
            strict_data=strict_data,
            run_reference=not skip_reference,
            ref_py=ref_py,
            output_dir=ds_out,
        )
        all_results[dataset_name] = result

        results_path = os.path.join(ds_out, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "model": getattr(args, f"{dataset_name}_model", None) or args.model,
                "gpu": gpu_name,
                "seed": args.seed,
                "num_steps": args.num_steps,
                "reference": args.reference,
                **result,
            }, f, indent=2)
        print(f"  Results saved: {results_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results: {summary_path}")


if __name__ == "__main__":
    main()
