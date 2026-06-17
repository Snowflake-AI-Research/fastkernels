"""Loaders and utilities for InstantNGP-style NeRF benchmarks."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import msgpack
import torch
import torch.nn as nn


_KB_ROOT = Path(__file__).resolve().parents[1]
_INSTANT_NGP_ROOT = _KB_ROOT / "third_party" / "instant-ngp"
_INSTANT_NGP_BUILD = _INSTANT_NGP_ROOT / "build"
_SNAPSHOT_CACHE = Path.home() / ".cache" / "kb_nano" / "instantngp"


@dataclass
class InstantNGPScene:
    scene_name: str
    scene_path: str
    width: int
    height: int
    num_views: int


@dataclass
class InstantNGPView:
    camera_matrix: torch.Tensor
    focal_length: torch.Tensor
    principal_point: torch.Tensor
    resolution: tuple[int, int]
    lens_params: torch.Tensor


def is_instantngp_model(model_name: str) -> bool:
    name = model_name.lower()
    return "instantngp" in name or "instant-ngp" in name


def _ensure_pyngp():
    build = str(_INSTANT_NGP_BUILD.resolve())
    scripts = str((_INSTANT_NGP_ROOT / "scripts").resolve())
    if build not in sys.path:
        sys.path.insert(0, build)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import pyngp as ngp  # type: ignore

    return ngp


def load_fox_scene(scene_name: str = "fox") -> InstantNGPScene:
    scene_root = _INSTANT_NGP_ROOT / "data" / "nerf" / scene_name
    transforms_path = scene_root / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(
            f"InstantNGP scene metadata not found at {transforms_path}"
        )
    obj = json.loads(transforms_path.read_text())
    return InstantNGPScene(
        scene_name=scene_name,
        scene_path=str(scene_root.resolve()),
        width=int(obj["w"]),
        height=int(obj["h"]),
        num_views=len(obj["frames"]),
    )


def load_fox_views(
    scene_name: str = "fox",
    train_steps: int = 50,
) -> list[InstantNGPView]:
    snapshot_path = ensure_fox_snapshot(train_steps=train_steps, scene_name=scene_name)
    obj = _load_snapshot_msgpack(snapshot_path)
    dataset = obj["snapshot"]["nerf"]["dataset"]
    views: list[InstantNGPView] = []
    for xform, metadata in zip(dataset["xforms"], dataset["metadata"], strict=True):
        lens = metadata["lens"]
        views.append(
            InstantNGPView(
                camera_matrix=torch.tensor(xform["start"], dtype=torch.float32),
                focal_length=torch.tensor(metadata["focal_length"], dtype=torch.float32),
                principal_point=torch.tensor(metadata["principal_point"], dtype=torch.float32),
                resolution=(int(metadata["resolution"][0]), int(metadata["resolution"][1])),
                lens_params=torch.tensor(
                    [
                        float(lens.get("k1", 0.0)),
                        float(lens.get("k2", 0.0)),
                        float(lens.get("p1", 0.0)),
                        float(lens.get("p2", 0.0)),
                    ],
                    dtype=torch.float32,
                ),
            )
        )
    return views


def _build_fox_snapshot_in_process(
    *,
    scene_name: str,
    train_steps: int,
    snapshot_path: str,
) -> None:
    scene = load_fox_scene(scene_name)
    ngp = _ensure_pyngp()
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.root_dir = str(_INSTANT_NGP_ROOT.resolve())
    testbed.load_training_data(scene.scene_path)
    testbed.shall_train = True
    for _ in range(train_steps):
        testbed.frame()
    testbed.save_snapshot(snapshot_path, False, True)


def _build_fox_snapshot_subprocess(
    *,
    scene_name: str,
    train_steps: int,
    snapshot_path: str,
) -> None:
    script = f"""
import sys
sys.path.insert(0, {str(_KB_ROOT)!r})
from infra.nerf_loader import _build_fox_snapshot_in_process
_build_fox_snapshot_in_process(
    scene_name={scene_name!r},
    train_steps={train_steps},
    snapshot_path={snapshot_path!r},
)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=str(_KB_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to build InstantNGP snapshot in helper subprocess.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def ensure_fox_snapshot(
    train_steps: int = 50,
    scene_name: str = "fox",
) -> str:
    _SNAPSHOT_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_path = _SNAPSHOT_CACHE / f"{scene_name}_{train_steps}steps.ingp"
    if snapshot_path.is_file():
        return str(snapshot_path)

    # Keep pyngp snapshot creation out of the current process so tinycudann
    # initialization for the native path does not inherit a broken CUDA state.
    _build_fox_snapshot_subprocess(
        scene_name=scene_name,
        train_steps=train_steps,
        snapshot_path=str(snapshot_path),
    )
    return str(snapshot_path)


def _load_snapshot_msgpack(snapshot_path: str) -> dict:
    with gzip.open(snapshot_path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)


def load_fox_aabb(
    scene_name: str = "fox",
    train_steps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    snapshot_path = ensure_fox_snapshot(train_steps=train_steps, scene_name=scene_name)
    obj = _load_snapshot_msgpack(snapshot_path)
    snapshot = obj["snapshot"]
    aabb_min = torch.tensor(snapshot["aabb"]["min"], dtype=torch.float32)
    aabb_max = torch.tensor(snapshot["aabb"]["max"], dtype=torch.float32)
    return aabb_min, aabb_max


def sample_real_fox_field_inputs(
    *,
    num_samples: int,
    device: str = "cuda",
    scene_name: str = "fox",
    train_steps: int = 50,
    num_views: int = 2,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor]:
    views = load_fox_views(scene_name=scene_name, train_steps=train_steps)
    if not views:
        raise ValueError(f"No training views found for scene {scene_name!r}")
    num_views = max(1, min(num_views, len(views)))
    aabb_min, aabb_max = load_fox_aabb(scene_name=scene_name, train_steps=train_steps)
    aabb_min = aabb_min.to(device=device)
    aabb_max = aabb_max.to(device=device)
    aabb_extent = (aabb_max - aabb_min).clamp_min(1e-6)

    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    positions_chunks: list[torch.Tensor] = []
    directions_chunks: list[torch.Tensor] = []
    remaining = num_samples
    eps = 1e-6

    while remaining > 0:
        candidate_count = max(remaining * 2, 4096)
        view_ids = torch.randint(0, num_views, (candidate_count,), generator=cpu_gen)
        xs = torch.randint(0, views[0].resolution[0], (candidate_count,), generator=cpu_gen)
        ys = torch.randint(0, views[0].resolution[1], (candidate_count,), generator=cpu_gen)
        depth_u = torch.rand(candidate_count, generator=cpu_gen)

        batch_positions: list[torch.Tensor] = []
        batch_directions: list[torch.Tensor] = []

        for view_idx in range(num_views):
            sel = (view_ids == view_idx).nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            view = views[view_idx]
            fx, fy = view.focal_length.to(device=device)
            cx, cy = view.principal_point.to(device=device)
            width = float(view.resolution[0])
            height = float(view.resolution[1])
            u = (xs[sel].to(device=device, dtype=torch.float32) + 0.5) / width
            v = (ys[sel].to(device=device, dtype=torch.float32) + 0.5) / height
            dirs_cam = torch.stack(
                [
                    (u - cx) * width / fx,
                    (v - cy) * height / fy,
                    torch.ones_like(u),
                ],
                dim=-1,
            )
            dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)
            camera_matrix = view.camera_matrix.to(device=device)
            rotation = camera_matrix[:, :3]
            origin = camera_matrix[:, 3]
            dirs_world = torch.nn.functional.normalize(dirs_cam @ rotation.T, dim=-1)
            origins = origin.unsqueeze(0).expand_as(dirs_world)

            safe_dirs = torch.where(
                dirs_world.abs() < eps,
                torch.where(dirs_world >= 0, torch.full_like(dirs_world, eps), torch.full_like(dirs_world, -eps)),
                dirs_world,
            )
            inv_dirs = 1.0 / safe_dirs
            t0 = (aabb_min - origins) * inv_dirs
            t1 = (aabb_max - origins) * inv_dirs
            t_near = torch.max(torch.minimum(t0, t1), dim=-1).values
            t_far = torch.min(torch.maximum(t0, t1), dim=-1).values
            t_start = torch.clamp(t_near, min=0.0) + eps
            valid = t_far > t_start
            if not valid.any():
                continue

            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            valid_idx_cpu = valid_idx.cpu()
            u = depth_u[sel][valid_idx_cpu].to(device=device, dtype=torch.float32)
            t = t_start[valid_idx] + u * (t_far[valid_idx] - t_start[valid_idx])
            pos_world = origins[valid_idx] + dirs_world[valid_idx] * t.unsqueeze(-1)
            positions = (pos_world - aabb_min) / aabb_extent
            directions = (dirs_world[valid_idx] + 1.0) * 0.5
            batch_positions.append(positions.clamp_(0.0, 1.0))
            batch_directions.append(directions.clamp_(0.0, 1.0))

        if not batch_positions:
            raise RuntimeError("Failed to sample valid InstantNGP field inputs from real fox views")

        positions = torch.cat(batch_positions, dim=0)
        directions = torch.cat(batch_directions, dim=0)
        take = min(remaining, positions.shape[0])
        positions_chunks.append(positions[:take])
        directions_chunks.append(directions[:take])
        remaining -= take

    return torch.cat(positions_chunks, dim=0), torch.cat(directions_chunks, dim=0)


class InstantNGPReference(nn.Module):
    def __init__(
        self,
        testbed,
        width: int,
        height: int,
        spp: int = 1,
        linear: bool = True,
    ):
        super().__init__()
        self.testbed = testbed
        self.width = width
        self.height = height
        self.spp = spp
        self.linear = linear

    def forward(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
    ) -> torch.Tensor:
        self.testbed.set_camera_to_training_view(view_index)
        image = self.testbed.render(
            width or self.width,
            height or self.height,
            spp or self.spp,
            self.linear,
        )
        return torch.from_numpy(image.copy())


def _load_instantngp_testbed(
    scene_name: str = "fox",
    train_steps: int = 50,
):
    ngp = _ensure_pyngp()
    scene = load_fox_scene(scene_name)
    snapshot_path = ensure_fox_snapshot(train_steps=train_steps, scene_name=scene_name)
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.root_dir = str(_INSTANT_NGP_ROOT.resolve())
    testbed.load_training_data(scene.scene_path)
    testbed.load_snapshot(snapshot_path)
    testbed.shall_train = False
    return scene, testbed


def load_reference_instantngp(
    scene_name: str = "fox",
    train_steps: int = 50,
    width: int | None = None,
    height: int | None = None,
    spp: int = 1,
):
    scene, testbed = _load_instantngp_testbed(
        scene_name=scene_name,
        train_steps=train_steps,
    )
    return InstantNGPReference(
        testbed=testbed,
        width=width or scene.width,
        height=height or scene.height,
        spp=spp,
        linear=True,
    ).eval()


def load_wrapped_instantngp(
    scene_name: str = "fox",
    train_steps: int = 50,
    width: int | None = None,
    height: int | None = None,
    spp: int = 1,
):
    from tasks.baseline.L4.instant_ngp import InstantNGP

    scene, testbed = _load_instantngp_testbed(
        scene_name=scene_name,
        train_steps=train_steps,
    )
    return InstantNGP(
        testbed=testbed,
        scene_name=scene_name,
        width=width or scene.width,
        height=height or scene.height,
        spp=spp,
        linear=True,
    ).eval()


def load_ours_instantngp(
    scene_name: str = "fox",
    train_steps: int = 50,
    width: int | None = None,
    height: int | None = None,
    spp: int = 1,
):
    return load_wrapped_instantngp(
        scene_name=scene_name,
        train_steps=train_steps,
        width=width,
        height=height,
        spp=spp,
    )
