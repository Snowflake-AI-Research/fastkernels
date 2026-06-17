"""Graphics scene/model loaders for 3D Gaussian Splatting."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


_SH_C0 = 0.28209479177387814
_PLY_REPO = "Voxel51/gaussian_splatting"
_PLY_ITERATION = 30000
_SCENE_ALIASES = {
    "poster": "train",  # Backward-compatible CLI alias; trained 3DGS scene.
}
_PLY_TYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass
class GaussianScene:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    viewmats: torch.Tensor
    Ks: torch.Tensor
    width: int
    height: int
    scene_name: str
    model_source: str = _PLY_REPO
    model_path: str = ""
    iteration: int = _PLY_ITERATION
    total_points: int = 0
    loaded_points: int = 0


def is_3dgs_model(model_name: str) -> bool:
    name = model_name.lower()
    return "3dgs" in name or "gaussian" in name or "splat" in name


def _resolve_scene_name(scene_name: str) -> str:
    return _SCENE_ALIASES.get(scene_name.lower(), scene_name)


def _download_gaussian_ply(scene_name: str, iteration: int) -> Path:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    filename = f"FO_dataset/{scene_name}/point_cloud/iteration_{iteration}/point_cloud.ply"
    return Path(
        hf_hub_download(
            repo_id=_PLY_REPO,
            filename=filename,
            repo_type="dataset",
        )
    )


def _read_ply_header(f) -> tuple[int, np.dtype]:
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    ply_format = None

    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("Unexpected EOF while reading PLY header")
        line = raw.decode("ascii").strip()
        if line.startswith("format "):
            ply_format = line.split()[1]
        elif line.startswith("element "):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif line.startswith("property ") and in_vertex:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("List vertex properties are not supported")
            if parts[1] not in _PLY_TYPE_MAP:
                raise ValueError(f"Unsupported PLY property type: {parts[1]}")
            properties.append((parts[2], _PLY_TYPE_MAP[parts[1]]))
        elif line == "end_header":
            break

    if ply_format != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY, got {ply_format!r}")
    if vertex_count is None:
        raise ValueError("PLY header does not contain a vertex element")
    return vertex_count, np.dtype(properties)


def _read_ply_vertices(path: Path) -> tuple[np.ndarray, int]:
    with path.open("rb") as f:
        vertex_count, dtype = _read_ply_header(f)
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
    if vertices.shape[0] != vertex_count:
        raise ValueError(
            f"Expected {vertex_count} PLY vertices, read {vertices.shape[0]}"
        )
    return vertices, vertex_count


def _field_stack(vertices: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in names if name not in vertices.dtype.names]
    if missing:
        raise ValueError(f"PLY is missing required fields: {missing}")
    return np.stack([vertices[name] for name in names], axis=-1).astype(np.float32)


def _load_gaussians_from_ply(
    path: Path,
    max_points: int | None,
) -> tuple[dict[str, np.ndarray], int, int]:
    vertices, total_points = _read_ply_vertices(path)
    if max_points is not None and max_points > 0 and total_points > max_points:
        indices = np.linspace(0, total_points - 1, max_points, dtype=np.int64)
        vertices = vertices[indices]

    means = _field_stack(vertices, ("x", "y", "z"))
    colors = 0.5 + _SH_C0 * _field_stack(vertices, ("f_dc_0", "f_dc_1", "f_dc_2"))
    colors = np.clip(colors, 0.0, 1.0)
    opacities = vertices["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities))
    scales = np.exp(_field_stack(vertices, ("scale_0", "scale_1", "scale_2")))
    quats = _field_stack(vertices, ("rot_0", "rot_1", "rot_2", "rot_3"))
    quats = quats / np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-8)

    arrays = {
        "means": np.ascontiguousarray(means),
        "colors": np.ascontiguousarray(colors),
        "opacities": np.ascontiguousarray(opacities),
        "scales": np.ascontiguousarray(scales),
        "quats": np.ascontiguousarray(quats),
    }
    return arrays, total_points, int(vertices.shape[0])


def _look_at_viewmat(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    forward = torch.nn.functional.normalize(center - eye, dim=0)
    right = torch.cross(forward, up, dim=0)
    if torch.linalg.norm(right) < 1e-6:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        right = torch.cross(forward, up, dim=0)
    right = torch.nn.functional.normalize(right, dim=0)
    down = torch.nn.functional.normalize(torch.cross(forward, right, dim=0), dim=0)

    view = torch.eye(4, dtype=torch.float32)
    view[:3, :3] = torch.stack([right, down, forward], dim=0)
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _make_orbit_cameras(
    means: np.ndarray,
    num_cameras: int,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = np.percentile(means, [5.0, 95.0], axis=0)
    center = torch.tensor((lo + hi) * 0.5, dtype=torch.float32)
    extent = torch.tensor(np.maximum(hi - lo, 1e-3), dtype=torch.float32)
    radius = max(float(torch.linalg.norm(extent).item()) * 1.35, 1.0)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    viewmats = []
    for idx in range(max(num_cameras, 1)):
        theta = 2.0 * math.pi * idx / max(num_cameras, 1)
        eye = center + torch.tensor(
            [
                math.cos(theta) * radius,
                math.sin(theta) * radius,
                0.35 * radius,
            ],
            dtype=torch.float32,
        )
        viewmats.append(_look_at_viewmat(eye, center, up))

    focal = 0.7 * float(max(width, height))
    K = torch.tensor(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    Ks = K.unsqueeze(0).expand(len(viewmats), -1, -1).contiguous()
    return torch.stack(viewmats, dim=0), Ks


def load_3dgs_scene(
    scene_name: str = "train",
    num_cameras: int = 2,
    max_points: int = 100000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    width: int = 1920,
    height: int = 1080,
    iteration: int = _PLY_ITERATION,
) -> GaussianScene:
    resolved_scene = _resolve_scene_name(scene_name)
    ply_path = _download_gaussian_ply(resolved_scene, iteration)
    arrays, total_points, loaded_points = _load_gaussians_from_ply(
        ply_path,
        max_points=max_points,
    )
    viewmats, Ks = _make_orbit_cameras(arrays["means"], num_cameras, width, height)

    return GaussianScene(
        means=torch.from_numpy(arrays["means"]).to(device=device, dtype=dtype),
        quats=torch.from_numpy(arrays["quats"]).to(device=device, dtype=dtype),
        scales=torch.from_numpy(arrays["scales"]).to(device=device, dtype=dtype),
        opacities=torch.from_numpy(arrays["opacities"]).to(device=device, dtype=dtype),
        colors=torch.from_numpy(arrays["colors"]).to(device=device, dtype=dtype),
        viewmats=viewmats.to(device=device, dtype=dtype),
        Ks=Ks.to(device=device, dtype=dtype),
        width=width,
        height=height,
        scene_name=resolved_scene,
        model_path=str(ply_path),
        iteration=iteration,
        total_points=total_points,
        loaded_points=loaded_points,
    )


def load_poster_scene(
    scene_name: str = "poster",
    num_cameras: int = 2,
    max_points: int = 100000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> GaussianScene:
    return load_3dgs_scene(
        scene_name=scene_name,
        num_cameras=num_cameras,
        max_points=max_points,
        device=device,
        dtype=dtype,
    )


class GSplatReference(torch.nn.Module):
    def __init__(self, scene: GaussianScene):
        super().__init__()
        self.scene = scene

    def forward(self, return_meta: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        from gsplat.rendering import rasterization

        return rasterization(
            self.scene.means,
            self.scene.quats,
            self.scene.scales,
            self.scene.opacities,
            self.scene.colors,
            self.scene.viewmats,
            self.scene.Ks,
            self.scene.width,
            self.scene.height,
            packed=False,
        )


def load_reference_3dgs(scene: GaussianScene):
    return GSplatReference(scene).eval()


def load_ours_3dgs(scene: GaussianScene):
    from tasks.baseline.L4.gaussian_splatting import GaussianSplatting

    return GaussianSplatting(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        colors=scene.colors,
        width=scene.width,
        height=scene.height,
        viewmats=scene.viewmats,
        Ks=scene.Ks,
    ).eval()
