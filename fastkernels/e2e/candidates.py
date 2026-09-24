"""Candidate-set helpers for ``fastkernels e2e``: working copies, per-model kernel lists,
and attributing a crash to a candidate kernel."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


def prepare_set(src: Path, work_root: Path) -> Path:
    """Writable working copy of a frozen candidate set (kernels JIT-build next to their own
    files). Re-used across runs while the source manifest is unchanged."""
    dst = work_root / src.name
    stamp = hashlib.sha256((src / "manifest.json").read_bytes()).hexdigest() \
        if (src / "manifest.json").is_file() else "none"
    marker = dst / ".source_manifest_sha256"
    if dst.exists() and marker.is_file() and marker.read_text() == stamp:
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    marker.write_text(stamp)
    return dst


def kernels_for(set_dir: Path, hf_name: str) -> list[str]:
    """Kernels (``L<n>:<stem>``) of this set that the model instantiates, from the set's
    manifest (``scenarios`` map built from the default capture); all kernels otherwise."""
    manifest = set_dir / "manifest.json"
    if manifest.is_file():
        scen = json.loads(manifest.read_text()).get("scenarios") or {}
        if hf_name in scen:
            return list(scen[hf_name])
    return sorted(f"{p.parent.name}:{p.stem}" for p in set_dir.glob("L[1-4]/*.py"))


_FRAME_RE = re.compile(r'File "([^"]+)", line \d+')


def culprits(log: str, set_dir: Path) -> list[str]:
    """Candidate kernels named in the LAST traceback of ``log``, innermost frame first.

    Frames are matched against files of the working set directory, so both kernels that
    were swapped in and kernels they import (``from ..L1.x import``) are found.
    """
    blocks = log.split("Traceback (most recent call last):")
    root = str(set_dir.resolve())
    for block in reversed(blocks[1:]):
        found: list[str] = []
        for path in _FRAME_RE.findall(block):
            try:
                rel = Path(path).resolve().relative_to(root)
            except (ValueError, OSError):
                continue
            if len(rel.parts) == 2 and rel.parts[0] in {"L1", "L2", "L3", "L4"} and rel.suffix == ".py":
                key = f"{rel.parts[0]}:{rel.stem}"
                if key in found:
                    found.remove(key)
                found.append(key)
        if found:
            return list(reversed(found))  # innermost (last printed) first
    return []
