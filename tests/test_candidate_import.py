"""Candidate import fallback and --standalone isolation."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from fastkernels.list import install_candidate_finder

_DEP = "fastkernels.tasks.baseline.L1._fk_test_dep"
_CAND_DEP = "fastkernels.tasks.candidate.L1._fk_test_dep"
_CAND_PARENT = "fastkernels.tasks.candidate.L3._fk_test_parent"


def _purge() -> None:
    for name in list(sys.modules):
        if name.startswith("fastkernels.tasks.candidate"):
            del sys.modules[name]
        elif name == _DEP:
            del sys.modules[name]


def _seed_baseline_dep(marker: str = "baseline") -> type:
    mod = types.ModuleType(_DEP)
    class Dep:
        pass
    Dep.marker = marker
    mod.Dep = Dep
    sys.modules[_DEP] = mod
    return Dep


@pytest.fixture
def cand_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("fastkernels.CANDIDATE_DIR", tmp_path)
    _purge()
    install_candidate_finder(standalone=False)
    yield tmp_path
    _purge()
    install_candidate_finder(standalone=False)


def _write(cand_dir: Path, rel: str, src: str) -> None:
    path = cand_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)


def test_missing_candidate_aliases_baseline(cand_dir: Path) -> None:
    base_cls = _seed_baseline_dep()
    mod = importlib.import_module(_CAND_DEP)
    assert mod.Dep is base_cls


def test_relative_import_falls_back_to_baseline(cand_dir: Path) -> None:
    base_cls = _seed_baseline_dep()
    _write(cand_dir, "L3/_fk_test_parent.py",
           "from ..L1._fk_test_dep import Dep\nmarker = Dep.marker\n")
    mod = importlib.import_module(_CAND_PARENT)
    assert mod.Dep is base_cls
    assert mod.marker == "baseline"


def test_relative_import_uses_candidate_when_present(cand_dir: Path) -> None:
    _seed_baseline_dep()
    _write(cand_dir, "L1/_fk_test_dep.py",
           "class Dep:\n    marker = 'candidate'\n")
    _write(cand_dir, "L3/_fk_test_parent.py",
           "from ..L1._fk_test_dep import Dep\nmarker = Dep.marker\n")
    mod = importlib.import_module(_CAND_PARENT)
    assert mod.marker == "candidate"


def test_standalone_ignores_other_candidates(cand_dir: Path) -> None:
    base_cls = _seed_baseline_dep()
    _write(cand_dir, "L1/_fk_test_dep.py",
           "class Dep:\n    marker = 'candidate'\n")
    _write(cand_dir, "L3/_fk_test_parent.py",
           "from ..L1._fk_test_dep import Dep\nmarker = Dep.marker\n")
    install_candidate_finder(standalone=True, keep={_CAND_PARENT})
    mod = importlib.import_module(_CAND_PARENT)
    assert mod.Dep is base_cls
    assert mod.marker == "baseline"
