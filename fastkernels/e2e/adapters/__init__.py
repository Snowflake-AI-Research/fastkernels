"""Model-family adapters for ``fastkernels e2e`` (see ``base.py`` for the contract).

Each module in this package defines one ``Adapter`` subclass; they are discovered
automatically, so adding a family never touches a shared file.
"""

from __future__ import annotations

import importlib
import pkgutil

from .base import Adapter, RunSpec, Timing

__all__ = ["Adapter", "RunSpec", "Timing", "all_adapters", "adapter_for", "adapter_by_name"]


def all_adapters() -> list[type[Adapter]]:
    found = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        found += [v for v in vars(mod).values()
                  if isinstance(v, type) and issubclass(v, Adapter) and v is not Adapter
                  and v.__module__ == mod.__name__]
    return found


def adapter_for(scenario) -> type[Adapter]:
    matches = [a for a in all_adapters() if a.handles(scenario)]
    if len(matches) != 1:
        raise LookupError(f"{scenario.hf_name}: expected exactly one adapter, found "
                          f"{[a.name for a in matches] or 'none'}")
    return matches[0]


def adapter_by_name(name: str) -> type[Adapter]:
    for a in all_adapters():
        if a.name == name:
            return a
    raise LookupError(f"no adapter named {name!r}")
