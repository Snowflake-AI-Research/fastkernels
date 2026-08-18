"""Inline used symbols from local baseline imports.

Given a baseline task file, collect the definitions (and their local
baseline dependencies) that the file actually uses, so those imports can
be replaced. External packages and ``fastkernels.infra`` stay as imports.

    python -m fastkernels.utils.inline_baseline path/to/baseline.py
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from fastkernels import BASELINE_DIR, KB_ROOT

_SKIP = set(dir(__builtins__)) | {"self", "cls"}


def _is_baseline(path: Path) -> bool:
    try:
        path.resolve().relative_to(BASELINE_DIR)
    except ValueError:
        return False
    return path.suffix == ".py"


def _mod_parts(path: Path) -> list[str]:
    rel = path.resolve().relative_to(KB_ROOT).with_suffix("")
    return [KB_ROOT.name, *rel.parts]


def _resolve_module(filepath: Path, level: int, module: str | None) -> Path | None:
    if level == 0:
        if not module or not module.startswith("fastkernels."):
            return None
        cand = (KB_ROOT.parent / module.replace(".", "/")).with_suffix(".py")
        return cand if cand.is_file() else None
    base = filepath.parent
    for _ in range(level - 1):
        base = base.parent
    if module:
        cand = (base / module.replace(".", "/")).with_suffix(".py")
        return cand if cand.is_file() else None
    return None


def _abs_module(filepath: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    pkg = _mod_parts(filepath)[:-1]
    for _ in range(node.level - 1):
        if not pkg:
            return node.module
        pkg = pkg[:-1]
    extra = node.module.split(".") if node.module else []
    return ".".join(pkg + extra) or None


def _used_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in _SKIP:
                out.add(child.id)
    return out


@lru_cache(maxsize=None)
def _parse(path: Path) -> tuple[str, ast.Module]:
    path = path.resolve()
    src = path.read_text()
    return src, ast.parse(src, filename=str(path))


def _bindings(path: Path) -> tuple[dict[str, ast.stmt], dict[str, tuple[Path | None, str, ast.stmt]]]:
    src, tree = _parse(path)
    defs: dict[str, ast.stmt] = {}
    imps: dict[str, tuple[Path | None, str, ast.stmt]] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defs[node.target.id] = node
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            dest = _resolve_module(path, node.level, node.module)
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "*":
                    imps[local] = (dest if dest and _is_baseline(dest) else None, "*", node)
                    continue
                if dest is None and node.module is None:
                    dest = _resolve_module(path, node.level, alias.name)
                imps[local] = (
                    dest if dest and _is_baseline(dest) else None,
                    alias.name,
                    node,
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imps[alias.asname or alias.name.split(".")[0]] = (None, alias.name, node)
    return defs, imps


def _seed(path: Path) -> list[tuple[Path, str]]:
    src, tree = _parse(path)
    defs, imps = _bindings(path)
    used = _used_names(tree) - set(defs)
    out: list[tuple[Path, str]] = []
    for name in used:
        rec = imps.get(name)
        if rec and rec[0] is not None:
            if rec[1] == "*":
                dest_defs, _ = _bindings(rec[0])
                if name in dest_defs:
                    out.append((rec[0], name))
            else:
                out.append((rec[0], rec[1]))
    out.extend(_follow_inner_imports(path, tree))
    return out


def _follow_inner_imports(path: Path, stmt: ast.AST) -> list[tuple[Path, str]]:
    extra: list[tuple[Path, str]] = []
    for node in ast.walk(stmt):
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue
        dest = _resolve_module(path, node.level, node.module)
        if dest is None or not _is_baseline(dest):
            continue
        for alias in node.names:
            if alias.name != "*":
                extra.append((dest, alias.name))
    return extra


def baseline_inline_source(path: Path | str) -> str:
    """Source to add to *path* so local baseline imports can be dropped."""
    root = Path(path).resolve()
    work = _seed(root)
    taken: dict[tuple[Path, str], ast.stmt] = {}
    needs: dict[tuple[Path, str], set[tuple[Path, str]]] = {}
    extras: dict[str, ast.stmt] = {}

    while work:
        cur, name = work.pop()
        key = (cur, name)
        if key in taken or cur == root:
            continue
        defs, imps = _bindings(cur)
        if name not in defs:
            rec = imps.get(name)
            if rec and rec[0] is not None and rec[1] not in ("*", name):
                work.append((rec[0], rec[1]))
            continue
        stmt = defs[name]
        taken[key] = stmt
        deps: set[tuple[Path, str]] = set()
        for n in _used_names(stmt):
            if n in defs and n != name:
                work.append((cur, n))
                deps.add((cur, n))
            elif n in imps:
                dest, exported, inode = imps[n]
                if dest is not None:
                    exp = n if exported == "*" else exported
                    work.append((dest, exp))
                    deps.add((dest, exp))
                else:
                    line = ast.unparse(_rewrite_import(cur, inode))
                    if not line.startswith("from __future__"):
                        extras[line] = inode
        for dep in _follow_inner_imports(cur, stmt):
            work.append(dep)
            deps.add(dep)
        needs[key] = deps

    emitted: set[tuple[Path, str]] = set()
    chunks: list[str] = []

    def emit(key: tuple[Path, str]) -> None:
        if key in emitted or key not in taken:
            return
        emitted.add(key)
        for dep in needs.get(key, ()):
            emit(dep)
        src, _ = _parse(key[0])
        piece = ast.get_source_segment(src, taken[key]) or ast.unparse(taken[key])
        chunks.append(_rewrite_stmt_imports(key[0], piece))

    for key in taken:
        emit(key)

    header = "\n".join(sorted(extras))
    body = "\n\n".join(chunks)
    return "\n\n".join(p for p in (header, body) if p).rstrip() + ("\n" if header or body else "")


def _import_is_baseline(path: Path, node: ast.ImportFrom) -> bool:
    dest = _resolve_module(path, node.level, node.module)
    if dest is not None:
        return _is_baseline(dest)
    if node.module is None:
        return all(
            (c := _resolve_module(path, node.level, a.name)) is not None and _is_baseline(c)
            for a in node.names
        )
    return False


def _rewrite_stmt_imports(path: Path, text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, int, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue
        dest = _resolve_module(path, node.level, node.module)
        abs_baseline = (
            dest is not None
            and _is_baseline(dest)
            and node.level == 0
            and (node.module or "").startswith("fastkernels.tasks.baseline")
        )
        if node.level == 0 and not abs_baseline:
            continue
        indent = lines[node.lineno - 1][: len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
        if _import_is_baseline(path, node) or abs_baseline:
            edits.append((node.lineno, node.end_lineno or node.lineno, None))
        elif node.level:
            edits.append(
                (node.lineno, node.end_lineno or node.lineno, indent + ast.unparse(_rewrite_import(path, node)) + "\n")
            )
    if not edits:
        return text
    drop: dict[int, str | None] = {}
    for start, end, repl in edits:
        for i in range(start, end + 1):
            drop[i] = repl if i == start else None
    out: list[str] = []
    for i, line in enumerate(lines, 1):
        if i not in drop:
            out.append(line)
        elif drop[i] is not None:
            out.append(drop[i])
    return "".join(out)


def _rewrite_import(path: Path, node: ast.stmt) -> ast.stmt:
    if not isinstance(node, ast.ImportFrom) or node.level == 0:
        return node
    mod = _abs_module(path, node)
    return ast.ImportFrom(module=mod, names=node.names, level=0)


def strip_baseline_imports(source: str, path: Path | str) -> str:
    """Drop imports that resolve to other baseline task files."""
    return _rewrite_stmt_imports(Path(path).resolve(), source)


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    print(baseline_inline_source(p), end="")
