"""Cyrillic inventory of a source tree (review aid for the stage 2/3 refactoring, not a gate).

``python -m tools.inventory [ROOT] [--json]`` walks ``ROOT/*.py`` (default ``src/trolobot``)
and counts, per file, string constants with Cyrillic by role: docstrings, comments, and the
other literals classified by where they sit (``field_desc``, ``cli_help``, ``log``, ``exc``,
``const:UPPER``, ``const:var``, ``field_default``, ``inline_in_func``, ``other``).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

CYRILLIC = re.compile("[Ѐ-ӿ]")
_LOG_METHODS = frozenset({"info", "warning", "error", "exception", "debug"})
_DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _has_cyrillic(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and bool(CYRILLIC.search(node.value))
    )


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOC_OWNERS) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _classify(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.keyword) and current.arg == "description":
            return "field_desc"
        if isinstance(current, ast.keyword) and current.arg == "help":
            return "cli_help"
        if isinstance(current, ast.Call):
            name = _call_name(current)
            if name in _LOG_METHODS:
                return "log"
            if name.endswith("Error") or name == "Exception":
                return "exc"
        if isinstance(current, ast.Raise):
            return "exc"
        if (
            isinstance(current, ast.Assign)
            and current.targets
            and isinstance(current.targets[0], ast.Name)
        ):
            upper = current.targets[0].id.lstrip("_").isupper()
            return "const:UPPER" if upper else "const:var"
        if isinstance(current, ast.AnnAssign) and isinstance(current.target, ast.Name):
            return "field_default"
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "inline_in_func"
    return "other"


def inventory_file(source: str) -> Counter[str]:
    """Counters for one module: ``doc``, ``cmt``, ``lit`` and the per-role classes."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    docstrings = _docstring_ids(tree)
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not _has_cyrillic(node):
            continue
        if id(node) in docstrings:
            counts["doc"] += 1
        else:
            counts["lit"] += 1
            counts["role:" + _classify(node, parents)] += 1
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and CYRILLIC.search(token.string):
            counts["cmt"] += 1
    return counts


def inventory(root: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for path in sorted(root.glob("*.py")):
        counts = inventory_file(path.read_text(encoding="utf-8"))
        if counts:
            result[path.name] = dict(sorted(counts.items()))
    return result


def totals(per_file: dict[str, dict[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for counts in per_file.values():
        total.update(counts)
    return dict(sorted(total.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.inventory")
    parser.add_argument("root", nargs="?", type=Path, default=Path("src/trolobot"))
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    per_file = inventory(args.root)
    if args.json:
        json.dump({"files": per_file, "total": totals(per_file)}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for name, counts in per_file.items():
        print(name, counts)
    print("TOTAL", totals(per_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
