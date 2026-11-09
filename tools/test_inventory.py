"""Test inventory (review aid, not a gate): per test function, the number of ``assert``
statements, ``pytest.raises`` uses and ``parametrize`` cases.

``python -m tools.test_inventory [TESTS_DIR] [--json]`` prints it;
``python -m tools.test_inventory --diff OLD.json NEW.json`` shows what changed between two
JSON inventories (attach that diff to a test-migration PR for the reviewer). Counts cannot
see an ``assert x == y`` degraded to ``assert True``; reviewers still read the diff.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _parametrize_call(decorator: ast.expr) -> ast.Call | None:
    if (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "parametrize"
    ):
        return decorator
    return None


def _case_count(call: ast.Call) -> int:
    if len(call.args) >= 2 and isinstance(call.args[1], (ast.List, ast.Tuple)):
        return len(call.args[1].elts)
    return 1


def _is_raises(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == "raises") or (
        isinstance(func, ast.Name) and func.id == "raises"
    )


def inventory_file(source: str) -> dict[str, dict[str, int]]:
    """``{test_name: {asserts, raises, parametrize, cases}}`` for one test module."""
    tree = ast.parse(source)
    result: dict[str, dict[str, int]] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}::")
            elif isinstance(node, _FUNCTIONS) and node.name.startswith("test"):
                calls = [
                    call
                    for decorator in node.decorator_list
                    if (call := _parametrize_call(decorator)) is not None
                ]
                cases = 1
                for call in calls:
                    cases *= _case_count(call)
                result[prefix + node.name] = {
                    "asserts": sum(isinstance(n, ast.Assert) for n in ast.walk(node)),
                    "raises": sum(_is_raises(n) for n in ast.walk(node)),
                    "parametrize": len(calls),
                    "cases": cases,
                }

    visit(tree.body, "")
    return result


def inventory(tests_dir: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for path in sorted(tests_dir.rglob("test_*.py")):
        for name, counts in inventory_file(path.read_text(encoding="utf-8")).items():
            result[f"{path.relative_to(tests_dir).as_posix()}::{name}"] = counts
    return result


def diff(old: dict[str, dict[str, int]], new: dict[str, dict[str, int]]) -> list[str]:
    """Human-readable lines: removed, added and changed tests."""
    lines: list[str] = []
    for node in sorted(old.keys() - new.keys()):
        lines.append(f"- removed {node}")
    for node in sorted(new.keys() - old.keys()):
        lines.append(f"+ added   {node}")
    for node in sorted(old.keys() & new.keys()):
        if old[node] != new[node]:
            lines.append(f"~ changed {node}: {old[node]} -> {new[node]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.test_inventory")
    parser.add_argument("tests_dir", nargs="?", type=Path, default=Path("tests"))
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("OLD", "NEW"))
    args = parser.parse_args(argv)
    if args.diff:
        old = json.loads(args.diff[0].read_text(encoding="utf-8"))
        new = json.loads(args.diff[1].read_text(encoding="utf-8"))
        lines = diff(old, new)
        print("\n".join(lines) if lines else "no differences")
        return 0
    inv = inventory(args.tests_dir)
    if args.json:
        json.dump(inv, sys.stdout, indent=1, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    for node, counts in inv.items():
        print(node, counts)
    print(f"TOTAL tests={len(inv)} asserts={sum(c['asserts'] for c in inv.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
