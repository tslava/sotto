"""Coverage gate (plan 3.6): total >= 93.0 % over ``src/`` and >= 95.0 % per new module.

Reads the JSON report of ``coverage json`` (``coverage.json`` by default). No per-module
comparison against a baseline: modules move and split. The total counts only files under
``src/``; the floor on new modules also covers ``tools/``. A listed module that exists on
disk but is missing from the report counts as a failure; one that does not exist yet (a
future ``src`` module) is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOTAL_FLOOR = 93.0
NEW_MODULE_FLOOR = 95.0

# Modules held to NEW_MODULE_FLOOR. Extend as stage 2 adds modules (PR-1 renames the package:
# the ``src/sotto`` entries below are the future names and are skipped until they exist).
NEW_MODULES: tuple[str, ...] = (
    "tools/golden/__main__.py",
    "tools/golden/adapter.py",
    "tools/golden/canonical.py",
    "tools/golden/runner.py",
    "tools/golden/schema.py",
    "tools/coverage_gate.py",
    "tools/check_language.py",
    "tools/extract_fragments.py",
    "tools/extract_filter_defaults.py",
    "tools/extract_config_keys.py",
    "tools/extract_locale.py",
    "tools/split_changelog.py",
    "tools/inventory.py",
    "tools/test_inventory.py",
    "src/sotto/persona.py",
    "src/sotto/fragments.py",
    "src/sotto/i18n.py",
    "src/sotto/errors.py",
    "src/sotto/packcheck.py",
    "src/sotto/wiring.py",
    "src/sotto/lang/__init__.py",
    "src/sotto/lang/base.py",
    "src/sotto/lang/en.py",
    "src/sotto/lang/ru.py",
)


def _percent(summary: dict[str, Any]) -> float:
    statements = int(summary["num_statements"])
    if statements == 0:
        return 100.0
    return 100.0 * (statements - int(summary["missing_lines"])) / statements


def check(
    report: dict[str, Any],
    *,
    repo: Path = Path("."),
    total_floor: float = TOTAL_FLOOR,
    module_floor: float = NEW_MODULE_FLOOR,
    modules: tuple[str, ...] | None = None,
) -> list[str]:
    """Problems found in a coverage JSON report; empty means the gate passes."""
    modules = NEW_MODULES if modules is None else modules
    files: dict[str, Any] = {Path(name).as_posix(): data for name, data in report["files"].items()}
    problems: list[str] = []

    statements = missing = 0
    for name, data in files.items():
        if name.startswith("src/"):
            statements += data["summary"]["num_statements"]
            missing += data["summary"]["missing_lines"]
    total = 100.0 * (statements - missing) / statements if statements else 100.0
    if total < total_floor:
        problems.append(f"total coverage of src/ is {total:.2f} %, floor is {total_floor:.1f} %")

    for module in modules:
        data = files.get(module)
        if data is None:
            if (repo / module).exists():
                problems.append(f"{module}: on disk but absent from the coverage report")
            continue
        percent = _percent(data["summary"])
        if percent < module_floor:
            problems.append(f"{module}: {percent:.2f} % < {module_floor:.1f} %")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.coverage_gate")
    parser.add_argument("report", nargs="?", type=Path, default=Path("coverage.json"))
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    problems = check(report)
    for problem in problems:
        print(f"COVERAGE GATE: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("coverage gate OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
