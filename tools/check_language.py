"""Language gate: nothing outside the allowed paths may contain Cyrillic.

Prints the number of Cyrillic characters per tracked file outside the allowed paths and a
total. Exit code is 0 by default; ``--enforce`` makes a non-empty report fail (CI uses it).
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

CYRILLIC = re.compile("[\u0400-\u04ff]")

# Paths where Cyrillic is legitimate (fnmatch patterns on posix paths from the repo root).
ALLOWED_PATHS: tuple[str, ...] = (
    "src/sotto/lang/ru.py",
    "tests/lang/*",
    "tests/fixtures/*",
    "tests/test_ast_equiv.py",  # self-test fixtures: Russian sample code
    "tools/ast_equiv.py",
)


def is_allowed(path: str, allowed: tuple[str, ...] = ALLOWED_PATHS) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in allowed)


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"], check=True, capture_output=True
    ).stdout
    return [name for name in out.decode().split("\0") if name]


def count_cyrillic(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    return len(CYRILLIC.findall(text))


def report(repo: Path, files: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in files:
        if is_allowed(name):
            continue
        n = count_cyrillic(repo / name)
        if n:
            counts[name] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.check_language")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--enforce", action="store_true", help="exit 1 when anything is found")
    args = parser.parse_args(argv)
    counts = report(args.repo, tracked_files(args.repo))
    for name, n in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{n:7d}  {name}")
    total = sum(counts.values())
    print(f"language gate (report-only): {len(counts)} files, {total} Cyrillic chars")
    return 1 if (args.enforce and counts) else 0


if __name__ == "__main__":
    sys.exit(main())
