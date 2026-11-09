"""CLI: ``python -m tools.golden record|compare ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.golden import runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.golden")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="run all scenarios and write canonical JSON files")
    record.add_argument("--pack", type=Path, required=True)
    record.add_argument("--scenarios", type=Path, required=True)
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--only", action="append", default=[], help="scenario id (repeatable)")

    compare = sub.add_parser("compare", help="run all scenarios and diff against a baseline")
    compare.add_argument("--pack", type=Path, required=True)
    compare.add_argument("--scenarios", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--allow", type=Path, default=None, help="ALLOWED_DIFFS.md")
    compare.add_argument("--only", action="append", default=[], help="scenario id (repeatable)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record":
        return runner.record(args.pack, args.scenarios, args.out, only=args.only)
    return runner.compare(
        args.pack, args.scenarios, args.baseline, allow=args.allow, only=args.only
    )


if __name__ == "__main__":
    sys.exit(main())
