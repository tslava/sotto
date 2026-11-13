"""Split the chat blocks of a pack's ``CHANGELOG.md`` into its ``chat_notes.md`` (plan 04, PR-5a).

The pre-split changelog has two blocks per release, "### Для чата" (what the owner posts in the
chat) and "### Для владельца". The pack keeps only the chat text, one ``## [X.Y.Z] - DATE``
section per release, which is what ``/changelog`` shows. The output is verified by parsing it back
with the engine's own parser: every release, date and chat block must survive byte for byte (after
the ``strip()`` both parsers apply). Texts are never printed.

Usage::

    python -m tools.split_changelog CHANGELOG.md --out chat_notes.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sotto.changelog import parse_changelog, parse_chat_notes


class SplitError(ValueError):
    """The changelog cannot be split without losing or altering a chat block."""


def split_changelog(text: str) -> str:
    """The ``chat_notes.md`` text for a ``CHANGELOG.md`` text (newest release first)."""
    releases = parse_changelog(text)
    if not releases:
        raise SplitError("no releases found in the changelog")
    sections: list[str] = []
    for release in releases:
        sections.append(f"## [{release.version}] - {release.date}\n\n{release.for_chat}".rstrip())
    notes = "\n\n".join(sections) + "\n"
    parsed = parse_chat_notes(notes)
    expected = [(r.version, r.date, r.for_chat) for r in releases]
    if [(r.version, r.date, r.for_chat) for r in parsed] != expected:
        raise SplitError("chat_notes.md does not parse back to the same chat blocks")
    return notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.split_changelog", description=__doc__)
    parser.add_argument("changelog", type=Path, help="the pack's CHANGELOG.md")
    parser.add_argument("--out", type=Path, required=True, help="chat_notes.md to write")
    args = parser.parse_args(argv)
    try:
        notes = split_changelog(args.changelog.read_text(encoding="utf-8"))
    except (OSError, SplitError) as exc:
        print(f"split_changelog: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(notes, encoding="utf-8")
    print(f"wrote {args.out} ({len(parse_chat_notes(notes))} releases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
