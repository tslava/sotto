"""A pack's chat_notes.md (release notes for the chat) and the version string.

The legacy Keep-a-Changelog format is covered in ``tests/legacy/test_legacy_changelog.py``.
"""

from __future__ import annotations

import sotto
from sotto.changelog import Release, chat_releases, latest_release, owner_releases, parse_chat_notes
from tests.packs import DEMO_PACK

NOTES = """## [1.2.0] - 2026-02-01

First line.
Second line.

## [1.1.0] — 2026-01-01

Only one.

## [1.0.0] - 2025-12-01

## [Unreleased]

Not a release.
"""


def test_sotto_version_is_0_9_1() -> None:
    assert sotto.__version__ == "0.9.1"


def test_latest_release_none_for_empty_text() -> None:
    assert latest_release("") is None


def test_parse_chat_notes_reads_headings_and_free_text() -> None:
    assert parse_chat_notes(NOTES) == [
        Release("1.2.0", "2026-02-01", "First line.\nSecond line.", ""),
        Release("1.1.0", "2026-01-01", "Only one.", ""),
        Release("1.0.0", "2025-12-01", "", ""),
    ]


def test_parse_chat_notes_of_nothing_is_empty() -> None:
    assert parse_chat_notes("") == []
    assert parse_chat_notes("no headings here") == []


def test_chat_releases_prefer_the_notes() -> None:
    assert chat_releases(NOTES, "")[0].version == "1.2.0"
    assert chat_releases("", "") == []


def test_owner_releases_of_nothing_is_empty() -> None:
    assert owner_releases("") == []


def test_demo_pack_chat_notes_parse() -> None:
    notes = (DEMO_PACK / "chat_notes.md").read_text(encoding="utf-8")
    releases = parse_chat_notes(notes)
    assert releases
    assert all(r.version and r.date for r in releases)
