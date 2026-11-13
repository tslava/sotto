"""A pack's chat_notes.md (release notes for the chat) and the version string.

The legacy Keep-a-Changelog format is covered in ``tests/legacy/test_legacy_changelog.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sotto
from sotto import changelog
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


ENGLISH = """\
## [Unreleased]

## [2.0.0] - 2026-02-03

### For the chat

Chat text.

### For the owner

Owner text.
"""


def test_english_block_headers_are_parsed() -> None:
    release = latest_release(ENGLISH)
    assert release is not None
    assert (release.for_chat, release.for_owner) == ("Chat text.", "Owner text.")


def test_owner_releases_fall_back_to_the_engine_changelog() -> None:
    assert owner_releases("", ENGLISH)[0].version == "2.0.0"
    # A pack changelog with releases wins over the engine's.
    pack = ENGLISH.replace("2.0.0", "1.0.0")
    assert owner_releases(pack, ENGLISH)[0].version == "1.0.0"
    assert owner_releases("", "") == []


def test_engine_changelog_text_reads_the_path_and_survives_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(ENGLISH, encoding="utf-8")
    monkeypatch.setattr(changelog, "ENGINE_CHANGELOG_PATH", path)
    assert changelog.engine_changelog_text() == ENGLISH
    monkeypatch.setattr(changelog, "ENGINE_CHANGELOG_PATH", tmp_path / "missing.md")
    assert changelog.engine_changelog_text() == ""


NOTES_ONLY = """\
## [Unreleased]

## [0.9.1] — 2026-11-05

### Notes

- Vocabulary offers use eight words.
- Fixed the stem of words ending in -mya.

## [0.9.0] — 2026-11-01

### Notes

- Alerts to the owner.
"""


def test_notes_block_is_the_owner_block() -> None:
    releases = changelog.parse_changelog(NOTES_ONLY)
    assert [r.version for r in releases] == ["0.9.1", "0.9.0"]
    assert releases[0].for_owner.startswith("- Vocabulary offers")
    assert releases[0].for_chat == ""
    assert owner_releases("", NOTES_ONLY)[1].for_owner == "- Alerts to the owner."
