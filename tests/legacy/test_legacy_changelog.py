"""Parsing of the legacy Keep-a-Changelog CHANGELOG.md (Russian block headers).

The legacy format keys on Russian block headers ("### Для чата" / "### Для владельца"); the
real legacy pack CHANGELOG.md is the control. Migration proof: removed together with
``personas/_legacy_ru`` in PR-9. The format-independent chat-notes tests are in
tests/test_changelog.py.
"""

from __future__ import annotations

import sotto
from sotto.changelog import (
    Release,
    chat_releases,
    latest_release,
    owner_releases,
    parse_changelog,
    parse_chat_notes,
)
from tests.packs import LEGACY_PACK

REAL_CHANGELOG = (LEGACY_PACK / "CHANGELOG.md").read_text(encoding="utf-8")


def test_sotto_version_is_0_9_1() -> None:
    assert sotto.__version__ == "0.9.1"


def test_parse_real_changelog_latest_is_0_9_1_with_date_and_nonempty_blocks() -> None:
    releases = parse_changelog(REAL_CHANGELOG)

    assert releases[0].version == "0.9.1"
    assert releases[0].date == "2026-11-05"
    assert releases[0].for_chat.strip() != ""
    assert releases[0].for_owner.strip() != ""


def test_parse_real_changelog_order_is_newest_first() -> None:
    releases = parse_changelog(REAL_CHANGELOG)

    versions = [r.version for r in releases]
    assert versions == [
        "0.9.1",
        "0.9.0",
        "0.8.1",
        "0.8.0",
        "0.7.0",
        "0.6.2",
        "0.6.1",
        "0.6.0",
        "0.5.0",
        "0.4.0",
        "0.3.0",
        "0.2.0",
        "0.1.0",
    ]


def test_parse_real_changelog_skips_unreleased() -> None:
    releases = parse_changelog(REAL_CHANGELOG)

    assert "Unreleased" not in [r.version for r in releases]
    assert all(r.version != "" for r in releases)


def test_latest_release_matches_first_of_parse() -> None:
    releases = parse_changelog(REAL_CHANGELOG)

    assert latest_release(REAL_CHANGELOG) == releases[0]


def test_latest_release_none_for_empty_text() -> None:
    assert latest_release("") is None


# PR-8: the legacy parser still keys on the Russian block headers.
def test_parse_changelog_accepts_em_dash_and_hyphen() -> None:
    text = (
        "## [Unreleased]\n\n"
        "### Для чата\n\n### Для владельца\n\n"
        "## [1.2.0] — 2026-01-02\n\n"
        "### Для чата\n\nEm dash.\n\n### Для владельца\n\nEm dash owner.\n\n"
        "## [1.1.0] - 2026-01-01\n\n"
        "### Для чата\n\nHyphen.\n\n### Для владельца\n\nHyphen owner.\n"
    )

    releases = parse_changelog(text)

    assert [r.version for r in releases] == ["1.2.0", "1.1.0"]
    assert releases[0].for_chat == "Em dash."
    assert releases[1].for_chat == "Hyphen."


def test_parse_changelog_release_missing_blocks_returns_empty_strings() -> None:
    text = "## [1.0.0] — 2026-01-01\n\nNo sections at all.\n"

    releases = parse_changelog(text)

    assert releases == [Release(version="1.0.0", date="2026-01-01", for_chat="", for_owner="")]


def test_parse_changelog_strips_trailing_link_references() -> None:
    text = (
        "## [1.0.0] — 2026-01-01\n\n"
        "### Для чата\n\nText.\n\n"
        "### Для владельца\n\nDescription.\n\n"
        "[1.0.0]: https://example.com/releases/tag/v1.0.0\n"
    )

    releases = parse_changelog(text)

    assert releases[0].for_owner == "Description."
    assert "https://example.com" not in releases[0].for_owner


def test_parse_changelog_ignores_unreleased_entirely() -> None:
    text = (
        "## [Unreleased]\n\n"
        "### Для чата\n\nMust not reach the result.\n\n"
        "### Для владельца\n\nNeither must this.\n\n"
        "## [1.0.0] — 2026-01-01\n\n"
        "### Для чата\n\nRelease.\n\n### Для владельца\n\nRelease owner.\n"
    )

    releases = parse_changelog(text)

    assert len(releases) == 1
    assert releases[0].version == "1.0.0"
    assert "Must not" not in releases[0].for_chat


NOTES = """## [1.2.0] - 2026-02-01

First line.
Second line.

## [1.1.0] — 2026-01-01

Only one.

## [1.0.0] - 2025-12-01

## [Unreleased]

Not a release.
"""


def test_parse_chat_notes_reads_headings_and_free_text() -> None:
    assert parse_chat_notes(NOTES) == [
        Release("1.2.0", "2026-02-01", "First line.\nSecond line.", ""),
        Release("1.1.0", "2026-01-01", "Only one.", ""),
        Release("1.0.0", "2025-12-01", "", ""),
    ]


def test_parse_chat_notes_of_nothing_is_empty() -> None:
    assert parse_chat_notes("") == []
    assert parse_chat_notes("no headings here") == []


def test_chat_releases_prefer_notes_and_fall_back_to_the_legacy_changelog() -> None:
    assert chat_releases(NOTES, REAL_CHANGELOG)[0].version == "1.2.0"
    assert chat_releases("", REAL_CHANGELOG) == parse_changelog(REAL_CHANGELOG)
    assert chat_releases("", "") == []


def test_owner_releases_come_from_the_legacy_changelog() -> None:
    assert owner_releases(REAL_CHANGELOG) == parse_changelog(REAL_CHANGELOG)
    assert owner_releases("") == []


def test_legacy_pack_chat_notes_equal_the_chat_blocks_of_its_changelog() -> None:
    notes = (LEGACY_PACK / "chat_notes.md").read_text(encoding="utf-8")
    from_notes = [(r.version, r.date, r.for_chat) for r in parse_chat_notes(notes)]
    from_changelog = [(r.version, r.date, r.for_chat) for r in parse_changelog(REAL_CHANGELOG)]
    assert from_notes == from_changelog
    assert from_notes
