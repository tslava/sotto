"""Parse CHANGELOG.md (Keep a Changelog) for the /changelog command.

The only source of release notes is CHANGELOG.md at the repository root, with two
sections per version ("### For chat" and "### For owner"); "## [Unreleased]" is not a
version. Pure functions without I/O: the caller (commands.py) reads the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sotto.lang.ru import LEGACY_FOR_CHAT_HEADER, LEGACY_FOR_OWNER_HEADER

# Version heading: "## [0.5.0] — 2026-09-16"; the dash may be long ("—") or a plain
# hyphen ("-"), with or without spaces around it. "## [Unreleased]" does not match:
# it has no version number and no date.
_VERSION_HEADER_RE = re.compile(
    r"^##\s*\[(?P<version>\d+\.\d+\.\d+)\]\s*[—-]\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)
# Any second-level heading ("## ..."): it ends a version body, including
# "## [Unreleased]" if it is not first.
_ANY_H2_RE = re.compile(r"^##\s", re.MULTILINE)
# Third-level heading ("### ..."): it ends a block body inside a version.
_ANY_H3_RE = re.compile(r"^###\s", re.MULTILINE)
# Block headers: "### Notes" is the single block of the engine's own CHANGELOG.md (read as the
# owner block); "For the chat"/"For the owner" and the legacy Russian ones a pack's
# CHANGELOG.md may still use.
_FOR_CHAT_RE = re.compile(
    rf"^###\s*(?:For the chat|{re.escape(LEGACY_FOR_CHAT_HEADER)})\s*$", re.MULTILINE
)
_FOR_OWNER_RE = re.compile(
    rf"^###\s*(?:For the owner|Notes|{re.escape(LEGACY_FOR_OWNER_HEADER)})\s*$", re.MULTILINE
)
# The engine's own CHANGELOG.md (next to ``src``; copied into the image).
ENGINE_CHANGELOG_PATH = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
# Link footnotes at the bottom of the file ("[0.5.0]: https://..."). They follow the
# last version with no "##"/"###" heading between them and the last block, so they
# must be cut off explicitly rather than relying on the next heading as a boundary.
_LINK_REF_RE = re.compile(r"^\[[^\]]+\]:\s", re.MULTILINE)


@dataclass(frozen=True)
class Release:
    """One version from CHANGELOG.md. Block texts have no headings and are stripped."""

    version: str
    date: str
    for_chat: str
    for_owner: str


def _section_text(body: str, header_re: re.Pattern[str]) -> str:
    match = header_re.search(body)
    if match is None:
        return ""
    start = match.end()
    next_h3 = _ANY_H3_RE.search(body, start)
    end = next_h3.start() if next_h3 else len(body)
    return body[start:end].strip()


def parse_changelog(text: str) -> list[Release]:
    """Versions in file order (newest first).

    "## [Unreleased]" is not a version.
    """
    h2_starts = [m.start() for m in _ANY_H2_RE.finditer(text)]
    releases: list[Release] = []
    for match in _VERSION_HEADER_RE.finditer(text):
        body_start = match.end()
        next_h2 = next((pos for pos in h2_starts if pos > match.start()), None)
        body_end = next_h2 if next_h2 is not None else len(text)
        body = text[body_start:body_end]

        link_ref = _LINK_REF_RE.search(body)
        if link_ref is not None:
            body = body[: link_ref.start()]

        releases.append(
            Release(
                version=match.group("version"),
                date=match.group("date"),
                for_chat=_section_text(body, _FOR_CHAT_RE),
                for_owner=_section_text(body, _FOR_OWNER_RE),
            )
        )
    return releases


def latest_release(text: str) -> Release | None:
    releases = parse_changelog(text)
    return releases[0] if releases else None


def parse_chat_notes(text: str) -> list[Release]:
    """Release notes for the chat from a pack's ``chat_notes.md``, newest first.

    Format: ``## [0.9.1] - 2026-11-05`` then free text up to the next ``## `` heading. The text
    is the release's chat block (``for_chat``); ``for_owner`` stays empty, the owner block is the
    engine's. Same heading regex as ``CHANGELOG.md`` (hyphen or em dash).
    """
    h2_starts = [m.start() for m in _ANY_H2_RE.finditer(text)]
    releases: list[Release] = []
    for match in _VERSION_HEADER_RE.finditer(text):
        next_h2 = next((pos for pos in h2_starts if pos > match.start()), None)
        body = text[match.end() : next_h2 if next_h2 is not None else len(text)]
        releases.append(
            Release(
                version=match.group("version"),
                date=match.group("date"),
                for_chat=body.strip(),
                for_owner="",
            )
        )
    return releases


def chat_releases(chat_notes: str, legacy_changelog: str) -> list[Release]:
    """What ``/changelog`` and ``/changelog <version>`` show: the pack's ``chat_notes.md`` when it
    has releases, otherwise the chat blocks of the pack's legacy ``CHANGELOG.md`` (transition)."""
    return parse_chat_notes(chat_notes) or parse_changelog(legacy_changelog)


def engine_changelog_text() -> str:
    """The engine's own ``CHANGELOG.md`` ("" when the file is not there)."""
    try:
        return ENGINE_CHANGELOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def owner_releases(legacy_changelog: str, engine_changelog: str = "") -> list[Release]:
    """What ``/changelog owner`` shows: the owner blocks of the pack's own ``CHANGELOG.md`` when it
    has releases, otherwise those of the engine's ``CHANGELOG.md`` (a pack without a changelog)."""
    return parse_changelog(legacy_changelog) or parse_changelog(engine_changelog)
