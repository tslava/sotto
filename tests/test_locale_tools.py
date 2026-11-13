"""tools/split_changelog.py and tools/extract_locale.py (plan 04, PR-5a)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tools import extract_locale, split_changelog

from sotto.changelog import parse_changelog, parse_chat_notes
from tests.packs import LEGACY_PACK

CHANGELOG = """# Changelog

## [Unreleased]

### Для чата

- Not yet.

## [1.1.0] — 2026-01-01

### Для чата

- Chat line one.
- Chat line two.

### Для владельца

- Owner line.

## [1.0.0] — 2025-12-01

### Для владельца

- Only owner text.

[1.1.0]: https://example.invalid/1.1.0
"""


# -- split_changelog -----------------------------------------------------------------


def test_split_changelog_keeps_every_release_and_chat_block() -> None:
    notes = split_changelog.split_changelog(CHANGELOG)
    assert notes == (
        "## [1.1.0] - 2026-01-01\n\n- Chat line one.\n- Chat line two.\n\n## [1.0.0] - 2025-12-01\n"
    )
    assert [(r.version, r.for_chat) for r in parse_chat_notes(notes)] == [
        ("1.1.0", "- Chat line one.\n- Chat line two."),
        ("1.0.0", ""),
    ]


def test_split_changelog_rejects_a_changelog_without_releases() -> None:
    with pytest.raises(split_changelog.SplitError, match="no releases"):
        split_changelog.split_changelog("# nothing")


def test_split_changelog_fails_when_the_notes_do_not_parse_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(split_changelog, "parse_chat_notes", lambda text: [])
    with pytest.raises(split_changelog.SplitError, match="parse back"):
        split_changelog.split_changelog(CHANGELOG)


def test_split_changelog_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "CHANGELOG.md"
    source.write_text(CHANGELOG, encoding="utf-8")
    out = tmp_path / "chat_notes.md"
    assert split_changelog.main([str(source), "--out", str(out)]) == 0
    assert "2 releases" in capsys.readouterr().out
    assert len(parse_chat_notes(out.read_text(encoding="utf-8"))) == 2
    assert split_changelog.main([str(tmp_path / "missing.md"), "--out", str(out)]) == 1
    assert "split_changelog:" in capsys.readouterr().err


def test_legacy_pack_chat_notes_are_what_the_tool_produces() -> None:
    text = (LEGACY_PACK / "CHANGELOG.md").read_text(encoding="utf-8")
    assert split_changelog.split_changelog(text) == (LEGACY_PACK / "chat_notes.md").read_text(
        encoding="utf-8"
    )
    assert len(parse_changelog(text)) == 13


# -- extract_locale ------------------------------------------------------------------

SOURCE = '''
"""Модульная строка (docstring не считается)."""
import logging

logger = logging.getLogger(__name__)

HEADER = "Привет всем"


async def reply(message, count, name):
    """Строка документации тоже не считается."""
    logger.info("лог не считается: %s", name)
    if count:
        return f"Всего {count}, звали {name.title}, ещё {len(name)} {'да' if count else 'нет'}."
    return "Пусто."


def english_only():
    return "No Russian here"
'''


def test_extract_literals_skips_docstrings_logging_and_non_cyrillic() -> None:
    found = extract_locale.extract_literals(SOURCE)
    assert [(lit.function, lit.template) for lit in found] == [
        ("<module>", "Привет всем"),
        ("reply", "Всего {}, звали {}, ещё {} {}."),
        ("reply", "да"),
        ("reply", "нет"),
        ("reply", "Пусто."),
    ]
    assert found[1].names == ("count", "title", "len", "expr4")


def test_draft_names_placeholders_after_the_expressions() -> None:
    draft = extract_locale.draft_catalog(SOURCE, "demo")
    assert draft["demo"]["reply_1"] == "Всего {count}, звали {title}, ещё {len} {expr4}."
    assert draft["demo"]["reply_4"] == "Пусто."


def _catalog(tmp_path: Path, data: dict[str, dict[str, object]]) -> Path:
    directory = tmp_path / "ru"
    directory.mkdir()
    for name, body in data.items():
        (directory / f"{name}.yaml").write_text(
            yaml.safe_dump({name: body}, allow_unicode=True), encoding="utf-8"
        )
    return directory


GOOD = {
    "t": {
        "header": "Привет всем",
        "total": "Всего {count}, звали {name}, ещё {n} {yes}.",
        "yes": "да",
        "no": "нет",
        "empty": "Пусто.",
        "plural": {"one": "{n} штука", "few": "{n} штуки"},
    }
}


def test_check_accepts_a_catalog_that_holds_every_literal(tmp_path: Path) -> None:
    good = {"t": {k: v for k, v in GOOD["t"].items() if k != "plural"}}
    assert extract_locale.check_catalog(_catalog(tmp_path, good), [("src.py", SOURCE)]) == []


def test_check_reports_missing_literals_and_invented_values(tmp_path: Path) -> None:
    data = {
        "t": {"header": "Привет всем", "invented": "Совсем другое", "plural": GOOD["t"]["plural"]}
    }
    errors = extract_locale.check_catalog(_catalog(tmp_path, data), [("src.py", SOURCE)])
    assert "src.py:14 (reply): literal not in the catalog" in errors
    assert "t.invented: value has no source literal inside it" in errors
    assert "t.plural#0: value has no source literal inside it" in errors


def test_normalize_value_unescapes_braces_and_flattens_fields() -> None:
    assert extract_locale.normalize_value("a {x!r} b {{c}} {y:.2f}") == "a {} b {c} {}"


def test_cli_draft_and_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "demo.py"
    source.write_text(SOURCE, encoding="utf-8")
    assert extract_locale.main(["draft", str(source)]) == 0
    drafted = yaml.safe_load(capsys.readouterr().out)
    assert drafted["demo"]["reply_4"] == "Пусто."
    good = _catalog(tmp_path, {"t": {k: v for k, v in GOOD["t"].items() if k != "plural"}})
    assert extract_locale.main(["check", "--catalog", str(good), str(source)]) == 0
    assert "matches 1 source file" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert extract_locale.main(["check", "--catalog", str(empty), str(source)]) == 1
    assert "literal not in the catalog" in capsys.readouterr().err
