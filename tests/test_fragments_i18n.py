"""Fragments, prompt slot tables and the owner-UI catalog (PR-2 stubs with the final API)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sotto.fragments import (
    PROMPT_NAMES,
    PROMPT_SLOTS,
    Fragments,
    load_fragments,
    slot_errors,
)
from sotto.i18n import Catalog, CatalogError, load_catalog_dir

# -- prompt slots ---------------------------------------------------------------


def test_thirteen_prompts_are_known() -> None:
    assert len(PROMPT_NAMES) == 13
    for required, allowed in PROMPT_SLOTS.values():
        assert required <= allowed


def test_slot_errors_accept_exact_slots() -> None:
    assert slot_errors("diary", "reply: {reply}") == []


def test_slot_errors_report_missing_and_extra() -> None:
    errors = slot_errors("judge", "{trigger} and {other}")

    assert "prompts/judge.txt: required slot {candidate} is missing" in errors
    assert "prompts/judge.txt: slot {other} is not allowed" in errors


def test_slot_errors_unknown_prompt() -> None:
    assert slot_errors("nope", "") == ["prompts/nope.txt: not a known prompt"]


def test_system_prompt_may_use_optional_slots_but_not_foreign_ones() -> None:
    body = "{context} {situation} {few_shot} {age} {weather} {x}"

    assert slot_errors("system", body) == ["prompts/system.txt: slot {x} is not allowed"]


# -- fragments ------------------------------------------------------------------


def test_fragments_lookup_and_render() -> None:
    frag = Fragments({"a": {"b": "hi {name}, {name}!", "n": 3}})

    assert frag.has("a.b")
    assert not frag.has("a.c")
    assert not frag.has("a.b.c")
    assert frag.get("a.b") == "hi {name}, {name}!"
    assert frag.render("a.b", name="Bob") == "hi Bob, Bob!"
    assert frag.render("a.b") == "hi {name}, {name}!"  # unknown slots stay literal
    with pytest.raises(KeyError):
        frag.get("a.c")
    with pytest.raises(KeyError):
        frag.get("a.b.c")
    with pytest.raises(TypeError):
        frag.render("a.n")


def test_fragments_render_never_uses_format_semantics() -> None:
    frag = Fragments({"k": "{{x}} {name}"})

    assert frag.render("k", name="{y}") == "{{x}} {y}"


def test_load_fragments(tmp_path: Path) -> None:
    assert load_fragments(tmp_path / "missing.yaml").data == {}
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_fragments(empty).data == {}
    good = tmp_path / "good.yaml"
    good.write_text("a:\n  b: c\n", encoding="utf-8")
    assert load_fragments(good).get("a.b") == "c"
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_fragments(bad)


# -- catalog --------------------------------------------------------------------


def _catalog_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    directory = tmp_path / "locale"
    directory.mkdir()
    for name, text in files.items():
        (directory / f"{name}.yaml").write_text(text, encoding="utf-8")
    return directory


def test_catalog_t_and_plural(tmp_path: Path) -> None:
    directory = _catalog_dir(
        tmp_path,
        {
            "status": (
                "status:\n"
                "  title: 'v{version}'\n"
                "  items:\n"
                "    one: '{n} item'\n"
                "    other: '{n} items'\n"
                "  nested:\n"
                "    deep: 'ok'\n"
            )
        },
    )

    catalog = load_catalog_dir(directory, "en")

    assert catalog.keys() == {"status.title", "status.items", "status.nested.deep"}
    assert catalog.t("status.title", version="1.2") == "v1.2"
    assert catalog.t("status.nested.deep") == "ok"
    assert catalog.plural("status.items", 1) == "1 item"
    assert catalog.plural("status.items", 5) == "5 items"
    assert catalog.missing == set()


def test_catalog_plural_uses_the_language_rule(tmp_path: Path) -> None:
    directory = _catalog_dir(
        tmp_path,
        {
            "s": (
                "s:\n  d:\n    one: '{n} a'\n    few: '{n} b'\n"
                "    many: '{n} c'\n    other: '{n} d'\n"
            )
        },
    )
    from sotto.lang import get_language

    catalog = load_catalog_dir(directory, "ru", plural_rule=get_language("ru").plural)

    assert catalog.plural("s.d", 2) == "2 b"
    assert catalog.plural("s.d", 11) == "11 c"


def test_catalog_plain_string_as_plural_and_missing_form_falls_back_to_other() -> None:
    catalog = Catalog("en", {"k": "{n} thing", "p": {"other": "{n} things"}})

    assert catalog.plural("k", 3) == "3 thing"
    assert catalog.plural("p", 1) == "1 things"


def test_catalog_plural_without_any_matching_form_raises() -> None:
    catalog = Catalog("en", {"p": {"few": "x"}})

    with pytest.raises(KeyError):
        catalog.plural("p", 1)


def test_catalog_missing_key_is_recorded_and_warned_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = Catalog("en", {})

    with caplog.at_level(logging.WARNING, logger="sotto.i18n"):
        assert catalog.t("a.b") == "a.b"
        assert catalog.t("a.b") == "a.b"
        assert catalog.plural("a.c", 2) == "a.c"

    assert catalog.missing == {"a.b", "a.c"}
    assert len([r for r in caplog.records if "a.b" in r.message]) == 1


def test_catalog_falls_back_to_the_fallback_mapping() -> None:
    catalog = Catalog("ru", {}, fallback={"k": "fallback {x}"})

    assert catalog.t("k", x=1) == "fallback 1"
    assert catalog.missing == set()


def test_catalog_t_on_a_plural_key_is_an_error() -> None:
    catalog = Catalog("en", {"p": {"one": "x", "other": "y"}})

    with pytest.raises(TypeError):
        catalog.t("p")


def test_load_catalog_dir_missing_directory_gives_an_empty_catalog(tmp_path: Path) -> None:
    assert load_catalog_dir(tmp_path / "nope", "en").keys() == set()


def test_load_catalog_dir_skips_empty_files(tmp_path: Path) -> None:
    directory = _catalog_dir(tmp_path, {"empty": ""})

    assert load_catalog_dir(directory, "en").keys() == set()


@pytest.mark.parametrize(
    "text",
    [
        "other:\n  a: b\n",  # top-level key is not the file stem
        "bad:\n  a: 5\n",  # not a string
        "bad:\n  a: 'x {'\n",  # bad placeholder syntax
        "bad:\n  a:\n    one: 'x {'\n    other: y\n",  # bad syntax inside a plural
        "- 1\n",  # not a mapping
    ],
)
def test_load_catalog_dir_rejects_malformed_files(tmp_path: Path, text: str) -> None:
    directory = _catalog_dir(tmp_path, {"bad": text})

    with pytest.raises(CatalogError):
        load_catalog_dir(directory, "en")
