"""Fragments: the schema (FRAGMENT_SPEC), the one-shot extractor and the pack's config keys.

All wording here is synthetic or already part of the repo's legacy pack; the extractor is
tested against tiny generated source trees, never against real engine history.

Migration proof: removed together with ``personas/_legacy_ru`` in PR-9.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from tools import extract_fragments as ef

from sotto.fragments import (
    ALL_FRAGMENT_KEYS,
    FRAGMENT_LISTS,
    FRAGMENT_MAPS,
    FRAGMENT_SPEC,
    LITERAL_BRACES,
    Fragments,
    fragment_errors,
    load_fragments,
    load_valid_fragments,
)
from tests.packs import LEGACY_FRAG, LEGACY_PACK, frag_with

# -- the schema ----------------------------------------------------------------------


def test_legacy_fragments_are_valid_and_complete() -> None:
    assert fragment_errors(LEGACY_FRAG.data) == []
    assert set(FRAGMENT_SPEC) | set(FRAGMENT_LISTS) | FRAGMENT_MAPS == ALL_FRAGMENT_KEYS
    assert len(ALL_FRAGMENT_KEYS) == 72


def test_extractor_table_matches_the_schema_exactly() -> None:
    assert {item.key for item in ef.ITEMS} == set(ALL_FRAGMENT_KEYS)
    assert len(ef.ITEMS) == len({item.key for item in ef.ITEMS})


def _legacy_data() -> dict[str, dict[str, object]]:
    loaded = yaml.safe_load((LEGACY_PACK / "fragments.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_schema_reports_missing_unknown_and_wrong_placeholders_by_key_only() -> None:
    data = _legacy_data()
    del data["weather"]["now_word"]
    data["weather"]["bogus"] = "x"
    data["situations"]["callback"] = "{name} only"
    data["situations"]["ambient"] = "has {surprise}"
    data["prompt"]["places_none"] = "stray } brace"

    errors = fragment_errors(data)

    assert errors == [
        "fragments.yaml: missing key weather.now_word",
        "fragments.yaml: unknown key weather.bogus",
        "fragments.yaml: situations.ambient placeholders are [surprise], expected [none]",
        "fragments.yaml: situations.callback placeholders are [name], expected [name, text]",
        "fragments.yaml: prompt.places_none contains a stray brace",
    ]
    # messages carry key names only, never the wording itself
    assert not any("stray }" in error for error in errors)


def test_schema_checks_value_types() -> None:
    data = _legacy_data()
    data["situations"]["late"] = 5
    data["render"]["weekdays"] = ["a", "b"]
    data["weather"]["wmo"] = {0: 1}
    data["render"]["accusative"] = []

    assert fragment_errors(data) == [
        "fragments.yaml: situations.late must be a string",
        "fragments.yaml: render.weekdays must be a list of 7 strings",
        "fragments.yaml: render.accusative must be a non-empty mapping",
        "fragments.yaml: weather.wmo values must be strings",
    ]


def test_json_reminders_may_keep_their_braces() -> None:
    assert set(FRAGMENT_SPEC) >= LITERAL_BRACES
    for key in LITERAL_BRACES:
        assert "{" in LEGACY_FRAG.text(key)
    assert fragment_errors(LEGACY_FRAG.data) == []


def test_load_valid_fragments_rejects_a_missing_or_broken_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_valid_fragments(tmp_path / "fragments.yaml")
    (tmp_path / "fragments.yaml").write_text("a:\n  b: c\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown key a\.b"):
        load_valid_fragments(tmp_path / "fragments.yaml")
    (tmp_path / "fragments.yaml").write_text("", encoding="utf-8")
    assert load_fragments(tmp_path / "fragments.yaml").data == {}
    (tmp_path / "fragments.yaml").write_text("- a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at top level"):
        load_fragments(tmp_path / "fragments.yaml")


def test_typed_accessors() -> None:
    frag = Fragments({"a": {"s": "x{n}y", "l": ["p", "q"], "m": {1: "one"}}})

    assert frag.text("a.s") == "x{n}y"
    assert frag.prefix("a.s") == "x"
    assert frag.items("a.l") == ["p", "q"]
    assert frag.table("a.m") == {1: "one"}
    assert frag.render("a.s", n="7") == "x7y"
    assert frag.has("a.s") and not frag.has("a.zzz") and not frag.has("a.s.deeper")
    for call, key in ((frag.text, "a.l"), (frag.items, "a.s"), (frag.table, "a.s")):
        with pytest.raises(TypeError):
            call(key)
    with pytest.raises(KeyError):
        frag.get("a.zzz")
    with pytest.raises(TypeError):
        frag.prefix("a.m")


def test_render_is_single_pass() -> None:
    frag = Fragments({"a": {"s": "{x} and {y}"}})

    assert frag.render("a.s", x="{y}", y="B") == "{y} and B"


# -- the extractor -------------------------------------------------------------------


def _tree(tmp_path: Path, **modules: str) -> Path:
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    for name, body in modules.items():
        (package / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path / "src"


def test_extractor_reads_constants_inline_literals_and_fstrings(tmp_path: Path) -> None:
    src = _tree(
        tmp_path,
        a="""
        GREETING = "привет " + "мир"
        TABLE = {1: "один", 2: ("два", "две")}
        _ALIAS = GREETING

        def render(name):
            '''doc ignored: привет'''
            first = "первый"
            second = "второй"
            return f"{first}-{name}: хвост {{ok}}"

        def tagged(n, text):
            return f"[метка #{n}] {text}"
        """,
        b='OTHER = "привет мир"\n',
    )
    cache: dict[str, ef._Module] = {}

    def read(source: ef.Source) -> object:
        return ef.read_source(src, "pkg", source, cache)

    assert read(ef.Const("a", "GREETING")) == "привет мир"
    assert read(ef.Const("a", "_ALIAS")) == "привет мир"
    assert read(ef.Const("a", "TABLE")) == {1: "один", 2: ["два", "две"]}
    assert read(ef.Inline("a", "render", 0)) == "первый"
    assert read(ef.Inline("a", "render", 1)) == "второй"
    assert read(ef.Inline("a", "render", 0, pattern="хвост")) == ": хвост {ok}"
    assert (
        read(ef.FString("a", "tagged", ("id", "text"), strip_suffix=" {text}")) == "[метка #{id}]"
    )
    assert read(ef.FString("a", "tagged", ("id", "text"))) == "[метка #{id}] {text}"


@pytest.mark.parametrize(
    "source",
    [
        ef.Const("a", "MISSING"),
        ef.Const("nomodule", "X"),
        ef.Inline("a", "nofunction", 0),
        ef.Inline("a", "tagged", 5),
        ef.FString("a", "tagged", ("only_one",)),
        ef.FString("a", "tagged", ("a", "b", "c")),
        ef.FString("a", "plain", ("x",)),
        ef.FString("a", "tagged", ("id", "text"), strip_suffix="nope"),
        ef.Const("a", "BAD_PLUS"),
        ef.Const("a", "BAD_CALL"),
        ef.Const("a", "BAD_SPREAD"),
    ],
)
def test_extractor_reports_unexpected_shapes(tmp_path: Path, source: ef.Source) -> None:
    src = _tree(
        tmp_path,
        a="""
        BAD_PLUS = 1 + 2
        BAD_CALL = str("x")
        BAD_SPREAD = {**{"a": 1}}

        def tagged(n, text):
            return f"[метка #{n}] {text}"

        def plain():
            return "без подстановок"
        """,
    )

    with pytest.raises(ef.ExtractError):
        ef.read_source(src, "pkg", source, {})


def test_extract_cross_checks_second_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _tree(
        tmp_path,
        a='ONE = "общее"\nTAG = "[метка #"\n\ndef f(n):\n    return f"[метка #{n}] x"\n',
        b='SAME = "общее"\nOTHER = "иное"\nPREFIX = "[метка #"\nWRONG = "[другая #"\n',
    )
    items = (
        ef.Item("g.one", ef.Const("a", "ONE"), (ef.Const("b", "SAME"),)),
        ef.Item("g.two", ef.Const("a", "ONE"), (ef.Const("b", "OTHER"),)),
        ef.Item("g.tag", ef.FString("a", "f", ("id",), " x"), (ef.Const("b", "PREFIX"),), "prefix"),
        ef.Item("g.bad", ef.FString("a", "f", ("id",), " x"), (ef.Const("b", "WRONG"),), "prefix"),
    )
    monkeypatch.setattr(ef, "ITEMS", items)

    flat, problems = ef.extract(src, "pkg")

    assert flat == {
        "g.one": "общее",
        "g.two": "общее",
        "g.tag": "[метка #{id}]",
        "g.bad": "[метка #{id}]",
    }
    assert problems == ["g.two: disagrees with b", "g.bad: disagrees with b"]


def test_cli_writes_the_nested_file_and_fails_on_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _tree(tmp_path, a='ONE = "один"\nTWO = "два"\n', b='SAME = "один"\n')
    out = tmp_path / "out.yaml"
    monkeypatch.setattr(
        ef,
        "ITEMS",
        (
            ef.Item("g.one", ef.Const("a", "ONE"), (ef.Const("b", "SAME"),)),
            ef.Item("h.two", ef.Const("a", "TWO")),
        ),
    )

    assert ef.main(["--src", str(src), "--package", "pkg", "--out", str(out)]) == 0
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == {
        "g": {"one": "один"},
        "h": {"two": "два"},
    }
    assert "wrote 2 keys" in capsys.readouterr().out

    monkeypatch.setattr(
        ef, "ITEMS", (ef.Item("g.one", ef.Const("a", "TWO"), (ef.Const("b", "SAME"),)),)
    )
    assert ef.main(["--src", str(src), "--package", "pkg", "--out", str(out)]) == 2
    assert "g.one: disagrees with b" in capsys.readouterr().err

    monkeypatch.setattr(ef, "ITEMS", (ef.Item("g.one", ef.Const("a", "MISSING")),))
    assert ef.main(["--src", str(src), "--package", "pkg", "--out", str(out)]) == 1
    assert "MISSING" in capsys.readouterr().err


def test_dump_keeps_unicode_and_survives_a_round_trip() -> None:
    nested = ef.nest({"a.x": "привет: {n}", "a.y": ["д", "е"], "b.z": {0: "ноль"}})

    text = ef.dump(nested)

    assert "привет" in text and "\\u" not in text
    assert yaml.safe_load(text) == nested


def test_frag_with_replaces_one_leaf_and_keeps_the_rest() -> None:
    changed = frag_with(placeholders__photo="[picture]")

    assert changed.text("placeholders.photo") == "[picture]"
    assert changed.text("placeholders.sticker") == LEGACY_FRAG.text("placeholders.sticker")
    assert LEGACY_FRAG.text("placeholders.photo") != "[picture]"
