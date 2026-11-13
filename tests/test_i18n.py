"""The owner-UI catalog: the English files, the legacy ru migration proof, fallbacks, parity,
and a static check that every catalog call in the engine matches the English texts."""

from __future__ import annotations

import ast
import logging
import re
import shutil
import string
from pathlib import Path

import pytest
import yaml

from sotto.errors import OwnerError
from sotto.i18n import (
    ENGINE_LOCALES_DIR,
    Catalog,
    catalog_parity,
    engine_catalog,
    load_catalog_dir,
    read_locale_dir,
)
from sotto.packcheck import EXIT_INVALID, EXIT_OK, locale_parity_errors, main
from sotto.persona import load_persona
from tests.packs import LEGACY_PACK, REPO_ROOT

EN = read_locale_dir(ENGINE_LOCALES_DIR / "en")
AREAS = {"help", "status", "commands", "alerts", "errors", "validation", "config_keys"}
SRC = REPO_ROOT / "src" / "sotto"


def _placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def _texts(entries: dict) -> list[str]:  # type: ignore[type-arg]
    out: list[str] = []
    for value in entries.values():
        out += [value] if isinstance(value, str) else list(value.values())
    return out


# -- the files ---------------------------------------------------------------------


def test_english_catalog_has_the_seven_owner_areas() -> None:
    assert {key.split(".")[0] for key in EN} == AREAS
    assert {path.stem for path in (ENGINE_LOCALES_DIR / "en").glob("*.yaml")} == AREAS


def test_english_catalog_is_plain_english_text() -> None:
    for text in _texts(EN):
        assert not re.search("[\u0400-\u04ff]", text), text
        assert "**" not in text and "`" not in text and "<b>" not in text, text
        assert len(text) <= 3500, text[:40]


def test_help_texts_fit_one_telegram_message() -> None:
    for entries in (EN,):
        help_text = entries["help.text"]
        assert isinstance(help_text, str) and len(help_text) <= 3500
        for command in (
            "/stop",
            "/mute",
            "/unmute",
            "/ex add",
            "/panic",
            "/resume",
            "/status",
            "/last",
            "/why",
            "/get",
            "/set",
            "/unset",
            "/prompt",
            "/rollback",
            "/ex last",
            "/ex rm",
            "/life",
            "/say",
            "/ask",
            "/memory",
            "/diary",
            "/jokes",
            "/people",
            "/words",
            "/changelog",
            "/help",
        ):
            assert command in help_text, command


def test_every_value_formats_with_its_own_placeholders() -> None:
    for entries in (EN,):
        for value in entries.values():
            for text in [value] if isinstance(value, str) else list(value.values()):
                params = {name: 1 for name in _placeholders(text)}
                text.format(**params)  # no stray braces


def test_plural_forms_follow_the_language_rule() -> None:
    en = Catalog("en", EN, plural_rule=lambda n: "one" if n == 1 else "other")
    assert en.plural("commands.words_count", 1) == "1 word"
    assert en.plural("commands.words_count", 2) == "2 words"


# -- every catalog call in the engine ----------------------------------------------


def _catalog_calls() -> tuple[list[tuple[str, set[str], str]], set[str]]:
    """``(key, parameter names, file)`` of the constant-key catalog calls, and every key-looking
    string constant (a call with a computed key still names its keys somewhere)."""
    calls: list[tuple[str, set[str], str]] = []
    mentioned: set[str] = set()
    pattern = re.compile(r"^(help|status|commands|alerts|errors|validation)\.[a-z_0-9]+$")
    for name in ("commands.py", "alerts.py", "stores.py", "config_models.py"):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and pattern.match(node.value)
            ):
                mentioned.add(node.value)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if callee == "OwnerError" and node.args and isinstance(node.args[0], ast.Constant):
                code = str(node.args[0].value)
                key = code if "." in code else f"errors.{code}"
                params = {k.arg for k in node.keywords if k.arg}
                if code.startswith("validation."):
                    params.add("key")  # filled in from the error location by load_config
                calls.append((key, params, name))
                mentioned.add(key)
                continue
            first = 1 if callee == "_t" else 0 if callee in {"t", "plural"} else None
            if first is None or len(node.args) <= first:
                continue
            arg = node.args[first]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                params = {k.arg for k in node.keywords if k.arg}
                if callee == "plural":
                    params.add("n")
                calls.append((arg.value, params, name))
    return calls, mentioned


def test_every_catalog_call_names_an_existing_key_with_all_its_placeholders() -> None:
    calls, _ = _catalog_calls()
    assert len(calls) > 100
    problems: list[str] = []
    for key, params, name in calls:
        if key not in EN:
            problems.append(f"{name}: unknown key {key}")
            continue
        value = EN[key]
        forms = [value] if isinstance(value, str) else list(value.values())
        needed = set().union(*(_placeholders(form) for form in forms))
        if needed - params:
            problems.append(f"{name}: {key} lacks {sorted(needed - params)}")
    assert problems == []


def test_every_english_key_is_used_by_the_engine() -> None:
    _, mentioned = _catalog_calls()
    # alerts.hint_<status> is built from the status code.
    mentioned |= {"alerts.hint_401", "alerts.hint_402"}
    # config_keys.<key> is read by describe_key for every config key.
    unused = {key for key in set(EN) - mentioned if not key.startswith("config_keys.")}
    assert sorted(unused) == []


# -- loading and fallback -----------------------------------------------------------


def test_engine_catalog_reads_english() -> None:
    catalog = engine_catalog()
    assert catalog.locale == "en"
    assert catalog.t("commands.say_sent") == "Sent."
    assert catalog.missing == set()


def test_engine_catalog_for_another_locale_falls_back_to_english(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = engine_catalog("ru")  # no locales/ru: every key comes from English
    with caplog.at_level(logging.WARNING, logger="sotto.i18n"):
        assert catalog.t("commands.say_sent") == "Sent."
        assert catalog.t("commands.say_sent") == "Sent."
    assert catalog.fell_back == {"commands.say_sent"}
    assert catalog.missing == set()
    assert len([r for r in caplog.records if "say_sent" in r.getMessage()]) == 1
    assert catalog.plural("commands.words_count", 3) == "3 words"


def test_catalog_entries_is_a_copy_of_the_locale_own_texts() -> None:
    catalog = Catalog("en", {"a.b": "x"}, fallback={"a.c": "y"})
    own = catalog.entries()
    own["new"] = "z"
    assert catalog.entries() == {"a.b": "x"}


def test_load_catalog_dir_uses_the_fallback_dir(tmp_path: Path) -> None:
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "x.yaml").write_text("x:\n  a: own\n", encoding="utf-8")
    (tmp_path / "fb").mkdir()
    (tmp_path / "fb" / "x.yaml").write_text("x:\n  a: fb\n  b: only\n", encoding="utf-8")
    catalog = load_catalog_dir(tmp_path / "own", "ru", fallback_dir=tmp_path / "fb")
    assert catalog.t("x.a") == "own"
    assert catalog.t("x.b") == "only"
    assert catalog.fell_back == {"x.b"}


def test_a_pack_without_its_own_locale_reads_the_engine_english(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    shutil.rmtree(pack / "locale")
    persona = load_persona(pack)
    assert persona.catalog.keys() == set()
    assert persona.catalog.t("commands.say_sent") == "Sent."


def test_an_english_pack_without_locale_dir_uses_the_engine_catalog(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    shutil.rmtree(pack / "locale")
    manifest = yaml.safe_load((pack / "pack.yaml").read_text(encoding="utf-8"))
    manifest["locale"] = "en"
    (pack / "pack.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    persona = load_persona(pack)
    assert persona.catalog.locale == "en"
    assert persona.catalog.keys() == set(EN)
    assert locale_parity_errors(persona) == []


# -- parity -------------------------------------------------------------------------


def test_parity_reports_every_kind_of_difference() -> None:
    reference = {
        "a.plain": "hello {name}",
        "a.gone": "bye",
        "a.plural": {"one": "{n} item", "other": "{n} items"},
        "a.kind": "text",
        "a.kind2": {"one": "x", "other": "y"},
        "a.pl": {"one": "{n} {what}", "other": "{n} {what}"},
    }
    other = {
        "a.plain": "hola {who}",
        "a.plural": {"one": "{n} cosa"},
        "a.kind": {"one": "x", "other": "y"},
        "a.kind2": "text",
        "a.pl": {"one": "{n} {what} {extra}", "other": "{n} {what}"},
        "a.extra": "?",
    }
    errors = catalog_parity(reference, other, ["one", "other"], other_name="xx")
    assert errors == [
        "xx: a.plain placeholders differ from English",
        "xx: missing key a.gone",
        "xx: a.plural lacks plural form 'other'",
        "xx: a.kind must be a plain string",
        "xx: a.kind2 must be plural forms",
        "xx: a.pl.one uses placeholders outside English",
        "xx: extra key a.extra",
    ]


def test_packcheck_locale_parity_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(LEGACY_PACK), "--locale-parity"]) == EXIT_OK
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    commands = pack / "locale" / "ru" / "commands.yaml"
    data = yaml.safe_load(commands.read_text(encoding="utf-8"))
    del data["commands"]["say_sent"]
    data["commands"]["set_done"] = "{key}"
    commands.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main([str(pack), "--locale-parity"]) == EXIT_INVALID
    err = capsys.readouterr().err
    assert "locale/ru: missing key commands.say_sent" in err
    assert "commands.set_done placeholders differ" in err
    assert main([str(pack)]) == EXIT_OK  # without the flag the pack is still valid


def test_packcheck_locale_parity_needs_a_locale_dir_for_a_non_english_pack(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    shutil.rmtree(pack / "locale")
    errors = locale_parity_errors(load_persona(pack))
    assert errors == ["locale/ru: directory not found (the pack reads English for every text)"]


# -- OwnerError ---------------------------------------------------------------------


def test_owner_error_renders_through_the_catalog_and_stays_a_value_error() -> None:
    error = OwnerError("ex_rm_range", index=4, total=2)
    assert isinstance(error, ValueError)
    assert error.code == "ex_rm_range" and error.params == {"index": 4, "total": 2}
    assert "ex_rm_range" in str(error)
    assert error.render(engine_catalog()) == "index 4 is out of range: there are 2 examples"
    assert str(OwnerError("set_persona")) == "set_persona"
