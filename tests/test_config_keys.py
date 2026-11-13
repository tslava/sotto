"""The config-key catalog (``config_keys.yaml``): coverage in every locale, the generator tool,
``describe_key`` with a catalog, and the validator errors that render through ``validation.*``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tools.extract_config_keys import escape_braces, leaf_descriptions, main, render

from sotto.config import ConfigError, describe_key, load_config
from sotto.config_models import Config
from sotto.errors import OwnerError
from sotto.i18n import ENGINE_LOCALES_DIR, Catalog, engine_catalog, read_locale_dir
from sotto.persona import load_persona
from tests.packs import LEGACY_PACK

_CYRILLIC = re.compile("[А-Яа-яЁё]")
EN_FILE = ENGINE_LOCALES_DIR / "en" / "config_keys.yaml"
RU_FILE = LEGACY_PACK / "locale" / "ru" / "config_keys.yaml"


def _leaf_keys() -> list[str]:
    return list(leaf_descriptions())


def _catalog_keys(directory: Path) -> set[str]:
    prefix = "config_keys."
    return {key[len(prefix) :] for key in read_locale_dir(directory) if key.startswith(prefix)}


# -- coverage ----------------------------------------------------------------------


def test_every_settable_key_has_an_entry_in_each_locale() -> None:
    settable = {key for key in _leaf_keys() if not key.startswith("persona.")}
    assert settable
    for directory in (ENGINE_LOCALES_DIR / "en", LEGACY_PACK / "locale" / "ru"):
        assert settable <= _catalog_keys(directory), directory


def test_the_catalogs_hold_exactly_the_leaf_keys() -> None:
    expected = set(_leaf_keys())
    assert _catalog_keys(ENGINE_LOCALES_DIR / "en") == expected
    assert _catalog_keys(LEGACY_PACK / "locale" / "ru") == expected


def test_every_leaf_field_has_an_english_description() -> None:
    for key, text in leaf_descriptions().items():
        assert text, key
        assert not _CYRILLIC.search(text), key
        assert len(text) <= 90, key


def test_the_english_file_is_what_the_generator_writes() -> None:
    assert EN_FILE.read_text(encoding="utf-8") == render(leaf_descriptions())
    assert main(["check", str(EN_FILE)]) == 0


def test_the_russian_file_is_russian_text_with_escaped_braces() -> None:
    ru = read_locale_dir(LEGACY_PACK / "locale" / "ru")
    texts = {k: v for k, v in ru.items() if k.startswith("config_keys.")}
    assert all(isinstance(v, str) and _CYRILLIC.search(v) for v in texts.values())
    catalog = Catalog("ru", ru)
    assert "{avoid}" in catalog.t("config_keys.filters.motif_avoid_window")


def test_generator_writes_and_checks_a_file(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "config_keys.yaml"
    assert main(["write", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == render(leaf_descriptions())
    assert main(["check", str(target)]) == 0
    target.write_text(target.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    assert main(["check", str(target)]) == 1


def test_escape_braces_round_trips_through_format() -> None:
    assert escape_braces("slot {diary} here") == "slot {{diary}} here"
    assert escape_braces("slot {diary} here").format() == "slot {diary} here"


# -- describe_key with a catalog -----------------------------------------------------


def test_describe_key_uses_the_catalog_text() -> None:
    cfg = Config()
    catalog = Catalog("xx", {"config_keys.behaviour.daily_cap": "From the catalog"})
    info = describe_key(cfg, cfg, {}, "behaviour.daily_cap", catalog)
    assert info is not None and info.description == "From the catalog"


def test_describe_key_falls_back_to_the_field_description() -> None:
    cfg = Config()
    field = leaf_descriptions()["behaviour.daily_cap"]
    for catalog in (None, Catalog("xx", {})):
        info = describe_key(cfg, cfg, {}, "behaviour.daily_cap", catalog)
        assert info is not None and info.description == field


def test_describe_key_reads_the_english_fallback_of_a_catalog() -> None:
    cfg = Config()
    english = engine_catalog("en")
    pack_like = Catalog("ru", {}, fallback=english.entries())
    info = describe_key(cfg, cfg, {}, "behaviour.daily_cap", pack_like)
    assert info is not None
    assert info.description == leaf_descriptions()["behaviour.daily_cap"]


def test_describe_key_keeps_braces_of_the_description() -> None:
    cfg = Config()
    info = describe_key(cfg, cfg, {}, "filters.motif_avoid_window", engine_catalog("en"))
    assert info is not None and "{avoid}" in info.description


def test_the_legacy_pack_describes_a_key_in_russian() -> None:
    persona = load_persona(LEGACY_PACK)
    cfg = load_config(persona.config_layers())
    info = describe_key(cfg, cfg, {}, "persona.timezone", persona.catalog)
    assert info is not None and _CYRILLIC.search(info.description)


# -- validators raise owner errors ---------------------------------------------------


def test_a_validator_error_carries_an_owner_error_with_the_key() -> None:
    with pytest.raises(ConfigError) as caught:
        load_config({"behaviour": {"morning_reply_window": ["07:00", "8am"]}})
    owner = caught.value.owner
    assert isinstance(owner, OwnerError)
    assert owner.code == "validation.time"
    assert owner.params == {"value": "8am", "key": "behaviour.morning_reply_window"}
    assert owner.render(engine_catalog("en")) == (
        "behaviour.morning_reply_window: invalid time '8am', expected HH:MM"
    )


def test_a_pydantic_range_error_has_no_owner_error() -> None:
    with pytest.raises(ConfigError) as caught:
        load_config({"behaviour": {"daily_cap": 9999}})
    assert caught.value.owner is None
    assert "behaviour.daily_cap" in str(caught.value)


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ({"behaviour": {"debounce_sec": [9, 1]}}, "validation.debounce_sec"),
        ({"behaviour": {"reactions": {"delay_sec": [5, 1]}}}, "validation.delay_sec"),
        ({"behaviour": {"reactions": {"emoji": {"😂": "x"}}}}, "validation.emoji_not_telegram"),
        ({"behaviour": {"reactions": {"emoji": {}}}}, "validation.emoji_empty"),
        ({"behaviour": {"followup": {"model": "no-slash"}}}, "validation.model_id"),
        ({"behaviour": {"checkin": {"after_min": [9, 1]}}}, "validation.after_min"),
        ({"behaviour": {"vocab": {"request_markers": ["("]}}}, "validation.regex"),
        ({"filters": {"motifs": {"label": ["("]}}}, "validation.motif_regex"),
        ({"places": {"search_center": [95.0, 0.0]}}, "validation.search_center"),
        (
            {"behaviour": {"reply_delay_buckets": [{"weight": 0.2, "range_sec": [1, 2]}]}},
            "validation.bucket_weights",
        ),
    ],
)
def test_every_validator_error_renders_through_the_english_catalog(
    data: dict[str, object], code: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        load_config(data)
    owner = caught.value.owner
    assert owner is not None and owner.code == code
    text = owner.render(engine_catalog("en"))
    assert text.startswith(str(owner.params["key"]) + ": ")
    assert not _CYRILLIC.search(text)


def test_the_legacy_pack_keeps_the_russian_emoji_message() -> None:
    owner = OwnerError("validation.emoji_not_telegram", key="behaviour.reactions.emoji", bad=["😂"])
    text = owner.render(load_persona(LEGACY_PACK).catalog)
    assert text == (
        "behaviour.reactions.emoji: ['😂'] не входит в набор реакций Telegram "
        "(например, вместо 😂 нужен 🤣)"
    )


def test_generator_refuses_a_key_without_a_description(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.extract_config_keys as tool

    monkeypatch.setattr(tool, "leaf_descriptions", lambda: {"a.b": "", "a.c": "text"})
    assert tool.main(["write", str(tmp_path / "x.yaml")]) == 1
    assert "a.b" in capsys.readouterr().err
    assert not (tmp_path / "x.yaml").exists()
