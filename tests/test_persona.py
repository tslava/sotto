"""load_persona: structure, errors, fail-closed selection (PR-2)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from sotto.fragments import PROMPT_NAMES
from sotto.i18n import read_locale_dir
from sotto.lang import UnknownLanguageError, default_config_layer, get_language
from sotto.persona import (
    DEMO_PACK_DIR,
    PACK_FORMAT,
    PackError,
    load_persona,
    persona_from_env,
    resolve_persona_dir,
)
from tests.packs import LEGACY_PACK


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    """A throwaway copy of the legacy pack that a test may break."""
    target = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, target)
    return target


def _write_manifest(pack: Path, **changes: object) -> None:
    data = yaml.safe_load((pack / "pack.yaml").read_text(encoding="utf-8"))
    data.update(changes)
    (pack / "pack.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_legacy_pack_loads_with_expected_shape() -> None:
    persona = load_persona(LEGACY_PACK)

    assert persona.name == "legacy_ru"
    assert persona.manifest.format == PACK_FORMAT
    assert persona.manifest.language == "ru"
    assert persona.lang.code == "ru"
    assert persona.root == LEGACY_PACK
    assert persona.config_path == LEGACY_PACK / "config.yaml"
    assert persona.config_layers() == [
        default_config_layer(persona.lang),
        LEGACY_PACK / "config.yaml",
    ]
    assert persona.few_shot_path == LEGACY_PACK / "few_shot.yaml"
    assert persona.stickers_path == LEGACY_PACK / "stickers.yaml"
    assert persona.places_manual_path == LEGACY_PACK / "places_manual.yaml"
    assert set(persona.prompts) == set(PROMPT_NAMES)
    assert len(persona.prompts) == 13
    assert persona.revision == "dev"
    assert persona.chat_notes == (LEGACY_PACK / "chat_notes.md").read_text(encoding="utf-8")
    assert persona.legacy_changelog == (LEGACY_PACK / "CHANGELOG.md").read_text(encoding="utf-8")
    assert persona.fragment_errors() == []
    assert persona.catalog.keys() == set(read_locale_dir(LEGACY_PACK / "locale" / "ru"))
    assert persona.catalog.keys()


def test_prompts_are_the_file_text_as_prompt_store_reads_it() -> None:
    persona = load_persona(LEGACY_PACK)

    for name in PROMPT_NAMES:
        text = (LEGACY_PACK / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
        assert persona.prompts[name] == text


def test_prompts_mapping_is_read_only() -> None:
    persona = load_persona(LEGACY_PACK)

    with pytest.raises(TypeError):
        persona.prompts["system"] = "x"  # type: ignore[index]


def test_legacy_prompts_have_exactly_their_slots() -> None:
    assert load_persona(LEGACY_PACK).prompt_errors() == []


def test_revision_comes_from_the_revision_file(pack: Path) -> None:
    (pack / ".revision").write_text("abc1234\n", encoding="utf-8")

    assert load_persona(pack).revision == "abc1234"


def test_chat_notes_and_fragments_are_read_when_present(pack: Path) -> None:
    (pack / "chat_notes.md").write_text("notes", encoding="utf-8")
    (pack / "fragments.yaml").write_text("placeholders:\n  photo: x\n", encoding="utf-8")

    persona = load_persona(pack)

    assert persona.chat_notes == "notes"
    assert persona.frag.get("placeholders.photo") == "x"


def test_locale_directory_follows_the_manifest(pack: Path) -> None:
    locale_dir = pack / "locale" / "xx"
    locale_dir.mkdir(parents=True)
    (locale_dir / "help.yaml").write_text("help:\n  title: Hi\n", encoding="utf-8")
    _write_manifest(pack, locale="xx")

    persona = load_persona(pack)

    assert persona.catalog.locale == "xx"
    assert persona.catalog.t("help.title") == "Hi"


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="does not exist"):
        load_persona(tmp_path / "nope")


def test_missing_manifest(pack: Path) -> None:
    (pack / "pack.yaml").unlink()

    with pytest.raises(PackError, match=r"pack\.yaml not found"):
        load_persona(pack)


@pytest.mark.parametrize("body", ["format: [", "- a\n- b\n", "format: 1\nname: x\n"])
def test_invalid_manifest(pack: Path, body: str) -> None:
    (pack / "pack.yaml").write_text(body, encoding="utf-8")

    with pytest.raises(PackError, match="invalid manifest"):
        load_persona(pack)


def test_unknown_manifest_key_is_rejected(pack: Path) -> None:
    _write_manifest(pack, surprise=1)

    with pytest.raises(PackError, match="invalid manifest"):
        load_persona(pack)


def test_unsupported_format(pack: Path) -> None:
    _write_manifest(pack, format=99)

    with pytest.raises(PackError, match="format 99"):
        load_persona(pack)


def test_unknown_language(pack: Path) -> None:
    _write_manifest(pack, language="xx")

    with pytest.raises(PackError, match="unknown language"):
        load_persona(pack)


def test_missing_required_files_are_all_listed(pack: Path) -> None:
    (pack / "config.yaml").unlink()
    (pack / "prompts" / "judge.txt").unlink()

    with pytest.raises(PackError) as info:
        load_persona(pack)

    assert "config.yaml" in str(info.value)
    assert "prompts/judge.txt" in str(info.value)


def test_malformed_fragments_file_is_a_pack_error(pack: Path) -> None:
    (pack / "fragments.yaml").write_text("- a\n", encoding="utf-8")

    with pytest.raises(PackError, match="fragments"):
        load_persona(pack)


def test_malformed_locale_file_is_a_pack_error(pack: Path) -> None:
    locale_dir = pack / "locale" / "ru"
    locale_dir.mkdir(parents=True, exist_ok=True)  # the legacy pack already has one
    (locale_dir / "help.yaml").write_text("other:\n  a: b\n", encoding="utf-8")

    with pytest.raises(PackError, match="top-level key"):
        load_persona(pack)


def test_prompt_errors_report_a_bad_slot(pack: Path) -> None:
    (pack / "prompts" / "diary.txt").write_text("no slots here {extra}", encoding="utf-8")

    errors = load_persona(pack).prompt_errors()

    assert "prompts/diary.txt: required slot {reply} is missing" in errors
    assert "prompts/diary.txt: slot {extra} is not allowed" in errors


# -- selection -----------------------------------------------------------------


def test_resolve_persona_dir_uses_the_explicit_directory() -> None:
    assert resolve_persona_dir(Path("/srv/pack"), allow_demo=False) == Path("/srv/pack")
    assert resolve_persona_dir(Path("/srv/pack"), allow_demo=True) == Path("/srv/pack")


def test_resolve_persona_dir_fails_closed_without_directory() -> None:
    with pytest.raises(PackError, match="PERSONA_DIR not set"):
        resolve_persona_dir(None, allow_demo=False)


def test_resolve_persona_dir_demo_opt_in() -> None:
    assert resolve_persona_dir(None, allow_demo=True) == DEMO_PACK_DIR
    assert DEMO_PACK_DIR.parts[-2:] == ("personas", "demo")


def test_persona_from_env_reads_persona_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONA_DIR", str(LEGACY_PACK))

    assert persona_from_env().name == "legacy_ru"


def test_persona_from_env_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("PERSONA_DIR")

    with pytest.raises(PackError, match="PERSONA_DIR not set"):
        persona_from_env()


# -- language registry ---------------------------------------------------------


def test_get_language_known_and_unknown() -> None:
    assert get_language("ru").code == "ru"
    assert get_language("en").code == "en"
    with pytest.raises(UnknownLanguageError):
        get_language("de")


@pytest.mark.parametrize(
    ("n", "category"),
    [(1, "one"), (21, "one"), (11, "many"), (2, "few"), (24, "few"), (12, "many"), (5, "many")],
)
def test_russian_plural_categories(n: int, category: str) -> None:
    assert get_language("ru").plural(n) == category


@pytest.mark.parametrize(("n", "category"), [(1, "one"), (0, "other"), (2, "other")])
def test_english_plural_categories(n: int, category: str) -> None:
    assert get_language("en").plural(n) == category


def test_demo_pack_fixture(demo_pack: object) -> None:
    assert demo_pack is not None  # skipped until personas/demo exists
