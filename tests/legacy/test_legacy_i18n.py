"""The legacy Russian pack's owner-UI catalog (locale/ru): parity with the English files, help
text size, placeholders and plural forms.

Migration proof: removed together with ``personas/_legacy_ru`` in PR-9.
"""

from __future__ import annotations

from sotto.errors import OwnerError
from sotto.i18n import ENGINE_LOCALES_DIR, Catalog, catalog_parity, read_locale_dir
from sotto.lang import get_language
from sotto.persona import load_persona
from tests.packs import LEGACY_PACK

EN = read_locale_dir(ENGINE_LOCALES_DIR / "en")
RU = read_locale_dir(LEGACY_PACK / "locale" / "ru")
AREAS = {"help", "status", "commands", "alerts", "errors", "validation", "config_keys"}


def _placeholders(text: str) -> set[str]:
    import string

    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_legacy_ru_catalog_has_the_same_seven_files() -> None:
    assert {path.stem for path in (LEGACY_PACK / "locale" / "ru").glob("*.yaml")} == AREAS


def test_english_and_legacy_ru_are_in_parity() -> None:
    assert catalog_parity(EN, RU, ["one", "few", "many"], other_name="ru") == []


def test_legacy_ru_help_text_fits_one_telegram_message() -> None:
    help_text = RU["help.text"]
    assert isinstance(help_text, str) and len(help_text) <= 3500
    for command in ("/stop", "/mute", "/status", "/set", "/help"):
        assert command in help_text, command


def test_every_legacy_ru_value_formats_with_its_own_placeholders() -> None:
    for value in RU.values():
        for text in [value] if isinstance(value, str) else list(value.values()):
            params = {name: 1 for name in _placeholders(text)}
            text.format(**params)  # no stray braces


def test_legacy_ru_plural_forms_follow_the_russian_rule() -> None:
    ru = Catalog("ru", RU, plural_rule=get_language("ru").plural)
    assert [ru.plural("commands.words_count", n) for n in (1, 2, 5, 11, 21, 22)] == [
        "1 слово",
        "2 слова",
        "5 слов",
        "11 слов",
        "21 слово",
        "22 слова",
    ]


def test_owner_error_renders_through_the_legacy_ru_catalog() -> None:
    error = OwnerError("ex_rm_range", index=4, total=2)
    ru = load_persona(LEGACY_PACK).catalog
    assert error.render(ru) == "индекс 4 вне диапазона: всего примеров 2"


def test_vocab_errors_render_through_the_legacy_ru_catalog() -> None:
    from sotto.vocab import parse_vocab

    ru = load_persona(LEGACY_PACK).catalog
    lang = get_language("ru")
    raw = "name: x\nwords:\n  - word: '!!!'\n    meaning: b\n  - {word: Сырюка}"
    try:
        parse_vocab(raw, lang)
    except OwnerError as exc:
        assert exc.render(ru) == (
            "words → слово #1 → word: Value error, в слове нет букв; "
            "words → слово #2 → meaning: Field required"
        )
    else:  # pragma: no cover
        raise AssertionError("expected an error")
    try:
        parse_vocab("- список", lang)
    except OwnerError as exc:
        assert exc.render(ru) == "ожидался YAML-объект с полями name и words"
    else:  # pragma: no cover
        raise AssertionError("expected an error")
