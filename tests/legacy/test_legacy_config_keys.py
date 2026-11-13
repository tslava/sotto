"""The legacy Russian pack's config-key catalog (locale/ru) and its owner messages.

Migration proof: removed together with ``personas/_legacy_ru`` in PR-9.
"""

from __future__ import annotations

import re
from pathlib import Path

from tools.extract_config_keys import leaf_descriptions

from sotto.config import describe_key, load_config
from sotto.errors import OwnerError
from sotto.i18n import Catalog, read_locale_dir
from sotto.persona import load_persona
from tests.packs import LEGACY_PACK

_CYRILLIC = re.compile("[А-Яа-яЁё]")


def _catalog_keys(directory: Path) -> set[str]:
    prefix = "config_keys."
    return {key[len(prefix) :] for key in read_locale_dir(directory) if key.startswith(prefix)}


def test_every_settable_key_has_an_entry_in_the_legacy_ru_catalog() -> None:
    settable = {key for key in leaf_descriptions() if not key.startswith("persona.")}
    assert settable <= _catalog_keys(LEGACY_PACK / "locale" / "ru")


def test_the_legacy_ru_catalog_holds_exactly_the_leaf_keys() -> None:
    assert _catalog_keys(LEGACY_PACK / "locale" / "ru") == set(leaf_descriptions())


def test_the_russian_file_is_russian_text_with_escaped_braces() -> None:
    ru = read_locale_dir(LEGACY_PACK / "locale" / "ru")
    texts = {k: v for k, v in ru.items() if k.startswith("config_keys.")}
    assert all(isinstance(v, str) and _CYRILLIC.search(v) for v in texts.values())
    catalog = Catalog("ru", ru)
    assert "{avoid}" in catalog.t("config_keys.filters.motif_avoid_window")


def test_the_legacy_pack_describes_a_key_in_russian() -> None:
    persona = load_persona(LEGACY_PACK)
    cfg = load_config(persona.config_layers())
    info = describe_key(cfg, cfg, {}, "persona.timezone", persona.catalog)
    assert info is not None and _CYRILLIC.search(info.description)


def test_the_legacy_pack_keeps_the_russian_emoji_message() -> None:
    owner = OwnerError("validation.emoji_not_telegram", key="behaviour.reactions.emoji", bad=["😂"])
    text = owner.render(load_persona(LEGACY_PACK).catalog)
    assert text == (
        "behaviour.reactions.emoji: ['😂'] не входит в набор реакций Telegram "
        "(например, вместо 😂 нужен 🤣)"
    )
