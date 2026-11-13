"""Language registry: ``get_language("ru")``."""

from __future__ import annotations

from collections.abc import Callable

from sotto.lang.base import Language, VocabRule, default_config_layer
from sotto.lang.en import EnglishLanguage
from sotto.lang.ru import RussianLanguage

_REGISTRY: dict[str, Callable[[], Language]] = {"ru": RussianLanguage, "en": EnglishLanguage}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(_REGISTRY)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "Language",
    "UnknownLanguageError",
    "VocabRule",
    "default_config_layer",
    "get_language",
]


class UnknownLanguageError(ValueError):
    """``pack.yaml`` names a language the engine does not know."""


def get_language(code: str) -> Language:
    factory = _REGISTRY.get(code)
    if factory is None:
        known = ", ".join(SUPPORTED_LANGUAGES)
        raise UnknownLanguageError(f"unknown language {code!r} (known: {known})")
    return factory()
