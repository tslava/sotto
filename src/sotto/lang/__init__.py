"""Language registry: ``get_language("ru")``."""

from __future__ import annotations

from sotto.lang.base import BaseLanguage, Language

SUPPORTED_LANGUAGES: tuple[str, ...] = ("ru", "en")

__all__ = [
    "SUPPORTED_LANGUAGES",
    "BaseLanguage",
    "Language",
    "UnknownLanguageError",
    "get_language",
]


class UnknownLanguageError(ValueError):
    """``pack.yaml`` names a language the engine does not know."""


def get_language(code: str) -> Language:
    if code not in SUPPORTED_LANGUAGES:
        known = ", ".join(SUPPORTED_LANGUAGES)
        raise UnknownLanguageError(f"unknown language {code!r} (known: {known})")
    return BaseLanguage(code)
