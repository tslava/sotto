"""The ``Language`` protocol: what is linguistic rather than persona data.

A language module answers questions the engine cannot answer without knowing the
language: how to fold text for comparison, which words are stop words, how a name
declines, how a vocabulary word is inflected, what a "fact" about a place may look
like, which generic marker lists exist, how plural categories work. Persona data
(declension of a motif label, district aliases, the character's own foreign words)
is NOT here; it lives in the pack.

Modules are looked up by ``get_language(code)`` and travel with the ``Persona``;
engine code receives the ``Language`` as an argument and never reads a global.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VocabRule:
    """How one vocabulary word is recognised in running text.

    A token matches when it starts with ``stem`` and the rest (after removing a
    reflexive suffix when ``reflexive`` is true) is one of ``endings``. ``reflexive``
    is ``None`` for words that are not verbs, ``True`` when the form must end with a
    reflexive suffix (``Language.reflexive_suffixes``). ``exact_forms`` are extra
    whole-token forms the stem rule cannot express (fleeting vowels).
    """

    stem: str
    endings: frozenset[str]
    reflexive: bool | None
    exact_forms: tuple[str, ...] = ()


class Language(Protocol):
    @property
    def code(self) -> str: ...

    @property
    def stopwords(self) -> frozenset[str]:
        """Words too common to make a repeated 4-gram meaningful (``dedup:self_echo``)."""
        ...

    @property
    def foreign_letters(self) -> str | None:
        """Regex character-class body of the script the persona must not write in
        (``"A-Za-z"`` for Russian); ``None`` switches the foreign-script checks off."""
        ...

    @property
    def reflexive_suffixes(self) -> tuple[str, ...]:
        """Suffixes of reflexive verb forms (empty when the language has none)."""
        ...

    def plural(self, n: int) -> str:
        """CLDR-style plural category: ``one``, ``few``, ``many`` or ``other``."""
        ...

    def fold(self, text: str) -> str:
        """Comparison form: casefolded, stress marks and letter variants removed."""
        ...

    def name_pattern(self, word: str) -> str:
        """Regex source (no ``re.compile``) for a name trigger in all inflected forms."""
        ...

    def name_forms(self, word: str) -> list[str]:
        """The forms ``name_pattern`` matches, for tests."""
        ...

    def vocab_rules(self, word: str) -> list[VocabRule]:
        """Stem/ending rules for one folded vocabulary word."""
        ...

    def vocab_prefix_tails(self, token: str) -> list[str]:
        """Verb tails left after cutting a known prefix off ``token``, in priority order."""
        ...

    def vocab_name_stem(self, word: str) -> str | None:
        """Stem of a dictionary name (first folded word) used to spot a request for it."""
        ...

    def fact_re(self) -> re.Pattern[str]:
        """Characters a one-line fact about a place may consist of (whole-string match)."""
        ...

    def default_filters(self) -> Mapping[str, list[str]]:
        """Generic marker lists for ``filters.*``; they sit below the pack's ``config.yaml``."""
        ...


def default_config_layer(lang: Language) -> dict[str, dict[str, list[str]]]:
    """The language's config layer: goes below the pack ``config.yaml`` (empty if no lists)."""
    filters = {key: list(value) for key, value in lang.default_filters().items()}
    return {"filters": filters} if filters else {}
