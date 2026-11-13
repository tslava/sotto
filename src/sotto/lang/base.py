"""The ``Language`` protocol: what is linguistic rather than persona data.

PR-2 fixes only the shape every later piece plugs into (``code`` and the plural rule).
The remaining methods listed in the architecture (name declension, vocabulary rules,
default filter lists, ...) arrive with the language modules in a later PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Language(Protocol):
    @property
    def code(self) -> str: ...

    def plural(self, n: int) -> str:
        """CLDR-style plural category: ``one``, ``few``, ``many`` or ``other``."""
        ...


@dataclass(frozen=True)
class BaseLanguage:
    """Placeholder implementation: knows the plural rule of the supported codes."""

    code: str

    def plural(self, n: int) -> str:
        count = abs(n)
        if self.code == "ru":
            if count % 10 == 1 and count % 100 != 11:
                return "one"
            if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
                return "few"
            return "many"
        return "one" if count == 1 else "other"
