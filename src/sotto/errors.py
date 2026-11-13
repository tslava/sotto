"""Owner-facing errors: a code plus parameters, rendered through the pack's catalog.

``OwnerError`` is a ``ValueError`` so existing ``except ValueError`` keeps working. The text the
owner reads is ``catalog.t(key, **params)`` (``commands.py`` renders it) where ``key`` is
``errors.<code>``, or the code itself when it already names a catalog area
(``validation.time``: the config validators). ``str(exc)`` is only a diagnostic for logs and tests.
Pydantic's own validation messages are not wrapped.
"""

from __future__ import annotations

from sotto.i18n import Catalog


class OwnerError(ValueError):
    """An error the bot owner reads in a private chat; ``code`` names ``errors.<code>``."""

    def __init__(self, code: str, **params: object) -> None:
        self.code = code
        self.params = params
        detail = ", ".join(f"{key}={value!r}" for key, value in params.items())
        super().__init__(f"{code}: {detail}" if detail else code)

    @property
    def catalog_key(self) -> str:
        return self.code if "." in self.code else f"errors.{self.code}"

    def render(self, catalog: Catalog) -> str:
        return catalog.t(self.catalog_key, **self.params)
