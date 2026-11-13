"""Owner-UI catalog: ``Catalog.t`` / ``Catalog.plural`` over ``locale/<code>/*.yaml``.

One YAML file per area; the top-level key of each file equals the file stem, values are
strings or plural mappings (``one``/``few``/``many``/``other``). Placeholders are
``{name}`` and are checked with ``string.Formatter`` at load. The catalog is a stub in
this PR: no owner text is read from it yet, a pack without ``locale/`` gets an empty one.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PLURAL_CATEGORIES = ("one", "few", "many", "other")


class CatalogError(ValueError):
    """The catalog files are malformed."""


def _placeholders(text: str) -> set[str]:
    try:
        return {name for _, name, _, _ in string.Formatter().parse(text) if name}
    except ValueError as exc:
        raise CatalogError(f"bad placeholder syntax in {text!r}: {exc}") from exc


def _flatten(prefix: str, node: Any, out: dict[str, str | dict[str, str]]) -> None:
    if isinstance(node, str):
        _placeholders(node)
        out[prefix] = node
    elif isinstance(node, Mapping) and node and set(node) <= set(PLURAL_CATEGORIES):
        forms = {str(k): str(v) for k, v in node.items()}
        for form in forms.values():
            _placeholders(form)
        out[prefix] = forms
    elif isinstance(node, Mapping):
        for key, child in node.items():
            _flatten(f"{prefix}.{key}", child, out)
    else:
        raise CatalogError(f"{prefix}: value must be a string, plural mapping or section")


class Catalog:
    """Dotted-key lookup. A missing key is recorded, warned about once and returned as is."""

    def __init__(
        self,
        locale: str,
        entries: Mapping[str, str | dict[str, str]] | None = None,
        *,
        fallback: Mapping[str, str | dict[str, str]] | None = None,
        plural_rule: Callable[[int], str] = lambda n: "one" if n == 1 else "other",
    ) -> None:
        self.locale = locale
        self._entries: dict[str, str | dict[str, str]] = dict(entries or {})
        self._fallback: dict[str, str | dict[str, str]] = dict(fallback or {})
        self._plural_rule = plural_rule
        self.missing: set[str] = set()

    def keys(self) -> set[str]:
        return set(self._entries)

    def _lookup(self, key: str) -> str | dict[str, str] | None:
        if key in self._entries:
            return self._entries[key]
        if key in self._fallback:
            return self._fallback[key]
        if key not in self.missing:
            logger.warning("catalog %s: missing key %r", self.locale, key)
        self.missing.add(key)
        return None

    def t(self, key: str, /, **params: object) -> str:
        value = self._lookup(key)
        if value is None:
            return key
        if isinstance(value, dict):
            raise TypeError(f"catalog key {key!r} is a plural; use plural()")
        return value.format(**params)

    def plural(self, key: str, n: int, /, **params: object) -> str:
        value = self._lookup(key)
        if value is None:
            return key
        if isinstance(value, str):
            return value.format(n=n, **params)
        form = self._plural_rule(n)
        text = value.get(form) or value.get("other")
        if text is None:
            raise KeyError(f"catalog key {key!r} has no form {form!r}")
        return text.format(n=n, **params)


def load_catalog_dir(
    directory: Path,
    locale: str,
    *,
    plural_rule: Callable[[int], str] = lambda n: "one" if n == 1 else "other",
) -> Catalog:
    """Read ``directory/*.yaml``; a missing directory gives an empty catalog."""
    entries: dict[str, str | dict[str, str]] = {}
    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if loaded is None:
                continue
            if not isinstance(loaded, dict) or set(loaded) != {path.stem}:
                raise CatalogError(f"{path}: the only top-level key must be {path.stem!r}")
            _flatten(path.stem, loaded[path.stem], entries)
    return Catalog(locale, entries, plural_rule=plural_rule)
