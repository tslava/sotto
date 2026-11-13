"""Owner-UI catalog: ``Catalog.t`` / ``Catalog.plural`` over ``locale/<code>/*.yaml``.

One YAML file per area (``help``, ``status``, ``commands``, ``alerts``, ``errors``,
``validation``, ``config_keys``); the top-level
key of each file equals the file stem, values are strings or plural mappings
(``one``/``few``/``many``/``other``). Placeholders are ``{name}`` and are checked with
``string.Formatter`` at load. The engine ships the English catalog in ``locales/en``; a pack may
carry ``locale/<code>/`` for its own locale and every key it lacks falls back to English with one
WARNING (a command never crashes on a missing text). ``catalog_parity`` is the pack-vs-English
check.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PLURAL_CATEGORIES = ("one", "few", "many", "other")
# The engine's own catalogs: ``<repo>/locales/<code>/*.yaml`` (copied into the image next to src).
ENGINE_LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"
FALLBACK_LOCALE = "en"

Entries = dict[str, str | dict[str, str]]


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
    """Dotted-key lookup. A key found nowhere is recorded, warned about once and returned as is;
    a key only the English fallback has is answered from it, with one warning."""

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
        # Keys found nowhere (``missing``) and keys answered by the fallback (``fell_back``).
        self.missing: set[str] = set()
        self.fell_back: set[str] = set()

    def keys(self) -> set[str]:
        return set(self._entries)

    def entries(self) -> dict[str, str | dict[str, str]]:
        """A copy of the locale's own entries (without the fallback)."""
        return dict(self._entries)

    def has(self, key: str) -> bool:
        """The locale or its English fallback carries ``key`` (no warning, nothing recorded)."""
        return key in self._entries or key in self._fallback

    def _lookup(self, key: str) -> str | dict[str, str] | None:
        if key in self._entries:
            return self._entries[key]
        if key in self._fallback:
            if key not in self.fell_back:
                logger.warning(
                    "catalog %s: key %r missing, using the English text", self.locale, key
                )
            self.fell_back.add(key)
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


def _read_dir(directory: Path) -> Entries:
    entries: Entries = {}
    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if loaded is None:
                continue
            if not isinstance(loaded, dict) or set(loaded) != {path.stem}:
                raise CatalogError(f"{path}: the only top-level key must be {path.stem!r}")
            _flatten(path.stem, loaded[path.stem], entries)
    return entries


def read_locale_dir(directory: Path) -> Entries:
    """Flat ``dotted.key -> text | plural forms`` of one locale directory (empty if missing)."""
    return _read_dir(directory)


def load_catalog_dir(
    directory: Path,
    locale: str,
    *,
    plural_rule: Callable[[int], str] = lambda n: "one" if n == 1 else "other",
    fallback_dir: Path | None = None,
) -> Catalog:
    """Read ``directory/*.yaml``; a missing directory gives an empty catalog.

    ``fallback_dir`` (normally ``ENGINE_LOCALES_DIR / "en"``) supplies the keys the locale lacks.
    """
    fallback = _read_dir(fallback_dir) if fallback_dir is not None else None
    return Catalog(locale, _read_dir(directory), fallback=fallback, plural_rule=plural_rule)


def engine_catalog(locale: str = FALLBACK_LOCALE) -> Catalog:
    """The engine's own catalog for ``locale`` (``locales/<locale>``), English as fallback."""
    plural_rule: Callable[[int], str] = _plural_rule_for(locale)
    return load_catalog_dir(
        ENGINE_LOCALES_DIR / locale,
        locale,
        plural_rule=plural_rule,
        fallback_dir=ENGINE_LOCALES_DIR / FALLBACK_LOCALE,
    )


def _plural_rule_for(locale: str) -> Callable[[int], str]:
    # Local import: sotto.lang does not depend on the catalog, keep this module a leaf.
    from sotto.lang import get_language

    return get_language(locale).plural


def catalog_parity(
    reference: Mapping[str, str | dict[str, str]],
    other: Mapping[str, str | dict[str, str]],
    plural_rule_categories: Sequence[str],
    *,
    other_name: str = "locale",
) -> list[str]:
    """Differences between the English ``reference`` and another locale (empty list = parity).

    Every reference key must exist, with the same placeholders; plural keys carry every category the
    locale's plural rule can return; the reverse direction (extra keys) is an error too.
    """
    errors: list[str] = []
    for key, ref_value in reference.items():
        if key not in other:
            errors.append(f"{other_name}: missing key {key}")
            continue
        value = other[key]
        if isinstance(ref_value, str) != isinstance(value, str):
            kind = "a plain string" if isinstance(ref_value, str) else "plural forms"
            errors.append(f"{other_name}: {key} must be {kind}")
            continue
        if isinstance(ref_value, str) and isinstance(value, str):
            if _placeholders(ref_value) != _placeholders(value):
                errors.append(f"{other_name}: {key} placeholders differ from English")
            continue
        assert isinstance(ref_value, dict) and isinstance(value, dict)
        for category in plural_rule_categories:
            if category not in value:
                errors.append(f"{other_name}: {key} lacks plural form {category!r}")
        ref_names = set().union(*(_placeholders(text) for text in ref_value.values()))
        for form, text in value.items():
            if _placeholders(text) - ref_names:
                errors.append(f"{other_name}: {key}.{form} uses placeholders outside English")
    for key in other:
        if key not in reference:
            errors.append(f"{other_name}: extra key {key}")
    return errors
