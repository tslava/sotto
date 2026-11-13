"""The persona pack: everything that makes the bot a particular character.

``load_persona(dir)`` reads a pack directory once (prompts as text, the rest by path) and
returns a frozen ``Persona``. The pack is chosen at run time by ``PERSONA_DIR``; there is no
module-level "active pack". A pack is::

    pack.yaml  config.yaml  few_shot.yaml  stickers.yaml  places_manual.yaml
    prompts/<name>.txt (13)  [fragments.yaml]  [locale/<code>/*.yaml]  [chat_notes.md]
    [CHANGELOG.md]  [.revision]

A pack's ``locale/<code>/`` may be partial or absent: every key it lacks is read from the engine's
``locales/en`` (``i18n``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from sotto.fragments import (
    PROMPT_NAMES,
    Fragments,
    fragment_errors,
    load_fragments,
    slot_errors,
)
from sotto.i18n import (
    ENGINE_LOCALES_DIR,
    FALLBACK_LOCALE,
    Catalog,
    CatalogError,
    load_catalog_dir,
)
from sotto.lang import Language, UnknownLanguageError, default_config_layer, get_language

logger = logging.getLogger(__name__)

PACK_FORMAT = 1
DEMO_PACK_DIR = Path(__file__).resolve().parents[2] / "personas" / "demo"


class PackError(ValueError):
    """The pack is missing, malformed or not selected."""


class Manifest(BaseModel):
    """``pack.yaml``."""

    model_config = ConfigDict(extra="forbid")

    format: int
    name: str
    language: str
    locale: str | None = None
    ship: list[str] = []


@dataclass(frozen=True)
class Persona:
    root: Path
    manifest: Manifest
    config_path: Path
    few_shot_path: Path
    prompts: Mapping[str, str]
    stickers_path: Path
    places_manual_path: Path
    chat_notes: str
    # Both release-note blocks of the pre-split pack ("CHANGELOG.md"); the transition source of
    # /changelog until the pack carries chat_notes.md.
    legacy_changelog: str
    frag: Fragments
    lang: Language
    catalog: Catalog
    revision: str

    @property
    def name(self) -> str:
        return self.manifest.name

    def config_layers(self) -> list[Path | Mapping[str, Any]]:
        """Config layers below the DB overrides, lowest first: the language's generic
        defaults, then the pack's ``config.yaml`` (which replaces them key by key)."""
        return [default_config_layer(self.lang), self.config_path]

    def prompt_errors(self) -> list[str]:
        """Slot problems of all prompts (empty list = fine)."""
        errors: list[str] = []
        for name in PROMPT_NAMES:
            errors += slot_errors(name, self.prompts[name])
        return errors

    def fragment_errors(self) -> list[str]:
        """Schema problems of ``fragments.yaml`` against ``FRAGMENT_SPEC`` (empty list = fine)."""
        return fragment_errors(self.frag.data)


def resolve_persona_dir(persona_dir: Path | None, allow_demo: bool) -> Path:
    """The pack directory to load. Fails closed: no ``PERSONA_DIR`` and no demo opt-in: error."""
    if persona_dir is not None:
        return persona_dir
    if allow_demo:
        return DEMO_PACK_DIR
    raise PackError("PERSONA_DIR not set (set it to the persona pack directory)")


def persona_from_env() -> Persona:
    """Pack chosen by ``PERSONA_DIR`` / ``ALLOW_DEMO_PERSONA`` (env or .env), for the CLIs."""
    # Local import: settings is a leaf module, persona must stay importable without it.
    from sotto.settings import PersonaSelector

    selector = PersonaSelector()
    return load_persona(resolve_persona_dir(selector.persona_dir, selector.allow_demo_persona))


def _read_manifest(root: Path) -> Manifest:
    path = root / "pack.yaml"
    if not path.is_file():
        raise PackError(f"{path}: pack.yaml not found")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = Manifest.model_validate(loaded)
    except (yaml.YAMLError, ValidationError) as exc:
        raise PackError(f"{path}: invalid manifest: {exc}") from exc
    if manifest.format != PACK_FORMAT:
        raise PackError(
            f"{path}: pack format {manifest.format} is not supported (engine reads {PACK_FORMAT})"
        )
    return manifest


def _read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def load_persona(path: Path) -> Persona:
    """Read and structurally validate a pack. Deeper checks live in ``packcheck``."""
    root = Path(path)
    if not root.is_dir():
        raise PackError(f"persona directory {root} does not exist")
    manifest = _read_manifest(root)

    required = [root / "config.yaml", root / "few_shot.yaml"]
    required += [root / "prompts" / f"{name}.txt" for name in PROMPT_NAMES]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise PackError(f"pack {root} is missing files: {', '.join(missing)}")

    prompts = {
        name: (root / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
        for name in PROMPT_NAMES
    }
    try:
        lang = get_language(manifest.language)
        locale = manifest.locale or manifest.language
        pack_locale_dir = root / "locale" / locale
        engine_dir = ENGINE_LOCALES_DIR / FALLBACK_LOCALE
        # A pack without its own texts for the locale reads the engine's English catalog.
        catalog = load_catalog_dir(
            pack_locale_dir
            if pack_locale_dir.is_dir() or locale != FALLBACK_LOCALE
            else engine_dir,
            locale,
            plural_rule=lang.plural,
            fallback_dir=engine_dir,
        )
        frag = load_fragments(root / "fragments.yaml")
    except (UnknownLanguageError, CatalogError, ValueError) as exc:
        raise PackError(f"pack {root}: {exc}") from exc

    revision = _read_optional(root / ".revision").strip() or "dev"
    return Persona(
        root=root,
        manifest=manifest,
        config_path=root / "config.yaml",
        few_shot_path=root / "few_shot.yaml",
        prompts=MappingProxyType(prompts),
        stickers_path=root / "stickers.yaml",
        places_manual_path=root / "places_manual.yaml",
        chat_notes=_read_optional(root / "chat_notes.md"),
        legacy_changelog=_read_optional(root / "CHANGELOG.md"),
        frag=frag,
        lang=lang,
        catalog=catalog,
        revision=revision,
    )
