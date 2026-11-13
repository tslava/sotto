"""Shared fixtures: the persona packs tests run against."""

from __future__ import annotations

import pytest

from sotto.persona import Persona, load_persona
from tests.packs import DEMO_PACK, LEGACY_PACK, RU_PACK


@pytest.fixture(autouse=True)
def _default_persona_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run against the English demo pack unless they say otherwise (PERSONA_DIR is set)."""
    monkeypatch.setenv("PERSONA_DIR", str(DEMO_PACK))
    monkeypatch.delenv("ALLOW_DEMO_PERSONA", raising=False)


@pytest.fixture
def legacy_pack() -> Persona:
    """The legacy Russian pack (``personas/_legacy_ru``); removed in PR-9."""
    return load_persona(LEGACY_PACK)


@pytest.fixture
def ru_pack() -> Persona:
    """The synthetic Russian pack (``tests/fixtures/ru_pack``) the language tests run on."""
    return load_persona(RU_PACK)


@pytest.fixture
def demo_pack() -> Persona:
    """The synthetic English demo pack (``personas/demo``)."""
    return load_persona(DEMO_PACK)
