"""Shared fixtures: the persona packs tests run against."""

from __future__ import annotations

import pytest

from sotto.persona import Persona, load_persona
from tests.packs import DEMO_PACK, LEGACY_PACK


@pytest.fixture(autouse=True)
def _default_persona_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run against the legacy pack unless they say otherwise (PERSONA_DIR is set)."""
    monkeypatch.setenv("PERSONA_DIR", str(LEGACY_PACK))
    monkeypatch.delenv("ALLOW_DEMO_PERSONA", raising=False)


@pytest.fixture
def legacy_pack() -> Persona:
    """The temporary default pack (``personas/_legacy_ru``) until the demo pack lands."""
    return load_persona(LEGACY_PACK)


@pytest.fixture
def demo_pack() -> Persona:
    """The synthetic English demo pack (``personas/demo``), skipped while it does not exist."""
    if not DEMO_PACK.is_dir():
        pytest.skip("personas/demo does not exist yet")
    return load_persona(DEMO_PACK)
