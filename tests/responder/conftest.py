"""Fixtures for the responder tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sotto.db import Database
from tests.demo import DEMO_PACK


@pytest.fixture(autouse=True)
def _demo_persona_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """The responder tests run against the English demo pack."""
    monkeypatch.setenv("PERSONA_DIR", str(DEMO_PACK))


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "bot.db")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()
