"""Fixtures of the language tests: they run on the synthetic Russian pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from sotto.db import Database
from tests.packs import RU_PACK


@pytest.fixture(autouse=True)
def _ru_default_persona_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """tests/lang run against the synthetic Russian pack (``tests/fixtures/ru_pack``)."""
    monkeypatch.setenv("PERSONA_DIR", str(RU_PACK))


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "bot.db")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()
