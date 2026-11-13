"""ConfigStore with the synthetic Russian pack."""

from __future__ import annotations

from pathlib import Path

from sotto.db import Database
from sotto.persona import load_persona
from sotto.stores import ConfigStore
from tests.packs import RU_PACK


async def test_config_store_patterns_carry_the_ru_pack_language(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    store = ConfigStore(load_persona(RU_PACK), db)
    await store.load()
    try:
        assert store.patterns().lang is not None
        assert store.patterns().lang.code == "ru"
    finally:
        await db.close()
