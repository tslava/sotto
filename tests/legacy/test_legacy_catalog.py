"""The legacy Russian pack's catalog through ConfigStore (pack locale over the engine catalog).

Migration proof: removed together with ``personas/_legacy_ru`` in PR-9.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sotto.db import Database
from sotto.persona import Persona, load_persona
from sotto.stores import ConfigStore
from tests.packs import LEGACY_PACK


def _legacy() -> Persona:
    return load_persona(LEGACY_PACK)


async def test_describe_reads_the_description_from_the_legacy_ru_catalog(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    store = ConfigStore(_legacy(), db)
    await store.load()
    try:
        info = store.describe("behaviour.daily_cap")

        assert info is not None
        assert info.description == "Суточный лимит ambient и «просто так» вместе"
    finally:
        await db.close()


async def test_describe_falls_back_to_the_english_catalog_for_a_pack_without_locale(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    shutil.rmtree(pack / "locale")
    store = ConfigStore(load_persona(pack), db)
    await store.load()
    try:
        info = store.describe("behaviour.daily_cap")

        assert info is not None
        assert info.description == "Daily limit of ambient and unprompted replies together"
    finally:
        await db.close()
