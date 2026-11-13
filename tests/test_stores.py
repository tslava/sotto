"""ConfigStore и PromptStore (этап 6): горячий /set, валидация, версии промпта/few-shot."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from sotto.db import Database
from sotto.errors import OwnerError
from sotto.persona import Persona, load_persona
from sotto.stores import ConfigStore, PromptStore, build_seed_plan, decide_seed
from tests.packs import LEGACY_PACK


def _legacy() -> Persona:
    return load_persona(LEGACY_PACK)


def _with_system(persona: Persona, text: str) -> Persona:
    """The same pack with another system prompt (a restart after an edit of the file)."""
    return replace(persona, prompts={**persona.prompts, "system": text})


# -- ConfigStore ---------------------------------------------------------------


def _write_minimal_config(path: Path) -> None:
    # Пустой yaml -> все поля Config берутся из дефолтов pydantic-моделей.
    path.write_text("", encoding="utf-8")


async def _make_config_store(tmp_path: Path) -> tuple[ConfigStore, Database]:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)
    store = ConfigStore(replace(_legacy(), config_path=config_path), db)
    await store.load()
    return store, db


async def test_set_valid_key_changes_get(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        assert store.get().behaviour.ambient_probability == 0.25

        old, new = await store.set("behaviour.ambient_probability", "0.9", changed_by=1, now=1000)

        assert old is None
        assert new == "0.9"
        assert store.get().behaviour.ambient_probability == 0.9
    finally:
        await db.close()


async def test_set_valid_key_rebuilds_patterns(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        assert store.patterns().topic_stop("виджет xyz123 виджет") is None

        await store.set("filters.topic_stop", "[xyz123]", changed_by=1, now=1000)

        assert store.patterns().topic_stop("виджет xyz123 виджет") is not None
    finally:
        await db.close()


async def test_set_invalid_value_raises_and_does_not_write_override(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        with pytest.raises(ValueError, match=r"behaviour\.daily_cap"):
            await store.set("behaviour.daily_cap", "9999", changed_by=1, now=1000)

        assert await db.get_overrides() == {}
        assert store.get().behaviour.daily_cap == 3  # дефолт не изменился
    finally:
        await db.close()


async def test_set_persona_key_raises_without_writing(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        with pytest.raises(OwnerError, match="persona") as info:
            await store.set("persona.name", "Вася", changed_by=1, now=1000)
        assert info.value.code == "set_persona"

        assert await db.get_overrides() == {}
    finally:
        await db.close()


async def test_set_unknown_section_raises(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        with pytest.raises(OwnerError) as info:
            await store.set("secret.token", "x", changed_by=1, now=1000)
        assert (info.value.code, info.value.params) == (
            "set_unknown_section",
            {"key": "secret.token"},
        )

        assert await db.get_overrides() == {}
    finally:
        await db.close()


async def test_unset_returns_default_value(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        await store.set("behaviour.daily_cap", "10", changed_by=1, now=1000)
        assert store.get().behaviour.daily_cap == 10

        default_value = await store.unset("behaviour.daily_cap", changed_by=1, now=2000)

        assert default_value == "3"
        assert store.get().behaviour.daily_cap == 3
        assert await db.get_overrides() == {}
    finally:
        await db.close()


async def test_flat_marks_overridden_keys(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        flat_before = {key: overridden for key, _value, overridden in store.flat()}
        assert flat_before["behaviour.daily_cap"] is False

        await store.set("behaviour.daily_cap", "5", changed_by=1, now=1000)

        flat_after = {key: (value, overridden) for key, value, overridden in store.flat()}
        assert flat_after["behaviour.daily_cap"] == ("5", True)
        assert flat_after["behaviour.ambient_probability"][1] is False
    finally:
        await db.close()


async def test_describe_returns_none_for_unknown_or_section(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        assert store.describe("behaviour.no_such_key") is None
        assert store.describe("behaviour.live_talk") is None
    finally:
        await db.close()


async def test_describe_before_override_shows_default_as_value(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        info = store.describe("behaviour.daily_cap")

        assert info is not None
        assert info.value == "3"
        assert info.default == "3"
        assert info.overridden is False
        assert info.type_name == "int"
        assert info.bounds == "0..50"
        assert info.description
    finally:
        await db.close()


async def test_describe_after_override_shows_new_value_and_old_default(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        await store.set("behaviour.daily_cap", "9", changed_by=1, now=1000)

        info = store.describe("behaviour.daily_cap")

        assert info is not None
        assert info.value == "9"
        assert info.default == "3"
        assert info.overridden is True
    finally:
        await db.close()


async def test_describe_persona_key_not_settable(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        info = store.describe("persona.name")

        assert info is not None
        assert info.settable is False
    finally:
        await db.close()


async def test_describe_reads_the_description_from_the_packs_catalog(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        info = store.describe("behaviour.daily_cap")

        assert info is not None
        # The legacy pack's ru catalog, not the English Field.description.
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


async def test_set_validator_error_is_an_owner_error_with_the_key(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        with pytest.raises(OwnerError) as caught:
            await store.set("behaviour.quiet_window", '["25:00", "07:00"]', changed_by=1, now=1000)

        assert caught.value.code == "validation.time"
        assert caught.value.params["key"] == "behaviour.quiet_window"
        assert caught.value.render(_legacy().catalog) == (
            "behaviour.quiet_window: invalid time '25:00', expected HH:MM"
        )
        assert "behaviour.quiet_window" not in await db.get_overrides()
    finally:
        await db.close()


async def test_set_bot_username_enables_mentions_bot(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        assert store.patterns().mentions_bot("привет @example_persona_bot как дела") is False

        store.set_bot_username("example_persona_bot")

        assert store.patterns().mentions_bot("привет @example_persona_bot как дела") is True
    finally:
        await db.close()


# -- PromptStore -----------------------------------------------------------------

_FEW_SHOT_SEED = """\
- name: Юля
  user: "привет"
  speak: true
  text: "И тебе привет."
- name: Гена
  user: "лол"
  speak: false
"""


async def _make_prompt_store(
    tmp_path: Path, prompt_body: str = "system v1"
) -> tuple[PromptStore, Database, Persona]:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    few_shot_path = tmp_path / "few_shot.yaml"
    few_shot_path.write_text(_FEW_SHOT_SEED, encoding="utf-8")
    persona = _with_system(replace(_legacy(), few_shot_path=few_shot_path), prompt_body)
    store = PromptStore(persona, db)
    return store, db, persona


async def test_load_empty_db_seeds_version_1_from_files(tmp_path: Path) -> None:
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()

        assert store.prompt_version() == 1
        assert store.system_prompt() == "system v1"
        assert store.few_shot_version() == 1
        assert len(store.examples()) == 2
        assert "И тебе привет." in store.few_shot_text()

        versions = await db.prompt_versions()
        assert len(versions) == 1
        assert versions[0].active is True
    finally:
        await db.close()


async def test_load_changed_prompt_file_creates_new_active_version(tmp_path: Path) -> None:
    store, db, persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        assert store.prompt_version() == 1

        store = PromptStore(_with_system(persona, "system v2"), db)
        await store.load()

        assert store.prompt_version() == 2
        assert store.system_prompt() == "system v2"
        versions = await db.prompt_versions()
        assert len(versions) == 2
    finally:
        await db.close()


async def test_load_unchanged_prompt_file_does_not_add_version(tmp_path: Path) -> None:
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        await store.load()
        await store.load()

        assert store.prompt_version() == 1
        versions = await db.prompt_versions()
        assert len(versions) == 1
    finally:
        await db.close()


async def test_rollback_prompt_to_1_restores_body(tmp_path: Path) -> None:
    store, db, persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()  # версия 1, "system v1"

        store = PromptStore(_with_system(persona, "system v2"), db)
        await store.load()  # версия 2, активная
        assert store.prompt_version() == 2

        ok = await store.rollback_prompt(1)

        assert ok is True
        assert store.prompt_version() == 1
        assert store.system_prompt() == "system v1"

        missing = await store.rollback_prompt(999)
        assert missing is False
        # неудачный rollback не меняет текущую версию
        assert store.prompt_version() == 1
    finally:
        await db.close()


async def test_load_after_rollback_does_not_reseed_new_version(tmp_path: Path) -> None:
    """Баг: /rollback 1, затем рестарт (новый load()) с файлом всё ещё на v2 тексте
    раньше засевал v3 из файла и тем самым отменял откат — load() сравнивал файл
    только с активной версией, а после отката активная (v1) отличается от файла
    (v2). Исправлено: файл сравнивается ВСЕМИ сохранёнными версиями."""
    store, db, persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()  # версия 1, "system v1"

        store = PromptStore(_with_system(persona, "system v2"), db)
        await store.load()  # версия 2, активная, файл теперь "system v2"
        assert store.prompt_version() == 2

        ok = await store.rollback_prompt(1)
        assert ok is True
        assert store.prompt_version() == 1

        # Рестарт процесса: новый PromptStore на тех же файле и БД. Файл на диске
        # всё ещё содержит "system v2" (никто его не откатывал), но это тело уже
        # есть среди сохранённых версий — новая версия не должна завестись, и
        # активной должна остаться версия 1 (результат отката).
        restarted = PromptStore(_with_system(persona, "system v2"), db)
        await restarted.load()

        assert restarted.prompt_version() == 1
        assert restarted.system_prompt() == "system v1"
        versions = await db.prompt_versions()
        assert len(versions) == 2  # версия 3 не появилась
    finally:
        await db.close()


async def test_load_new_file_text_after_rollback_still_adds_version(tmp_path: Path) -> None:
    """Если после отката файл на диске меняют на текст, которого ещё нет ни в одной
    версии, load() всё равно должен завести новую версию (а не молча остаться на
    откаченной) — сравнение идёт по телу, а не по факту "был ли откат"."""
    store, db, persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        store = PromptStore(_with_system(persona, "system v2"), db)
        await store.load()
        assert await store.rollback_prompt(1) is True

        store = PromptStore(_with_system(persona, "system v3 - brand new"), db)
        await store.load()

        assert store.prompt_version() == 3
        assert store.system_prompt() == "system v3 - brand new"
        versions = await db.prompt_versions()
        assert len(versions) == 3
    finally:
        await db.close()


async def test_add_example_bumps_few_shot_version_and_is_findable(tmp_path: Path) -> None:
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        assert store.few_shot_version() == 1

        new_version = await store.add_example(
            name="Тёма", user="что делаешь", text="Ничего особенного.", now=1000
        )

        assert new_version == 2
        assert store.few_shot_version() == 2

        examples = store.examples()
        assert len(examples) == 3
        added = examples[-1]
        assert added.name == "Тёма"
        assert added.user == "что делаешь"
        assert added.speak is True
        assert added.text == "Ничего особенного."

        assert "Ничего особенного." in store.few_shot_text()

        # active_few_shot должен отражать новую версию в БД
        active = await db.active_few_shot()
        assert active is not None
        assert active[0] == 2
    finally:
        await db.close()


async def test_remove_example_index_1_removes_last_added(tmp_path: Path) -> None:
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        await store.add_example(
            name="Тёма", user="что делаешь", text="Ничего особенного.", now=1000
        )
        assert store.few_shot_version() == 2

        new_version = await store.remove_example(1, now=2000)

        assert new_version == 3
        assert store.few_shot_version() == 3
        examples = store.examples()
        assert len(examples) == 2
        assert all(e.text != "Ничего особенного." for e in examples)
    finally:
        await db.close()


async def test_concurrent_add_example_serializes_and_keeps_both(tmp_path: Path) -> None:
    """Без asyncio.Lock два параллельных add_example читают один и тот же
    self.examples() до того, как любой из них запишет новую версию — потерялся
    бы один добавленный пример, и версий стало бы +1 вместо +2. С локом оба
    вызова исполняются строго по очереди."""
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        assert store.few_shot_version() == 1

        versions = await asyncio.gather(
            store.add_example(name="Тёма", user="что делаешь", text="Ничего.", now=1000),
            store.add_example(name="Ксюша", user="как сам", text="Бывает.", now=1001),
        )

        assert sorted(versions) == [2, 3]
        assert store.few_shot_version() == 3

        texts = {e.text for e in store.examples()}
        assert "Ничего." in texts
        assert "Бывает." in texts

        all_versions = await db.few_shot_version_bodies()
        assert len(all_versions) == 3
    finally:
        await db.close()


async def test_remove_example_out_of_range_raises_and_keeps_version(tmp_path: Path) -> None:
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        await store.load()
        assert store.few_shot_version() == 1

        with pytest.raises(OwnerError) as info:
            await store.remove_example(99, now=1000)
        assert (info.value.code, info.value.params) == ("ex_rm_range", {"index": 99, "total": 2})

        assert store.few_shot_version() == 1
        assert len(store.examples()) == 2
    finally:
        await db.close()


# -- seed plan (PR-2) ----------------------------------------------------------------


def test_decide_seed_rules() -> None:
    assert decide_seed("text", []) == "seed"
    assert decide_seed("text", ["other"]) == "seed"
    assert decide_seed("text", ["old", "text"]) == "keep"
    assert decide_seed("  text\n", ["text"]) == "keep"  # surrounding whitespace is ignored
    assert decide_seed("text", ["\ntext  "]) == "keep"
    assert decide_seed("tex t", ["text"]) == "seed"  # inner whitespace is not


def test_build_seed_plan_carries_hashes_and_active_versions() -> None:
    plan = build_seed_plan(
        prompt_text="p",
        few_shot_text="f",
        prompt_bodies=["p"],
        few_shot_bodies=[],
        prompt_active_version=3,
        few_shot_active_version=None,
    )

    assert (plan.prompt, plan.few_shot) == ("keep", "seed")
    assert plan.prompt_sha256 == hashlib.sha256(b"p").hexdigest()
    assert plan.prompt_active_version == 3
    assert plan.few_shot_active_version is None


async def test_plan_seed_before_and_after_load(tmp_path: Path) -> None:
    store, db, _persona = await _make_prompt_store(tmp_path)
    try:
        before = await store.plan_seed()
        assert (before.prompt, before.few_shot) == ("seed", "seed")
        assert before.prompt_active_version is None

        await store.load()

        after = await store.plan_seed()
        assert (after.prompt, after.few_shot) == ("keep", "keep")
        assert after.prompt_active_version == 1
        assert after.few_shot_active_version == 1
        assert after.prompt_sha256 == before.prompt_sha256
        # plan_seed is pure: it never created a version
        assert len(await db.prompt_versions()) == 1
    finally:
        await db.close()


async def test_config_store_patterns_carry_the_persona_language(tmp_path: Path) -> None:
    store, db = await _make_config_store(tmp_path)
    try:
        assert store.patterns().lang is not None
        assert store.patterns().lang.code == "ru"
        store.set_bot_username("x_bot")  # rebuilds the patterns
        assert store.patterns().lang is not None
    finally:
        await db.close()
