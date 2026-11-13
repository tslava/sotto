"""Тесты для sotto.app. main() не запускается — только setup_logging.

Полная сборка процесса (Settings -> ConfigStore -> PromptStore -> Bot -> Deps ->
Dispatcher -> Responder) проверяется сквозным тестом в
tests/test_commands_integration.py, не здесь: main() требует реального aiogram
Bot.get_me() (сетевой вызов), поэтому напрямую не вызывается ни в одном тесте.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from sotto.app import load_startup_persona, setup_logging
from sotto.settings import Settings
from tests.packs import LEGACY_PACK


def test_setup_logging_does_not_raise() -> None:
    setup_logging("DEBUG")
    logging.getLogger("sotto").info("still works")


def test_setup_logging_accepts_default_level() -> None:
    setup_logging("INFO")


# -- persona selection at startup (fail-closed) ---------------------------------


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str) -> Settings:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.delenv("PERSONA_DIR", raising=False)
    monkeypatch.delenv("ALLOW_DEMO_PERSONA", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_startup_persona_loads_from_persona_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(monkeypatch, tmp_path, PERSONA_DIR=str(LEGACY_PACK))

    with caplog.at_level(logging.INFO, logger="sotto.app"):
        persona = load_startup_persona(settings)

    assert persona.name == "legacy_ru"
    assert "persona: legacy_ru rev dev" in caplog.text


def test_startup_without_persona_dir_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)

    with pytest.raises(SystemExit, match="PERSONA_DIR not set"):
        load_startup_persona(settings)


def test_startup_demo_opt_in_without_demo_pack_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path, ALLOW_DEMO_PERSONA="true")
    monkeypatch.setattr("sotto.persona.DEMO_PACK_DIR", tmp_path / "no-demo")

    with pytest.raises(SystemExit, match="persona pack error"):
        load_startup_persona(settings)


def test_startup_broken_pack_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path, PERSONA_DIR=str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()

    with pytest.raises(SystemExit, match="persona pack error"):
        load_startup_persona(settings)


def test_startup_bad_prompt_slots_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    (pack / "prompts" / "judge.txt").write_text("no slots", encoding="utf-8")
    settings = _settings(monkeypatch, tmp_path, PERSONA_DIR=str(pack))

    with pytest.raises(SystemExit, match=r"required slot \{candidate\}"):
        load_startup_persona(settings)


def test_startup_warns_about_deprecated_path_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(
        monkeypatch,
        tmp_path,
        PERSONA_DIR=str(LEGACY_PACK),
        CONFIG_PATH="/old/config.yaml",
        STICKERS_PATH="/old/stickers.yaml",
    )

    with caplog.at_level(logging.WARNING, logger="sotto.app"):
        load_startup_persona(settings)

    warned = [r.message for r in caplog.records if "игнорируется" in r.message]
    assert [m.split()[0] for m in warned] == ["CONFIG_PATH", "STICKERS_PATH"]


async def test_main_exits_before_touching_the_database_without_persona(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sotto.app import main

    _settings(monkeypatch, tmp_path, DB_PATH=str(tmp_path / "bot.db"))

    with pytest.raises(SystemExit, match="PERSONA_DIR not set"):
        await main()

    assert not (tmp_path / "bot.db").exists()
