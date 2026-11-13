"""packcheck CLI: pack validation, boot dry-run, read-only seed dry-run."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest
import yaml

from sotto.db import Database
from sotto.fragments import ALL_FRAGMENT_KEYS
from sotto.packcheck import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_SEED,
    EXIT_USAGE,
    boot_check,
    check_pack,
    main,
    read_seed_plan,
)
from sotto.persona import Persona, load_persona
from sotto.stores import PromptStore
from tests.packs import LEGACY_PACK


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    target = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, target)
    return target


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _seeded_db(tmp_path: Path, persona: Persona) -> Path:
    """A database that has run PromptStore.load for ``persona`` (what production has)."""
    path = tmp_path / "live.db"
    db = Database(path)
    await db.connect()
    await PromptStore(persona, db).load()
    await db.close()
    return path


# -- static checks ---------------------------------------------------------------


def test_legacy_pack_is_valid() -> None:
    assert check_pack(load_persona(LEGACY_PACK)) == []


def _set_ship(pack: Path, ship: list[str]) -> Persona:
    manifest = yaml.safe_load((pack / "pack.yaml").read_text(encoding="utf-8"))
    manifest["ship"] = ship
    (pack / "pack.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return load_persona(pack)


def test_ship_must_cover_every_file_the_loader_requires(pack: Path) -> None:
    ship = [
        "pack.yaml",
        "config.yaml",
        "few_shot.yaml",
        "fragments.yaml",
        "prompts/",
    ]
    assert check_pack(_set_ship(pack, ship)) == []
    for left_out in ("pack.yaml", "config.yaml", "few_shot.yaml", "fragments.yaml", "prompts/"):
        errors = check_pack(_set_ship(pack, [e for e in ship if e != left_out]))
        if left_out == "prompts/":
            assert len(errors) == 13
            assert errors[0] == "pack.yaml: ship does not cover required file 'prompts/system.txt'"
        else:
            assert errors == [f"pack.yaml: ship does not cover required file {left_out!r}"]


def test_ship_accepts_single_prompt_files_and_directories_without_slash(pack: Path) -> None:
    from sotto.fragments import PROMPT_NAMES

    ship = ["pack.yaml", "config.yaml", "few_shot.yaml", "fragments.yaml"]
    ship += [f"prompts/{name}.txt" for name in PROMPT_NAMES]
    assert check_pack(_set_ship(pack, ship)) == []
    ship = ["pack.yaml", "config.yaml", "few_shot.yaml", "fragments.yaml", "prompts"]
    assert check_pack(_set_ship(pack, ship)) == []


def test_main_valid_pack_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(LEGACY_PACK)]) == EXIT_OK

    assert "is valid" in capsys.readouterr().out


def test_main_missing_pack_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope")]) == EXIT_INVALID

    assert "does not exist" in capsys.readouterr().err


def test_ship_entry_that_does_not_exist(pack: Path) -> None:
    data = yaml.safe_load((pack / "pack.yaml").read_text(encoding="utf-8"))
    data["ship"].append("golden/")
    (pack / "pack.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    assert check_pack(load_persona(pack)) == ["pack.yaml: ship entry 'golden/' does not exist"]


def test_prompt_with_a_disallowed_slot(pack: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = pack / "prompts" / "vision.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\n{surprise}\n", encoding="utf-8")

    assert main([str(pack)]) == EXIT_INVALID

    assert "prompts/vision.txt: slot {surprise} is not allowed" in capsys.readouterr().err


def test_prompt_with_a_missing_required_slot(pack: Path) -> None:
    (pack / "prompts" / "judge.txt").write_text("no slots", encoding="utf-8")

    errors = check_pack(load_persona(pack))

    assert "prompts/judge.txt: required slot {candidate} is missing" in errors
    assert "prompts/judge.txt: required slot {trigger} is missing" in errors


def test_invalid_config(pack: Path) -> None:
    (pack / "config.yaml").write_text("behaviour:\n  daily_cap: 9999\n", encoding="utf-8")

    errors = check_pack(load_persona(pack))

    assert len(errors) == 1
    assert errors[0].startswith("config.yaml: invalid config: behaviour.daily_cap")


def test_invalid_few_shot_stickers_and_places(pack: Path) -> None:
    (pack / "few_shot.yaml").write_text("not: a list\n", encoding="utf-8")
    (pack / "stickers.yaml").write_text("stickers: 5\n", encoding="utf-8")
    (pack / "places_manual.yaml").write_text("- 5\n", encoding="utf-8")

    errors = check_pack(load_persona(pack))

    assert [e.split(":")[0] for e in errors] == [
        "few_shot.yaml",
        "stickers.yaml",
        "places_manual.yaml",
    ]


def test_missing_catalog_keys_are_an_error(pack: Path) -> None:
    persona = load_persona(pack)
    persona.catalog.t("never.defined")

    assert check_pack(persona) == ["catalog: missing keys ['never.defined']"]


def test_fragment_spec_is_checked(pack: Path) -> None:
    data = yaml.safe_load((LEGACY_PACK / "fragments.yaml").read_text(encoding="utf-8"))
    del data["vocab"]["offer_tail"]
    data["situations"]["life"] = "no slot at all"
    data["placeholders"]["extra"] = "unknown"
    (pack / "fragments.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    errors = check_pack(load_persona(pack))

    assert errors == [
        "fragments.yaml: missing key vocab.offer_tail",
        "fragments.yaml: unknown key placeholders.extra",
        "fragments.yaml: situations.life placeholders are [none], expected [text]",
    ]


def test_pack_without_fragments_reports_every_key(pack: Path) -> None:
    (pack / "fragments.yaml").unlink()

    errors = check_pack(load_persona(pack))

    # The manifest still ships the file that is gone, then every fragment key is missing.
    assert errors[0] == "pack.yaml: ship entry 'fragments.yaml' does not exist"
    assert len(errors) == len(ALL_FRAGMENT_KEYS) + 1
    assert all(error.startswith("fragments.yaml: missing key ") for error in errors[1:])


# -- boot ------------------------------------------------------------------------


async def test_boot_check_builds_the_wiring_for_the_legacy_pack() -> None:
    assert await boot_check(load_persona(LEGACY_PACK)) == []


async def test_boot_check_reports_a_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sotto.wiring

    def explode(**kwargs: object) -> None:
        raise RuntimeError("wiring broke")

    monkeypatch.setattr(sotto.wiring, "build_llm_stack", explode)

    assert await boot_check(load_persona(LEGACY_PACK)) == ["boot: RuntimeError: wiring broke"]


def test_main_boot_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(LEGACY_PACK), "--boot"]) == EXIT_OK


def test_main_boot_failure_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sotto.wiring

    def explode(**kwargs: object) -> None:
        raise RuntimeError("wiring broke")

    monkeypatch.setattr(sotto.wiring, "build_llm_stack", explode)

    assert main([str(LEGACY_PACK), "--boot"]) == EXIT_INVALID

    assert "wiring broke" in capsys.readouterr().err


# -- seed dry-run ------------------------------------------------------------------


async def test_seed_dry_run_passes_when_the_database_matches_the_pack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    persona = load_persona(LEGACY_PACK)
    db_path = await _seeded_db(tmp_path, persona)

    assert main([str(LEGACY_PACK), "--db", str(db_path), "--seed-dry-run"]) == EXIT_OK

    assert "prompt=keep few_shot=keep" in capsys.readouterr().out


async def test_seed_dry_run_is_read_only(tmp_path: Path) -> None:
    db_path = await _seeded_db(tmp_path, load_persona(LEGACY_PACK))
    before = _sha(db_path)
    wal = tmp_path / "live.db-wal"
    wal_size = wal.stat().st_size if wal.exists() else 0

    assert main([str(LEGACY_PACK), "--db", str(db_path), "--seed-dry-run"]) == EXIT_OK

    # SQLite may create the empty -wal/-shm companions of a WAL database; no page is written.
    assert _sha(db_path) == before
    assert (wal.stat().st_size if wal.exists() else 0) == wal_size


async def test_seed_dry_run_fails_on_a_changed_prompt(
    tmp_path: Path, pack: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = await _seeded_db(tmp_path, load_persona(pack))
    system = pack / "prompts" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")

    assert main([str(pack), "--db", str(db_path), "--seed-dry-run"]) == EXIT_SEED

    captured = capsys.readouterr()
    assert "prompt=seed few_shot=keep" in captured.out
    assert "would seed a new version of: prompt" in captured.err


async def test_seed_dry_run_allow_seed_accepts_an_intended_change(
    tmp_path: Path, pack: Path
) -> None:
    db_path = await _seeded_db(tmp_path, load_persona(pack))
    few_shot = pack / "few_shot.yaml"
    few_shot.write_text(few_shot.read_text(encoding="utf-8") + "\n# edit\n", encoding="utf-8")
    argv = [str(pack), "--db", str(db_path), "--seed-dry-run"]

    few_shot.write_text("- {name: A, user: b, speak: true, text: c}\n", encoding="utf-8")
    assert main(argv) == EXIT_SEED
    assert main([*argv, "--allow-seed", "few_shot"]) == EXIT_OK


async def test_seed_dry_run_whitespace_only_change_does_not_seed(
    tmp_path: Path, pack: Path
) -> None:
    db_path = await _seeded_db(tmp_path, load_persona(pack))
    system = pack / "prompts" / "system.txt"
    system.write_text("\n" + system.read_text(encoding="utf-8").rstrip() + "\n\n", encoding="utf-8")

    assert main([str(pack), "--db", str(db_path), "--seed-dry-run"]) == EXIT_OK


async def test_seed_dry_run_matches_the_real_load(tmp_path: Path, pack: Path) -> None:
    """The dry-run decision is the one ``PromptStore.load`` then acts on."""
    db_path = await _seeded_db(tmp_path, load_persona(pack))
    system = pack / "prompts" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
    persona = load_persona(pack)

    plan = read_seed_plan(persona, db_path)
    db = Database(db_path)
    await db.connect()
    try:
        store = PromptStore(persona, db)
        assert (await store.plan_seed()) == plan
        await store.load()
        assert store.prompt_version() == 2
    finally:
        await db.close()
    assert plan.prompt == "seed"


def test_seed_dry_run_on_a_database_without_tables_plans_a_seed(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    sqlite3.connect(db_path).close()

    plan = read_seed_plan(load_persona(LEGACY_PACK), db_path)

    assert (plan.prompt, plan.few_shot) == ("seed", "seed")
    assert plan.prompt_active_version is None


def test_seed_dry_run_needs_a_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(LEGACY_PACK), "--seed-dry-run"]) == EXIT_USAGE
    assert main([str(LEGACY_PACK), "--db", str(tmp_path / "x.db"), "--seed-dry-run"]) == EXIT_USAGE

    err = capsys.readouterr().err
    assert "needs --db" in err
    assert "not found" in err


def test_seed_dry_run_unreadable_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"this is not a sqlite database" * 100)

    assert main([str(LEGACY_PACK), "--db", str(garbage), "--seed-dry-run"]) == EXIT_USAGE

    assert "cannot read" in capsys.readouterr().err
