"""Persona pack validator: ``python -m sotto.packcheck DIR [--boot] [--db PATH --seed-dry-run]``.

Prints file names and counts only, never prompt or config text (the pack may be private).

Exit codes: 0 pack is fine; 1 the pack is invalid; 2 bad usage or unreadable database;
3 the seed dry-run found that starting would create a new prompt / few-shot version.

Checks, in order: the pack loads (manifest, required files), every ``ship`` entry exists,
every prompt has exactly the slots it may have (``fragments.PROMPT_SLOTS``), the config
validates through ``load_config(layers)``, few-shot / stickers / manual places parse and
the catalog has no missing keys. ``--boot`` then builds the process wiring (stores, checkers,
responder, routers) against a temporary database and a dummy API key, without any network call.
``--seed-dry-run`` opens ``--db`` read-only and reports what ``PromptStore.load`` would do.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from sotto.config import load_config
from sotto.persona import PackError, Persona, load_persona
from sotto.stores import SeedPlan, build_seed_plan

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2
EXIT_SEED = 3

SEED_TARGETS = ("prompt", "few_shot")


def check_pack(persona: Persona) -> list[str]:
    """Static checks of a loaded pack (no database, no wiring). Empty list = fine."""
    errors: list[str] = []

    for entry in persona.manifest.ship:
        if not (persona.root / entry).exists():
            errors.append(f"pack.yaml: ship entry {entry!r} does not exist")

    errors += persona.prompt_errors()

    try:
        load_config(persona.config_layers())
    except ValueError as exc:
        errors.append(f"config.yaml: {exc}")

    # Imported here: these modules pull in more of the engine than the loader needs.
    from sotto.few_shot import load_few_shot
    from sotto.places_fill import load_manual
    from sotto.stickers import load_catalog

    try:
        load_few_shot(persona.few_shot_path)
    except ValueError as exc:
        errors.append(f"few_shot.yaml: {exc}")
    try:
        load_catalog(persona.stickers_path)
    except ValueError as exc:
        errors.append(f"stickers.yaml: {exc}")
    try:
        load_manual(persona.places_manual_path)
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"places_manual.yaml: {exc}")

    errors += persona.fragment_errors()

    if persona.catalog.missing:
        errors.append(f"catalog: missing keys {sorted(persona.catalog.missing)}")
    return errors


def read_seed_plan(persona: Persona, db_path: Path) -> SeedPlan:
    """``PromptStore`` seed decision on a database opened read-only (no migration, no write)."""
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:

        def rows(sql: str) -> list[tuple[Any, ...]]:
            try:
                return conn.execute(sql).fetchall()
            except sqlite3.OperationalError:
                return []  # table not created yet: no versions stored

        prompt_bodies = [str(r[0]) for r in rows("SELECT body FROM prompt_versions")]
        few_shot_bodies = [str(r[0]) for r in rows("SELECT body_yaml FROM few_shot_versions")]
        prompt_active = rows("SELECT version FROM prompt_versions WHERE active = 1 LIMIT 1")
        few_shot_active = rows("SELECT version FROM few_shot_versions WHERE active = 1 LIMIT 1")
    finally:
        conn.close()
    return build_seed_plan(
        prompt_text=persona.prompts["system"],
        few_shot_text=persona.few_shot_path.read_text(encoding="utf-8"),
        prompt_bodies=prompt_bodies,
        few_shot_bodies=few_shot_bodies,
        prompt_active_version=int(prompt_active[0][0]) if prompt_active else None,
        few_shot_active_version=int(few_shot_active[0][0]) if few_shot_active else None,
    )


class _NullBot:
    """Stands in for aiogram.Bot in the boot dry-run: nothing is ever sent."""

    async def send_message(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("packcheck --boot must not send messages")

    async def send_sticker(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("packcheck --boot must not send messages")

    async def send_chat_action(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("packcheck --boot must not send messages")

    async def set_message_reaction(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("packcheck --boot must not send messages")

    async def download(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("packcheck --boot must not download files")


async def boot_check(persona: Persona) -> list[str]:
    """Build the process wiring for ``persona`` on a temp database. Empty list = it boots."""
    # Imported here: the wiring pulls in aiogram and the whole engine.
    from sotto.bot import Deps, build_router
    from sotto.commands import build_commands_router
    from sotto.db import Database
    from sotto.reactions import ReactionScheduler
    from sotto.stores import ConfigStore, PromptStore
    from sotto.vocab import VocabStore
    from sotto.weather import WeatherClient
    from sotto.wiring import build_llm_stack

    def no_network(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("packcheck --boot must not use the network")

    errors: list[str] = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
    weather: WeatherClient | None = None
    with tempfile.TemporaryDirectory(prefix="packcheck-") as tmp:
        db = Database(Path(tmp) / "boot.db")
        try:
            await db.connect()
            config_store = ConfigStore(persona, db)
            await config_store.load()
            prompt_store = PromptStore(persona, db)
            await prompt_store.load()
            vocab_store = VocabStore(db)
            await vocab_store.load()
            config_store.set_bot_username("packcheck_bot")
            cfg = config_store.get()
            rng = random.Random(0)
            bot = _NullBot()
            weather = WeatherClient(config_store.get, http=http)
            stack = build_llm_stack(
                persona=persona,
                config_store=config_store,
                prompt_store=prompt_store,
                db=db,
                bot=bot,
                weather=weather,
                vocab_store=vocab_store,
                rng=rng,
                api_key="packcheck-dummy-key",
                http=http,
                admin_user_id=0,
                chat_id=0,
                bot_user_id=1,
            )
            deps = Deps(
                settings=_boot_settings(),
                config_getter=config_store.get,
                db=db,
                bot_user_id=1,
                reserved_names={cfg.persona.name, cfg.persona.display_name}
                | set(cfg.persona.name_triggers),
                patterns_getter=config_store.patterns,
                rng=rng,
                config_store=config_store,
                prompt_store=prompt_store,
                bot_username="packcheck_bot",
                persona=persona,
                bot=bot,
                weather=weather,
                vocab_store=vocab_store,
                followup=stack.followup,
                vision=stack.vision,
                sticker_catalog_enabled=stack.sticker_catalog_enabled,
                memorizer=stack.memorizer,
                responder=stack.responder,
            )
            deps.reaction_scheduler = ReactionScheduler(
                bot=bot,
                db=db,
                cfg_getter=config_store.get,
                chooser=stack.reaction_chooser,
                rng=rng,
            )
            build_commands_router(deps)
            build_router(deps)
        except Exception as exc:
            errors.append(f"boot: {type(exc).__name__}: {exc}")
        finally:
            await http.aclose()
            if weather is not None:
                await weather.aclose()
            await db.close()
    return errors


def _boot_settings() -> Any:
    from sotto.settings import Settings

    return Settings(_env_file=None, bot_token="packcheck-dummy-token")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sotto.packcheck", description=__doc__)
    parser.add_argument("pack", type=Path, help="persona pack directory")
    parser.add_argument("--boot", action="store_true", help="build the process wiring (no network)")
    parser.add_argument("--db", type=Path, default=None, help="bot database for --seed-dry-run")
    parser.add_argument(
        "--seed-dry-run",
        action="store_true",
        help="fail (exit 3) if starting on --db would seed a new prompt/few-shot version",
    )
    parser.add_argument(
        "--allow-seed",
        action="append",
        default=[],
        choices=SEED_TARGETS,
        help="an intended change: do not fail on a seed of this kind (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed_dry_run and args.db is None:
        print("packcheck: --seed-dry-run needs --db", file=sys.stderr)
        return EXIT_USAGE

    try:
        persona = load_persona(args.pack)
    except PackError as exc:
        print(f"packcheck: {exc}", file=sys.stderr)
        return EXIT_INVALID

    errors = check_pack(persona)
    if not errors and args.boot:
        errors = asyncio.run(boot_check(persona))
    if errors:
        for line in errors:
            print(f"packcheck: {line}", file=sys.stderr)
        return EXIT_INVALID
    print(f"packcheck: pack {persona.name!r} ({persona.manifest.language}) is valid")

    if args.seed_dry_run:
        if not args.db.is_file():
            print(f"packcheck: database {args.db} not found", file=sys.stderr)
            return EXIT_USAGE
        try:
            plan = read_seed_plan(persona, args.db)
        except sqlite3.Error as exc:
            print(f"packcheck: cannot read {args.db}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        unexpected = [
            name
            for name, action in (("prompt", plan.prompt), ("few_shot", plan.few_shot))
            if action == "seed" and name not in args.allow_seed
        ]
        print(
            f"packcheck: seed plan: prompt={plan.prompt} few_shot={plan.few_shot} "
            f"(sha256 prompt {plan.prompt_sha256[:12]}, few_shot {plan.few_shot_sha256[:12]})"
        )
        if unexpected:
            print(
                f"packcheck: starting would seed a new version of: {', '.join(unexpected)}",
                file=sys.stderr,
            )
            return EXIT_SEED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
