"""Process assembly: settings, DB, config/prompt stores, bot, background tasks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import sys

import httpx
from aiogram import Bot, Dispatcher

from sotto.alerts import OwnerAlerter
from sotto.bot import Deps, build_router
from sotto.commands import build_commands_router
from sotto.db import Database
from sotto.llm import LLMClient
from sotto.persona import PackError, Persona, load_persona, resolve_persona_dir
from sotto.reactions import ReactionChooser, ReactionScheduler
from sotto.responder import Responder
from sotto.retention import retention_loop
from sotto.settings import Settings
from sotto.stores import ConfigStore, PromptStore
from sotto.vocab import VocabStore
from sotto.weather import Place, WeatherClient
from sotto.wiring import build_llm_stack

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: str) -> None:
    """stdout, format from CLAUDE.md. Secrets never reach it."""
    logging.basicConfig(stream=sys.stdout, level=level, format=_LOG_FORMAT)


def load_startup_persona(settings: Settings) -> Persona:
    """Persona pack for the process. No PERSONA_DIR and no ALLOW_DEMO_PERSONA -> SystemExit.

    Fail-closed: a lost mount in production must not silently start the demo persona in
    the real chat. Pack errors (missing files, broken manifest, prompt slots) also stop
    the process before connecting to the DB and Telegram.
    """
    for name in settings.deprecated_fields_set():
        logger.warning(
            "%s is set in the environment but ignored: files come from PERSONA_DIR", name
        )
    try:
        persona = load_persona(
            resolve_persona_dir(settings.persona_dir, settings.allow_demo_persona)
        )
        errors = persona.prompt_errors() + persona.fragment_errors()
    except PackError as exc:
        raise SystemExit(f"persona pack error: {exc}") from exc
    if errors:
        raise SystemExit("persona pack error: " + "; ".join(errors))
    logger.info("persona: %s rev %s (%s)", persona.name, persona.revision, persona.root)
    return persona


async def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    persona = load_startup_persona(settings)

    db = Database(settings.db_path)
    await db.connect()

    bot: Bot | None = None
    retention_task: asyncio.Task[None] | None = None
    morning_task: asyncio.Task[None] | None = None
    spontaneous_task: asyncio.Task[None] | None = None
    checkin_task: asyncio.Task[None] | None = None
    callback_task: asyncio.Task[None] | None = None
    memory_task: asyncio.Task[None] | None = None
    llm: LLMClient | None = None
    alerter: OwnerAlerter | None = None
    responder: Responder | None = None
    reaction_scheduler: ReactionScheduler | None = None
    weather: WeatherClient | None = None
    try:
        config_store = ConfigStore(persona, db)
        await config_store.load()

        prompt_store = PromptStore(persona, db)
        await prompt_store.load()

        # Vocabulary dictionaries live in the DB: the owner uploads them as a file in
        # a DM; they are not in the repository.
        vocab_store = VocabStore(db, persona.lang)
        await vocab_store.load()

        bot = Bot(token=settings.bot_token.get_secret_value())
        me = await bot.get_me()
        config_store.set_bot_username(me.username or "")

        cfg = config_store.get()
        reserved_names = {
            cfg.persona.name,
            cfg.persona.display_name,
            *cfg.persona.name_triggers,
            me.username or "",
        } - {""}

        rng = random.Random()

        if settings.admin_user_id == 0:
            logger.warning("owner commands disabled: admin_user_id is not set")

        # Weather: no API key needed, so the client is always created. The home point
        # comes from .env, not from config: the repository is public and the owner's
        # coordinates are not in it. If only one coordinate is set the client runs
        # without a home point: the "outside the window" block is empty, but weather
        # for a place asked about in the chat still works.
        home: Place | None = None
        if settings.weather_latitude is not None and settings.weather_longitude is not None:
            home = Place(
                name=settings.weather_home_name,
                latitude=settings.weather_latitude,
                longitude=settings.weather_longitude,
            )
        else:
            logger.warning("weather disabled: WEATHER_LATITUDE/WEATHER_LONGITUDE are not set")
        weather = WeatherClient(config_store.get, home=home)

        deps = Deps(
            settings=settings,
            config_getter=config_store.get,
            db=db,
            bot_user_id=me.id,
            reserved_names=reserved_names,
            patterns_getter=config_store.patterns,
            rng=rng,
            config_store=config_store,
            prompt_store=prompt_store,
            frag=persona.frag,
            bot_username=me.username or "",
            persona=persona,
            bot=bot,
            weather=weather,
            vocab_store=vocab_store,
        )

        dispatcher = Dispatcher()
        # Commands router FIRST: otherwise commands in the allowed chat would reach the
        # gate of the main router and be recorded in messages as ordinary text.
        dispatcher.include_router(build_commands_router(deps))
        dispatcher.include_router(build_router(deps))

        api_key = settings.openrouter_api_key
        reaction_chooser: ReactionChooser | None = None
        if api_key is None:
            logger.warning("LLM disabled, no replies are generated: openrouter_api_key is not set")
        else:
            # LLMClient/Judge/Responder are created as soon as there is a key,
            # regardless of whether llm.main_model/llm.judge_model are set now: both
            # change hot via /set, and Responder/Judge check for an empty model on every
            # call (llm:no_model, judge.check -> []), not once at process start.
            if not cfg.llm.main_model:
                logger.warning("llm.main_model is not set: no replies are generated until /set")
            if not cfg.llm.judge_model:
                logger.warning("judge disabled: llm.judge_model is not set")

            http = httpx.AsyncClient(timeout=cfg.llm.timeout_sec)
            stack = build_llm_stack(
                persona=persona,
                config_store=config_store,
                prompt_store=prompt_store,
                db=db,
                bot=bot,
                weather=weather,
                vocab_store=vocab_store,
                rng=rng,
                api_key=api_key.get_secret_value(),
                http=http,
                admin_user_id=settings.admin_user_id,
                chat_id=settings.allowed_chat_id,
                bot_user_id=me.id,
            )
            llm = stack.llm
            alerter = stack.alerter
            responder = stack.responder
            reaction_chooser = stack.reaction_chooser
            deps.followup = stack.followup
            deps.vision = stack.vision
            deps.sticker_catalog_enabled = stack.sticker_catalog_enabled
            deps.memorizer = stack.memorizer
            deps.responder = responder

        # The reaction scheduler is always needed: the pause before a reaction does not
        # depend on the LLM. Without a chooser (no key) it rolls dice, as before semantic
        # reactions; warn once at startup so semantic: true does not look functional.
        if reaction_chooser is None and cfg.behaviour.reactions.semantic:
            logger.warning("reactions are picked by dice: openrouter_api_key is not set")
        reaction_scheduler = ReactionScheduler(
            bot=bot,
            db=db,
            cfg_getter=config_store.get,
            chooser=reaction_chooser,
            rng=rng,
        )
        deps.reaction_scheduler = reaction_scheduler

        retention_task = asyncio.create_task(retention_loop(db, config_store.get))
        if responder is not None:
            await responder.restore_pending()
            morning_task = asyncio.create_task(responder.morning_job())
            spontaneous_task = asyncio.create_task(responder.spontaneous_job())
            checkin_task = asyncio.create_task(responder.checkin_job())
            callback_task = asyncio.create_task(responder.callback_job())
        if deps.memorizer is not None:
            memory_task = asyncio.create_task(deps.memorizer.job())

        logger.info(
            "started as @%s, allowed_chat_id=%s, discovery=%s",
            me.username,
            settings.allowed_chat_id,
            settings.allowed_chat_id == 0,
        )

        await dispatcher.start_polling(bot, allowed_updates=["message", "edited_message"])
    finally:
        for task in (
            memory_task,
            callback_task,
            checkin_task,
            spontaneous_task,
            morning_task,
            retention_task,
        ):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if responder is not None:
            await responder.shutdown()
        if reaction_scheduler is not None:
            await reaction_scheduler.shutdown()
        if alerter is not None:
            await alerter.shutdown()
        if llm is not None:
            await llm.aclose()
        if weather is not None:
            await weather.aclose()
        if bot is not None:
            # aiogram closes the session itself on a normal exit from start_polling; a
            # repeat call is idempotent and is needed here for an exception before start_polling.
            await bot.session.close()
        await db.close()
