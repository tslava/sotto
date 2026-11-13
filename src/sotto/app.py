"""Сборка процесса: настройки, БД, конфиг/промпт-хранилища, бот, фоновые таски.

PLAN.md, этапы 1 и 6.
"""

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
    """stdout, формат из CLAUDE.md. Секреты сюда никогда не попадают."""
    logging.basicConfig(stream=sys.stdout, level=level, format=_LOG_FORMAT)


def load_startup_persona(settings: Settings) -> Persona:
    """Пакет персоны для процесса. Нет PERSONA_DIR и нет ALLOW_DEMO_PERSONA -> SystemExit.

    Fail-closed: потерянный монтаж на проде не должен молча запустить демо-персону в
    настоящем чате. Ошибки пакета (нет файлов, кривой манифест, слоты промптов) тоже
    останавливают процесс до подключения к БД и к Telegram.
    """
    for name in settings.deprecated_fields_set():
        logger.warning("%s задан в окружении, но игнорируется: файлы берутся из PERSONA_DIR", name)
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

        # Словари словечек (CLAUDE.md, "словечки из словаря") живут в БД: владелец
        # грузит их файлом в личку, в репозитории их нет.
        vocab_store = VocabStore(db)
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
            logger.warning("команды владельца отключены: admin_user_id не задан")

        # Погода (CLAUDE.md, "Интерфейсы: погода"): ключ API не нужен, клиент
        # создаётся всегда. Домашняя точка — из .env, а не из конфига: репозиторий
        # публичный, координаты владельца в нём не лежат. Заданы не обе — клиент
        # живёт без домашней точки: блок «за окном» пуст, а погода по месту,
        # про которое спросили в чате, работает.
        home: Place | None = None
        if settings.weather_latitude is not None and settings.weather_longitude is not None:
            home = Place(
                name=settings.weather_home_name,
                latitude=settings.weather_latitude,
                longitude=settings.weather_longitude,
            )
        else:
            logger.warning(
                "погода за окном выключена: WEATHER_LATITUDE/WEATHER_LONGITUDE не заданы"
            )
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
        # Роутер команд ПЕРВЫМ: иначе команды в разрешённом чате попали бы в гейт
        # основного роутера и записались бы в messages как обычный текст.
        dispatcher.include_router(build_commands_router(deps))
        dispatcher.include_router(build_router(deps))

        api_key = settings.openrouter_api_key
        reaction_chooser: ReactionChooser | None = None
        if api_key is None:
            logger.warning("LLM отключён, ответы не генерируются: openrouter_api_key не задан")
        else:
            # LLMClient/Judge/Responder создаются, как только есть ключ, независимо
            # от того, заданы ли llm.main_model/llm.judge_model сейчас — оба меняются
            # на горячую через /set, и Responder/Judge сами проверяют пустую модель
            # на каждом вызове (llm:no_model, judge.check -> []), а не один раз при
            # старте процесса.
            if not cfg.llm.main_model:
                logger.warning("llm.main_model не задан: ответы не генерируются до /set")
            if not cfg.llm.judge_model:
                logger.warning("судья отключён: llm.judge_model не задан")

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

        # Планировщик реакций нужен всегда: пауза перед реакцией не зависит от LLM.
        # Без чузера (ключ не задан) он бросает кубик, как до семантических реакций —
        # предупреждаем один раз на старте, чтобы semantic: true не выглядел рабочим.
        if reaction_chooser is None and cfg.behaviour.reactions.semantic:
            logger.warning("реакции выбираются кубиком: openrouter_api_key не задан")
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
            # aiogram закрывает сессию сама при штатном выходе из start_polling; повторный
            # вызов идемпотентен и здесь нужен на случай исключения до start_polling.
            await bot.session.close()
        await db.close()
