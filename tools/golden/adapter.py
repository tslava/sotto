"""The ONLY module that knows how to build the engine and call it.

Layout assumed here (``Pack``): a directory holding ``config.yaml``, ``few_shot.yaml``,
``prompts/*.txt``, ``CHANGELOG.md``, ``stickers.yaml``, ``places_manual.yaml``. The engine
package name (``trolobot``) appears only in the imports below; later PRs adapt this file
(package rename, pack loader) and nothing else in ``tools/golden``.

Determinism: ``time.time`` and ``asyncio.sleep`` are replaced by a virtual clock for the run,
``random.Random(1)`` drives every dice, LLM and weather go through ``httpx.MockTransport``,
the database is a temp SQLite file opened through the real ``Database``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import random
import re
import time
import traceback
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import yaml
from aiogram.types import Chat, Document, Message, PhotoSize, Sticker, User, Voice

from tools.golden.schema import Scenario, parse_offset
from trolobot import alerts as engine_alerts
from trolobot import filters as engine_filters
from trolobot import motifs as engine_motifs
from trolobot import places as engine_places
from trolobot import places_fill as engine_places_fill
from trolobot import postprocess as engine_postprocess
from trolobot import prompt as engine_prompt
from trolobot import sanitize as engine_sanitize
from trolobot import stickers_fill as engine_stickers_fill
from trolobot import vocab as engine_vocab
from trolobot import weather as engine_weather
from trolobot.alerts import OwnerAlerter
from trolobot.bot import Deps, build_router
from trolobot.chat_memory import ChatMemorizer
from trolobot.commands import build_commands_router
from trolobot.config import flatten_config, load_config
from trolobot.config_models import Config
from trolobot.db import Database, MessageRow, PlaceRow
from trolobot.diary import DiaryExtractor
from trolobot.followup import FollowupChecker
from trolobot.gate_types import GateMessage, Trigger
from trolobot.judge import Judge
from trolobot.llm import LLMClient
from trolobot.morph import name_forms, name_pattern
from trolobot.reactions import (
    REPLY_UNAVAILABLE,
    ReactionChooser,
    ReactionScheduler,
    palette_without_repeats,
)
from trolobot.responder import Responder
from trolobot.settings import Settings
from trolobot.stickers import StickerChooser, load_catalog
from trolobot.stores import ConfigStore, PromptStore
from trolobot.timeutil import day_key, week_key
from trolobot.vision import VisionDescriber
from trolobot.vocab import VocabStore
from trolobot.weather import Place, WeatherClient, WeatherPlaceExtractor

CHAT_ID = -1001234500001
OWNER_ID = 7001
BOT_USER_ID = 999001
BOT_USERNAME = "golden_bot"
DEFAULT_CLOCK = "2026-11-04 12:00:00"
RNG_SEED = 1

# A sleep this long or longer, from a background task, waits for an explicit release
# (pending reply timers, reaction pauses); shorter sleeps (debounce, typing) pass at once.
HOLD_MIN_SEC = 10.0

_REAL_SLEEP = asyncio.sleep

_DEFAULT_HOME = {"name": "Золотово", "lat": 52.1, "lon": 17.1}
_DEFAULT_FORECAST = {
    "current": {"temperature_2m": 7.4, "weather_code": 3},
    "daily": {
        "temperature_2m_min": [3.1, 1.2],
        "temperature_2m_max": [9.6, 6.0],
        "weather_code": [61, 3],
    },
}

# Prompt files of the pack, by the Settings attribute that holds their path.
_PROMPT_FILES = {
    "prompt_path": "system.txt",
    "judge_prompt_path": "judge.txt",
    "sticker_prompt_path": "sticker.txt",
    "followup_prompt_path": "followup.txt",
    "checkin_prefilter_prompt_path": "checkin_prefilter.txt",
    "name_check_prompt_path": "name_check.txt",
    "reaction_prompt_path": "reaction.txt",
    "memory_prompt_path": "memory.txt",
    "diary_prompt_path": "diary.txt",
    "jokes_prompt_path": "memory_jokes.txt",
    "threads_prompt_path": "memory_threads.txt",
    "vision_prompt_path": "vision.txt",
    "weather_place_prompt_path": "weather_place.txt",
}


# --------------------------------------------------------------------------- #
# Virtual time
# --------------------------------------------------------------------------- #


class VirtualClock:
    """Virtual wall clock plus a discrete-event ``sleep``."""

    def __init__(self, start: int) -> None:
        self.start = start
        self.value = float(start)
        self.main: asyncio.Task[Any] | None = None
        self._waiters: list[tuple[float, int, asyncio.Event, asyncio.Task[Any]]] = []
        self._seq = 0

    def now(self) -> int:
        return int(self.value)

    def elapsed(self) -> int:
        return int(self.value) - self.start

    async def sleep(self, seconds: float = 0.0, result: Any = None) -> Any:
        seconds = max(float(seconds), 0.0)
        task = asyncio.current_task()
        if seconds < HOLD_MIN_SEC or task is None or task is self.main:
            self.value += seconds
            await _REAL_SLEEP(0)
            return result
        event = asyncio.Event()
        self._seq += 1
        self._waiters.append((self.value + seconds, self._seq, event, task))
        await event.wait()
        return result

    def waiting(self, task: asyncio.Task[Any]) -> bool:
        return any(waiter[3] is task for waiter in self._waiters)

    def has_waiters(self) -> bool:
        return bool(self._waiters)

    def release_next(self, *, up_to: float | None = None) -> bool:
        if not self._waiters:
            return False
        self._waiters.sort(key=lambda item: (item[0], item[1]))
        target, _seq, event, _task = self._waiters[0]
        if up_to is not None and target > up_to:
            return False
        self._waiters.pop(0)
        self.value = max(self.value, target)
        event.set()
        return True


@contextlib.contextmanager
def _patched_runtime(clock: VirtualClock) -> Iterator[None]:
    """``time.time`` and ``asyncio.sleep`` follow the virtual clock for one scenario."""
    real_time = time.time
    real_sleep = asyncio.sleep
    time.time = lambda: clock.value  # deliberate seam: no wall clock
    asyncio.sleep = clock.sleep  # type: ignore[assignment]  # deliberate seam: virtual sleep
    try:
        yield
    finally:
        time.time = real_time
        asyncio.sleep = real_sleep


def _to_epoch(local: str, tz: str) -> int:
    parsed = datetime.fromisoformat(local.strip().replace(" ", "T"))
    return int(parsed.replace(tzinfo=ZoneInfo(tz)).timestamp())


# --------------------------------------------------------------------------- #
# Fakes: Telegram bot, LLM and weather transports
# --------------------------------------------------------------------------- #


@dataclass
class _Sent:
    message_id: int


class FakeBot:
    """Stands in for ``aiogram.Bot``: records every outgoing call, downloads fake bytes."""

    def __init__(self, clock: VirtualClock) -> None:
        self._clock = clock
        self._next_id = 9000
        self.events: list[dict[str, Any]] = []

    def _log(self, event: dict[str, Any]) -> None:
        event["t"] = self._clock.elapsed()
        if event["op"] == "typing" and self.events and self.events[-1]["op"] == "typing":
            self.events[-1]["n"] += 1
            return
        self.events.append(event)

    async def send_message(
        self, chat_id: int, text: str, *, reply_to_message_id: int | None = None
    ) -> _Sent:
        self._next_id += 1
        self._log(
            {
                "op": "send_message",
                "chat": chat_id,
                "text": text,
                "reply_to": reply_to_message_id,
                "id": self._next_id,
            }
        )
        return _Sent(self._next_id)

    async def send_sticker(
        self, chat_id: int, sticker: str, *, reply_to_message_id: int | None = None
    ) -> _Sent:
        self._next_id += 1
        self._log(
            {
                "op": "send_sticker",
                "chat": chat_id,
                "sticker": sticker,
                "reply_to": reply_to_message_id,
                "id": self._next_id,
            }
        )
        return _Sent(self._next_id)

    async def send_chat_action(self, chat_id: int, action: str) -> bool:
        self._log({"op": "typing", "chat": chat_id, "action": action, "n": 1})
        return True

    async def set_message_reaction(
        self, chat_id: int, message_id: int, reaction: Any = None
    ) -> bool:
        emojis = [getattr(item, "emoji", str(item)) for item in (reaction or [])]
        self._log({"op": "reaction", "chat": chat_id, "message": message_id, "emoji": emojis})
        return True

    async def download(self, file: str, destination: Any = None) -> io.BytesIO:
        return io.BytesIO(b"golden-photo:" + file.encode())


class Recorder:
    """Canned LLM and weather HTTP endpoints; keeps the request log."""

    def __init__(self, scenario: Scenario) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies = list(scenario.llm)
        self._weather = scenario.weather

    def _next_reply(self) -> str | dict[str, Any]:
        if self._replies:
            return self._replies.pop(0)
        return {"speak": False, "text": ""}

    def llm_handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(
            {
                "kind": "llm",
                "model": body.get("model"),
                "max_tokens": body.get("max_tokens"),
                "temperature": body.get("temperature"),
                "messages": body.get("messages"),
            }
        )
        reply = self._next_reply()
        if isinstance(reply, dict) and "status" in reply:
            payload = {"error": {"message": str(reply.get("message", "golden error"))}}
            return httpx.Response(int(reply["status"]), json=payload)
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
            },
        )

    def weather_handler(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        host = request.url.host
        if host.startswith("geocoding"):
            self.requests.append({"kind": "geocode", "params": params})
            found = self._weather.places.get(params.get("name", ""))
            if found is None:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"results": [found]})
        self.requests.append({"kind": "weather", "params": params})
        return httpx.Response(200, json=self._weather.forecast or _DEFAULT_FORECAST)


# --------------------------------------------------------------------------- #
# Engine assembly (mirrors app.main without Telegram and without polling)
# --------------------------------------------------------------------------- #


@dataclass
class Engine:
    pack: Path
    scenario: Scenario
    clock: VirtualClock
    db: Database
    settings: Settings
    bot: FakeBot
    recorder: Recorder
    rng: random.Random
    config_store: ConfigStore
    prompt_store: PromptStore
    vocab_store: VocabStore
    llm: LLMClient
    alerter: OwnerAlerter
    judge: Judge
    followup: FollowupChecker
    vision: VisionDescriber
    reaction_chooser: ReactionChooser
    reaction_scheduler: ReactionScheduler
    sticker_chooser: StickerChooser | None
    memorizer: ChatMemorizer
    weather: WeatherClient
    weather_places: WeatherPlaceExtractor
    diary: DiaryExtractor
    responder: Responder
    deps: Deps
    http_clients: list[httpx.AsyncClient] = field(default_factory=list)

    def cfg(self) -> Config:
        return self.config_store.get()

    def user_name(self, user_id: int) -> str:
        return self.scenario.seed.users.get(user_id, f"User{user_id}")

    def busy_tasks(self) -> list[asyncio.Task[Any]]:
        """Every live task except the scenario driver and tasks parked in a held sleep."""
        return [
            task
            for task in asyncio.all_tasks()
            if task is not self.clock.main and not task.done() and not self.clock.waiting(task)
        ]

    async def quiesce(self) -> None:
        """Wait until every background task is finished or parked in a held sleep."""
        quiet_rounds = 0
        for _ in range(200_000):
            if self.busy_tasks():
                quiet_rounds = 0
            else:
                quiet_rounds += 1
                if quiet_rounds >= 3:
                    return
            await _REAL_SLEEP(0.0005)
        raise RuntimeError("golden: background tasks did not settle")

    async def drain(self) -> None:
        """Run everything pending, releasing held timers in time order."""
        while True:
            await self.quiesce()
            if not self.clock.release_next():
                return

    async def advance(self, seconds: float) -> None:
        """Move virtual time forward, firing held timers that come due on the way."""
        end = self.clock.value + seconds
        while True:
            await self.quiesce()
            if not self.clock.release_next(up_to=end):
                break
        await self.quiesce()
        self.clock.value = max(self.clock.value, end)

    async def close(self) -> None:
        await self.responder.shutdown()
        await self.reaction_scheduler.shutdown()
        await self.alerter.shutdown()
        for client in self.http_clients:
            await client.aclose()
        await self.db.close()


def pack_timezone(pack: Path) -> str:
    raw = yaml.safe_load((pack / "config.yaml").read_text(encoding="utf-8"))
    return str(raw["persona"]["timezone"])


def _settings_for(pack: Path, workdir: Path, scenario: Scenario) -> Settings:
    paths: dict[str, Any] = {
        attr: pack / "prompts" / filename for attr, filename in _PROMPT_FILES.items()
    }
    weather = scenario.weather
    home = _DEFAULT_HOME if weather.home == "default" else None
    if not isinstance(weather.home, str):
        home = {"name": weather.home.name, "lat": weather.home.lat, "lon": weather.home.lon}
    return Settings(
        _env_file=None,
        bot_token="golden-token",
        admin_user_id=OWNER_ID,
        allowed_chat_id=CHAT_ID,
        openrouter_api_key="golden-key",
        google_places_key=None,
        db_path=workdir / "bot.db",
        config_path=pack / "config.yaml",
        few_shot_path=pack / "few_shot.yaml",
        stickers_path=pack / "stickers.yaml",
        changelog_path=pack / "CHANGELOG.md",
        weather_latitude=home["lat"] if home else None,
        weather_longitude=home["lon"] if home else None,
        weather_home_name=home["name"] if home else "",
        log_level="WARNING",
        **paths,
    )


async def _seed(engine_db: Database, scenario: Scenario, clock: VirtualClock, tz: str) -> None:
    seed = scenario.seed
    now = clock.now()

    def at(offset: int | str) -> int:
        return now + parse_offset(offset)

    for message in seed.messages:
        await engine_db.insert_message(
            tg_message_id=message.id,
            chat_id=CHAT_ID,
            user_id=message.user,
            display_name=seed.users.get(message.user, f"User{message.user}"),
            text=message.text,
            reply_to_tg_message_id=message.reply_to,
            is_bot=message.bot,
            created_at=at(message.at),
        )
    for reply in seed.bot_replies:
        await engine_db.insert_bot_reply(
            tg_message_id=reply.id,
            reply_to_tg_message_id=reply.reply_to,
            trigger=reply.trigger,
            trigger_tg_message_id=reply.trigger_msg,
            text=reply.text,
            prompt_version=1,
            few_shot_version=1,
            delay_sec=reply.delay,
            created_at=at(reply.at),
        )
    for key, value in seed.state.items():
        if value.startswith("@"):  # "@-30m": a timestamp relative to the scenario clock
            value = str(now + parse_offset(value[1:]))
        base, _, suffix = key.partition("@")
        if suffix == "day":
            key = day_key(base, now, tz)
        elif suffix == "week":
            key = week_key(base, now, tz)
        await engine_db.set_state(key, value)
    for life in seed.life:
        await engine_db.insert_life_event(text=life.text, created_at=at(life.at))
    for fact in seed.diary:
        await engine_db.insert_self_fact(
            text=fact.text, bot_reply_tg_message_id=None, created_at=at(fact.at)
        )
    for memory in seed.memory:
        await engine_db.insert_chat_memory(
            period_start=at(memory.start),
            period_end=at(memory.end),
            text=memory.text,
            created_at=now,
        )
    for joke in seed.jokes:
        joke_id = await engine_db.insert_joke(text=joke.text, created_at=at(joke.at))
        if joke.used_at is not None:
            await engine_db.mark_joke_used(joke_id, at(joke.used_at))
    for thread in seed.threads:
        thread_id = await engine_db.insert_people_thread(
            user_id=thread.user,
            display_name=seed.users.get(thread.user, f"User{thread.user}"),
            text=thread.text,
            created_at=at(thread.at),
        )
        if thread.closed:
            await engine_db.close_people_thread(thread_id, now)
    for place in seed.places:
        await engine_db.upsert_place(
            PlaceRow(
                place_id=place.id,
                name=place.name,
                district=place.district,
                category=place.category,
                rating=place.rating,
                reviews=place.reviews,
                price_level=place.price_level,
                quiet=place.quiet,
                fact=place.fact,
                operational=place.operational,
                refreshed_at=now,
            )
        )
    for pending in seed.pending:
        await engine_db.insert_pending(
            trigger_tg_message_id=pending.trigger_msg,
            user_id=pending.user,
            trigger=pending.trigger,
            due_at=at(pending.due),
            created_at=at(pending.created),
        )
    for night in seed.night:
        await engine_db.enqueue_night(
            tg_message_id=night.id,
            user_id=night.user,
            display_name=seed.users.get(night.user, f"User{night.user}"),
            text=night.text,
            created_at=at(night.at),
        )
    for user_id in seed.mutes:
        await engine_db.add_mute(user_id, seed.users.get(user_id, f"User{user_id}"), OWNER_ID, now)


async def build_engine(
    pack: Path, scenario: Scenario, workdir: Path, clock: VirtualClock
) -> Engine:
    tz = pack_timezone(pack)
    settings = _settings_for(pack, workdir, scenario)
    recorder = Recorder(scenario)
    rng = random.Random(RNG_SEED)
    bot = FakeBot(clock)

    db = Database(settings.db_path)
    await db.connect()
    await _seed(db, scenario, clock, tz)
    for key, value in scenario.config.items():
        await db.set_override(key, value, OWNER_ID, clock.now())

    config_store = ConfigStore(settings.config_path, db)
    await config_store.load()
    prompt_store = PromptStore(db, settings.prompt_path, settings.few_shot_path)
    await prompt_store.load()
    vocab_store = VocabStore(db)
    for raw in scenario.seed.vocab:
        await vocab_store.put(raw, clock.now())
    await vocab_store.load()
    config_store.set_bot_username(BOT_USERNAME)

    cfg = config_store.get()
    reserved = {
        cfg.persona.name,
        cfg.persona.display_name,
        *cfg.persona.name_triggers,
        BOT_USERNAME,
    } - {""}

    home: Place | None = None
    if settings.weather_latitude is not None and settings.weather_longitude is not None:
        home = Place(
            name=settings.weather_home_name,
            latitude=settings.weather_latitude,
            longitude=settings.weather_longitude,
        )
    llm_http = httpx.AsyncClient(transport=httpx.MockTransport(recorder.llm_handler), timeout=30)
    weather_http = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder.weather_handler), timeout=5
    )
    weather = WeatherClient(config_store.get, home=home, http=weather_http, clock=clock.now)

    alerter = OwnerAlerter(
        bot=bot, db=db, admin_user_id=OWNER_ID, cfg_getter=config_store.get, clock=clock.now
    )
    llm = LLMClient(
        api_key="golden-key",
        cfg_getter=config_store.get,
        db=db,
        http=llm_http,
        on_refusal=alerter.provider_refused,
        clock=clock.now,
    )

    def read(attr: str) -> str:
        path: Path = getattr(settings, attr)
        return path.read_text(encoding="utf-8")

    judge = Judge(llm, config_store.get, read("judge_prompt_path"))
    followup = FollowupChecker(
        llm,
        config_store.get,
        read("followup_prompt_path"),
        read("checkin_prefilter_prompt_path"),
        read("name_check_prompt_path"),
    )
    vision = VisionDescriber(llm, config_store.get, read("vision_prompt_path"))
    reaction_chooser = ReactionChooser(llm, config_store.get, read("reaction_prompt_path"))
    catalog = load_catalog(settings.stickers_path)
    enabled_stickers = sum(1 for sticker in catalog.stickers if sticker.enabled)
    sticker_chooser = (
        StickerChooser(llm, config_store.get, catalog, read("sticker_prompt_path"))
        if enabled_stickers
        else None
    )
    memorizer = ChatMemorizer(
        llm,
        db,
        config_store.get,
        read("memory_prompt_path"),
        chat_id=CHAT_ID,
        clock=clock.now,
        jokes_prompt=read("jokes_prompt_path"),
        threads_prompt=read("threads_prompt_path"),
        patterns_getter=config_store.patterns,
    )
    weather_places = WeatherPlaceExtractor(llm, config_store.get, read("weather_place_prompt_path"))
    diary = DiaryExtractor(llm, config_store.get, read("diary_prompt_path"))
    responder = Responder(
        bot=bot,
        db=db,
        cfg_getter=config_store.get,
        llm=llm,
        judge=judge,
        sticker_chooser=sticker_chooser,
        weather=weather,
        weather_places=weather_places,
        diary=diary,
        vocab_getter=vocab_store.get,
        patterns_getter=config_store.patterns,
        prompt_store=prompt_store,
        rng=rng,
        chat_id=CHAT_ID,
        bot_user_id=BOT_USER_ID,
        followup=followup,
        clock=clock.now,
        sleep=clock.sleep,
    )
    reaction_scheduler = ReactionScheduler(
        bot=bot,
        db=db,
        cfg_getter=config_store.get,
        chooser=reaction_chooser,
        rng=rng,
        clock=clock.now,
    )
    deps = Deps(
        settings=settings,
        config_getter=config_store.get,
        db=db,
        bot_user_id=BOT_USER_ID,
        reserved_names=reserved,
        patterns_getter=config_store.patterns,
        rng=rng,
        config_store=config_store,
        prompt_store=prompt_store,
        bot_username=BOT_USERNAME,
        bot=bot,
        weather=weather,
        vocab_store=vocab_store,
        clock=clock.now,
        followup=followup,
        vision=vision,
        reaction_scheduler=reaction_scheduler,
        sticker_catalog_enabled=enabled_stickers,
        memorizer=memorizer,
        responder=responder,
    )
    return Engine(
        pack=pack,
        scenario=scenario,
        clock=clock,
        db=db,
        settings=settings,
        bot=bot,
        recorder=recorder,
        rng=rng,
        config_store=config_store,
        prompt_store=prompt_store,
        vocab_store=vocab_store,
        llm=llm,
        alerter=alerter,
        judge=judge,
        followup=followup,
        vision=vision,
        reaction_chooser=reaction_chooser,
        reaction_scheduler=reaction_scheduler,
        sticker_chooser=sticker_chooser,
        memorizer=memorizer,
        weather=weather,
        weather_places=weather_places,
        diary=diary,
        responder=responder,
        deps=deps,
        http_clients=[llm_http, weather_http],
    )


# --------------------------------------------------------------------------- #
# Dumps
# --------------------------------------------------------------------------- #

_HASHED_COLUMNS = {("prompt_versions", "body"), ("few_shot_versions", "body_yaml")}


async def dump_db(db: Database) -> dict[str, list[dict[str, Any]]]:
    """Every non-empty table, rows in rowid order; big prompt bodies reduced to a hash."""
    conn = db._conn
    assert conn is not None
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    names = [row["name"] for row in await cursor.fetchall()]
    dump: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        cursor = await conn.execute(f"SELECT * FROM {name} ORDER BY rowid")
        rows = []
        for row in await cursor.fetchall():
            item = {key: row[key] for key in row.keys()}  # noqa: SIM118  (sqlite3.Row)
            for key in list(item):
                if (name, key) in _HASHED_COLUMNS and isinstance(item[key], str):
                    body: str = item[key]
                    item[key] = {
                        "sha256": hashlib.sha256(body.encode()).hexdigest(),
                        "len": len(body),
                    }
            rows.append(item)
        if rows:
            dump[name] = rows
    return dump


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, re.Pattern):
        return value.pattern
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return value.name
    return value


# --------------------------------------------------------------------------- #
# Telegram objects
# --------------------------------------------------------------------------- #


def _group_chat() -> Chat:
    return Chat(id=CHAT_ID, type="supergroup", title="golden chat")


def _private_chat() -> Chat:
    return Chat(id=OWNER_ID, type="private")


def _user(engine: Engine, user_id: int) -> User:
    if user_id == BOT_USER_ID:
        return User(id=BOT_USER_ID, is_bot=True, first_name="Bot", username=BOT_USERNAME)
    full = engine.user_name(user_id)
    first, _, last = full.partition(" ")
    return User(id=user_id, is_bot=False, first_name=first, last_name=last or None)


def _message(engine: Engine, spec: dict[str, Any], *, private: bool = False) -> Message:
    chat = _private_chat() if private else _group_chat()
    date = datetime.fromtimestamp(engine.clock.now(), tz=UTC)
    reply: Message | None = None
    if spec.get("reply_to") is not None:
        reply_user = BOT_USER_ID if spec.get("reply_to_bot") else int(spec.get("reply_user", 0))
        reply = Message(
            message_id=int(spec["reply_to"]),
            date=date,
            chat=chat,
            from_user=_user(engine, reply_user) if reply_user else None,
            text=spec.get("reply_text", "..."),
        )
    photo = None
    if spec.get("photo"):
        photo = [
            PhotoSize(file_id="photo-small", file_unique_id="u1", width=320, height=240),
            PhotoSize(file_id="photo-big", file_unique_id="u2", width=1280, height=960),
        ]
    sticker = (
        Sticker(
            file_id="sticker-1",
            file_unique_id="s1",
            type="regular",
            width=512,
            height=512,
            is_animated=False,
            is_video=False,
        )
        if spec.get("sticker")
        else None
    )
    voice = Voice(file_id="voice-1", file_unique_id="v1", duration=3) if spec.get("voice") else None
    document = Document(file_id="doc-1", file_unique_id="d1") if spec.get("document") else None
    user_id = int(spec.get("user", OWNER_ID))
    return Message(
        message_id=int(spec.get("msg", 1)),
        date=date,
        chat=chat,
        from_user=_user(engine, user_id),
        text=spec.get("text"),
        caption=spec.get("caption"),
        photo=photo,
        sticker=sticker,
        voice=voice,
        document=document,
        reply_to_message=reply,
    )


@contextlib.contextmanager
def _capture_answers() -> Iterator[list[dict[str, Any]]]:
    """Owner commands answer through ``Message.answer``; collect the texts instead."""
    answers: list[dict[str, Any]] = []
    original = Message.answer

    async def fake_answer(self: Message, text: str, **kwargs: Any) -> None:
        answers.append({"chat": self.chat.id, "text": text})

    Message.answer = fake_answer  # type: ignore[method-assign,assignment]  # test seam
    try:
        yield answers
    finally:
        Message.answer = original  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

ActionFn = Callable[[Engine, dict[str, Any]], Awaitable[Any]]


async def _insert_pass_message(engine: Engine, spec: dict[str, Any]) -> GateMessage:
    user = int(spec["user"])
    await engine.db.insert_message(
        tg_message_id=int(spec["msg"]),
        chat_id=CHAT_ID,
        user_id=user,
        display_name=engine.user_name(user),
        text=spec["text"],
        reply_to_tg_message_id=spec.get("reply_to"),
        is_bot=False,
        created_at=engine.clock.now(),
    )
    return GateMessage(
        chat_id=CHAT_ID,
        tg_message_id=int(spec["msg"]),
        user_id=user,
        is_bot=False,
        text=spec["text"],
        reply_to_bot=bool(spec.get("reply_to_bot", False)),
        created_at=engine.clock.now(),
    )


async def _act_gate_pass(engine: Engine, args: dict[str, Any]) -> Any:
    """Feed gate PASS events to the responder as the bot handler would, then run to idle."""
    for spec in args["passes"]:
        engine.clock.value += parse_offset(spec.get("advance", 0))
        message = await _insert_pass_message(engine, spec)
        await engine.responder.on_gate_pass(
            message, Trigger(spec["trigger"]), engine.user_name(int(spec["user"]))
        )
        await engine.quiesce()  # the debounce task finishes, pending timers stay held
    await engine.drain()
    return None


async def _act_ingest(engine: Engine, args: dict[str, Any]) -> Any:
    """Run fake Telegram messages through the real message handler."""
    handler = build_router(engine.deps).message.handlers[0].callback
    for spec in args["messages"]:
        engine.clock.value += parse_offset(spec.get("advance", 0))
        await handler(_message(engine, spec))
        await engine.quiesce()
    await engine.drain()
    return None


async def _act_command(engine: Engine, args: dict[str, Any]) -> Any:
    handler = build_commands_router(engine.deps).message.handlers[0].callback
    outputs: list[Any] = []
    for spec in args["messages"]:
        engine.clock.value += parse_offset(spec.get("advance", 0))
        private = spec.get("chat", "private") == "private"
        spec = {"user": OWNER_ID, **spec}
        with _capture_answers() as answers:
            await handler(_message(engine, spec, private=private))
        outputs.append({"command": spec.get("text") or spec.get("caption"), "answers": answers})
        await engine.quiesce()
    await engine.drain()
    return outputs


async def _act_advance(engine: Engine, args: dict[str, Any]) -> Any:
    await engine.advance(parse_offset(args["by"]))
    return None


async def _act_morning(engine: Engine, args: dict[str, Any]) -> Any:
    await engine.responder._run_morning_once()
    await engine.drain()
    return None


async def _act_spontaneous(engine: Engine, args: dict[str, Any]) -> Any:
    await engine.responder._maybe_spontaneous()
    await engine.drain()
    return None


async def _act_checkin(engine: Engine, args: dict[str, Any]) -> Any:
    await engine.responder._maybe_checkin()
    await engine.drain()
    return None


async def _act_callback(engine: Engine, args: dict[str, Any]) -> Any:
    await engine.responder._maybe_callback()
    await engine.drain()
    return None


async def _act_restore_pending(engine: Engine, args: dict[str, Any]) -> Any:
    await engine.responder.restore_pending()
    await engine.drain()
    return None


async def _act_life(engine: Engine, args: dict[str, Any]) -> Any:
    event_id = await engine.db.insert_life_event(text=args["text"], created_at=engine.clock.now())
    event = await engine.db.life_event(event_id)
    assert event is not None
    outcome = await engine.responder.announce_life(event)
    await engine.drain()
    return _jsonable(outcome)


async def _act_say(engine: Engine, args: dict[str, Any]) -> Any:
    outcome = await engine.responder.say(args["text"])
    await engine.drain()
    return _jsonable(outcome)


async def _act_preview(engine: Engine, args: dict[str, Any]) -> Any:
    user = int(args.get("user", 0))
    name = args.get("name") or engine.user_name(user)
    outcome = await engine.responder.preview(args["text"], name, user)
    await engine.drain()
    return _jsonable(outcome)


# --- group B: auxiliary model prompts ---


async def _context_rows(engine: Engine, limit: int = 10) -> list[MessageRow]:
    return await engine.db.recent_messages(CHAT_ID, limit)


async def _act_aux(engine: Engine, args: dict[str, Any]) -> Any:
    which = args["which"]
    now = engine.clock.now()
    cfg = engine.cfg()
    rows = await _context_rows(engine, int(args.get("context", 10)))
    replies = list(args.get("recent_replies", []))
    if which == "judge":
        return await engine.judge.check(
            candidate=args["candidate"], trigger_text=args.get("trigger", ""), now=now
        )
    if which == "followup_check":
        return await engine.followup.check(
            text=args["text"],
            display_name=args.get("name", "Гость"),
            context_rows=rows,
            recent_replies=replies,
            now=now,
        )
    if which == "name_check":
        return await engine.followup.check_name(
            text=args["text"],
            display_name=args.get("name", "Гость"),
            context_rows=rows,
            recent_replies=replies,
            now=now,
        )
    if which == "batch":
        return await engine.followup.check_batch(rows=rows, recent_replies=replies, now=now)
    if which == "reaction":
        recent = list(args.get("recent_reactions", []))
        reactions_cfg = cfg.behaviour.reactions
        allowed = palette_without_repeats(reactions_cfg.emoji, recent, reactions_cfg.max_repeat)
        reply_to: Any = None
        if args.get("reply_to") == "unavailable":
            reply_to = REPLY_UNAVAILABLE
        elif args.get("reply_to") is not None:
            reply_to = await engine.db.message_by_tg_id(CHAT_ID, int(args["reply_to"]))
        return await engine.reaction_chooser.choose(
            text=args["text"],
            display_name=args.get("name", "Гость"),
            context_rows=rows,
            allowed=allowed,
            now=now,
            reply_to=reply_to,
        )
    if which == "sticker":
        if engine.sticker_chooser is None:
            return {"skipped": "no enabled stickers in the pack"}
        chosen = await engine.sticker_chooser.choose(
            reply_text=args["reply"],
            trigger_text=args.get("trigger", ""),
            exclude_ids=set(args.get("exclude", [])),
            now=now,
        )
        return None if chosen is None else _jsonable(chosen.model_dump())
    if which == "vision":
        return await engine.vision.describe(
            b"golden-image-bytes", mime="image/jpeg", caption=args.get("caption", ""), now=now
        )
    if which == "weather_place":
        return await engine.weather_places.extract(args["text"], now=now)
    if which == "diary":
        return await engine.diary.extract(args["reply"], now=now)
    if which == "memory":
        start = now + parse_offset(args["start"])
        end = now + parse_offset(args["end"])
        row = await engine.memorizer.summarize_period(start, end, now=now)
        return _jsonable(row)
    if which == "places_fact":
        return await engine_places_fill.compress_fact(
            args["reviews"], llm=engine.llm, model=cfg.llm.main_model, now=now
        )
    if which == "stickers_ocr":
        return await engine_stickers_fill.recognize_sticker(
            b"golden-sticker-bytes", llm=engine.llm, model=cfg.llm.main_model, now=now
        )
    raise ValueError(f"unknown aux {which!r}")


# --- group C: deterministic functions ---


def _row(engine: Engine, spec: dict[str, Any], index: int) -> MessageRow:
    user = int(spec.get("user", 0))
    return MessageRow(
        id=index,
        tg_message_id=index,
        chat_id=CHAT_ID,
        user_id=user,
        display_name=spec.get("name") or engine.user_name(user),
        text=spec["text"],
        reply_to_tg_message_id=None,
        is_bot=False,
        created_at=engine.clock.now() - 60 * (50 - index),
    )


def _call_soften(engine: Engine, case: dict[str, Any]) -> Any:
    fixed = engine_postprocess.soften(
        case["text"], recent_replies=case.get("recent", []), cfg=engine.cfg().filters
    )
    return {"text": fixed.text, "fixes": list(fixed.fixes)}


async def _call_check_output(engine: Engine, case: dict[str, Any]) -> Any:
    cfg = engine.cfg()
    context_rows = [_row(engine, spec, i) for i, spec in enumerate(case.get("context", []))]
    ctx = engine_filters.FilterContext(
        cfg=cfg,
        recent_replies=list(case.get("recent", [])),
        context_rows=context_rows,
        places_names=list(case.get("places", [])),
        participant_names=sorted({row.display_name or "" for row in context_rows} - {""}),
        bot_names=[cfg.persona.name, cfg.persona.display_name, *cfg.persona.name_triggers],
        muted_names=list(case.get("muted", [])),
        patterns=engine.config_store.patterns(),
        system_prompt=engine.prompt_store.system_prompt(),
        trigger_text=case.get("trigger", ""),
        now=engine.clock.now(),
    )
    verdict = await engine_filters.check_output(case["text"], ctx)
    return {"ok": verdict.ok, "reason": verdict.reason, "reasons": list(verdict.reasons)}


_PATTERN_METHODS = {
    "topic_stop",
    "injection",
    "logistics",
    "name_trigger",
    "mentions_bot",
    "urgent",
    "places_request",
    "weather_request",
    "model_talk",
    "assistant_marker",
    "unsafe_note",
    "grumpy",
}


def _call_patterns(engine: Engine, case: dict[str, Any]) -> Any:
    method = case["method"]
    if method not in _PATTERN_METHODS:
        raise ValueError(f"unknown Patterns method {method!r}")
    return getattr(engine.config_store.patterns(), method)(case["text"])


def _call_sanitize_name(engine: Engine, case: dict[str, Any]) -> Any:
    return engine_sanitize.sanitize_display_name(
        case.get("raw"), int(case.get("user", 1)), engine.deps.reserved_names
    )


def _call_media_placeholder(engine: Engine, case: dict[str, Any]) -> Any:
    fields = ("sticker", "photo", "voice", "audio", "video", "video_note", "animation", "document")
    message = SimpleNamespace(
        **{name: (object() if name == case["kind"] else None) for name in fields}
    )
    return engine_sanitize.media_placeholder(message)


def _call_normalize_text(engine: Engine, case: dict[str, Any]) -> Any:
    return engine_sanitize.normalize_text(case.get("raw"))


def _call_name_forms(engine: Engine, case: dict[str, Any]) -> Any:
    word = case["word"]
    return {"pattern": name_pattern(word), "forms": name_forms(word)}


def _call_vocab(engine: Engine, case: dict[str, Any]) -> Any:
    vocab = engine.vocab_store.get()
    vocab_cfg = engine.cfg().behaviour.vocab
    text = case["text"]
    return {
        "found": [entry.word for entry in vocab.find(text)],
        "requested": engine_vocab.vocab_requested(text, vocab, vocab_cfg),
        "explain": engine_vocab.explain_requested(text, vocab_cfg),
    }


def _call_motifs(engine: Engine, case: dict[str, Any]) -> Any:
    patterns = engine.config_store.patterns()
    recent = list(case.get("recent", []))
    used = engine_motifs.used_motifs(recent, int(case.get("window", 10)), patterns.motifs)
    count = engine_motifs.story_count(recent, int(case.get("window", 10)), patterns.story_markers)
    share = engine_motifs.self_share(
        [(item["trigger"], item["text"]) for item in case.get("replies", [])],
        patterns.self_markers,
    )
    return {
        "motifs_in": engine_motifs.motifs_in(case.get("text", ""), patterns.motifs),
        "used": used,
        "story_count": count,
        "avoid": engine_motifs.render_avoid(used, no_story=bool(case.get("no_story", False))),
        "self_share": engine_motifs.render_self_share(share, 3),
        "speaks_about_self": engine_motifs.speaks_about_self(
            case.get("text", ""), patterns.self_markers
        ),
    }


def _call_validate_fact(engine: Engine, case: dict[str, Any]) -> Any:
    return engine_places.validate_fact(case["raw"])


async def _call_places_menu(engine: Engine, case: dict[str, Any]) -> Any:
    rows = await engine.db.places_all()
    return engine_places.render_places_menu(rows)


def _call_weather(engine: Engine, case: dict[str, Any]) -> Any:
    tz = engine.cfg().persona.timezone
    w = engine_weather.Weather(
        temp_now=float(case["temp"]),
        code_now=int(case["code"]),
        today_min=float(case["min"]),
        today_max=float(case["max"]),
        today_code=int(case["code"]),
        tomorrow_min=float(case["tmin"]),
        tomorrow_max=float(case["tmax"]),
        tomorrow_code=int(case["tcode"]),
        fetched_at=engine.clock.now(),
    )
    now = engine.clock.now()
    return {
        "home": engine_weather.render_weather(w, tz, now, case.get("home", "")),
        "place": engine_weather.render_weather_place(case.get("place", "Гдеугодно"), w, tz, now),
        "none": engine_weather.render_weather(None, tz, now),
        "describe": engine_weather.describe(int(case["code"])),
    }


def _call_parse_reply(engine: Engine, case: dict[str, Any]) -> Any:
    return _jsonable(engine_prompt.parse_reply(case["raw"]))


def _call_situations(engine: Engine, case: dict[str, Any]) -> Any:
    items = [(name, text) for name, text in case.get("items", [])]
    rows = [_row(engine, spec, i) for i, spec in enumerate(case.get("rows", []))]
    return {
        "addressed": engine_prompt.situation_addressed(items) if items else "",
        "followup": engine_prompt.situation_followup(items) if items else "",
        "checkin": engine_prompt.situation_checkin(rows) if rows else "",
        "life": engine_prompt.situation_life(case.get("life", "")),
        "callback": engine_prompt.situation_callback("Имя", case.get("life", "")),
    }


def _call_prompts(engine: Engine, case: dict[str, Any]) -> Any:
    system = engine.prompt_store.system_prompt()
    return {
        "few_shot_text": engine.prompt_store.few_shot_text(),
        "few_shot_examples": len(engine.prompt_store.examples()),
        "few_shot_version": engine.prompt_store.few_shot_version(),
        "prompt_version": engine.prompt_store.prompt_version(),
        "system_prompt": system,
    }


_CALLS: dict[str, Callable[[Engine, dict[str, Any]], Any]] = {
    "prompts": _call_prompts,
    "soften": _call_soften,
    "check_output": _call_check_output,
    "patterns": _call_patterns,
    "sanitize_name": _call_sanitize_name,
    "media_placeholder": _call_media_placeholder,
    "normalize_text": _call_normalize_text,
    "name_forms": _call_name_forms,
    "vocab": _call_vocab,
    "motifs": _call_motifs,
    "validate_fact": _call_validate_fact,
    "places_menu": _call_places_menu,
    "weather": _call_weather,
    "parse_reply": _call_parse_reply,
    "situations": _call_situations,
}


async def _act_call(engine: Engine, args: dict[str, Any]) -> Any:
    fn = _CALLS[args["fn"]]
    results = []
    for case in args["cases"]:
        value = fn(engine, case)
        if asyncio.iscoroutine(value):
            value = await value
        results.append({"case": case, "result": _jsonable(value)})
    return results


# --- group D: config ---


async def _act_config_dump(engine: Engine, args: dict[str, Any]) -> Any:
    base = load_config(engine.settings.config_path)
    return {
        "flat": flatten_config(engine.cfg()),
        "flat_yaml_only": flatten_config(base),
        "overrides": await engine.db.get_overrides(),
    }


async def _act_describe_keys(engine: Engine, args: dict[str, Any]) -> Any:
    cards = {}
    for key in args["keys"]:
        info = engine.config_store.describe(key)
        cards[key] = None if info is None else _jsonable(info)
    return cards


async def _act_pattern_dump(engine: Engine, args: dict[str, Any]) -> Any:
    patterns = engine.config_store.patterns()
    return {name.lstrip("_"): _jsonable(value) for name, value in sorted(vars(patterns).items())}


async def _act_alert(engine: Engine, args: dict[str, Any]) -> Any:
    out: dict[str, Any] = {}
    interval = engine.cfg().llm.alert_interval_hours
    out["rendered"] = [
        {
            "status": case["status"],
            "text": engine_alerts.render_alert(
                status=int(case["status"]),
                model=case.get("model", "vendor/model"),
                message=case.get("message", "golden message"),
                interval_hours=interval,
            ),
        }
        for case in args.get("cases", [])
    ]
    if args.get("flow"):
        # A refused main-model call must reach the owner through the alert hook.
        try:
            await engine.llm.call(
                [{"role": "user", "content": "ping"}],
                model=engine.cfg().llm.main_model,
                max_tokens=10,
                now=engine.clock.now(),
            )
        except Exception as exc:
            out["error"] = getattr(exc, "reason", type(exc).__name__)
        await engine.drain()
    return out


_ACTIONS: dict[str, ActionFn] = {
    "gate_pass": _act_gate_pass,
    "ingest": _act_ingest,
    "command": _act_command,
    "advance": _act_advance,
    "morning": _act_morning,
    "spontaneous": _act_spontaneous,
    "checkin": _act_checkin,
    "callback": _act_callback,
    "restore_pending": _act_restore_pending,
    "life": _act_life,
    "say": _act_say,
    "preview": _act_preview,
    "aux": _act_aux,
    "call": _act_call,
    "config_dump": _act_config_dump,
    "describe_keys": _act_describe_keys,
    "pattern_dump": _act_pattern_dump,
    "alert": _act_alert,
}


# --------------------------------------------------------------------------- #
# Entry point used by the runner
# --------------------------------------------------------------------------- #


async def _run(pack: Path, scenario: Scenario, workdir: Path) -> dict[str, Any]:
    tz = pack_timezone(pack)
    clock = VirtualClock(_to_epoch(scenario.clock or DEFAULT_CLOCK, tz))
    clock.main = asyncio.current_task()
    with _patched_runtime(clock):
        engine = await build_engine(pack, scenario, workdir, clock)
        try:
            results = []
            for action in scenario.actions:
                args = action.model_dump()
                kind = args.pop("kind")
                results.append(await _ACTIONS[kind](engine, args))
            await engine.drain()

            out: dict[str, Any] = {
                "id": scenario.id,
                "group": scenario.group,
                "clock": {
                    "start": datetime.fromtimestamp(clock.start, ZoneInfo(tz)).isoformat(),
                    "elapsed_sec": clock.elapsed(),
                },
            }
            capture = set(scenario.capture)
            if os.environ.get("GOLDEN_CAPTURE_ALL"):  # debugging aid, never used for baselines
                capture |= {"requests", "sent", "db", "result"}
            if "result" in capture:
                out["result"] = _jsonable(results)
            if "requests" in capture:
                out["requests"] = engine.recorder.requests
            if "sent" in capture:
                out["sent"] = engine.bot.events
            if "db" in capture:
                out["db"] = await dump_db(engine.db)
            return out
        finally:
            await engine.close()


class _ErrorCollector(logging.Handler):
    """Collects the exception types the engine logged with ``logger.exception``.

    Unexpected exceptions swallowed by background tasks would otherwise hide inside a
    scenario output that "looks fine"; their types (never their texts or logger names)
    become part of the output.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.types: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info and record.exc_info[0] is not None:
            self.types.append(record.exc_info[0].__name__)
            if os.environ.get("GOLDEN_DEBUG"):
                traceback.print_exception(*record.exc_info)


async def run_scenario(pack: Path, scenario: Scenario, workdir: Path) -> dict[str, Any]:
    """Run one scenario in the current event loop; returns the JSON-able output."""
    collector = _ErrorCollector()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(collector)
    root.setLevel(logging.ERROR)
    try:
        out = await _run(pack, scenario, workdir)
    finally:
        root.removeHandler(collector)
        root.setLevel(previous_level)
    out["logged_exceptions"] = collector.types
    return out
