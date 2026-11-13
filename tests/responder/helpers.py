"""Shared fakes and builders for the responder tests."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from sotto.config_models import Config
from sotto.db import Database, MessageRow, PendingRow
from sotto.gate_types import GateMessage, Trigger
from sotto.judge import Judge
from sotto.llm import LLMClient
from sotto.patterns import Patterns
from sotto.responder import Responder
from sotto.stickers import Sticker
from sotto.vocab import Vocab
from sotto.weather import Place, Weather
from tests.demo import DEMO_FRAG, EN, demo_config, en_vocab, parse_en_vocab

PLACES_NONE = DEMO_FRAG.text("prompt.places_none")


SITUATION_AMBIENT = DEMO_FRAG.text("situations.ambient")


SITUATION_LATE = DEMO_FRAG.text("situations.late")


SITUATION_MORNING = DEMO_FRAG.text("situations.morning")


CHAT_ID = -100123456


BOT_USER_ID = 999


BOT_USERNAME = "gusbot"


PROMPT_TEMPLATE = (
    "You are Gus, you are {age} years old.\n"
    "Examples:\n{few_shot}\n"
    "{context}\n{recent_replies}\n{places}\n{situation}"
)


# The {life} slot is deliberately absent from the plain PROMPT_TEMPLATE: most tests do not
# touch it. This variant is for the tests that check life really reaches the system prompt.
PROMPT_TEMPLATE_WITH_LIFE = (
    "You are Gus, you are {age} years old.\n"
    "{life}\n"
    "Examples:\n{few_shot}\n"
    "{context}\n{recent_replies}\n{places}\n{situation}"
)


# The same for the {weather} slot.
PROMPT_TEMPLATE_WITH_WEATHER = (
    "You are Gus, you are {age} years old.\n"
    "{weather}\n"
    "Examples:\n{few_shot}\n"
    "{context}\n{recent_replies}\n{places}\n{situation}"
)


# The same for the {chat_memory} slot.
PROMPT_TEMPLATE_WITH_CHAT_MEMORY = (
    "You are Gus, you are {age} years old.\n"
    "{chat_memory}\n"
    "{life}\n"
    "Examples:\n{few_shot}\n"
    "{context}\n{recent_replies}\n{places}\n{situation}"
)


LONDON = ZoneInfo("Europe/London")


DAY_NOW = int(datetime(2026, 1, 10, 15, 0, tzinfo=LONDON).timestamp())


NIGHT_NOW = int(datetime(2026, 1, 10, 3, 0, tzinfo=LONDON).timestamp())


def _config() -> Config:
    cfg = demo_config()
    cfg.llm.main_model = "test/model"
    return cfg


@dataclass
class SentMessage:
    message_id: int


class FakeBot:
    """Fake ``_BotLike``: writes every call to one log so tests can check the order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []
        self.sent: list[tuple[int, str, int | None]] = []
        self.sent_stickers: list[tuple[int, str, int | None]] = []
        self._next_id = 5000

    async def send_message(
        self, chat_id: int, text: str, *, reply_to_message_id: int | None = None
    ) -> SentMessage:
        self._next_id += 1
        self.sent.append((chat_id, text, reply_to_message_id))
        self.events.append(("send", str(chat_id), text))
        return SentMessage(message_id=self._next_id)

    async def send_sticker(
        self, chat_id: int, sticker: str, *, reply_to_message_id: int | None = None
    ) -> SentMessage:
        self._next_id += 1
        self.sent_stickers.append((chat_id, sticker, reply_to_message_id))
        self.events.append(("sticker", str(chat_id), sticker))
        return SentMessage(message_id=self._next_id)

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.events.append(("typing", str(chat_id), action))


_REAL_IO_STEP_SEC = 0.001  # real pause while a coroutine is in I/O (the aiosqlite thread)


class FakeClock:
    """Virtual time: sleep() does not advance it, it waits for an explicit alarm.

    ``run_until(task)`` loops "wake the earliest sleeper, advance the time to its moment,
    give the event loop a tick" until the task finishes. ``run_until_idle()`` does the same
    until there are no sleepers and no runnable coroutines left.

    A woken coroutine may need an arbitrary number of loop switches before it finishes or
    sleeps again (real I/O such as httpx or aiosqlite, nested awaits). Counting each of those
    as a round made the needed round budget depend on the machine, so waiting for "the loop has
    not caught up with the next sleep() yet" is not counted: after each time advance the clock
    waits in tiny real pauses (see ``_drain``) until a new sleeper appears or the task is done.
    The pauses are real because the test database is aiosqlite, which runs in its own thread
    and gets no time from empty event-loop spins. If no sleeper appears and the task does not
    finish within a few real seconds, that is a genuine stall and ``run_until`` fails naming
    the live tasks. The task is also re-checked after the loop, so finishing exactly on the
    last tick of the budget is not a failure.
    """

    def __init__(self, start: int) -> None:
        self.value = float(start)
        self._waiters: list[tuple[float, asyncio.Event]] = []

    def now(self) -> int:
        return int(self.value)

    async def sleep(self, seconds: float) -> None:
        target = self.value + max(seconds, 0.0)
        event = asyncio.Event()
        self._waiters.append((target, event))
        await event.wait()

    async def _tick(self) -> None:
        if self._waiters:
            self._waiters.sort(key=lambda item: item[0])
            target, event = self._waiters.pop(0)
            if target > self.value:
                self.value = target
            event.set()
            await asyncio.sleep(0)
        else:
            # No sleepers: the coroutine is in real I/O right now (aiosqlite runs in its own
            # thread). An empty sleep(0) gives that thread no time, a tiny real pause does.
            await asyncio.sleep(_REAL_IO_STEP_SEC)

    async def _drain(self, task: asyncio.Task[None], max_wait_sec: float = 5.0) -> bool:
        """Let the coroutine catch up after a wake-up: wait in real time (steps of
        _REAL_IO_STEP_SEC) until a new sleeper appears or the task finishes. The FakeClock
        time does not move. True if that happened; False if nothing happened within
        max_wait_sec of real time: a genuine stall, not scheduler lag or the aiosqlite thread."""
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            if self._waiters or task.done():
                return True
            await asyncio.sleep(_REAL_IO_STEP_SEC)
        return bool(self._waiters) or task.done()

    async def run_until(self, task: asyncio.Task[None], max_rounds: int = 10_000) -> None:
        for _ in range(max_rounds):
            if task.done():
                return
            if not self._waiters:
                caught_up = await self._drain(task)
                if task.done():
                    return
                if not caught_up:
                    alive = sorted(
                        repr(t) for t in asyncio.all_tasks() if t is not task and not t.done()
                    )
                    raise AssertionError(
                        "FakeClock.run_until: no sleepers and no task completion "
                        f"within 5 s of real waiting; task {task!r} "
                        f"is alive; other live tasks: {alive}"
                    )
            await self._tick()
        if task.done():
            return
        raise AssertionError("FakeClock.run_until: the task did not finish")

    async def run_until_idle(self, max_rounds: int = 10_000) -> None:
        idle_streak = 0
        for _ in range(max_rounds):
            had_waiters = bool(self._waiters)
            await self._tick()
            if not had_waiters:
                idle_streak += 1
                if idle_streak >= 20:
                    return
            else:
                idle_streak = 0
        raise AssertionError("FakeClock.run_until_idle: did not settle")


async def _drive[T](clock: FakeClock, coro: Awaitable[T]) -> asyncio.Task[T]:
    """Wrap a coroutine in a Task and run it to the end through the FakeClock.

    Generic in the return type: ``announce_life``/``say`` return ``SendOutcome``, not
    ``None`` like the ordinary handlers here. ``FakeClock.run_until`` is typed for
    ``Task[None]`` because it only uses ``task.done()`` and never reads the result, which is
    exactly why the cast is safe.
    """
    task: asyncio.Task[T] = asyncio.ensure_future(coro)
    await clock.run_until(task)  # type: ignore[arg-type]  # run_until never reads .result()
    return task


Handler = Callable[[httpx.Request], httpx.Response]


def _make_llm(
    cfg: Config, db_: Database, handler: Handler
) -> tuple[LLMClient, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    client = LLMClient(api_key="test-key", cfg_getter=lambda: cfg, db=db_, http=http)
    return client, calls


def _ok_response(text: str = "Happens.") -> httpx.Response:
    content = json.dumps({"speak": True, "text": text}, ensure_ascii=False)
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5},
    }
    return httpx.Response(200, json=body)


def _varied_responses(*texts: str) -> Handler:
    """Handler that returns a different text on each call (the last one repeats after that).

    For tests that expect several replies in a row: the same text twice would be cut by the
    output filter (``dedup:jaccard``), because the ``dedup`` stage is in
    ``filters.enforce_stages`` and cuts even with ``shadow: true``.
    """
    state = {"index": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        index = min(state["index"], len(texts) - 1)
        state["index"] += 1
        return _ok_response(texts[index])

    return handler


def _silent_response() -> httpx.Response:
    content = json.dumps({"speak": False, "text": ""})
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"cost": 0.0005, "prompt_tokens": 10, "completion_tokens": 2},
    }
    return httpx.Response(200, json=body)


def _invalid_json_response() -> httpx.Response:
    body = {
        "choices": [{"message": {"content": "this is not json at all, just plain text"}}],
        "usage": {"cost": 0.0002, "prompt_tokens": 5, "completion_tokens": 2},
    }
    return httpx.Response(200, json=body)


def _checkin_response(text: str, reply_to: int | None) -> httpx.Response:
    """Model answer for the "checkin" trigger: JSON with a reply_to field, as
    JSON_REMINDER_CHECKIN really asks for."""
    content = json.dumps({"speak": True, "text": text, "reply_to": reply_to}, ensure_ascii=False)
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5},
    }
    return httpx.Response(200, json=body)


def _fail_handler(_request: httpx.Request) -> httpx.Response:
    raise AssertionError("the LLM must not have been called")


class FixedRandom:
    """Fake random.Random: .random() always returns the given value (see test_gate.py)."""

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


class MinRandom:
    """Fake random.Random that returns the minimum of every range: .random() -> 0.0 (always
    the first, fastest delay bucket), .randint(lo, hi) -> lo, .uniform(lo, hi) -> lo. For the
    cooldown-as-delay tests (earliest): the base (unshifted) delay is as small as possible, so
    they reliably tell whether the due_at shift applied."""

    def random(self) -> float:
        return 0.0

    def randint(self, lo: int, hi: int) -> int:
        return lo

    def uniform(self, lo: float, hi: float) -> float:
        return lo


class MaxRandom:
    """Fake random.Random that returns the maximum of every range: .random() -> 0.999999
    (always the last, slowest delay bucket), .randint(lo, hi) -> hi, .uniform(lo, hi) -> lo.
    For the hot-window delay cap tests (``hot_window.mention_max_delay_sec``): the base
    (uncapped) delay is clearly above the cap, so they reliably tell whether the cap applied."""

    def random(self) -> float:
        return 0.999999

    def randint(self, lo: int, hi: int) -> int:
        return hi

    def uniform(self, lo: float, hi: float) -> float:
        return lo


class FakeStickerChooser:
    """Fake ``StickerChooser`` with no real LLM call: these tests check the integration (when
    the chooser is called and what happens after), not the model's choice (test_stickers.py
    covers that)."""

    def __init__(self, sticker: Sticker | None = None) -> None:
        self.sticker = sticker
        self.calls: list[dict[str, object]] = []

    async def choose(
        self, *, reply_text: str, trigger_text: str, exclude_ids: set[int], now: int
    ) -> Sticker | None:
        self.calls.append(
            {
                "reply_text": reply_text,
                "trigger_text": trigger_text,
                "exclude_ids": exclude_ids,
                "now": now,
            }
        )
        return self.sticker


class FakePromptStore:
    """Fake stores.PromptStore: swaps the system prompt, few-shot and their versions.

    Returns fixed values, but through methods (not Responder fields), so the tests can check
    that Responder really reads prompt_store on every _respond and does not cache it in the
    constructor.
    """

    def __init__(
        self,
        *,
        prompt: str = PROMPT_TEMPLATE,
        few_shot: str = 'Gene: hi\n{"speak": true, "text": "You too."}',
        prompt_version: int = 1,
        few_shot_version: int = 1,
    ) -> None:
        self.prompt = prompt
        self.few_shot = few_shot
        self.prompt_version_value = prompt_version
        self.few_shot_version_value = few_shot_version

    def system_prompt(self) -> str:
        return self.prompt

    def prompt_version(self) -> int:
        return self.prompt_version_value

    def few_shot_text(self) -> str:
        return self.few_shot

    def few_shot_version(self) -> int:
        return self.few_shot_version_value


class FakeWeatherClient:
    """Fake WeatherClient: a snapshot and a geocoder without the network."""

    def __init__(
        self,
        snapshot: Weather | None,
        *,
        home_name: str = "Town",
        place_snapshot: Weather | None = None,
        geocoded: Place | None = None,
    ) -> None:
        self.snapshot = snapshot
        self._home_name = home_name
        self.place_snapshot = place_snapshot if place_snapshot is not None else snapshot
        self.geocoded = geocoded
        self.geocode_calls: list[str] = []

    @property
    def home_name(self) -> str:
        return self._home_name

    async def get(self, place: Place | None = None) -> Weather | None:
        return self.place_snapshot if place is not None else self.snapshot

    async def geocode(self, query: str) -> Place | None:
        self.geocode_calls.append(query)
        return self.geocoded


class FakeWeatherPlaces:
    """Fake WeatherPlaceExtractor: returns a prepared place name."""

    def __init__(self, place: str | None) -> None:
        self.place = place
        self.calls: list[str] = []

    async def extract(self, text: str, *, now: int) -> str | None:
        self.calls.append(text)
        return self.place


class FakeDiary:
    """Fake DiaryExtractor: returns a prepared fact and records the calls; ``error`` raises an
    exception, ``gate`` waits for an event (to check that the chat reply does not wait for the
    extraction)."""

    def __init__(
        self,
        fact: str | None = "went to the bathhouse",
        *,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.fact = fact
        self.error = error
        self.gate = gate
        self.calls: list[str] = []

    async def extract(self, reply_text: str, *, now: int) -> str | None:
        self.calls.append(reply_text)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.fact


async def _drain_diary(responder: Responder) -> None:
    """Wait for the diary background tasks (in production nobody waits for them)."""
    await asyncio.gather(*list(responder._diary_tasks))


def _weather_snapshot(fetched_at: int = DAY_NOW) -> Weather:
    return Weather(
        temp_now=9.4,
        code_now=3,
        today_min=4.2,
        today_max=11.6,
        today_code=3,
        tomorrow_min=-2.6,
        tomorrow_max=3.4,
        tomorrow_code=61,
        fetched_at=fetched_at,
    )


def _make_responder(
    db_: Database,
    cfg: Config,
    llm: LLMClient,
    bot: FakeBot,
    clock: FakeClock,
    *,
    seed: int = 0,
    rng: random.Random | FixedRandom | MinRandom | None = None,
    judge: Judge | None = None,
    sticker_chooser: FakeStickerChooser | None = None,
    prompt_store: FakePromptStore | None = None,
    followup: FakeFollowup | None = None,
    weather: FakeWeatherClient | None = None,
    weather_places: FakeWeatherPlaces | None = None,
    diary: FakeDiary | None = None,
    vocab: Vocab | None = None,
) -> Responder:
    patterns = Patterns(cfg.filters, cfg.persona.name_triggers, BOT_USERNAME, EN)
    return Responder(
        bot=bot,
        db=db_,
        cfg_getter=lambda: cfg,
        llm=llm,
        judge=judge,
        sticker_chooser=sticker_chooser,  # type: ignore[arg-type]
        followup=followup,  # type: ignore[arg-type]  # FakeFollowup: the narrow check/check_batch protocol
        weather=weather,  # type: ignore[arg-type]  # FakeWeatherClient: get/geocode/home_name
        weather_places=weather_places,  # type: ignore[arg-type]  # FakeWeatherPlaces: extract
        diary=diary,  # type: ignore[arg-type]  # FakeDiary: the narrow extract protocol
        vocab_getter=(lambda: vocab) if vocab is not None else None,
        patterns_getter=lambda: patterns,
        prompt_store=prompt_store if prompt_store is not None else FakePromptStore(),
        rng=rng if rng is not None else random.Random(seed),  # type: ignore[arg-type]
        chat_id=CHAT_ID,
        bot_user_id=BOT_USER_ID,
        clock=clock.now,
        sleep=clock.sleep,
        frag=DEMO_FRAG,
    )


class FakeFollowup:
    """Fake ``FollowupChecker`` for checkin: ``check`` and ``check_batch`` without an LLM.
    Records the call arguments: tests must be sure the selected row (``check``) is what gets
    checked, not the whole list.

    ``check_batch`` lets the whole batch through by default (returns all the numbers): most
    ``_maybe_checkin`` tests check the logic AFTER the prefilter (the main model's row choice,
    the repeated ``check``), not the prefilter itself, and must not lose the main model call
    because of it."""

    def __init__(self, addressed: bool, batch_addressed: list[int] | None = None) -> None:
        self.addressed = addressed
        self.batch_addressed = batch_addressed
        self.calls: list[dict[str, object]] = []
        self.batch_calls: list[dict[str, object]] = []

    async def check(
        self,
        *,
        text: str,
        display_name: str,
        context_rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
        now: int,
    ) -> bool:
        self.calls.append(
            {
                "text": text,
                "display_name": display_name,
                "context_rows": list(context_rows),
                "recent_replies": list(recent_replies),
                "now": now,
            }
        )
        return self.addressed

    async def check_batch(
        self,
        *,
        rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
        now: int,
    ) -> list[int]:
        self.batch_calls.append(
            {"rows": list(rows), "recent_replies": list(recent_replies), "now": now}
        )
        if self.batch_addressed is not None:
            return self.batch_addressed
        return list(range(1, len(rows) + 1))


async def _noop_open_hot_window(_cfg: Config, _now: int) -> None:
    """Fake ``Responder._maybe_open_hot_window``, for tests that check the counters BEFORE the
    shared tail of ``_generate_and_send`` opens or extends the hot window."""


async def _tick_until(
    clock: FakeClock, predicate: Callable[[], bool], max_rounds: int = 20_000
) -> None:
    """Tick the FakeClock until predicate() becomes True. For endless background loops
    (morning_job/spontaneous_job) that run_until cannot finish, because their task is never
    done()."""
    for _ in range(max_rounds):
        if predicate():
            return
        await clock._tick()
    raise AssertionError("_tick_until: the condition was never met")


def _gate_message(
    *, tg_message_id: int, user_id: int, text: str, created_at: int, reply_to_bot: bool = False
) -> GateMessage:
    return GateMessage(
        chat_id=CHAT_ID,
        tg_message_id=tg_message_id,
        user_id=user_id,
        is_bot=False,
        text=text,
        reply_to_bot=reply_to_bot,
        created_at=created_at,
    )


async def _insert_message(
    db_: Database,
    *,
    tg_message_id: int,
    user_id: int,
    display_name: str,
    text: str,
    created_at: int,
    reply_to: int | None = None,
) -> None:
    await db_.insert_message(
        tg_message_id=tg_message_id,
        chat_id=CHAT_ID,
        user_id=user_id,
        display_name=display_name,
        text=text,
        reply_to_tg_message_id=reply_to,
        is_bot=False,
        created_at=created_at,
    )


def _payload(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content)  # type: ignore[no-any-return]


CALLBACK_AUTHOR_ID = 5


async def _insert_bot_reply(db_: Database, *, created_at: int, tg_message_id: int = 1) -> None:
    await db_.insert_bot_reply(
        tg_message_id=tg_message_id,
        reply_to_tg_message_id=None,
        trigger="ambient",
        trigger_tg_message_id=None,
        text="a reply from Gus",
        prompt_version=1,
        few_shot_version=1,
        delay_sec=0,
        created_at=created_at,
    )


PROMPT_TEMPLATE_WITH_AVOID = (
    "You are Gus, you are {age} years old.\n"
    "Examples:\n{few_shot}\n"
    "{context}\n{recent_replies}\n{places}\n{avoid}\n{situation}"
)


async def _ambient(clock: FakeClock, responder: Responder) -> None:
    await _drive(
        clock,
        responder._respond(
            trigger=Trigger.AMBIENT, trigger_msg_id=50, user_id=None, situation="", delay_sec=0
        ),
    )


PROMPT_TEMPLATE_WITH_VOCAB = (
    "You are Gus, you are {age} years old.\n"
    "{vocab}\n"
    "Examples:\n{few_shot}\n"
    "{context}\n{recent_replies}\n{places}\n{situation}"
)


VOCAB_YAML = """
name: backyard
about: backyard slang
words:
  - {word: sloshmire, meaning: Slushy weather}
  - {word: zoofa, meaning: A call to sit down}
  - {word: hlooss, meaning: A biting wind}
"""


def _test_vocab() -> Vocab:
    return en_vocab([parse_en_vocab(VOCAB_YAML)])


async def _insert_human(db_: Database, tg_id: int, text: str, at: int) -> None:
    await db_.insert_message(
        tg_message_id=tg_id,
        chat_id=CHAT_ID,
        user_id=7,
        display_name="Gene",
        text=text,
        reply_to_tg_message_id=None,
        is_bot=False,
        created_at=at,
    )


def _pending_row(pending_id: int, trigger: str, due_at: int) -> PendingRow:
    return PendingRow(
        id=pending_id,
        trigger_tg_message_id=50,
        user_id=5,
        trigger=trigger,
        due_at=due_at,
        created_at=DAY_NOW - 60,
        done_at=None,
    )
