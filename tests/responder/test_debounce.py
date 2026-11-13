"""Responder tests: debounce."""

from __future__ import annotations

import asyncio
import random

import httpx

from sotto.db import Database
from sotto.gate_types import Trigger
from sotto.llm import LLMClient
from sotto.patterns import Patterns
from sotto.responder import Responder
from sotto.timeutil import day_key
from tests.demo import DEMO_FRAG, EN
from tests.responder.helpers import (
    BOT_USER_ID,
    BOT_USERNAME,
    CHAT_ID,
    DAY_NOW,
    FakeBot,
    FakeClock,
    FakePromptStore,
    _config,
    _gate_message,
    _insert_message,
    _make_llm,
    _make_responder,
    _ok_response,
)


# Three ambient PASSes within 2 seconds produce one reply.
async def test_ambient_debounce_collapses_into_one_reply(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        for i in range(3):
            await _insert_message(
                db,
                tg_message_id=100 + i,
                user_id=5 + i,
                display_name=f"User{i}",
                text="hello everyone",
                created_at=DAY_NOW + i,
            )
            msg = _gate_message(
                tg_message_id=100 + i, user_id=5 + i, text="hello everyone", created_at=DAY_NOW + i
            )
            await responder.on_gate_pass(msg, Trigger.AMBIENT, f"User{i}")

        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Happens."
        assert bot.sent[0][2] is None  # ambient always goes to the stream, not as a reply

        tz = cfg.persona.timezone
        ambient_key = day_key("ambient_count", clock.now(), tz)
        assert await db.get_state(ambient_key) == "1"
    finally:
        await responder.shutdown()
        await llm.aclose()


# A second ambient PASS waits for the lock while the first hangs in the LLM; after the lock
# is released the cooldown recheck cuts it.
async def test_ambient_generation_is_serialized_and_recheck_blocks_second(
    db: Database,
) -> None:
    cfg = _config()
    # The hot window is off on purpose: this test is about generation serialization and the
    # recheck of the ambient daily budget/cooldown, not about the hot window. With it on, the
    # first send would open the window and lift the cooldown for the second, mixing two
    # independent behaviours in one test. The hot window has its own tests.
    cfg.behaviour.hot_window.enabled = False
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        started.set()
        await release.wait()
        return _ok_response("Happens.")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = LLMClient(api_key="test-key", cfg_getter=lambda: cfg, db=db, http=http)
    bot = FakeBot()

    async def instant_sleep(_seconds: float) -> None:
        await asyncio.sleep(0)

    patterns = Patterns(cfg.filters, cfg.persona.name_triggers, BOT_USERNAME, EN)
    responder = Responder(
        bot=bot,
        db=db,
        cfg_getter=lambda: cfg,
        llm=llm,
        patterns_getter=lambda: patterns,
        prompt_store=FakePromptStore(),
        rng=random.Random(0),
        chat_id=CHAT_ID,
        bot_user_id=BOT_USER_ID,
        clock=lambda: DAY_NOW,
        sleep=instant_sleep,
        frag=DEMO_FRAG,
    )
    try:
        msg1 = _gate_message(
            tg_message_id=940, user_id=5, text="hello everyone", created_at=DAY_NOW
        )
        msg2 = _gate_message(
            tg_message_id=941, user_id=6, text="the weather really is lovely", created_at=DAY_NOW
        )

        task1: asyncio.Task[None] = asyncio.ensure_future(
            responder._handle_debounced(Trigger.AMBIENT, msg1, "User1")
        )
        await started.wait()

        task2: asyncio.Task[None] = asyncio.ensure_future(
            responder._handle_debounced(Trigger.AMBIENT, msg2, "User2")
        )
        # Let task2 advance to waiting on _respond_lock (task1 holds it).
        for _ in range(10):
            await asyncio.sleep(0)

        release.set()
        await task1
        await task2

        assert len(calls) == 1  # the second LLM call never happened
        assert len(bot.sent) == 1

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:recheck_ambient_cooldown") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()
