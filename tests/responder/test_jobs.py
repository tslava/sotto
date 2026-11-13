"""Responder tests: jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from sotto.db import Database
from sotto.timeutil import day_key, in_window
from tests.responder.helpers import (
    DAY_NOW,
    NIGHT_NOW,
    SITUATION_MORNING,
    FakeBot,
    FakeClock,
    FixedRandom,
    _config,
    _drive,
    _fail_handler,
    _insert_message,
    _make_llm,
    _make_responder,
    _ok_response,
    _payload,
    _tick_until,
    _varied_responses,
)


# Morning job: two night messages -> one call, MORNING situation, both marked answered.
async def test_morning_job_answers_all_night_messages_once(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Morning all, I was busy."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        night_id_1 = await db.enqueue_night(
            tg_message_id=200,
            user_id=5,
            display_name="Gene",
            text="gus, are you there?",
            created_at=NIGHT_NOW,
        )
        night_id_2 = await db.enqueue_night(
            tg_message_id=201,
            user_id=6,
            display_name="Nora",
            text="gus, are you asleep?",
            created_at=NIGHT_NOW + 60,
        )

        await _drive(clock, responder._run_morning_once())

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert SITUATION_MORNING in user_content

        unanswered = await db.night_unanswered()
        assert unanswered == []

        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT answered_at FROM night_queue WHERE id IN (?, ?)", (night_id_1, night_id_2)
        )
        rows = await cursor.fetchall()
        assert all(row["answered_at"] is not None for row in rows)

        tz = cfg.persona.timezone
        assert await db.get_state(day_key("mention_count", clock.now(), tz)) is None
    finally:
        await responder.shutdown()
        await llm.aclose()


# morning_job survives one failing iteration (night_unanswered raises once); the next
# iteration runs normally.
async def test_morning_job_survives_failing_iteration_and_continues(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Morning all, I was busy."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.enqueue_night(
            tg_message_id=910,
            user_id=5,
            display_name="Gene",
            text="gus, are you there?",
            created_at=NIGHT_NOW,
        )

        real_night_unanswered = db.night_unanswered
        state = {"raised": False}

        async def flaky_night_unanswered() -> list[object]:
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("night_unanswered is on fire")
            return await real_night_unanswered()  # type: ignore[return-value]

        monkeypatch.setattr(db, "night_unanswered", flaky_night_unanswered)

        task: asyncio.Task[None] = asyncio.ensure_future(responder.morning_job())
        try:
            with caplog.at_level(logging.ERROR):
                await _tick_until(clock, lambda: len(bot.sent) >= 1)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "morning_job" in errors[0].message

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Morning all, I was busy."
    finally:
        await responder.shutdown()
        await llm.aclose()


# morning_job loop: sleeps until a moment inside the 07:00-08:00 window, answers once, and
# wakes the next day (two days in a row).
async def test_morning_job_loop_wakes_within_window_two_consecutive_days(
    db: Database,
) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _varied_responses("Good morning.", "Hello all."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)  # 2026-01-10 15:00 London, outside the window, after it
    responder = _make_responder(db, cfg, llm, bot, clock)
    tz = cfg.persona.timezone
    window = cfg.behaviour.morning_reply_window

    await db.enqueue_night(
        tg_message_id=920, user_id=5, display_name="Gene", text="morning one", created_at=DAY_NOW
    )

    task: asyncio.Task[None] = asyncio.ensure_future(responder.morning_job())
    try:
        await _tick_until(clock, lambda: len(bot.sent) >= 1)
        first_send_at = clock.now()
        assert in_window(first_send_at, tz, window)

        await db.enqueue_night(
            tg_message_id=921,
            user_id=6,
            display_name="Nora",
            text="morning two",
            created_at=clock.now(),
        )

        await _tick_until(clock, lambda: len(bot.sent) >= 2)
        second_send_at = clock.now()
        assert in_window(second_send_at, tz, window)

        # The window is fixed and the moment inside it is random, so the gap between the two
        # runs must fit one day plus or minus the window width (1 hour).
        gap = second_send_at - first_send_at
        assert 23 * 3600 <= gap <= 25 * 3600
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await responder.shutdown()
        await llm.aclose()

    assert len(calls) == 2
    assert len(bot.sent) == 2


# spontaneous_job / _maybe_spontaneous: the "just because" conditions.
async def test_maybe_spontaneous_sends_once_and_bumps_counters(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Missing the shed."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)  # daytime, inside the default spontaneous window (10:00-21:00)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=FixedRandom(0.0))
    try:
        await _drive(clock, responder._maybe_spontaneous())

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Missing the shed."
        assert bot.sent[0][2] is None  # to the stream, no reply

        tz = cfg.persona.timezone
        assert await db.get_state(day_key("ambient_count", clock.now(), tz)) == "1"
        assert await db.get_state("last_ambient_at") is not None
        from sotto.timeutil import week_key

        assert await db.get_state(week_key("spontaneous_count", clock.now(), tz)) == "1"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_spontaneous_skipped_when_chat_recently_active(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=FixedRandom(0.0))
    try:
        await _insert_message(
            db,
            tg_message_id=930,
            user_id=5,
            display_name="Gene",
            text="hello",
            created_at=DAY_NOW - 60,  # a minute ago, less than min_quiet_hours (3h)
        )

        await responder._maybe_spontaneous()

        assert calls == []
        assert bot.sent == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_spontaneous_skipped_when_week_budget_exhausted(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=FixedRandom(0.0))
    tz = cfg.persona.timezone
    try:
        from sotto.timeutil import week_key

        await db.set_state(
            week_key("spontaneous_count", DAY_NOW, tz), str(cfg.behaviour.spontaneous.per_week)
        )

        await responder._maybe_spontaneous()

        assert calls == []
        assert bot.sent == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_spontaneous_skipped_outside_window(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(NIGHT_NOW)  # 3:00, outside spontaneous.window (10:00-21:00)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=FixedRandom(0.0))
    try:
        await responder._maybe_spontaneous()

        assert calls == []
        assert bot.sent == []
    finally:
        await responder.shutdown()
        await llm.aclose()
