"""Responder tests: callback."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx
import pytest

from sotto.db import Database, PlaceRow
from sotto.stickers import Sticker
from sotto.timeutil import day_key
from tests.demo import DEMO_FRAG
from tests.responder.helpers import (
    CALLBACK_AUTHOR_ID,
    CHAT_ID,
    DAY_NOW,
    NIGHT_NOW,
    FakeBot,
    FakeClock,
    FakeStickerChooser,
    _config,
    _drive,
    _fail_handler,
    _insert_bot_reply,
    _insert_message,
    _make_llm,
    _make_responder,
    _ok_response,
    _payload,
    _silent_response,
    _tick_until,
)


async def _callback_setup(
    db_: Database,
    *,
    thread_age_days: int = 3,
    author_last_msg_ago_sec: int = 3600,
    name: str = "Gene",
    text: str = "is waiting to hear back after the audition",
) -> int:
    """A thread of an author who wrote in the chat recently, plus that message itself."""
    await _insert_message(
        db_,
        tg_message_id=900,
        user_id=CALLBACK_AUTHOR_ID,
        display_name=name,
        text="hello everyone",
        created_at=DAY_NOW - author_last_msg_ago_sec,
    )
    return await db_.insert_people_thread(
        user_id=CALLBACK_AUTHOR_ID,
        display_name=name,
        text=text,
        created_at=DAY_NOW - thread_age_days * 86400,
    )


async def _assert_callback_untouched(db_: Database, thread_id: int, calls: list[object]) -> None:
    assert calls == []
    assert [t.id for t in await db_.open_people_threads()] == [thread_id]
    assert await db_.get_state("callback_last_at") is None


async def test_maybe_callback_sends_question_and_closes_thread(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Gene, how did it turn out?"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        thread_id = await _callback_setup(db)

        await _drive(clock, responder._maybe_callback())

        assert len(calls) == 1
        assert bot.sent == [(CHAT_ID, "Gene, how did it turn out?", None)]  # not a reply
        user_message = _payload(calls[0])["messages"][1]["content"]  # type: ignore[index]
        assert (
            DEMO_FRAG.text("situations.callback")
            .replace("{name}", "Gene")
            .replace("{text}", "is waiting to hear back after the audition")
            in user_message
        )
        thread = (await db.people_threads_all())[0]
        assert thread.id == thread_id
        assert thread.closed is True
        # the time moved on for "typing", the closing is already after the send
        assert thread.asked_at is not None
        assert thread.asked_at >= DAY_NOW
        assert await db.get_state("callback_last_at") == str(thread.asked_at)

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:callback") == 1
        replies = await db.last_bot_replies(1)
        assert replies[0].trigger == "callback"
        assert replies[0].text == "Gene, how did it turn out?"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_closes_thread_even_if_cancelled_mid_send(db: Database) -> None:
    """The thread is closed before the attempt: a cancellation in the middle of the send
    (SIGTERM) does not lead to the same story being asked about a second time after a restart."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, FakeBot(), clock)

    async def cancelled(**_kwargs: object) -> None:
        raise asyncio.CancelledError

    responder._respond = cancelled  # type: ignore[method-assign]  # swapped for a cancellation
    try:
        thread_id = await _callback_setup(db)

        with pytest.raises(asyncio.CancelledError):
            await responder._maybe_callback()

        assert [t.id for t in await db.open_people_threads()] == []
        assert [t.id for t in await db.people_threads_all()] == [thread_id]
        assert await db.get_state("callback_last_at") == str(DAY_NOW)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_does_not_touch_counters_or_open_hot_window(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.hot_window.open_on_any_reply = True  # an ordinary trigger would open the window
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Gene, how did it turn out?"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _callback_setup(db)

        await _drive(clock, responder._maybe_callback())

        assert len(bot.sent) == 1
        tz = cfg.persona.timezone
        for key in ("mention_count", "ambient_count", "sticker_count"):
            assert await db.get_state(day_key(key, DAY_NOW, tz)) is None
        assert await db.get_state("last_ambient_at") is None
        assert await db.get_state("last_mention_reply_at") is None
        assert await db.get_state(f"last_mention_reply_at:{CALLBACK_AUTHOR_ID}") is None
        assert await db.get_state("hot_until") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_system_prompt_has_no_places_and_no_sticker(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Gene, how did it turn out?"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    chooser = FakeStickerChooser(Sticker(id=1, file_id="F", text="ok", enabled=True))
    responder = _make_responder(db, cfg, llm, bot, clock, sticker_chooser=chooser)
    try:
        await _callback_setup(db)
        await db.upsert_place(
            PlaceRow(
                place_id="p1",
                name="Pub X",
                district="Kettle Hill",
                category="pub",
                rating=4.5,
                reviews=100,
                price_level=1,
                quiet=1,
                fact="quiet",
                operational=1,
                refreshed_at=DAY_NOW,
            )
        )

        await _drive(clock, responder._maybe_callback())

        user_message = _payload(calls[0])["messages"][1]["content"]  # type: ignore[index]
        assert "Pub X" not in user_message
        assert bot.sent_stickers == []
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_silence_closes_thread_and_sets_last_at(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _silent_response())
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _callback_setup(db)

        await _drive(clock, responder._maybe_callback())

        assert len(calls) == 1
        assert bot.sent == []
        assert await db.open_people_threads() == []
        assert await db.get_state("callback_last_at") == str(DAY_NOW)
        assert dict(await db.filter_log_summary(0)).get("llm:silent") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_filter_cut_closes_thread(db: Database) -> None:
    cfg = _config()
    # style:exclaim cuts even in shadow (enforce_stages): no reply, thread closed
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Gene! So how did it go!"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _callback_setup(db)

        await _drive(clock, responder._maybe_callback())

        assert bot.sent == []
        assert await db.open_people_threads() == []
        assert await db.get_state("callback_last_at") == str(DAY_NOW)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_llm_error_closes_thread(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: httpx.Response(500, json={}))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _callback_setup(db)

        await _drive(clock, responder._maybe_callback())

        assert len(calls) == 1
        assert bot.sent == []
        assert await db.open_people_threads() == []
        assert await db.get_state("callback_last_at") == str(DAY_NOW)
        assert dict(await db.filter_log_summary(0)).get("llm:http") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_picks_oldest_eligible_and_skips_muted_and_inactive(
    db: Database,
) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Julie, how did it turn out?"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        # the oldest is by a muted author; next comes an author who has not written for a long time
        await _insert_message(
            db, tg_message_id=1, user_id=11, display_name="Moss", text="I am here",
            created_at=DAY_NOW - 3600,
        )  # fmt: skip
        await db.add_mute(11, "Moss", 999, DAY_NOW - 5000)
        muted = await db.insert_people_thread(
            user_id=11, display_name="Moss", text="repairs", created_at=DAY_NOW - 9 * 86400
        )
        await _insert_message(
            db, tg_message_id=2, user_id=12, display_name="Quincy", text="wrote long ago",
            created_at=DAY_NOW - 8 * 86400,
        )  # fmt: skip
        inactive = await db.insert_people_thread(
            user_id=12, display_name="Quincy", text="a trip", created_at=DAY_NOW - 8 * 86400
        )
        await _insert_message(
            db, tg_message_id=3, user_id=13, display_name="Julie", text="hello",
            created_at=DAY_NOW - 7200,
        )  # fmt: skip
        eligible = await db.insert_people_thread(
            user_id=13,
            display_name="Julie",
            text="is choosing a van",
            created_at=DAY_NOW - 5 * 86400,
        )
        await db.insert_people_thread(
            user_id=13, display_name="Julie", text="a purchase", created_at=DAY_NOW - 4 * 86400
        )

        await _drive(clock, responder._maybe_callback())

        assert len(calls) == 1
        user_message = _payload(calls[0])["messages"][1]["content"]  # type: ignore[index]
        assert (
            DEMO_FRAG.text("situations.callback")
            .replace("{name}", "Julie")
            .replace("{text}", "is choosing a van")
            in user_message
        )
        closed = {t.id for t in await db.people_threads_all() if t.closed}
        assert closed == {eligible}
        open_ids = {t.id for t in await db.open_people_threads()}
        assert muted in open_ids
        assert inactive in open_ids
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_deletes_expired_thread_without_asking(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _callback_setup(db, thread_age_days=15)

        await responder._maybe_callback()

        assert calls == []
        assert bot.sent == []
        # Deleted, not closed: /people must not show "asked" without a question.
        assert await db.people_threads_all() == []
        # the expired story was not asked about, so the interval between questions is not set
        assert await db.get_state("callback_last_at") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_no_open_threads_does_nothing(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        thread_id = await _callback_setup(db)
        await db.close_people_thread(thread_id, DAY_NOW - 100)

        await responder._maybe_callback()

        assert calls == []
        assert await db.get_state("callback_last_at") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_disabled(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.callback.enabled = False
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_blocked_by_panic_and_stop(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)

        await db.set_state("panic", "1")
        await responder._maybe_callback()
        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]

        await db.delete_state("panic")
        await db.set_state("stop_until", str(DAY_NOW + 1000))
        await responder._maybe_callback()
        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_blocked_by_topic_cooldown(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)
        await db.set_state("topic_cooldown_until", str(DAY_NOW + 1000))

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_blocked_at_night(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(NIGHT_NOW))
    try:
        thread_id = await _callback_setup(db)

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_blocked_outside_window(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    # 21:00 local: outside callback.window (11:00-20:00) but also outside the night silence
    clock = FakeClock(DAY_NOW + 6 * 3600)
    responder = _make_responder(db, cfg, llm, FakeBot(), clock)
    try:
        thread_id = await _callback_setup(db)

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_respects_min_days(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)
        await db.set_state("callback_last_at", str(DAY_NOW - 2 * 86400))

        await responder._maybe_callback()

        assert calls == []
        assert [t.id for t in await db.open_people_threads()] == [thread_id]
        assert await db.get_state("callback_last_at") == str(DAY_NOW - 2 * 86400)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_asks_again_after_min_days(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Gene, how did it turn out?"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _callback_setup(db)
        await db.set_state("callback_last_at", str(DAY_NOW - 3 * 86400))

        await _drive(clock, responder._maybe_callback())

        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_waits_for_quiet_in_chat(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)
        await _insert_message(
            db, tg_message_id=901, user_id=6, display_name="Julie", text="typing now",
            created_at=DAY_NOW - 60,
        )  # fmt: skip

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_blocked_by_presence_cap(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)
        # one human message -> allowance 2; two bot replies today use up the ceiling
        await _insert_bot_reply(db, created_at=DAY_NOW - 7200, tg_message_id=1)
        await _insert_bot_reply(db, created_at=DAY_NOW - 3 * 3600, tg_message_id=2)

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_callback_blocked_by_min_gap(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        thread_id = await _callback_setup(db)
        await _insert_bot_reply(db, created_at=DAY_NOW - 60)

        await responder._maybe_callback()

        await _assert_callback_untouched(db, thread_id, calls)  # type: ignore[arg-type]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_callback_job_survives_failing_iteration(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, FakeBot(), clock)
    try:
        state = {"calls": 0}

        async def flaky() -> None:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("callback is on fire")

        monkeypatch.setattr(responder, "_maybe_callback", flaky)

        task: asyncio.Task[None] = asyncio.ensure_future(responder.callback_job())
        try:
            with caplog.at_level(logging.ERROR):
                await _tick_until(clock, lambda: state["calls"] >= 2)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "callback_job" in errors[0].message
    finally:
        await responder.shutdown()
        await llm.aclose()
