"""Responder tests: checkin."""

from __future__ import annotations

from sotto.db import Database
from tests.responder.helpers import (
    DAY_NOW,
    NIGHT_NOW,
    FakeBot,
    FakeClock,
    FakeFollowup,
    MinRandom,
    _checkin_response,
    _config,
    _drive,
    _fail_handler,
    _insert_bot_reply,
    _insert_message,
    _make_llm,
    _make_responder,
    _silent_response,
)


async def test_maybe_checkin_schedules_due_from_closed_hot_window(db: Database) -> None:
    """due is computed from the last known (already closed) hot_until, not from now."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        hot_until = DAY_NOW - 10  # the window has already closed
        await db.set_state("hot_until", str(hot_until))
        await _insert_bot_reply(db, created_at=hot_until)

        await responder._maybe_checkin()

        assert calls == []
        after_min_lo = cfg.behaviour.checkin.after_min[0]
        assert await db.get_state("checkin_due") == str(hot_until + after_min_lo * 60)
        assert await db.get_state("checkin_due_hot_until") == str(hot_until)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_skipped_while_hot_window_still_open(db: Database) -> None:
    """The phone is still in hand: followup copes, no separate check runs, and
    checkin_due is not computed at all."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("hot_until", str(DAY_NOW + 900))
        await _insert_bot_reply(db, created_at=DAY_NOW - 100)

        await responder._maybe_checkin()

        assert calls == []
        assert await db.get_state("checkin_due") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_no_new_messages_reschedules_without_llm_call(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        last_reply_at = DAY_NOW - 20000
        await _insert_bot_reply(db, created_at=last_reply_at)
        # hot_window was never set -> due_ref = last_reply_at (see the _maybe_checkin
        # docstring); the due time has already come.
        await db.set_state("checkin_due", str(DAY_NOW - 10))
        await db.set_state("checkin_due_hot_until", str(last_reply_at))

        await responder._maybe_checkin()

        assert calls == []
        assert bot.sent == []
        after_min_lo = cfg.behaviour.checkin.after_min[0]
        assert await db.get_state("checkin_last_at") == str(DAY_NOW)
        assert await db.get_state("checkin_due") == str(DAY_NOW + after_min_lo * 60)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_sends_reply_to_selected_message(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(
        cfg, db, lambda _req: _checkin_response("Fair enough, the shed is calling.", 2)
    )
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        last_reply_at = DAY_NOW - 20000
        await _insert_bot_reply(db, created_at=last_reply_at)
        await _insert_message(
            db,
            tg_message_id=701,
            user_id=5,
            display_name="Gene",
            text="how are you, Gus?",
            created_at=last_reply_at + 10,
        )
        await _insert_message(
            db,
            tg_message_id=702,
            user_id=6,
            display_name="Julie",
            text="gus, are you even alive?",
            created_at=last_reply_at + 20,
        )
        await db.set_state("checkin_due", str(DAY_NOW - 10))
        await db.set_state("checkin_due_hot_until", str(last_reply_at))

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Fair enough, the shed is calling."
        assert bot.sent[0][2] == 702  # reply_to=2 -> the second message (Julie, tg 702)

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:checkin") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_speak_false_logs_silent_and_reschedules(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _silent_response())
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        last_reply_at = DAY_NOW - 20000
        await _insert_bot_reply(db, created_at=last_reply_at)
        await _insert_message(
            db,
            tg_message_id=800,
            user_id=5,
            display_name="Gene",
            text="hello",
            created_at=last_reply_at + 10,
        )
        await db.set_state("checkin_due", str(DAY_NOW - 10))
        await db.set_state("checkin_due_hot_until", str(last_reply_at))

        await responder._maybe_checkin()

        assert len(calls) == 1
        assert bot.sent == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:silent") == 1
        after_min_lo = cfg.behaviour.checkin.after_min[0]
        assert await db.get_state("checkin_last_at") == str(DAY_NOW)
        assert await db.get_state("checkin_due") == str(DAY_NOW + after_min_lo * 60)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_dead_topic_deletes_due_and_skips(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        topic_max_hours = cfg.behaviour.checkin.topic_max_hours
        last_reply_at = DAY_NOW - (topic_max_hours + 1) * 3600
        await _insert_bot_reply(db, created_at=last_reply_at)
        await db.set_state("checkin_due", str(DAY_NOW - 10))
        await db.set_state("checkin_due_hot_until", str(last_reply_at))

        await responder._maybe_checkin()

        assert calls == []
        assert await db.get_state("checkin_due") is None
        assert await db.get_state("checkin_due_hot_until") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_blocked_by_panic(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("panic", "1")
        await _insert_bot_reply(db, created_at=DAY_NOW - 20000)

        await responder._maybe_checkin()

        assert calls == []
        assert await db.get_state("checkin_due") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_blocked_by_stop(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("stop_until", str(DAY_NOW + 1000))
        await _insert_bot_reply(db, created_at=DAY_NOW - 20000)

        await responder._maybe_checkin()

        assert calls == []
        assert await db.get_state("checkin_due") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_blocked_during_quiet_window(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(NIGHT_NOW)  # 3:00 is inside the default quiet_window (01:00-07:00)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_bot_reply(db, created_at=NIGHT_NOW - 20000)

        await responder._maybe_checkin()

        assert calls == []
        assert await db.get_state("checkin_due") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def _prepare_checkin(db_: Database, *, last_reply_at: int, human_created_at: int) -> None:
    """State in which _maybe_checkin would get as far as calling the model."""
    await _insert_bot_reply(db_, created_at=last_reply_at)
    await _insert_message(
        db_,
        tg_message_id=701,
        user_id=5,
        display_name="Gene",
        text="how are you, Gus?",
        created_at=human_created_at,
    )
    await _insert_message(
        db_,
        tg_message_id=702,
        user_id=6,
        display_name="Julie",
        text="gus, are you even alive?",
        created_at=human_created_at + 10,
    )
    await db_.set_state("checkin_due", str(last_reply_at + 10))
    await db_.set_state("checkin_due_hot_until", str(last_reply_at))


async def test_maybe_checkin_waits_for_quiet_chat(db: Database) -> None:
    """Someone just wrote in the chat: the check-in is postponed without moving due, and the
    next poll tries again."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        last_reply_at = DAY_NOW - 20000
        due = last_reply_at + 10
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=DAY_NOW - 60)

        await responder._maybe_checkin()

        assert calls == []
        assert bot.sent == []
        assert await db.get_state("checkin_due") == str(due)  # due not moved
        assert await db.get_state("checkin_last_at") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_quiet_min_zero_does_not_wait(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.checkin.quiet_min = 0
    llm, calls = _make_llm(cfg, db, lambda _req: _checkin_response("Happens.", 2))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        await _prepare_checkin(db, last_reply_at=DAY_NOW - 20000, human_created_at=DAY_NOW - 60)

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_without_reply_to_stays_silent(db: Database) -> None:
    """An answer "to everyone", without a message number, is no longer sent."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _checkin_response("Hello, everyone.", None))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        last_reply_at = DAY_NOW - 20000
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=last_reply_at + 10)

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert bot.sent == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:checkin_no_target") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_followup_rejects_selected_message(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _checkin_response("Happens.", 2))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    followup = FakeFollowup(addressed=False)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom(), followup=followup)
    try:
        last_reply_at = DAY_NOW - 20000
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=last_reply_at + 10)

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert bot.sent == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:checkin_not_addressed") == 1

        # The selected row (number 2, Julie) is what was checked; the rest went into the
        # check context.
        assert len(followup.calls) == 1
        assert followup.calls[0]["text"] == "gus, are you even alive?"
        assert followup.calls[0]["display_name"] == "Julie"
        context_rows = followup.calls[0]["context_rows"]
        assert isinstance(context_rows, list)
        assert [row.tg_message_id for row in context_rows] == [701]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_followup_confirms_selected_message(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _checkin_response("Alive, why wouldn't I be.", 2))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    followup = FakeFollowup(addressed=True)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom(), followup=followup)
    try:
        last_reply_at = DAY_NOW - 20000
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=last_reply_at + 10)

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][2] == 702
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:checkin") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


# The cheap prefilter before checkin.


async def test_maybe_checkin_prefilter_empty_skips_main_model(db: Database) -> None:
    """The prefilter saw nothing for Gus: the main model is not called at all,
    checkin_last_at/checkin_due are moved as usual (as after llm:silent), and the filter_log
    gets send:checkin_prefilter_no."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    followup = FakeFollowup(addressed=True, batch_addressed=[])
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom(), followup=followup)
    try:
        last_reply_at = DAY_NOW - 20000
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=last_reply_at + 10)

        await responder._maybe_checkin()

        assert calls == []
        assert bot.sent == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:checkin_prefilter_no") == 1
        assert "send:checkin_prefilter_yes" not in summary
        after_min_lo = cfg.behaviour.checkin.after_min[0]
        assert await db.get_state("checkin_last_at") == str(DAY_NOW)
        assert await db.get_state("checkin_due") == str(DAY_NOW + after_min_lo * 60)
        assert len(followup.batch_calls) == 1
        rows_seen = followup.batch_calls[0]["rows"]
        assert isinstance(rows_seen, list)
        assert len(rows_seen) == 2
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_prefilter_nonempty_calls_main_model_with_full_rows(
    db: Database,
) -> None:
    """The prefilter marked only one message: the main model still gets ALL the checkin_rows
    (context matters), and the filter_log gets send:checkin_prefilter_yes."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _checkin_response("Happens.", 2))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    followup = FakeFollowup(addressed=True, batch_addressed=[2])
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom(), followup=followup)
    try:
        last_reply_at = DAY_NOW - 20000
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=last_reply_at + 10)

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert len(bot.sent) == 1
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:checkin_prefilter_yes") == 1
        assert summary.get("send:checkin") == 1
        assert len(followup.batch_calls) == 1
        rows_seen = followup.batch_calls[0]["rows"]
        assert isinstance(rows_seen, list)
        assert len(rows_seen) == 2
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_maybe_checkin_prefilter_disabled_skips_cheap_call(db: Database) -> None:
    """behaviour.checkin.prefilter=false: the cheap batch check is not called at all and the
    main model is called directly."""
    cfg = _config()
    cfg.behaviour.checkin.prefilter = False
    llm, calls = _make_llm(cfg, db, lambda _req: _checkin_response("Happens.", 2))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    followup = FakeFollowup(addressed=True)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom(), followup=followup)
    try:
        last_reply_at = DAY_NOW - 20000
        await _prepare_checkin(db, last_reply_at=last_reply_at, human_created_at=last_reply_at + 10)

        await _drive(clock, responder._maybe_checkin())

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert followup.batch_calls == []
        summary = dict(await db.filter_log_summary(0))
        assert "send:checkin_prefilter_yes" not in summary
        assert "send:checkin_prefilter_no" not in summary
    finally:
        await responder.shutdown()
        await llm.aclose()
