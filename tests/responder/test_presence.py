"""Responder tests: presence."""

from __future__ import annotations

import pytest

from sotto.db import Database, PendingRow
from sotto.gate_types import Trigger
from tests.responder.helpers import (
    DAY_NOW,
    FakeBot,
    FakeClock,
    MinRandom,
    _config,
    _drive,
    _fail_handler,
    _insert_bot_reply,
    _insert_message,
    _make_llm,
    _make_responder,
    _ok_response,
    _pending_row,
)

# Presence ceiling and the pause between replies.


async def _fill_presence(db_: Database, *, humans: int, bot_replies: int, now: int) -> None:
    """A day in which people wrote ``humans`` messages and the bot replied ``bot_replies``
    times. All within the local day of ``now``."""
    for index in range(humans):
        await _insert_message(
            db_,
            tg_message_id=6000 + index,
            user_id=5,
            display_name="Gene",
            text="chatter",
            created_at=now - 3600 + index,
        )
    for index in range(bot_replies):
        await db_.insert_bot_reply(
            tg_message_id=7000 + index,
            reply_to_tg_message_id=None,
            trigger="ambient",
            trigger_tg_message_id=None,
            text="a reply from Gus",
            prompt_version=1,
            few_shot_version=1,
            delay_sec=0,
            created_at=now - 3600 + index,
        )


async def test_presence_cap_blocks_ambient_before_model(db: Database) -> None:
    """20 human messages at max_share 0.15 and free_replies 2 -> allowance 5; the bot's
    fifth reply has already used the ceiling, so the model is not called at all."""
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0  # tested separately, would get in the way here
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _fill_presence(db, humans=20, bot_replies=5, now=DAY_NOW)

        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=6000,
                user_id=5,
                situation="",
                delay_sec=0,
            ),
        )

        assert calls == []
        assert bot.sent == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:presence_cap") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_presence_under_cap_lets_ambient_through(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _fill_presence(db, humans=20, bot_replies=4, now=DAY_NOW)

        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=6000,
                user_id=5,
                situation="",
                delay_sec=0,
            ),
        )

        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_presence_cap_does_not_block_life_and_say(db: Database) -> None:
    """/life and /say are the owner's button: neither the ceiling nor the pause holds them."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Sold it, yep."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _fill_presence(db, humans=20, bot_replies=9, now=DAY_NOW)

        event_id = await db.insert_life_event(text="sold the handcart", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None
        life_task = await _drive(clock, responder.announce_life(event))
        assert life_task.result().sent is True

        say_task = await _drive(clock, responder.say("Well, hello."))
        assert say_task.result().sent is True
        assert len(bot.sent) == 2
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_recheck_presence_cuts_pending_name_address(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _fill_presence(db, humans=20, bot_replies=5, now=DAY_NOW)
        pending_id = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=50,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await responder._fire_pending(row)

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:recheck_presence") == 1
        assert bot.sent == []
        assert calls == []
        assert await db.load_pending() == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_recheck_presence_lets_reply_to_bot_through(db: Database) -> None:
    """A reply to the bot's message passes under the ceiling, also at send time."""
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Oh, come on."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _fill_presence(db, humans=20, bot_replies=5, now=DAY_NOW)
        pending_id = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="reply",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=50,
            user_id=5,
            trigger="reply",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await _drive(clock, responder._fire_pending(row))

        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_recheck_presence_lets_mention_through(db: Database) -> None:
    """A2: an @mention passes under the ceiling, also at send time."""
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Oh, come on."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _fill_presence(db, humans=20, bot_replies=5, now=DAY_NOW)
        pid = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        await _drive(clock, responder._fire_pending(_pending_row(pid, "mention", DAY_NOW)))
        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


@pytest.mark.parametrize("trigger", ["reply", "mention", "name"])
async def test_recheck_ignores_topic_cooldown_for_direct_address(
    db: Database, trigger: str
) -> None:
    """A3: the stop-topic cooldown does not cut a direct address at send time."""
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Oh, come on."))
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, FakeBot(), clock)
    try:
        await db.set_state("topic_cooldown_until", str(DAY_NOW + 600))
        row = _pending_row(1, trigger, DAY_NOW)
        assert await responder._recheck(row, DAY_NOW, cfg) is None
        ambient = _pending_row(2, "ambient", DAY_NOW)
        assert await responder._recheck(ambient, DAY_NOW, cfg) == "send:recheck_topic"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_recheck_keeps_topic_cooldown_for_followup(db: Database) -> None:
    """A followup is "probably for me", not a direct address: the topic cooldown cuts it."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Oh, come on."))
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        await db.set_state("topic_cooldown_until", str(DAY_NOW + 600))
        row = _pending_row(1, "followup", DAY_NOW)
        assert await responder._recheck(row, DAY_NOW, cfg) == "send:recheck_topic"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_min_gap_delays_pending_instead_of_dropping_it(db: Database) -> None:
    """A direct address is not dropped by the pause: the pending moves to last + min_gap."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        last_reply_at = DAY_NOW - 100
        await _insert_bot_reply(db, created_at=last_reply_at, tg_message_id=800)
        pending_id = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=50,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await responder._fire_pending_inner(row)

        rows = await db.load_pending()
        assert len(rows) == 1  # not marked done: the reply is still ahead
        assert rows[0].due_at == last_reply_at + cfg.behaviour.min_gap_sec + 5  # MinRandom -> 5
        assert calls == []
        assert bot.sent == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_min_gap_elapsed_lets_pending_fire(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Yep."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_bot_reply(
            db, created_at=DAY_NOW - cfg.behaviour.min_gap_sec - 1, tg_message_id=800
        )
        pending_id = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=50,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await _drive(clock, responder._fire_pending(row))

        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_min_gap_cuts_ambient_with_silence(db: Database) -> None:
    """An unaddressed trigger does not wait out the pause: silence, the model is not called."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_bot_reply(db, created_at=DAY_NOW - 100, tg_message_id=800)

        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=900,
                user_id=None,
                situation="",
                delay_sec=0,
            ),
        )

        assert calls == []
        assert bot.sent == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:min_gap") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_min_gap_zero_disables_the_pause(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_bot_reply(db, created_at=DAY_NOW - 1, tg_message_id=800)

        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=900,
                user_id=None,
                situation="",
                delay_sec=0,
            ),
        )

        assert len(calls) == 1
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()
