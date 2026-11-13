"""Responder tests: pending."""

from __future__ import annotations

import asyncio
import logging
import random

import httpx
import pytest

from sotto.db import Database, PendingRow
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
    NIGHT_NOW,
    SITUATION_AMBIENT,
    SITUATION_LATE,
    FakeBot,
    FakeClock,
    FakePromptStore,
    MinRandom,
    _config,
    _drive,
    _fail_handler,
    _gate_message,
    _insert_message,
    _make_llm,
    _make_responder,
    _ok_response,
    _payload,
    _pending_row,
    _silent_response,
    _varied_responses,
)


def _addressed(name: str, text: str) -> str:
    """The single-address situation as the demo pack renders it."""
    template = DEMO_FRAG.text("situations.addressed_single")
    return template.replace("{name}", name).replace("{text}", text)


def _multi_line(name: str, text: str) -> str:
    quote_open, quote_close = (
        DEMO_FRAG.text("render.quote_open"),
        DEMO_FRAG.text("render.quote_close"),
    )
    return f"- {name}: {quote_open}{text}{quote_close}"


# A mention creates a pending reply, sent no earlier than 30 s later, counters set


async def test_mention_creates_pending_and_fires_after_delay(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Hello to you too."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=10,
            user_id=5,
            display_name="Gene",
            text="@gusbot how are you?",
            created_at=DAY_NOW,
        )
        msg = _gate_message(
            tg_message_id=10, user_id=5, text="@gusbot how are you?", created_at=DAY_NOW
        )
        await responder.on_gate_pass(msg, Trigger.MENTION, "Gene")

        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        # The debounce is done but the pending has not fired yet: nothing is sent.
        pending_rows = await db.load_pending()
        assert len(pending_rows) == 1
        assert pending_rows[0].done_at is None
        assert bot.sent == []

        pending_id = pending_rows[0].id
        task = responder._pending_tasks[pending_id]
        start = clock.value
        await clock.run_until(task)

        assert clock.value - start >= 30  # fast bucket: 30 s minimum
        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Hello to you too."
        assert (await db.recent_bot_replies(5)) == ["Hello to you too."]

        tz = cfg.persona.timezone
        mention_key = day_key("mention_count", clock.now(), tz)
        assert await db.get_state(mention_key) == "1"
        assert await db.get_state("last_mention_reply_at") is not None
        assert await db.get_state("last_mention_reply_at:5") is not None
    finally:
        await responder.shutdown()
        await llm.aclose()


# A second mention while waiting collapses into the same pending


async def test_second_mention_collapses_pending_without_duplicate(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        msg1 = _gate_message(
            tg_message_id=20, user_id=5, text="gus, you there?", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg1, "Gene")

        pending_after_1 = await db.load_pending()
        assert len(pending_after_1) == 1
        pending_id = pending_after_1[0].id
        task1 = responder._pending_tasks[pending_id]
        assert not task1.done()  # nothing was advanced, the timer has not fired yet

        msg2 = _gate_message(
            tg_message_id=21, user_id=6, text="marlow, anyone home?", created_at=DAY_NOW + 2
        )
        await responder._handle_debounced(Trigger.NAME, msg2, "Nora")

        pending_after_2 = await db.load_pending()
        assert len(pending_after_2) == 1  # not a second task, the same id
        assert pending_after_2[0].id == pending_id

        fast_hi = cfg.behaviour.reply_delay_buckets[0].range_sec[1]
        assert pending_after_2[0].due_at - (DAY_NOW + 2) <= fast_hi
        assert responder._pending_tasks[pending_id] is not task1  # the old timer was replaced

        assert calls == []  # generation has not happened yet
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_mention_during_in_flight_pending_gets_own_pending(db: Database) -> None:
    """While a pending is being generated, a new address does not collapse into it: otherwise
    its timer would fire a second time after the reply (a double reply)."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        msg1 = _gate_message(
            tg_message_id=20, user_id=5, text="gus, you there?", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg1, "Gene")
        first_id = (await db.load_pending())[0].id
        first_items = list(responder._pending_info.get(first_id, []))
        responder._in_flight.add(first_id)  # as if _fire_pending were already calling the model

        msg2 = _gate_message(
            tg_message_id=21, user_id=6, text="marlow, anyone home?", created_at=DAY_NOW + 2
        )
        await responder._handle_debounced(Trigger.NAME, msg2, "Nora")

        pending = await db.load_pending()
        assert len(pending) == 2
        second = next(p for p in pending if p.id != first_id)
        assert second.trigger_tg_message_id == 21
        assert responder._pending_info.get(first_id, []) == first_items
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


# The mention cooldown never drops the reply, it shifts due_at to no earlier than
# earliest = max(last_mention_reply_at + chat_cooldown,
# last_mention_reply_at:<user> + user_cooldown).


async def test_mention_without_active_cooldown_uses_normal_bucket_delay(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        # Neither last_mention_reply_at nor last_mention_reply_at:<user> exists: earliest is
        # small (0 + cooldown), due is not shifted, the delay is the usual one (first bucket).
        msg = _gate_message(
            tg_message_id=10, user_id=5, text="gus, how are you?", created_at=DAY_NOW
        )
        await responder.on_gate_pass(msg, Trigger.NAME, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        rows = await db.load_pending()
        assert len(rows) == 1
        fast_lo = cfg.behaviour.reply_delay_buckets[0].range_sec[0]
        assert rows[0].due_at - rows[0].created_at == fast_lo
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_second_mention_after_reply_delayed_until_chat_cooldown_elapses(
    db: Database,
) -> None:
    """A second address 20 s after a sent reply, with mention_chat_cooldown_sec=90 active: the
    new pending gets due >= last_reply_at + 90 (not the usual short delay), and when it fires the
    reply is still sent. The cooldown no longer drops the address, it only postpones it."""
    cfg = _config()
    assert cfg.behaviour.mention_chat_cooldown_sec == 90
    # The pause between replies would postpone this pending once more and is covered by
    # separate tests; this one is about the mention cooldown, so the pause is switched off.
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, _varied_responses("Answer one.", "Answer two."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        msg1 = _gate_message(tg_message_id=10, user_id=5, text="gus, hello", created_at=DAY_NOW)
        await responder.on_gate_pass(msg1, Trigger.NAME, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        pending_rows = await db.load_pending()
        assert len(pending_rows) == 1
        pending_id = pending_rows[0].id
        await clock.run_until(responder._pending_tasks[pending_id])

        assert len(bot.sent) == 1
        last_reply_raw = await db.get_state("last_mention_reply_at")
        assert last_reply_raw is not None
        last_reply_at = int(last_reply_raw)

        # The second address is from another person, 20 s after the sent reply.
        clock.value = float(last_reply_at + 20)
        msg2 = _gate_message(
            tg_message_id=11, user_id=6, text="marlow, anyone home?", created_at=last_reply_at + 20
        )
        await responder.on_gate_pass(msg2, Trigger.NAME, "Nora")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        pending_rows_2 = await db.load_pending()
        assert len(pending_rows_2) == 1
        assert pending_rows_2[0].due_at >= last_reply_at + cfg.behaviour.mention_chat_cooldown_sec

        task2 = responder._pending_tasks[pending_rows_2[0].id]
        await clock.run_until(task2)
        assert len(bot.sent) == 2  # the cooldown postponed it but did not drop it
        assert len(calls) == 2
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_collapse_does_not_move_due_earlier_than_cooldown_floor(db: Database) -> None:
    """Collapsing (update_pending_due) also obeys earliest: even the fast bucket cannot pull
    due_at before the end of the active chat cooldown."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        # Simulate a recent reply to an address: the chat cooldown is still active.
        await db.set_state("last_mention_reply_at", str(DAY_NOW - 10))
        earliest = DAY_NOW - 10 + cfg.behaviour.mention_chat_cooldown_sec

        msg1 = _gate_message(
            tg_message_id=20, user_id=5, text="gus, you there?", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg1, "Gene")

        pending_after_1 = await db.load_pending()
        assert len(pending_after_1) == 1
        pending_id = pending_after_1[0].id
        assert pending_after_1[0].due_at >= earliest

        msg2 = _gate_message(
            tg_message_id=21, user_id=6, text="marlow, anyone home?", created_at=DAY_NOW + 2
        )
        await responder._handle_debounced(Trigger.NAME, msg2, "Nora")

        pending_after_2 = await db.load_pending()
        assert len(pending_after_2) == 1
        assert pending_after_2[0].id == pending_id  # collapsed, not a second task
        assert pending_after_2[0].due_at >= earliest
        assert calls == []  # generation has not happened yet
    finally:
        await responder.shutdown()
        await llm.aclose()


# An urgent message gets the fast bucket


async def test_urgent_message_gets_fast_bucket(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        msg = _gate_message(
            tg_message_id=30,
            user_id=5,
            text="@gusbot where are we going today?",
            created_at=DAY_NOW,
        )
        await responder.on_gate_pass(msg, Trigger.MENTION, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        rows = await db.load_pending()
        assert len(rows) == 1
        assert rows[0].due_at - rows[0].created_at <= cfg.behaviour.urgent_max_delay_sec
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


# A pending whose due falls into the night goes to night_queue, pending done, no LLM call


async def test_pending_due_in_quiet_window_goes_to_night_queue(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(NIGHT_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        pending_id = await db.insert_pending(
            trigger_tg_message_id=40,
            user_id=5,
            trigger="mention",
            due_at=NIGHT_NOW,
            created_at=NIGHT_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=40,
            user_id=5,
            trigger="mention",
            due_at=NIGHT_NOW,
            created_at=NIGHT_NOW - 60,
            done_at=None,
        )

        await responder._fire_pending(row)

        night_rows = await db.night_unanswered()
        assert len(night_rows) == 1
        assert night_rows[0].tg_message_id == 40
        assert (
            night_rows[0].display_name == DEMO_FRAG.prefix("placeholders.participant").strip()
        )  # not in memory -> placeholder

        pending_after = await db.load_pending()
        assert pending_after == []

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:night") == 1
        assert calls == []
        assert bot.sent == []
    finally:
        await responder.shutdown()
        await llm.aclose()


# /stop during the wait -> recheck_stop, nothing sent


async def test_recheck_catches_stop_during_wait(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("stop_until", str(DAY_NOW + 1000))
        pending_id = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=50,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await responder._fire_pending(row)

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:recheck_stop") == 1
        assert bot.sent == []
        assert calls == []

        pending_after = await db.load_pending()
        assert pending_after == []
    finally:
        await responder.shutdown()
        await llm.aclose()


# restore_pending: a recently overdue one fires at once with SITUATION_LATE; an old one is a restart


async def test_restore_pending_late_flag_and_restart(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Lost in my own thoughts."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        # Queued an hour ago, due almost now: overdue is small (< threshold), but the full
        # delay since created_at is large -> SITUATION_LATE when it fires.
        recent_id = await db.insert_pending(
            trigger_tg_message_id=90,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW - 50,
            created_at=DAY_NOW - 3600,
        )
        # due is 700 s overdue (> late_reply_threshold_sec=600) -> restart.
        old_id = await db.insert_pending(
            trigger_tg_message_id=91,
            user_id=6,
            trigger="mention",
            due_at=DAY_NOW - 700,
            created_at=DAY_NOW - 4000,
        )

        await responder.restore_pending()

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:restart") == 1

        pending_after_restore = await db.load_pending()
        remaining_ids = {row.id for row in pending_after_restore}
        assert remaining_ids == {recent_id}
        assert old_id not in responder._pending_tasks
        assert recent_id in responder._pending_tasks

        await clock.run_until(responder._pending_tasks[recent_id])

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert SITUATION_LATE in user_content
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


# Reply mode


async def test_reply_mode_depends_on_messages_after_and_delay(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _varied_responses("Okay.", "Right.", "Got it."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        # (a) there are messages after the trigger -> reply_to = trigger id
        await _insert_message(
            db, tg_message_id=10, user_id=5, display_name="Gene", text="trigger", created_at=DAY_NOW
        )
        await _insert_message(
            db,
            tg_message_id=11,
            user_id=6,
            display_name="Nora",
            text="something else",
            created_at=DAY_NOW + 1,
        )
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=10, user_id=5, situation="", delay_sec=5
            ),
        )
        assert bot.sent[-1][2] == 10

        # (b) no messages after and fast -> None
        await _insert_message(
            db,
            tg_message_id=20,
            user_id=5,
            display_name="Gene",
            text="another trigger",
            created_at=DAY_NOW + 10,
        )
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=20, user_id=5, situation="", delay_sec=5
            ),
        )
        assert bot.sent[-1][2] is None

        # (c) ambient -> always None, even with a long delay
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=20,
                user_id=5,
                situation="",
                delay_sec=999_999,
            ),
        )
        assert bot.sent[-1][2] is None
    finally:
        await responder.shutdown()
        await llm.aclose()


# For an address, the situation names the one who really addressed the bot (a live bug: with
# 30 messages of context the model answered the most striking one, not the one who called).


async def test_mention_reply_states_who_addressed_it(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Hello to you too."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=500,
            user_id=5,
            display_name="Gene",
            text="@gusbot how are you?",
            created_at=DAY_NOW,
        )
        msg = _gate_message(
            tg_message_id=500, user_id=5, text="@gusbot how are you?", created_at=DAY_NOW
        )
        await responder.on_gate_pass(msg, Trigger.MENTION, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        pending_id = (await db.load_pending())[0].id
        await clock.run_until(responder._pending_tasks[pending_id])

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert _addressed("Gene", "@gusbot how are you?") in user_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_ambient_reply_has_no_addressed_situation_line(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=501,
            user_id=5,
            display_name="Gene",
            text="hi everyone",
            created_at=DAY_NOW,
        )
        msg = _gate_message(tg_message_id=501, user_id=5, text="hi everyone", created_at=DAY_NOW)
        await responder.on_gate_pass(msg, Trigger.AMBIENT, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert " just spoke to you" not in user_content
        assert DEMO_FRAG.text("situations.addressed_multi_header") not in user_content
        assert SITUATION_AMBIENT in user_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_mention_late_reply_has_both_addressed_and_late_situation(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Lost in my own thoughts."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        pending_id = await db.insert_pending(
            trigger_tg_message_id=502,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 3600,
        )
        # The address was queued in this same process -> _pending_info holds the pair.
        responder._pending_info[pending_id] = [("Gene", "gus, are you there?")]
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=502,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 3600,
            done_at=None,
        )

        await _drive(clock, responder._fire_pending(row))

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert _addressed("Gene", "gus, are you there?") in user_content
        assert SITUATION_LATE in user_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_restored_pending_takes_address_from_messages_after_restart(db: Database) -> None:
    """After a process restart _pending_info is empty: the name and text of the address are
    restored from messages (message_by_tg_id by trigger_tg_message_id)."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Well, yes."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=503,
            user_id=5,
            display_name="Gene",
            text="gus, how are you?",
            created_at=DAY_NOW - 30,
        )
        pending_id = await db.insert_pending(
            trigger_tg_message_id=503,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 30,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=503,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 30,
            done_at=None,
        )
        # No entry in _pending_info, as if the process restarted between queueing the pending
        # and its firing.
        assert pending_id not in responder._pending_info

        await _drive(clock, responder._fire_pending(row))

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert _addressed("Gene", "gus, how are you?") in user_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_collapsed_mentions_list_all_addressers_in_situation(db: Database) -> None:
    """Collapsing the debounce buffer appends the new address to those already gathered for
    this pending instead of replacing them: the situation lists everyone who addressed the bot."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Hello all."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        msg1 = _gate_message(
            tg_message_id=520, user_id=5, text="gus, you there?", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg1, "Gene")

        msg2 = _gate_message(
            tg_message_id=521, user_id=6, text="marlow, anyone home?", created_at=DAY_NOW + 2
        )
        await responder._handle_debounced(Trigger.NAME, msg2, "Nora")

        pending_id = (await db.load_pending())[0].id
        assert responder._pending_info[pending_id] == [
            ("Gene", "gus, you there?"),
            ("Nora", "marlow, anyone home?"),
        ]
        await clock.run_until(responder._pending_tasks[pending_id])

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("situations.addressed_multi_header") in user_content
        assert _multi_line("Gene", "gus, you there?") in user_content
        assert _multi_line("Nora", "marlow, anyone home?") in user_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_restored_pending_collects_multiple_addresses_from_messages(db: Database) -> None:
    """Restoring after a restart also gathers several addresses, not only the original
    trigger: any reply to the bot or mention/name_trigger message since the pending's
    created_at counts as an address too."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Hello all."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=530,
            user_id=5,
            display_name="Gene",
            text="@gusbot how are you?",
            created_at=DAY_NOW - 20,
        )
        await _insert_message(
            db,
            tg_message_id=531,
            user_id=6,
            display_name="Nora",
            text="marlow, you there?",
            created_at=DAY_NOW - 10,
        )
        pending_id = await db.insert_pending(
            trigger_tg_message_id=530,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 20,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=530,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 20,
            done_at=None,
        )

        await _drive(clock, responder._fire_pending(row))

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("situations.addressed_multi_header") in user_content
        assert _multi_line("Gene", "@gusbot how are you?") in user_content
        assert _multi_line("Nora", "marlow, you there?") in user_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_collect_addressed_items_skips_messages_of_later_pending(db: Database) -> None:
    """After a restart pending A does not pick up an address that has its own pending B
    (B appeared while A was being generated), otherwise it would be answered twice."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        await _insert_message(
            db,
            tg_message_id=600,
            user_id=5,
            display_name="Gene",
            text="marlow, you there?",
            created_at=DAY_NOW - 100,
        )
        await _insert_message(
            db,
            tg_message_id=601,
            user_id=6,
            display_name="Nora",
            text="marlow, answer me too",
            created_at=DAY_NOW - 90,
        )
        await _insert_message(
            db,
            tg_message_id=602,
            user_id=7,
            display_name="Mike",
            text="marlow, and then me",
            created_at=DAY_NOW - 50,
        )
        a_id = await db.insert_pending(
            trigger_tg_message_id=600,
            user_id=5,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 100,
        )
        await db.insert_pending(
            trigger_tg_message_id=602,
            user_id=7,
            trigger="name",
            due_at=DAY_NOW + 60,
            created_at=DAY_NOW - 50,
        )
        row_a = next(p for p in await db.load_pending() if p.id == a_id)

        items = await responder._collect_addressed_items(row_a)

        assert [name for name, _ in items] == ["Gene", "Nora"]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_collect_addressed_items_same_second_boundary_by_message_id(
    db: Database,
) -> None:
    """The boundary between pendings is by message id, not by seconds: an address in the same
    second as the next pending's creation, but earlier than its trigger, stays."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        for tg_id, name, at in ((610, "Gene", DAY_NOW - 100), (611, "Nora", DAY_NOW - 50)):
            await _insert_message(
                db,
                tg_message_id=tg_id,
                user_id=tg_id,
                display_name=name,
                text="marlow, hello",
                created_at=at,
            )
        await _insert_message(
            db,
            tg_message_id=612,
            user_id=612,
            display_name="Mike",
            text="marlow, me too",
            created_at=DAY_NOW - 50,
        )
        a_id = await db.insert_pending(
            trigger_tg_message_id=610,
            user_id=610,
            trigger="name",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 100,
        )
        await db.insert_pending(
            trigger_tg_message_id=612,
            user_id=612,
            trigger="name",
            due_at=DAY_NOW + 60,
            created_at=DAY_NOW - 50,
        )
        row_a = next(p for p in await db.load_pending() if p.id == a_id)

        items = await responder._collect_addressed_items(row_a)

        assert [name for name, _ in items] == ["Gene", "Nora"]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_collect_addressed_items_caps_at_five(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=540,
            user_id=5,
            display_name="Trigger",
            text="marlow, hiya",
            created_at=DAY_NOW - 100,
        )
        for i in range(6):
            await _insert_message(
                db,
                tg_message_id=541 + i,
                user_id=10 + i,
                display_name=f"User{i}",
                text="marlow, hello",
                created_at=DAY_NOW - 90 + i,
            )
        row = PendingRow(
            id=999999,
            trigger_tg_message_id=540,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 100,
            done_at=None,
        )

        items = await responder._collect_addressed_items(row)

        assert len(items) == 5
        names = [name for name, _ in items]
        assert "Trigger" not in names  # the oldest one is pushed out by the cap
        assert "User0" not in names
        assert names == ["User1", "User2", "User3", "User4", "User5"]
    finally:
        await responder.shutdown()
        await llm.aclose()


# _fire_pending survives an error in db.recent_messages: it is logged, the pending is NOT
# marked done (an unhandled exception inside generation is not a filter_log outcome), the
# process lives on and the next pending is handled normally.


async def test_fire_pending_survives_recent_messages_error_and_continues(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Okay, alive."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        real_recent_messages = db.recent_messages
        state = {"raised": False}

        async def flaky_recent_messages(chat_id: int, limit: int) -> list[object]:
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("db is on fire")
            return await real_recent_messages(chat_id, limit)  # type: ignore[return-value]

        monkeypatch.setattr(db, "recent_messages", flaky_recent_messages)

        pending_id_1 = await db.insert_pending(
            trigger_tg_message_id=900,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row1 = PendingRow(
            id=pending_id_1,
            trigger_tg_message_id=900,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        with caplog.at_level(logging.ERROR):
            await responder._fire_pending(row1)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert bot.sent == []
        assert calls == []
        # The pending is NOT marked done: an unhandled exception inside generation (before
        # the model call), not a filter_log outcome. restore_pending on the next start must
        # see it again and retry (see the test below about surviving an exception in the LLM
        # call itself).
        pending_after_failure = await db.load_pending()
        assert [row.id for row in pending_after_failure] == [pending_id_1]
        assert pending_after_failure[0].done_at is None

        # The process is alive: the next PASS (a new pending) is handled normally.
        pending_id_2 = await db.insert_pending(
            trigger_tg_message_id=901,
            user_id=6,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row2 = PendingRow(
            id=pending_id_2,
            trigger_tg_message_id=901,
            user_id=6,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )
        await _drive(clock, responder._fire_pending(row2))

        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Okay, alive."
        assert len(calls) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


# Incident: mark_pending_done used to be set at the start of handling, so SIGTERM during the
# model call lost the reply for good (restore_pending no longer saw it, already done). Now done
# is not set on an unhandled exception inside llm.call itself: the pending stays in the DB and
# the next restore_pending (emulating a process restart) picks it up and the retry succeeds.


async def test_pending_stays_undone_on_llm_exception_and_retries_after_restore(
    db: Database,
) -> None:
    cfg = _config()
    state = {"raised": False}

    def handler(_request: httpx.Request) -> httpx.Response:
        if not state["raised"]:
            state["raised"] = True
            # Not an httpx.HTTPError/TimeoutException: LLMClient neither catches it nor turns it
            # into an LLMError, exactly like a "transport drop during SIGTERM".
            raise RuntimeError("transport aborted (simulated SIGTERM)")
        return _ok_response("Back after the restart.")

    llm, calls = _make_llm(cfg, db, handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        pending_id = await db.insert_pending(
            trigger_tg_message_id=950,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=950,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await responder._fire_pending(row)

        assert bot.sent == []
        assert len(calls) == 1
        pending_after_crash = await db.load_pending()
        assert [r.id for r in pending_after_crash] == [pending_id]
        assert pending_after_crash[0].done_at is None

        # Emulate a process restart: restore_pending on the new start sees the unfinished
        # pending again (the overdue time is small: a timer at once, not send:restart).
        await responder.restore_pending()
        assert pending_id in responder._pending_tasks

        await clock.run_until(responder._pending_tasks[pending_id])

        assert len(calls) == 2
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Back after the restart."
        assert await db.load_pending() == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_pending_marked_done_after_speak_false_final_decision(db: Database) -> None:
    """speak=false is the final decision to stay silent (the filter_log outcome llm:silent),
    so the pending is marked done."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _silent_response())
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        pending_id = await db.insert_pending(
            trigger_tg_message_id=970,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=970,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await responder._fire_pending(row)

        assert bot.sent == []
        assert len(calls) == 1
        assert await db.load_pending() == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:silent") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_pending_marked_done_after_successful_send(db: Database) -> None:
    """A successful send: the pending is marked done after insert_bot_reply, that is, after the
    reply has already appeared in bot_replies."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Well, hello."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        pending_id = await db.insert_pending(
            trigger_tg_message_id=980,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        row = PendingRow(
            id=pending_id,
            trigger_tg_message_id=980,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
            done_at=None,
        )

        await _drive(clock, responder._fire_pending(row))

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert (await db.recent_bot_replies(5)) == ["Well, hello."]
        assert await db.load_pending() == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_shutdown_waits_for_inflight_generation_before_cancelling(db: Database) -> None:
    """shutdown() during a hanging model call (_respond_lock is held) does not cancel the
    generation at once: it waits (asyncio.wait_for on the lock) until the LLM fake, stuck on an
    Event, is released; the reply must get out in time."""
    cfg = _config()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return _ok_response("Finished after shutdown.")

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
        msg = _gate_message(tg_message_id=990, user_id=5, text="hi everyone", created_at=DAY_NOW)
        gen_task: asyncio.Task[None] = asyncio.ensure_future(
            responder._handle_debounced(Trigger.AMBIENT, msg, "User1")
        )
        await started.wait()  # generation is inside llm.call, holding _respond_lock

        assert responder._respond_lock.locked()
        shutdown_task: asyncio.Task[None] = asyncio.ensure_future(responder.shutdown())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not shutdown_task.done()  # waits for the lock, does not cancel until the LLM answers

        release.set()  # the LLM "answers": the process lived long enough to get the reply
        await gen_task
        await shutdown_task

        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Finished after shutdown."
    finally:
        await llm.aclose()


# ---------------------------------------------------------------------------
# Followup: the cheap "is this for me?" check turns a message into an address. bot.py has
# already filtered the candidates by FOLLOWUP_REASONS/hot_until/checker.check(); responder.py
# gets on_gate_pass(gm, Trigger.FOLLOWUP, display_name) and must behave as with any other
# address (mention/reply/name).
# ---------------------------------------------------------------------------


async def test_followup_creates_pending_like_other_address_triggers(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("That happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _insert_message(
            db,
            tg_message_id=700,
            user_id=5,
            display_name="Gene",
            text="what a day that was",
            created_at=DAY_NOW,
        )
        msg = _gate_message(
            tg_message_id=700, user_id=5, text="what a day that was", created_at=DAY_NOW
        )
        await responder.on_gate_pass(msg, Trigger.FOLLOWUP, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        pending_rows = await db.load_pending()
        assert len(pending_rows) == 1
        assert pending_rows[0].trigger == "followup"

        await clock.run_until(responder._pending_tasks[pending_rows[0].id])

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        followup = DEMO_FRAG.text("situations.followup_single")
        expected = followup.replace("{name}", "Gene").replace("{text}", "what a day that was")
        assert expected in user_content

        last = await db.last_bot_replies(1)
        assert last[0].trigger == "followup"

        tz = cfg.persona.timezone
        assert await db.get_state(day_key("mention_count", clock.now(), tz)) == "1"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_followup_mention_delay_capped_by_hot_window(db: Database) -> None:
    """followup is an address, so in the hot window its delay is capped by
    mention_max_delay_sec too, like mention/reply/name."""
    cfg = _config()
    cfg.behaviour.hot_window.mention_max_delay_sec = 120
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("That happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("hot_until", str(DAY_NOW + 1000))
        msg = _gate_message(
            tg_message_id=701, user_id=5, text="what a day that was", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.FOLLOWUP, msg, "Gene")

        rows = await db.load_pending()
        assert len(rows) == 1
        assert rows[0].due_at == DAY_NOW + cfg.behaviour.hot_window.mention_max_delay_sec
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_llm_retry_keeps_collapsed_address_and_earlier_due(db: Database) -> None:
    """A6: while the model call was running, a new address collapsed into the pending; the
    retry overwrites neither it nor the earlier due_at."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Oh come on."))
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        pid = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW + 30,
            created_at=DAY_NOW - 60,
        )
        responder._pending_info[pid] = [("Pete", "what do you think?")]
        assert await responder._retry_pending_later(pid, 50, [("Vic", "@bot hello")], DAY_NOW)
        pending = await db.load_pending()
        assert pending[0].due_at == DAY_NOW + 30
        assert responder._pending_info[pid] == [
            ("Vic", "@bot hello"),
            ("Pete", "what do you think?"),
        ]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_llm_http_failure_retries_address_pending_once(db: Database) -> None:
    """A6: llm:http on an address -> retry in 120-300 s, a second failure -> silence."""
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, lambda _req: httpx.Response(500, text="boom"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        pid = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        responder._pending_info[pid] = [("Vic", "@bot hello")]
        await responder._fire_pending(_pending_row(pid, "mention", DAY_NOW))

        pending = await db.load_pending()
        assert [p.id for p in pending] == [pid]
        assert DAY_NOW + 120 <= pending[0].due_at <= DAY_NOW + 300
        assert responder._pending_info[pid] == [("Vic", "@bot hello")]
        assert pid in responder._pending_tasks
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:llm_retry") == 1
        assert "llm:http" not in summary

        # Cancel the timer (it would fire on its own), drive the second failure by hand.
        responder._pending_tasks.pop(pid).cancel()
        await responder._fire_pending(_pending_row(pid, "mention", pending[0].due_at))

        assert await db.load_pending() == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("send:llm_retry") == 1
        assert summary.get("llm:http") == 1
        assert bot.sent == []
        assert len(calls) == 2
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_llm_http_failure_does_not_retry_ambient(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, _calls = _make_llm(cfg, db, lambda _req: httpx.Response(500, text="boom"))
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, FakeBot(), clock)
    try:
        await responder._respond(
            trigger=Trigger.AMBIENT, trigger_msg_id=None, user_id=None, situation="", delay_sec=0
        )
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:http") == 1
        assert "send:llm_retry" not in summary
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_llm_circuit_open_does_not_retry_address_pending(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.min_gap_sec = 0
    llm, calls = _make_llm(cfg, db, _fail_handler)
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, FakeBot(), clock)
    try:
        await db.set_state("llm_circuit_until", str(DAY_NOW + 600))
        pid = await db.insert_pending(
            trigger_tg_message_id=50,
            user_id=5,
            trigger="mention",
            due_at=DAY_NOW,
            created_at=DAY_NOW - 60,
        )
        await responder._fire_pending(_pending_row(pid, "mention", DAY_NOW))
        assert await db.load_pending() == []
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:circuit_open") == 1
        assert "send:llm_retry" not in summary
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()
