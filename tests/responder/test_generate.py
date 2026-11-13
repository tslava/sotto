"""Responder tests: generate."""

from __future__ import annotations

import asyncio
import logging

import pytest

from sotto.db import Database
from sotto.gate_types import Trigger
from sotto.timeutil import day_key
from tests.responder.helpers import (
    CHAT_ID,
    DAY_NOW,
    PROMPT_TEMPLATE,
    FakeBot,
    FakeClock,
    FakePromptStore,
    _config,
    _drive,
    _fail_handler,
    _gate_message,
    _insert_human,
    _invalid_json_response,
    _make_llm,
    _make_responder,
    _ok_response,
    _payload,
    _silent_response,
    _varied_responses,
)


async def test_speak_false_is_silent(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _silent_response())
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await responder._respond(
            trigger=Trigger.MENTION, trigger_msg_id=60, user_id=5, situation="", delay_sec=40
        )

        assert bot.sent == []
        assert len(calls) == 1
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:silent") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_invalid_json_is_logged_and_silent(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _invalid_json_response())
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await responder._respond(
            trigger=Trigger.MENTION, trigger_msg_id=70, user_id=5, situation="", delay_sec=40
        )

        assert bot.sent == []
        assert len(calls) == 1
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:invalid_json") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_llm_budget_error_is_logged(db: Database) -> None:
    cfg = _config()
    cfg.llm.daily_budget_usd = 0.01
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    tz = cfg.persona.timezone
    await db.add_state_float(day_key("llm_spent_usd", DAY_NOW, tz), 1.0)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await responder._respond(
            trigger=Trigger.MENTION, trigger_msg_id=80, user_id=5, situation="", delay_sec=40
        )

        assert bot.sent == []
        assert calls == []  # llm:budget stops it before the network
        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:budget") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_typing_called_before_send(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(
        cfg, db, lambda _req: _ok_response("A somewhat longer answer for the typing indicator.")
    )
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=300, user_id=5, situation="", delay_sec=5
            ),
        )

        kinds = [event[0] for event in bot.events]
        assert "typing" in kinds
        send_index = kinds.index("send")
        typing_index = kinds.index("typing")
        assert typing_index < send_index
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_prompt_store_is_read_fresh_on_each_respond(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _varied_responses("An answer.", "Another answer."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    prompt_store = FakePromptStore(
        prompt=PROMPT_TEMPLATE, few_shot="Julie: hello\n{}", prompt_version=1, few_shot_version=1
    )
    responder = _make_responder(db, cfg, llm, bot, clock, prompt_store=prompt_store)
    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=1000, user_id=5, situation="", delay_sec=5
            ),
        )
        payload = _payload(calls[0])
        system_content = payload["messages"][0]["content"]  # type: ignore[index]
        assert "Julie: hello" in system_content

        last = await db.last_bot_replies(1)
        assert last[0].prompt_version == 1
        assert last[0].few_shot_version == 1

        # Change the versions on the PromptStore "from outside" (as /rollback and /ex add do):
        # without recreating the Responder the next _respond must see the new values.
        prompt_store.prompt = "A new system prompt for {age}, few-shot: {few_shot}"
        prompt_store.few_shot = "Gene: bye\n{}"
        prompt_store.prompt_version_value = 2
        prompt_store.few_shot_version_value = 3

        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=1001, user_id=5, situation="", delay_sec=5
            ),
        )
        payload2 = _payload(calls[1])
        system_content_2 = payload2["messages"][0]["content"]  # type: ignore[index]
        assert "A new system prompt" in system_content_2
        assert "Gene: bye" in system_content_2

        last2 = await db.last_bot_replies(1)
        assert last2[0].prompt_version == 2
        assert last2[0].few_shot_version == 3
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_no_main_model_skips_llm_and_logs_no_model(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    """LLM/Judge/Responder are created in app.py as soon as there is an openrouter_api_key,
    regardless of llm.main_model: an empty model must not crash the Responder, only quietly
    cut the answer and stay off the network."""
    cfg = _config()
    cfg.llm.main_model = ""
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        msg = _gate_message(tg_message_id=900, user_id=5, text="hello everyone", created_at=DAY_NOW)
        with caplog.at_level(logging.WARNING):
            await responder.on_gate_pass(msg, Trigger.AMBIENT, "Gene")
            assert responder._debounce_task is not None
            await clock.run_until(responder._debounce_task)

        assert calls == []
        assert bot.sent == []

        summary = dict(await db.filter_log_summary(0))
        assert summary.get("llm:no_model") == 1

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "main_model" in warnings[0].message
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_main_model_set_hot_enables_llm_on_next_respond(db: Database) -> None:
    """After a hot change of llm.main_model (as ConfigStore.set would do) the next _respond of
    the same Responder must call the LLM, with no Responder rebuild and no process restart."""
    cfg = _config()
    cfg.llm.main_model = ""
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        first_msg = _gate_message(
            tg_message_id=901, user_id=5, text="hello everyone", created_at=DAY_NOW
        )
        await responder.on_gate_pass(first_msg, Trigger.AMBIENT, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        assert calls == []
        assert bot.sent == []

        # Hot model change (the same thing ConfigStore.set does via /set).
        cfg.llm.main_model = "test/model"

        await asyncio.sleep(0)  # let the previous debounce task settle completely
        second_msg = _gate_message(
            tg_message_id=902, user_id=5, text="how is the mood", created_at=DAY_NOW + 30
        )
        await responder.on_gate_pass(second_msg, Trigger.AMBIENT, "Gene")
        assert responder._debounce_task is not None
        await clock.run_until(responder._debounce_task)

        assert len(calls) == 1
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Happens."
    finally:
        await responder.shutdown()
        await llm.aclose()


async def _side_effects(db_: Database) -> dict[str, int]:
    conn = db_._conn
    assert conn is not None
    counts = {}
    for table in ("filter_log", "bot_replies", "state", "self_facts", "pending_replies"):
        cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cursor.fetchone()
        assert row is not None
        counts[table] = row[0]
    return counts


async def test_preview_builds_addressed_prompt_and_writes_nothing(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Fine, one can live with it."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    await _insert_human(db, 40, "who is up for a pint?", DAY_NOW - 60)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        before = await _side_effects(db)
        outcome = await responder.preview("Gus, how are you?", "Sam")

        assert outcome.sent is True
        assert outcome.text == "Fine, one can live with it."
        assert outcome.reason == "send:mention"
        user = _payload(calls[0])["messages"][1]["content"]  # type: ignore[index]
        assert "Sam" in user and "Gus, how are you?" in user
        assert "who is up for a pint?" in user  # fresh chat context
        assert bot.sent == []
        assert bot.events == []
        after = await _side_effects(db)
        # The only write is the llm_calls counter and the spend in state (the call is real).
        assert {k: v for k, v in after.items() if k != "state"} == {
            k: v for k, v in before.items() if k != "state"
        }
        assert await db.get_state(day_key("mention_count", DAY_NOW, cfg.persona.timezone)) is None
        assert await db.get_state("last_mention_reply_at") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_preview_puts_message_into_context_like_live_chat(db: Database) -> None:
    """The /ask message goes at the end of the context in full (in the situation it is cut to
    300), it is not in the DB; the filter sees it the way it does in the chat: an echo is cut."""
    cfg = _config()
    cfg.filters.shadow = False
    long_text = (
        "Gus, " + "a very long story about the neighbour and the kettle, " * 9 + "so what to do?"
    )
    assert len(long_text) > 300
    echo = "The neighbour brought an old kettle from the market yesterday"
    responses = iter([_ok_response("Nothing to do, really."), _ok_response(echo + ".")])
    llm, calls = _make_llm(cfg, db, lambda _req: next(responses))
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        await responder.preview(long_text, "Sam", 7)
        user = _payload(calls[0])["messages"][1]["content"]  # type: ignore[index]
        assert f"Sam: {long_text}" in user
        assert await db.recent_messages(CHAT_ID, 10) == []

        cut = await responder.preview(f"{echo}, can you imagine?", "Sam", 7)
        assert cut.sent is False
        assert "regex:echo" in cut.reasons
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_preview_silent_and_cut(db: Database) -> None:
    cfg = _config()
    cfg.filters.shadow = False
    responses = iter([_silent_response(), _ok_response("One! Two! Three!")])
    llm, _calls = _make_llm(cfg, db, lambda _req: next(responses))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        silent = await responder.preview("Gus?", "Sam")
        assert (silent.sent, silent.reason) == (False, "llm:silent")

        cut = await responder.preview("Gus?", "Sam")
        assert cut.sent is False
        assert cut.reason == "style:exclaim"
        assert "style:exclaim" in cut.reasons
        assert cut.text == "One! Two! Three!"
        assert bot.sent == []
        conn = db._conn
        assert conn is not None
        cursor = await conn.execute("SELECT COUNT(*) FROM filter_log")
        row = await cursor.fetchone()
        assert row is not None and row[0] == 0
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_preview_without_main_model(db: Database) -> None:
    cfg = _config()
    cfg.llm.main_model = ""
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    responder = _make_responder(db, cfg, llm, FakeBot(), FakeClock(DAY_NOW))
    try:
        outcome = await responder.preview("Gus?", "Sam")
        assert (outcome.sent, outcome.reason) == (False, "llm:no_model")
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()
