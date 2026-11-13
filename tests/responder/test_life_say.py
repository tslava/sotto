"""Responder tests: life_say."""

from __future__ import annotations

import httpx
import pytest

from sotto import filters as filters_module
from sotto.db import Database
from sotto.filters import FilterVerdict
from sotto.gate_types import Trigger
from sotto.stickers import Sticker
from sotto.timeutil import day_key
from tests.responder.helpers import (
    DAY_NOW,
    PROMPT_TEMPLATE_WITH_LIFE,
    FakeBot,
    FakeClock,
    FakePromptStore,
    FakeStickerChooser,
    MaxRandom,
    MinRandom,
    _config,
    _drive,
    _fail_handler,
    _gate_message,
    _make_llm,
    _make_responder,
    _noop_open_hot_window,
    _ok_response,
    _payload,
)


async def test_announce_life_sends_and_marks_event_announced(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Well, so he sold it after all."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        event_id = await db.insert_life_event(
            text="sold the old trailer, bought a Ford", created_at=DAY_NOW
        )
        event = await db.life_event(event_id)
        assert event is not None

        task = await _drive(clock, responder.announce_life(event))
        outcome = task.result()

        assert outcome.sent is True
        assert outcome.text == "Well, so he sold it after all."
        assert outcome.reason == "send:life"
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Well, so he sold it after all."
        assert bot.sent[0][2] is None  # not a reply

        announced = await db.life_event(event_id)
        assert announced is not None
        assert announced.announced_tg_message_id is not None

        last = await db.last_bot_replies(1)
        assert last[0].trigger == "life"
        assert last[0].tg_message_id == announced.announced_tg_message_id
        # now is fixed at the start of announce_life, before the typing pause: the same moment
        # as bot_replies.created_at (clock.now() after _drive is already different, the
        # typing loop has advanced the FakeClock).
        assert announced.announced_at == last[0].created_at

        # The situation (the news retold) goes into the user message, not into system.
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert "sold the old trailer, bought a Ford" in user_content

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:life" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_does_not_touch_mention_or_ambient_counters(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        task = await _drive(clock, responder.announce_life(event))
        assert task.result().sent is True

        tz = cfg.persona.timezone
        assert await db.get_state(day_key("mention_count", clock.now(), tz)) is None
        assert await db.get_state(day_key("ambient_count", clock.now(), tz)) is None
        assert await db.get_state("last_mention_reply_at") is None
        assert await db.get_state("last_ambient_at") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_never_chooses_sticker(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    chooser = FakeStickerChooser(sticker=Sticker(id=1, file_id="F1", text="T", enabled=True))
    responder = _make_responder(db, cfg, llm, bot, clock, sticker_chooser=chooser)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        await _drive(clock, responder.announce_life(event))

        assert chooser.calls == []
        assert bot.sent_stickers == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_blocked_by_panic(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("panic", "1")
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        outcome = await responder.announce_life(event)

        assert outcome.sent is False
        assert outcome.reason == "blocked:panic"
        assert bot.sent == []
        still = await db.life_event(event_id)
        assert still is not None
        assert still.announced_at is None

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:blocked_panic" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_blocked_by_stop(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("stop_until", str(DAY_NOW + 1000))
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        outcome = await responder.announce_life(event)

        assert outcome.sent is False
        assert outcome.reason == "blocked:stop"
        assert bot.sent == []

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:blocked_stop" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_llm_error_returns_reason_and_does_not_send(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: httpx.Response(500, text="boom"))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        outcome = await responder.announce_life(event)

        assert outcome.sent is False
        assert outcome.reason == "llm:http"
        assert bot.sent == []
        still = await db.life_event(event_id)
        assert still is not None
        assert still.announced_at is None

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "llm:http" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_filter_cut_not_shadow_returns_sent_false_with_candidate(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    cfg.filters.shadow = False
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("A candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(text: str, ctx: object, judge: object = None) -> FilterVerdict:
        return FilterVerdict(ok=False, reason="regex:length", reasons=("regex:length",))

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        outcome = await responder.announce_life(event)

        assert outcome.sent is False
        assert outcome.text == "A candidate."
        assert outcome.reason == "regex:length"
        assert bot.sent == []

        still = await db.life_event(event_id)
        assert still is not None
        assert still.announced_at is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_shadow_mode_still_sends(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    cfg.filters.shadow = True
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("A candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(text: str, ctx: object, judge: object = None) -> FilterVerdict:
        return FilterVerdict(ok=False, reason="regex:length", reasons=("regex:length",))

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        task = await _drive(clock, responder.announce_life(event))
        outcome = task.result()

        assert outcome.sent is True
        assert outcome.reason == "send:life"
        assert len(bot.sent) == 1

        announced = await db.life_event(event_id)
        assert announced is not None
        assert announced.announced_at is not None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_life_slot_filled_in_system_for_ordinary_ambient_reply(db: Database) -> None:
    """The {life} slot is ALWAYS filled, for any trigger, not only for announce_life."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    prompt_store = FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_LIFE)
    responder = _make_responder(db, cfg, llm, bot, clock, prompt_store=prompt_store)
    try:
        await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW - 100)

        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=900, user_id=None, situation="", delay_sec=0
            ),
        )

        payload = _payload(calls[0])
        system_content = payload["messages"][0]["content"]  # type: ignore[index]
        assert "sold the old trailer" in system_content
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_say_sends_text_verbatim_without_model(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        task = await _drive(clock, responder.say("Well hello, then, anyway."))
        outcome = task.result()

        assert outcome.sent is True
        assert outcome.text == "Well hello, then, anyway."
        assert outcome.reason == "send:say"
        assert calls == []  # the model was not called
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Well hello, then, anyway."

        last = await db.last_bot_replies(1)
        assert last[0].trigger == "say"
        assert last[0].text == "Well hello, then, anyway."
        assert last[0].trigger_tg_message_id is None

        assert await db.get_state("replies_since_sticker") == "1"

        tz = cfg.persona.timezone
        assert await db.get_state(day_key("mention_count", clock.now(), tz)) is None
        assert await db.get_state(day_key("ambient_count", clock.now(), tz)) is None

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:say" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_say_blocked_by_panic(db: Database) -> None:
    cfg = _config()
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("panic", "1")

        outcome = await responder.say("some text")

        assert outcome.sent is False
        assert outcome.reason == "blocked:panic"
        assert bot.sent == []
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_say_blocked_by_stop(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("stop_until", str(DAY_NOW + 1000))

        outcome = await responder.say("some text")

        assert outcome.sent is False
        assert outcome.reason == "blocked:stop"
        assert bot.sent == []
    finally:
        await responder.shutdown()
        await llm.aclose()


# ---------------------------------------------------------------------------
# The hot window after /life and /say: announce_life/say open or extend the window, addresses
# are capped by mention_max_delay_sec, and ambient inside the window lives on its own budget
# (hot_ambient_count) instead of the daily one.
# ---------------------------------------------------------------------------


async def test_announce_life_opens_hot_window(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Sold it, yes."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        task = await _drive(clock, responder.announce_life(event))
        assert task.result().sent is True

        # now is fixed at the start of announce_life, before the typing pause (the same logic
        # as announced_at in test_announce_life_sends_and_marks_event_announced).
        hot_until_raw = await db.get_state("hot_until")
        assert hot_until_raw == str(DAY_NOW + cfg.behaviour.hot_window.minutes * 60)
        assert await db.get_state("hot_ambient_count") == "0"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_does_not_open_hot_window_when_disabled(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.hot_window.enabled = False
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Sold it, yes."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        await _drive(clock, responder.announce_life(event))

        assert await db.get_state("hot_until") is None
        assert await db.get_state("hot_ambient_count") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_does_not_open_hot_window_when_not_sent(db: Database) -> None:
    """Blocked by panic -> outcome.sent=False -> the window does not open."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("panic", "1")
        event_id = await db.insert_life_event(text="sold the old trailer", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        outcome = await responder.announce_life(event)
        assert outcome.sent is False

        assert await db.get_state("hot_until") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_extends_existing_hot_window_and_resets_counter(db: Database) -> None:
    """A repeated /life inside an already open window extends it afresh from now and resets
    the counter: the old ambient replies of that window do not carry over to the new one."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("That happens too."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("hot_until", str(DAY_NOW + 60))
        await db.set_state("hot_ambient_count", "3")

        event_id = await db.insert_life_event(text="got a new dog", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        await _drive(clock, responder.announce_life(event))

        expected_hot_until = DAY_NOW + cfg.behaviour.hot_window.minutes * 60
        assert await db.get_state("hot_until") == str(expected_hot_until)
        assert await db.get_state("hot_ambient_count") == "0"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_say_opens_hot_window(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        task = await _drive(clock, responder.say("Well hello."))
        assert task.result().sent is True

        assert await db.get_state("hot_until") == str(
            DAY_NOW + cfg.behaviour.hot_window.minutes * 60
        )
        assert await db.get_state("hot_ambient_count") == "0"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_say_does_not_open_hot_window_when_blocked(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("stop_until", str(DAY_NOW + 1000))

        outcome = await responder.say("some text")
        assert outcome.sent is False

        assert await db.get_state("hot_until") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_mention_delay_capped_by_hot_window(db: Database) -> None:
    """Without the window (MaxRandom) the ordinary bucket would give 3600s; inside the window
    the answer to an address cannot be postponed beyond mention_max_delay_sec."""
    cfg = _config()
    cfg.behaviour.hot_window.mention_max_delay_sec = 120
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MaxRandom())
    try:
        await db.set_state("hot_until", str(DAY_NOW + 1000))

        msg = _gate_message(
            tg_message_id=30, user_id=5, text="gus, how are you?", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg, "Gene")

        rows = await db.load_pending()
        assert len(rows) == 1
        assert rows[0].due_at == DAY_NOW + cfg.behaviour.hot_window.mention_max_delay_sec
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_mention_delay_not_capped_outside_hot_window(db: Database) -> None:
    """The same scenario (MaxRandom, the long bucket) but with no open window: the cap does
    not apply, the delay stays ordinary (unrelated to mention_max_delay_sec)."""
    cfg = _config()
    cfg.behaviour.hot_window.mention_max_delay_sec = 120
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MaxRandom())
    try:
        msg = _gate_message(
            tg_message_id=31, user_id=5, text="gus, how are you?", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg, "Gene")

        rows = await db.load_pending()
        assert len(rows) == 1
        assert rows[0].due_at - DAY_NOW > cfg.behaviour.hot_window.mention_max_delay_sec
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_mention_earliest_capped_by_hot_window(db: Database) -> None:
    """The cooldown shift (earliest) inside the window is also capped by
    mention_max_delay_sec: without the cap earliest would be DAY_NOW-10+200=DAY_NOW+190."""
    cfg = _config()
    cfg.behaviour.mention_chat_cooldown_sec = 200
    cfg.behaviour.hot_window.mention_max_delay_sec = 120
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock, rng=MinRandom())
    try:
        await db.set_state("hot_until", str(DAY_NOW + 1000))
        await db.set_state("last_mention_reply_at", str(DAY_NOW - 10))

        msg = _gate_message(
            tg_message_id=32, user_id=5, text="gus, where are you", created_at=DAY_NOW
        )
        await responder._handle_debounced(Trigger.NAME, msg, "Gene")

        rows = await db.load_pending()
        assert len(rows) == 1
        # without the window earliest would be DAY_NOW+190; the window caps it at DAY_NOW+120,
        # MinRandom.randint -> 5 on top of earliest.
        assert rows[0].due_at == DAY_NOW + 120 + 5
        assert calls == []
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_ambient_recheck_hot_cap_blocks_send_in_hot_window(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.hot_window.ambient_cap = 1
    llm, calls = _make_llm(cfg, db, _fail_handler)
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("hot_until", str(DAY_NOW + 1000))
        await db.set_state("hot_ambient_count", "1")

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
        assert summary.get("send:recheck_hot_cap") == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_ambient_reply_in_hot_window_increments_hot_counter_not_daily(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.set_state("hot_until", str(DAY_NOW + 1000))
        # _maybe_open_hot_window is stubbed out on purpose: this test checks which counter is
        # incremented BEFORE the tail that opens or extends the window (the window opens after
        # any successful send, not only announce_life/say, and would reset hot_ambient_count
        # in the same call). A separate test below checks that the window really reopens
        # after an ambient reply.
        responder._maybe_open_hot_window = _noop_open_hot_window  # type: ignore[method-assign]

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

        assert len(bot.sent) == 1
        tz = cfg.persona.timezone
        assert await db.get_state(day_key("ambient_count", clock.now(), tz)) is None
        assert await db.get_state("last_ambient_at") is None
        assert await db.get_state("hot_ambient_count") == "1"

        last = await db.last_bot_replies(1)
        assert last[0].trigger == "ambient"  # bot_replies.trigger stays ambient

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:ambient_hot" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_ambient_reply_outside_hot_window_uses_daily_budget_as_before(db: Database) -> None:
    """Regression: with no previously opened window an ambient reply spends the daily budget
    (not the "hot" one) and does NOT open a new window: the shared tail of
    ``_generate_and_send`` only does that with ``hot_window.open_on_any_reply``."""
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
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

        tz = cfg.persona.timezone
        assert await db.get_state(day_key("ambient_count", clock.now(), tz)) == "1"
        assert await db.get_state("last_ambient_at") is not None
        assert await db.get_state("hot_ambient_count") is None
        assert await db.get_state("hot_until") is None

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:ambient" for reason, _count in logs)
        assert not any(reason == "send:ambient_hot" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_morning_reply_does_not_open_hot_window_by_default(db: Database) -> None:
    """An ordinary reply (the morning one included) does not open the window:
    ``hot_window.open_on_any_reply`` is false by default, the window stays the /life and /say
    button."""
    cfg = _config()
    assert cfg.behaviour.hot_window.open_on_any_reply is False
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Good morning, was busy."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.enqueue_night(
            tg_message_id=200,
            user_id=5,
            display_name="Gene",
            text="gus are you there?",
            created_at=DAY_NOW - 3600,
        )

        await _drive(clock, responder._run_morning_once())

        assert len(bot.sent) == 1
        assert await db.get_state("hot_until") is None
        assert await db.get_state("hot_ambient_count") is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_morning_reply_opens_hot_window_when_open_on_any_reply(db: Database) -> None:
    """The old behaviour comes back with a flag: with ``open_on_any_reply: true`` the shared
    tail of _generate_and_send opens the window again after any successful send."""
    cfg = _config()
    cfg.behaviour.hot_window.open_on_any_reply = True
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Good morning, was busy."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)
    try:
        await db.enqueue_night(
            tg_message_id=201,
            user_id=5,
            display_name="Gene",
            text="gus are you there?",
            created_at=DAY_NOW - 3600,
        )

        await _drive(clock, responder._run_morning_once())

        assert len(bot.sent) == 1
        hot_until_raw = await db.get_state("hot_until")
        assert hot_until_raw is not None
        assert (
            int(hot_until_raw) > DAY_NOW
        )  # the window is open into the future from the moment of the reply
        assert await db.get_state("hot_ambient_count") == "0"
    finally:
        await responder.shutdown()
        await llm.aclose()
