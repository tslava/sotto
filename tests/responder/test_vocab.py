"""Responder tests: vocab."""

from __future__ import annotations

from sotto.db import Database
from sotto.gate_types import Trigger
from sotto.responder import Responder
from sotto.timeutil import day_key
from tests.demo import DEMO_FRAG
from tests.responder.helpers import (
    DAY_NOW,
    PROMPT_TEMPLATE_WITH_VOCAB,
    FakeBot,
    FakeClock,
    FakePromptStore,
    _ambient,
    _config,
    _drive,
    _insert_human,
    _make_llm,
    _make_responder,
    _ok_response,
    _payload,
    _test_vocab,
    _varied_responses,
)


async def test_vocab_meaning_of_word_from_chat_goes_to_system(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.vocab.probability = 0.0
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    await _insert_human(db, 40, "Gus, what is sloshmire?", DAY_NOW - 60)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _ambient(clock, responder)

        system = _payload(calls[0])["messages"][0]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("vocab.meanings_header") in system
        assert "sloshmire: slushy weather" in system
        assert DEMO_FRAG.text("vocab.offer_header") not in system
        assert "{vocab}" not in system
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_vocab_offer_then_counted_and_paused(db: Database) -> None:
    """The dice came up -> words are offered; a reply with a word -> the day counter and
    "last"; the next generation -> not offered (the word is inside recent_window), and its own
    word is not slipped in even as a meaning without an address."""
    cfg = _config()
    cfg.behaviour.vocab.probability = 1.0
    llm, calls = _make_llm(
        cfg, db, _varied_responses("Zoofa, why stand there.", "Well, it happens.")
    )
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _ambient(clock, responder)
        first = _payload(calls[0])["messages"][0]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("vocab.offer_header") + " (backyard slang)" in first
        assert len(bot.sent) == 1
        assert await db.get_state(day_key("vocab_count", DAY_NOW, cfg.persona.timezone)) == "1"
        assert await db.get_state("vocab_last") == f"{DAY_NOW}|zoofa"

        clock.value += max(cfg.behaviour.min_gap_sec, cfg.behaviour.chat_cooldown_min * 60) + 60
        await _ambient(clock, responder)
        second = _payload(calls[1])["messages"][0]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("vocab.offer_header") not in second
        assert DEMO_FRAG.text("vocab.meanings_header") not in second
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_vocab_unsolicited_word_over_quota_is_cut(db: Database) -> None:
    """A prompt hint does not hold the limit: the model may repeat the word on its own. The
    output filter cuts such a reply (dedup:vocab), even in shadow."""
    cfg = _config()
    cfg.behaviour.vocab.probability = 0.0
    llm, _calls = _make_llm(
        cfg, db, _varied_responses("Zoofa, why stand there.", "Zoofa already, come on.")
    )
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _ambient(clock, responder)
        assert len(bot.sent) == 1  # the first word on its own is allowed

        clock.value += max(cfg.behaviour.min_gap_sec, cfg.behaviour.chat_cooldown_min * 60) + 60
        await _ambient(clock, responder)
        assert len(bot.sent) == 1
        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT stage, reason FROM filter_log WHERE verdict = 'cut' ORDER BY id"
        )
        assert ("dedup", "dedup:vocab") in [tuple(row) for row in await cursor.fetchall()]
        assert await db.get_state(day_key("vocab_count", DAY_NOW, cfg.persona.timezone)) == "1"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_vocab_word_from_question_passes_over_quota(db: Database) -> None:
    """ "Gus, what is zoofa?": the word is in the question, so a reply with it is not cut, and
    the meaning of its own recent word reaches the prompt on a direct address."""
    cfg = _config()
    cfg.behaviour.vocab.probability = 0.0
    tz = cfg.persona.timezone
    await db.set_state(day_key("vocab_count", DAY_NOW, tz), str(cfg.behaviour.vocab.daily_cap))
    await db.insert_bot_reply(
        tg_message_id=30,
        reply_to_tg_message_id=None,
        trigger="ambient",
        trigger_tg_message_id=None,
        text="Zoofa on the bench.",
        prompt_version=1,
        few_shot_version=1,
        delay_sec=0,
        created_at=DAY_NOW - 3600,
    )
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Zoofa means sit down."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION,
                trigger_msg_id=50,
                user_id=7,
                situation="",
                delay_sec=0,
                trigger_text="Gus, what is zoofa?",
                addressed_items=[("Gene", "Gus, what is zoofa?")],
            ),
        )
        system = _payload(calls[0])["messages"][0]["content"]  # type: ignore[index]
        assert "zoofa: a call to sit down" in system
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_vocab_slot_empty_without_vocab_or_when_disabled(db: Database) -> None:
    cfg = _config()
    cfg.behaviour.vocab.probability = 1.0
    llm, calls = _make_llm(cfg, db, _varied_responses("Happens.", "Well, it happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    await _insert_human(db, 40, "sloshmire out there", DAY_NOW - 60)
    plain = _make_responder(
        db, cfg, llm, bot, clock, prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB)
    )
    disabled_cfg = _config()
    disabled_cfg.behaviour.vocab.enabled = False
    disabled = _make_responder(
        db,
        disabled_cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _ambient(clock, plain)
        clock.value += max(cfg.behaviour.min_gap_sec, cfg.behaviour.chat_cooldown_min * 60) + 60
        await _ambient(clock, disabled)

        assert len(calls) == 2
        for call in calls:
            system = _payload(call)["messages"][0]["content"]  # type: ignore[index]
            assert DEMO_FRAG.text("vocab.offer_header") not in system
            assert DEMO_FRAG.text("vocab.meanings_header") not in system
            assert "{vocab}" not in system
    finally:
        await plain.shutdown()
        await disabled.shutdown()
        await llm.aclose()


async def _mention(clock: FakeClock, responder: Responder, text: str, trigger: Trigger) -> None:
    await _drive(
        clock,
        responder._respond(
            trigger=trigger,
            trigger_msg_id=50,
            user_id=7,
            situation="",
            delay_sec=0,
            trigger_text=text,
            addressed_items=[("Gene", text)],
        ),
    )


async def test_vocab_request_offers_words_without_dice_and_quota(db: Database) -> None:
    """ "Say something backyard-style": words are offered without the dice and the limit, and a
    reply with them passes the filter even over the daily ceiling."""
    cfg = _config()
    cfg.behaviour.vocab.probability = 0.0
    tz = cfg.persona.timezone
    await db.set_state(day_key("vocab_count", DAY_NOW, tz), str(cfg.behaviour.vocab.daily_cap))
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Hlooss today, zoofa and off home."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _mention(clock, responder, "Gus, say something backyard-style", Trigger.MENTION)
        system = _payload(calls[0])["messages"][0]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("vocab.request_header") in system
        assert all(word in system for word in ("sloshmire", "zoofa", "hlooss"))
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_vocab_explain_own_word_after_not_understood(db: Database) -> None:
    """ "Did not get it" after his "zoofa" (a followup from the gate): the meaning of his word is
    in the prompt, and an explanation using the word passes the filter although it just sounded."""
    cfg = _config()
    cfg.behaviour.vocab.probability = 0.0
    cfg.behaviour.vocab.max_meanings = 1
    await db.insert_bot_reply(
        tg_message_id=30,
        reply_to_tg_message_id=None,
        trigger="ambient",
        trigger_tg_message_id=None,
        text="Zoofa on the bench before the hlooss starts.",
        prompt_version=1,
        few_shot_version=1,
        delay_sec=0,
        created_at=DAY_NOW - 600,
    )
    await _insert_human(db, 40, "sloshmire out there", DAY_NOW - 300)
    llm, calls = _make_llm(
        cfg, db, lambda _req: _ok_response("Zoofa means sit down, and hlooss is a wind.")
    )
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        await _mention(clock, responder, "I didn't get that", Trigger.FOLLOWUP)
        system = _payload(calls[0])["messages"][0]["content"]  # type: ignore[index]
        # His words come first, both of them, even with max_meanings=1.
        assert "zoofa: a call to sit down" in system
        assert "hlooss: a biting wind" in system
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_vocab_not_offered_for_life_news(db: Database) -> None:
    """/life is the owner's text: no vocabulary words are mixed in."""
    cfg = _config()
    cfg.behaviour.vocab.probability = 1.0
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Got a new van."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(
        db,
        cfg,
        llm,
        bot,
        clock,
        prompt_store=FakePromptStore(prompt=PROMPT_TEMPLATE_WITH_VOCAB),
        vocab=_test_vocab(),
    )
    try:
        event_id = await db.insert_life_event(text="got a van in the sloshmire", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None
        task = await _drive(clock, responder.announce_life(event))
        assert task.result().sent is True

        system = _payload(calls[0])["messages"][0]["content"]  # type: ignore[index]
        assert DEMO_FRAG.text("vocab.offer_header") not in system
        assert DEMO_FRAG.text("vocab.meanings_header") not in system
    finally:
        await responder.shutdown()
        await llm.aclose()
