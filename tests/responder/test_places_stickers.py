"""Responder tests: places_stickers."""

from __future__ import annotations

from sotto.db import Database, PlaceRow
from sotto.gate_types import Trigger
from sotto.stickers import Sticker
from sotto.timeutil import day_key
from tests.demo import DEMO_FRAG
from tests.responder.helpers import (
    DAY_NOW,
    PLACES_NONE,
    FakeBot,
    FakeClock,
    FakeStickerChooser,
    _config,
    _drive,
    _make_llm,
    _make_responder,
    _ok_response,
    _payload,
)

PLACES_MENU_HEADER = DEMO_FRAG.text("render.places_menu_header")
STICKER_TAG = DEMO_FRAG.text("placeholders.sticker_tag")


def _place_row(
    place_id: str,
    name: str,
    *,
    district: str = "Kettle Hill",
    category: str = "craft",
    rating: float = 4.6,
    reviews: int = 100,
    quiet: bool = False,
    fact: str = "quiet",
    operational: bool = True,
) -> PlaceRow:
    return PlaceRow(
        place_id=place_id,
        name=name,
        district=district,
        category=category,
        rating=rating,
        reviews=reviews,
        price_level=2,
        quiet=quiet,
        fact=fact,
        operational=operational,
        refreshed_at=DAY_NOW,
    )


async def test_places_request_in_mention_sends_full_menu_and_marks_trigger_places(
    db: Database,
) -> None:
    """Owner decision after a live test: the regex does not cover live speech, so on a direct
    address the whole venue cache goes into the prompt (not the 1-2 places select_places would
    filter), and both "was a place asked about" and the choice are left to the model.
    patterns.places_request still matches "where should we go" and switches the recorded
    trigger to "places", but only for statistics."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("There is one place, a quiet one."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    await db.upsert_place(_place_row("p1", "The Lantern Room", quiet=True, fact="quiet"))
    await db.upsert_place(_place_row("p2", "Barrow's Hatch", quiet=False, fact="noisy"))

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION,
                trigger_msg_id=10,
                user_id=5,
                situation="",
                delay_sec=5,
                trigger_text="Gus, recommend somewhere quiet to go",
            ),
        )

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert PLACES_MENU_HEADER in user_content
        assert "The Lantern Room" in user_content
        assert "Barrow's Hatch" in user_content

        assert len(bot.sent) == 1
        last = await db.last_bot_replies(1)
        assert last[0].trigger == "places"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_places_block_absent_for_ambient_even_with_places_request_text(
    db: Database,
) -> None:
    """Ambient/spontaneous/morning never get the places block, even if trigger_text (usually
    empty for ambient, but the contract is explicit: "never") looks like a request for places."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    await db.upsert_place(_place_row("p1", "The Lantern Room", quiet=True))

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=11,
                user_id=5,
                situation="",
                delay_sec=0,
                trigger_text="Gus, recommend somewhere quiet to go",
            ),
        )

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert "The Lantern Room" not in user_content
        assert PLACES_NONE in user_content

        last = await db.last_bot_replies(1)
        assert last[0].trigger == Trigger.AMBIENT.value
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_mention_without_places_words_still_gets_full_menu_and_keeps_trigger(
    db: Database,
) -> None:
    """Owner live test: phrases like "any decent beer round here" are not caught by the regex,
    so the venue list goes into the prompt on ANY direct address, even when trigger_text is
    not about a place at all: "was it asked about" is the model's call. The trigger in
    bot_replies stays the original (mention): the regex did not fire on this text, so the
    "places" statistics trigger is untouched."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("Hello to you too."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    await db.upsert_place(_place_row("p1", "The Lantern Room", quiet=True))

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION,
                trigger_msg_id=12,
                user_id=5,
                situation="",
                delay_sec=5,
                trigger_text="Gus, how are you doing?",
            ),
        )

        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert PLACES_MENU_HEADER in user_content
        assert "The Lantern Room" in user_content

        last = await db.last_bot_replies(1)
        assert last[0].trigger == Trigger.MENTION.value
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_places_request_with_empty_places_table_uses_places_none_and_marks_trigger(
    db: Database,
) -> None:
    """Code review: an address with places_request but an empty places table still puts
    PLACES_NONE (not an empty string) into the user message, and the trigger in bot_replies
    stays "places": the request was about places even if there is nothing to say."""
    cfg = _config()
    llm, calls = _make_llm(cfg, db, lambda _req: _ok_response("No idea where to go, really."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION,
                trigger_msg_id=16,
                user_id=5,
                situation="",
                delay_sec=5,
                trigger_text="Gus, recommend somewhere to go",
            ),
        )

        assert len(calls) == 1
        payload = _payload(calls[0])
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert PLACES_NONE in user_content

        last = await db.last_bot_replies(1)
        assert last[0].trigger == "places"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_sticker_chosen_sends_sticker_not_text(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Hello to you too."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    chooser = FakeStickerChooser(
        sticker=Sticker(
            id=3, file_id="FILE3", emoji="😂", text="Well, look at you", when="", enabled=True
        )
    )
    responder = _make_responder(db, cfg, llm, bot, clock, sticker_chooser=chooser)
    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=50, user_id=None, situation="", delay_sec=0
            ),
        )

        assert bot.sent == []  # send_message was NOT called
        assert len(bot.sent_stickers) == 1
        assert bot.sent_stickers[0][1] == "FILE3"
        assert len(chooser.calls) == 1

        replies = await db.recent_bot_replies(5)
        assert replies == [f"{STICKER_TAG.format(id=3)} Well, look at you"]

        assert await db.get_state("replies_since_sticker") == "0"
        sticker_key = day_key("sticker_count", clock.now(), cfg.persona.timezone)
        assert await db.get_state(sticker_key) == "1"

        # The address/ambient budget counters are the shared tail, the same as for text.
        ambient_key = day_key("ambient_count", clock.now(), cfg.persona.timezone)
        assert await db.get_state(ambient_key) == "1"

        logs = await db.filter_log_summary(clock.now() - 10)
        assert any(reason == "send:sticker" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_sticker_chooser_returns_none_sends_text_as_before(db: Database) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    chooser = FakeStickerChooser(sticker=None)
    responder = _make_responder(db, cfg, llm, bot, clock, sticker_chooser=chooser)
    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=51, user_id=None, situation="", delay_sec=0
            ),
        )

        assert len(chooser.calls) == 1
        assert bot.sent_stickers == []
        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Happens."

        assert await db.get_state("replies_since_sticker") == "1"
        sticker_key = day_key("sticker_count", clock.now(), cfg.persona.timezone)
        assert await db.get_state(sticker_key) is None
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_sticker_chooser_not_called_when_min_replies_between_not_elapsed(
    db: Database,
) -> None:
    cfg = _config()
    cfg.behaviour.stickers.min_replies_between = 4
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    chooser = FakeStickerChooser(
        sticker=Sticker(id=1, file_id="F1", text="Must not be chosen", enabled=True)
    )
    responder = _make_responder(db, cfg, llm, bot, clock, sticker_chooser=chooser)
    await db.set_state("replies_since_sticker", "1")
    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=52, user_id=None, situation="", delay_sec=0
            ),
        )

        assert chooser.calls == []  # the chooser is not called at all
        assert bot.sent_stickers == []
        assert len(bot.sent) == 1
        assert await db.get_state("replies_since_sticker") == "2"
    finally:
        await responder.shutdown()
        await llm.aclose()
