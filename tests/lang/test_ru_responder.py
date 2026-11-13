"""Responder behaviour that depends on the Russian language: the venue filter (Latin words inside
Russian text) only exists for a language with a foreign script. Uses the synthetic Russian pack."""

from __future__ import annotations

from sotto.config_models import Config
from sotto.db import Database, PlaceRow
from sotto.gate_types import Trigger
from sotto.llm import LLMClient
from sotto.patterns import Patterns
from sotto.responder import Responder
from tests.packs import RU, RU_FRAG, ru_config
from tests.responder.helpers import (
    BOT_USERNAME,
    DAY_NOW,
    FakeBot,
    FakeClock,
    _drive,
    _make_llm,
    _make_responder,
    _ok_response,
)


def _place_row(place_id: str, name: str) -> PlaceRow:
    return PlaceRow(
        place_id=place_id,
        name=name,
        district="Заречье",
        category="craft",
        rating=4.6,
        reviews=100,
        price_level=2,
        quiet=False,
        fact="тихо",
        operational=True,
        refreshed_at=DAY_NOW,
    )


def _russian_responder(
    db: Database, llm: LLMClient, bot: FakeBot, clock: FakeClock
) -> tuple[Config, Responder]:
    """Responder on the synthetic Russian pack: the venue filter (Latin words inside Russian
    text) only exists for languages that have a foreign script."""
    cfg = ru_config()
    cfg.llm.main_model = "test/model"
    cfg.filters.shadow = False
    responder = _make_responder(db, cfg, llm, bot, clock)
    patterns = Patterns(cfg.filters, cfg.persona.name_triggers, BOT_USERNAME, RU)
    responder.patterns_getter = lambda: patterns
    responder.frag = RU_FRAG
    return cfg, responder


async def test_filter_context_places_names_passes_real_place_through_venue_regex(
    db: Database,
) -> None:
    """FilterContext.places_names is always filled from db.places_names(): a real venue from
    the table is not cut by regex:venue, even when the trigger is not about places."""
    cfg = ru_config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Был вчера в Zielony Kot."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    _cfg, responder = _russian_responder(db, llm, bot, clock)

    # "Zielony Kot" is deliberately in neither filters.known_places nor
    # filters.places_whitelist by default: this checks places_names specifically.
    await db.upsert_place(_place_row("p1", "Zielony Kot"))

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=13, user_id=None, situation="", delay_sec=0
            ),
        )

        assert len(bot.sent) == 1
        assert bot.sent[0][1] == "Был вчера в Zielony Kot."
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_filter_context_places_names_still_cuts_fabricated_venue(
    db: Database,
) -> None:
    """A made-up venue (not from places, not from known_places/whitelist) is cut by
    regex:venue even though places_names is now a non-empty list."""
    cfg = ru_config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Был вчера в Fikcyjny Bar."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    _cfg, responder = _russian_responder(db, llm, bot, clock)

    await db.upsert_place(_place_row("p1", "Zielony Kot"))

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=14, user_id=None, situation="", delay_sec=0
            ),
        )

        assert bot.sent == []
        logs = await db.filter_log_summary(DAY_NOW - 10)
        assert any(reason == "regex:venue" for reason, _count in logs)
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_announce_life_latin_from_note_passes_venue_filter(db: Database) -> None:
    """Латинское название из заметки владельца («Hyundai») не режется regex:venue:
    текст события уходит как trigger_text и попадает в белый список фильтра."""
    cfg = ru_config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Взял Hyundai, старую продал."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    _cfg, responder = _russian_responder(db, llm, bot, clock)
    try:
        event_id = await db.insert_life_event(
            text="продал тележку, взял Hyundai 2015 года", created_at=DAY_NOW
        )
        event = await db.life_event(event_id)
        assert event is not None

        task = await _drive(clock, responder.announce_life(event))
        outcome = task.result()

        assert outcome.sent is True
        assert outcome.reason == "send:life"
        assert len(bot.sent) == 1
    finally:
        await responder.shutdown()
        await llm.aclose()
