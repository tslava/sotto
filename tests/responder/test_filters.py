"""Responder tests: filters."""

from __future__ import annotations

import pytest

from sotto import filters as filters_module
from sotto.db import Database
from sotto.filters import FilterContext, FilterVerdict
from sotto.gate_types import Trigger
from sotto.patterns import Patterns
from tests.responder.helpers import (
    DAY_NOW,
    PROMPT_TEMPLATE,
    FakeBot,
    FakeClock,
    _config,
    _drive,
    _insert_message,
    _make_llm,
    _make_responder,
    _ok_response,
)

# shadow=True sends and logs shadow=1; shadow=False does not send.


async def test_shadow_mode_sends_but_logs_cut(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    cfg.filters.shadow = True
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(text: str, ctx: object, judge: object = None) -> FilterVerdict:
        return FilterVerdict(ok=False, reason="regex:length", reasons=("regex:length",))

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=400, user_id=5, situation="", delay_sec=5
            ),
        )
        assert len(bot.sent) == 1

        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT shadow, verdict, reason FROM filter_log WHERE reason = ?", ("regex:length",)
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["shadow"] == 1
        assert row["verdict"] == "cut"
    finally:
        await responder.shutdown()
        await llm.aclose()

    # shadow=False -> not sent (early return before typing, no need to drive it)
    cfg2 = _config()
    cfg2.filters.shadow = False
    llm2, _calls2 = _make_llm(cfg2, db, lambda _req: _ok_response("Candidate two."))
    bot2 = FakeBot()
    clock2 = FakeClock(DAY_NOW)
    responder2 = _make_responder(db, cfg2, llm2, bot2, clock2)
    try:
        await responder2._respond(
            trigger=Trigger.MENTION, trigger_msg_id=401, user_id=5, situation="", delay_sec=5
        )
        assert bot2.sent == []

        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT shadow FROM filter_log WHERE trigger_tg_message_id = ?", (401,)
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["shadow"] == 0
    finally:
        await responder2.shutdown()
        await llm2.aclose()


# FilterContext is fully assembled: trigger_text from the pending, muted_names via
# db.display_names(muted_user_ids()), bot_names/system_prompt from the Responder.


async def test_filter_context_wired_with_trigger_text_and_muted_names(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens, old friend."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    # A muted author: their latest name in messages must end up in muted_names.
    await _insert_message(
        db,
        tg_message_id=500,
        user_id=7,
        display_name="Gene",
        text="gus, hello",
        created_at=DAY_NOW,
    )
    conn = db._conn
    assert conn is not None
    await conn.execute(
        "INSERT INTO muted_users (user_id, display_name, muted_by, created_at) VALUES (?, ?, ?, ?)",
        (7, "Gene", 1, DAY_NOW),
    )
    await conn.commit()

    real_check_output = filters_module.check_output
    captured: dict[str, FilterContext] = {}

    async def spy_check_output(
        text: str, ctx: FilterContext, judge: object = None
    ) -> FilterVerdict:
        captured["ctx"] = ctx
        return await real_check_output(text, ctx, judge)  # type: ignore[arg-type]

    monkeypatch.setattr(filters_module, "check_output", spy_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION,
                trigger_msg_id=500,
                user_id=7,
                situation="",
                delay_sec=5,
                trigger_text="gus, hello",
            ),
        )

        ctx = captured["ctx"]
        assert ctx.trigger_text == "gus, hello"
        assert ctx.system_prompt == PROMPT_TEMPLATE
        assert ctx.muted_names == ["Gene"]
        # patterns_getter() -> Patterns goes into FilterContext, not just cfg.filters:
        # one regex compilation for the whole check_output, no rebuilding.
        assert isinstance(ctx.patterns, Patterns)
        assert ctx.bot_names[0] == cfg.persona.name
        assert ctx.bot_names[1] == cfg.persona.display_name
        assert "Gene" in ctx.participant_names
        assert len(ctx.recent_replies) <= 50
    finally:
        await responder.shutdown()
        await llm.aclose()


# Ambient/pending without trigger text -> trigger_text="" in FilterContext.


async def test_ambient_reply_has_empty_trigger_text(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    real_check_output = filters_module.check_output
    captured: dict[str, FilterContext] = {}

    async def spy_check_output(
        text: str, ctx: FilterContext, judge: object = None
    ) -> FilterVerdict:
        captured["ctx"] = ctx
        return await real_check_output(text, ctx, judge)  # type: ignore[arg-type]

    monkeypatch.setattr(filters_module, "check_output", spy_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.AMBIENT, trigger_msg_id=600, user_id=5, situation="", delay_sec=0
            ),
        )
        assert captured["ctx"].trigger_text == ""
    finally:
        await responder.shutdown()
        await llm.aclose()


# A not-ok verdict with several reasons -> one filter_log row per reason.


async def test_multiple_reasons_produce_one_filter_log_row_each(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(
        text: str, ctx: FilterContext, judge: object = None
    ) -> FilterVerdict:
        return FilterVerdict(
            ok=False,
            reason="regex:length",
            reasons=("regex:length", "style:exclaim"),
        )

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=700, user_id=5, situation="", delay_sec=5
            ),
        )

        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT stage, reason FROM filter_log "
            "WHERE trigger_tg_message_id = ? AND verdict = 'cut' ORDER BY id",
            (700,),
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        assert rows == [
            {"stage": "regex", "reason": "regex:length"},
            {"stage": "style", "reason": "style:exclaim"},
        ]
    finally:
        await responder.shutdown()
        await llm.aclose()


# The judge is passed to check_output.


async def test_judge_is_passed_through_to_check_output(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Happens."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)

    class _DummyJudge:
        async def check(self, *, candidate: str, trigger_text: str, now: int) -> list[str]:
            return []

    dummy_judge = _DummyJudge()
    responder = _make_responder(db, cfg, llm, bot, clock, judge=dummy_judge)  # type: ignore[arg-type]

    captured: dict[str, object] = {}

    async def fake_check_output(
        text: str, ctx: FilterContext, judge: object = None
    ) -> FilterVerdict:
        captured["judge"] = judge
        return FilterVerdict(ok=True, reason="pass", reasons=())

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=800, user_id=5, situation="", delay_sec=5
            ),
        )
        assert captured["judge"] is dummy_judge
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_enforce_stages_cut_dedup_even_in_shadow_mode(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    cfg.filters.shadow = True
    assert cfg.filters.enforce_stages == ["dedup", "style"]
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(text: str, ctx: object, judge: object = None) -> FilterVerdict:
        return FilterVerdict(
            ok=False,
            reason="regex:length",
            reasons=("regex:length", "dedup:motif"),
        )

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=810, user_id=5, situation="", delay_sec=5
            ),
        )

        assert bot.sent == []

        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT stage, reason, shadow FROM filter_log "
            "WHERE trigger_tg_message_id = ? AND verdict = 'cut' ORDER BY id",
            (810,),
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        # Both reasons are logged, but shadow=0: the reply was really cut.
        assert rows == [
            {"stage": "regex", "reason": "regex:length", "shadow": 0},
            {"stage": "dedup", "reason": "dedup:motif", "shadow": 0},
        ]
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_enforce_stages_cut_reason_is_the_enforced_one_in_shadow(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In shadow only an enforced stage cuts, and it becomes the reason in SendOutcome."""
    cfg = _config()
    cfg.filters.shadow = True
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(text: str, ctx: object, judge: object = None) -> FilterVerdict:
        return FilterVerdict(
            ok=False,
            reason="regex:length",
            reasons=("regex:length", "style:story_quota"),
        )

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)

    try:
        event_id = await db.insert_life_event(text="sold the handcart", created_at=DAY_NOW)
        event = await db.life_event(event_id)
        assert event is not None

        task = await _drive(clock, responder.announce_life(event))
        outcome = task.result()

        assert outcome.sent is False
        assert outcome.reason == "style:story_quota"
    finally:
        await responder.shutdown()
        await llm.aclose()


async def test_non_enforced_stage_still_passes_in_shadow(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stages outside enforce_stages are still only logged in shadow."""
    cfg = _config()
    cfg.filters.shadow = True
    cfg.filters.enforce_stages = ["dedup"]
    llm, _calls = _make_llm(cfg, db, lambda _req: _ok_response("Candidate."))
    bot = FakeBot()
    clock = FakeClock(DAY_NOW)
    responder = _make_responder(db, cfg, llm, bot, clock)

    async def fake_check_output(text: str, ctx: object, judge: object = None) -> FilterVerdict:
        return FilterVerdict(ok=False, reason="style:exclaim", reasons=("style:exclaim",))

    monkeypatch.setattr(filters_module, "check_output", fake_check_output)

    try:
        await _drive(
            clock,
            responder._respond(
                trigger=Trigger.MENTION, trigger_msg_id=811, user_id=5, situation="", delay_sec=5
            ),
        )

        assert len(bot.sent) == 1

        conn = db._conn
        assert conn is not None
        cursor = await conn.execute(
            "SELECT shadow FROM filter_log WHERE trigger_tg_message_id = ?", (811,)
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["shadow"] == 1
    finally:
        await responder.shutdown()
        await llm.aclose()
