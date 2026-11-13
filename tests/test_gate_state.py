from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sotto.db import Database
from sotto.gate_state import load_gate_state
from sotto.gate_types import GateMessage
from tests.demo import demo_config

CONFIG = demo_config()


def _msg(
    *,
    chat_id: int = 1,
    tg_message_id: int = 1,
    user_id: int = 10,
    is_bot: bool = False,
    text: str = "hello",
    reply_to_bot: bool = False,
    created_at: int = 1000,
) -> GateMessage:
    return GateMessage(
        chat_id=chat_id,
        tg_message_id=tg_message_id,
        user_id=user_id,
        is_bot=is_bot,
        text=text,
        reply_to_bot=reply_to_bot,
        created_at=created_at,
    )


async def test_load_gate_state_empty_db_gives_defaults(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        state = await load_gate_state(db, CONFIG, _msg(), now=1_768_003_200)

        assert state.panic is False
        assert state.stop_until is None
        assert state.topic_cooldown_until is None
        assert state.muted_user_ids == frozenset()
        assert state.mention_count_today == 0
        assert state.last_mention_reply_at is None
        assert state.last_mention_reply_at_user is None
        assert state.ambient_count_today == 0
        assert state.last_ambient_at is None
        assert state.recent == ()
        assert state.hot_until is None
        assert state.hot_ambient_count == 0
        assert state.human_messages_today == 0
        assert state.bot_replies_today == 0
    finally:
        await db.close()


async def test_load_gate_state_reads_filled_keys(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        now = 1_768_003_200  # 2026-01-10 00:00 UTC
        msg = _msg(user_id=42, created_at=now)

        from sotto.timeutil import day_key

        await db.set_state("panic", "1")
        await db.set_state("stop_until", "12345")
        await db.set_state("topic_cooldown_until", "6789")
        await db.set_state("last_mention_reply_at", "111")
        await db.set_state("last_mention_reply_at:42", "222")
        await db.set_state("last_ambient_at", "333")
        await db.set_state(day_key("mention_count", now, CONFIG.persona.timezone), "5")
        await db.set_state(day_key("ambient_count", now, CONFIG.persona.timezone), "7")
        await db.set_state("hot_until", str(now + 1800))
        await db.set_state("hot_ambient_count", "2")

        conn = db._conn
        assert conn is not None
        await conn.execute(
            "INSERT INTO muted_users (user_id, display_name, muted_by, created_at) "
            "VALUES (?, ?, ?, ?)",
            (99, "Muted", 1, now),
        )
        await conn.commit()

        state = await load_gate_state(db, CONFIG, msg, now)

        assert state.panic is True
        assert state.stop_until == 12345
        assert state.topic_cooldown_until == 6789
        assert state.muted_user_ids == frozenset({99})
        assert state.mention_count_today == 5
        assert state.last_mention_reply_at == 111
        assert state.last_mention_reply_at_user == 222
        assert state.ambient_count_today == 7
        assert state.last_ambient_at == 333
        assert state.hot_until == now + 1800
        assert state.hot_ambient_count == 2
    finally:
        await db.close()


async def test_load_gate_state_panic_requires_exact_value_one(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        await db.set_state("panic", "0")
        state = await load_gate_state(db, CONFIG, _msg(), now=1000)
        assert state.panic is False
    finally:
        await db.close()


async def test_load_gate_state_garbage_numeric_value_falls_back_to_default(
    tmp_path: Path, caplog: object
) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        await db.set_state("stop_until", "not-a-number")

        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="sotto.gate_state"):  # type: ignore[attr-defined]
            state = await load_gate_state(db, CONFIG, _msg(), now=1000)

        assert state.stop_until is None
        assert "stop_until" in caplog.text  # type: ignore[attr-defined]
        assert "WARNING" in caplog.text  # type: ignore[attr-defined]
    finally:
        await db.close()


async def test_load_gate_state_garbage_count_falls_back_to_zero(
    tmp_path: Path, caplog: object
) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        now = 1_768_003_200
        from sotto.timeutil import day_key

        key = day_key("mention_count", now, CONFIG.persona.timezone)
        await db.set_state(key, "garbage")

        with caplog.at_level(logging.WARNING, logger="sotto.gate_state"):  # type: ignore[attr-defined]
            state = await load_gate_state(db, CONFIG, _msg(), now)

        assert state.mention_count_today == 0
        assert "WARNING" in caplog.text  # type: ignore[attr-defined]
    finally:
        await db.close()


async def test_load_gate_state_recent_activity_within_window(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        now = 100_000
        window_sec = CONFIG.behaviour.live_talk.window_min * 60

        await db.insert_message(
            tg_message_id=1,
            chat_id=1,
            user_id=10,
            display_name="A",
            text="hi",
            reply_to_tg_message_id=None,
            is_bot=False,
            created_at=now - window_sec + 1,  # inside the window
        )
        await db.insert_message(
            tg_message_id=2,
            chat_id=1,
            user_id=11,
            display_name="B",
            text="old",
            reply_to_tg_message_id=None,
            is_bot=False,
            created_at=now - window_sec - 100,  # outside the window
        )

        state = await load_gate_state(db, CONFIG, _msg(chat_id=1, created_at=now), now)

        assert [a.user_id for a in state.recent] == [10]
    finally:
        await db.close()


async def test_load_gate_state_mention_count_key_uses_london_local_day(tmp_path: Path) -> None:
    """now = 00:30 London (summer, UTC+1) = 23:30 UTC the previous day.

    The mention_count key must use the London date (the next UTC day),
    not the UTC date.
    """
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        # 2026-07-15 23:30 UTC == 2026-07-16 00:30 Europe/London (summer, UTC+1)
        now = int(datetime(2026, 7, 15, 23, 30, tzinfo=UTC).timestamp())

        await db.set_state("mention_count:2026-07-16", "9")  # London date
        await db.set_state("mention_count:2026-07-15", "1")  # UTC date, must not be used

        state = await load_gate_state(db, CONFIG, _msg(created_at=now), now)

        assert state.mention_count_today == 9
    finally:
        await db.close()


async def test_load_gate_state_counts_presence_by_local_day(tmp_path: Path) -> None:
    """Presence cap: the counters are taken from the tables since local midnight in
    persona.timezone, yesterday is not counted."""
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        tz = ZoneInfo(CONFIG.persona.timezone)
        now = int(datetime(2026, 9, 10, 15, 0, tzinfo=tz).timestamp())
        midnight = int(datetime(2026, 9, 10, 0, 0, tzinfo=tz).timestamp())

        for tg_id, created_at in ((1, midnight - 60), (2, midnight), (3, now - 10)):
            await db.insert_message(
                tg_message_id=tg_id,
                chat_id=1,
                user_id=10,
                display_name="Gene",
                text="some text",
                reply_to_tg_message_id=None,
                is_bot=False,
                created_at=created_at,
            )
        for tg_id, created_at in ((100, midnight - 30), (101, now - 5)):
            await db.insert_bot_reply(
                tg_message_id=tg_id,
                reply_to_tg_message_id=None,
                trigger="ambient",
                trigger_tg_message_id=None,
                text="a reply",
                prompt_version=1,
                few_shot_version=1,
                delay_sec=0,
                created_at=created_at,
            )

        state = await load_gate_state(db, CONFIG, _msg(created_at=now), now=now)

        assert state.human_messages_today == 2
        assert state.bot_replies_today == 1
    finally:
        await db.close()
