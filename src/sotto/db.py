"""Database: connection, migrations and repository methods on top of SQLite.

SQL is written by hand, without an ORM. The schema lives in schema.sql and is applied
in full when PRAGMA user_version == 0. Further migrations are numbered scripts
in MIGRATIONS, applied in ascending version order.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite

from sotto.gate_types import RecentActivity, StateChange

# Numbered migrations on top of the initial schema (user_version == 1). The key is the
# target version, the value is an SQL script applied through executescript. The script
# for version 2 is the same CREATE TABLE IF NOT EXISTS as in schema.sql (fresh
# DBs already get the table from schema.sql; IF NOT EXISTS here covers the case
# when someone reaches this point with the table already present).
MIGRATIONS: dict[int, str] = {
    2: """
    CREATE TABLE IF NOT EXISTS life_events (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        announced_at INTEGER,
        announced_tg_message_id INTEGER
    );
    """,
    3: """
    CREATE TABLE IF NOT EXISTS chat_memory (
        id INTEGER PRIMARY KEY,
        period_start INTEGER NOT NULL,
        period_end INTEGER NOT NULL,
        text TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chat_memory_period_end ON chat_memory (period_end);
    """,
    4: """
    CREATE TABLE IF NOT EXISTS self_facts (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        bot_reply_tg_message_id INTEGER,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_self_facts_created ON self_facts (created_at);
    """,
    5: """
    CREATE TABLE IF NOT EXISTS chat_jokes (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_used_at INTEGER,
        uses INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS people_threads (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        display_name TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        asked_at INTEGER,
        closed INTEGER NOT NULL DEFAULT 0
    );
    """,
    6: """
    CREATE TABLE IF NOT EXISTS vocab_dicts (
        name TEXT PRIMARY KEY,
        body TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        updated_at INTEGER NOT NULL
    );
    """,
}


@dataclass(frozen=True, slots=True)
class LifeEventRow:
    """A persona life event entered by the owner through /life."""

    id: int
    text: str
    created_at: int
    announced_at: int | None
    announced_tg_message_id: int | None


@dataclass(frozen=True, slots=True)
class ChatMemoryRow:
    """A recap of one chat period.

    ``period_start`` is inclusive, ``period_end`` exclusive; both boundaries fall
    on local midnight (persona.timezone), see chat_memory.py.
    """

    id: int
    period_start: int
    period_end: int
    text: str
    created_at: int


@dataclass(frozen=True, slots=True)
class SelfFactRow:
    """A fact about the persona's own doings extracted from its reply (day diary)."""

    id: int
    text: str
    bot_reply_tg_message_id: int | None
    created_at: int


@dataclass(frozen=True, slots=True)
class JokeRow:
    """A chat joke extracted by the weekly recap."""

    id: int
    text: str
    created_at: int
    last_used_at: int | None
    uses: int


@dataclass(frozen=True, slots=True)
class VocabDictRow:
    """A dictionary of words loaded by the owner.

    ``body`` is the source YAML as is: vocab.py parses it, the DB only stores it."""

    name: str
    body: str
    enabled: bool
    updated_at: int


@dataclass(frozen=True, slots=True)
class PeopleThreadRow:
    """An open thread of a person for the "so how did it go?" question."""

    id: int
    user_id: int
    display_name: str
    text: str
    created_at: int
    asked_at: int | None
    closed: bool


@dataclass(frozen=True, slots=True)
class MessageRow:
    id: int
    tg_message_id: int | None
    chat_id: int | None
    user_id: int | None
    display_name: str | None
    text: str | None
    reply_to_tg_message_id: int | None
    is_bot: bool
    created_at: int | None


@dataclass(frozen=True, slots=True)
class PurgeStats:
    messages: int
    night_queue: int
    pending_replies: int
    filter_log_texts: int
    state_keys: int
    # Long-term chat memory recaps live for their own period (behaviour.chat_memory.keep_days),
    # not message_retention_days, so they are purged by a separate call from run_retention.
    # The field has a default so that existing code that builds PurgeStats positionally
    # does not break.
    chat_memory_deleted: int = 0
    # Day diary facts also have their own period (diary.keep_days),
    # purged by a separate call from run_retention.
    self_facts_deleted: int = 0
    # Chat jokes and people threads: jokes by
    # chat_memory.keep_days, threads by callback.max_age_days.
    jokes_deleted: int = 0
    threads_deleted: int = 0


@dataclass(frozen=True, slots=True)
class PendingRow:
    id: int
    trigger_tg_message_id: int
    user_id: int
    trigger: str
    due_at: int
    created_at: int
    done_at: int | None


@dataclass(frozen=True, slots=True)
class NightRow:
    id: int
    tg_message_id: int
    user_id: int
    display_name: str
    text: str
    created_at: int
    answered_at: int | None


@dataclass(frozen=True, slots=True)
class BotReplyRow:
    id: int
    tg_message_id: int
    trigger: str
    trigger_tg_message_id: int | None
    text: str
    prompt_version: int
    few_shot_version: int
    delay_sec: int
    created_at: int


@dataclass(frozen=True, slots=True)
class VersionRow:
    version: int
    note: str
    active: bool
    created_at: int


@dataclass(frozen=True, slots=True)
class PlaceRow:
    """One row of the places table (see schema.sql).

    price_level comes from Google, is unreliable and sometimes missing,
    hence int | None. quiet and fact are set by hand or by the fill script
    (places_fill.py), not by Google; quiet is always a bool.
    """

    place_id: str
    name: str
    district: str
    category: str
    rating: float
    reviews: int
    price_level: int | None
    quiet: bool
    fact: str
    operational: bool
    refreshed_at: int


def _state_key_date_suffix(key: str) -> str | None:
    """The suffix after the last ':', or None if the key has no colon."""
    if ":" not in key:
        return None
    return key.rsplit(":", 1)[-1]


def _parse_state_key_date(suffix: str) -> date | None:
    """The date that a key suffix stands for: YYYY-MM-DD or YYYY-Www (Monday).

    Returns None if the suffix looks like neither of these forms:
    such keys (e.g. ":<user_id>") are not touched by retention.
    """
    try:
        return date.fromisoformat(suffix)
    except ValueError:
        pass
    match = re.fullmatch(r"(\d{4})-W(\d{2})", suffix)
    if match is None:
        return None
    year, week = int(match.group(1)), int(match.group(2))
    try:
        return date.fromisocalendar(year, week, 1)
    except ValueError:
        return None


def _is_expired_state_key(key: str, cutoff_date: date) -> bool:
    suffix = _state_key_date_suffix(key)
    if suffix is None:
        return False
    key_date = _parse_state_key_date(suffix)
    if key_date is None:
        return False
    return key_date < cutoff_date


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        # Serializes all writing methods (including commit): one aiosqlite connection
        # per process, concurrent writes would otherwise race at the SQLite level
        # and could interleave multi-statement operations such as purge_older_than.
        # Calling code does not think about serialization.
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self._path)
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await self.migrate()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected: call connect() first")
        return self._conn

    async def migrate(self) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            version = int(row[0]) if row is not None else 0

            if version == 0:
                schema_sql = (
                    importlib.resources.files("sotto").joinpath("schema.sql").read_text("utf-8")
                )
                await conn.executescript(schema_sql)
                cursor = await conn.execute("PRAGMA user_version")
                row = await cursor.fetchone()
                version = int(row[0]) if row is not None else 0

            for target_version in sorted(MIGRATIONS):
                if target_version <= version:
                    continue
                await conn.executescript(MIGRATIONS[target_version])
                await conn.execute(f"PRAGMA user_version = {target_version}")
                await conn.commit()
                version = target_version

    async def insert_message(
        self,
        *,
        tg_message_id: int,
        chat_id: int,
        user_id: int,
        display_name: str,
        text: str,
        reply_to_tg_message_id: int | None,
        is_bot: bool,
        created_at: int,
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO messages "
                "(tg_message_id, chat_id, user_id, display_name, text, "
                "reply_to_tg_message_id, is_bot, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tg_message_id,
                    chat_id,
                    user_id,
                    display_name,
                    text,
                    reply_to_tg_message_id,
                    int(is_bot),
                    created_at,
                ),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_message: INSERT did not return a rowid")
            return cursor.lastrowid

    async def recent_messages(self, chat_id: int, limit: int) -> list[MessageRow]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, tg_message_id, chat_id, user_id, display_name, text, "
            "reply_to_tg_message_id, is_bot, created_at "
            "FROM messages WHERE chat_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        messages = [
            MessageRow(
                id=row["id"],
                tg_message_id=row["tg_message_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                display_name=row["display_name"],
                text=row["text"],
                reply_to_tg_message_id=row["reply_to_tg_message_id"],
                is_bot=bool(row["is_bot"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
        messages.reverse()
        return messages

    async def get_state(self, key: str) -> str | None:
        conn = self._require_conn()
        cursor = await conn.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return None if row is None else str(row["value"])

    async def set_state(self, key: str, value: str) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await conn.commit()

    async def delete_state(self, key: str) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute("DELETE FROM state WHERE key = ?", (key,))
            await conn.commit()

    async def get_overrides(self) -> dict[str, str]:
        conn = self._require_conn()
        cursor = await conn.execute("SELECT key, value FROM config_overrides")
        rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def purge_older_than(self, cutoff: int, tz: str = "UTC") -> PurgeStats:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
            messages_deleted = cursor.rowcount

            cursor = await conn.execute(
                "DELETE FROM night_queue WHERE answered_at IS NOT NULL AND answered_at < ?",
                (cutoff,),
            )
            night_queue_deleted = cursor.rowcount

            cursor = await conn.execute(
                "DELETE FROM pending_replies WHERE done_at IS NOT NULL AND done_at < ?",
                (cutoff,),
            )
            pending_replies_deleted = cursor.rowcount

            cursor = await conn.execute(
                "UPDATE filter_log SET candidate_text = NULL "
                "WHERE created_at < ? AND candidate_text IS NOT NULL",
                (cutoff,),
            )
            filter_log_texts_cleared = cursor.rowcount

            # The bot's day is local (persona.timezone), not UTC: the :YYYY-MM-DD/:YYYY-Www suffix
            # is compared with the cutoff date in tz, otherwise the key of "today" could
            # be deleted several hours earlier or later than that day really begins for the bot.
            cutoff_date = datetime.fromtimestamp(cutoff, ZoneInfo(tz)).date()
            cursor = await conn.execute("SELECT key FROM state")
            state_rows = await cursor.fetchall()
            expired_keys = [
                str(row["key"])
                for row in state_rows
                if _is_expired_state_key(row["key"], cutoff_date)
            ]
            state_keys_deleted = 0
            if expired_keys:
                placeholders = ",".join("?" for _ in expired_keys)
                cursor = await conn.execute(
                    f"DELETE FROM state WHERE key IN ({placeholders})", expired_keys
                )
                state_keys_deleted = cursor.rowcount

            await conn.commit()

            return PurgeStats(
                messages=messages_deleted,
                night_queue=night_queue_deleted,
                pending_replies=pending_replies_deleted,
                filter_log_texts=filter_log_texts_cleared,
                state_keys=state_keys_deleted,
            )

    async def insert_filter_log(
        self,
        *,
        trigger_tg_message_id: int | None,
        candidate_text: str | None,
        verdict: str,
        stage: str,
        reason: str,
        shadow: bool,
        created_at: int,
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO filter_log "
                "(trigger_tg_message_id, candidate_text, verdict, stage, reason, shadow, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trigger_tg_message_id,
                    candidate_text,
                    verdict,
                    stage,
                    reason,
                    int(shadow),
                    created_at,
                ),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_filter_log: INSERT did not return a rowid")
            return cursor.lastrowid

    async def muted_user_ids(self) -> frozenset[int]:
        conn = self._require_conn()
        cursor = await conn.execute("SELECT user_id FROM muted_users")
        rows = await cursor.fetchall()
        return frozenset(int(row["user_id"]) for row in rows)

    async def display_names(self, user_ids: Sequence[int]) -> dict[int, str]:
        """The latest (by created_at) display_name of each user_id from messages.

        An empty list -> {}. One query for all user_ids, ordered by
        recency (created_at, id DESC): the first occurrence of each user_id in the
        result is its freshest name.
        """
        if not user_ids:
            return {}
        conn = self._require_conn()
        unique_ids = list(dict.fromkeys(user_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        cursor = await conn.execute(
            f"SELECT user_id, display_name FROM messages "
            f"WHERE user_id IN ({placeholders}) "
            f"ORDER BY created_at DESC, id DESC",
            unique_ids,
        )
        rows = await cursor.fetchall()
        result: dict[int, str] = {}
        for row in rows:
            user_id = int(row["user_id"])
            if user_id not in result:
                result[user_id] = str(row["display_name"])
        return result

    async def recent_activity(self, chat_id: int, since: int) -> list[RecentActivity]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT user_id, created_at FROM messages "
            "WHERE chat_id = ? AND is_bot = 0 AND created_at >= ? "
            "ORDER BY created_at, id",
            (chat_id, since),
        )
        rows = await cursor.fetchall()
        return [
            RecentActivity(user_id=row["user_id"], created_at=row["created_at"]) for row in rows
        ]

    async def enqueue_night(
        self,
        *,
        tg_message_id: int,
        user_id: int,
        display_name: str,
        text: str,
        created_at: int,
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO night_queue "
                "(tg_message_id, user_id, display_name, text, created_at, answered_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (tg_message_id, user_id, display_name, text, created_at),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("enqueue_night: INSERT did not return a rowid")
            return cursor.lastrowid

    async def apply_state_changes(self, changes: Iterable[StateChange]) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            applied = False
            for change in changes:
                applied = True
                if change.value is None:
                    await conn.execute("DELETE FROM state WHERE key = ?", (change.key,))
                else:
                    await conn.execute(
                        "INSERT INTO state (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (change.key, change.value),
                    )
            if applied:
                await conn.commit()

    async def filter_log_summary(self, since: int) -> list[tuple[str, int]]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT reason, COUNT(*) AS cnt FROM filter_log WHERE created_at >= ? "
            "GROUP BY reason ORDER BY cnt DESC, reason",
            (since,),
        )
        rows = await cursor.fetchall()
        return [(str(row["reason"]), int(row["cnt"])) for row in rows]

    async def recent_reactions(self, limit: int) -> list[str]:
        """The latest reactions set (filter_log react:sent), chronological."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT candidate_text FROM filter_log WHERE stage = 'react' AND reason = 'react:sent' "
            "AND candidate_text IS NOT NULL ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        emoji = [str(row["candidate_text"]) for row in rows]
        emoji.reverse()
        return emoji

    async def recent_bot_replies(self, limit: int) -> list[str]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT text FROM bot_replies ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        texts = [str(row["text"]) for row in rows]
        texts.reverse()
        return texts

    async def insert_bot_reply(
        self,
        *,
        tg_message_id: int,
        reply_to_tg_message_id: int | None,
        trigger: str,
        trigger_tg_message_id: int | None,
        text: str,
        prompt_version: int,
        few_shot_version: int,
        delay_sec: int,
        created_at: int,
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO bot_replies "
                "(tg_message_id, reply_to_tg_message_id, trigger, trigger_tg_message_id, text, "
                "prompt_version, few_shot_version, delay_sec, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tg_message_id,
                    reply_to_tg_message_id,
                    trigger,
                    trigger_tg_message_id,
                    text,
                    prompt_version,
                    few_shot_version,
                    delay_sec,
                    created_at,
                ),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_bot_reply: INSERT did not return a rowid")
            return cursor.lastrowid

    async def increment_state(self, key: str, by: int = 1) -> int:
        """Atomic read-modify-write of an int counter in state, entirely under write_lock.

        Does not reuse get_state/set_state: their own acquisition of write_lock
        (asyncio.Lock is not reentrant) would deadlock inside an already held lock.
        """
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("SELECT value FROM state WHERE key = ?", (key,))
            row = await cursor.fetchone()
            current = int(row["value"]) if row is not None else 0
            new_value = current + by
            await conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(new_value)),
            )
            await conn.commit()
            return new_value

    async def add_state_float(self, key: str, by: float) -> float:
        """Atomic read-modify-write of a float counter in state, stored as str(value)."""
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("SELECT value FROM state WHERE key = ?", (key,))
            row = await cursor.fetchone()
            current = float(row["value"]) if row is not None else 0.0
            new_value = current + by
            await conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(new_value)),
            )
            await conn.commit()
            return new_value

    async def insert_pending(
        self,
        *,
        trigger_tg_message_id: int,
        user_id: int,
        trigger: str,
        due_at: int,
        created_at: int,
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO pending_replies "
                "(trigger_tg_message_id, user_id, trigger, due_at, created_at, done_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (trigger_tg_message_id, user_id, trigger, due_at, created_at),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_pending: INSERT did not return a rowid")
            return cursor.lastrowid

    async def update_pending_due(self, pending_id: int, due_at: int) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "UPDATE pending_replies SET due_at = ? WHERE id = ?", (due_at, pending_id)
            )
            await conn.commit()

    async def mark_pending_done(self, pending_id: int, done_at: int) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "UPDATE pending_replies SET done_at = ? WHERE id = ?", (done_at, pending_id)
            )
            await conn.commit()

    async def load_pending(self) -> list[PendingRow]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, trigger_tg_message_id, user_id, trigger, due_at, created_at, done_at "
            "FROM pending_replies WHERE done_at IS NULL ORDER BY due_at, id"
        )
        rows = await cursor.fetchall()
        return [
            PendingRow(
                id=row["id"],
                trigger_tg_message_id=row["trigger_tg_message_id"],
                user_id=row["user_id"],
                trigger=row["trigger"],
                due_at=row["due_at"],
                created_at=row["created_at"],
                done_at=row["done_at"],
            )
            for row in rows
        ]

    async def night_unanswered(self) -> list[NightRow]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, tg_message_id, user_id, display_name, text, created_at, answered_at "
            "FROM night_queue WHERE answered_at IS NULL ORDER BY created_at, id"
        )
        rows = await cursor.fetchall()
        return [
            NightRow(
                id=row["id"],
                tg_message_id=row["tg_message_id"],
                user_id=row["user_id"],
                display_name=row["display_name"],
                text=row["text"],
                created_at=row["created_at"],
                answered_at=row["answered_at"],
            )
            for row in rows
        ]

    async def mark_night_answered(self, ids: Sequence[int], answered_at: int) -> None:
        if not ids:
            return
        conn = self._require_conn()
        async with self._write_lock:
            placeholders = ",".join("?" for _ in ids)
            await conn.execute(
                f"UPDATE night_queue SET answered_at = ? WHERE id IN ({placeholders})",
                (answered_at, *ids),
            )
            await conn.commit()

    async def messages_after(self, chat_id: int, tg_message_id: int) -> int:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM messages "
            "WHERE chat_id = ? AND tg_message_id > ? AND is_bot = 0",
            (chat_id, tg_message_id),
        )
        row = await cursor.fetchone()
        return int(row["cnt"]) if row is not None else 0

    async def last_message_at(self, chat_id: int) -> int | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT MAX(created_at) AS max_created_at FROM messages "
            "WHERE chat_id = ? AND is_bot = 0",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is None or row["max_created_at"] is None:
            return None
        return int(row["max_created_at"])

    async def count_messages_today(self, chat_id: int, day_start: int) -> int:
        """How many human messages there are in the chat since local midnight (the presence
        cap). ``day_start`` is computed by the caller:
        ``timeutil.day_start(now, persona.timezone)``."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM messages "
            "WHERE chat_id = ? AND is_bot = 0 AND created_at >= ?",
            (chat_id, day_start),
        )
        row = await cursor.fetchone()
        return int(row["cnt"]) if row is not None else 0

    async def count_bot_replies_today(self, day_start: int) -> int:
        """How many bot replies went to the chat since local midnight: the denominator of the same
        presence cap. All replies count, including /life and /say: they are
        the bot's presence in the chat, even though they are not limited by the cap."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM bot_replies WHERE created_at >= ?",
            (day_start,),
        )
        row = await cursor.fetchone()
        return int(row["cnt"]) if row is not None else 0

    async def last_bot_reply_at(self) -> int | None:
        """created_at of the last bot reply (bot_replies): the moment from which
        responder._maybe_checkin considers the topic dead and takes messages_since."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT MAX(created_at) AS max_created_at FROM bot_replies")
        row = await cursor.fetchone()
        if row is None or row["max_created_at"] is None:
            return None
        return int(row["max_created_at"])

    # -- control from Telegram --------------------------------------------

    async def set_override(self, key: str, value: str, changed_by: int, now: int) -> str | None:
        """UPSERT into config_overrides + a config_audit row, in one transaction.

        Returns the previous value (None if the key was not overridden).
        """
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("SELECT value FROM config_overrides WHERE key = ?", (key,))
            row = await cursor.fetchone()
            old_value = None if row is None else str(row["value"])
            await conn.execute(
                "INSERT INTO config_overrides (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, value, now),
            )
            await conn.execute(
                "INSERT INTO config_audit (key, old_value, new_value, changed_by, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, old_value, value, changed_by, now),
            )
            await conn.commit()
            return old_value

    async def delete_override(self, key: str, changed_by: int, now: int) -> str | None:
        """DELETE from config_overrides + a config_audit row (new_value=NULL), in one transaction.

        Returns the value before deletion (None if the key was not overridden either).
        """
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("SELECT value FROM config_overrides WHERE key = ?", (key,))
            row = await cursor.fetchone()
            old_value = None if row is None else str(row["value"])
            await conn.execute("DELETE FROM config_overrides WHERE key = ?", (key,))
            await conn.execute(
                "INSERT INTO config_audit (key, old_value, new_value, changed_by, created_at) "
                "VALUES (?, ?, NULL, ?, ?)",
                (key, old_value, changed_by, now),
            )
            await conn.commit()
            return old_value

    async def audit_stop(
        self, key: str, changed_by: int, now: int, new_value: str | None = None
    ) -> None:
        """A config_audit row without changing config_overrides, for /stop, /panic, /life, /say.

        new_value is the event/reply text for /life and /say; None for toggles.
        """
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO config_audit (key, old_value, new_value, changed_by, created_at) "
                "VALUES (?, NULL, ?, ?, ?)",
                (key, new_value, changed_by, now),
            )
            await conn.commit()

    async def add_mute(self, user_id: int, display_name: str, muted_by: int, now: int) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO muted_users (user_id, display_name, muted_by, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET display_name = excluded.display_name, "
                "muted_by = excluded.muted_by, created_at = excluded.created_at",
                (user_id, display_name, muted_by, now),
            )
            await conn.commit()

    async def remove_mute(self, user_id: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM muted_users WHERE user_id = ?", (user_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def last_bot_replies(self, n: int) -> list[BotReplyRow]:
        """The last n bot replies, newest first."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, tg_message_id, trigger, trigger_tg_message_id, text, prompt_version, "
            "few_shot_version, delay_sec, created_at FROM bot_replies "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (n,),
        )
        rows = await cursor.fetchall()
        return [
            BotReplyRow(
                id=row["id"],
                tg_message_id=row["tg_message_id"],
                trigger=row["trigger"],
                trigger_tg_message_id=row["trigger_tg_message_id"],
                text=row["text"],
                prompt_version=row["prompt_version"],
                few_shot_version=row["few_shot_version"],
                delay_sec=row["delay_sec"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def prompt_versions(self) -> list[VersionRow]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT version, note, active, created_at FROM prompt_versions ORDER BY version"
        )
        rows = await cursor.fetchall()
        return [
            VersionRow(
                version=row["version"],
                note=row["note"],
                active=bool(row["active"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def active_prompt(self) -> tuple[int, str] | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT version, body FROM prompt_versions WHERE active = 1 LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return (int(row["version"]), str(row["body"]))

    async def add_prompt_version(self, body: str, note: str, now: int) -> int:
        """New version = max(version)+1, becomes active, the others get active=0.

        One transaction: without it a concurrent call could read the same
        max(version) twice and leave two active versions.
        """
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("SELECT MAX(version) AS max_version FROM prompt_versions")
            row = await cursor.fetchone()
            next_version = (
                1 if row is None or row["max_version"] is None else int(row["max_version"]) + 1
            )
            await conn.execute("UPDATE prompt_versions SET active = 0")
            await conn.execute(
                "INSERT INTO prompt_versions (version, body, note, active, created_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (next_version, body, note, now),
            )
            await conn.commit()
            return next_version

    async def activate_prompt(self, version: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "SELECT 1 FROM prompt_versions WHERE version = ?", (version,)
            )
            exists = await cursor.fetchone()
            if exists is None:
                return False
            await conn.execute("UPDATE prompt_versions SET active = 0")
            await conn.execute(
                "UPDATE prompt_versions SET active = 1 WHERE version = ?", (version,)
            )
            await conn.commit()
            return True

    async def active_few_shot(self) -> tuple[int, str] | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT version, body_yaml FROM few_shot_versions WHERE active = 1 LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return (int(row["version"]), str(row["body_yaml"]))

    async def add_few_shot_version(self, body_yaml: str, note: str, now: int) -> int:
        """Like add_prompt_version, but for few_shot_versions: independent numbering."""
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("SELECT MAX(version) AS max_version FROM few_shot_versions")
            row = await cursor.fetchone()
            next_version = (
                1 if row is None or row["max_version"] is None else int(row["max_version"]) + 1
            )
            await conn.execute("UPDATE few_shot_versions SET active = 0")
            await conn.execute(
                "INSERT INTO few_shot_versions (version, body_yaml, note, active, created_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (next_version, body_yaml, note, now),
            )
            await conn.commit()
            return next_version

    async def activate_few_shot(self, version: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "SELECT 1 FROM few_shot_versions WHERE version = ?", (version,)
            )
            exists = await cursor.fetchone()
            if exists is None:
                return False
            await conn.execute("UPDATE few_shot_versions SET active = 0")
            await conn.execute(
                "UPDATE few_shot_versions SET active = 1 WHERE version = ?", (version,)
            )
            await conn.commit()
            return True

    async def message_by_tg_id(self, chat_id: int, tg_message_id: int) -> MessageRow | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, tg_message_id, chat_id, user_id, display_name, text, "
            "reply_to_tg_message_id, is_bot, created_at FROM messages "
            "WHERE chat_id = ? AND tg_message_id = ? ORDER BY id DESC LIMIT 1",
            (chat_id, tg_message_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return MessageRow(
            id=row["id"],
            tg_message_id=row["tg_message_id"],
            chat_id=row["chat_id"],
            user_id=row["user_id"],
            display_name=row["display_name"],
            text=row["text"],
            reply_to_tg_message_id=row["reply_to_tg_message_id"],
            is_bot=bool(row["is_bot"]),
            created_at=row["created_at"],
        )

    async def messages_since(
        self, chat_id: int, since: int, limit: int | None = None
    ) -> list[MessageRow]:
        """Human chat messages with created_at > since, chronological.

        Without ``limit`` (the default) it is used by
        ``responder._collect_addressed_items`` to restore the accumulated
        addresses to the bot after a process restart (when ``_pending_info``
        is empty), and returns all matching rows. With ``limit`` it is used by
        ``responder._maybe_checkin``: the last ``limit`` rows, but still in
        chronological order.
        """
        conn = self._require_conn()
        if limit is None:
            cursor = await conn.execute(
                "SELECT id, tg_message_id, chat_id, user_id, display_name, text, "
                "reply_to_tg_message_id, is_bot, created_at FROM messages "
                "WHERE chat_id = ? AND is_bot = 0 AND created_at > ? ORDER BY created_at, id",
                (chat_id, since),
            )
            rows = await cursor.fetchall()
        else:
            cursor = await conn.execute(
                "SELECT id, tg_message_id, chat_id, user_id, display_name, text, "
                "reply_to_tg_message_id, is_bot, created_at FROM messages "
                "WHERE chat_id = ? AND is_bot = 0 AND created_at > ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (chat_id, since, limit),
            )
            rows = list(await cursor.fetchall())
            rows.reverse()
        return [
            MessageRow(
                id=row["id"],
                tg_message_id=row["tg_message_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                display_name=row["display_name"],
                text=row["text"],
                reply_to_tg_message_id=row["reply_to_tg_message_id"],
                is_bot=bool(row["is_bot"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def bot_reply_by_tg_id(self, tg_message_id: int) -> BotReplyRow | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, tg_message_id, trigger, trigger_tg_message_id, text, prompt_version, "
            "few_shot_version, delay_sec, created_at FROM bot_replies WHERE tg_message_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (tg_message_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return BotReplyRow(
            id=row["id"],
            tg_message_id=row["tg_message_id"],
            trigger=row["trigger"],
            trigger_tg_message_id=row["trigger_tg_message_id"],
            text=row["text"],
            prompt_version=row["prompt_version"],
            few_shot_version=row["few_shot_version"],
            delay_sec=row["delay_sec"],
            created_at=row["created_at"],
        )

    async def prompt_version_bodies(self) -> list[str]:
        """Bodies of all stored system prompt versions (to seed PromptStore)."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT body FROM prompt_versions")
        rows = await cursor.fetchall()
        return [str(row["body"]) for row in rows]

    async def few_shot_version_bodies(self) -> list[str]:
        """Bodies of all stored few-shot versions (to seed PromptStore)."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT body_yaml FROM few_shot_versions")
        rows = await cursor.fetchall()
        return [str(row["body_yaml"]) for row in rows]

    # -- places ---------------------------------------------------------------

    async def upsert_place(self, row: PlaceRow) -> None:
        """UPSERT by place_id (filling the cache, places_fill.py)."""
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO places "
                "(place_id, name, district, category, rating, reviews, price_level, quiet, "
                "fact, operational, refreshed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(place_id) DO UPDATE SET "
                "name = excluded.name, district = excluded.district, "
                "category = excluded.category, rating = excluded.rating, "
                "reviews = excluded.reviews, price_level = excluded.price_level, "
                "quiet = excluded.quiet, fact = excluded.fact, "
                "operational = excluded.operational, refreshed_at = excluded.refreshed_at",
                (
                    row.place_id,
                    row.name,
                    row.district,
                    row.category,
                    row.rating,
                    row.reviews,
                    row.price_level,
                    int(row.quiet),
                    row.fact,
                    int(row.operational),
                    row.refreshed_at,
                ),
            )
            await conn.commit()

    async def places_all(self, *, operational_only: bool = True) -> list[PlaceRow]:
        """All places, by name. operational_only=True (the default) means only working ones."""
        conn = self._require_conn()
        query = (
            "SELECT place_id, name, district, category, rating, reviews, price_level, "
            "quiet, fact, operational, refreshed_at FROM places"
        )
        params: tuple[object, ...] = ()
        if operational_only:
            query += " WHERE operational = ?"
            params = (1,)
        query += " ORDER BY name"
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            PlaceRow(
                place_id=row["place_id"],
                name=row["name"],
                district=row["district"],
                category=row["category"],
                rating=row["rating"],
                reviews=row["reviews"],
                price_level=row["price_level"],
                quiet=bool(row["quiet"]),
                fact=row["fact"] or "",
                operational=bool(row["operational"]),
                refreshed_at=row["refreshed_at"],
            )
            for row in rows
        ]

    async def places_names(self) -> list[str]:
        """Names of working places only: the whitelist for filters.py."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT name FROM places WHERE operational = 1 ORDER BY name")
        rows = await cursor.fetchall()
        return [str(row["name"]) for row in rows]

    async def mark_places_not_seen(self, seen_ids: Sequence[str], now: int) -> int:
        """After a places_fill.py run with Google: mark as non-operational the places that were
        operational but were not met in this run (closed, filtered out
        by rating/reviews or simply did not hit any query): operational=0,
        refreshed_at=now. ``--manual-only`` records (place_id with the "manual:" prefix)
        are never found by Google, so they are left alone regardless of seen_ids.
        Returns the number of rows marked.
        """
        conn = self._require_conn()
        async with self._write_lock:
            if seen_ids:
                placeholders = ",".join("?" for _ in seen_ids)
                query = (
                    "UPDATE places SET operational = 0, refreshed_at = ? "
                    f"WHERE operational = 1 AND place_id NOT LIKE 'manual:%' "
                    f"AND place_id NOT IN ({placeholders})"
                )
                params: tuple[object, ...] = (now, *seen_ids)
            else:
                query = (
                    "UPDATE places SET operational = 0, refreshed_at = ? "
                    "WHERE operational = 1 AND place_id NOT LIKE 'manual:%'"
                )
                params = (now,)
            cursor = await conn.execute(query, params)
            await conn.commit()
            return cursor.rowcount

    # -- life events (/life) ------------------------------------------------

    async def insert_life_event(self, *, text: str, created_at: int) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO life_events (text, created_at, announced_at, "
                "announced_tg_message_id) VALUES (?, ?, NULL, NULL)",
                (text, created_at),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_life_event: INSERT did not return a rowid")
            return cursor.lastrowid

    async def life_events(self) -> list[LifeEventRow]:
        """All events, by created_at asc, id asc (oldest first, as in the prompt)."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, text, created_at, announced_at, announced_tg_message_id "
            "FROM life_events ORDER BY created_at, id"
        )
        rows = await cursor.fetchall()
        return [
            LifeEventRow(
                id=row["id"],
                text=row["text"],
                created_at=row["created_at"],
                announced_at=row["announced_at"],
                announced_tg_message_id=row["announced_tg_message_id"],
            )
            for row in rows
        ]

    async def life_event(self, event_id: int) -> LifeEventRow | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, text, created_at, announced_at, announced_tg_message_id "
            "FROM life_events WHERE id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return LifeEventRow(
            id=row["id"],
            text=row["text"],
            created_at=row["created_at"],
            announced_at=row["announced_at"],
            announced_tg_message_id=row["announced_tg_message_id"],
        )

    async def delete_life_event(self, event_id: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM life_events WHERE id = ?", (event_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def mark_life_event_announced(
        self, event_id: int, *, tg_message_id: int, now: int
    ) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "UPDATE life_events SET announced_at = ?, announced_tg_message_id = ? WHERE id = ?",
                (now, tg_message_id, event_id),
            )
            await conn.commit()

    # -- long-term chat memory ------------------------------------------------

    async def insert_chat_memory(
        self, *, period_start: int, period_end: int, text: str, created_at: int
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO chat_memory (period_start, period_end, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (period_start, period_end, text, created_at),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_chat_memory: INSERT did not return a rowid")
            return cursor.lastrowid

    async def chat_memories(self, limit: int) -> list[ChatMemoryRow]:
        """The last ``limit`` recaps, returned chronologically (oldest first).

        ``limit <= 0`` -> an empty list: ``behaviour.chat_memory.in_prompt = 0``
        means "do not mix memory into the prompt", and no extra query is needed here.
        """
        if limit <= 0:
            return []
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, period_start, period_end, text, created_at FROM chat_memory "
            "ORDER BY period_end DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows = list(await cursor.fetchall())
        rows.reverse()
        return [
            ChatMemoryRow(
                id=row["id"],
                period_start=row["period_start"],
                period_end=row["period_end"],
                text=row["text"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def chat_memory(self, memory_id: int) -> ChatMemoryRow | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, period_start, period_end, text, created_at FROM chat_memory WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ChatMemoryRow(
            id=row["id"],
            period_start=row["period_start"],
            period_end=row["period_end"],
            text=row["text"],
            created_at=row["created_at"],
        )

    async def chat_memory_count(self) -> int:
        """How many recaps there are in the DB in total, for the /status line; there is no need
        to pull them as a list (over a year they add up to fifty or more)."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT COUNT(*) AS n FROM chat_memory")
        row = await cursor.fetchone()
        return 0 if row is None else int(row["n"])

    async def delete_chat_memory(self, memory_id: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM chat_memory WHERE id = ?", (memory_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def last_chat_memory_end(self) -> int | None:
        """max(period_end): the next recap period starts from this moment."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT MAX(period_end) AS end FROM chat_memory")
        row = await cursor.fetchone()
        if row is None or row["end"] is None:
            return None
        return int(row["end"])

    async def purge_chat_memory_older_than(self, cutoff: int) -> int:
        """Delete recaps whose period ended before ``cutoff`` (by period_end)."""
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM chat_memory WHERE period_end < ?", (cutoff,))
            await conn.commit()
            return int(cursor.rowcount)

    # -- day diary ------------------------------------------------------------

    async def insert_self_fact(
        self, *, text: str, bot_reply_tg_message_id: int | None, created_at: int
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO self_facts (text, bot_reply_tg_message_id, created_at) "
                "VALUES (?, ?, ?)",
                (text, bot_reply_tg_message_id, created_at),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_self_fact: INSERT did not return a rowid")
            return cursor.lastrowid

    async def self_facts_since(self, since: int) -> list[SelfFactRow]:
        """Facts with ``created_at >= since``, in ascending time order."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, text, bot_reply_tg_message_id, created_at FROM self_facts "
            "WHERE created_at >= ? ORDER BY created_at ASC, id ASC",
            (since,),
        )
        return [
            SelfFactRow(
                id=row["id"],
                text=row["text"],
                bot_reply_tg_message_id=row["bot_reply_tg_message_id"],
                created_at=row["created_at"],
            )
            for row in await cursor.fetchall()
        ]

    async def delete_self_fact(self, fact_id: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM self_facts WHERE id = ?", (fact_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def purge_self_facts_older_than(self, cutoff: int) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM self_facts WHERE created_at < ?", (cutoff,))
            await conn.commit()
            return int(cursor.rowcount)

    # -- chat jokes and people threads --------------------------------------

    async def insert_joke(self, *, text: str, created_at: int) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO chat_jokes (text, created_at) VALUES (?, ?)", (text, created_at)
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_joke: INSERT did not return a rowid")
            return cursor.lastrowid

    async def jokes(self, limit: int) -> list[JokeRow]:
        """The last ``limit`` jokes, newest first."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, text, created_at, last_used_at, uses FROM chat_jokes "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [
            JokeRow(
                id=row["id"],
                text=row["text"],
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
                uses=row["uses"],
            )
            for row in await cursor.fetchall()
        ]

    async def delete_joke(self, joke_id: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM chat_jokes WHERE id = ?", (joke_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def mark_joke_used(self, joke_id: int, now: int) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "UPDATE chat_jokes SET last_used_at = ?, uses = uses + 1 WHERE id = ?",
                (now, joke_id),
            )
            await conn.commit()

    async def purge_jokes_older_than(self, cutoff: int) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM chat_jokes WHERE created_at < ?", (cutoff,))
            await conn.commit()
            return int(cursor.rowcount)

    async def insert_people_thread(
        self, *, user_id: int, display_name: str, text: str, created_at: int
    ) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "INSERT INTO people_threads (user_id, display_name, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, display_name, text, created_at),
            )
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("insert_people_thread: INSERT did not return a rowid")
            return cursor.lastrowid

    async def _people_threads(self, where: str) -> list[PeopleThreadRow]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, user_id, display_name, text, created_at, asked_at, closed "
            f"FROM people_threads {where} ORDER BY created_at ASC, id ASC"
        )
        return [
            PeopleThreadRow(
                id=row["id"],
                user_id=row["user_id"],
                display_name=row["display_name"],
                text=row["text"],
                created_at=row["created_at"],
                asked_at=row["asked_at"],
                closed=bool(row["closed"]),
            )
            for row in await cursor.fetchall()
        ]

    async def open_people_threads(self) -> list[PeopleThreadRow]:
        """Open threads, oldest first."""
        return await self._people_threads("WHERE closed = 0")

    async def people_threads_all(self) -> list[PeopleThreadRow]:
        return await self._people_threads("")

    async def close_people_thread(self, thread_id: int, now: int) -> None:
        """Close a thread: ``asked_at = now``, ``closed = 1``."""
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "UPDATE people_threads SET asked_at = ?, closed = 1 WHERE id = ?",
                (now, thread_id),
            )
            await conn.commit()

    async def delete_people_thread(self, thread_id: int) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM people_threads WHERE id = ?", (thread_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def purge_people_threads_older_than(self, cutoff: int) -> int:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "DELETE FROM people_threads WHERE created_at < ?", (cutoff,)
            )
            await conn.commit()
            return int(cursor.rowcount)

    async def last_message_at_by_user(self, chat_id: int, user_id: int) -> int | None:
        """When this person last wrote in the chat (None: never)."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT MAX(created_at) AS max_created_at FROM messages "
            "WHERE chat_id = ? AND user_id = ? AND is_bot = 0",
            (chat_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None or row["max_created_at"] is None:
            return None
        return int(row["max_created_at"])

    async def user_id_by_display_name(
        self, chat_id: int, name: str, since: int, until: int
    ) -> int | None:
        """user_id by an EXACT display_name match among human messages with
        ``since <= created_at < until``. If two or more people wrote under this name,
        None: a person's thread must not go to their namesake."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT DISTINCT user_id FROM messages WHERE chat_id = ? AND is_bot = 0 "
            "AND display_name = ? AND created_at >= ? AND created_at < ? "
            "AND user_id IS NOT NULL LIMIT 2",
            (chat_id, name, since, until),
        )
        rows = list(await cursor.fetchall())
        if len(rows) != 1:
            return None
        return int(rows[0]["user_id"])

    async def first_message_at(self, chat_id: int) -> int | None:
        """created_at of the earliest chat message: the starting point of the catch-up
        run of long-term memory (chat_memory.run_due) when there are no recaps yet."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT MIN(created_at) AS first FROM messages WHERE chat_id = ?", (chat_id,)
        )
        row = await cursor.fetchone()
        if row is None or row["first"] is None:
            return None
        return int(row["first"])

    async def messages_between(
        self,
        chat_id: int,
        start: int,
        end: int,
        *,
        exclude_user_ids: Iterable[int] = (),
    ) -> list[MessageRow]:
        """Human chat messages within ``[start, end)``, chronological.

        ``exclude_user_ids`` are muted participants: their messages never get
        into the recap (a deliberate privacy decision).
        """
        conn = self._require_conn()
        excluded = sorted({int(user_id) for user_id in exclude_user_ids})
        sql = (
            "SELECT id, tg_message_id, chat_id, user_id, display_name, text, "
            "reply_to_tg_message_id, is_bot, created_at FROM messages "
            "WHERE chat_id = ? AND is_bot = 0 AND created_at >= ? AND created_at < ?"
        )
        params: list[int] = [chat_id, start, end]
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            sql += f" AND (user_id IS NULL OR user_id NOT IN ({placeholders}))"
            params.extend(excluded)
        sql += " ORDER BY created_at, id"
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            MessageRow(
                id=row["id"],
                tg_message_id=row["tg_message_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                display_name=row["display_name"],
                text=row["text"],
                reply_to_tg_message_id=row["reply_to_tg_message_id"],
                is_bot=bool(row["is_bot"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def bot_replies_between(self, start: int, end: int) -> list[BotReplyRow]:
        """Bot replies within ``[start, end)``, chronological: the second half of
        the material for a period recap."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT id, tg_message_id, trigger, trigger_tg_message_id, text, prompt_version, "
            "few_shot_version, delay_sec, created_at FROM bot_replies "
            "WHERE created_at >= ? AND created_at < ? ORDER BY created_at, id",
            (start, end),
        )
        rows = await cursor.fetchall()
        return [
            BotReplyRow(
                id=row["id"],
                tg_message_id=row["tg_message_id"],
                trigger=row["trigger"],
                trigger_tg_message_id=row["trigger_tg_message_id"],
                text=row["text"],
                prompt_version=row["prompt_version"],
                few_shot_version=row["few_shot_version"],
                delay_sec=row["delay_sec"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- dictionary words -------------------------------------------------------

    async def upsert_vocab_dict(self, name: str, body: str, now: int) -> None:
        """A new dictionary is enabled; an existing one gets a new body, ``enabled`` is kept."""
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO vocab_dicts (name, body, enabled, updated_at) VALUES (?, ?, 1, ?) "
                "ON CONFLICT(name) DO UPDATE SET body = excluded.body, "
                "updated_at = excluded.updated_at",
                (name, body, now),
            )
            await conn.commit()

    async def vocab_dicts(self) -> list[VocabDictRow]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT name, body, enabled, updated_at FROM vocab_dicts ORDER BY name"
        )
        return [
            VocabDictRow(
                name=row["name"],
                body=row["body"],
                enabled=bool(row["enabled"]),
                updated_at=row["updated_at"],
            )
            for row in await cursor.fetchall()
        ]

    async def set_vocab_dict_enabled(self, name: str, enabled: bool) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute(
                "UPDATE vocab_dicts SET enabled = ? WHERE name = ?", (int(enabled), name)
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def delete_vocab_dict(self, name: str) -> bool:
        conn = self._require_conn()
        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM vocab_dicts WHERE name = ?", (name,))
            await conn.commit()
            return cursor.rowcount > 0
