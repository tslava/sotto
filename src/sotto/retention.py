"""Retention: a one-off purge run and a periodic background task.

`retention_loop` does not depend on config.py: the config comes through `cfg_getter`,
typed with a structural Protocol, so only an object with
`.behaviour.message_retention_days` and `.persona.timezone` is required.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from sotto.db import Database, PurgeStats

logger = logging.getLogger(__name__)


class _RetentionChatMemory(Protocol):
    @property
    def keep_days(self) -> int: ...


class _RetentionDiary(Protocol):
    @property
    def keep_days(self) -> int: ...


class _RetentionCallback(Protocol):
    @property
    def max_age_days(self) -> int: ...


class _RetentionBehaviour(Protocol):
    # Properties, not attributes: mypy checks Protocol attributes invariantly
    # and would not accept BehaviourConfig in place of _RetentionBehaviour.
    @property
    def message_retention_days(self) -> int: ...
    @property
    def chat_memory(self) -> _RetentionChatMemory: ...
    @property
    def diary(self) -> _RetentionDiary: ...
    @property
    def callback(self) -> _RetentionCallback: ...


class _RetentionPersona(Protocol):
    @property
    def timezone(self) -> str: ...


class _RetentionConfig(Protocol):
    @property
    def behaviour(self) -> _RetentionBehaviour: ...
    @property
    def persona(self) -> _RetentionPersona: ...


async def run_retention(
    db: Database,
    retention_days: int,
    tz: str,
    now: int,
    chat_memory_keep_days: int | None = None,
    diary_keep_days: int | None = None,
    callback_max_age_days: int | None = None,
) -> PurgeStats:
    """Delete everything older than `retention_days` relative to `now` and log the totals.

    `tz` is the bot timezone (persona.timezone): the days that dated state keys are
    tied to are computed locally, not in UTC.

    `chat_memory_keep_days` is a separate, much longer period for the long-term chat
    memory: summaries outlive the messages themselves. None leaves the memory
    untouched (old callers and tests that pass three arguments).

    `diary_keep_days` is the retention of day-diary facts; None leaves them untouched.

    Chat jokes live as long as the summaries (`chat_memory_keep_days`); people's
    threads live `callback_max_age_days`; None leaves them untouched.
    """
    cutoff = now - retention_days * 86400
    stats = await db.purge_older_than(cutoff, tz)
    if chat_memory_keep_days is not None:
        chat_memory_deleted = await db.purge_chat_memory_older_than(
            now - chat_memory_keep_days * 86400
        )
        jokes_deleted = await db.purge_jokes_older_than(now - chat_memory_keep_days * 86400)
        stats = replace(stats, chat_memory_deleted=chat_memory_deleted, jokes_deleted=jokes_deleted)
    if diary_keep_days is not None:
        self_facts_deleted = await db.purge_self_facts_older_than(now - diary_keep_days * 86400)
        stats = replace(stats, self_facts_deleted=self_facts_deleted)
    if callback_max_age_days is not None:
        threads_deleted = await db.purge_people_threads_older_than(
            now - callback_max_age_days * 86400
        )
        stats = replace(stats, threads_deleted=threads_deleted)
    total = (
        stats.messages
        + stats.night_queue
        + stats.pending_replies
        + stats.filter_log_texts
        + stats.state_keys
        + stats.chat_memory_deleted
        + stats.self_facts_deleted
        + stats.jokes_deleted
        + stats.threads_deleted
    )
    if total:
        logger.info(
            "retention purge: messages=%d night_queue=%d pending_replies=%d "
            "filter_log_texts=%d state_keys=%d chat_memory=%d self_facts=%d jokes=%d threads=%d",
            stats.messages,
            stats.night_queue,
            stats.pending_replies,
            stats.filter_log_texts,
            stats.state_keys,
            stats.chat_memory_deleted,
            stats.self_facts_deleted,
            stats.jokes_deleted,
            stats.threads_deleted,
        )
    return stats


async def retention_loop(
    db: Database,
    cfg_getter: Callable[[], _RetentionConfig],
    interval_sec: int = 3600,
) -> None:
    """Endless retention loop.

    Stopped by cancelling the task; CancelledError is propagated.
    """
    while True:
        try:
            cfg = cfg_getter()
            retention_days = cfg.behaviour.message_retention_days
            tz = cfg.persona.timezone
            keep_days = cfg.behaviour.chat_memory.keep_days
            diary_keep_days = cfg.behaviour.diary.keep_days
            await run_retention(
                db,
                retention_days,
                tz,
                int(time.time()),
                keep_days,
                diary_keep_days,
                cfg.behaviour.callback.max_age_days,
            )
        except asyncio.CancelledError:
            logger.info("retention_loop: cancelled, stopping")
            raise
        except Exception:
            logger.exception("retention_loop: iteration failed")
        await asyncio.sleep(interval_sec)
