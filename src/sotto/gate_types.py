"""Gate types. Pure data, no I/O and no aiogram.

The gate decides whether a message is a reason to reply. It writes nothing:
state changes are returned in Decision and applied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Trigger(StrEnum):
    MENTION = "mention"  # @username in the text
    REPLY = "reply"  # reply to a bot message
    NAME = "name"  # a word from persona.name_triggers
    FOLLOWUP = "followup"  # no address, but a cheap check judged it addressed to the bot
    AMBIENT = "ambient"  # no address, joining a live conversation


class Verdict(StrEnum):
    PASS = "pass"  # consider it: debounce and generation follow
    QUEUE_NIGHT = "queue_night"  # addressed at night: into night_queue, answered in the morning
    DROP = "drop"  # silence, the cause is in reason


@dataclass(frozen=True, slots=True)
class GateMessage:
    chat_id: int
    tg_message_id: int
    user_id: int
    is_bot: bool
    text: str  # already normalize_text
    reply_to_bot: bool  # reply to a bot message
    created_at: int  # unix seconds


@dataclass(frozen=True, slots=True)
class RecentActivity:
    """One non-bot message within the live_talk window. Only who and when matter."""

    user_id: int
    created_at: int


@dataclass(frozen=True, slots=True)
class GateState:
    """Snapshot of the state at decision time. Built from the state and messages tables."""

    panic: bool
    stop_until: int | None
    topic_cooldown_until: int | None
    muted_user_ids: frozenset[int]
    mention_count_today: int
    last_mention_reply_at: int | None  # per chat
    last_mention_reply_at_user: int | None  # per author of the current message
    ambient_count_today: int  # ambient + spontaneous
    last_ambient_at: int | None
    recent: tuple[RecentActivity, ...]  # non-bot messages in live_talk.window_min, with the current
    # Hot window after /life and /say: defaults come last so that existing positional
    # calls (replay.py, tests) keep working; the window is never opened there
    # (hot_until is always None).
    hot_until: int | None = None
    hot_ambient_count: int = 0
    # Presence cap: how many human messages and bot replies there already were in the
    # local day. Zero defaults come last for the same reason as for the hot window:
    # existing positional calls (replay.py, tests) keep working, and a zero human
    # counter leaves the bot only free_replies.
    human_messages_today: int = 0
    bot_replies_today: int = 0


@dataclass(frozen=True, slots=True)
class StateChange:
    key: str
    value: str | None  # None = delete the key


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    trigger: Trigger | None  # on PASS and QUEUE_NIGHT
    reason: str  # "gate:is_bot", "gate:night_queued", ... ; on PASS "pass:<trigger>"
    state_changes: tuple[StateChange, ...] = ()


class PatternsLike(Protocol):
    """Interface of patterns.Patterns, so the gate can be tested with a fake."""

    def topic_stop(self, text: str) -> str | None: ...
    def injection(self, text: str) -> str | None: ...
    def logistics(self, text: str) -> str | None: ...
    def name_trigger(self, text: str) -> str | None: ...
    def mentions_bot(self, text: str) -> bool: ...
