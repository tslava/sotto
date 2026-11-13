"""Responder: orchestrates generation and sending (PLAN.md, stage 3).

The only module that calls ``bot.send_message``/``bot.send_sticker``/
``bot.send_chat_action``. Everything else (gate, DB, LLM, prompt, filter) is already
built; this module glues it together:

- debounce of incoming PASS decisions (one ``asyncio.Task`` per chat, the timer resets on
  every new PASS inside the ``debounce_sec`` window);
- delayed replies to direct addresses through ``pending_replies`` (the in-memory timer is
  only an alarm clock, the data survives a restart);
- the text is regenerated only after the delay, on a fresh context;
- the morning job and the unprompted ("spontaneous") job.

The bot is injected as an object implementing ``_BotLike``, a narrow protocol with
``send_message``/``send_sticker``/``send_chat_action``, so tests fake it with a plain class
instead of pulling in aiogram. For the same reason ``clock`` and ``sleep`` are injected
(real time and ``asyncio.sleep`` by default): tests swap in controllable fakes and never
wait real seconds.

Special case: after a process restart ``restore_pending`` restores delayed replies from
``PendingRow`` alone (no text and no display_name, since the DB does not store them for
pending_replies). If such a restored wait lands in the night window, ``night_queue``
gets a placeholder row with ``text=""`` and a generic participant name, not the real
message. For pendings created during the current process the text and name are in memory
(``_pending_info``) and are used as usual.

Several addresses to the bot can pile up while one pending is being debounced (several
people wrote while the bot was silent). ``_pending_info`` keeps a list of
``(display_name, text)`` pairs; each merge appends an item instead of replacing the
previous one. When generating a reply to an address (``_generate_and_send``) the whole
``situation`` is built from that list through ``prompt.situation_addressed``, so the model
knows exactly whom to answer instead of answering the most prominent message in the
context window. If the list is empty after a restart, ``_fire_pending_inner`` rebuilds it
from ``messages`` (``_collect_addressed_items``): every human message with
``created_at >= pending.created_at`` flagged as an address (reply to a bot message,
mention or name trigger) plus the original trigger itself, at most the last 5.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from sotto import filters
from sotto.chat_memory import render_chat_memory
from sotto.config_models import Config, HotWindowConfig
from sotto.db import Database, JokeRow, LifeEventRow, MessageRow, PendingRow
from sotto.delays import debounce_seconds, fast_delay, pick_delay
from sotto.diary import DiaryExtractor, render_diary
from sotto.filters import FilterContext
from sotto.followup import FollowupChecker
from sotto.fragments import Fragments
from sotto.gate import PRESENCE_EXEMPT
from sotto.gate_state import load_gate_state
from sotto.gate_types import GateMessage, Trigger
from sotto.jokes import joke_used, render_jokes
from sotto.judge import Judge
from sotto.llm import LLMClient, LLMError
from sotto.motifs import render_avoid, story_count, used_motifs
from sotto.patterns import Patterns
from sotto.persona import Persona
from sotto.places import render_places_menu
from sotto.postprocess import soften
from sotto.prompt import (
    build_messages,
    parse_reply,
    render_context,
    render_life,
    situation_addressed,
    situation_callback,
    situation_checkin,
    situation_followup,
    situation_life,
)
from sotto.stickers import Sticker, StickerChooser, recent_sticker_ids, sticker_allowed
from sotto.timeutil import (
    day_key,
    day_start,
    in_window,
    local_date,
    parse_hhmm,
    seconds_until,
    week_key,
)
from sotto.vocab import (
    Vocab,
    VocabEntry,
    explain_requested,
    offer_allowed,
    pick_offer,
    render_vocab,
    vocab_over_quota,
    vocab_requested,
)
from sotto.weather import (
    WeatherClient,
    WeatherPlaceExtractor,
    render_weather,
    render_weather_place,
)

logger = logging.getLogger(__name__)

# Trigger priority when the debounce buffer collapses addresses:
# reply > mention > name > followup > ambient.
_TRIGGER_PRIORITY: dict[Trigger, int] = {
    Trigger.REPLY: 4,
    Trigger.MENTION: 3,
    Trigger.NAME: 2,
    Trigger.FOLLOWUP: 1,
    Trigger.AMBIENT: 0,
}

# followup: the cheap check judged the message to be addressed to the bot. From here on it
# behaves like a regular address: pending, cooldown shift, mention_count, hot-window delay
# cap, the usual reply_to rule, the whole places cache in the prompt, stickers as for an
# address.
_ADDRESS_TRIGGER_VALUES = (
    Trigger.MENTION.value,
    Trigger.REPLY.value,
    Trigger.NAME.value,
    Trigger.FOLLOWUP.value,
)

# Provider failures after which an address is postponed once instead of being dropped
# silently, and the postponement range in seconds.
_LLM_RETRY_REASONS = frozenset({"llm:http", "llm:timeout"})
_LLM_RETRY_DELAY_SEC = (120, 300)

# Cap on how many addresses situation_addressed is built from (and how many
# _collect_addressed_items restores after a restart): the last 5 at most.
_ADDRESSED_ITEMS_LIMIT = 5

# On any direct address (mention/reply/name) the whole places cache (render_places_menu)
# goes into the prompt, and the model decides from the instruction in the block itself
# whether a place was asked about: regex detection of place requests does not cover live
# speech. ambient/spontaneous/morning never get the list (no unsolicited recommendations).
# patterns.places_request is kept for statistics only: when it matches the address
# trigger_text, the recorded trigger (bot_replies.trigger / filter_log.reason) is replaced
# with "places". The address budget (mention_count/last_mention_reply_at) is still counted
# for the original trigger_value, and is_address does not change.
_PLACES_TRIGGER = "places"

_TYPING_ACTION = "typing"
_TYPING_STEP_SEC = 4.0
_CHARS_PER_SEC = 15.0
_SECONDS_PER_HOUR = 3600

# Ambient and spontaneous share one daily budget (ambient_count/last_ambient_at,
# GateState.ambient_count_today), so _respond_inner rechecks them the same way.
_AMBIENT_LIKE_TRIGGER_VALUES = (Trigger.AMBIENT.value, "spontaneous")

# Pause after a failed iteration of a background loop (morning_job/spontaneous_job), so a
# repeating error does not turn into a busy loop (same as retention_loop).
_ERROR_RETRY_SEC = 60.0

# shutdown(): if a generation is running (_respond_lock is held), how long to wait for it to
# finish normally before cancelling timers; a graceful SIGTERM must not tear a model call.
_SHUTDOWN_GENERATION_WAIT_SEC = 10.0

# FilterContext.recent_replies holds the last 50 bot replies, regardless of
# cfg.behaviour.recent_replies_memory, which only limits the {recent_replies} block in the
# generation prompt itself.
_FILTER_RECENT_REPLIES_LIMIT = 50

# llm.main_model may be empty (LLM enabled live by the OpenRouter key alone, the model set
# later via /set). Warn about that at most once per 10 minutes, otherwise every PASS in a
# chat without a model would flood the log.
_NO_MODEL_WARN_INTERVAL_SEC = 600

# Offer a sticker only for a direct address or ambient: morning and spontaneous speak into
# the void, where a sticker would look out of place, so they are excluded.
_STICKER_ELIGIBLE_TRIGGER_VALUES = (*_ADDRESS_TRIGGER_VALUES, Trigger.AMBIENT.value)

# "replies_since_sticker" was never set (no sticker was ever sent in this chat): treat it as
# plenty of replies since the last sticker, so min_replies_between cannot block it.
_NO_STICKER_YET_REPLIES_SINCE = 10**9

# Pause before sending a sticker: short and fixed rather than based on text length (a
# sticker has no reply text length, unlike typing before text).
_STICKER_TYPING_RANGE_SEC = (2.0, 3.0)

# "Come back and check" is a plain string, not a Trigger enum member (like
# "spontaneous"/"morning"/"say"/"life"): it is a background job, not a reaction to a gate
# PASS, so it needs no slot in the debounce priority (_TRIGGER_PRIORITY).
_CHECKIN_TRIGGER = "checkin"

# State keys of checkin_job/_maybe_checkin. checkin_due is the moment of the next check;
# checkin_due_hot_until is the "hot_until" value (or last_bot_reply_at if the hot window is
# off) that this checkin_due was last computed from. A new or changed hot_until (a new bot
# reply reopened the window) invalidates the computed due and forces a recompute from
# scratch; otherwise a stale due computed from the previous window close would fire right
# after the bot started talking again.
_CHECKIN_DUE_KEY = "checkin_due"
_CHECKIN_DUE_REF_KEY = "checkin_due_hot_until"
_CHECKIN_LAST_AT_KEY = "checkin_last_at"

# "How did it go?" callback is also a plain string, like "checkin": a background job, not an
# address (NOT in _ADDRESS_TRIGGER_VALUES), with no places, stickers, reply or budget
# counters. Its own state key records when it last asked.
_CALLBACK_TRIGGER = "callback"
_CALLBACK_LAST_AT_KEY = "callback_last_at"

# Vocabulary words: the daily counter, the "last used" value for /status
# ("<unix>|<word>"), and how many of the bot's own latest replies to scan so it can explain
# a word it just said itself.
_LIFE_TRIGGER = "life"
_VOCAB_COUNT_KEY = "vocab_count"
_VOCAB_LAST_KEY = "vocab_last"
_VOCAB_OWN_REPLIES = 2

# How many of the persona's latest replies go to the cheap "is this for me?" check when
# vetting a message picked by checkin.
_CHECKIN_FOLLOWUP_RECENT_REPLIES = 3

# Triggers the bot picks itself, with no address to it: the presence cap and the gap between
# replies cut exactly these, silently; there is nothing to postpone and no reason to wait.
# Addresses are limited by _recheck (the cap) and the pending postponement (the gap);
# /life and /say, the owner's buttons, are not limited at all.
_UNSOLICITED_TRIGGER_VALUES = (
    Trigger.AMBIENT.value,
    "spontaneous",
    "morning",
    _CHECKIN_TRIGGER,
)


def _unique_participant_names(context_rows: list[MessageRow]) -> list[str]:
    """Unique non-bot display_names from context_rows, in order of first appearance."""
    names: list[str] = []
    seen: set[str] = set()
    for row in context_rows:
        if row.is_bot or not row.display_name or row.display_name in seen:
            continue
        seen.add(row.display_name)
        names.append(row.display_name)
    return names


@dataclass(frozen=True)
class SendOutcome:
    """Outcome of one send attempt for /life and /say.

    reason is the same string recorded in filter_log: "send:life", "send:say",
    "llm:silent", a filter cut reason, "blocked:panic"/"blocked:stop".
    """

    sent: bool
    text: str
    reason: str
    tg_message_id: int | None = None


class _SentMessageLike(Protocol):
    @property
    def message_id(self) -> int: ...


class _BotLike(Protocol):
    """Narrow protocol instead of ``aiogram.Bot``: tests fake it without aiogram."""

    async def send_message(
        self, chat_id: int, text: str, *, reply_to_message_id: int | None = None
    ) -> _SentMessageLike: ...

    async def send_sticker(
        self, chat_id: int, sticker: str, *, reply_to_message_id: int | None = None
    ) -> _SentMessageLike: ...

    async def send_chat_action(self, chat_id: int, action: str) -> object: ...


class _PromptStoreLike(Protocol):
    """Narrow protocol instead of ``stores.PromptStore``.

    Read anew on every ``_respond`` rather than cached in the constructor, so a hot edit of
    the prompt or few-shot (``/rollback``, ``/ex add``) is picked up by the very next reply
    without restarting the process.
    """

    def system_prompt(self) -> str: ...
    def prompt_version(self) -> int: ...
    def few_shot_text(self) -> str: ...
    def few_shot_version(self) -> int: ...


def _trigger_value(trigger: Trigger | str) -> str:
    return trigger.value if isinstance(trigger, Trigger) else str(trigger)


def _window_hours(window: tuple[str, str]) -> float:
    start = parse_hhmm(window[0])
    end = parse_hhmm(window[1])
    start_sec = start.hour * _SECONDS_PER_HOUR + start.minute * 60
    end_sec = end.hour * _SECONDS_PER_HOUR + end.minute * 60
    if end_sec <= start_sec:
        end_sec += 24 * _SECONDS_PER_HOUR
    return (end_sec - start_sec) / _SECONDS_PER_HOUR


def _cut_reason(cfg: Config, reasons: Sequence[str]) -> str | None:
    """Whether the output filter cuts, and for what reason.

    Shadow mode stays the gentle mode for new layers, but the stages in
    filters.enforce_stages (dedup and style by default) always cut: repeats and a broken
    style are what got the bot stopped by the chat, and shadow let them through. In shadow
    mode the reason is the enforced stage; without shadow it is the first reason. None
    means the reply goes out."""
    enforce_stages = set(cfg.filters.enforce_stages)
    enforced = [reason for reason in reasons if reason.split(":", 1)[0] in enforce_stages]
    if not cfg.filters.shadow:
        return reasons[0] if reasons else None
    return enforced[0] if enforced else None


@dataclass(frozen=True)
class PreviewOutcome:
    """Outcome of ``Responder.preview`` for ``/ask``: what the persona would answer, with
    nothing sent. ``reason`` is as in ``SendOutcome``: "send:mention", "llm:silent",
    "llm:<error>", a filter cut reason. ``reasons`` lists every filter reason that fired,
    including ones shadow mode let through; ``fixes`` lists post-processing fixes."""

    sent: bool
    text: str
    reason: str
    reasons: tuple[str, ...] = ()
    fixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _VocabPlan:
    """The ``{vocab}`` slot and the data for ``dedup:vocab``."""

    block: str = ""
    over_quota: bool = False
    allowed: frozenset[VocabEntry] = frozenset()


@dataclass(frozen=True)
class _BuiltPrompt:
    """Outcome of ``Responder._build_prompt``: the messages for the model and what the
    filter, stickers, jokes and vocabulary need after the reply."""

    messages: list[dict[str, str]]
    context_rows: list[MessageRow]
    recent_replies: list[str]
    filter_recent_replies: list[str]
    patterns: Patterns
    system_prompt: str
    is_address: bool
    record_trigger: str
    joke_rows: list[JokeRow]
    vocab: Vocab | None
    vocab_over: bool
    vocab_allowed: frozenset[VocabEntry]


class Responder:
    def __init__(
        self,
        *,
        bot: _BotLike,
        db: Database,
        cfg_getter: Callable[[], Config],
        llm: LLMClient,
        judge: Judge | None = None,
        sticker_chooser: StickerChooser | None = None,
        followup: FollowupChecker | None = None,
        weather: WeatherClient | None = None,
        weather_places: WeatherPlaceExtractor | None = None,
        diary: DiaryExtractor | None = None,
        vocab_getter: Callable[[], Vocab] | None = None,
        patterns_getter: Callable[[], Patterns],
        prompt_store: _PromptStoreLike,
        rng: random.Random,
        chat_id: int,
        bot_user_id: int,
        clock: Callable[[], int] = lambda: int(time.time()),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        persona: Persona | None = None,
        frag: Fragments,
    ) -> None:
        self.bot = bot
        self.db = db
        self.cfg_getter = cfg_getter
        self.llm = llm
        self.judge = judge
        self.sticker_chooser = sticker_chooser
        # Cheap "is this for me?" check. Not needed by the gate here (bot.py calls it there)
        # but by checkin: the line the main model picked is rechecked by a second model.
        # None -> no check.
        self.followup = followup
        # Weather is background life for the persona and goes into the system prompt for any
        # trigger. None -> no weather block at all. weather_places extracts a place name from
        # the question with a cheap model; None -> weather is always the home one only.
        self.weather = weather
        self.weather_places = weather_places
        # Day diary: after a successful text send a cheap model extracts a fact about the
        # persona's doings from the reply. None -> no facts are extracted (the {diary} slot is
        # still filled with what is already recorded).
        self.diary = diary
        self._diary_tasks: set[asyncio.Task[None]] = set()
        # Words from the /words dictionaries: a getter, not an object, since the owner loads
        # dictionaries live. None -> the {vocab} slot is empty.
        self.vocab_getter = vocab_getter
        self.patterns_getter = patterns_getter
        self.prompt_store = prompt_store
        # Persona pack: situation texts, block texts and placeholders come from its fragments.
        self.persona = persona
        self.frag = frag
        self.rng = rng
        self.chat_id = chat_id
        self.bot_user_id = bot_user_id
        self._clock = clock
        self.sleep = sleep

        self._debounce_buffer: list[tuple[Trigger, GateMessage, str]] = []
        self._debounce_task: asyncio.Task[None] | None = None
        self._pending_tasks: dict[int, asyncio.Task[None]] = {}
        # pending_id -> [(display_name, text), ...]: every address accumulated during this
        # pending's life (a merge appends, never replaces). Only for pendings created in this
        # process; after a restart the list is empty and _fire_pending_inner restores it from
        # the DB (_collect_addressed_items).
        self._pending_info: dict[int, list[tuple[str, str]]] = {}
        # pending_ids already postponed after a provider failure: only one postponement, kept
        # in memory, so it is not repeated after a restart.
        self._llm_retried: set[int] = set()
        # pendings being generated right now: a new address does not merge into them, or its
        # timer would fire a second time after the reply and the bot would answer twice. A new
        # address in the meantime gets its own pending.
        self._in_flight: set[int] = set()
        # Serializes generation + send + counters of one Responder: without it two PASS
        # decisions waiting for the LLM in parallel could both pass the same budget check.
        self._respond_lock = asyncio.Lock()
        # Timestamp (in clock()/now scale) of the last WARNING about an empty llm.main_model,
        # throttled by _NO_MODEL_WARN_INTERVAL_SEC so every PASS does not spam the log.
        self._last_no_model_warn_at: int | None = None

    # ------------------------------------------------------------------ #
    # Receiving a PASS from the gate: debounce, merge, creating a pending.
    # ------------------------------------------------------------------ #

    async def on_gate_pass(self, msg: GateMessage, trigger: Trigger, display_name: str) -> None:
        self._debounce_buffer.append((trigger, msg, display_name))

        if self._debounce_task is not None:
            if not self._debounce_task.done():
                self._debounce_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._debounce_task
            else:
                # The task is already done (normally; the cancellation in _debounce_fire was
                # swallowed): fetch its exception just in case, so asyncio does not complain
                # about "Task exception was never retrieved".
                with contextlib.suppress(asyncio.CancelledError):
                    self._debounce_task.exception()

        cfg = self.cfg_getter()
        wait = debounce_seconds(cfg.behaviour, self.rng)
        self._debounce_task = asyncio.ensure_future(self._debounce_fire(wait))

    async def _debounce_fire(self, wait: float) -> None:
        try:
            await self.sleep(wait)
            buffer = self._debounce_buffer
            self._debounce_buffer = []
            if not buffer:
                return
            trigger, msg, display_name = self._pick_strongest(buffer)
            await self._handle_debounced(trigger, msg, display_name)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("debounce fire failed")

    def _pick_strongest(
        self, buffer: list[tuple[Trigger, GateMessage, str]]
    ) -> tuple[Trigger, GateMessage, str]:
        best = max(_TRIGGER_PRIORITY[item[0]] for item in buffer)
        candidates = [item for item in buffer if _TRIGGER_PRIORITY[item[0]] == best]
        return candidates[-1]

    async def _handle_debounced(
        self, trigger: Trigger, msg: GateMessage, display_name: str
    ) -> None:
        try:
            await self._handle_debounced_inner(trigger, msg, display_name)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "handle debounced failed: trigger=%s trigger_msg_id=%s", trigger, msg.tg_message_id
            )

    async def _handle_debounced_inner(
        self, trigger: Trigger, msg: GateMessage, display_name: str
    ) -> None:
        if trigger is Trigger.AMBIENT:
            await self._respond(
                trigger=Trigger.AMBIENT,
                trigger_msg_id=msg.tg_message_id,
                user_id=msg.user_id,
                situation=self.frag.text("situations.ambient"),
                delay_sec=0,
                trigger_text="",
            )
            return

        cfg = self.cfg_getter()
        now = self._clock()
        hot_until = await self._hot_until(cfg, now)
        earliest = await self._mention_earliest_due(cfg, msg.user_id, now)
        if hot_until is not None:
            earliest = min(earliest, now + cfg.behaviour.hot_window.mention_max_delay_sec)
        pending_rows = [
            row for row in await self.db.load_pending() if row.id not in self._in_flight
        ]
        if pending_rows:
            pending = pending_rows[0]
            new_due = now + fast_delay(cfg.behaviour, self.rng)
            if hot_until is not None:
                new_due = self._cap_hot_mention_delay(new_due, now, cfg.behaviour.hot_window)
            new_due = self._apply_cooldown_floor(new_due, earliest)
            await self.db.update_pending_due(pending.id, new_due)
            # Merge: the new address is appended to those already accumulated for this
            # pending instead of replacing them (several people may have written to the bot
            # while it waited); situation_addressed shows them all at generation time.
            self._pending_info.setdefault(pending.id, []).append((display_name, msg.text))
            row = PendingRow(
                id=pending.id,
                trigger_tg_message_id=pending.trigger_tg_message_id,
                user_id=pending.user_id,
                trigger=pending.trigger,
                due_at=new_due,
                created_at=pending.created_at,
                done_at=None,
            )
            self._schedule_pending_timer(row, new_due)
        else:
            urgent = self.patterns_getter().urgent(msg.text)
            due = now + pick_delay(cfg.behaviour, self.rng, urgent=urgent)
            if hot_until is not None:
                due = self._cap_hot_mention_delay(due, now, cfg.behaviour.hot_window)
            due = self._apply_cooldown_floor(due, earliest)
            pending_id = await self.db.insert_pending(
                trigger_tg_message_id=msg.tg_message_id,
                user_id=msg.user_id,
                trigger=trigger.value,
                due_at=due,
                created_at=now,
            )
            self._pending_info[pending_id] = [(display_name, msg.text)]
            row = PendingRow(
                id=pending_id,
                trigger_tg_message_id=msg.tg_message_id,
                user_id=msg.user_id,
                trigger=trigger.value,
                due_at=due,
                created_at=now,
                done_at=None,
            )
            self._schedule_pending_timer(row, due)

    async def _mention_earliest_due(self, cfg: Config, user_id: int, now: int) -> int:
        """An address cooldown becomes a delay, not a refusal: the gate (step 6) no longer
        drops on cooldown; instead the reply to an address cannot go out before
        ``earliest``, the maximum of the per-chat cooldown and the per-author cooldown. The
        last_*_at values are read from state directly (like ``_recheck_ambient_budget``);
        a missing one counts as 0, which on the unix scale is always below ``now`` and so
        shifts nothing."""
        behaviour = cfg.behaviour
        last_chat_raw = await self.db.get_state("last_mention_reply_at")
        last_chat = int(last_chat_raw) if last_chat_raw is not None else 0
        last_user_raw = await self.db.get_state(f"last_mention_reply_at:{user_id}")
        last_user = int(last_user_raw) if last_user_raw is not None else 0
        return max(
            last_chat + behaviour.mention_chat_cooldown_sec,
            last_user + behaviour.mention_cooldown_sec,
        )

    def _apply_cooldown_floor(self, due: int, earliest: int) -> int:
        """due, but not earlier than earliest; when it had to be shifted, add a small random
        spread (5-30s) on top of earliest so all shifted replies do not hit the same second,
        and log the shift at INFO."""
        if due < earliest:
            due = earliest + self.rng.randint(5, 30)
            logger.info("mention delayed by cooldown until %s", due)
        return due

    async def _hot_until(self, cfg: Config, now: int) -> int | None:
        """``hot_until`` from state if the hot window is enabled and still open, else
        ``None``. A shared primitive: used both for the delay cap of a reply to an address
        (``_handle_debounced_inner``) and for the "in the window" flag of an ambient reply
        (``_generate_and_send``). Read from state directly, like
        ``_mention_earliest_due``, not through the gate, which ignores addresses entirely."""
        if not cfg.behaviour.hot_window.enabled:
            return None
        raw = await self.db.get_state("hot_until")
        if raw is None:
            return None
        try:
            hot_until = int(raw)
        except ValueError:
            return None
        return hot_until if hot_until > now else None

    def _cap_hot_mention_delay(self, due: int, now: int, hot_window: HotWindowConfig) -> int:
        """In the hot window a reply to an address cannot be postponed beyond
        ``mention_max_delay_sec``; the cooldown shift (``_apply_cooldown_floor``) is applied
        on top of this cap."""
        capped = now + hot_window.mention_max_delay_sec
        if due > capped:
            logger.info("hot window: mention delay capped")
            return capped
        return due

    # ------------------------------------------------------------------ #
    # Pending timer and firing.
    # ------------------------------------------------------------------ #

    def _schedule_pending_timer(self, row: PendingRow, due_at: int) -> None:
        existing = self._pending_tasks.get(row.id)
        if existing is not None and not existing.done():
            existing.cancel()
        self._pending_tasks[row.id] = asyncio.ensure_future(
            self._pending_wait_and_fire(row, due_at)
        )

    async def _pending_wait_and_fire(self, row: PendingRow, due_at: int) -> None:
        try:
            wait = max(0.0, due_at - self._clock())
            await self.sleep(wait)
            fresh = PendingRow(
                id=row.id,
                trigger_tg_message_id=row.trigger_tg_message_id,
                user_id=row.user_id,
                trigger=row.trigger,
                due_at=due_at,
                created_at=row.created_at,
                done_at=None,
            )
            await self._fire_pending(fresh)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pending wait and fire failed: pending_id=%s", row.id)

    async def _fire_pending(self, row: PendingRow) -> None:
        self._in_flight.add(row.id)
        try:
            await self._fire_pending_inner(row)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("fire pending failed: pending_id=%s", row.id)
        finally:
            self._in_flight.discard(row.id)

    async def _fire_pending_inner(self, row: PendingRow) -> None:
        self._pending_tasks.pop(row.id, None)
        cfg = self.cfg_getter()
        tz = cfg.persona.timezone
        now = self._clock()

        if in_window(now, tz, cfg.behaviour.quiet_window):
            items = self._pending_info.pop(row.id, None)
            if items:
                display_name, text = items[-1]
            else:
                display_name, text = self.frag.prefix("placeholders.participant").strip(), ""
            if not text:
                logger.warning(
                    "night queue: pending %s restored without original text (restart), "
                    "writing placeholder for display_name=%r",
                    row.id,
                    display_name,
                )
            await self.db.enqueue_night(
                tg_message_id=row.trigger_tg_message_id,
                user_id=row.user_id,
                display_name=display_name,
                text=text,
                created_at=row.created_at,
            )
            await self.db.mark_pending_done(row.id, now)
            await self.db.insert_filter_log(
                trigger_tg_message_id=row.trigger_tg_message_id,
                candidate_text=None,
                verdict="cut",
                stage="send",
                reason="send:night",
                shadow=False,
                created_at=now,
            )
            return

        # Gap between replies: an address is not dropped but postponed, so the reply goes
        # out just not back to back with the bot's previous reply.
        delayed_due = await self._min_gap_due(cfg, now)
        if delayed_due is not None:
            await self.db.update_pending_due(row.id, delayed_due)
            logger.info("min gap: pending delayed until %s", delayed_due)
            self._schedule_pending_timer(row, delayed_due)
            return

        recheck_reason = await self._recheck(row, now, cfg)
        if recheck_reason is not None:
            self._pending_info.pop(row.id, None)
            await self.db.mark_pending_done(row.id, now)
            await self.db.insert_filter_log(
                trigger_tg_message_id=row.trigger_tg_message_id,
                candidate_text=None,
                verdict="cut",
                stage="send",
                reason=recheck_reason,
                shadow=False,
                created_at=now,
            )
            return

        addressed_items = self._pending_info.pop(row.id, None)
        if not addressed_items:
            # The process restarted between creating the pending and its firing, so
            # _pending_info is empty. Restore the accumulated addresses from messages.
            addressed_items = await self._collect_addressed_items(row)
        pending_text = addressed_items[-1][1] if addressed_items else ""
        delay_sec = now - row.created_at
        # mark_pending_done is NOT called here (incident: it used to be called before
        # generation, so a SIGTERM during the model call lost the reply forever and
        # restore_pending no longer saw it). pending_id goes into _respond/_generate_and_send
        # and is marked done only on the processing outcome, see _finish_pending.
        await self._respond(
            trigger=row.trigger,
            trigger_msg_id=row.trigger_tg_message_id,
            user_id=row.user_id,
            situation="",
            delay_sec=delay_sec,
            trigger_text=pending_text,
            addressed_items=addressed_items,
            pending_id=row.id,
        )

    async def _collect_addressed_items(self, row: PendingRow) -> list[tuple[str, str]]:
        """Restores the addresses to the bot for a pending lost in a restart
        (``_pending_info`` is empty: the process restarted between creating the pending and
        its firing, and the in-memory (display_name, text) pairs are gone).

        Takes the chat's human messages with ``created_at >= row.created_at`` (the moment
        the pending was created) and keeps those that look like an address to the bot: a
        reply to a bot message (``bot_reply_by_tg_id`` is not None), or ``mentions_bot``,
        or ``name_trigger``. The original trigger (``row.trigger_tg_message_id``) is added
        explicitly and always first: the debounce may have given it a created_at slightly
        earlier than ``row.created_at`` (the pending's created_at is when it was created,
        not the original message's time), and the created_at filter would miss it. At most
        the last ``_ADDRESSED_ITEMS_LIMIT`` items; prompt.situation_addressed trims too,
        but we do it here as well to avoid holding extra items in memory on long merges."""
        patterns = self.patterns_getter()
        items: list[tuple[str, str]] = []
        seen_ids: set[int] = set()

        def _add(msg_row: MessageRow) -> None:
            if msg_row.tg_message_id is None or msg_row.tg_message_id in seen_ids:
                return
            seen_ids.add(msg_row.tg_message_id)
            items.append((msg_row.display_name or "", msg_row.text or ""))

        trigger_msg = await self.db.message_by_tg_id(self.chat_id, row.trigger_tg_message_id)
        if trigger_msg is not None:
            _add(trigger_msg)

        # Addresses that have their own pending (they came in while this one was being
        # generated, see _in_flight) are not taken here, or after a restart they would be
        # answered twice. The boundary is the next open pending's trigger: "next" by
        # insertion order (id), the boundary by tg_message_id (monotonic in the chat), not
        # by seconds, since several can fit in one second.
        later_trigger_ids = [
            other.trigger_tg_message_id
            for other in await self.db.load_pending()
            if other.id > row.id
        ]
        until_id = min(later_trigger_ids, default=None)

        candidates = await self.db.messages_since(self.chat_id, row.created_at)
        for candidate in candidates:
            if (
                until_id is not None
                and candidate.tg_message_id is not None
                and candidate.tg_message_id >= until_id
            ):
                continue
            text = candidate.text or ""
            addressed = False
            if candidate.reply_to_tg_message_id is not None:
                bot_reply = await self.db.bot_reply_by_tg_id(candidate.reply_to_tg_message_id)
                addressed = bot_reply is not None
            if not addressed and patterns.mentions_bot(text):
                addressed = True
            if not addressed and patterns.name_trigger(text) is not None:
                addressed = True
            if addressed:
                _add(candidate)

        return items[-_ADDRESSED_ITEMS_LIMIT:]

    async def _presence_over_cap(self, cfg: Config, now: int) -> bool:
        """The daily presence cap is reached. Counts from the same tables as ``gate_state``;
        the counters may have changed while the reply waited for its delay or the lock."""
        presence = cfg.behaviour.presence
        if not presence.enabled:
            return False
        midnight = day_start(now, cfg.persona.timezone)
        return presence.over_cap(
            human_messages_today=await self.db.count_messages_today(self.chat_id, midnight),
            bot_replies_today=await self.db.count_bot_replies_today(midnight),
        )

    async def _min_gap_due(self, cfg: Config, now: int) -> int | None:
        """The moment before which the bot's next message cannot go out, or None if the gap
        has already elapsed.

        The gap is counted from the last bot reply in ``bot_replies``, i.e. from any
        message: text, sticker, ``/life``, ``/say``. A small random spread on top (5-30s),
        like ``_apply_cooldown_floor``, so several postponed replies do not hit the same
        second."""
        min_gap = cfg.behaviour.min_gap_sec
        if min_gap <= 0:
            return None
        last_reply_at = await self.db.last_bot_reply_at()
        if last_reply_at is None or now - last_reply_at >= min_gap:
            return None
        return last_reply_at + min_gap + self.rng.randint(5, 30)

    async def _recheck(self, row: PendingRow, now: int, cfg: Config) -> str | None:
        """Only the deterministic gate steps (1-5, 7 and the address limits).

        The dice (step 11) and live conversation (step 9) are deliberately not rechecked,
        or the recheck would cut most already-approved ambient replies.

        The address cooldown (per chat and per person) is not rechecked here either: the
        gate (step 6) does not check it at all, since the cooldown is a delay, not a
        refusal. By construction this pending's due_at is already no earlier than earliest
        (see _mention_earliest_due/_apply_cooldown_floor in _handle_debounced_inner), so
        by firing time the cooldown cannot be violated.
        """
        synthetic = GateMessage(
            chat_id=self.chat_id,
            tg_message_id=row.trigger_tg_message_id,
            user_id=row.user_id,
            is_bot=False,
            text="",
            reply_to_bot=False,
            created_at=now,
        )
        state = await load_gate_state(self.db, cfg, synthetic, now)
        if state.panic:
            return "send:recheck_panic"
        if state.stop_until is not None and state.stop_until > now:
            return "send:recheck_stop"
        if row.user_id in state.muted_user_ids:
            return "send:recheck_muted"
        # The stop-topic cooldown does not silence addresses: check non-addresses only.
        # followup is not an explicit address but "probably to me": the cooldown still
        # applies to it, or it would go out right after a message with a stop topic.
        if (
            (row.trigger not in _ADDRESS_TRIGGER_VALUES or row.trigger == Trigger.FOLLOWUP.value)
            and state.topic_cooldown_until is not None
            and state.topic_cooldown_until > now
        ):
            return "send:recheck_topic"
        # Presence cap: reply and @mention always pass under the cap, as in the gate
        # (PRESENCE_EXEMPT); other causes (name, followup) may have used up the daily share
        # by send time.
        if row.trigger not in {
            t.value for t in PRESENCE_EXEMPT
        } and cfg.behaviour.presence.over_cap(
            human_messages_today=state.human_messages_today,
            bot_replies_today=state.bot_replies_today,
        ):
            return "send:recheck_presence"
        if state.mention_count_today >= cfg.behaviour.mention_daily_cap:
            return "send:recheck_mention_cap"
        return None

    # ------------------------------------------------------------------ #
    # Generation and sending.
    # ------------------------------------------------------------------ #

    async def _finish_pending(self, pending_id: int | None, done_at: int) -> None:
        """Marks the pending done, but only on the processing outcome (a successful send
        plus insert_bot_reply, or any filter_log silence outcome: recheck failure,
        night_queue, llm:no_model, llm:invalid_json, llm:silent, a filter cut outside
        shadow, LLMError). Called explicitly at each such point rather than
        unconditionally at the end of _respond_inner: otherwise an unhandled exception
        inside generation (see except Exception in _respond) would also mark the pending
        done via finally, which was the cause of the incident (SIGTERM during llm.call: the
        reply was lost forever, restore_pending no longer saw it).

        A consciously accepted risk: if the process is killed (SIGKILL, no chance for
        finally) after bot.send_message but before this call, the next start's
        restore_pending picks up the same pending and sends a duplicate. A rare case;
        tolerating it is cheaper than losing a reply.
        """
        if pending_id is not None:
            await self.db.mark_pending_done(pending_id, done_at)
            self._llm_retried.discard(pending_id)

    async def _respond(
        self,
        *,
        trigger: Trigger | str,
        trigger_msg_id: int | None,
        user_id: int | None,
        situation: str,
        delay_sec: int,
        trigger_text: str = "",
        addressed_items: list[tuple[str, str]] | None = None,
        pending_id: int | None = None,
        checkin_rows: Sequence[MessageRow] | None = None,
    ) -> None:
        try:
            await self._respond_inner(
                trigger=trigger,
                trigger_msg_id=trigger_msg_id,
                user_id=user_id,
                situation=situation,
                delay_sec=delay_sec,
                trigger_text=trigger_text,
                addressed_items=addressed_items,
                pending_id=pending_id,
                checkin_rows=checkin_rows,
            )
        except LLMError as exc:
            now = self._clock()
            if (
                pending_id is not None
                and _trigger_value(trigger) in _ADDRESS_TRIGGER_VALUES
                and exc.reason in _LLM_RETRY_REASONS
                and pending_id not in self._llm_retried
                and await self._retry_pending_later(
                    pending_id, trigger_msg_id, addressed_items, now
                )
            ):
                return
            await self.db.insert_filter_log(
                trigger_tg_message_id=trigger_msg_id,
                candidate_text=None,
                verdict="cut",
                stage="llm",
                reason=exc.reason,
                shadow=False,
                created_at=now,
            )
            await self._finish_pending(pending_id, now)
        except Exception:
            logger.exception(
                "responder failed: trigger=%s trigger_msg_id=%s", trigger, trigger_msg_id
            )
            # Deliberately NOT mark_pending_done: an unhandled exception (a transport break
            # during SIGTERM, any failure inside _generate_and_send) must not "bury" the
            # pending; restore_pending picks it up again on the next start (see
            # _finish_pending).

    async def _retry_pending_later(
        self,
        pending_id: int,
        trigger_msg_id: int | None,
        addressed_items: list[tuple[str, str]] | None,
        now: int,
    ) -> bool:
        """A provider failure does not bury an address: once, postpone the pending by 2-5
        minutes instead of staying silent. False means the pending was not found; act as before."""
        row = next((r for r in await self.db.load_pending() if r.id == pending_id), None)
        if row is None:
            return False
        self._llm_retried.add(pending_id)
        due = now + self.rng.randint(*_LLM_RETRY_DELAY_SEC)
        # While the model call ran, a new address may have merged into this pending: it
        # already moved due_at closer and appended itself to _pending_info. Overwrite
        # neither.
        if now < row.due_at < due:
            due = row.due_at
        await self.db.insert_filter_log(
            trigger_tg_message_id=trigger_msg_id,
            candidate_text=None,
            verdict="cut",
            stage="send",
            reason="send:llm_retry",
            shadow=False,
            created_at=now,
        )
        await self.db.update_pending_due(pending_id, due)
        merged = list(addressed_items or [])
        merged.extend(item for item in self._pending_info.get(pending_id, []) if item not in merged)
        if merged:
            self._pending_info[pending_id] = merged
        logger.info("llm failure: pending %s retried at %s", pending_id, due)
        self._schedule_pending_timer(row, due)
        return True

    async def _respond_inner(
        self,
        *,
        trigger: Trigger | str,
        trigger_msg_id: int | None,
        user_id: int | None,
        situation: str,
        delay_sec: int,
        trigger_text: str = "",
        addressed_items: list[tuple[str, str]] | None = None,
        pending_id: int | None = None,
        checkin_rows: Sequence[MessageRow] | None = None,
    ) -> None:
        # One Responder generates and sends strictly one reply at a time: without this lock
        # two PASS decisions waiting for the LLM in parallel could both pass the same budget
        # check (ambient_count/last_ambient_at) while both wait for the model, and both send.
        async with self._respond_lock:
            cfg = self.cfg_getter()
            tz = cfg.persona.timezone
            trigger_value = _trigger_value(trigger)
            now = self._clock()

            # The ambient/spontaneous budget recheck and the ambient "hot window" flag live
            # inside _generate_and_send (see its docstring): hot is read exactly once, before
            # the model call, so the recheck and the post-send counters stay consistent even
            # if the window closes while we wait for the LLM.
            await self._generate_and_send(
                cfg=cfg,
                tz=tz,
                trigger_value=trigger_value,
                trigger_msg_id=trigger_msg_id,
                user_id=user_id,
                situation=situation,
                delay_sec=delay_sec,
                now=now,
                trigger_text=trigger_text,
                addressed_items=addressed_items,
                pending_id=pending_id,
                checkin_rows=checkin_rows,
            )

    async def _recheck_ambient_budget(self, cfg: Config, now: int, hot: bool) -> str | None:
        """Fresh ambient/spontaneous budget check after taking _respond_lock.

        Uses the same fields as the gate (step 10: GateState.ambient_count_today /
        last_ambient_at) but reads them straight from state, not from the snapshot taken
        before waiting for the lock, which may have gone stale while this reply queued.

        ``hot`` is the ambient reply's "hot window" flag, computed once in
        ``_generate_and_send`` and passed in rather than reread: in the window the daily
        ambient budget is not spent at all, and the window's own budget is checked instead
        (``hot_ambient_count`` / ``hot_window.ambient_cap``). For spontaneous ``hot`` is
        always ``False``: the hot window does not apply to it.
        """
        if hot:
            hot_count_raw = await self.db.get_state("hot_ambient_count")
            hot_count = int(hot_count_raw) if hot_count_raw is not None else 0
            if hot_count >= cfg.behaviour.hot_window.ambient_cap:
                return "send:recheck_hot_cap"
            return None

        tz = cfg.persona.timezone
        ambient_count_raw = await self.db.get_state(day_key("ambient_count", now, tz))
        ambient_count = int(ambient_count_raw) if ambient_count_raw is not None else 0
        if ambient_count >= cfg.behaviour.daily_cap:
            return "send:recheck_ambient_cap"

        last_ambient_raw = await self.db.get_state("last_ambient_at")
        last_ambient_at = int(last_ambient_raw) if last_ambient_raw is not None else None
        if (
            last_ambient_at is not None
            and last_ambient_at + cfg.behaviour.chat_cooldown_min * 60 > now
        ):
            return "send:recheck_ambient_cooldown"
        return None

    async def _append_place_weather(
        self,
        block: str,
        *,
        cfg: Config,
        tz: str,
        now: int,
        is_address: bool,
        trigger_text: str,
        trigger_msg_id: int | None,
        patterns: Patterns,
        record: bool = True,
    ) -> str:
        """Second line of the weather block, about the place that was asked about.

        The cheap call is made only when the cause is obvious: a direct address (in
        ambient and unprompted replies the persona does not bring up the weather in another
        city himself), ``place_lookup`` is on and the ``filters.weather_request`` regex
        matched. After that the model decides: the regex only opens the door, and "was a
        place asked about at all" is the model's job.

        The home point is not duplicated: if the question is about it, the first block has
        said it all. Any failure (the model found no place, the geocoder found no
        coordinates) is a ``filter_log`` row and the previous block, with no exceptions
        escaping.
        """
        weather_cfg = cfg.behaviour.weather
        if (
            self.weather is None
            or self.weather_places is None
            or not is_address
            or not weather_cfg.enabled
            or not weather_cfg.place_lookup
            or not trigger_text
            or not patterns.weather_request(trigger_text)
        ):
            return block

        home_name = self.weather.home_name
        query = await self.weather_places.extract(trigger_text, now=now)
        if query is None:
            await self._weather_log(record, trigger_msg_id, "weather:no_place", None, now, cut=True)
            return block
        if home_name and query.casefold() == home_name.casefold():
            return block

        place = await self.weather.geocode(query)
        if place is None:
            await self._weather_log(
                record, trigger_msg_id, "weather:not_found", query, now, cut=True
            )
            return block
        if home_name and place.name.casefold() == home_name.casefold():
            return block

        line = render_weather_place(
            place.name, await self.weather.get(place), tz, now, frag=self.frag
        )
        if not line:
            await self._weather_log(
                record, trigger_msg_id, "weather:not_found", place.name, now, cut=True
            )
            return block

        await self._weather_log(record, trigger_msg_id, "weather:place", place.name, now, cut=False)
        return f"{block}\n{line}" if block else line

    async def _weather_log(
        self,
        record: bool,
        trigger_msg_id: int | None,
        reason: str,
        candidate: str | None,
        now: int,
        *,
        cut: bool,
    ) -> None:
        if not record:
            return
        await self.db.insert_filter_log(
            trigger_tg_message_id=trigger_msg_id,
            candidate_text=candidate,
            verdict="cut" if cut else "pass",
            stage="weather",
            reason=reason,
            shadow=False,
            created_at=now,
        )

    async def _build_prompt(
        self,
        *,
        cfg: Config,
        tz: str,
        trigger_value: str,
        trigger_msg_id: int | None,
        situation: str,
        delay_sec: int,
        now: int,
        trigger_text: str,
        addressed_items: list[tuple[str, str]] | None,
        record: bool = True,
        extra_row: MessageRow | None = None,
    ) -> _BuiltPrompt:
        """Builds the prompt for the main model, shared by the live reply
        (``_generate_and_send``) and the ``/ask`` preview (``preview``), so the owner sees
        exactly what would have gone to the chat. ``record=False`` means no ``filter_log``
        writes (weather by place); the rest of the build only reads the DB.
        ``extra_row`` is a message not in the DB (preview): appended to the end of the
        context in memory, and the ``context_window`` size is kept.
        """
        context_rows = await self.db.recent_messages(self.chat_id, cfg.behaviour.context_window)
        if extra_row is not None:
            context_rows = [*context_rows, extra_row][-cfg.behaviour.context_window :]
        recent_replies_list = await self.db.recent_bot_replies(cfg.behaviour.recent_replies_memory)
        context = render_context(context_rows)
        recent_replies = "\n".join(recent_replies_list)
        # The last 50 replies are read once per generation and reused three times: the
        # {avoid} slot, soften (the style:emoji_freq window) and FilterContext, so the DB is
        # not queried again.
        filter_recent_replies = await self.db.recent_bot_replies(_FILTER_RECENT_REPLIES_LIMIT)
        patterns = self.patterns_getter()
        # {avoid} is built BEFORE the model call from the same replies list that dedup:motif
        # and style:story_quota will later check: warning is cheaper than cutting a finished
        # reply and staying silent.
        avoid_block = render_avoid(
            used_motifs(
                filter_recent_replies,
                cfg.filters.motif_avoid_window,
                patterns.motifs,
                frag=self.frag,
            ),
            no_story=(
                story_count(
                    filter_recent_replies,
                    cfg.filters.story_window,
                    patterns.story_markers,
                    frag=self.frag,
                )
                >= cfg.filters.story_max
            ),
            frag=self.frag,
        )
        # Read the prompt/few-shot from prompt_store anew on every _respond rather than
        # caching in the constructor: /rollback and /ex add must be picked up by the very
        # next reply, without restarting the Responder.
        system_prompt = self.prompt_store.system_prompt()
        few_shot = self.prompt_store.few_shot_text()
        age = cfg.persona.age(local_date(now, tz))

        # Places: only on a request in an address (mention/reply/name), never for
        # ambient/spontaneous/morning (no unsolicited recommendations).
        is_address = trigger_value in _ADDRESS_TRIGGER_VALUES
        if is_address:
            # An address (mention/reply/name/followup) builds its own situation from the
            # accumulated addresses (there may be several if several people wrote to the bot
            # during debounce merging) instead of the one the caller passed: without an
            # explicit "X addressed you", in a 30-message context window the model answers
            # the most prominent message rather than whoever actually addressed it (a live
            # bug, the reason this was added). followup's addressing is only probable (a
            # cheap check, not the gate), so it uses its own situation_followup form with an
            # explicit caveat and the right to stay silent. A late reply is still appended
            # as a separate line.
            if addressed_items:
                situation = (
                    situation_followup(addressed_items, frag=self.frag)
                    if trigger_value == Trigger.FOLLOWUP.value
                    else situation_addressed(addressed_items, frag=self.frag)
                )
            else:
                situation = ""
            if delay_sec > cfg.behaviour.late_reply_threshold_sec:
                late = self.frag.text("situations.late")
                situation = f"{situation}\n{late}" if situation else late
        # A direct address -> the whole places cache goes into the prompt, always,
        # regardless of the text; the regex below only affects the recorded trigger (stats).
        places_block = (
            render_places_menu(await self.db.places_all(), frag=self.frag) if is_address else ""
        )
        places_requested = is_address and patterns.places_request(trigger_text)
        record_trigger = _PLACES_TRIGGER if places_requested else trigger_value

        # The {life} slot is always filled, for any trigger: it is the persona's memory, not
        # data limited to an address.
        life_block = render_life(await self.db.life_events(), tz, frag=self.frag)

        # The {diary} slot is also always filled, for any trigger: the facts the persona has
        # already told today about his doings.
        midnight = day_start(now, tz)
        today_facts = await self.db.self_facts_since(midnight)
        week_days = cfg.behaviour.diary.week_days
        week_facts = (
            [
                fact
                for fact in await self.db.self_facts_since(midnight - week_days * 86400)
                if fact.created_at < midnight
            ]
            if week_days > 0
            else []
        )
        diary_block = render_diary(today_facts, week_facts, tz, frag=self.frag)

        # The {chat_memory} slot is also always filled, for any trigger: recaps of past
        # weeks outlive the messages themselves and replace what retention already purged.
        chat_memory_block = render_chat_memory(
            await self.db.chat_memories(cfg.behaviour.chat_memory.in_prompt), tz, frag=self.frag
        )

        # The {jokes} slot is also always filled, for any trigger. The rows are read once:
        # after sending, the same rows are used to check which joke was said.
        jokes_cfg = cfg.behaviour.jokes
        joke_rows = (
            await self.db.jokes(jokes_cfg.in_prompt)
            if jokes_cfg.enabled and jokes_cfg.in_prompt > 0
            else []
        )
        jokes_block = render_jokes(
            joke_rows, now=now, cooldown_days=jokes_cfg.cooldown_days, frag=self.frag
        )

        # The {vocab} slot: meanings of words that came up in the conversation and, rarely,
        # an offer to use one.
        vocab = self.vocab_getter() if self.vocab_getter is not None else None
        vocab_plan = await self._vocab_block(
            cfg,
            vocab,
            tz=tz,
            now=now,
            trigger_value=trigger_value,
            trigger_text=trigger_text,
            context_rows=context_rows,
            recent_replies=filter_recent_replies,
            is_address=is_address,
        )

        # Weather is background for ANY trigger: both ambient and an address need it. The
        # client swallows its own errors and keeps its own cache, so there is no try and no
        # freshness check here. No home point (coordinates in .env) -> get() returns None
        # and the block is empty.
        weather_block = ""
        if self.weather is not None:
            weather_block = render_weather(
                await self.weather.get(),
                tz,
                now,
                home_name=self.weather.home_name,
                frag=self.frag,
            )
            weather_block = await self._append_place_weather(
                weather_block,
                cfg=cfg,
                tz=tz,
                now=now,
                is_address=is_address,
                trigger_text=trigger_text,
                trigger_msg_id=trigger_msg_id,
                patterns=patterns,
                record=record,
            )

        # checkin is the only trigger where the model itself says which of the messages
        # listed in the situation it answers (the "reply_to" field); other triggers use the
        # usual JSON reminder (the build_messages default).
        build_messages_kwargs: dict[str, str] = {}
        if trigger_value == _CHECKIN_TRIGGER:
            build_messages_kwargs["json_reminder"] = self.frag.text("prompt.json_reminder_checkin")
        messages = build_messages(
            system_prompt,
            age=age,
            few_shot=few_shot,
            context=context,
            recent_replies=recent_replies,
            places=places_block,
            situation=situation,
            avoid=avoid_block,
            life=life_block,
            diary=diary_block,
            chat_memory=chat_memory_block,
            jokes=jokes_block,
            weather=weather_block,
            vocab=vocab_plan.block,
            **build_messages_kwargs,
            frag=self.frag,
        )
        return _BuiltPrompt(
            messages=messages,
            context_rows=context_rows,
            recent_replies=recent_replies_list,
            filter_recent_replies=filter_recent_replies,
            patterns=patterns,
            system_prompt=system_prompt,
            is_address=is_address,
            record_trigger=record_trigger,
            joke_rows=joke_rows,
            vocab=vocab,
            vocab_over=vocab_plan.over_quota,
            vocab_allowed=vocab_plan.allowed,
        )

    async def _filter_context(
        self, cfg: Config, prompt: _BuiltPrompt, *, trigger_text: str, now: int
    ) -> FilterContext:
        muted_ids = await self.db.muted_user_ids()
        muted_names = list((await self.db.display_names(list(muted_ids))).values())
        # Always (not only for the "places" trigger): the regex:venue/regex:latin whitelist
        # must know the real venues regardless of whether they were asked about right now.
        places_names = await self.db.places_names()
        return FilterContext(
            cfg=cfg,
            recent_replies=prompt.filter_recent_replies,
            context_rows=prompt.context_rows,
            places_names=places_names,
            participant_names=_unique_participant_names(prompt.context_rows),
            bot_names=[cfg.persona.name, cfg.persona.display_name, *cfg.persona.name_triggers],
            muted_names=muted_names,
            patterns=prompt.patterns,
            system_prompt=prompt.system_prompt,
            trigger_text=trigger_text,
            now=now,
            vocab=prompt.vocab,
            vocab_over_quota=prompt.vocab_over,
            vocab_allowed=prompt.vocab_allowed,
            frag=self.frag,
        )

    async def preview(self, text: str, display_name: str, user_id: int = 0) -> PreviewOutcome:
        """``/ask``: what the persona would answer if ``display_name`` addressed him in the
        chat with ``text``: the same prompt with a fresh chat context, the same model,
        post-processing and output filter. Sends and writes nothing: no filter_log, no
        bot_replies, no counters, diary, jokes, vocabulary or hot window. The model calls
        are real and count toward the shared budget (llm_calls, judge, weather by place).
        No _respond_lock: nothing goes to the chat, so there is no reason to wait for
        someone else's generation.

        The message itself is appended to the context in memory only: in the live chat the
        handler has already recorded it before generation, and without it the model would
        see the text only in situation (truncated to 300 characters) and the filter would
        not see it at all
        (regex:echo, regex:starts_name)."""
        cfg = self.cfg_getter()
        tz = cfg.persona.timezone
        now = self._clock()
        if not cfg.llm.main_model:
            return PreviewOutcome(sent=False, text="", reason="llm:no_model")
        prompt = await self._build_prompt(
            cfg=cfg,
            tz=tz,
            trigger_value=Trigger.MENTION.value,
            trigger_msg_id=None,
            situation="",
            delay_sec=0,
            now=now,
            trigger_text=text,
            addressed_items=[(display_name, text)],
            record=False,
            extra_row=MessageRow(
                id=0,
                tg_message_id=None,
                chat_id=self.chat_id,
                user_id=user_id,
                display_name=display_name,
                text=text,
                reply_to_tg_message_id=None,
                is_bot=False,
                created_at=now,
            ),
        )
        try:
            result = await self.llm.call(
                prompt.messages, model=cfg.llm.main_model, max_tokens=cfg.llm.max_tokens, now=now
            )
        except LLMError as exc:
            return PreviewOutcome(sent=False, text="", reason=exc.reason)
        reply = parse_reply(result.text)
        if reply is None:
            return PreviewOutcome(sent=False, text=result.text, reason="llm:invalid_json")
        if not reply.speak:
            return PreviewOutcome(sent=False, text="", reason="llm:silent")
        fixed = soften(
            reply.text,
            recent_replies=prompt.filter_recent_replies,
            cfg=cfg.filters,
            frag=self.frag,
        )
        if not fixed.text:
            return PreviewOutcome(
                sent=False, text=reply.text, reason="fix:empty", fixes=fixed.fixes
            )
        filter_ctx = await self._filter_context(cfg, prompt, trigger_text=text, now=now)
        verdict = await filters.check_output(fixed.text, filter_ctx, self.judge)
        reasons = () if verdict.ok else (verdict.reasons or (verdict.reason,))
        cut_reason = _cut_reason(cfg, reasons) if reasons else None
        return PreviewOutcome(
            sent=cut_reason is None,
            text=fixed.text,
            reason=cut_reason or f"send:{Trigger.MENTION.value}",
            reasons=tuple(reasons),
            fixes=fixed.fixes,
        )

    async def _generate_and_send(
        self,
        *,
        cfg: Config,
        tz: str,
        trigger_value: str,
        trigger_msg_id: int | None,
        user_id: int | None,
        situation: str,
        delay_sec: int,
        now: int,
        trigger_text: str = "",
        addressed_items: list[tuple[str, str]] | None = None,
        pending_id: int | None = None,
        checkin_rows: Sequence[MessageRow] | None = None,
    ) -> SendOutcome:
        """Returns a SendOutcome with the same reason that goes to filter_log. Needed by
        ``announce_life``/``say``, which call this method directly (bypassing debounce and
        pending) and must tell the owner through commands.py whether the text was sent.
        ``_respond_inner`` ignores the result; the behaviour of ordinary triggers (gate
        PASS, morning, spontaneous) does not change.

        ``hot`` (the "hot window" flag) is read exactly once here, at the very start,
        before the model call, so the budget check (``_recheck_ambient_budget``) and the
        post-send counters stay consistent even if the window closes while we wait for the
        LLM. It applies only to ``ambient``: spontaneous and addresses are untouched by the
        hot window (addresses are handled separately by ``_handle_debounced_inner``).
        """
        hot = False
        if trigger_value == Trigger.AMBIENT.value:
            hot = await self._hot_until(cfg, now) is not None

        if trigger_value in _AMBIENT_LIKE_TRIGGER_VALUES:
            # While this reply waited for the lock, another could have gone out and updated
            # ambient_count/last_ambient_at (or hot_ambient_count): reread them from scratch
            # instead of relying on the state seen before taking the lock.
            recheck_reason = await self._recheck_ambient_budget(cfg, now, hot)
            if recheck_reason is not None:
                await self.db.insert_filter_log(
                    trigger_tg_message_id=trigger_msg_id,
                    candidate_text=None,
                    verdict="cut",
                    stage="send",
                    reason=recheck_reason,
                    shadow=False,
                    created_at=now,
                )
                await self._finish_pending(pending_id, now)
                return SendOutcome(sent=False, text="", reason=recheck_reason)

        if trigger_value in _UNSOLICITED_TRIGGER_VALUES:
            # Presence cap and gap between replies: before the model call, since an
            # unsolicited cause is worth neither money nor space in the chat. Addresses do
            # not get here: _recheck limits them (the cap) and the pending postponement (the
            # gap); /life and /say, the owner's buttons, are not limited at all.
            unsolicited_reason: str | None = None
            if await self._presence_over_cap(cfg, now):
                unsolicited_reason = "send:presence_cap"
            elif await self._min_gap_due(cfg, now) is not None:
                unsolicited_reason = "send:min_gap"
            if unsolicited_reason is not None:
                await self.db.insert_filter_log(
                    trigger_tg_message_id=trigger_msg_id,
                    candidate_text=None,
                    verdict="cut",
                    stage="send",
                    reason=unsolicited_reason,
                    shadow=False,
                    created_at=now,
                )
                await self._finish_pending(pending_id, now)
                return SendOutcome(sent=False, text="", reason=unsolicited_reason)

        if not cfg.llm.main_model:
            # The LLM is on (there is an openrouter_api_key) but no model is set yet. That is
            # not a call error but the normal state before the first /set llm.main_model.
            # Stay silent, with no network trip.
            await self.db.insert_filter_log(
                trigger_tg_message_id=trigger_msg_id,
                candidate_text=None,
                verdict="cut",
                stage="llm",
                reason="llm:no_model",
                shadow=False,
                created_at=now,
            )
            if (
                self._last_no_model_warn_at is None
                or now - self._last_no_model_warn_at >= _NO_MODEL_WARN_INTERVAL_SEC
            ):
                self._last_no_model_warn_at = now
                logger.warning("llm.main_model is not set, no reply generated")
            await self._finish_pending(pending_id, now)
            return SendOutcome(sent=False, text="", reason="llm:no_model")

        prompt = await self._build_prompt(
            cfg=cfg,
            tz=tz,
            trigger_value=trigger_value,
            trigger_msg_id=trigger_msg_id,
            situation=situation,
            delay_sec=delay_sec,
            now=now,
            trigger_text=trigger_text,
            addressed_items=addressed_items,
        )
        recent_replies_list = prompt.recent_replies
        filter_recent_replies = prompt.filter_recent_replies
        is_address = prompt.is_address
        record_trigger = prompt.record_trigger
        joke_rows = prompt.joke_rows
        vocab = prompt.vocab
        messages = prompt.messages

        result = await self.llm.call(
            messages, model=cfg.llm.main_model, max_tokens=cfg.llm.max_tokens, now=now
        )

        reply = parse_reply(result.text)
        if reply is None:
            await self.db.insert_filter_log(
                trigger_tg_message_id=trigger_msg_id,
                candidate_text=None,
                verdict="cut",
                stage="llm",
                reason="llm:invalid_json",
                shadow=False,
                created_at=now,
            )
            await self._finish_pending(pending_id, now)
            return SendOutcome(sent=False, text="", reason="llm:invalid_json")

        if not reply.speak:
            await self.db.insert_filter_log(
                trigger_tg_message_id=trigger_msg_id,
                candidate_text=None,
                verdict="cut",
                stage="llm",
                reason="llm:silent",
                shadow=False,
                created_at=now,
            )
            await self._finish_pending(pending_id, now)
            return SendOutcome(sent=False, text="", reason="llm:silent")

        # checkin: the model gave the number of the message it answers (a 1-based index into
        # checkin_rows). A valid number yields both reply_to_message_id (below, together
        # with the ordinary address) and trigger_text (that message's text, for the output
        # filter's regex:venue/regex:latin, as if it were an ordinary address). An invalid
        # or missing number leaves both empty, and the reply goes to the chat with no
        # reply-to (like ambient).
        checkin_reply_to_message_id: int | None = None
        if trigger_value == _CHECKIN_TRIGGER and checkin_rows is not None:
            reply_to_index = reply.reply_to
            if reply_to_index is None or not 1 <= reply_to_index <= len(checkin_rows):
                # A reply "to everyone" with no addressee is no longer sent: the model must
                # name the number of the message it answers, otherwise it is not a reply but
                # butting in.
                await self.db.insert_filter_log(
                    trigger_tg_message_id=trigger_msg_id,
                    candidate_text=reply.text,
                    verdict="cut",
                    stage="send",
                    reason="send:checkin_no_target",
                    shadow=False,
                    created_at=now,
                )
                await self._finish_pending(pending_id, now)
                return SendOutcome(sent=False, text=reply.text, reason="send:checkin_no_target")

            selected_index = reply_to_index - 1
            selected_row = checkin_rows[selected_index]
            checkin_reply_to_message_id = selected_row.tg_message_id
            trigger_text = selected_row.text or ""
            if self.followup is not None and not await self.followup.check(
                text=trigger_text,
                display_name=selected_row.display_name or "",
                context_rows=[
                    row for index, row in enumerate(checkin_rows) if index != selected_index
                ],
                recent_replies=recent_replies_list[-_CHECKIN_FOLLOWUP_RECENT_REPLIES:],
                now=now,
            ):
                # The second, cheap model saw no address to the persona here, so stay silent:
                # the main model tends to treat any conversation it once touched as addressed
                # to itself.
                await self.db.insert_filter_log(
                    trigger_tg_message_id=trigger_msg_id,
                    candidate_text=reply.text,
                    verdict="cut",
                    stage="send",
                    reason="send:checkin_not_addressed",
                    shadow=False,
                    created_at=now,
                )
                await self._finish_pending(pending_id, now)
                return SendOutcome(sent=False, text=reply.text, reason="send:checkin_not_addressed")

        # Soft fixes BEFORE the output filter: dashes and too frequent or extra emoji are
        # cosmetics, not a character violation, so they are fixed by hand rather than cut.
        fixed = soften(
            reply.text,
            recent_replies=filter_recent_replies,
            cfg=cfg.filters,
            frag=self.frag,
        )
        if fixed.fixes:
            for fix_reason in fixed.fixes:
                await self.db.insert_filter_log(
                    trigger_tg_message_id=trigger_msg_id,
                    candidate_text=reply.text,
                    verdict="fix",
                    stage="fix",
                    reason=fix_reason,
                    shadow=False,
                    created_at=now,
                )
            logger.info("fix: %s", ", ".join(fixed.fixes))
        text = fixed.text

        if not text:
            # The reply was a single emoji, so it is empty after the fix: stay silent, as
            # with llm:silent.
            await self.db.insert_filter_log(
                trigger_tg_message_id=trigger_msg_id,
                candidate_text=reply.text,
                verdict="cut",
                stage="fix",
                reason="fix:empty",
                shadow=False,
                created_at=now,
            )
            await self._finish_pending(pending_id, now)
            return SendOutcome(sent=False, text=reply.text, reason="fix:empty")

        filter_ctx = await self._filter_context(cfg, prompt, trigger_text=trigger_text, now=now)
        verdict = await filters.check_output(text, filter_ctx, self.judge)
        if not verdict.ok:
            # reasons can be empty only for a hand-built foreign FilterVerdict (tests); then
            # use verdict.reason.
            reasons = verdict.reasons or (verdict.reason,)
            cut_reason = _cut_reason(cfg, reasons)
            cut = cut_reason is not None
            # One filter_log row per fired reason (for per-layer shadow statistics);
            # shadow=1 means "recorded but sent".
            for reason in reasons:
                stage = reason.split(":", 1)[0] if reason else "filter"
                await self.db.insert_filter_log(
                    trigger_tg_message_id=trigger_msg_id,
                    candidate_text=text,
                    verdict="cut",
                    stage=stage,
                    reason=reason,
                    shadow=not cut,
                    created_at=now,
                )
            if cut_reason is not None:
                logger.info("cut: %s", ", ".join(reasons))
                await self._finish_pending(pending_id, now)
                return SendOutcome(sent=False, text=text, reason=cut_reason)

        reply_to_message_id: int | None = None
        if is_address and trigger_msg_id is not None:
            after_count = await self.db.messages_after(self.chat_id, trigger_msg_id)
            if after_count > 0 or delay_sec > cfg.behaviour.reply_as_reply_after_sec:
                reply_to_message_id = trigger_msg_id
        elif trigger_value == _CHECKIN_TRIGGER:
            reply_to_message_id = checkin_reply_to_message_id

        # Second, cheap LLM call: the finished (filter-passed) text may be replaced by a
        # sticker from the catalog; the main model does not know about stickers at all.
        # None -> send the text, as before.
        sticker = await self._maybe_choose_sticker(
            cfg=cfg,
            tz=tz,
            trigger_value=trigger_value,
            reply_text=text,
            trigger_text=trigger_text,
            now=now,
        )

        if sticker is not None:
            await self._run_typing("…", duration=self.rng.uniform(*_STICKER_TYPING_RANGE_SEC))
            sticker_sent = await self.bot.send_sticker(
                self.chat_id, sticker.file_id, reply_to_message_id=reply_to_message_id
            )
            sent_message_id = sticker_sent.message_id
            # The sticker number in the text: this is how it reaches recent_replies / the
            # model's context and /last, and how recent_sticker_ids restores the "recently
            # used" set at the next choice.
            sent_text = (
                f"{self.frag.render('placeholders.sticker_tag', id=str(sticker.id))} {sticker.text}"
            )
            send_reason = "send:sticker"
        else:
            await self._run_typing(text)
            text_sent = await self.bot.send_message(
                self.chat_id, text, reply_to_message_id=reply_to_message_id
            )
            sent_message_id = text_sent.message_id
            sent_text = text
            # The hot window has its own reason in filter_log (for /why statistics);
            # bot_replies.trigger stays record_trigger ("ambient") as before.
            send_reason = "send:ambient_hot" if hot else f"send:{record_trigger}"

        await self.db.insert_bot_reply(
            tg_message_id=sent_message_id,
            reply_to_tg_message_id=reply_to_message_id,
            trigger=record_trigger,
            trigger_tg_message_id=trigger_msg_id,
            text=sent_text,
            prompt_version=self.prompt_store.prompt_version(),
            few_shot_version=self.prompt_store.few_shot_version(),
            delay_sec=delay_sec,
            created_at=now,
        )

        # Address/ambient/spontaneous budget counters: a shared tail, the same whether text
        # or a sticker went out (only the way of sending and what went into bot_replies.text
        # above differ).
        if is_address:
            await self.db.increment_state(day_key("mention_count", now, tz))
            await self.db.set_state("last_mention_reply_at", str(now))
            if user_id is not None:
                await self.db.set_state(f"last_mention_reply_at:{user_id}", str(now))
        elif trigger_value == Trigger.AMBIENT.value:
            if hot:
                # The hot window does not eat the daily ambient budget: it has its own
                # counter per window, and ambient_count(day)/last_ambient_at are untouched.
                await self.db.increment_state("hot_ambient_count")
            else:
                await self.db.increment_state(day_key("ambient_count", now, tz))
                await self.db.set_state("last_ambient_at", str(now))
        elif trigger_value == "spontaneous":
            await self.db.increment_state(day_key("ambient_count", now, tz))
            await self.db.set_state("last_ambient_at", str(now))
            await self.db.increment_state(week_key("spontaneous_count", now, tz))
        # morning: counters do not change, it is not part of the address budget.

        if sticker is not None:
            await self.db.increment_state(day_key("sticker_count", now, tz))
            await self.db.set_state("replies_since_sticker", "0")
        else:
            await self.db.increment_state("replies_since_sticker")

        await self.db.insert_filter_log(
            trigger_tg_message_id=trigger_msg_id,
            candidate_text=text,
            verdict="pass",
            stage="send",
            reason=send_reason,
            shadow=False,
            created_at=now,
        )
        await self._finish_pending(pending_id, now)
        if sticker is None:
            self._spawn_diary(sent_text, sent_message_id, now)
            # A joke the persona returned to the chat rests for cooldown_days days.
            for joke in joke_rows:
                if joke_used(sent_text, joke.text, self.patterns_getter().lang):
                    await self.db.mark_joke_used(joke.id, now)
            await self._note_vocab_used(vocab, sent_text, tz=tz, now=now)
        # The hot window from the SHARED tail only if the owner enabled
        # hot_window.open_on_any_reply. By default only /life and /say open the window:
        # "be more attentive after every own reply" in practice meant the bot extended its
        # own presence in the chat indefinitely. The "how did it go?" callback does not open
        # the window.
        if cfg.behaviour.hot_window.open_on_any_reply and trigger_value != _CALLBACK_TRIGGER:
            await self._maybe_open_hot_window(cfg, now)
        return SendOutcome(
            sent=True, text=sent_text, reason=send_reason, tg_message_id=sent_message_id
        )

    async def _vocab_block(
        self,
        cfg: Config,
        vocab: Vocab | None,
        *,
        tz: str,
        now: int,
        trigger_value: str,
        trigger_text: str,
        context_rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
        is_address: bool,
    ) -> _VocabPlan:
        """The ``{vocab}`` slot and everything the ``dedup:vocab`` filter needs from it.
        The owner's news (``life``) is his own text: vocabulary words are neither mixed in
        nor cut."""
        vocab_cfg = cfg.behaviour.vocab
        if vocab is None or not vocab or not vocab_cfg.enabled or trigger_value == _LIFE_TRIGGER:
            return _VocabPlan()
        count_raw = await self.db.get_state(day_key(_VOCAB_COUNT_KEY, now, tz))
        count_today = int(count_raw) if count_raw is not None else 0
        over_quota = vocab_over_quota(
            vocab,
            cfg=vocab_cfg,
            recent_replies=recent_replies,
            count_today=count_today,
            frag=self.frag,
        )
        # A request ("say it in your own words") and "did not understand" only on an
        # address: in ambient nobody addressed them to him.
        requested = is_address and vocab_requested(trigger_text, vocab, vocab_cfg)
        explain = is_address and explain_requested(trigger_text, vocab_cfg)
        # "What did you just say?" means a word in his own recent reply. Only in answer to
        # an address: otherwise the meaning of his own word in every prompt nudged the model
        # to repeat it.
        own_words: list[VocabEntry] = []
        if is_address:
            for reply in recent_replies[-_VOCAB_OWN_REPLIES:]:
                own_words.extend(e for e in vocab.find(reply) if e not in own_words)
        # "Did not understand": the meanings of his words first, so they do not fall out of
        # max_meanings.
        meanings: list[VocabEntry] = list(own_words) if explain else []
        sources = [trigger_text]
        if vocab_cfg.meaning_context > 0:
            sources.extend(row.text or "" for row in context_rows[-vocab_cfg.meaning_context :])
        for source in sources:
            meanings.extend(e for e in vocab.find(source) if e not in meanings)
        meanings.extend(e for e in own_words if e not in meanings)
        meanings = meanings[: max(vocab_cfg.max_meanings, len(own_words) if explain else 0)]

        offer: list[VocabEntry] = []
        if requested:
            # Asked directly: no dice, no daily cap and no pause.
            offer = pick_offer(
                vocab,
                recent_replies=recent_replies,
                size=vocab_cfg.request_size,
                rng=self.rng,
                exclude=meanings,
                fill=True,
                frag=self.frag,
            )
            logger.info("vocab requested: %s", ", ".join(entry.word for entry in offer))
        elif offer_allowed(
            cfg=vocab_cfg,
            vocab=vocab,
            recent_replies=recent_replies,
            count_today=count_today,
            rng=self.rng,
            frag=self.frag,
        ):
            offer = pick_offer(
                vocab,
                recent_replies=recent_replies,
                size=vocab_cfg.offer_size,
                rng=self.rng,
                exclude=meanings,
                frag=self.frag,
            )
            if offer:
                logger.info("vocab offer: %s", ", ".join(entry.word for entry in offer))
        # What the filter lets through even over the limit: what was offered on request and
        # his own recent words when asked to explain them.
        allowed = frozenset([*(offer if requested else ()), *(own_words if explain else ())])
        return _VocabPlan(
            block=render_vocab(offer, meanings, requested=requested, frag=self.frag),
            over_quota=over_quota,
            allowed=allowed,
        )

    async def _note_vocab_used(self, vocab: Vocab | None, text: str, *, tz: str, now: int) -> None:
        """A word was heard in the sent reply: bump the daily counter and the "last" value
        for /status. Whether it was offered or the model recalled it itself does not matter:
        the quota is shared."""
        if vocab is None or not vocab:
            return
        used = vocab.find(text)
        if not used:
            return
        await self.db.increment_state(day_key(_VOCAB_COUNT_KEY, now, tz))
        await self.db.set_state(_VOCAB_LAST_KEY, f"{now}|{used[0].word}")
        logger.info("vocab used: %s", ", ".join(entry.word for entry in used))

    async def _panic_or_stop_reason(self, now: int) -> str | None:
        """ "panic"/"stop" if the bot is silent right now. Reads ``state`` the same way as
        ``gate_state.load_gate_state``/``_recheck``: panic is the string "1", stop_until is
        an int greater than ``now``. Used by ``announce_life``/``say``, which have no gate
        on their path (the owner's "publish now" button bypasses the night window, limits
        and the dice, but not panic/stop)."""
        panic_raw = await self.db.get_state("panic")
        if panic_raw is not None and panic_raw == "1":
            return "panic"
        stop_until_raw = await self.db.get_state("stop_until")
        if stop_until_raw is not None:
            try:
                stop_until = int(stop_until_raw)
            except ValueError:
                stop_until = None
            if stop_until is not None and stop_until > now:
                return "stop"
        return None

    async def _maybe_open_hot_window(self, cfg: Config, now: int) -> None:
        """After a successful ``/life`` or ``/say`` send, open the hot window: half an hour
        of livelier behaviour than usual, while the conversation most likely revolves
        around the dropped news/reply. A repeated ``/life``/``/say`` inside an already open
        window extends it anew from ``now`` (overwrites ``hot_until``), and the window
        counter is reset with it: earlier ambient replies of that window do not count
        toward the new limit."""
        hot_window = cfg.behaviour.hot_window
        if not hot_window.enabled or hot_window.minutes <= 0:
            return
        await self.db.set_state("hot_until", str(now + hot_window.minutes * 60))
        await self.db.set_state("hot_ambient_count", "0")

    def _spawn_diary(self, text: str, tg_message_id: int, sent_at: int) -> None:
        """Starts background extraction of a fact from a reply that went to the chat. The
        task lives outside ``_respond_lock`` and does not delay the reply; it is kept in a
        set (otherwise the event loop may let it be garbage collected) and cancelled in
        ``shutdown``. Stickers do not get here: the caller does not pass them."""
        if self.diary is None or not self.cfg_getter().behaviour.diary.enabled:
            return
        task = asyncio.ensure_future(self._extract_diary(text, tg_message_id, sent_at))
        self._diary_tasks.add(task)
        task.add_done_callback(self._diary_tasks.discard)

    async def _extract_diary(self, text: str, tg_message_id: int, sent_at: int) -> None:
        """Body of the background task: no exception (except cancellation) escapes.

        The fact is dated by the reply time (``sent_at``), not the moment the task got to
        run: otherwise a reply a minute before midnight would land on tomorrow. The fact
        goes into the system prompt for a week, so commands and talk about the model are
        not let in (``Patterns.unsafe_note``)."""
        if self.diary is None:
            return
        try:
            fact = await self.diary.extract(text, now=self._clock())
            if fact is None:
                return
            hit = self.patterns_getter().unsafe_note(fact)
            if hit is not None:
                logger.warning("diary: fact dropped, marker matched %r", hit)
                return
            await self.db.insert_self_fact(
                text=fact, bot_reply_tg_message_id=tg_message_id, created_at=sent_at
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("diary: background fact extraction failed")

    async def announce_life(self, event: LifeEventRow) -> SendOutcome:
        """Publishes a life event (``/life``/``/life post``) right now, bypassing
        debounce, pending, the night window, daily limits and the dice (the owner's command
        in the private chat is a "publish now" button, not a queue). Only
        ``panic``/``stop_until`` hold it.

        Under ``_respond_lock``, the same lock as ordinary generation: without it a
        parallel ambient reply and ``/life`` could read/write the shared budget at the same
        time (although "life" is not part of it, the model call and the bot_replies write
        must still go one at a time, as everywhere in this class)."""
        async with self._respond_lock:
            cfg = self.cfg_getter()
            now = self._clock()

            blocked = await self._panic_or_stop_reason(now)
            if blocked is not None:
                await self.db.insert_filter_log(
                    trigger_tg_message_id=None,
                    candidate_text=None,
                    verdict="cut",
                    stage="send",
                    reason=f"send:blocked_{blocked}",
                    shadow=False,
                    created_at=now,
                )
                return SendOutcome(sent=False, text="", reason=f"blocked:{blocked}")

            try:
                outcome = await self._generate_and_send(
                    cfg=cfg,
                    tz=cfg.persona.timezone,
                    trigger_value="life",
                    trigger_msg_id=None,
                    user_id=None,
                    situation=situation_life(event.text, frag=self.frag),
                    delay_sec=0,
                    now=now,
                    # The event text as trigger_text: Latin words from the owner's note
                    # ("Hyundai") go into the output filter's regex:venue/regex:latin
                    # whitelist, as if a person in the chat had named them. This does not
                    # affect places and stickers for "life".
                    trigger_text=event.text,
                )
            except LLMError as exc:
                await self.db.insert_filter_log(
                    trigger_tg_message_id=None,
                    candidate_text=None,
                    verdict="cut",
                    stage="llm",
                    reason=exc.reason,
                    shadow=False,
                    created_at=now,
                )
                return SendOutcome(sent=False, text="", reason=exc.reason)

            if outcome.sent and outcome.tg_message_id is not None:
                await self.db.mark_life_event_announced(
                    event.id, tg_message_id=outcome.tg_message_id, now=now
                )
            # The window after /life always opens, regardless of
            # hot_window.open_on_any_reply: it is the owner's button, and the conversation
            # really does revolve around the dropped news. A repeated call with
            # open_on_any_reply=true is harmless: the values are the same.
            if outcome.sent:
                await self._maybe_open_hot_window(cfg, now)
            return outcome

    async def say(self, text: str) -> SendOutcome:
        """Sends ``text`` to the chat verbatim, with no model and no output filter
        (``/say``): the same "publish now" button with the same limits (panic/stop only);
        does not write to the ``life_events`` memory."""
        async with self._respond_lock:
            cfg = self.cfg_getter()
            now = self._clock()

            blocked = await self._panic_or_stop_reason(now)
            if blocked is not None:
                await self.db.insert_filter_log(
                    trigger_tg_message_id=None,
                    candidate_text=None,
                    verdict="cut",
                    stage="send",
                    reason=f"send:blocked_{blocked}",
                    shadow=False,
                    created_at=now,
                )
                return SendOutcome(sent=False, text="", reason=f"blocked:{blocked}")

            await self._run_typing(text)
            sent_message = await self.bot.send_message(self.chat_id, text)
            await self.db.insert_bot_reply(
                tg_message_id=sent_message.message_id,
                reply_to_tg_message_id=None,
                trigger="say",
                trigger_tg_message_id=None,
                text=text,
                prompt_version=self.prompt_store.prompt_version(),
                few_shot_version=self.prompt_store.few_shot_version(),
                delay_sec=0,
                created_at=now,
            )
            await self.db.increment_state("replies_since_sticker")
            await self.db.insert_filter_log(
                trigger_tg_message_id=None,
                candidate_text=text,
                verdict="pass",
                stage="send",
                reason="send:say",
                shadow=False,
                created_at=now,
            )
            self._spawn_diary(text, sent_message.message_id, now)
            await self._maybe_open_hot_window(cfg, now)
            return SendOutcome(
                sent=True, text=text, reason="send:say", tg_message_id=sent_message.message_id
            )

    async def _maybe_choose_sticker(
        self,
        *,
        cfg: Config,
        tz: str,
        trigger_value: str,
        reply_text: str,
        trigger_text: str,
        now: int,
    ) -> Sticker | None:
        """None without a single DB/network trip if there is no chooser or the trigger does
        not qualify (morning/spontaneous: no butting in with a sticker where even the text
        is optional). Otherwise the budget check (``sticker_allowed``, before the chooser
        call: if min_replies_between is not satisfied the chooser is not called at all)
        and, if not blocked, the choice itself."""
        if self.sticker_chooser is None:
            return None
        if trigger_value not in _STICKER_ELIGIBLE_TRIGGER_VALUES:
            return None

        stickers_cfg = cfg.behaviour.stickers
        replies_since_raw = await self.db.get_state("replies_since_sticker")
        replies_since = (
            int(replies_since_raw)
            if replies_since_raw is not None
            else _NO_STICKER_YET_REPLIES_SINCE
        )
        count_today_raw = await self.db.get_state(day_key("sticker_count", now, tz))
        count_today = int(count_today_raw) if count_today_raw is not None else 0
        if not sticker_allowed(
            cfg=stickers_cfg, replies_since=replies_since, count_today=count_today
        ):
            return None

        recent_texts = await self.db.recent_bot_replies(stickers_cfg.recent_window * 3)
        exclude_ids = recent_sticker_ids(recent_texts, stickers_cfg.recent_window, frag=self.frag)
        return await self.sticker_chooser.choose(
            reply_text=reply_text, trigger_text=trigger_text, exclude_ids=exclude_ids, now=now
        )

    async def _run_typing(self, text: str, *, duration: float | None = None) -> None:
        """A typing loop for ``duration`` seconds. By default (``None``) it is based on the
        length of ``text`` (an ordinary text reply); an explicit ``duration`` is a short
        pause before a sticker, which has no "reply length"."""
        if duration is None:
            duration = len(text) / _CHARS_PER_SEC
        elapsed = 0.0
        while elapsed < duration:
            await self.bot.send_chat_action(self.chat_id, _TYPING_ACTION)
            step = min(_TYPING_STEP_SEC, duration - elapsed)
            await self.sleep(step)
            elapsed += step

    # ------------------------------------------------------------------ #
    # Restart, morning, unprompted, shutdown.
    # ------------------------------------------------------------------ #

    async def restore_pending(self) -> None:
        cfg = self.cfg_getter()
        now = self._clock()
        rows = await self.db.load_pending()
        for row in rows:
            if row.due_at >= now - cfg.behaviour.late_reply_threshold_sec:
                self._schedule_pending_timer(row, row.due_at)
            else:
                await self.db.mark_pending_done(row.id, now)
                await self.db.insert_filter_log(
                    trigger_tg_message_id=row.trigger_tg_message_id,
                    candidate_text=None,
                    verdict="cut",
                    stage="send",
                    reason="send:restart",
                    shadow=False,
                    created_at=now,
                )

    async def morning_job(self) -> None:
        """An infinite loop: each pass sleeps until a random moment inside
        ``morning_reply_window``, answers the night messages, then sleeps until the start
        of the next window. Like ``retention.retention_loop``: a failure of one iteration is
        logged and does not stop the loop; after an error the loop sleeps
        ``_ERROR_RETRY_SEC`` (through the injected ``sleep``) to avoid a busy loop, and
        recomputes the window on the next pass."""
        while True:
            try:
                await self._morning_iteration()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("morning_job: iteration failed")
                await self.sleep(_ERROR_RETRY_SEC)

    async def _morning_iteration(self) -> None:
        cfg = self.cfg_getter()
        tz = cfg.persona.timezone
        start_hhmm, end_hhmm = cfg.behaviour.morning_reply_window
        now = self._clock()
        if in_window(now, tz, cfg.behaviour.morning_reply_window):
            remaining = seconds_until(now, tz, end_hhmm)
            wait = self.rng.uniform(0, remaining) if remaining > 0 else 0.0
        else:
            start_wait = seconds_until(now, tz, start_hhmm)
            duration = _window_hours(cfg.behaviour.morning_reply_window) * _SECONDS_PER_HOUR
            wait = start_wait + self.rng.uniform(0, max(duration, 0.0))

        await self.sleep(wait)
        await self._run_morning_once()

        cfg = self.cfg_getter()
        tz = cfg.persona.timezone
        start_hhmm = cfg.behaviour.morning_reply_window[0]
        await self.sleep(seconds_until(self._clock(), tz, start_hhmm))

    async def _run_morning_once(self) -> None:
        rows = await self.db.night_unanswered()
        if not rows:
            return
        await self._respond(
            trigger="morning",
            trigger_msg_id=None,
            user_id=None,
            situation=self.frag.text("situations.morning"),
            delay_sec=0,
        )
        await self.db.mark_night_answered([row.id for row in rows], self._clock())

    async def spontaneous_job(self) -> None:
        """An infinite loop: once an hour checks the "unprompted" conditions. Like
        ``retention.retention_loop``: a failure of one iteration is logged and does not
        stop the loop; after an error the loop sleeps ``_ERROR_RETRY_SEC`` instead of the
        hourly interval to avoid a busy loop."""
        while True:
            try:
                await self.sleep(float(_SECONDS_PER_HOUR))
                await self._maybe_spontaneous()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("spontaneous_job: iteration failed")
                await self.sleep(_ERROR_RETRY_SEC)

    async def _maybe_spontaneous(self) -> None:
        cfg = self.cfg_getter()
        tz = cfg.persona.timezone
        now = self._clock()
        spontaneous_cfg = cfg.behaviour.spontaneous

        week_raw = await self.db.get_state(week_key("spontaneous_count", now, tz))
        week_count = int(week_raw) if week_raw is not None else 0
        if week_count >= spontaneous_cfg.per_week:
            return

        if not in_window(now, tz, spontaneous_cfg.window):
            return

        last_msg_at = await self.db.last_message_at(self.chat_id)
        if (
            last_msg_at is not None
            and now - last_msg_at < spontaneous_cfg.min_quiet_hours * _SECONDS_PER_HOUR
        ):
            return

        day_raw = await self.db.get_state(day_key("ambient_count", now, tz))
        day_count = int(day_raw) if day_raw is not None else 0
        if day_count >= cfg.behaviour.daily_cap:
            return

        if state_reason := await self._spontaneous_gate_blocked(cfg, now):
            logger.debug("spontaneous skipped: %s", state_reason)
            return

        window_hours = _window_hours(spontaneous_cfg.window)
        if window_hours <= 0:
            return
        probability = spontaneous_cfg.per_week / (window_hours * 7)
        if self.rng.random() >= probability:
            return

        await self._respond(
            trigger="spontaneous",
            trigger_msg_id=None,
            user_id=None,
            situation=self.frag.text("situations.spontaneous"),
            delay_sec=0,
        )

    async def _spontaneous_gate_blocked(self, cfg: Config, now: int) -> str | None:
        tz = cfg.persona.timezone
        synthetic = GateMessage(
            chat_id=self.chat_id,
            tg_message_id=0,
            user_id=0,
            is_bot=False,
            text="",
            reply_to_bot=False,
            created_at=now,
        )
        state = await load_gate_state(self.db, cfg, synthetic, now)
        if state.panic:
            return "panic"
        if state.stop_until is not None and state.stop_until > now:
            return "stop"
        if in_window(now, tz, cfg.behaviour.quiet_window):
            return "night"
        if state.topic_cooldown_until is not None and state.topic_cooldown_until > now:
            return "topic_cooldown"
        return None

    # ------------------------------------------------------------------ #
    # "Come back and check".
    # ------------------------------------------------------------------ #

    async def checkin_job(self) -> None:
        """An infinite loop: every ``cfg.behaviour.checkin.poll_sec`` checks whether it is
        time to "come back and check" the chat after the hot window has closed. Like
        ``spontaneous_job``/``retention.retention_loop``: a failure of one iteration is
        logged and does not stop the loop; after an error the loop sleeps
        ``_ERROR_RETRY_SEC`` instead of the usual period to avoid a busy loop. The period
        is reread from the config on every pass, so ``/set behaviour.checkin.poll_sec`` is
        picked up without a restart."""
        while True:
            try:
                poll_sec = self.cfg_getter().behaviour.checkin.poll_sec
                await self.sleep(float(poll_sec))
                await self._maybe_checkin()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("checkin_job: iteration failed")
                await self.sleep(_ERROR_RETRY_SEC)

    async def _maybe_checkin(self) -> None:
        """One pass of the job ("come back and check"), step by step:

        enabled -> not panic/stop/night/topic_cooldown (``_spontaneous_gate_blocked``, the
        same checks as for "unprompted" but without the budget ones: checkin has no daily
        or weekly limit of its own) -> the hot window is no longer open (phone in hand,
        ``followup`` copes on its own) -> there is at least one bot reply and no more than
        ``topic_max_hours`` have passed since it (otherwise the topic is dead:
        ``checkin_due`` is deleted and the next pass starts from scratch) -> ``checkin_due``
        was computed from the current ``hot_until`` (if it changed because a new bot reply
        reopened the window, recompute) and has arrived -> there are human messages after
        ``max(last_reply_at, checkin_last_at)`` -> the cheap prefilter
        (``behaviour.checkin.prefilter``) looks at the whole batch at once and decides
        whether to call the main model at all: "no" writes ``send:checkin_prefilter_no``
        and exits without a call, "yes" (or the prefilter is off / ``followup`` is not set)
        goes to ``_respond`` with ``situation_checkin``. Whether or not messages were found
        and the prefilter passed, the next ``checkin_due`` is always reset to
        ``now + randint(after_min)*60``: the loop goes on until the topic dies."""
        cfg = self.cfg_getter()
        checkin_cfg = cfg.behaviour.checkin
        if not checkin_cfg.enabled:
            return

        now = self._clock()
        if await self._spontaneous_gate_blocked(cfg, now):
            return

        if await self._hot_until(cfg, now) is not None:
            # The phone is still in hand: the hot window is open and the cheap followup
            # check already catches addressed messages, so no separate check is needed.
            return

        last_reply_at = await self.db.last_bot_reply_at()
        if last_reply_at is None:
            return
        if now - last_reply_at > checkin_cfg.topic_max_hours * _SECONDS_PER_HOUR:
            await self.db.delete_state(_CHECKIN_DUE_KEY)
            await self.db.delete_state(_CHECKIN_DUE_REF_KEY)
            return

        # due is computed from the last known hot_until (even an already closed one): the
        # moment the conversation was still "warm" around. If the hot window is fully off
        # (the raw key was never set), count from last_reply_at, the same reference point
        # hot_window would have used were it on.
        hot_until_raw = await self.db.get_state("hot_until")
        try:
            due_ref = int(hot_until_raw) if hot_until_raw is not None else last_reply_at
        except ValueError:
            due_ref = last_reply_at

        stored_due_raw = await self.db.get_state(_CHECKIN_DUE_KEY)
        stored_ref_raw = await self.db.get_state(_CHECKIN_DUE_REF_KEY)
        try:
            stored_ref = int(stored_ref_raw) if stored_ref_raw is not None else None
        except ValueError:
            stored_ref = None

        if stored_due_raw is None or stored_ref != due_ref:
            due = due_ref + self.rng.randint(*checkin_cfg.after_min) * 60
            await self.db.set_state(_CHECKIN_DUE_KEY, str(due))
            await self.db.set_state(_CHECKIN_DUE_REF_KEY, str(due_ref))
        else:
            try:
                due = int(stored_due_raw)
            except ValueError:
                due = due_ref + self.rng.randint(*checkin_cfg.after_min) * 60
                await self.db.set_state(_CHECKIN_DUE_KEY, str(due))

        if now < due:
            return

        # Quiet: "come back and check" is about a conversation that has already cooled. If
        # someone wrote in the chat just now, the bot waits: due is not moved, and the next
        # poll tries again.
        quiet_sec = checkin_cfg.quiet_min * 60
        if quiet_sec > 0:
            last_human_at = await self.db.last_message_at(self.chat_id)
            if last_human_at is not None and now - last_human_at < quiet_sec:
                return

        checkin_last_raw = await self.db.get_state(_CHECKIN_LAST_AT_KEY)
        try:
            checkin_last_at = int(checkin_last_raw) if checkin_last_raw is not None else 0
        except ValueError:
            checkin_last_at = 0
        since = max(last_reply_at, checkin_last_at)

        rows = await self.db.messages_since(self.chat_id, since, checkin_cfg.max_messages)
        if rows:
            call_main_model = True
            if checkin_cfg.prefilter and self.followup is not None:
                # Cheap prefilter: seven checks out of seven once ended in llm:silent, about
                # 20 cents a day paid for silence. The cheap model looks at the whole batch at
                # once and decides whether to call the main one at all.
                recent_replies = await self.db.recent_bot_replies(_CHECKIN_FOLLOWUP_RECENT_REPLIES)
                addressed = await self.followup.check_batch(
                    rows=rows, recent_replies=recent_replies, now=now
                )
                if addressed:
                    await self.db.insert_filter_log(
                        trigger_tg_message_id=None,
                        candidate_text=None,
                        verdict="pass",
                        stage="send",
                        reason="send:checkin_prefilter_yes",
                        shadow=False,
                        created_at=now,
                    )
                else:
                    call_main_model = False
                    await self.db.insert_filter_log(
                        trigger_tg_message_id=None,
                        candidate_text=None,
                        verdict="cut",
                        stage="send",
                        reason="send:checkin_prefilter_no",
                        shadow=False,
                        created_at=now,
                    )
                    logger.info("checkin prefilter: nothing for Fyodor, main model not called")

            if call_main_model:
                await self._respond(
                    trigger=_CHECKIN_TRIGGER,
                    trigger_msg_id=None,
                    user_id=None,
                    situation=situation_checkin(rows, frag=self.frag),
                    delay_sec=0,
                    checkin_rows=rows,
                )

        # The next check is always after_min from "now" (not from due_ref): the decision
        # "replied -> window open again, stayed silent -> the next check after_min later"
        # does not depend on whether there were messages on this pass.
        after_now = self._clock()
        await self.db.set_state(_CHECKIN_LAST_AT_KEY, str(after_now))
        await self.db.set_state(
            _CHECKIN_DUE_KEY, str(after_now + self.rng.randint(*checkin_cfg.after_min) * 60)
        )

    # ------------------------------------------------------------------ #
    # "How did it go?" callback.
    # ------------------------------------------------------------------ #

    async def callback_job(self) -> None:
        """An infinite loop like ``checkin_job``: every ``cfg.behaviour.checkin.poll_sec``
        checks whether it is time to ask a person how their story ended. A failed iteration
        is logged and the loop lives on."""
        while True:
            try:
                poll_sec = self.cfg_getter().behaviour.checkin.poll_sec
                await self.sleep(float(poll_sec))
                await self._maybe_callback()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("callback_job: iteration failed")
                await self.sleep(_ERROR_RETRY_SEC)

    async def _maybe_callback(self) -> None:
        """One pass: enabled -> not panic/stop/night/topic_cooldown -> the ``callback.window``
        -> ``min_days`` days passed since the last question -> the chat is quiet
        (``checkin.quiet_min``) -> not over the presence cap -> the gap between replies has
        elapsed -> the oldest open thread: one older than ``max_age_days`` is closed, a muted
        author and an author who has not written to the chat for long are skipped. Any
        generation outcome (sent, silence, cut) closes the thread and sets
        ``callback_last_at``: we do not ask about the same thing twice."""
        cfg = self.cfg_getter()
        callback_cfg = cfg.behaviour.callback
        if not callback_cfg.enabled:
            return

        tz = cfg.persona.timezone
        now = self._clock()
        if await self._spontaneous_gate_blocked(cfg, now):
            return
        if not in_window(now, tz, callback_cfg.window):
            return

        last_raw = await self.db.get_state(_CALLBACK_LAST_AT_KEY)
        if last_raw is not None:
            try:
                if now - int(last_raw) < callback_cfg.min_days * 86400:
                    return
            except ValueError:
                pass

        quiet_sec = cfg.behaviour.checkin.quiet_min * 60
        if quiet_sec > 0:
            last_human_at = await self.db.last_message_at(self.chat_id)
            if last_human_at is not None and now - last_human_at < quiet_sec:
                return

        if await self._presence_over_cap(cfg, now):
            return
        if await self._min_gap_due(cfg, now) is not None:
            return

        muted = await self.db.muted_user_ids()
        chosen = None
        for thread in await self.db.open_people_threads():
            if now - thread.created_at > callback_cfg.max_age_days * 86400:
                # Not "asked" but stale: delete rather than close, so /people does not show
                # a question that was never asked (retention would delete it anyway).
                await self.db.delete_people_thread(thread.id)
                continue
            if thread.user_id in muted:
                continue
            author_last = await self.db.last_message_at_by_user(self.chat_id, thread.user_id)
            if author_last is None or now - author_last > callback_cfg.author_active_days * 86400:
                continue
            chosen = thread
            break
        if chosen is None:
            return

        # The thread is closed BEFORE the attempt: any outcome (send, silence, cut,
        # failure, cancellation mid-send) must not lead to asking about the same thing a
        # second time. The cost is a rare question lost to a SIGTERM before sending;
        # repeating someone's personal story is worse.
        await self.db.close_people_thread(chosen.id, now)
        await self.db.set_state(_CALLBACK_LAST_AT_KEY, str(now))
        await self._respond(
            trigger=_CALLBACK_TRIGGER,
            trigger_msg_id=None,
            user_id=chosen.user_id,
            situation=situation_callback(chosen.display_name, chosen.text, frag=self.frag),
            delay_sec=0,
        )

    async def shutdown(self) -> None:
        """Cancels the debounce and pending timers. The pendings themselves stay in the DB:
        the next restore_pending (after a restart) picks them up again.

        If a generation is running at stop time (``_respond_lock`` is held: a model call or
        a send), give it up to ``_SHUTDOWN_GENERATION_WAIT_SEC`` to finish normally instead
        of tearing it with a cancel: a graceful SIGTERM must not cut a model call off
        mid-word (the incident: mark_pending_done set before generation lost the reply
        forever). If it does not finish in the allotted time, cancel as usual; the pending
        stays in the DB (done is not set) and restore_pending picks it up on the next start.
        """
        if self._respond_lock.locked():
            try:
                await asyncio.wait_for(
                    self._respond_lock.acquire(), timeout=_SHUTDOWN_GENERATION_WAIT_SEC
                )
            except TimeoutError:
                pass
            else:
                self._respond_lock.release()

        tasks: list[asyncio.Task[None]] = []
        for diary_task in list(self._diary_tasks):
            if not diary_task.done():
                diary_task.cancel()
            tasks.append(diary_task)
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
            tasks.append(self._debounce_task)
        for task in self._pending_tasks.values():
            if not task.done():
                task.cancel()
            tasks.append(task)
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending_tasks.clear()
        self._diary_tasks.clear()
