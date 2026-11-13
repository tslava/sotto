"""Emoji reactions to messages that the gate cut by dice.

When the gate cuts a message for a non-deterministic reason (``gate:dice`` or
``gate:ambient_cooldown``), instead of complete silence the bot sometimes sets a
reaction through the Bot API ``setMessageReaction``. It is cheap: no model call,
no new message, and it gives a presence effect. The reaction decision lives outside
the gate (``gate.py`` is unchanged): this module is called from ``bot.py`` after
the gate returned ``Verdict.DROP``.

``pick_reaction`` is a pure function, like ``should_consider`` in ``gate.py``: the input is
the drop reason, the author, a state snapshot, the config, ``rng`` and ``now``; the
output is an emoji or ``None``. It does not change state; the caller (``react``)
does that after the reaction is sent successfully.

A reaction 0.3 seconds after a message looks mechanical, and a dice-based choice
puts a reaction on anything. So two layers sit on top of the same prefilter:

* ``ReactionScheduler``: a ``delay_sec`` pause before the reaction (a person reads
  first); after the pause the same deterministic conditions (cooldown, cap, same
  author) are re-checked, because anything could have happened in the chat meanwhile;
* ``ReactionChooser``: a cheap model call (the same approach as ``judge.Judge`` and
  ``followup.FollowupChecker``): a short prompt with the chat context and the message,
  and a strictly JSON answer ``{"emoji": "<one from the list>"|null}``. With ``semantic``
  no dice is thrown at all: the model is the filter, and ``pick_reaction`` returns a stub.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, Protocol

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.types import ReactionTypeEmoji, ReactionTypeUnion

from sotto.config_models import Config, ReactionsConfig
from sotto.db import Database, MessageRow
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError
from sotto.prompt import render_context
from sotto.timeutil import day_key

logger = logging.getLogger(__name__)

# A separate model call counter for reactions: the cheap check must not eat the
# call cap of the main model (same approach as followup_calls/vision_count).
_COUNTER_KEY = "reaction_calls"

# Truncation of the message text in filter_log (react:declined), like _LOG_TEXT_MAX_LEN in bot.py.
_CANDIDATE_TEXT_MAX_LEN = 200

# Truncation of the model reason in the log: it may retell a participant's message.
_REASON_LOG_MAX_LEN = 100

# The {reply_to} slot: the message the author replied to. Without it the model reads
# a reply out of context and sets a laughing reaction on a complaint.
_REPLY_TO_MAX_LEN = 300


class _ReplyUnavailable(Enum):
    """The reply target exists but is not visible: purged by retention, not yet written
    by the handler (a photo is being described) or without text. Kept apart from "not a
    reply" so the model knows it lacks context; the prompt says to answer null in that case."""

    UNAVAILABLE = "unavailable"


REPLY_UNAVAILABLE: Final = _ReplyUnavailable.UNAVAILABLE

# None: not a reply; MessageRow: target found; REPLY_UNAVAILABLE: a reply with an invisible target.
ReplyTarget = MessageRow | Literal[_ReplyUnavailable.UNAVAILABLE] | None

# Duplicates the delimiter stripping from sanitize.normalize_text/prompt.py/judge.py:
# just in case, clean once more what goes into the reaction-choice prompt.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_SLOT_RE = re.compile(r"\{(context|name|text|emoji|reply_to)\}")


def _render_palette(allowed: Mapping[str, str]) -> str:
    """The {emoji} slot: one line per "emoji: when it fits". Without descriptions the model
    distinguished only "funny" and "approve" and used one laughing emoji almost always."""
    return "\n".join(f"{e} — {meaning}" if meaning else e for e, meaning in allowed.items())


def palette_without_repeats(
    emoji: Mapping[str, str], recent: Sequence[str], max_repeat: int
) -> dict[str, str]:
    """The palette without an emoji that was used the last ``max_repeat`` times in a row.

    Measured on 150 past cases: the model ignores a request in the prompt not to
    repeat itself, so the code limits repeats. ``recent`` is chronological.
    The palette's only emoji is never removed: an empty choice is worse than a repeat."""
    palette = dict(emoji)
    if max_repeat <= 0 or len(recent) < max_repeat or len(palette) <= 1:
        return palette
    tail = {e.replace("\ufe0f", "") for e in recent[-max_repeat:]}
    if len(tail) == 1:
        palette.pop(tail.pop(), None)
    return palette


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _render_reply_to(row: ReplyTarget, frag: Fragments) -> str:
    if row is None:
        return str(frag.text("placeholders.reply_to_none"))
    if row is REPLY_UNAVAILABLE or not row.text:
        return str(frag.text("placeholders.reply_to_unavailable"))
    text = row.text
    if len(text) > _REPLY_TO_MAX_LEN:
        text = text[:_REPLY_TO_MAX_LEN].rstrip() + "…"
    return _strip_fake_delimiters(f"{row.display_name}: {text}").strip()


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


# Reasons for which a reaction is sometimes set instead of silence. All of them are
# non-deterministic: "dice" (gate step 11), the ambient cooldown (step 10) and the cheap
# model refusing to treat a mention of the name as an address: the bot was talked
# about, there is no need to join the conversation, and a quiet nod is exactly the
# gesture needed.
# All other reasons (panic/stop/muted/topic/night/logistics/injection/mention_cap/
# ambient_cap...) mean full silence, never a reaction.
# "gate:not_live" is the most frequent refusal reason, and without it reactions almost
# never appeared: a chat where one person writes in a row never gave a reason to nod.
REACT_REASONS: frozenset[str] = frozenset(
    {"gate:dice", "gate:ambient_cooldown", "gate:not_live", "followup:name_no"}
)


class ReactionBotLike(Protocol):
    """A narrow protocol instead of ``aiogram.Bot``: tests fake it without aiogram
    (modelled on ``responder._BotLike``).

    ``reaction`` is typed as ``list[ReactionTypeUnion] | None`` (the type of the real
    ``Bot.set_message_reaction``, not only ``ReactionTypeEmoji``): ``list`` in Python
    is invariant in its parameter, so a narrower type here would fail the structural
    mypy check when a real ``aiogram.Bot`` is passed into ``Deps.bot``.
    """

    async def set_message_reaction(
        self,
        chat_id: int,
        message_id: int,
        reaction: list[ReactionTypeUnion] | None = None,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class ReactionState:
    """Snapshot of the reaction state at decision time (modelled on ``GateState``)."""

    last_reaction_at: int | None
    last_reaction_user_id: int | None
    count_today: int


def _parse_int_state(key: str, raw: str | None) -> int | None:
    """int(raw) with a None default; garbage in the value -> WARNING in the log, no crash
    (duplicates gate_state._parse_int_state, which is private to its module)."""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("reactions: garbage value %r for state key %r, using default", raw, key)
        return None


def pick_reaction(
    *,
    drop_reason: str,
    user_id: int,
    state: ReactionState,
    cfg: ReactionsConfig,
    rng: random.Random,
    now: int,
) -> str | None:
    """Decide whether to set a reaction on a cut message, and which one.

    Order of checks: enabled -> reason -> cap -> cooldown -> same user -> dice. The dice
    comes last so that ``rng`` is spent only when a reaction is possible at all:
    tests with a fixed seed are more stable that way (just like step eleven of the
    gate, which is the last).

    With ``cfg.semantic`` the "dice" step is skipped and a stub is returned: the first
    of ``cfg.emoji``. The model (``ReactionChooser``) is the filter instead of
    probability and it also picks the concrete emoji; all that matters here is
    "a reaction is possible in principle".
    """
    if not cfg.enabled:
        return None
    if drop_reason not in REACT_REASONS:
        return None
    if state.count_today >= cfg.daily_cap:
        return None
    if state.last_reaction_at is not None and state.last_reaction_at + cfg.cooldown_min * 60 > now:
        return None
    if state.last_reaction_user_id is not None and state.last_reaction_user_id == user_id:
        return None
    if cfg.semantic:
        return next(iter(cfg.emoji))
    if rng.random() >= cfg.probability:
        return None
    return rng.choice(list(cfg.emoji))


async def load_reaction_state(db: Database, tz: str, now: int) -> ReactionState:
    """State snapshot for pick_reaction. The counter's day follows persona.timezone,
    like mention_count/ambient_count (timeutil.day_key)."""
    last_reaction_at_raw = await db.get_state("last_reaction_at")
    last_reaction_user_id_raw = await db.get_state("last_reaction_user_id")
    count_key = day_key("reaction_count", now, tz)
    count_raw = await db.get_state(count_key)
    return ReactionState(
        last_reaction_at=_parse_int_state("last_reaction_at", last_reaction_at_raw),
        last_reaction_user_id=_parse_int_state("last_reaction_user_id", last_reaction_user_id_raw),
        count_today=_parse_int_state("reaction_count", count_raw) or 0,
    )


async def react(
    bot: ReactionBotLike,
    db: Database,
    *,
    chat_id: int,
    tg_message_id: int,
    user_id: int,
    emoji: str,
    tz: str,
    now: int,
) -> bool:
    """Set a reaction and update the state. A failure (reactions are forbidden in the chat,
    the message is already deleted) -> WARNING + filter_log react:error, state is left alone.

    Other exceptions are not caught: the handler's general except in bot.py catches them.
    """
    try:
        await bot.set_message_reaction(
            chat_id=chat_id,
            message_id=tg_message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        logger.warning("reaction failed on message %s: %s", tg_message_id, exc)
        await db.insert_filter_log(
            trigger_tg_message_id=tg_message_id,
            candidate_text=emoji,
            verdict="cut",
            stage="react",
            reason="react:error",
            shadow=False,
            created_at=now,
        )
        return False

    await db.set_state("last_reaction_at", str(now))
    await db.set_state("last_reaction_user_id", str(user_id))
    await db.increment_state(day_key("reaction_count", now, tz))
    await db.insert_filter_log(
        trigger_tg_message_id=tg_message_id,
        candidate_text=emoji,
        verdict="pass",
        stage="react",
        reason="react:sent",
        shadow=False,
        created_at=now,
    )
    logger.info("reaction %s on %s", emoji, tg_message_id)
    return True


@dataclass(frozen=True, slots=True)
class _ChoiceVerdict:
    """A parsed answer of the reaction-choice model: ``emoji`` None means "no reaction needed"."""

    emoji: str | None
    reason: str


def _parse_choice(raw: str) -> _ChoiceVerdict | None:
    """Parse the answer as strictly and as tolerant of wrappers as ``judge._parse_verdict``:
    strip ```json``` wrappers, find the first "{" and ``json.JSONDecoder.raw_decode`` from it.
    Any failure or a wrong key type -> None (meaning "no reaction")."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        emoji_value = data.get("emoji")
        if emoji_value is not None and not isinstance(emoji_value, str):
            return None
        # Luna sometimes writes null as a string: that is a refusal, not an emoji outside the list.
        if isinstance(emoji_value, str) and emoji_value.strip().lower() == "null":
            emoji_value = None

        reason_value = data.get("reason", "")
        if reason_value is None:
            reason_value = ""
        if not isinstance(reason_value, str):
            return None

        return _ChoiceVerdict(emoji=emoji_value or None, reason=reason_value)
    except Exception:
        # The model answer is untrusted external text: any parse failure means
        # "no reaction", not a crash.
        return None


class ReactionChooser:
    """A cheap model call: is a reaction fitting for this message and which one.

    The same approach as ``judge.Judge``/``followup.FollowupChecker``: a short prompt
    from a file (``prompts/reaction.txt``) with slots, data inside ``<<<CHAT ... >>>``,
    a strictly JSON answer. Counted by its own ``reaction_calls`` counter with its own
    cap ``reactions.semantic_daily_cap``: reactions must not eat the main model's
    call budget.

    A failure (empty model, ``LLMError``, invalid JSON, an emoji not from the list) gives
    ``None``: there is simply no reaction.
    """

    def __init__(
        self,
        llm: LLMClient,
        cfg_getter: Callable[[], Config],
        prompt_template: str,
        *,
        frag: Fragments,
    ) -> None:
        self._llm = llm
        self._cfg_getter = cfg_getter
        self._prompt_template = prompt_template
        self._frag = frag

    async def choose(
        self,
        *,
        text: str,
        display_name: str,
        context_rows: Sequence[MessageRow],
        allowed: Mapping[str, str],
        now: int,
        reply_to: ReplyTarget = None,
    ) -> str | None:
        """``allowed`` maps emoji -> when it fits; ``reply_to`` is the message the author
        replied to: ``None`` is not a reply, ``REPLY_UNAVAILABLE`` is a reply whose target
        is not visible."""
        cfg = self._cfg_getter()
        reactions_cfg = cfg.behaviour.reactions
        model = reactions_cfg.model or cfg.llm.judge_model
        if not model:
            return None
        if not allowed:
            return None

        context = _strip_fake_delimiters(render_context(list(context_rows))).strip()
        slot_values = {
            "context": context or self._frag.text("placeholders.context_empty"),
            "name": _strip_fake_delimiters(display_name).strip(),
            "text": _strip_fake_delimiters(text).strip(),
            "emoji": _render_palette(allowed),
            "reply_to": _render_reply_to(reply_to, self._frag),
        }
        system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._prompt_template)
        messages = [{"role": "system", "content": system}]

        try:
            result = await self._llm.call(
                messages,
                model=model,
                max_tokens=reactions_cfg.max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=reactions_cfg.semantic_daily_cap,
            )
        except LLMError as exc:
            logger.warning("reaction chooser llm error: reason=%s", exc.reason)
            return None

        verdict = _parse_choice(result.text)
        if verdict is None:
            logger.warning("reaction chooser: invalid answer: %r", result.text[:80])
            return None
        if verdict.emoji is None:
            logger.info("reaction chooser: none (%s)", verdict.reason[:_REASON_LOG_MAX_LEN])
            return None
        # The model may answer with a variation-selector form of a listed emoji: ignore it.
        chosen = verdict.emoji.replace("\ufe0f", "")
        if chosen not in {e.replace("\ufe0f", "") for e in allowed}:
            logger.warning("reaction chooser: emoji outside allowed list")
            return None

        # The model reason may retell the message: only its beginning goes to the log.
        logger.info(
            "reaction chooser: %s (%s)", verdict.emoji, verdict.reason[:_REASON_LOG_MAX_LEN]
        )
        return chosen


class ReactionScheduler:
    """Set a reaction not at once but after a pause, and re-check the conditions after it.

    ``schedule`` does not wait: it starts a background task (modelled on the debounce
    in ``responder.Responder``) and returns immediately, so the message handler does
    not hold up chat processing for minutes. The task sleeps ``delay_sec``, re-reads
    the state (during the pause the bot could have replied, set another reaction or
    hit the cap) and only then decides finally.

    Unfinished tasks are simply lost when the process stops: a reaction is an
    optional decoration, there is no point restoring it after a restart.
    """

    def __init__(
        self,
        *,
        bot: ReactionBotLike,
        db: Database,
        cfg_getter: Callable[[], Config],
        chooser: ReactionChooser | None,
        rng: random.Random,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._bot = bot
        self._db = db
        self._cfg_getter = cfg_getter
        self._chooser = chooser
        self._rng = rng
        self._clock = clock
        self._tasks: set[asyncio.Task[None]] = set()

    def schedule(
        self,
        *,
        chat_id: int,
        tg_message_id: int,
        user_id: int,
        text: str,
        display_name: str,
    ) -> None:
        task = asyncio.ensure_future(
            self._run(
                chat_id=chat_id,
                tg_message_id=tg_message_id,
                user_id=user_id,
                text=text,
                display_name=display_name,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(
        self,
        *,
        chat_id: int,
        tg_message_id: int,
        user_id: int,
        text: str,
        display_name: str,
    ) -> None:
        try:
            await self._run_inner(
                chat_id=chat_id,
                tg_message_id=tg_message_id,
                user_id=user_id,
                text=text,
                display_name=display_name,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # A reaction is a decoration: its failure must not bring down the handler (it has
            # already returned) or neighbouring tasks.
            logger.exception("reaction task failed on message %s", tg_message_id)

    async def _run_inner(
        self,
        *,
        chat_id: int,
        tg_message_id: int,
        user_id: int,
        text: str,
        display_name: str,
    ) -> None:
        lo, hi = self._cfg_getter().behaviour.reactions.delay_sec
        await asyncio.sleep(self._rng.uniform(lo, hi))

        # The config is re-read after the pause: in a minute or two the owner could have changed
        # anything via /set, and the current values apply, not those taken before the sleep.
        cfg = self._cfg_getter()
        reactions_cfg = cfg.behaviour.reactions
        tz = cfg.persona.timezone
        now = self._clock()

        state = await load_reaction_state(self._db, tz, now)
        if not self._recheck(state, reactions_cfg, user_id=user_id, now=now):
            await self._db.insert_filter_log(
                trigger_tg_message_id=tg_message_id,
                candidate_text=None,
                verdict="cut",
                stage="react",
                reason="react:recheck",
                shadow=False,
                created_at=now,
            )
            return

        if reactions_cfg.semantic and self._chooser is not None:
            rows = await self._db.recent_messages(chat_id, reactions_cfg.context_messages + 1)
            context_rows = [row for row in rows if row.tg_message_id != tg_message_id]
            recent = (
                await self._db.recent_reactions(reactions_cfg.max_repeat)
                if reactions_cfg.max_repeat > 0
                else []
            )
            emoji = await self._chooser.choose(
                text=text,
                display_name=display_name,
                context_rows=context_rows,
                allowed=palette_without_repeats(
                    reactions_cfg.emoji, recent, reactions_cfg.max_repeat
                ),
                now=now,
                reply_to=await self._reply_target(chat_id, tg_message_id),
            )
            if emoji is None:
                await self._db.insert_filter_log(
                    trigger_tg_message_id=tg_message_id,
                    candidate_text=text[:_CANDIDATE_TEXT_MAX_LEN],
                    verdict="cut",
                    stage="react",
                    reason="react:declined",
                    shadow=False,
                    created_at=now,
                )
                return
        else:
            fallback = self._fallback_emoji(reactions_cfg)
            if fallback is None:
                return
            emoji = fallback

        await react(
            self._bot,
            self._db,
            chat_id=chat_id,
            tg_message_id=tg_message_id,
            user_id=user_id,
            emoji=emoji,
            tz=tz,
            now=now,
        )

    async def _reply_target(self, chat_id: int, tg_message_id: int) -> ReplyTarget:
        """The message the author replied to: the message itself is already written by the handler
        before the gate, and the reply link is in it. A reply to a bot message does not get here:
        such a reply passes the gate as an address, and reactions are set only on cut messages."""
        message = await self._db.message_by_tg_id(chat_id, tg_message_id)
        if message is None or message.reply_to_tg_message_id is None:
            return None
        target = await self._db.message_by_tg_id(chat_id, message.reply_to_tg_message_id)
        return REPLY_UNAVAILABLE if target is None else target

    @staticmethod
    def _recheck(state: ReactionState, cfg: ReactionsConfig, *, user_id: int, now: int) -> bool:
        """The same deterministic checks as in ``pick_reaction``, but after the
        pause and without the dice: during the sleep the bot could have set a reaction for the
        same person, hit the daily cap or started a cooldown."""
        if state.count_today >= cfg.daily_cap:
            return False
        if (
            state.last_reaction_at is not None
            and state.last_reaction_at + cfg.cooldown_min * 60 > now
        ):
            return False
        # The same person in a row: the reaction would look intrusive.
        return state.last_reaction_user_id != user_id

    def _fallback_emoji(self, cfg: ReactionsConfig) -> str | None:
        """An emoji without the model. With ``semantic: false`` the dice was already thrown by
        ``pick_reaction``, so only the emoji remains to be chosen. With ``semantic: true``
        and no chooser (LLM not configured) ``pick_reaction`` skipped the dice, so
        it is thrown here: otherwise a reaction would be set on every suitable cut."""
        if cfg.semantic and self._rng.random() >= cfg.probability:
            return None
        return self._rng.choice(list(cfg.emoji))

    async def shutdown(self) -> None:
        """Cancel all waiting tasks. Unfinished reactions are lost, which is fine."""
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
