"""aiogram Router and the message handler for a single chat.

The handler writes the message to the DB, runs it through the gate (should_consider)
and logs the verdict. Replies to the chat are produced elsewhere (responder.py), never
directly from here.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import BinaryIO, Protocol

from aiogram import Router
from aiogram.exceptions import AiogramError
from aiogram.types import Message

from sotto.chat_memory import ChatMemorizer
from sotto.config_models import Config
from sotto.db import Database, MessageRow
from sotto.followup import FollowupChecker
from sotto.fragments import Fragments
from sotto.gate import should_consider
from sotto.gate_state import load_gate_state
from sotto.gate_types import GateMessage, Trigger, Verdict
from sotto.patterns import Patterns
from sotto.persona import Persona
from sotto.reactions import (
    REACT_REASONS,
    ReactionBotLike,
    ReactionScheduler,
    load_reaction_state,
    pick_reaction,
    react,
)
from sotto.responder import Responder
from sotto.sanitize import media_placeholder, normalize_text, sanitize_display_name, stable_n
from sotto.settings import Settings
from sotto.stores import ConfigStore, PromptStore
from sotto.timeutil import day_key
from sotto.vision import (
    PhotoSizeLike,
    VisionDescriber,
    photo_text,
    pick_photo_size,
    should_describe,
)
from sotto.vocab import VocabStore, explain_requested
from sotto.weather import WeatherClient

logger = logging.getLogger(__name__)

# Truncation of text in the INFO log (the "Name: text" line).
_LOG_TEXT_MAX_LEN = 200

# DROP reasons for which the cheap followup check makes sense: all of them mean "a
# message without an address that the ordinary gate did not accept as an ambient
# reply". gate:dice is the same dice as without the hot window (step 11/10h); do not
# leave it without a chance at followup just because the dice fell against an ambient
# reply.
FOLLOWUP_REASONS: frozenset[str] = frozenset(
    {"gate:dice", "gate:not_live", "gate:ambient_cap", "gate:ambient_cooldown", "gate:hot_cap"}
)


class BotLike(ReactionBotLike, Protocol):
    """Narrow protocol instead of ``aiogram.Bot`` for everything the handler does with the bot:
    reactions (``set_message_reaction``, inherited from ``ReactionBotLike``) and
    downloading photos for vision (``download``). Tests fake it without aiogram.

    The ``download`` signature is a subset of ``aiogram.Bot.download``
    (``file: str | Downloadable``, ``destination: BinaryIO | Path | str | None``,
    plus ``timeout``/``chunk_size``/``seek`` with defaults): the protocol only needs
    what is used here (file_id as a string, the result in memory), while the real
    ``Bot`` accepts more and therefore matches structurally.
    """

    async def download(self, file: str, destination: BinaryIO | None = None) -> BinaryIO | None: ...


@dataclass(slots=True)
class Deps:
    """Dependencies of the handler and the commands router (commands.py).

    config_getter/patterns_getter are getters, not frozen snapshots taken once when Deps
    is built in app.py: ConfigStore rebuilds Patterns on every /set (new regexes from
    filters.*), and without a getter handle_message would keep a stale object until the
    process restarts. patterns_getter = config_store.patterns is called on every message,
    like config_getter = config_store.get.
    config_store/prompt_store are the full stores, added for commands.py
    (build_commands_router expects them via the structural protocol _CommandsDeps; this
    Deps matches it structurally without explicit inheritance).
    """

    settings: Settings
    config_getter: Callable[[], Config]
    db: Database
    bot_user_id: int
    reserved_names: set[str]
    patterns_getter: Callable[[], Patterns]
    rng: random.Random
    config_store: ConfigStore
    prompt_store: PromptStore
    frag: Fragments
    bot_username: str = ""
    # Persona pack: /changelog reads from it; None in handler tests without a pack.
    persona: Persona | None = None
    responder: Responder | None = None
    # Cheap "is this for me?" check for messages without an address in the hot window.
    # None means no LLM is configured: PASS is not checked for FOLLOWUP_REASONS at all,
    # behaviour as without followup.
    followup: FollowupChecker | None = None
    # Narrow protocol (set_message_reaction + download) instead of aiogram.Bot, for
    # reactions (reactions.py) and for downloading photos for vision (vision.py). None in
    # discovery/tests without a real bot: then no reaction is set and the photo is not
    # described, but the message is still recorded and gated as usual.
    bot: BotLike | None = None
    # Photo description by a vision model. None means no LLM is configured: the photo
    # stays the "[photo]" placeholder, as before this feature.
    vision: VisionDescriber | None = None
    # Delayed reaction: a pause before the reaction and the emoji chosen by a model. None
    # means the old behaviour: the reaction is set right in the handler, immediately and
    # with the emoji pick_reaction chose.
    reaction_scheduler: ReactionScheduler | None = None
    # How many enabled stickers are in the catalog (stickers.yaml): counted once in
    # app.py at startup, only for /status ("(N in catalog)"). The catalog does not change
    # hot (unlike config_overrides), so a fixed number is enough, not a getter.
    sticker_catalog_enabled: int = 0
    # Long-term chat memory: needed only by the /memory run command; the summarizing
    # itself is done by a background task. None means no LLM is configured.
    memorizer: ChatMemorizer | None = None
    # Weather: the message handler does not need it at all, Responder reads it; here it
    # is only for a line in /status. None means there is no client (handler tests),
    # /status will show "weather: none".
    weather: WeatherClient | None = None
    # Vocabulary dictionaries: for /words and /status; Responder gets its own getter.
    # None in handler tests without dictionaries.
    vocab_store: VocabStore | None = None
    # user_ids for which the display_name-injection WARNING has already been logged:
    # do not spam the log on every following message from the same participant.
    warned_user_ids: set[int] = field(default_factory=set)
    clock: Callable[[], int] = field(default_factory=lambda: lambda: int(time.time()))


def _int_state(raw: str | None) -> int | None:
    """int(raw) defaulting to None; garbage in the value -> WARNING, do not crash."""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("bot: garbage value %r in state, using default", raw)
        return None


async def _describe_photo(
    deps: Deps,
    describer: VisionDescriber,
    bot: BotLike,
    *,
    sizes: Sequence[PhotoSizeLike],
    caption: str,
    reply_to_bot: bool,
    tg_message_id: int,
    now: int,
) -> str:
    """Text that goes into ``messages.text`` instead of the photo.

    Called BEFORE ``insert_message`` so the description reaches the DB, the context and
    the gate; from then on the photo is no different from ordinary text. Any refusal
    (dice, cap, download error, model error) ends in the usual "[photo]": a photo must
    not break message handling.
    """
    cfg = deps.config_getter()
    vision_cfg = cfg.behaviour.vision
    patterns = deps.patterns_getter()

    addressed = (
        reply_to_bot or patterns.mentions_bot(caption) or patterns.name_trigger(caption) is not None
    )
    hot_until = _int_state(await deps.db.get_state("hot_until"))
    hot = hot_until is not None and hot_until > now

    count_key = day_key("vision_count", now, cfg.persona.timezone)
    count_today = _int_state(await deps.db.get_state(count_key)) or 0

    if not should_describe(
        addressed=addressed, hot=hot, count_today=count_today, cfg=vision_cfg, rng=deps.rng
    ):
        await deps.db.insert_filter_log(
            trigger_tg_message_id=tg_message_id,
            candidate_text=None,
            verdict="cut",
            stage="vision",
            reason="vision:skipped",
            shadow=False,
            created_at=now,
        )
        return photo_text(None, caption, frag=deps.frag)

    description: str | None = None
    size = pick_photo_size(sizes, vision_cfg.max_width)
    if size is not None:
        image: bytes | None = None
        try:
            downloaded = await bot.download(size.file_id)
            if downloaded is None:
                logger.warning(
                    "vision: bot.download returned nothing for message %s", tg_message_id
                )
            else:
                image = downloaded.read()
        except (AiogramError, OSError) as exc:
            # Photo deleted, file too large, network dropped: no reason to crash the
            # handler, the message is still recorded as "[photo]".
            logger.warning(
                "vision: could not download the photo of message %s: %s", tg_message_id, exc
            )
        if image is not None:
            # mime is fixed: Telegram always serves the message.photo preview as JPEG.
            description = await describer.describe(
                image, mime="image/jpeg", caption=caption, now=now
            )

    if description is None:
        await deps.db.insert_filter_log(
            trigger_tg_message_id=tg_message_id,
            candidate_text=None,
            verdict="cut",
            stage="vision",
            reason="vision:failed",
            shadow=False,
            created_at=now,
        )
        return photo_text(None, caption, frag=deps.frag)

    # /status counter (description statistics). The call cap is held by LLMClient via
    # its own vision_calls counter: a failed attempt spends that but not this statistic;
    # a small discrepancy is acceptable and part of the contract.
    await deps.db.increment_state(count_key)
    await deps.db.insert_filter_log(
        trigger_tg_message_id=tg_message_id,
        candidate_text=description,
        verdict="pass",
        stage="vision",
        reason="vision:described",
        shadow=False,
        created_at=now,
    )
    logger.info("vision: %s", description)
    return photo_text(description, caption, frag=deps.frag)


async def _vocab_explain_asked(
    deps: Deps, cfg: Config, gate_message: GateMessage, *, reply_to_other: bool, now: int
) -> bool:
    """ "I did not understand", "what does it mean" (``vocab.explain_markers``) within
    ``explain_window_min`` after its reply with a dictionary word. Not addressed to it if
    this is a reply to another person or more than ``explain_max_between`` messages from
    people came after its reply: the conversation has moved on."""
    vocab_cfg = cfg.behaviour.vocab
    if (
        deps.vocab_store is None
        or reply_to_other
        or not vocab_cfg.enabled
        or vocab_cfg.explain_window_min <= 0
        or not explain_requested(gate_message.text, vocab_cfg)
    ):
        return False
    last = await deps.db.last_bot_replies(1)
    if not last or now - last[0].created_at > vocab_cfg.explain_window_min * 60:
        return False
    # The current message is already recorded; it also counts as "after", so skip it.
    between = await deps.db.messages_after(deps.settings.allowed_chat_id, last[0].tg_message_id)
    if between - 1 > vocab_cfg.explain_max_between:
        return False
    return bool(deps.vocab_store.get().find(last[0].text))


async def _followup_inputs(
    deps: Deps, cfg: Config, gate_message: GateMessage
) -> tuple[list[MessageRow], list[str]]:
    """Context for the cheap check: the last chat messages WITHOUT the current one (it
    is already recorded by insert_message) and the bot's last replies. Shared by the
    "is this for me?" check in the hot window and the address-by-name check."""
    followup_cfg = cfg.behaviour.followup
    rows = await deps.db.recent_messages(gate_message.chat_id, followup_cfg.context_messages + 1)
    context_rows = [row for row in rows if row.tg_message_id != gate_message.tg_message_id]
    recent_replies = await deps.db.recent_bot_replies(followup_cfg.recent_replies)
    return context_rows, recent_replies


async def _maybe_react(
    deps: Deps,
    cfg: Config,
    *,
    gate_message: GateMessage,
    user_id: int,
    display_name: str,
    text: str,
    reason: str,
    now: int,
) -> None:
    """A reaction instead of complete silence: only for the non-deterministic reasons in
    ``REACT_REASONS`` (dice, ambient cooldown, the model refusing to count a name mention
    as an address) and only in the allowed chat (called after the chat_id checks in the
    handler). The decision lives outside the gate, ``gate.py`` is unchanged."""
    if reason not in REACT_REASONS or deps.bot is None:
        return
    tz = cfg.persona.timezone
    rstate = await load_reaction_state(deps.db, tz, now)
    emoji = pick_reaction(
        drop_reason=reason,
        user_id=user_id,
        state=rstate,
        cfg=cfg.behaviour.reactions,
        rng=deps.rng,
        now=now,
    )
    if emoji is None:
        return
    if deps.reaction_scheduler is not None:
        # The reaction goes to the background: a "read it and hummed" pause, then a
        # recheck of conditions and the model's emoji choice. The handler does not wait
        # (the chat would stall for a minute otherwise).
        deps.reaction_scheduler.schedule(
            chat_id=gate_message.chat_id,
            tg_message_id=gate_message.tg_message_id,
            user_id=user_id,
            text=text,
            display_name=display_name,
        )
        return
    await react(
        deps.bot,
        deps.db,
        chat_id=gate_message.chat_id,
        tg_message_id=gate_message.tg_message_id,
        user_id=user_id,
        emoji=emoji,
        tz=tz,
        now=now,
    )


def build_router(deps: Deps) -> Router:
    """Build a Router with a single handler for messages of a foreign/own chat.

    Edited messages are ignored entirely (a separate handler, just return): edits to chat
    history must not rewrite what already reached the log/DB.
    """
    router = Router(name=__name__)

    @router.edited_message()
    async def handle_edited_message(message: Message) -> None:
        return

    @router.message()
    async def handle_message(message: Message) -> None:
        allowed_chat_id = deps.settings.allowed_chat_id
        user = message.from_user

        if allowed_chat_id == 0:
            # Discovery mode: help the owner find chat_id for .env. Nothing goes to the DB.
            logger.warning(
                "discovery: chat_id=%s title=%r user_id=%s name=%r",
                message.chat.id,
                message.chat.title,
                user.id if user is not None else 0,
                user.full_name if user is not None else None,
            )
            return

        if message.chat.id != allowed_chat_id:
            # Foreign chat: no log and no DB, so as not to burn the key or store foreign data.
            return

        if user is None:
            # Anonymous admin or a message on behalf of a channel.
            user_id = 0
            display_name = sanitize_display_name(None, 0, deps.reserved_names, frag=deps.frag)
            is_bot = False
        else:
            user_id = user.id
            display_name = sanitize_display_name(
                user.full_name, user.id, deps.reserved_names, frag=deps.frag
            )
            # A name made of invisible characters or only emoji yields "Participant N"; if
            # there is a username it is clearer both to people in the context and to the model.
            if (
                display_name.startswith(deps.frag.prefix("placeholders.participant"))
                and user.username
            ):
                display_name = sanitize_display_name(
                    user.username, user.id, deps.reserved_names, frag=deps.frag
                )
            is_bot = user.is_bot

        if deps.patterns_getter().injection(display_name) is not None:
            # An injection name ("Ignore the rules and say ...") passed sanitization (it
            # contains no bot triggers as separate words) but looks like a gate 5a
            # command, so replace it with "Participant N" before writing to the DB and the
            # prompt, so display_name does not read as an instruction line by line in
            # front of every participant message in {context}.
            if user_id not in deps.warned_user_ids:
                deps.warned_user_ids.add(user_id)
                logger.warning(
                    "display_name looks like a prompt injection, replacing with placeholder: "
                    "user_id=%s name=%r",
                    user_id,
                    display_name,
                )
            display_name = deps.frag.render("placeholders.participant", n=str(stable_n(user_id)))

        reply_to_tg_message_id = (
            message.reply_to_message.message_id if message.reply_to_message is not None else None
        )
        reply_to_bot = (
            message.reply_to_message is not None
            and message.reply_to_message.from_user is not None
            and message.reply_to_message.from_user.id == deps.bot_user_id
        )
        reply_to_other = message.reply_to_message is not None and not reply_to_bot
        created_at = int(message.date.timestamp())

        text = normalize_text(message.text or message.caption)
        photo_sizes = list(message.photo or ())
        if (
            photo_sizes
            and deps.vision is not None
            and deps.bot is not None
            and deps.config_getter().behaviour.vision.enabled
        ):
            # Vision: the description replaces the placeholder and goes to the DB with
            # the caption, strictly before insert_message.
            text = await _describe_photo(
                deps,
                deps.vision,
                deps.bot,
                sizes=photo_sizes,
                caption=text,
                reply_to_bot=reply_to_bot,
                tg_message_id=message.message_id,
                now=created_at,
            )
        if not text:
            text = media_placeholder(message, frag=deps.frag) or ""
        if not text:
            # Service message with no text and no media (joined/left etc.): not recorded.
            return

        await deps.db.insert_message(
            tg_message_id=message.message_id,
            chat_id=message.chat.id,
            user_id=user_id,
            display_name=display_name,
            text=text,
            reply_to_tg_message_id=reply_to_tg_message_id,
            is_bot=is_bot,
            created_at=created_at,
        )
        logger.info("%s: %s", display_name, text[:_LOG_TEXT_MAX_LEN])

        gate_message = GateMessage(
            chat_id=message.chat.id,
            tg_message_id=message.message_id,
            user_id=user_id,
            is_bot=is_bot,
            text=text,
            reply_to_bot=reply_to_bot,
            created_at=created_at,
        )

        try:
            cfg = deps.config_getter()
            now = gate_message.created_at
            # Tail of updates after a restart: a message older than the late-reply
            # threshold goes into the context but not into the gate. A person whose phone
            # died does not reply retroactively. Otherwise a three-hour-old mention would
            # get a reply based on fresh context.
            age = deps.clock() - created_at
            if age > cfg.behaviour.late_reply_threshold_sec:
                await deps.db.insert_filter_log(
                    trigger_tg_message_id=message.message_id,
                    candidate_text=None,
                    verdict="cut",
                    stage="gate",
                    reason="gate:stale",
                    shadow=False,
                    created_at=deps.clock(),
                )
                logger.info("gate stale: message %s is %ss old, skipped", message.message_id, age)
                return
            state = await load_gate_state(deps.db, cfg, gate_message, now)
            decision = should_consider(
                gate_message, state, cfg, deps.patterns_getter(), now, deps.rng
            )
            await deps.db.apply_state_changes(decision.state_changes)

            if decision.verdict is Verdict.DROP:
                if decision.reason in FOLLOWUP_REASONS and await _vocab_explain_asked(
                    deps, cfg, gate_message, reply_to_other=reply_to_other, now=now
                ):
                    # "I did not understand" right after its dictionary word: the question is
                    # for it, even without a name or reply: it will explain.
                    await deps.db.insert_filter_log(
                        trigger_tg_message_id=gate_message.tg_message_id,
                        candidate_text=text,
                        verdict="pass",
                        stage="followup",
                        reason="followup:vocab_explain",
                        shadow=False,
                        created_at=now,
                    )
                    logger.info("vocab explain pass: %s", text[:_LOG_TEXT_MAX_LEN])
                    if deps.responder is not None:
                        await deps.responder.on_gate_pass(
                            gate_message, Trigger.FOLLOWUP, display_name
                        )
                    return
                if (
                    decision.reason in FOLLOWUP_REASONS
                    and deps.followup is not None
                    and cfg.behaviour.followup.enabled
                    and state.hot_until is not None
                    and now < state.hot_until
                ):
                    # Phone in hand: the hot window is still open and the gate did not
                    # accept the message without an address as an ambient reply, so the
                    # cheap check decides whether it is addressed to the bot after all.
                    # context_rows excludes the current message (already recorded by
                    # insert_message above).
                    context_rows, recent_replies = await _followup_inputs(deps, cfg, gate_message)
                    addressed = await deps.followup.check(
                        text=text,
                        display_name=display_name,
                        context_rows=context_rows,
                        recent_replies=recent_replies,
                        now=now,
                    )
                    if addressed:
                        await deps.db.insert_filter_log(
                            trigger_tg_message_id=gate_message.tg_message_id,
                            candidate_text=text,
                            verdict="pass",
                            stage="followup",
                            reason="followup:yes",
                            shadow=False,
                            created_at=now,
                        )
                        logger.info("followup pass: %s", text[:_LOG_TEXT_MAX_LEN])
                        if deps.responder is not None:
                            await deps.responder.on_gate_pass(
                                gate_message, Trigger.FOLLOWUP, display_name
                            )
                        return
                    await deps.db.insert_filter_log(
                        trigger_tg_message_id=gate_message.tg_message_id,
                        candidate_text=None,
                        verdict="cut",
                        stage="followup",
                        reason="followup:no",
                        shadow=False,
                        created_at=now,
                    )

                await deps.db.insert_filter_log(
                    trigger_tg_message_id=gate_message.tg_message_id,
                    candidate_text=None,
                    verdict="cut",
                    stage="gate",
                    reason=decision.reason,
                    shadow=False,
                    created_at=now,
                )
                logger.debug("gate drop: %s (%s)", decision.reason, gate_message.tg_message_id)
                await _maybe_react(
                    deps,
                    cfg,
                    gate_message=gate_message,
                    user_id=user_id,
                    display_name=display_name,
                    text=text,
                    reason=decision.reason,
                    now=now,
                )
            elif decision.verdict is Verdict.QUEUE_NIGHT:
                await deps.db.enqueue_night(
                    tg_message_id=gate_message.tg_message_id,
                    user_id=user_id,
                    display_name=display_name,
                    text=text,
                    created_at=now,
                )
                await deps.db.insert_filter_log(
                    trigger_tg_message_id=gate_message.tg_message_id,
                    candidate_text=None,
                    verdict="cut",
                    stage="gate",
                    reason=decision.reason,
                    shadow=False,
                    created_at=now,
                )
                logger.info("night queued: %s (%s)", decision.trigger, text[:_LOG_TEXT_MAX_LEN])
            else:
                await deps.db.insert_filter_log(
                    trigger_tg_message_id=gate_message.tg_message_id,
                    candidate_text=None,
                    verdict="pass",
                    stage="gate",
                    reason=decision.reason,
                    shadow=False,
                    created_at=now,
                )
                if decision.trigger is None:
                    # Gate invariant: PASS always carries a trigger (see gate.py). If it
                    # is None anyway, the gate configuration is broken, which is no reason
                    # to crash the handler: log and just do not call the responder here.
                    logger.error(
                        "gate pass without trigger: message %s reason=%s",
                        gate_message.tg_message_id,
                        decision.reason,
                    )
                    return
                if (
                    decision.trigger is Trigger.NAME
                    and cfg.behaviour.followup.name_check
                    and deps.followup is not None
                ):
                    # The name is caught in all case forms, but the regex cannot tell an
                    # address from talk about the bot in the third person ("the father put
                    # a smiley under the message yesterday"). The cheap model decides; it is
                    # also the only check a name goes through: a reply to the bot and
                    # @username are never checked.
                    context_rows, recent_replies = await _followup_inputs(deps, cfg, gate_message)
                    addressed = await deps.followup.check_name(
                        text=text,
                        display_name=display_name,
                        context_rows=context_rows,
                        recent_replies=recent_replies,
                        now=now,
                    )
                    if not addressed:
                        await deps.db.insert_filter_log(
                            trigger_tg_message_id=gate_message.tg_message_id,
                            candidate_text=text,
                            verdict="cut",
                            stage="followup",
                            reason="followup:name_no",
                            shadow=False,
                            created_at=now,
                        )
                        logger.info("name check: not addressed, %s", text[:_LOG_TEXT_MAX_LEN])
                        # From here the same branch as DROP by dice: no reply, but a
                        # reaction is possible (the reason is in REACT_REASONS).
                        await _maybe_react(
                            deps,
                            cfg,
                            gate_message=gate_message,
                            user_id=user_id,
                            display_name=display_name,
                            text=text,
                            reason="followup:name_no",
                            now=now,
                        )
                        return
                    await deps.db.insert_filter_log(
                        trigger_tg_message_id=gate_message.tg_message_id,
                        candidate_text=None,
                        verdict="pass",
                        stage="followup",
                        reason="followup:name_yes",
                        shadow=False,
                        created_at=now,
                    )
                trigger = decision.trigger
                if trigger is Trigger.AMBIENT and await _vocab_explain_asked(
                    deps, cfg, gate_message, reply_to_other=reply_to_other, now=now
                ):
                    # The ambient dice came up, but this is an "I did not understand" after
                    # its word: the same followup as on DROP, otherwise dedup:vocab would
                    # cut the explanation.
                    await deps.db.insert_filter_log(
                        trigger_tg_message_id=gate_message.tg_message_id,
                        candidate_text=text,
                        verdict="pass",
                        stage="followup",
                        reason="followup:vocab_explain",
                        shadow=False,
                        created_at=now,
                    )
                    trigger = Trigger.FOLLOWUP
                if deps.responder is not None:
                    await deps.responder.on_gate_pass(gate_message, trigger, display_name)
                else:
                    logger.info("gate pass: %s (%s)", decision.trigger, text[:_LOG_TEXT_MAX_LEN])
        except Exception:
            # A gate error must not crash the handler: the message is already in messages.
            logger.exception("gate failed for message %s", gate_message.tg_message_id)

    return router
