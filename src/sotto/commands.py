"""Router with commands for managing the bot from Telegram.

`build_commands_router` returns a Router with a single handler for all messages that
start with "/". The integrator (app.py) must include this router in the Dispatcher
FIRST, before the main message router (sotto.bot.build_router), otherwise commands in
the allowed chat would reach the gate and be recorded in messages as ordinary text.
aiogram stops propagation at the first handler that returns something other than None
or skip_this_handler, so the same message will not reach the main router a second time.

The handler decides itself who may do what: the "foreign chat" gate from bot.py does
not apply here, the commands router comes before it.

To avoid depending on the concrete store classes (ConfigStore/PromptStore), structural
Protocols are defined here (`_ConfigStoreLike`, `_PromptStoreLike`, `_DbLike` and the
helper row protocols) with exactly the methods the commands need. The real classes
(ConfigStore/PromptStore/Database) satisfy them structurally, without explicit
inheritance.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, BinaryIO, Protocol

from aiogram import F, Router
from aiogram.types import Message, User

from sotto import __version__
from sotto.changelog import Release, chat_releases, engine_changelog_text, owner_releases
from sotto.chat_memory import format_period
from sotto.config import KeyInfo
from sotto.config_models import Config
from sotto.db import (
    ChatMemoryRow,
    JokeRow,
    LifeEventRow,
    PeopleThreadRow,
    SelfFactRow,
)
from sotto.errors import OwnerError
from sotto.few_shot import FewShot
from sotto.fragments import Fragments
from sotto.i18n import Catalog, engine_catalog
from sotto.motifs import render_self_share, self_share
from sotto.patterns import Patterns
from sotto.persona import Persona
from sotto.sanitize import normalize_text, sanitize_display_name
from sotto.settings import Settings
from sotto.timeutil import day_key, day_start, local_dt
from sotto.weather import Place, Weather, describe, format_temp

if TYPE_CHECKING:
    # For annotations only: "from __future__ import annotations" makes them strings, so
    # this import does not run at runtime. A direct (non-TYPE_CHECKING) import of
    # sotto.responder is undesirable here: it drags the aiogram chain into commands.py
    # and risks a circular import (bot.py already imports both responder and commands).
    from sotto.db import VocabDictRow
    from sotto.responder import PreviewOutcome, SendOutcome
    from sotto.vocab import Vocab, VocabDict

logger = logging.getLogger(__name__)

# Length cap for replies to the owner ("no longer than 3500 characters").
_MAX_REPLY_LEN = 3500
# Prompt body preview in /prompt without "full".
_PROMPT_PREVIEW_LEN = 1500
# How many summaries /memory list shows (the reply is still cut to 3500 characters).
_MEMORY_LIST_LIMIT = 20
# How many characters of freshly created summaries to show in the /memory run reply.
_MEMORY_RUN_PREVIEW_LEN = 1500

# /status: window of the "about itself" metric, in days.
_SELF_SHARE_DAYS = 3

# /diary: list window ("for the week").
_DIARY_LIST_DAYS = 7

# /jokes and /people: cap on the jokes list
# (the reply is still cut to 3500 characters).
_JOKES_LIST_LIMIT = 1000

# /words: cap on the dictionary file size.
_WORDS_MAX_FILE_BYTES = 512 * 1024


class _MessageRowLike(Protocol):
    """The subset of db.MessageRow needed by /ex add.

    Members are declared with ``@property`` (not as plain attributes) because
    db.MessageRow is a frozen dataclass: its fields are read-only, while a plain
    attribute in a Protocol requires settable compatibility (invariant both ways) and
    does not match a read-only field. ``@property`` in a Protocol is exactly a read-only
    member, which a frozen dataclass satisfies structurally.
    """

    @property
    def display_name(self) -> str | None: ...
    @property
    def text(self) -> str | None: ...


class _BotReplyRowLike(Protocol):
    """The subset of db.BotReplyRow needed by /ex add and /last. See _MessageRowLike."""

    @property
    def trigger(self) -> str: ...
    @property
    def trigger_tg_message_id(self) -> int | None: ...
    @property
    def text(self) -> str: ...
    @property
    def prompt_version(self) -> int: ...
    @property
    def few_shot_version(self) -> int: ...
    @property
    def delay_sec(self) -> int: ...
    @property
    def created_at(self) -> int: ...


class _VersionRowLike(Protocol):
    """The subset of db.VersionRow needed by /prompt (to count "total M"). See _MessageRowLike."""

    @property
    def version(self) -> int: ...


class _DbLike(Protocol):
    """Database methods used by commands.py.

    Only signatures per the contract are declared here, without importing db.py (some
    of the methods, e.g. add_mute/remove_mute/last_bot_replies/audit_stop/
    message_by_tg_id/bot_reply_by_tg_id and the prompt versions, live elsewhere).
    """

    async def get_state(self, key: str) -> str | None: ...
    async def set_state(self, key: str, value: str) -> None: ...
    async def delete_state(self, key: str) -> None: ...
    async def audit_stop(
        self, key: str, changed_by: int, now: int, new_value: str | None = None
    ) -> None: ...
    async def add_mute(self, user_id: int, display_name: str, muted_by: int, now: int) -> None: ...
    async def remove_mute(self, user_id: int) -> bool: ...
    # Sequence (not list) in the return type: list is invariant in its parameter, so
    # list[_BotReplyRowLike] would not accept Database.last_bot_replies(), whose real
    # type is list[BotReplyRow] (BotReplyRow is a subtype of _BotReplyRowLike, but
    # List[BotReplyRow] is still not a subtype of List[_BotReplyRowLike]). Sequence
    # is declared covariant in its element, so it does not have this problem.
    async def last_bot_replies(self, n: int) -> Sequence[_BotReplyRowLike]: ...
    async def message_by_tg_id(
        self, chat_id: int, tg_message_id: int
    ) -> _MessageRowLike | None: ...
    async def bot_reply_by_tg_id(self, tg_message_id: int) -> _BotReplyRowLike | None: ...
    async def filter_log_summary(self, since: int) -> list[tuple[str, int]]: ...
    # -- presence cap ----------------------------------------------------------
    async def count_messages_today(self, chat_id: int, day_start: int) -> int: ...
    async def count_bot_replies_today(self, day_start: int) -> int: ...
    async def load_pending(self) -> Sequence[object]: ...
    async def night_unanswered(self) -> Sequence[object]: ...
    async def prompt_versions(self) -> Sequence[_VersionRowLike]: ...
    # -- life events (/life, /say) ---------------------------------------------
    async def insert_life_event(self, *, text: str, created_at: int) -> int: ...
    async def life_events(self) -> Sequence[LifeEventRow]: ...
    async def life_event(self, event_id: int) -> LifeEventRow | None: ...
    async def delete_life_event(self, event_id: int) -> bool: ...
    # -- long-term chat memory (/memory) ---------------------------------------
    async def chat_memories(self, limit: int) -> Sequence[ChatMemoryRow]: ...
    async def chat_memory(self, memory_id: int) -> ChatMemoryRow | None: ...
    async def chat_memory_count(self) -> int: ...
    async def delete_chat_memory(self, memory_id: int) -> bool: ...
    async def last_chat_memory_end(self) -> int | None: ...
    # -- daily diary (/diary) --------------------------------------------------
    async def self_facts_since(self, since: int) -> Sequence[SelfFactRow]: ...
    async def bot_replies_between(self, start: int, end: int) -> Sequence[_BotReplyRowLike]: ...
    async def delete_self_fact(self, fact_id: int) -> bool: ...
    # -- chat jokes and people's stories (/jokes, /people) ---------------------
    async def jokes(self, limit: int) -> Sequence[JokeRow]: ...
    async def delete_joke(self, joke_id: int) -> bool: ...
    async def people_threads_all(self) -> Sequence[PeopleThreadRow]: ...
    async def delete_people_thread(self, thread_id: int) -> bool: ...
    # -- /ask: the owner's name as the chat context sees it ---------------------
    async def display_names(self, user_ids: Sequence[int]) -> dict[int, str]: ...


class _ResponderLike(Protocol):
    """The subset of Responder needed by /life and /say.

    Only these two methods, not the full Responder. SendOutcome in annotations comes
    from the TYPE_CHECKING import above (see its comment on why commands.py does not
    import sotto.responder directly)."""

    async def announce_life(self, event: LifeEventRow) -> SendOutcome: ...
    async def say(self, text: str) -> SendOutcome: ...
    async def preview(self, text: str, display_name: str, user_id: int = 0) -> PreviewOutcome: ...


class _ChatMemorizerLike(Protocol):
    """The subset of chat_memory.ChatMemorizer needed by /memory run.

    Only ``run_due``: listing and deletion go straight through the DB. None means the
    LLM part is off (no key): then /memory run says so and starts nothing."""

    async def run_due(self, *, now: int) -> Sequence[ChatMemoryRow]: ...


class _VocabStoreLike(Protocol):
    """The subset of vocab.VocabStore needed by /words and /status."""

    def get(self) -> Vocab: ...
    async def put(self, raw: str, now: int) -> VocabDict: ...
    async def set_enabled(self, name: str, enabled: bool) -> bool: ...
    async def remove(self, name: str) -> bool: ...
    async def dicts(self) -> list[tuple[VocabDictRow, VocabDict | None]]: ...


class _DownloaderLike(Protocol):
    """Download a file by file_id: a subset of aiogram.Bot.download (like bot.BotLike)."""

    async def download(self, file: str, destination: BinaryIO | None = None) -> BinaryIO | None: ...


class _ConfigStoreLike(Protocol):
    """The stores.ConfigStore methods used by commands.py."""

    def get(self) -> Config: ...
    async def set(
        self, key: str, raw_value: str, changed_by: int, now: int
    ) -> tuple[str | None, str]: ...
    async def unset(self, key: str, changed_by: int, now: int) -> str | None: ...
    def flat(self) -> list[tuple[str, str, bool]]: ...
    def describe(self, key: str) -> KeyInfo | None: ...
    def patterns(self) -> Patterns: ...


class _PromptStoreLike(Protocol):
    """The stores.PromptStore methods used by commands.py."""

    def system_prompt(self) -> str: ...
    def prompt_version(self) -> int: ...
    def few_shot_version(self) -> int: ...
    async def rollback_prompt(self, version: int) -> bool: ...
    async def add_example(self, name: str, user: str, text: str, now: int) -> int: ...
    async def remove_example(self, index: int, now: int) -> int: ...
    def examples(self) -> list[FewShot]: ...


class _WeatherLike(Protocol):
    """The subset of WeatherClient needed by /status: one weather snapshot. Like the
    other protocols here it is structural, so tests do not have to build a real httpx
    client."""

    async def get(self, place: Place | None = None) -> Weather | None: ...


class _CommandsDeps(Protocol):
    """Dependencies of build_commands_router. bot.Deps is later extended with these fields.

    responder is used by /life and /say (announce_life/say) through _ResponderLike, a
    structural protocol rather than the concrete Responder type, so commands.py does not
    pull responder.py (and with it the aiogram chain) in at runtime. responder may be
    None (LLM part off: Settings.openrouter_api_key is None).

    Members are declared with ``@property``, not as plain attributes: commands.py only
    reads deps.* and never assigns; per PEP 544 a plain (settable) Protocol attribute
    requires the type to match invariantly both ways, while ``@property`` only requires
    read compatibility (covariant). bot.Deps is a plain (non-frozen) dataclass with more
    specific field types (ConfigStore, PromptStore, Database, Responder | None), and
    without ``@property`` it would not match these protocol types structurally.
    """

    @property
    def settings(self) -> Settings: ...
    @property
    def persona(self) -> Persona | None: ...
    @property
    def frag(self) -> Fragments: ...
    @property
    def config_store(self) -> _ConfigStoreLike: ...
    @property
    def prompt_store(self) -> _PromptStoreLike: ...
    @property
    def db(self) -> _DbLike: ...
    @property
    def responder(self) -> _ResponderLike | None: ...
    @property
    def bot_user_id(self) -> int: ...
    @property
    def bot_username(self) -> str: ...
    @property
    def sticker_catalog_enabled(self) -> int: ...
    @property
    def memorizer(self) -> _ChatMemorizerLike | None: ...
    @property
    def weather(self) -> _WeatherLike | None: ...
    @property
    def vocab_store(self) -> _VocabStoreLike | None: ...
    @property
    def bot(self) -> _DownloaderLike | None: ...


def _truncate(text: str) -> str:
    if len(text) <= _MAX_REPLY_LEN:
        return text
    return text[: _MAX_REPLY_LEN - 1] + "…"


async def _reply(message: Message, text: str) -> None:
    await message.answer(_truncate(text), parse_mode=None)


def _catalog(deps: _CommandsDeps) -> Catalog:
    """Owner-UI texts of the pack's locale; a deps without a persona reads the engine's English."""
    return deps.persona.catalog if deps.persona is not None else engine_catalog()


def _t(deps: _CommandsDeps, key: str, /, **params: object) -> str:
    return _catalog(deps).t(key, **params)


def _is_owner(user: User | None, deps: _CommandsDeps) -> bool:
    return (
        user is not None
        and deps.settings.admin_user_id != 0
        and user.id == deps.settings.admin_user_id
    )


def _resolve_mention_target(message: Message) -> tuple[int, str] | None:
    """Target of /mute or /unmute: a reply or a text_mention entity.

    A plain @username (entity type "mention") is not supported: the username is not
    stored in the DB, so there is nothing to look the user up by; ignored silently.
    """
    reply = message.reply_to_message
    if reply is not None and reply.from_user is not None:
        return reply.from_user.id, reply.from_user.full_name
    for entity in message.entities or ():
        if entity.type == "text_mention" and entity.user is not None:
            return entity.user.id, entity.user.full_name
    return None


async def _cmd_stop(message: Message, deps: _CommandsDeps, now: int, user: User | None) -> None:
    changed_by = user.id if user is not None else 0
    await deps.db.set_state("stop_until", str(now + 86400))
    await deps.db.audit_stop("stop", changed_by, now)
    logger.info("stop: silenced until %s by user_id=%s", now + 86400, changed_by)


async def _cmd_mute(message: Message, deps: _CommandsDeps, now: int, user: User | None) -> None:
    """Without a reply/mention, mutes the sender himself, allowed to everyone. By reply
    or mention on SOMEONE ELSE's message, owner only (otherwise it is a ready-made
    harassment tool: "the bot no longer sees you"). Owner target or bot target:
    silently nothing, whoever invoked the command."""
    if user is None:
        return
    target = _resolve_mention_target(message)
    if target is None:
        target_id, target_name = user.id, user.full_name
    else:
        target_id, target_name = target
        if target_id != user.id and not _is_owner(user, deps):
            return
    if target_id == deps.settings.admin_user_id or target_id == deps.bot_user_id:
        return
    display_name = sanitize_display_name(target_name, target_id, set(), frag=deps.frag)
    await deps.db.add_mute(target_id, display_name, user.id, now)
    await deps.db.audit_stop(f"mute:{target_id}", user.id, now)
    logger.info("mute: user_id=%s by user_id=%s", target_id, user.id)


async def _cmd_unmute(message: Message, deps: _CommandsDeps, now: int, user: User | None) -> None:
    """Without a reply/mention, unmutes the sender himself, allowed to everyone.
    By reply or mention on someone else, owner only."""
    if user is None:
        return
    target = _resolve_mention_target(message)
    target_id = user.id if target is None else target[0]
    if target is not None and target_id != user.id and not _is_owner(user, deps):
        return
    await deps.db.remove_mute(target_id)
    await deps.db.audit_stop(f"unmute:{target_id}", user.id, now)
    logger.info("unmute: user_id=%s by user_id=%s", target_id, user.id)


async def _cmd_ex_add(message: Message, deps: _CommandsDeps, now: int) -> None:
    reply = message.reply_to_message
    if reply is None or reply.from_user is None or reply.from_user.id != deps.bot_user_id:
        return
    bot_reply = await deps.db.bot_reply_by_tg_id(reply.message_id)
    if bot_reply is None:
        return

    trigger_msg = None
    if bot_reply.trigger_tg_message_id is not None:
        trigger_msg = await deps.db.message_by_tg_id(
            message.chat.id, bot_reply.trigger_tg_message_id
        )

    if trigger_msg is not None and trigger_msg.display_name and trigger_msg.text:
        name, user_text = trigger_msg.display_name, trigger_msg.text
    else:
        # Chat-facing: the pair is stored as few-shot data the model sees, so the pack words it.
        name = deps.frag.text("placeholders.ex_add_name")
        user_text = deps.frag.text("placeholders.ex_add_user")

    await deps.prompt_store.add_example(name=name, user=user_text, text=bot_reply.text, now=now)
    logger.info("ex add: %r -> %r", user_text, bot_reply.text)


async def _cmd_panic(message: Message, deps: _CommandsDeps, now: int, user: User) -> None:
    await deps.db.set_state("panic", "1")
    await deps.db.audit_stop("panic", user.id, now)
    logger.info("panic set")
    await _reply(message, _t(deps, "commands.panic_done"))


async def _cmd_resume(message: Message, deps: _CommandsDeps, now: int, user: User) -> None:
    """Clear panic, stop_until and the LLM circuit breaker (llm_circuit_until/
    llm_error_streak). The actions are unconditional (as before for panic/stop_until),
    but the reply lists only what was actually set, otherwise the owner would think the
    bot had been paused when it had not."""
    panic_was_set = await deps.db.get_state("panic") is not None
    stop_was_set = await deps.db.get_state("stop_until") is not None
    streak_before = await deps.db.get_state("llm_error_streak")
    circuit_was_open = await deps.db.get_state("llm_circuit_until") is not None

    await deps.db.delete_state("panic")
    await deps.db.delete_state("stop_until")
    await deps.db.delete_state("llm_circuit_until")
    await deps.db.set_state("llm_error_streak", "0")
    await deps.db.audit_stop("resume", user.id, now)
    logger.info(
        "resume: panic=%s stop=%s circuit=%s", panic_was_set, stop_was_set, circuit_was_open
    )

    cleared: list[str] = []
    if panic_was_set:
        cleared.append("panic")
    if stop_was_set:
        cleared.append("stop")
    if circuit_was_open:
        streak = streak_before if streak_before is not None else "?"
        cleared.append(_t(deps, "commands.resume_item_circuit", streak=streak))

    if cleared:
        text = _t(deps, "commands.resume_cleared", items=", ".join(cleared))
    else:
        text = _t(deps, "commands.resume_done")
    await _reply(message, text)


def _llm_last_refusal(deps: _CommandsDeps, raw: str | None, now: int, tz: str) -> str:
    """ "402 at 13:28 26.10" for the last day, else "none in the last day". "Errors in a
    row" would not show this: the cheap models kept passing and resetting the streak
    while the main one got 402 on every call."""
    try:
        at_raw, status = (raw or "").split("|", 1)
        at = int(at_raw)
    except ValueError:
        return _t(deps, "status.refusal_none")
    if now - at >= 86400:
        return _t(deps, "status.refusal_none")
    return _t(deps, "status.refusal", status=status, time=local_dt(at, tz).strftime("%H:%M %d.%m"))


async def _cmd_status(message: Message, deps: _CommandsDeps, now: int) -> None:
    cfg = deps.config_store.get()
    tz = cfg.persona.timezone

    panic = await deps.db.get_state("panic") == "1"
    stop_until_raw = await deps.db.get_state("stop_until")
    # Nobody deletes the stop_until key after it expires (the gate just compares it with
    # now), so show the date only while the stop is in force, otherwise /status would
    # report a stop that no longer exists.
    stop_until = _t(deps, "status.none")
    if stop_until_raw is not None and int(stop_until_raw) > now:
        stop_until = local_dt(int(stop_until_raw), tz).strftime("%H:%M %d.%m")

    circuit_until_raw = await deps.db.get_state("llm_circuit_until")
    circuit_state = _t(deps, "status.circuit_closed")
    if circuit_until_raw is not None:
        circuit_until = int(circuit_until_raw)
        if circuit_until > now:
            circuit_state = _t(
                deps,
                "status.circuit_open",
                time=local_dt(circuit_until, tz).strftime("%H:%M"),
            )
    llm_error_streak = await deps.db.get_state("llm_error_streak") or "0"
    llm_last_refusal = _llm_last_refusal(deps, await deps.db.get_state("llm_last_refusal"), now, tz)

    ambient = int(await deps.db.get_state(day_key("ambient_count", now, tz)) or "0")
    mention = int(await deps.db.get_state(day_key("mention_count", now, tz)) or "0")
    llm_calls = int(await deps.db.get_state(day_key("llm_calls", now, tz)) or "0")
    llm_spent = float(await deps.db.get_state(day_key("llm_spent_usd", now, tz)) or "0")
    reactions = int(await deps.db.get_state(day_key("reaction_count", now, tz)) or "0")
    reaction_calls = int(await deps.db.get_state(day_key("reaction_calls", now, tz)) or "0")
    stickers = int(await deps.db.get_state(day_key("sticker_count", now, tz)) or "0")
    followup_calls = int(await deps.db.get_state(day_key("followup_calls", now, tz)) or "0")
    vision_count = int(await deps.db.get_state(day_key("vision_count", now, tz)) or "0")
    vocab_line = await _vocab_status_line(deps, cfg, now)

    # Presence cap: computed from the tables, as in the gate, not from a state counter.
    presence_cfg = cfg.behaviour.presence
    midnight = day_start(now, tz)
    human_today = await deps.db.count_messages_today(deps.settings.allowed_chat_id, midnight)
    bot_today = await deps.db.count_bot_replies_today(midnight)
    presence_line = _t(
        deps,
        "status.presence",
        bot=bot_today,
        allowance=presence_cfg.allowance(human_today),
        people=human_today,
    )

    # "About itself" metric: look only, nothing is cut.
    self_rows = await deps.db.bot_replies_between(now - _SELF_SHARE_DAYS * 86400, now + 1)
    self_line = render_self_share(
        self_share(
            ((row.trigger, row.text) for row in self_rows),
            deps.config_store.patterns().self_markers,
            frag=deps.frag,
        ),
        _SELF_SHARE_DAYS,
        frag=deps.frag,
    )

    diary_calls = int(await deps.db.get_state(day_key("diary_calls", now, tz)) or "0")
    diary_today = len(await deps.db.self_facts_since(midnight))

    jokes_total = len(await deps.db.jokes(_JOKES_LIST_LIMIT))
    people_open = sum(1 for thread in await deps.db.people_threads_all() if not thread.closed)
    callback_last_raw = await deps.db.get_state("callback_last_at")
    try:
        callback_last = int(callback_last_raw) if callback_last_raw is not None else None
    except ValueError:
        callback_last = None
    callback_text = (
        _t(deps, "status.callback_last", date=local_dt(callback_last, tz).strftime("%d.%m"))
        if callback_last is not None
        else _t(deps, "status.none")
    )

    pending = len(await deps.db.load_pending())
    night_queue = len(await deps.db.night_unanswered())

    life_events = await deps.db.life_events()
    life_unsent = sum(1 for event in life_events if event.announced_at is None)

    memory_total = await deps.db.chat_memory_count()
    memory_end = await deps.db.last_chat_memory_end()
    memory_line = (
        _t(
            deps,
            "status.memory",
            total=memory_total,
            date=local_dt(memory_end, tz).strftime("%d.%m.%Y"),
        )
        if memory_end is not None
        else _t(deps, "status.memory_none")
    )

    hot_until_raw = await deps.db.get_state("hot_until")
    try:
        hot_until = int(hot_until_raw) if hot_until_raw is not None else None
    except ValueError:
        hot_until = None
    if hot_until is not None and hot_until > now:
        hot_ambient_count = int(await deps.db.get_state("hot_ambient_count") or "0")
        hot_line = _t(
            deps,
            "status.hot",
            time=local_dt(hot_until, tz).strftime("%H:%M"),
            count=hot_ambient_count,
            cap=cfg.behaviour.hot_window.ambient_cap,
        )
    else:
        hot_line = _t(deps, "status.hot_none")

    checkin_due_raw = await deps.db.get_state("checkin_due")
    try:
        checkin_due = int(checkin_due_raw) if checkin_due_raw is not None else None
    except ValueError:
        checkin_due = None
    checkin_line = (
        _t(deps, "status.checkin", time=local_dt(checkin_due, tz).strftime("%H:%M"))
        if checkin_due is not None
        else _t(deps, "status.checkin_none")
    )

    # Weather: get() may go to the network if the cache is stale; /status is called
    # rarely, so this is fine. No home point (coordinates not set in .env) or the client
    # is off -> "weather: none".
    weather_line = _t(deps, "status.weather_none")
    weather_client = deps.weather
    snapshot = await weather_client.get() if weather_client is not None else None
    if snapshot is not None:
        now_desc = describe(snapshot.code_now, frag=deps.frag)
        weather_line = _t(
            deps,
            "status.weather",
            temp=format_temp(snapshot.temp_now),
            desc=f", {now_desc}" if now_desc else "",
            time=local_dt(snapshot.fetched_at, tz).strftime("%H:%M"),
        )

    yes, no = _t(deps, "status.yes"), _t(deps, "status.no")
    lines = [
        _t(deps, "status.header", version=__version__),
        _t(deps, "status.panic", value=yes if panic else no),
        _t(deps, "status.stop_until", value=stop_until),
        _t(
            deps,
            "status.circuit",
            state=circuit_state,
            streak=llm_error_streak,
            refusal=llm_last_refusal,
        ),
        _t(
            deps,
            "status.prompt",
            prompt=deps.prompt_store.prompt_version(),
            few_shot=deps.prompt_store.few_shot_version(),
        ),
        _t(
            deps,
            "status.models",
            main=cfg.llm.main_model or "-",
            judge=cfg.llm.judge_model or "-",
        ),
        _t(deps, "status.shadow", value=yes if cfg.filters.shadow else no),
        _t(
            deps,
            "status.today",
            ambient=ambient,
            mention=mention,
            llm_calls=llm_calls,
            llm_spent=llm_spent,
            reactions=reactions,
            reactions_cap=cfg.behaviour.reactions.daily_cap,
            reaction_calls=reaction_calls,
            semantic_cap=cfg.behaviour.reactions.semantic_daily_cap,
            stickers=stickers,
            stickers_cap=cfg.behaviour.stickers.daily_cap,
            catalog=deps.sticker_catalog_enabled,
        ),
        _t(deps, "status.pending", count=pending),
        _t(deps, "status.night_queue", count=night_queue),
        _t(deps, "status.life_events", total=len(life_events), unsent=life_unsent),
        memory_line,
        hot_line,
        _t(
            deps,
            "status.followup",
            calls=followup_calls,
            cap=cfg.behaviour.followup.daily_cap,
        ),
        _t(deps, "status.vision", count=vision_count, cap=cfg.behaviour.vision.daily_cap),
        _t(
            deps,
            "status.diary",
            today=diary_today,
            calls=diary_calls,
            cap=cfg.behaviour.diary.daily_cap,
        ),
        _t(
            deps,
            "status.jokes",
            jokes=jokes_total,
            people=people_open,
            callback=callback_text,
        ),
        checkin_line,
        presence_line,
        self_line,
        vocab_line,
        weather_line,
    ]
    await _reply(message, "\n".join(lines))


async def _vocab_status_line(deps: _CommandsDeps, cfg: Config, now: int) -> str:
    """ "vocab: 1/2 dictionaries (100 words), today 1/2, last word 16.10"."""
    store = deps.vocab_store
    if store is None:
        return _t(deps, "status.vocab_none")
    tz = cfg.persona.timezone
    dicts = await store.dicts()
    enabled = sum(1 for row, _ in dicts if row.enabled)
    words = len(store.get().entries)
    today = int(await deps.db.get_state(day_key("vocab_count", now, tz)) or "0")
    last = _t(deps, "status.vocab_last_none")
    last_raw = await deps.db.get_state("vocab_last")
    if last_raw is not None:
        at_raw, _, word = last_raw.partition("|")
        try:
            last = f"{word} {local_dt(int(at_raw), tz).strftime('%d.%m')}"
        except ValueError:
            last = _t(deps, "status.vocab_last_none")
    return _t(
        deps,
        "status.vocab",
        enabled=enabled,
        total=len(dicts),
        words=words,
        today=today,
        cap=cfg.behaviour.vocab.daily_cap,
        last=last,
    )


async def _cmd_last(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    n = 5
    if args:
        try:
            n = int(args[0])
        except ValueError:
            n = 5
    n = max(1, min(n, 20))

    tz = deps.config_store.get().persona.timezone
    rows = await deps.db.last_bot_replies(n)
    lines = [
        _t(
            deps,
            "commands.last_line",
            time=local_dt(row.created_at, tz).strftime("%H:%M %d.%m"),
            trigger=row.trigger,
            prompt=row.prompt_version,
            few_shot=row.few_shot_version,
            delay=row.delay_sec,
            text=row.text,
        )
        for row in rows
    ]
    await _reply(message, "\n".join(lines) if lines else _t(deps, "commands.empty"))


async def _cmd_why(message: Message, deps: _CommandsDeps, now: int, args: list[str]) -> None:
    hours = 1
    if args:
        try:
            hours = int(args[0])
        except ValueError:
            hours = 1
    hours = max(1, min(hours, 168))

    summary = await deps.db.filter_log_summary(now - hours * 3600)
    lines = [f"{reason} {count}" for reason, count in summary]
    await _reply(message, "\n".join(lines) if lines else _t(deps, "commands.empty"))


def _render_key_card(deps: _CommandsDeps, info: KeyInfo) -> str:
    lines = [f"{info.key}: {info.value}"]
    if info.bounds:
        lines.append(_t(deps, "commands.key_type_bounds", type=info.type_name, bounds=info.bounds))
    else:
        lines.append(_t(deps, "commands.key_type", type=info.type_name))
    if info.description:
        lines.append(info.description)
    if info.overridden:
        lines.append(_t(deps, "commands.key_overridden", default=info.default))
    if not info.settable:
        lines.append(_t(deps, "commands.key_yaml_only"))
    return "\n".join(lines)


async def _cmd_get(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    arg = args[0] if args else ""
    if arg:
        info = deps.config_store.describe(arg)
        if info is not None:
            await _reply(message, _render_key_card(deps, info))
            return

    lines = [
        f"{key}{'*' if overridden else ''}: {value}"
        for key, value, overridden in deps.config_store.flat()
        if key.startswith(arg)
    ]
    if not lines:
        await _reply(message, _t(deps, "commands.empty"))
        return
    lines.append(_t(deps, "commands.get_hint"))
    await _reply(message, "\n".join(lines))


async def _cmd_set(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    if len(args) < 2:
        await _reply(message, _t(deps, "commands.set_usage"))
        return
    key, value = args[0], " ".join(args[1:])
    old, new = await deps.config_store.set(key, value, admin_user_id, now)
    await _reply(message, _t(deps, "commands.set_done", key=key, old=old, new=new))


async def _cmd_unset(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    if not args:
        await _reply(message, _t(deps, "commands.unset_usage"))
        return
    key = args[0]
    default = await deps.config_store.unset(key, admin_user_id, now)
    if default is None:
        await _reply(message, _t(deps, "commands.unset_not_overridden", key=key))
    else:
        await _reply(message, _t(deps, "commands.unset_done", key=key, default=default))


async def _cmd_prompt(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    full = bool(args) and args[0].lower() == "full"
    body = deps.prompt_store.system_prompt()
    total = len(await deps.db.prompt_versions())
    header = _t(
        deps, "commands.prompt_header", version=deps.prompt_store.prompt_version(), total=total
    )

    if full:
        text = f"{header}\n\n{body}"
    else:
        preview = body[:_PROMPT_PREVIEW_LEN]
        if len(body) > _PROMPT_PREVIEW_LEN:
            preview += "…"
        text = f"{header}\n\n{preview}"
    await _reply(message, text)


async def _cmd_rollback(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    if not args:
        await _reply(message, _t(deps, "commands.rollback_usage"))
        return
    try:
        version = int(args[0])
    except ValueError:
        await _reply(message, _t(deps, "commands.rollback_bad_number", arg=args[0]))
        return
    if await deps.prompt_store.rollback_prompt(version):
        await _reply(message, _t(deps, "commands.rollback_done", version=version))
    else:
        await _reply(message, _t(deps, "commands.rollback_not_found", version=version))


async def _cmd_ex_last(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    n = 5
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            n = 5
    n = max(1, n)

    lines = []
    for example in deps.prompt_store.examples()[-n:]:
        payload = json.dumps(
            {"speak": example.speak, "text": example.text},
            ensure_ascii=False,
            separators=(", ", ": "),
        )
        lines.append(
            _t(deps, "commands.ex_line", name=example.name, user=example.user, payload=payload)
        )
    await _reply(message, "\n".join(lines) if lines else _t(deps, "commands.empty"))


async def _cmd_ex_rm(message: Message, deps: _CommandsDeps, now: int, args: list[str]) -> None:
    n = 1
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            n = 1
    n = max(1, n)
    version = await deps.prompt_store.remove_example(n, now)
    await _reply(message, _t(deps, "commands.ex_rm_done", version=version, n=n))


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _command_arg_text(raw_text: str, tokens: list[str]) -> str:
    """Text after the command, with its original inner spaces/newlines (for /say,
    "verbatim"); only outer whitespace is stripped. args (tokens.split()) is no good
    for this: it collapses inner whitespace."""
    if not tokens:
        return ""
    return raw_text[len(tokens[0]) :].strip()


def _is_blocked(outcome: SendOutcome) -> bool:
    return outcome.reason in ("blocked:panic", "blocked:stop")


def _life_outcome_text(deps: _CommandsDeps, outcome: SendOutcome) -> str:
    """Shared part of the reply of /life <text> and /life post N after announce_life."""
    if outcome.sent:
        return _t(deps, "commands.life_sent", text=outcome.text)
    if _is_blocked(outcome):
        return _t(deps, "commands.blocked")
    text = _t(deps, "commands.life_not_sent", reason=outcome.reason)
    if outcome.text:
        text += "\n" + _t(deps, "commands.candidate", text=outcome.text)
    return text


async def _cmd_life(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    args: list[str],
    raw_text: str,
    tokens: list[str],
) -> None:
    sub = args[0].lower() if args else ""
    if sub == "list":
        await _cmd_life_list(message, deps)
        return
    if sub == "rm":
        await _cmd_life_rm(message, deps, now, admin_user_id, args)
        return
    if sub == "post":
        await _cmd_life_post(message, deps, now, admin_user_id, args)
        return

    text = normalize_text(_command_arg_text(raw_text, tokens))
    if not text:
        await _reply(message, _t(deps, "commands.life_usage"))
        return

    event_id = await deps.db.insert_life_event(text=text, created_at=now)
    await deps.db.audit_stop("life:add", admin_user_id, now, text)
    logger.info("life add: #%s %r", event_id, text)

    if deps.responder is None:
        await _reply(message, _t(deps, "commands.life_no_llm", id=event_id))
        return

    event = LifeEventRow(
        id=event_id, text=text, created_at=now, announced_at=None, announced_tg_message_id=None
    )
    outcome = await deps.responder.announce_life(event)
    await _reply(
        message,
        _t(deps, "commands.life_saved", id=event_id, outcome=_life_outcome_text(deps, outcome)),
    )


async def _cmd_life_list(message: Message, deps: _CommandsDeps) -> None:
    tz = deps.config_store.get().persona.timezone
    events = await deps.db.life_events()
    if not events:
        await _reply(message, _t(deps, "commands.life_none"))
        return
    lines = [
        _t(
            deps,
            "commands.life_line",
            id=event.id,
            date=local_dt(event.created_at, tz).strftime("%d.%m.%Y"),
            mark=(
                _t(deps, "commands.life_mark_sent")
                if event.announced_at is not None
                else _t(deps, "commands.dash")
            ),
            text=event.text,
        )
        for event in events
    ]
    await _reply(message, "\n".join(lines))


async def _cmd_life_rm(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    event_id = _parse_int(args[1]) if len(args) > 1 else None
    if event_id is None:
        await _reply(message, _t(deps, "commands.life_usage"))
        return
    event = await deps.db.life_event(event_id)
    if event is None:
        await _reply(message, _t(deps, "commands.life_not_found", id=event_id))
        return
    await deps.db.delete_life_event(event_id)
    await deps.db.audit_stop("life:rm", admin_user_id, now, event.text)
    logger.info("life rm: #%s %r", event_id, event.text)
    await _reply(message, _t(deps, "commands.life_removed", id=event_id))


async def _cmd_life_post(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    event_id = _parse_int(args[1]) if len(args) > 1 else None
    if event_id is None:
        await _reply(message, _t(deps, "commands.life_usage"))
        return
    event = await deps.db.life_event(event_id)
    if event is None:
        await _reply(message, _t(deps, "commands.life_not_found", id=event_id))
        return
    await deps.db.audit_stop("life:post", admin_user_id, now, event.text)
    logger.info("life post: #%s %r", event_id, event.text)

    if deps.responder is None:
        await _reply(message, _t(deps, "commands.life_post_no_llm"))
        return

    outcome = await deps.responder.announce_life(event)
    await _reply(message, _life_outcome_text(deps, outcome))


async def _cmd_memory(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    args: list[str],
) -> None:
    """/memory [list|rm N|run]: long-term chat memory.

    A bare /memory is /memory list: the owner will look at what the bot remembers about
    you more often than change anything.
    """
    sub = args[0].lower() if args else "list"
    if sub == "list":
        await _cmd_memory_list(message, deps)
        return
    if sub == "rm":
        await _cmd_memory_rm(message, deps, now, admin_user_id, args)
        return
    if sub == "run":
        await _cmd_memory_run(message, deps, now, admin_user_id)
        return
    await _reply(message, _t(deps, "commands.memory_usage"))


async def _cmd_memory_list(message: Message, deps: _CommandsDeps) -> None:
    tz = deps.config_store.get().persona.timezone
    rows = await deps.db.chat_memories(_MEMORY_LIST_LIMIT)
    if not rows:
        await _reply(message, _t(deps, "commands.memory_none"))
        return
    blocks = [
        _t(
            deps,
            "commands.memory_item",
            id=row.id,
            period=format_period(row.period_start, row.period_end, tz),
            text=row.text,
        )
        for row in rows
    ]
    await _reply(message, "\n\n".join(blocks))


async def _cmd_memory_rm(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    memory_id = _parse_int(args[1]) if len(args) > 1 else None
    if memory_id is None:
        await _reply(message, _t(deps, "commands.memory_usage"))
        return
    row = await deps.db.chat_memory(memory_id)
    if row is None:
        await _reply(message, _t(deps, "commands.memory_not_found", id=memory_id))
        return
    await deps.db.delete_chat_memory(memory_id)
    await deps.db.audit_stop("memory:rm", admin_user_id, now, row.text)
    logger.info("memory rm: #%s, chars %d", memory_id, len(row.text))
    await _reply(message, _t(deps, "commands.memory_removed", id=memory_id))


async def _cmd_diary(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    """/diary [rm N]: facts about itself extracted from replies.

    A bare /diary lists the week as "#N dd.mm HH:MM text"; /diary rm N deletes a fact
    (if the model pulled out the wrong thing, it stops reaching the {diary} slot)."""
    if not args:
        tz = deps.config_store.get().persona.timezone
        rows = await deps.db.self_facts_since(now - _DIARY_LIST_DAYS * 86400)
        if not rows:
            await _reply(message, _t(deps, "commands.diary_none"))
            return
        lines = [
            _t(
                deps,
                "commands.diary_line",
                id=row.id,
                time=local_dt(row.created_at, tz).strftime("%d.%m %H:%M"),
                text=row.text,
            )
            for row in rows
        ]
        await _reply(message, "\n".join(lines))
        return
    if args[0].lower() != "rm":
        await _reply(message, _t(deps, "commands.diary_usage"))
        return
    fact_id = _parse_int(args[1]) if len(args) > 1 else None
    if fact_id is None:
        await _reply(message, _t(deps, "commands.diary_usage"))
        return
    rows = await deps.db.self_facts_since(0)
    fact = next((row for row in rows if row.id == fact_id), None)
    if fact is None:
        await _reply(message, _t(deps, "commands.diary_not_found", id=fact_id))
        return
    await deps.db.delete_self_fact(fact_id)
    await deps.db.audit_stop("diary:rm", admin_user_id, now, fact.text)
    logger.info("diary rm: #%s", fact_id)
    await _reply(message, _t(deps, "commands.diary_removed", id=fact_id))


async def _cmd_jokes(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    """/jokes [rm N]: chat jokes it remembers. /jokes rm N removes a joke from the
    {jokes} slot."""
    rows = await deps.db.jokes(_JOKES_LIST_LIMIT)
    if not args:
        if not rows:
            await _reply(message, _t(deps, "commands.jokes_none"))
            return
        tz = deps.config_store.get().persona.timezone
        lines = [
            _t(
                deps,
                "commands.jokes_line",
                id=row.id,
                text=row.text,
                used=(
                    local_dt(row.last_used_at, tz).strftime("%d.%m")
                    if row.last_used_at
                    else _t(deps, "commands.dash")
                ),
            )
            for row in rows
        ]
        await _reply(message, "\n".join(lines))
        return
    joke_id = _parse_int(args[1]) if args[0].lower() == "rm" and len(args) > 1 else None
    if joke_id is None:
        await _reply(message, _t(deps, "commands.jokes_usage"))
        return
    joke = next((row for row in rows if row.id == joke_id), None)
    if joke is None:
        await _reply(message, _t(deps, "commands.jokes_not_found", id=joke_id))
        return
    await deps.db.delete_joke(joke_id)
    await deps.db.audit_stop("jokes:rm", admin_user_id, now, joke.text)
    logger.info("jokes rm: #%s", joke_id)
    await _reply(message, _t(deps, "commands.jokes_removed", id=joke_id))


async def _cmd_people(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    """/people [rm N]: unfinished people's stories for the "how did it go?" follow-up.
    These are notes about specific people: the owner sees all of them and can delete
    any."""
    rows = await deps.db.people_threads_all()
    if not args:
        if not rows:
            await _reply(message, _t(deps, "commands.people_none"))
            return
        tz = deps.config_store.get().persona.timezone
        lines = [
            _t(
                deps,
                "commands.people_line",
                id=row.id,
                name=row.display_name,
                text=row.text,
                state=(
                    _t(
                        deps,
                        "commands.people_asked",
                        date=local_dt(row.asked_at, tz).strftime("%d.%m"),
                    )
                    if row.asked_at is not None
                    else _t(deps, "commands.people_open")
                ),
            )
            for row in rows
        ]
        await _reply(message, "\n".join(lines))
        return
    thread_id = _parse_int(args[1]) if args[0].lower() == "rm" and len(args) > 1 else None
    if thread_id is None:
        await _reply(message, _t(deps, "commands.people_usage"))
        return
    thread = next((row for row in rows if row.id == thread_id), None)
    if thread is None:
        await _reply(message, _t(deps, "commands.people_not_found", id=thread_id))
        return
    await deps.db.delete_people_thread(thread_id)
    await deps.db.audit_stop("people:rm", admin_user_id, now, thread.text)
    logger.info("people rm: #%s", thread_id)
    await _reply(message, _t(deps, "commands.people_removed", id=thread_id))


async def _cmd_memory_run(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int
) -> None:
    """Forced run, outside run_window, so there is no waiting until night."""
    if deps.memorizer is None:
        await _reply(message, _t(deps, "commands.memory_run_off"))
        return
    await deps.db.audit_stop("memory:run", admin_user_id, now)
    tz = deps.config_store.get().persona.timezone
    created = await deps.memorizer.run_due(now=now)
    logger.info("memory run: added %d", len(created))
    if not created:
        await _reply(message, _t(deps, "commands.memory_run_none"))
        return
    blocks = [
        _t(
            deps,
            "commands.memory_item",
            id=row.id,
            period=format_period(row.period_start, row.period_end, tz),
            text=row.text,
        )
        for row in created
    ]
    preview = "\n\n".join(blocks)[:_MEMORY_RUN_PREVIEW_LEN]
    await _reply(message, _t(deps, "commands.memory_run_done", count=len(created), preview=preview))


async def _cmd_say(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    raw_text: str,
    tokens: list[str],
) -> None:
    text = _command_arg_text(raw_text, tokens)
    if not text:
        await _reply(message, _t(deps, "commands.say_usage"))
        return

    await deps.db.audit_stop("say", admin_user_id, now, text)
    logger.info("say: %r", text)

    if deps.responder is None:
        await _reply(message, _t(deps, "commands.say_off"))
        return

    outcome = await deps.responder.say(text)
    if outcome.sent:
        await _reply(message, _t(deps, "commands.say_sent"))
    else:
        await _reply(message, _t(deps, "commands.blocked"))


def _preview_text(deps: _CommandsDeps, outcome: PreviewOutcome) -> str:
    """Reply to /ask: what would go to the chat, or why it would stay silent."""
    if outcome.sent:
        lines = [_t(deps, "commands.ask_would_answer", text=outcome.text)]
        if outcome.reasons:
            lines.append(_t(deps, "commands.ask_shadow", reasons=", ".join(outcome.reasons)))
    elif outcome.reason == "llm:silent":
        lines = [_t(deps, "commands.ask_silent")]
    elif outcome.reason.startswith("llm:"):
        lines = [_t(deps, "commands.ask_llm_error", reason=outcome.reason)]
        if outcome.text:
            lines.append(_t(deps, "commands.ask_model_returned", text=outcome.text))
    else:
        lines = [
            _t(
                deps,
                "commands.ask_filtered",
                reasons=", ".join(outcome.reasons) or outcome.reason,
            )
        ]
        if outcome.text:
            lines.append(_t(deps, "commands.candidate", text=outcome.text))
    if outcome.fixes:
        lines.append(_t(deps, "commands.ask_fixes", fixes=", ".join(outcome.fixes)))
    lines.append(_t(deps, "commands.ask_not_sent"))
    return "\n".join(lines)


async def _cmd_ask(
    message: Message, deps: _CommandsDeps, user: User, raw_text: str, tokens: list[str]
) -> None:
    """/ask <text>: what the persona would answer if the owner addressed it in the chat.
    Sends and writes nothing."""
    text = normalize_text(_command_arg_text(raw_text, tokens))
    if not text:
        await _reply(message, _t(deps, "commands.ask_usage"))
        return
    if deps.responder is None:
        await _reply(message, _t(deps, "commands.ask_off"))
        return
    # The name is the one the owner already has in the chat context; if he has not
    # written in the chat, the Telegram name through the same sanitize as for messages.
    known = await deps.db.display_names([user.id])
    name = known.get(user.id) or sanitize_display_name(
        user.full_name, user.id, set(), frag=deps.frag
    )
    logger.info("ask preview: %r", text)
    outcome = await deps.responder.preview(text, name, user.id)
    await _reply(message, _preview_text(deps, outcome))


def _plural_words(deps: _CommandsDeps, n: int) -> str:
    return _catalog(deps).plural("commands.words_count", n)


def _words_too_big(deps: _CommandsDeps) -> str:
    return _t(deps, "commands.words_too_big", kb=_WORDS_MAX_FILE_BYTES // 1024)


async def _cmd_words_upload(
    message: Message, deps: _CommandsDeps, store: _VocabStoreLike, now: int, admin_user_id: int
) -> None:
    """A document captioned /words: upload a dictionary (the same name replaces it)."""
    document = message.document
    assert document is not None  # guaranteed by the caller
    if document.file_size is not None and document.file_size > _WORDS_MAX_FILE_BYTES:
        await _reply(message, _words_too_big(deps))
        return
    if deps.bot is None:
        await _reply(message, _t(deps, "commands.words_no_download"))
        return
    buffer = io.BytesIO()
    await deps.bot.download(document.file_id, destination=buffer)
    data = buffer.getvalue()
    if len(data) > _WORDS_MAX_FILE_BYTES:
        await _reply(message, _words_too_big(deps))
        return
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        await _reply(message, _t(deps, "commands.words_not_utf8"))
        return
    try:
        parsed = await store.put(raw, now)
    except ValueError as exc:
        detail = exc.render(_catalog(deps)) if isinstance(exc, OwnerError) else str(exc)
        await _reply(message, _t(deps, "commands.words_parse_error", error=detail))
        return
    await deps.db.audit_stop("words:put", admin_user_id, now, parsed.name)
    logger.info("words put: %r, %d words", parsed.name, len(parsed.words))
    enabled = next((row.enabled for row, _ in await store.dicts() if row.name == parsed.name), True)
    state = _t(deps, "commands.words_state_on" if enabled else "commands.words_state_off_hint")
    await _reply(
        message,
        _t(
            deps,
            "commands.words_loaded",
            name=parsed.name,
            words=_plural_words(deps, len(parsed.words)),
            state=state,
        ),
    )


def _words_usage(deps: _CommandsDeps) -> str:
    return _t(deps, "commands.words_usage")


def _resolve_dict_name(
    dicts: Sequence[tuple[VocabDictRow, VocabDict | None]], name: str
) -> VocabDictRow | None:
    """Dictionary name: exact match, otherwise the only case-insensitive one."""
    exact = [row for row, _ in dicts if row.name == name]
    if exact:
        return exact[0]
    folded = [row for row, _ in dicts if row.name.casefold() == name.casefold()]
    return folded[0] if len(folded) == 1 else None


async def _cmd_words(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    args: list[str],
) -> None:
    """/words: vocabulary dictionaries."""
    store = deps.vocab_store
    if store is None:
        await _reply(message, _t(deps, "commands.words_unavailable"))
        return
    if message.document is not None:
        await _cmd_words_upload(message, deps, store, now, admin_user_id)
        return
    dicts = await store.dicts()
    tz = deps.config_store.get().persona.timezone
    if not args:
        if not dicts:
            await _reply(message, _t(deps, "commands.words_none") + "\n" + _words_usage(deps))
            return
        lines = []
        for row, parsed in dicts:
            size = (
                _plural_words(deps, len(parsed.words))
                if parsed is not None
                else _t(deps, "commands.words_unparsable")
            )
            lines.append(
                _t(
                    deps,
                    "commands.words_line",
                    name=row.name,
                    size=size,
                    state=_t(deps, "commands.words_on" if row.enabled else "commands.words_off"),
                    date=local_dt(row.updated_at, tz).strftime("%d.%m"),
                )
            )
        lines.append(_words_usage(deps))
        await _reply(message, "\n".join(lines))
        return
    sub_cmd = args[0].lower()
    name = " ".join(args[1:]).strip()
    if sub_cmd not in {"show", "on", "off", "rm"} or not name:
        await _reply(message, _words_usage(deps))
        return
    found = _resolve_dict_name(dicts, name)
    if found is None:
        await _reply(message, _t(deps, "commands.words_no_dict", name=name))
        return
    row = found
    if sub_cmd == "show":
        parsed = next(p for r, p in dicts if r.name == row.name)
        if parsed is None:
            await _reply(message, _t(deps, "commands.words_show_unparsable", name=row.name))
            return
        words_text = _plural_words(deps, len(parsed.words))
        if parsed.about:
            header = _t(
                deps,
                "commands.words_header_about",
                name=row.name,
                words=words_text,
                about=parsed.about,
            )
        else:
            header = _t(deps, "commands.words_header", name=row.name, words=words_text)
        lines = [header] + [f"{item.word}: {item.meaning}" for item in parsed.words]
        await _reply(message, "\n".join(lines))
        return
    if sub_cmd == "rm":
        await store.remove(row.name)
        await deps.db.audit_stop("words:rm", admin_user_id, now, row.name)
        logger.info("words rm: %r", row.name)
        await _reply(message, _t(deps, "commands.words_removed", name=row.name))
        return
    enabled = sub_cmd == "on"
    await store.set_enabled(row.name, enabled)
    await deps.db.audit_stop(f"words:{sub_cmd}", admin_user_id, now, row.name)
    logger.info("words %s: %r", sub_cmd, row.name)
    await _reply(
        message,
        _t(deps, "commands.words_enabled" if enabled else "commands.words_disabled", name=row.name),
    )


async def _cmd_changelog(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    """/changelog [owner|<version>]: release notes.

    The chat block is the pack's `chat_notes.md`, or without it (transition period) the
    "For chat" block from the pack's CHANGELOG.md; the owner block comes from the
    pack's CHANGELOG.md. Files are read at startup: changing the pack means a restart.
    """
    persona = deps.persona
    chat_notes = persona.chat_notes if persona is not None else ""
    legacy = persona.legacy_changelog if persona is not None else ""

    owner_view = bool(args) and args[0].lower() == "owner"
    releases = (
        owner_releases(legacy, engine_changelog_text())
        if owner_view
        else chat_releases(chat_notes, legacy)
    )
    if not releases:
        await _reply(message, _t(deps, "commands.changelog_missing"))
        return

    found: Release | None
    version_args = args[1:] if owner_view else args
    if not version_args:
        found = releases[0]
    else:
        raw = version_args[0].lower()
        wanted = raw[1:] if raw.startswith("v") else raw
        found = next((r for r in releases if r.version == wanted), None)
        if found is None:
            await _reply(message, _t(deps, "commands.changelog_no_version", arg=version_args[0]))
            return
    release = found

    body = release.for_owner if owner_view else release.for_chat
    await _reply(
        message,
        _t(
            deps,
            "commands.changelog_release",
            version=release.version,
            date=release.date,
            body=body,
        ),
    )


def build_commands_router(deps: _CommandsDeps) -> Router:
    """Build a Router with a single command-dispatcher handler.

    Must be included in the Dispatcher FIRST, before sotto.bot.build_router: commands
    must not reach the gate and the log/DB of the main handler. aiogram itself stops
    propagation after the first handler whose function returned without an exception,
    so an explicit "cancel propagation" is not needed; only the order in which the
    integrator registers the routers matters.
    """
    router = Router(name=__name__)

    @router.message((F.text & F.text.startswith("/")) | (F.caption & F.caption.startswith("/")))
    async def handle_command(message: Message) -> None:
        text = message.text or message.caption or ""
        tokens = text.split()
        if not tokens:
            return
        cmd_token = tokens[0][1:]
        if "@" in cmd_token:
            cmd_part, _, suffix = cmd_token.partition("@")
            # The "@other_bot" suffix: in Telegram group chats a command is delivered to
            # all bots at once; execute only the one addressed to us (case-insensitive),
            # otherwise e.g. /stop@weatherbot would stop us too.
            if suffix.lower() != deps.bot_username.lower():
                return
            cmd = cmd_part.lower()
        else:
            cmd = cmd_token.lower()
        args = tokens[1:]

        user = message.from_user
        now = int(message.date.timestamp())

        is_chat = message.chat.id == deps.settings.allowed_chat_id
        is_owner_private = (
            message.chat.type == "private"
            and deps.settings.admin_user_id != 0
            and user is not None
            and user.id == deps.settings.admin_user_id
        )
        if not is_chat and not is_owner_private:
            return

        try:
            if is_chat:
                if cmd == "stop":
                    await _cmd_stop(message, deps, now, user)
                elif cmd == "mute":
                    await _cmd_mute(message, deps, now, user)
                elif cmd == "unmute":
                    await _cmd_unmute(message, deps, now, user)
                elif cmd == "ex" and args and args[0].lower() == "add" and _is_owner(user, deps):
                    await _cmd_ex_add(message, deps, now)
                # Unknown/foreign command in the chat: silence, no log and no reply.
            else:
                assert user is not None  # guaranteed by is_owner_private above
                admin_user_id = deps.settings.admin_user_id
                if cmd == "panic":
                    await _cmd_panic(message, deps, now, user)
                elif cmd == "resume":
                    await _cmd_resume(message, deps, now, user)
                elif cmd == "status":
                    await _cmd_status(message, deps, now)
                elif cmd == "last":
                    await _cmd_last(message, deps, args)
                elif cmd == "why":
                    await _cmd_why(message, deps, now, args)
                elif cmd == "get":
                    await _cmd_get(message, deps, args)
                elif cmd == "set":
                    await _cmd_set(message, deps, now, admin_user_id, args)
                elif cmd == "unset":
                    await _cmd_unset(message, deps, now, admin_user_id, args)
                elif cmd == "prompt":
                    await _cmd_prompt(message, deps, args)
                elif cmd == "rollback":
                    await _cmd_rollback(message, deps, args)
                elif cmd == "ex" and args and args[0].lower() == "last":
                    await _cmd_ex_last(message, deps, args)
                elif cmd == "ex" and args and args[0].lower() == "rm":
                    await _cmd_ex_rm(message, deps, now, args)
                elif cmd == "life":
                    await _cmd_life(message, deps, now, admin_user_id, args, text, tokens)
                elif cmd == "say":
                    await _cmd_say(message, deps, now, admin_user_id, text, tokens)
                elif cmd == "ask":
                    await _cmd_ask(message, deps, user, text, tokens)
                elif cmd == "memory":
                    await _cmd_memory(message, deps, now, admin_user_id, args)
                elif cmd == "diary":
                    await _cmd_diary(message, deps, now, admin_user_id, args)
                elif cmd == "jokes":
                    await _cmd_jokes(message, deps, now, admin_user_id, args)
                elif cmd == "people":
                    await _cmd_people(message, deps, now, admin_user_id, args)
                elif cmd == "words":
                    await _cmd_words(message, deps, now, admin_user_id, args)
                elif cmd == "changelog":
                    await _cmd_changelog(message, deps, args)
                else:  # /help and any unknown command
                    await _reply(message, _t(deps, "help.text"))
        except Exception as exc:
            logger.exception("command /%s failed", cmd)
            if is_owner_private:
                detail = exc.render(_catalog(deps)) if isinstance(exc, OwnerError) else str(exc)
                await _reply(message, _t(deps, "errors.generic", message=detail))

    return router
