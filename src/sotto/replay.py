"""Replay the gate (and, with ``--generate``, the full pipeline) on chat history.

``python -m sotto.replay <result.json|bot.db> [--config config.yaml] [--seed 1]
  [--bot-username example_persona_bot] [--bot-user-id 0] [--verbose]
  [--from-db] [--chat-id ID] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
  [--generate [--judge] [--max-calls 20]]``

``source`` is either ``result.json`` (a Telegram Desktop export) or the bot's own
database (``bot.db``/``*.sqlite``/``*.sqlite3`` by extension, or any path with
``--from-db``), for cases where the owner cannot export the history (the group forbids
saving content). For a database the ``messages`` table is read directly with a single
read-only SELECT through a sqlite3 ``mode=ro`` URI, without ``Database.connect()``/
``migrate()``, so the live file and its WAL are not touched. The chat is chosen by
``--chat-id`` or by the only ``chat_id`` in the table (several is an error with a
hint). Bot messages (``is_bot=1``) stay in the stream; ``--bot-user-id``, if not given
explicitly, is taken from their ``user_id`` automatically.
``--since``/``--until`` (a local date in ``persona.timezone``) cut the history at both
ends and work for both sources.

Without ``--generate`` it runs ``gate.should_consider`` over every history message
with in-memory state (no sqlite, nothing is sent or written anywhere). On PASS it
assumes the reply was sent, since otherwise daily limits and cooldowns cannot be checked.

With ``--generate`` every PASS additionally calls the same generation path as
``Responder._generate_and_send``: fresh context is the last ``context_window``
messages of the export before the current one, ``recent_replies`` are the replies
already generated in this run, ``build_messages`` -> ``LLMClient.call`` (a real
OpenRouter HTTP call, key from ``Settings().openrouter_api_key``) -> ``parse_reply``
-> ``filters.check_output`` with a full ``FilterContext`` (the judge only with
``--judge``). Nothing is sent or written to the persistent DB: LLM budget state
(``llm_calls``/``llm_spent_usd``/circuit) lives in ``InMemoryStateStore`` only for the
duration of the run, but the config limits (``daily_calls_cap``, ``daily_budget_usd``,
the circuit breaker) apply for real, plus a hard cap of its own, ``--max-calls``, on
the run itself so repeated runs do not waste budget.

``--max-calls`` counts REAL network calls (main model + judge), not the number of
PASSes with generation: before each PASS generation we read ``llm_calls`` for the
current day from ``InMemoryStateStore`` and, if it is already ``>= max_calls``, skip
generation (gate counting by day continues as usual). The check happens before the
call, so the generation that triggers it can raise the counter by one more main call
and, if layers 1-2 passed or shadow is on, by a judge call; exceeding ``max_calls`` by
at most one judge call is allowed and is not a bug. The final report line prints the
actual number of network calls from the store
(``InMemoryStateStore.total_llm_calls()``), not the number of generation attempts.

``gate.py`` and ``patterns.py`` were originally written in parallel by other agents;
the import inside ``_run_replay_async()`` stays lazy for the same reason (historically,
so the export parser tests did not fail before they existed).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sqlite3
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from sotto.config import load_config
from sotto.config_models import Config
from sotto.db import MessageRow
from sotto.export_parser import ExportMessage, parse_export
from sotto.few_shot import load_few_shot, render_few_shot
from sotto.filters import FilterContext, check_output
from sotto.fragments import Fragments
from sotto.gate_types import (
    GateMessage,
    GateState,
    RecentActivity,
    StateChange,
    Trigger,
    Verdict,
)
from sotto.judge import Judge
from sotto.lang import default_config_layer
from sotto.llm import LLMClient, LLMError
from sotto.patterns import Patterns
from sotto.persona import Persona, load_persona, persona_from_env
from sotto.prompt import build_messages, parse_reply, render_context
from sotto.sanitize import sanitize_display_name
from sotto.settings import Settings
from sotto.timeutil import day_key, local_date, local_dt

logger = logging.getLogger(__name__)

_TEXT_TRUNCATE_LEN = 120
_TOP_DROP_REASONS = 5
_DB_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}

# FilterContext.recent_replies is the last 50 replies, regardless of
# cfg.behaviour.recent_replies_memory, which limits {recent_replies} in the generation
# prompt itself; see the same technique in responder.py.
_FILTER_RECENT_REPLIES = 50


class ReplayState:
    """In-memory snapshot of gate state for replay, without sqlite, gate or patterns.

    A separate class so the ``recent`` window and counter increments are testable
    without the gate.py/patterns.py modules.
    """

    def __init__(self, tz: str, window_min: int) -> None:
        self.tz = tz
        self.window_min = window_min
        self.panic = False
        self.stop_until: int | None = None
        self.topic_cooldown_until: int | None = None
        self.muted_user_ids: frozenset[int] = frozenset()
        self.last_mention_reply_at: int | None = None
        self.last_mention_reply_at_user: dict[int, int] = {}
        self.last_ambient_at: int | None = None
        self.mention_count: dict[str, int] = {}
        self.ambient_count: dict[str, int] = {}
        self._recent: deque[RecentActivity] = deque()

    def push_recent(self, user_id: int, created_at: int) -> None:
        """Add a non-bot message to the live_talk window and drop stale ones."""
        self._recent.append(RecentActivity(user_id=user_id, created_at=created_at))
        self._prune_recent(created_at)

    def _prune_recent(self, now: int) -> None:
        cutoff = now - self.window_min * 60
        while self._recent and self._recent[0].created_at < cutoff:
            self._recent.popleft()

    def recent_window(self, now: int) -> tuple[RecentActivity, ...]:
        self._prune_recent(now)
        return tuple(self._recent)

    def gate_state(self, msg: GateMessage, now: int) -> GateState:
        mention_key = day_key("mention_count", now, self.tz)
        ambient_key = day_key("ambient_count", now, self.tz)
        return GateState(
            panic=self.panic,
            stop_until=self.stop_until,
            topic_cooldown_until=self.topic_cooldown_until,
            muted_user_ids=self.muted_user_ids,
            mention_count_today=self.mention_count.get(mention_key, 0),
            last_mention_reply_at=self.last_mention_reply_at,
            last_mention_reply_at_user=self.last_mention_reply_at_user.get(msg.user_id),
            ambient_count_today=self.ambient_count.get(ambient_key, 0),
            last_ambient_at=self.last_ambient_at,
            recent=self.recent_window(now),
        )

    def apply_state_changes(self, changes: Iterable[StateChange]) -> None:
        """The gate's only StateChange is topic_cooldown_until."""
        for change in changes:
            if change.key == "topic_cooldown_until":
                self.topic_cooldown_until = None if change.value is None else int(change.value)

    def apply_pass(self, trigger: Trigger, msg: GateMessage, now: int) -> None:
        """PASS => count the reply as sent: increment counters and last_*_at."""
        if trigger in (Trigger.MENTION, Trigger.REPLY, Trigger.NAME):
            key = day_key("mention_count", now, self.tz)
            self.mention_count[key] = self.mention_count.get(key, 0) + 1
            self.last_mention_reply_at = now
            self.last_mention_reply_at_user[msg.user_id] = now
        elif trigger is Trigger.AMBIENT:
            key = day_key("ambient_count", now, self.tz)
            self.ambient_count[key] = self.ambient_count.get(key, 0) + 1
            self.last_ambient_at = now


@dataclass
class DayStats:
    date: str
    messages: int = 0
    pass_mention: int = 0
    pass_reply: int = 0
    pass_name: int = 0
    pass_ambient: int = 0
    night: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def pass_total(self) -> int:
        return self.pass_mention + self.pass_reply + self.pass_name + self.pass_ambient

    def record_pass(self, trigger: Trigger) -> None:
        if trigger is Trigger.MENTION:
            self.pass_mention += 1
        elif trigger is Trigger.REPLY:
            self.pass_reply += 1
        elif trigger is Trigger.NAME:
            self.pass_name += 1
        elif trigger is Trigger.AMBIENT:
            self.pass_ambient += 1

    def record_drop(self, reason: str) -> None:
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1


@dataclass
class GenerateStats:
    """Counters of a ``--generate`` run: calls, what would be sent, cuts, cost.

    ``calls`` is filled at the end of the run from ``InMemoryStateStore.total_llm_calls()``:
    the real number of network calls (main model + judge), not the number of PASSes for
    which generation was started (see ``--max-calls`` in the module docstring).
    """

    calls: int = 0
    would_send: int = 0
    cost_usd: float = 0.0
    cut_reasons: dict[str, int] = field(default_factory=dict)

    def record_cut(self, reason: str) -> None:
        self.cut_reasons[reason] = self.cut_reasons.get(reason, 0) + 1


class InMemoryStateStore:
    """In-memory implementation of ``llm._StateStore`` for ``LLMClient`` in replay.

    ``day_key()`` counters (llm_calls/llm_spent_usd, error streak, circuit) live only in
    the run's memory, without sqlite, but config budgets (daily_calls_cap,
    daily_budget_usd, the circuit breaker) apply for real, exactly as in the real
    ``Database``.
    """

    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    async def get_state(self, key: str) -> str | None:
        return self._state.get(key)

    async def set_state(self, key: str, value: str) -> None:
        self._state[key] = value

    async def increment_state(self, key: str, by: int = 1) -> int:
        current = int(self._state.get(key, "0"))
        new_value = current + by
        self._state[key] = str(new_value)
        return new_value

    async def add_state_float(self, key: str, by: float) -> float:
        current = float(self._state.get(key, "0"))
        new_value = current + by
        self._state[key] = str(new_value)
        return new_value

    async def llm_calls_for_day(self, now: int, tz: str) -> int:
        """``llm_calls`` for the day of ``now`` (by ``persona.timezone``); used by
        ``--max-calls`` before each new PASS generation."""
        raw = await self.get_state(day_key("llm_calls", now, tz))
        return int(raw) if raw is not None else 0

    def total_llm_calls(self) -> int:
        """Sum of ``llm_calls:<day>`` over all days of the run: the real number of network
        calls (main model + judge), regardless of how many days the export covers.
        Used for the final report line."""
        return sum(int(value) for key, value in self._state.items() if key.startswith("llm_calls:"))


@dataclass
class _GenerateContext:
    """Everything generation needs during the run: client, judge, templates, store."""

    llm: LLMClient
    judge: Judge | None
    prompt_template: str
    few_shot_text: str
    store: InMemoryStateStore
    frag: Fragments


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


def _build_generate_context(cfg: Config, use_judge: bool, persona: Persona) -> _GenerateContext:
    """Build an LLMClient (+ a judge with ``--judge``) for a ``--generate`` run.

    The key comes from ``Settings().openrouter_api_key`` (the same .env as the bot).
    No key -> a clear error instead of an obscure failure inside LLMClient.
    """
    settings = Settings()
    if settings.openrouter_api_key is None:
        raise RuntimeError(
            '--generate needs OPENROUTER_API_KEY: set it in .env (see the README, section "Replay")'
        )

    prompt_template = persona.prompts["system"]
    few_shot_items = load_few_shot(persona.few_shot_path)
    few_shot_text = render_few_shot(few_shot_items)

    store = InMemoryStateStore()
    http = httpx.AsyncClient(timeout=cfg.llm.timeout_sec)
    llm = LLMClient(
        api_key=settings.openrouter_api_key.get_secret_value(),
        cfg_getter=lambda: cfg,
        db=store,
        http=http,
    )

    judge: Judge | None = None
    if use_judge:
        judge = Judge(llm, lambda: cfg, persona.prompts["judge"], frag=persona.frag)

    return _GenerateContext(
        llm=llm,
        judge=judge,
        prompt_template=prompt_template,
        few_shot_text=few_shot_text,
        store=store,
        frag=persona.frag,
    )


def situation_for(trigger: Trigger, *, frag: Fragments) -> str:
    """Situation for generation in replay, as in Responder for the same trigger.

    Ambient gets ``situations.ambient``, otherwise replay would test an old prompt
    shape. Addresses in replay still go without ``situation_addressed``, a known
    simplification of replay.
    """
    return frag.text("situations.ambient") if trigger is Trigger.AMBIENT else ""


async def _generate_one(
    ctx: _GenerateContext,
    cfg: Config,
    tz: str,
    exported: ExportMessage,
    trigger: Trigger,
    display_name: str,
    context_rows: list[MessageRow],
    generated_replies: list[str],
    stats: GenerateStats,
    patterns: Patterns,
) -> str:
    """One PASS with ``--generate`` on: generation + output filter.

    Returns the finished report line: "time | name | trigger | message -> reply | verdict".
    By the same path as ``Responder._generate_and_send``: a real LLM call, then
    ``filters.check_output`` with a full ``FilterContext``. Nothing is sent, only
    printed and counted in ``stats``. The real number of network calls is counted not
    here but from ``ctx.store.total_llm_calls()`` at the end of the run (``stats.calls``,
    see the module docstring about ``--max-calls``).
    """
    when = local_dt(exported.created_at, tz).strftime("%Y-%m-%d %H:%M:%S")
    trigger_text = (
        exported.text if trigger in (Trigger.MENTION, Trigger.REPLY, Trigger.NAME) else ""
    )

    age = cfg.persona.age(local_date(exported.created_at, tz))
    recent_for_prompt = generated_replies[-cfg.behaviour.recent_replies_memory :]
    messages = build_messages(
        ctx.prompt_template,
        age=age,
        few_shot=ctx.few_shot_text,
        context=render_context(context_rows),
        recent_replies="\n".join(recent_for_prompt),
        places="",
        situation=situation_for(trigger, frag=ctx.frag),
        frag=ctx.frag,
    )

    def _line(reply_text: str, verdict_str: str) -> str:
        return (
            f"{when} | {display_name} | {trigger.value} | "
            f"{_truncate(exported.text)} → {_truncate(reply_text)} | {verdict_str}"
        )

    try:
        result = await ctx.llm.call(
            messages,
            model=cfg.llm.main_model,
            max_tokens=cfg.llm.max_tokens,
            now=exported.created_at,
        )
    except LLMError as exc:
        stats.record_cut(exc.reason)
        return _line("(no reply)", f"cut: {exc.reason}")

    stats.cost_usd += result.cost_usd

    reply = parse_reply(result.text)
    if reply is None:
        stats.record_cut("llm:invalid_json")
        return _line("(no reply)", "cut: llm:invalid_json")
    if not reply.speak:
        stats.record_cut("llm:silent")
        return _line("(silent)", "cut: llm:silent")

    participant_names = _unique_participant_names(context_rows)
    bot_names = [cfg.persona.name, cfg.persona.display_name, *cfg.persona.name_triggers]
    filter_ctx = FilterContext(
        cfg=cfg,
        recent_replies=generated_replies[-_FILTER_RECENT_REPLIES:],
        context_rows=context_rows,
        places_names=[],
        participant_names=participant_names,
        bot_names=bot_names,
        muted_names=[],  # /mute is not simulated in replay
        patterns=patterns,
        system_prompt=ctx.prompt_template,
        trigger_text=trigger_text,
        now=exported.created_at,
        frag=ctx.frag,
    )
    verdict = await check_output(reply.text, filter_ctx, ctx.judge)

    if verdict.ok:
        stats.would_send += 1
        generated_replies.append(reply.text)
        return _line(reply.text, "pass")

    reasons = verdict.reasons or (verdict.reason,)
    for reason in reasons:
        stats.record_cut(reason)
    if cfg.filters.shadow:
        # shadow: counts and cuts only in the report; it would really have sent it.
        stats.would_send += 1
        generated_replies.append(reply.text)
    return _line(reply.text, ", ".join(reasons))


def _is_db_source(source: Path, from_db: bool) -> bool:
    return from_db or source.suffix.lower() in _DB_EXTENSIONS


def _row_to_export_message(row: sqlite3.Row) -> ExportMessage | None:
    """``messages`` row -> ``ExportMessage``, or ``None`` when required fields are
    missing (should not happen on a production DB, but replay is no place to crash over it)."""
    tg_message_id, user_id, created_at = row["tg_message_id"], row["user_id"], row["created_at"]
    if tg_message_id is None or user_id is None or created_at is None:
        logger.warning("replay: skipping db message row missing a required field: %r", tuple(row))
        return None
    reply_to = row["reply_to_tg_message_id"]
    return ExportMessage(
        tg_message_id=int(tg_message_id),
        user_id=int(user_id),
        display_name=str(row["display_name"] or ""),
        text=str(row["text"] or ""),
        reply_to_tg_message_id=int(reply_to) if reply_to is not None else None,
        created_at=int(created_at),
    )


def _read_db_messages(db_path: Path, chat_id: int | None) -> tuple[list[ExportMessage], int]:
    """Read history from the bot's database: one read-only SELECT over ``messages``,
    without ``Database``/``connect()``/``migrate()``; the sqlite3 ``mode=ro`` URI
    guarantees the live file (and its WAL) is untouched.

    Returns (messages, bot_user_id): ``bot_user_id`` is derived from rows with
    ``is_bot=1`` (the most frequent ``user_id`` among them), 0 if there are none;
    the caller uses it only if ``--bot-user-id`` is not given explicitly.

    ``chat_id=None`` -> the only ``chat_id`` in the table is used; several different
    ones -> ``ValueError`` with a hint to use ``--chat-id``.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path}: database file not found")

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if chat_id is None:
            id_rows = conn.execute("SELECT DISTINCT chat_id FROM messages").fetchall()
            chat_ids = sorted({row[0] for row in id_rows if row[0] is not None})
            if not chat_ids:
                raise ValueError(f"{db_path}: the messages table is empty")
            if len(chat_ids) > 1:
                raise ValueError(
                    f"{db_path}: messages holds several chat_ids {chat_ids}; "
                    "pick the chat to replay with --chat-id <id>"
                )
            chat_id = chat_ids[0]

        raw_rows = conn.execute(
            "SELECT tg_message_id, user_id, display_name, text, "
            "reply_to_tg_message_id, created_at, is_bot "
            "FROM messages WHERE chat_id = ? ORDER BY created_at, tg_message_id",
            (chat_id,),
        ).fetchall()
    finally:
        conn.close()

    bot_ids = Counter(
        int(row["user_id"]) for row in raw_rows if row["is_bot"] and row["user_id"] is not None
    )
    detected_bot_user_id = bot_ids.most_common(1)[0][0] if bot_ids else 0

    messages: list[ExportMessage] = []
    for row in raw_rows:
        parsed = _row_to_export_message(row)
        if parsed is not None:
            messages.append(parsed)

    return messages, detected_bot_user_id


def _parse_date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from exc


def _filter_by_date_range(
    messages: list[ExportMessage], tz: str, since: date | None, until: date | None
) -> list[ExportMessage]:
    """Drop messages outside [--since, --until] by local date (persona.timezone).

    ``since`` is from 00:00 of that day inclusive; ``until`` is through the end of that
    day inclusive (up to 00:00 of the next).
    """
    if since is not None:
        since_ts = int(datetime.combine(since, time.min, tzinfo=ZoneInfo(tz)).timestamp())
        messages = [m for m in messages if m.created_at >= since_ts]
    if until is not None:
        until_ts = int(
            datetime.combine(until + timedelta(days=1), time.min, tzinfo=ZoneInfo(tz)).timestamp()
        )
        messages = [m for m in messages if m.created_at < until_ts]
    return messages


def _load_messages(
    args: argparse.Namespace, cfg: Config, frag: Fragments
) -> tuple[list[ExportMessage], int]:
    """Messages of the source (``result.json`` or the bot's database) + effective bot_user_id.

    For the bot's database ``--bot-user-id``, if not given (0), is taken automatically
    from the ``is_bot=1`` rows. ``--since``/``--until`` apply to both sources alike.
    """
    source = Path(args.source)
    from_db = bool(getattr(args, "from_db", False))
    chat_id = getattr(args, "chat_id", None)
    bot_user_id = int(args.bot_user_id)

    if _is_db_source(source, from_db):
        messages, detected_bot_user_id = _read_db_messages(source, chat_id)
        if bot_user_id == 0:
            bot_user_id = detected_bot_user_id
    else:
        messages = parse_export(source, cfg.persona.timezone, frag=frag)

    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    messages = _filter_by_date_range(messages, cfg.persona.timezone, since, until)
    return messages, bot_user_id


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sotto.replay",
        description=(
            "Runs the gate (and, with --generate, the full generation pipeline and the "
            "output filter) over a chat history: a Telegram Desktop export (result.json) "
            "or the bot's own database (bot.db/*.sqlite/*.sqlite3, or any path with "
            "--from-db). Sends nothing and writes nothing to the database."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to result.json (an export) or to bot.db/*.sqlite(3) (the bot database)",
    )
    parser.add_argument(
        "--persona",
        type=Path,
        default=None,
        help="Persona pack directory (default: PERSONA_DIR from the environment/.env)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit config.yaml instead of the persona pack's config",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for random.Random (gate step 11)")
    parser.add_argument(
        "--bot-username", default="", help="Bot username without @, used to detect mentions"
    )
    parser.add_argument(
        "--bot-user-id",
        type=int,
        default=0,
        help=(
            "The bot's user_id in the source; 0 = unknown/auto-detect. For a bot "
            "database, when unset, it is detected from the is_bot=1 rows"
        ),
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        dest="from_db",
        help="Treat source as a bot database even if the extension is not .db/.sqlite/.sqlite3",
    )
    parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        dest="chat_id",
        help="chat_id when reading a bot database; required if messages holds several chats",
    )
    parser.add_argument(
        "--since",
        type=_parse_date_arg,
        default=None,
        help="Not before this local date, inclusive (YYYY-MM-DD, persona.timezone)",
    )
    parser.add_argument(
        "--until",
        type=_parse_date_arg,
        default=None,
        help="Not after this local date, inclusive (YYYY-MM-DD, persona.timezone)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every PASS line separately")
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "On every PASS, really generate a reply and run it through the output "
            "filter (needs OPENROUTER_API_KEY in .env). Nothing is sent."
        ),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable the LLM judge (layer 3 of the output filter) with --generate",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=20,
        dest="max_calls",
        help="Cap on real network LLM calls per --generate run (default 20)",
    )
    return parser.parse_args(argv)


def _truncate(text: str, limit: int = _TEXT_TRUNCATE_LEN) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


async def _run_replay_async(args: argparse.Namespace) -> str:
    # gate.py was written in parallel historically, hence the lazy import (see the module
    # docstring).
    # patterns.py is imported at module level (also needed for the Patterns type in _generate_one).
    from sotto.gate import should_consider

    generate = bool(getattr(args, "generate", False))
    use_judge = bool(getattr(args, "judge", False))
    max_calls = int(getattr(args, "max_calls", 20))

    persona_arg = getattr(args, "persona", None)
    config_arg = getattr(args, "config", None)
    # The pack is always needed: media placeholders and the stub name come from its fragments.
    persona = load_persona(Path(persona_arg)) if persona_arg is not None else persona_from_env()
    if config_arg is not None:
        cfg = load_config([default_config_layer(persona.lang), Path(config_arg)])
    else:
        cfg = load_config(persona.config_layers())
    patterns = Patterns(
        cfg.filters,
        cfg.persona.name_triggers,
        args.bot_username,
        persona.lang,
    )
    reserved = {
        cfg.persona.name,
        cfg.persona.display_name,
        *cfg.persona.name_triggers,
        args.bot_username,
    }

    export_messages, effective_bot_user_id = _load_messages(args, cfg, persona.frag)

    rng = random.Random(args.seed)
    state = ReplayState(cfg.persona.timezone, cfg.behaviour.live_talk.window_min)
    bot_message_ids: set[int] = set()

    day_stats: dict[str, DayStats] = {}
    verbose_lines: list[str] = []
    generate_lines: list[str] = []
    gen_stats = GenerateStats()

    gen_ctx: _GenerateContext | None = None
    if generate:
        gen_ctx = _build_generate_context(cfg, use_judge, persona)

    context_buffer: deque[MessageRow] = deque(maxlen=cfg.behaviour.context_window)
    generated_replies: list[str] = []

    try:
        for exported in export_messages:
            is_bot = effective_bot_user_id != 0 and exported.user_id == effective_bot_user_id
            reply_to_bot = (
                exported.reply_to_tg_message_id is not None
                and exported.reply_to_tg_message_id in bot_message_ids
            )
            display_name = sanitize_display_name(
                exported.display_name, exported.user_id, reserved, frag=persona.frag
            )

            msg = GateMessage(
                chat_id=0,
                tg_message_id=exported.tg_message_id,
                user_id=exported.user_id,
                is_bot=is_bot,
                text=exported.text,
                reply_to_bot=reply_to_bot,
                created_at=exported.created_at,
            )

            if not is_bot:
                state.push_recent(msg.user_id, msg.created_at)

            context_buffer.append(
                MessageRow(
                    id=exported.tg_message_id,
                    tg_message_id=exported.tg_message_id,
                    chat_id=0,
                    user_id=exported.user_id,
                    display_name=display_name,
                    text=exported.text,
                    reply_to_tg_message_id=exported.reply_to_tg_message_id,
                    is_bot=is_bot,
                    created_at=exported.created_at,
                )
            )

            gate_state = state.gate_state(msg, msg.created_at)
            decision = should_consider(msg, gate_state, cfg, patterns, msg.created_at, rng)
            state.apply_state_changes(decision.state_changes)

            date_str = local_date(msg.created_at, cfg.persona.timezone).isoformat()
            stats = day_stats.setdefault(date_str, DayStats(date=date_str))
            stats.messages += 1

            if decision.verdict == Verdict.PASS:
                assert decision.trigger is not None
                state.apply_pass(decision.trigger, msg, msg.created_at)
                stats.record_pass(decision.trigger)
                if args.verbose:
                    when = local_dt(msg.created_at, cfg.persona.timezone).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    verbose_lines.append(
                        f"{when} | {display_name} | {decision.trigger.value} | "
                        f"{_truncate(exported.text)}"
                    )
                if gen_ctx is not None:
                    # The --max-calls cap counts REAL network calls (main model + judge),
                    # not the number of PASSes: read llm_calls for this message's day
                    # from the store before deciding whether to generate more. The check
                    # is before the call, so the generation below may raise the counter
                    # by one (main) or two (main + judge) more calls: exceeding max_calls
                    # by at most one judge call is allowed, see the module docstring.
                    calls_so_far = await gen_ctx.store.llm_calls_for_day(
                        exported.created_at, cfg.persona.timezone
                    )
                    if calls_so_far < max_calls:
                        line = await _generate_one(
                            gen_ctx,
                            cfg,
                            cfg.persona.timezone,
                            exported,
                            decision.trigger,
                            display_name,
                            list(context_buffer),
                            generated_replies,
                            gen_stats,
                            patterns,
                        )
                        generate_lines.append(line)
            elif decision.verdict == Verdict.QUEUE_NIGHT:
                stats.night += 1
            else:
                stats.record_drop(decision.reason)

            if is_bot:
                bot_message_ids.add(exported.tg_message_id)
    finally:
        if gen_ctx is not None:
            # Real number of network calls for the whole run (not the number of PASS
            # generations); printed in the report's final line.
            gen_stats.calls = gen_ctx.store.total_llm_calls()
            await gen_ctx.llm.aclose()

    return _format_report(
        day_stats,
        verbose_lines,
        verbose=args.verbose,
        generate_lines=generate_lines,
        gen_stats=gen_stats if generate else None,
    )


def run_replay(args: argparse.Namespace) -> str:
    """Run the gate (and, with ``--generate``, generation) over the export; return the report."""
    return asyncio.run(_run_replay_async(args))


def _format_report(
    day_stats: dict[str, DayStats],
    verbose_lines: list[str],
    verbose: bool,
    generate_lines: list[str] | None = None,
    gen_stats: GenerateStats | None = None,
) -> str:
    lines: list[str] = []

    if generate_lines:
        lines.append("Generation:")
        lines.extend(generate_lines)
        lines.append("")
    elif verbose and verbose_lines:
        lines.append("PASS:")
        lines.extend(verbose_lines)
        lines.append("")

    lines.append("date       | msgs | pass mention/reply/name/ambient | night | drop (top-5)")
    for date_str in sorted(day_stats):
        stats = day_stats[date_str]
        top_drops = sorted(stats.drop_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[
            :_TOP_DROP_REASONS
        ]
        drops_str = ", ".join(f"{reason}={count}" for reason, count in top_drops) or "-"
        lines.append(
            f"{stats.date} | {stats.messages:4d} | "
            f"{stats.pass_mention}/{stats.pass_reply}/{stats.pass_name}/{stats.pass_ambient} | "
            f"{stats.night:5d} | {drops_str}"
        )

    days = len(day_stats)
    total_messages = sum(s.messages for s in day_stats.values())
    total_pass = sum(s.pass_total for s in day_stats.values())
    total_ambient = sum(s.pass_ambient for s in day_stats.values())
    days_without_pass = sum(1 for s in day_stats.values() if s.pass_total == 0)
    avg_pass = total_pass / days if days else 0.0
    avg_ambient = total_ambient / days if days else 0.0

    lines.append("")
    lines.append(
        f"Total: days={days}, messages={total_messages}, "
        f"pass={total_pass} (avg {avg_pass:.2f}/day), "
        f"ambient={total_ambient} (avg {avg_ambient:.2f}/day), "
        f"days without any pass={days_without_pass}"
    )

    if gen_stats is not None:
        top_cuts = sorted(gen_stats.cut_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[
            :_TOP_DROP_REASONS
        ]
        cuts_str = ", ".join(f"{reason}={count}" for reason, count in top_cuts) or "-"
        lines.append(
            f"Generation: calls={gen_stats.calls}, would send={gen_stats.would_send}, "
            f"cut (top 5): {cuts_str}, spent $={gen_stats.cost_usd:.4f}"
        )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(run_replay(args))


if __name__ == "__main__":
    main()
