"""Long-term chat memory (CLAUDE.md, "long-term chat memory").

Messages live for ``behaviour.message_retention_days`` (30 days by default), yet the
persona should remember what was discussed months ago. Once a week (the step is
``chat_memory.period_days``) one model call compresses the past period into a few lines;
the recap sits in the ``chat_memory`` table longer than the messages themselves
(``keep_days``) and is mixed into the system prompt through the ``{chat_memory}`` slot.

This is a deliberate owner decision on privacy: a recap of other people's conversations is
kept longer than the conversations. As compensation, messages of muted participants never
reach the recap (``exclude_user_ids`` in ``messages_between``).

Period boundaries are always local midnight in ``persona.timezone``, and the step counts
calendar days, not ``period_days * 86400``: across a DST change a day lasts 23 or 25 hours,
and second-based arithmetic would drift the boundary from midnight to 23:00/01:00 and then
further and further. ``_add_days`` recomputes via the local date and is therefore
independent of DST.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sotto.config_models import Config
from sotto.db import ChatMemoryRow, Database
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError
from sotto.patterns import Patterns
from sotto.sanitize import normalize_text
from sotto.timeutil import in_window, local_date

logger = logging.getLogger(__name__)

# Fewer lines than this in a period: nothing to recap, the model is not called.
_MIN_LINES = 5

# job() loop modelled on retention.retention_loop: once an hour check whether it is time.
_JOB_INTERVAL_SEC = 3600
# After a failed iteration, a short pause instead of the usual one, so there is no busy
# loop yet no wait of a whole hour either (same approach as responder.checkin_job).
_ERROR_RETRY_SEC = 60

# People's data goes inside <<<CHAT ... >>>, forged delimiters are stripped from it
# (same approach as prompt.py/judge.py/followup.py).
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_SLOT_RE = re.compile(r"\{(period|chat|limit)\}")

# Jokes and people's threads (CLAUDE.md, "chat jokes and people's threads"): length caps
# from the contract; answer tokens: jokes from the contract, threads with room for JSON.
_JOKE_MAX_LEN = 80
_JOKES_MAX_TOKENS = 200
_THREAD_MAX_LEN = 120
_THREADS_MAX_TOKENS = 300
_THREADS_MAX_PER_PERIOD = 5
_EXISTING_JOKES_LOOKUP = 200
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

# Leading list markers and numbering that the model likes to add despite the
# instruction. They are stripped: a recap line starting with "- " would reach the
# system prompt as a bullet and filters.regex:prompt_leak would treat it as an instruction.
_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]+\s+|\d+[.)]\s+)")
# A line where the model started talking about the answer format rather than the chat.
_JSON_RE = re.compile(r"json", re.IGNORECASE)


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _parse_json_object(raw: str) -> dict[str, object] | None:
    """Tolerant parse of a model answer (like ``judge._parse_verdict``): strip the ```json```
    fence, take the first "{" and ``raw_decode``. Any failure -> None."""
    try:
        candidate = raw.strip()
        match = _CODE_FENCE_RE.match(candidate)
        if match is not None:
            candidate = match.group(1).strip()
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # "found nothing", not a crash.
        return None


def _clean_note(raw: object, max_len: int) -> str | None:
    """A line from a model answer -> a note for the DB/prompt: no delimiters, bullet or
    line breaks; empty or longer than ``max_len`` (the model retold instead of
    compressing) -> None."""
    if not isinstance(raw, str):
        return None
    cleaned = normalize_text(_BULLET_RE.sub("", _strip_fake_delimiters(raw))).strip()
    if not cleaned or len(cleaned) > max_len:
        return None
    return cleaned


def _local_midnight(ts: int, tz: str) -> int:
    """Start of the local day that ``ts`` falls into."""
    zone = ZoneInfo(tz)
    day = local_date(ts, tz)
    return int(datetime(day.year, day.month, day.day, tzinfo=zone).timestamp())


def _add_days(ts: int, days: int, tz: str) -> int:
    """``ts`` (local midnight) plus ``days`` calendar days, which is midnight again.

    Calendar days specifically: on the spring-forward day there are 23 hours, on the fall-back
    day 25, and adding ``days * 86400`` would shift the period boundary off midnight.
    """
    zone = ZoneInfo(tz)
    day = local_date(ts, tz) + timedelta(days=days)
    return int(datetime(day.year, day.month, day.day, tzinfo=zone).timestamp())


def format_period(period_start: int, period_end: int, tz: str) -> str:
    """ "08.09-14.09.2026": the period end is exclusive, so show the last day.

    Public: the same period format labels recaps in ``/memory list`` in commands.py."""
    start_day = local_date(period_start, tz)
    # period_end is exclusive and falls on the midnight of the next day, so the last
    # day of the period is period_end minus one second.
    end_day = local_date(max(period_end - 1, period_start), tz)
    return f"{start_day.strftime('%d.%m')}–{end_day.strftime('%d.%m.%Y')}"


def render_chat_memory(rows: Sequence[ChatMemoryRow], tz: str, *, frag: Fragments) -> str:
    """Long-memory block for the system prompt. Empty -> "" (the whole paragraph disappears).

    Recap lines are indented by two spaces and NEVER start with "- ":
    ``filters.regex:prompt_leak`` treats bullet lines as the instructional part of the
    prompt, and a memory starting with a hyphen would cut the persona's reply as a prompt
    leak.
    """
    if not rows:
        return ""
    lines = [frag.text("render.chat_memory_header")]
    for row in rows:
        lines.append(f"{format_period(row.period_start, row.period_end, tz)}:")
        for line in row.text.splitlines():
            cleaned = line.strip()
            if cleaned:
                lines.append(f"  {cleaned}")
    return "\n".join(lines)


def _clean_summary(raw: str, max_chars: int) -> str:
    """Model answer -> recap body: one line per memory, no junk.

    Empty lines are dropped, leading bullets and numbering are stripped (see
    ``render_chat_memory``), lines where the model started talking about JSON instead of the
    chat are dropped whole. Trimming to ``max_chars`` is at a line boundary: a phrase cut
    mid-word looks worse in the prompt than one line fewer.
    """
    kept: list[str] = []
    total = 0
    for line in raw.splitlines():
        cleaned = normalize_text(_BULLET_RE.sub("", line))
        if not cleaned or _JSON_RE.search(cleaned):
            continue
        extra = len(cleaned) + (1 if kept else 0)
        if total + extra > max_chars:
            break
        kept.append(cleaned)
        total += extra
    return "\n".join(kept)


class ChatMemorizer:
    def __init__(
        self,
        llm: LLMClient,
        db: Database,
        cfg_getter: Callable[[], Config],
        prompt_template: str,
        chat_id: int,
        clock: Callable[[], int] = lambda: int(time.time()),
        interval_sec: int = _JOB_INTERVAL_SEC,
        jokes_prompt: str = "",
        threads_prompt: str = "",
        patterns_getter: Callable[[], Patterns] | None = None,
    ) -> None:
        self._llm = llm
        self._db = db
        self._cfg_getter = cfg_getter
        self._prompt_template = prompt_template
        self._chat_id = chat_id
        self._clock = clock
        self._interval_sec = interval_sec
        self._jokes_prompt = jokes_prompt
        self._threads_prompt = threads_prompt
        # Jokes and threads go into the system prompt for a long time: commands and talk
        # about the model are not let in (Patterns.unsafe_note). None means no check (tests).
        self._patterns_getter = patterns_getter

    def _unsafe(self, text: str) -> bool:
        if self._patterns_getter is None:
            return False
        hit = self._patterns_getter().unsafe_note(text)
        if hit is not None:
            logger.warning("chat memory: note dropped, marker %r matched", hit)
        return hit is not None

    async def summarize_period(self, start: int, end: int, *, now: int) -> ChatMemoryRow | None:
        """One model call for the period ``[start, end)``; a DB row or None.

        None (without a model call) if the period has fewer than ``_MIN_LINES`` lines:
        nothing to recap and a call costs money. None (after a call) if the model answered
        with nothing or failed: skipping a period is better than junk in the system prompt
        forever.
        """
        cfg = self._cfg_getter()
        memory_cfg = cfg.behaviour.chat_memory
        model = memory_cfg.model or cfg.llm.main_model
        if not model:
            logger.warning("chat memory: model not set, recap skipped")
            return None

        muted = await self._db.muted_user_ids()
        messages = await self._db.messages_between(
            self._chat_id, start, end, exclude_user_ids=muted
        )
        bot_replies = await self._db.bot_replies_between(start, end)

        bot_name = cfg.persona.name
        lines: list[tuple[int, str]] = [
            (row.created_at or 0, f"{row.display_name}: {row.text}") for row in messages if row.text
        ]
        lines.extend((row.created_at, f"{bot_name}: {row.text}") for row in bot_replies if row.text)
        lines.sort(key=lambda item: item[0])
        tail = [line for _ts, line in lines[-memory_cfg.max_messages :]]
        if len(tail) < _MIN_LINES:
            logger.info(
                "chat memory: period %s skipped, %d lines",
                format_period(start, end, cfg.persona.timezone),
                len(tail),
            )
            return None

        chat_block = _strip_fake_delimiters("\n".join(tail))
        slot_values = {
            "period": format_period(start, end, cfg.persona.timezone),
            "chat": chat_block,
        }
        prompt = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._prompt_template)

        try:
            result = await self._llm.call(
                [{"role": "system", "content": prompt}],
                model=model,
                max_tokens=memory_cfg.max_tokens,
                now=now,
            )
        except LLMError as exc:
            logger.warning("chat memory llm error: reason=%s", exc.reason)
            return None

        text = _clean_summary(result.text, memory_cfg.max_chars)
        if not text:
            logger.warning(
                "chat memory: empty recap for period %s",
                format_period(start, end, cfg.persona.timezone),
            )
            return None

        memory_id = await self._db.insert_chat_memory(
            period_start=start, period_end=end, text=text, created_at=now
        )
        logger.info(
            "chat memory: period %s recapped, %d lines, %d chars",
            format_period(start, end, cfg.persona.timezone),
            len(tail),
            len(text),
        )
        # Two extra calls over the same period lines. The recap is already saved, so any
        # failure of theirs (except cancellation) does not break the recap.
        try:
            await self._extract_jokes(cfg, model, start, end, chat_block, now)
            await self._extract_threads(cfg, model, start, end, chat_block, muted, now)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("chat memory: joke/thread extraction failed")
        return ChatMemoryRow(
            id=memory_id, period_start=start, period_end=end, text=text, created_at=now
        )

    async def _call_extra(
        self,
        template: str,
        *,
        model: str,
        max_tokens: int,
        period: str,
        chat_block: str,
        limit: int,
        now: int,
        what: str,
    ) -> dict[str, object] | None:
        slot_values = {"period": period, "chat": chat_block, "limit": str(limit)}
        prompt = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], template)
        try:
            result = await self._llm.call(
                [{"role": "system", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                now=now,
            )
        except LLMError as exc:
            logger.warning("chat memory %s llm error: reason=%s", what, exc.reason)
            return None
        data = _parse_json_object(result.text)
        if data is None:
            logger.warning("chat memory %s: model answer is not JSON", what)
        return data

    async def _extract_jokes(
        self, cfg: Config, model: str, start: int, end: int, chat_block: str, now: int
    ) -> None:
        jokes_cfg = cfg.behaviour.jokes
        if not (jokes_cfg.enabled and self._jokes_prompt and jokes_cfg.per_period > 0):
            return
        data = await self._call_extra(
            self._jokes_prompt,
            model=model,
            max_tokens=_JOKES_MAX_TOKENS,
            period=format_period(start, end, cfg.persona.timezone),
            chat_block=chat_block,
            limit=jokes_cfg.per_period,
            now=now,
            what="jokes",
        )
        raw_items = data.get("jokes") if data is not None else None
        if not isinstance(raw_items, list):
            return
        known = {row.text.casefold() for row in await self._db.jokes(_EXISTING_JOKES_LOOKUP)}
        saved = 0
        for raw in raw_items:
            joke = _clean_note(raw, _JOKE_MAX_LEN)
            if joke is None or joke.casefold() in known or self._unsafe(joke):
                continue
            known.add(joke.casefold())
            await self._db.insert_joke(text=joke, created_at=now)
            saved += 1
            if saved >= jokes_cfg.per_period:
                break
        logger.info("chat memory: jokes saved %d", saved)

    async def _extract_threads(
        self,
        cfg: Config,
        model: str,
        start: int,
        end: int,
        chat_block: str,
        muted: frozenset[int],
        now: int,
    ) -> None:
        if not (cfg.behaviour.callback.enabled and self._threads_prompt):
            return
        data = await self._call_extra(
            self._threads_prompt,
            model=model,
            max_tokens=_THREADS_MAX_TOKENS,
            period=format_period(start, end, cfg.persona.timezone),
            chat_block=chat_block,
            limit=_THREADS_MAX_PER_PERIOD,
            now=now,
            what="threads",
        )
        raw_items = data.get("threads") if data is not None else None
        if not isinstance(raw_items, list):
            return
        bot_names = {
            name.casefold() for name in (cfg.persona.name, cfg.persona.display_name) if name
        }
        # The same thread must not be saved twice, neither from one model answer nor on
        # top of one already open: otherwise "so how did it go?" would be asked again.
        known = {(row.user_id, row.text.casefold()) for row in await self._db.open_people_threads()}
        saved = 0
        for item in raw_items[:_THREADS_MAX_PER_PERIOD]:
            if not isinstance(item, dict):
                continue
            name = _clean_note(item.get("name"), _JOKE_MAX_LEN)
            text = _clean_note(item.get("text"), _THREAD_MAX_LEN)
            if name is None or text is None or name.casefold() in bot_names:
                continue
            if self._unsafe(text):
                continue
            # Only an exact display_name match among the period's messages, and only if
            # one person wrote under that name: a name from the model's answer is no reason
            # to attach a thread to a stranger or a namesake.
            user_id = await self._db.user_id_by_display_name(self._chat_id, name, start, end)
            if user_id is None or user_id in muted:
                continue
            key = (user_id, text.casefold())
            if key in known:
                continue
            known.add(key)
            await self._db.insert_people_thread(
                user_id=user_id, display_name=name, text=text, created_at=now
            )
            saved += 1
        logger.info("chat memory: threads saved %d", saved)

    async def run_due(self, *, now: int) -> list[ChatMemoryRow]:
        """Recaps every period that fits entirely before the start of the current day.

        A period always ends on a local-day boundary, so "today" is never recapped, only
        completed days. No recaps yet -> catch up ``backfill_periods`` periods back, but not
        earlier than the chat's first message. A repeated run in the same window does
        nothing: ``last_chat_memory_end`` already equals the end of the last period.
        """
        cfg = self._cfg_getter()
        memory_cfg = cfg.behaviour.chat_memory
        tz = cfg.persona.timezone
        period_days = memory_cfg.period_days

        end = _local_midnight(now, tz)
        start = await self._db.last_chat_memory_end()
        if start is None:
            first_at = await self._db.first_message_at(self._chat_id)
            if first_at is None:
                return []
            start = max(
                _add_days(end, -period_days * memory_cfg.backfill_periods, tz),
                _local_midnight(first_at, tz),
            )

        created: list[ChatMemoryRow] = []
        while True:
            period_end = _add_days(start, period_days, tz)
            if period_end > end:
                break
            row = await self.summarize_period(start, period_end, now=now)
            if row is not None:
                created.append(row)
            start = period_end
        return created

    async def job(self) -> None:
        """Endless loop modelled on ``retention.retention_loop``: once an hour it checks
        whether the current moment falls into ``chat_memory.run_window`` and if so calls
        ``run_due``. A second pass within the same window is safe: ``run_due`` will find
        nothing. Cancellation is propagated, any other error is logged and the loop lives on."""
        while True:
            try:
                cfg = self._cfg_getter()
                memory_cfg = cfg.behaviour.chat_memory
                now = self._clock()
                if memory_cfg.enabled and in_window(
                    now, cfg.persona.timezone, memory_cfg.run_window
                ):
                    await self.run_due(now=now)
            except asyncio.CancelledError:
                logger.info("chat memory job: cancelled, stopping")
                raise
            except Exception:
                logger.exception("chat memory job: iteration failed")
                await asyncio.sleep(_ERROR_RETRY_SEC)
                continue
            await asyncio.sleep(self._interval_sec)
