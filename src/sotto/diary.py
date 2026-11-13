"""Diary of the day (CLAUDE.md, "diary of the day").

The reason: through ``/say`` the persona once said it was heading out somewhere and minutes
later said it was doing something incompatible, and the chat noticed. The line was in "your
latest replies" but the model does not hold it as a fact. ``DiaryExtractor`` is a cheap LLM
call (the same approach as ``followup.FollowupChecker``): from the final reply text it pulls
a fact about the persona's own doings, and ``render_diary`` assembles the day's facts into a
paragraph for the ``{diary}`` slot of the system prompt: "stick to this, do not contradict".

The module does not touch the DB: the caller (``responder.py``) writes the fact.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence

from sotto.config_models import Config
from sotto.db import SelfFactRow
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError
from sotto.sanitize import normalize_text
from sotto.timeutil import local_dt

logger = logging.getLogger(__name__)

_COUNTER_KEY = "diary_calls"
_FACT_MAX_LEN = 80

_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_SLOT_RE = re.compile(r"\{(reply)\}")


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def _parse_fact(raw: str) -> str | None:
    """``{"fact": "..."|null}`` -> a normalized fact or None.

    Parsing is as tolerant as ``judge._parse_verdict``: strip the ```json``` fence,
    the first "{" and ``raw_decode``. A non-string, an empty string and one longer than
    80 characters (the model retold instead of compressing) give None."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None
        value = data.get("fact")
        if not isinstance(value, str):
            return None
        fact = normalize_text(_strip_fake_delimiters(value))
        if not fact or len(fact) > _FACT_MAX_LEN:
            return None
        return fact
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # "no fact", not a crash.
        return None


class DiaryExtractor:
    def __init__(
        self, llm: LLMClient, cfg_getter: Callable[[], Config], prompt_template: str
    ) -> None:
        self._llm = llm
        self._cfg_getter = cfg_getter
        self._prompt_template = prompt_template

    async def extract(self, reply_text: str, *, now: int) -> str | None:
        """A fact about the persona's doings from its reply, or None. Any failure (empty
        model, ``LLMError``, not JSON, too long a fact) gives None; the chat reply does not
        depend on it."""
        cfg = self._cfg_getter()
        diary_cfg = cfg.behaviour.diary
        model = diary_cfg.model or cfg.llm.judge_model
        if not model:
            return None

        slot_values = {"reply": _strip_fake_delimiters(reply_text).strip()}
        system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._prompt_template)
        try:
            result = await self._llm.call(
                [{"role": "system", "content": system}],
                model=model,
                max_tokens=diary_cfg.max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=diary_cfg.daily_cap,
            )
        except LLMError as exc:
            logger.warning("diary llm error: reason=%s", exc.reason)
            return None

        fact = _parse_fact(result.text)
        if fact is None:
            logger.info("diary: no fact in the reply")
            return None
        logger.info("diary: %s", fact)
        return fact


def _clean(text: str) -> str:
    return _strip_fake_delimiters(text).strip().rstrip(".;")


def render_diary(
    today: Sequence[SelfFactRow], week: Sequence[SelfFactRow], tz: str, *, frag: Fragments
) -> str:
    """Block for the ``{diary}`` slot. Empty -> "" (the slot's paragraph in system.txt disappears).

    Today looks like "10:46 went to the sauna; 12:39 tinkering with the boat.", the week like
    "Mon fixed the fence; Wed bought paint." (weekday and time are local per tz). Lines do
    not start with "- " (``filters.regex:prompt_leak``)."""
    lines: list[str] = []
    if today:
        parts = [f"{local_dt(row.created_at, tz):%H:%M} {_clean(row.text)}" for row in today]
        lines.append(frag.text("render.diary_today_header") + "; ".join(parts) + ".")
    if week:
        weekdays = frag.items("render.weekdays")
        parts = [
            f"{weekdays[local_dt(row.created_at, tz).weekday()]} {_clean(row.text)}" for row in week
        ]
        lines.append(frag.text("render.diary_week_header") + "; ".join(parts) + ".")
    return "\n".join(lines)
