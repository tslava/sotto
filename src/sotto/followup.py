"""FollowupChecker: a cheap semantic "is this for me?" check (CLAUDE.md,
"attention like a live person: phone in hand").

While the hot window is open after ``/life``/``/say`` (and, later, after any of the
persona's own replies), a person may write to the bot without a reply, without the name and
without ``@``, simply continuing the conversation. The ordinary gate does not treat such a
message as addressing (``gate:not_live``/``gate:dice``/``gate:ambient_cap``/
``gate:ambient_cooldown``/``gate:hot_cap``). ``FollowupChecker`` is a second, cheap LLM call
(the same approach as ``judge.Judge`` and ``stickers.StickerChooser``): a short prompt with
recent chat context, the persona's latest replies and the new message, answered strictly as
JSON ``{"addressed": true|false, "reason": "..."}``. A "yes" turns the message into a
``Trigger.FOLLOWUP`` addressing.

The same approach is reused twice more: ``check_batch`` is a cheap prefilter for a batch
before the "come back and check" pass, and ``check_name`` confirms addressing by name
(CLAUDE.md, "name in cases and name-addressing check"), where the regex sees the name in any
case but cannot tell "Fedya, where are you?" from "the old man put a smiley yesterday".

The module does not touch the DB: context and recent replies are passed as arguments
(``bot.py`` prepares them and writes ``filter_log``). A failure (invalid JSON, LLMError,
empty model) is resolved in each check's own cheaper direction:
``check`` -> ``False`` (silence is cheaper than answering the wrong person),
``check_name`` -> ``True`` (missing an addressing is worse than one extra reply),
``check_batch`` -> all numbers (let the main model decide).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sotto.config_models import Config
from sotto.db import MessageRow
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError
from sotto.prompt import render_context, render_numbered_messages

logger = logging.getLogger(__name__)

_COUNTER_KEY = "followup_calls"

# Duplicates the delimiter stripping of sanitize.normalize_text/prompt.py/judge.py:
# whatever goes into the check prompt is cleaned once more, just in case.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_SLOT_RE = re.compile(r"\{(context|recent_replies|name|text)\}")
_BATCH_SLOT_RE = re.compile(r"\{(messages|recent_replies|names)\}")


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


@dataclass(frozen=True, slots=True)
class _FollowupVerdict:
    addressed: bool
    reason: str


def _parse_verdict(raw: str, key: str = "addressed") -> _FollowupVerdict | None:
    """Parses a check answer as strictly and tolerantly as ``judge._parse_verdict``:
    strip the ```json``` fence, find the first "{" and ``json.JSONDecoder.raw_decode`` from
    it. Any failure or a wrong boolean type -> None. What None means is up to the caller:
    for ``check`` it is "not addressed", for ``check_name`` the opposite, "could not check,
    so answer".

    ``key`` is the name of the boolean field in the answer: ``addressed`` for the hot-window
    check, ``answer`` for the name-addressing check (different prompts, same approach)."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        addressed = data.get(key)
        if not isinstance(addressed, bool):
            return None

        reason_value = data.get("reason", "")
        if reason_value is None:
            reason_value = ""
        if not isinstance(reason_value, str):
            return None

        return _FollowupVerdict(addressed=addressed, reason=reason_value)
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # "not addressed", not a crash.
        return None


def _parse_batch_verdict(raw: str) -> list[int] | None:
    """Parses the cheap prefilter's ``{"addressed": [numbers]}`` the same way as
    ``_parse_verdict``/``judge._parse_verdict``: strip the ```json``` fence, find the first
    "{" and a tolerant ``raw_decode``. Any failure or a non-list ``addressed`` -> None
    ("could not check": the caller returns all numbers). List items that are not integers
    (``bool`` is not a number either: ``true``/``false`` instead of a number is a typical
    JSON slip) are simply dropped instead of failing the whole parse; the caller checks the
    range anyway."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        addressed = data.get("addressed")
        if not isinstance(addressed, list):
            return None

        return [item for item in addressed if isinstance(item, int) and not isinstance(item, bool)]
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # "could not check", not a crash.
        return None


class FollowupChecker:
    def __init__(
        self,
        llm: LLMClient,
        cfg_getter: Callable[[], Config],
        prompt_template: str,
        batch_prompt_template: str = "",
        name_prompt_template: str = "",
        *,
        frag: Fragments,
    ) -> None:
        self._frag = frag
        self._llm = llm
        self._cfg_getter = cfg_getter
        self._prompt_template = prompt_template
        self._batch_prompt_template = batch_prompt_template
        self._name_prompt_template = name_prompt_template

    def _render_single(
        self,
        template: str,
        *,
        text: str,
        display_name: str,
        context_rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
    ) -> str:
        """Fills the {context}/{recent_replies}/{name}/{text} slots in one pass (shared by
        ``check`` and ``check_name``: different prompts, same slots)."""
        context = _strip_fake_delimiters(render_context(list(context_rows))).strip()
        recent = _strip_fake_delimiters("\n".join(recent_replies)).strip()
        slot_values = {
            "context": context or self._frag.text("placeholders.context_empty"),
            "recent_replies": recent or self._frag.text("placeholders.recent_replies_empty"),
            "name": _strip_fake_delimiters(display_name).strip(),
            "text": _strip_fake_delimiters(text).strip(),
        }
        return _SLOT_RE.sub(lambda m: slot_values[m.group(1)], template)

    async def check(
        self,
        *,
        text: str,
        display_name: str,
        context_rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
        now: int,
    ) -> bool:
        cfg = self._cfg_getter()
        followup_cfg = cfg.behaviour.followup
        model = followup_cfg.model or cfg.llm.judge_model
        if not model:
            return False

        system = self._render_single(
            self._prompt_template,
            text=text,
            display_name=display_name,
            context_rows=context_rows,
            recent_replies=recent_replies,
        )
        messages = [{"role": "system", "content": system}]

        try:
            result = await self._llm.call(
                messages,
                model=model,
                max_tokens=followup_cfg.max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=followup_cfg.daily_cap,
            )
        except LLMError as exc:
            logger.warning("followup checker llm error: reason=%s", exc.reason)
            return False

        verdict = _parse_verdict(result.text)
        if verdict is None:
            logger.info("followup verdict: invalid")
            return False

        logger.info("followup verdict: addressed=%s reason=%s", verdict.addressed, verdict.reason)
        return verdict.addressed

    async def check_name(
        self,
        *,
        text: str,
        display_name: str,
        context_rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
        now: int,
    ) -> bool:
        """Confirms addressing by name (CLAUDE.md, "name in cases and name-addressing
        check").

        The regex now catches the name in every case but cannot tell an addressing
        ("Fedya, where are you?") from talk about the persona in the third person ("the old
        man put a smiley under the message yesterday"). The cheap model tells them apart: the
        same mechanics as ``check``, with a different prompt and answer field (``answer``).

        A failure (no model, no template, ``LLMError``, invalid JSON) gives ``True``, the
        mirror of ``check``: there silence is cheaper than an error, here missing a real
        addressing is worse than one extra reply.
        """
        cfg = self._cfg_getter()
        followup_cfg = cfg.behaviour.followup
        model = followup_cfg.model or cfg.llm.judge_model
        if not model or not self._name_prompt_template:
            logger.warning("name check: no model or prompt template, treating as addressed")
            return True

        system = self._render_single(
            self._name_prompt_template,
            text=text,
            display_name=display_name,
            context_rows=context_rows,
            recent_replies=recent_replies,
        )
        messages = [{"role": "system", "content": system}]

        try:
            result = await self._llm.call(
                messages,
                model=model,
                max_tokens=followup_cfg.name_check_max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=followup_cfg.daily_cap,
            )
        except LLMError as exc:
            logger.warning("name check llm error: reason=%s", exc.reason)
            return True

        verdict = _parse_verdict(result.text, key="answer")
        if verdict is None:
            logger.warning("name check: invalid verdict, treating as addressed")
            return True

        logger.info("name check verdict: answer=%s reason=%s", verdict.addressed, verdict.reason)
        return verdict.addressed

    async def check_batch(
        self,
        *,
        rows: Sequence[MessageRow],
        recent_replies: Sequence[str],
        now: int,
    ) -> list[int]:
        """Cheap prefilter for the "come back and check" pass (CLAUDE.md, "cheap prefilter
        for the 'come back and check' pass"): one cheap-model call over the whole batch
        ``rows`` accumulated since the persona's last reply, instead of the main model on
        every ``responder._maybe_checkin`` check.

        Returns 1-based numbers of messages in ``rows`` addressed to the persona or directly
        continuing its topic. An empty list means "no need to call the main model, nothing
        here is for the persona". When there is nothing to check with (no model, no
        ``batch_prompt_template``, ``LLMError``, invalid JSON) a cheaper decision is not
        possible, so ALL numbers are returned: let the main model decide, costlier but with a
        chance to answer.
        """
        if not rows:
            return []

        all_numbers = list(range(1, len(rows) + 1))
        cfg = self._cfg_getter()
        followup_cfg = cfg.behaviour.followup
        model = followup_cfg.model or cfg.llm.judge_model
        if not model or not self._batch_prompt_template:
            logger.warning("checkin prefilter: no model or prompt template, not filtering")
            return all_numbers

        messages_block = _strip_fake_delimiters(render_numbered_messages(rows)).strip()
        recent = _strip_fake_delimiters("\n".join(recent_replies)).strip()

        # How the persona is called in the chat: without this the cheap model knows only the
        # persona's name and drops addressings by a nickname; they never reach the main model.
        persona = cfg.persona
        names = list(dict.fromkeys([persona.name, persona.display_name, *persona.name_triggers]))
        slot_values = {
            "messages": messages_block or self._frag.text("placeholders.messages_empty"),
            "recent_replies": recent or self._frag.text("placeholders.recent_replies_empty"),
            "names": ", ".join(name for name in names if name),
        }
        system = _BATCH_SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._batch_prompt_template)
        messages = [{"role": "system", "content": system}]

        try:
            result = await self._llm.call(
                messages,
                model=model,
                max_tokens=cfg.behaviour.checkin.prefilter_max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=followup_cfg.daily_cap,
            )
        except LLMError as exc:
            logger.warning("checkin prefilter llm error: reason=%s", exc.reason)
            return all_numbers

        numbers = _parse_batch_verdict(result.text)
        if numbers is None:
            logger.warning("checkin prefilter: invalid verdict, not filtering")
            return all_numbers

        filtered = [number for number in numbers if 1 <= number <= len(rows)]
        logger.info("checkin prefilter verdict: addressed=%s", filtered)
        return filtered
