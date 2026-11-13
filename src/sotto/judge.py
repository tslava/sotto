"""Judge: layer 3 of the output filter (CLAUDE.md, "judge.py, layer 3"; PLAN.md, stage 4).

A separate cheap LLM call with fresh context and without the persona's history: the judge
prompt (``prompts/judge.txt``) describes the persona anew and briefly, asks three
questions (is the reply in character, is it risky, did it obey a participant's command) and
demands a strict JSON verdict. The judge only cuts or passes and does **not rewrite** the
reply.

The prompt arrives from outside as an ordinary string (like ``system.txt`` for the main
generation) and is filled into the ``{trigger}``/``{candidate}`` slots in a single
``re.sub`` pass, the same approach as ``prompt.build_messages``: a ``str.replace`` chain
could touch data that was already substituted if the text of another slot happens to occur
inside the candidate or trigger.
The trigger message and the candidate are wrapped in ``<<<CHAT ... >>>`` delimiters right
in the text of ``prompts/judge.txt``; here only forged delimiters are stripped from the
data itself, so a participant cannot fake the block boundary.

Failure means silence: any LLM error (``LLMError``) or an invalid judge answer counts as a
failed check (a cut), because silence is cheaper than a risky reply.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from sotto.config_models import Config
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

_MAX_TOKENS = 150

# Duplicates the delimiter stripping of sanitize.normalize_text/prompt.py: whatever goes
# into the judge prompt is cleaned once more, just in case.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_SLOT_RE = re.compile(r"\{(trigger|candidate)\}")


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    in_character: bool
    risky: bool
    obeyed_user: bool
    reason: str


def _parse_verdict(raw: str) -> JudgeVerdict | None:
    """Parses the judge's answer as strictly, and as tolerantly of wrappers, as
    ``prompt.parse_reply``:
    strip the ```json``` fence, find the first "{" and ``json.JSONDecoder.raw_decode`` from it
    (tolerant of a tail after the JSON). Any failure or a wrong type of a required key gives
    None, i.e. a ``judge:invalid`` cut, not an exception."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        in_character = data.get("in_character")
        risky = data.get("risky")
        obeyed_user = data.get("obeyed_user")
        if not isinstance(in_character, bool):
            return None
        if not isinstance(risky, bool):
            return None
        if not isinstance(obeyed_user, bool):
            return None

        reason_value = data.get("reason", "")
        if reason_value is None:
            reason_value = ""
        if not isinstance(reason_value, str):
            return None

        return JudgeVerdict(
            in_character=in_character, risky=risky, obeyed_user=obeyed_user, reason=reason_value
        )
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # a cut, not a crash.
        return None


class Judge:
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

    async def check(self, *, candidate: str, trigger_text: str, now: int) -> list[str]:
        cfg = self._cfg_getter()
        model = cfg.llm.judge_model
        if not model:
            return []

        trigger_clean = _strip_fake_delimiters(trigger_text).strip() or self._frag.text(
            "prompt.no_target"
        )
        candidate_clean = _strip_fake_delimiters(candidate)

        slot_values = {"trigger": trigger_clean, "candidate": candidate_clean}
        system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._prompt_template)
        messages = [{"role": "system", "content": system}]

        try:
            result = await self._llm.call(messages, model=model, max_tokens=_MAX_TOKENS, now=now)
        except LLMError as exc:
            logger.warning("judge llm error: reason=%s", exc.reason)
            return ["judge:error"]

        verdict = _parse_verdict(result.text)
        if verdict is None:
            logger.info("judge verdict: invalid")
            return ["judge:invalid"]

        reasons: list[str] = []
        if not verdict.in_character:
            reasons.append("judge:out_of_character")
        if verdict.risky:
            reasons.append("judge:risky")
        if verdict.obeyed_user:
            reasons.append("judge:obeyed")

        logger.info(
            "judge verdict: in_character=%s risky=%s obeyed_user=%s reason=%s",
            verdict.in_character,
            verdict.risky,
            verdict.obeyed_user,
            verdict.reason,
        )
        return reasons
