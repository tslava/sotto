"""Stickers instead of a text reply: a second call to a cheap model (CLAUDE.md,
"Interfaces: stickers"; PLAN.md, section "Stickers").

The main model knows nothing about the sticker catalog: the menu is never mixed into its
prompt (the owner's decision: a costlier main request is not worth it, and reply speed does
not matter). Instead the FINISHED text that already passed the output filter, together with
the trigger message and the catalog, goes to a separate cheap call (the same model as the
judge, ``cfg.llm.judge_model``, or its own ``cfg.behaviour.stickers.model``), which either
names a sticker number or answers ``null``, in which case the text goes to the chat as before.

The prompt lives in ``prompts/sticker.txt`` and, like ``prompts/judge.txt``, wraps the data
({trigger}/{reply}) in ``<<<CHAT ... >>>`` delimiters itself; here only forged delimiters
are stripped from the data (the same approach as ``judge.py``), and slots are filled in one
``re.sub`` pass.

The catalog (``stickers.yaml``) is built offline by the vision-based ``stickers_fill.py``
script and edited by the owner by hand; ``load_catalog`` is a pure load with no network,
and a missing/empty file means "no stickers", not an error.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from sotto.config_models import Config, StickersConfig
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Duplicates the delimiter stripping of sanitize.normalize_text/prompt.py/judge.py:
# whatever goes into the chooser prompt is cleaned once more, just in case.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_SLOT_RE = re.compile(r"\{(menu|trigger|reply)\}")

# "no file -> WARNING once" (CLAUDE.md) means once per process, not per path: hot-reloading
# the catalog is not planned in this bot, so a repeated load_catalog call with the same
# missing file would only spam the same warning.
_missing_catalog_warned = False


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


class Sticker(BaseModel):
    id: int
    file_id: str
    emoji: str = ""
    text: str
    when: str = ""
    enabled: bool = True


class StickerCatalog(BaseModel):
    set_name: str = ""
    stickers: list[Sticker] = Field(default_factory=list)


def load_catalog(path: Path) -> StickerCatalog:
    """``stickers.yaml`` -> ``StickerCatalog``. File missing or empty -> an empty catalog
    (everything else in the bot works as before, there are just no stickers).
    Broken YAML/data that does not fit the schema -> ``ValueError`` (including
    ``pydantic.ValidationError``, which subclasses it)."""
    global _missing_catalog_warned
    if not path.exists():
        if not _missing_catalog_warned:
            _missing_catalog_warned = True
            logger.warning("stickers: file %s not found, sticker catalog is empty", path)
        return StickerCatalog()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid stickers yaml {path}: {exc}") from exc

    if not raw:
        return StickerCatalog()
    if not isinstance(raw, dict):
        raise ValueError(f"stickers file {path} must contain a YAML mapping")
    return StickerCatalog.model_validate(raw)


def sticker_tag_re(frag: Fragments) -> re.Pattern[str]:
    """Regex with which responder.py extracts a sticker id from the text written by
    insert_bot_reply (see render_sticker_menu/StickerChooser.choose below and the
    responder.py module docstring on "recent_ids"): "[sticker #3] Nice one" -> 3.
    Built from the package's ``placeholders.sticker_tag``: the text before and after ``{id}``
    is escaped, and ``(\\d+)`` stands in place of ``{id}``."""
    template: str = frag.text("placeholders.sticker_tag")
    head, _, tail = template.partition("{id}")
    return re.compile(f"^{re.escape(head)}(\\d+){re.escape(tail)}")


def render_sticker_menu(stickers: list[Sticker], *, frag: Fragments) -> str:
    """ "1: "Nice one" - surprise, admiration\\n2: ..." - enabled only."""
    quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
    lines = []
    for sticker in stickers:
        if not sticker.enabled:
            continue
        if sticker.when:
            lines.append(f"{sticker.id}: {quote_open}{sticker.text}{quote_close} — {sticker.when}")
        else:
            lines.append(f"{sticker.id}: {quote_open}{sticker.text}{quote_close}")
    return "\n".join(lines)


def parse_choice(raw: str, valid_ids: set[int]) -> int | None:
    """Parses the chooser's answer as strictly and tolerantly as ``judge._parse_verdict``/
    ``prompt.parse_reply``: strip the ```json``` fence, find the first "{" and
    ``json.JSONDecoder.raw_decode`` from it. Expects ``{"sticker": int|null}``;
    anything that is not an integer from ``valid_ids`` (including ``bool``, a subclass of
    ``int`` in Python but not what is meant here) means ``None``."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        sticker_id = data.get("sticker")
        if sticker_id is None or isinstance(sticker_id, bool):
            return None
        if not isinstance(sticker_id, int):
            return None
        if sticker_id not in valid_ids:
            return None
        return sticker_id
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # "no sticker chosen", not a crash.
        return None


def recent_sticker_ids(texts: Sequence[str], window: int, *, frag: Fragments) -> set[int]:
    """Ids of stickers used in the bot's last ``window`` replies marked with the
    ``sticker_tag_re`` tag: not over all texts but over the last ``window`` sticker ones.
    ``texts`` is in chronological order (as ``Database.recent_bot_replies`` returns it),
    the freshest last."""
    tag_re = sticker_tag_re(frag)
    ids: list[int] = []
    for text in reversed(texts):
        match = tag_re.match(text)
        if match is not None:
            ids.append(int(match.group(1)))
            if len(ids) >= window:
                break
    return set(ids)


def sticker_allowed(*, cfg: StickersConfig, replies_since: int, count_today: int) -> bool:
    """Whether a sticker can be considered at all right now, before the chooser is called
    (CLAUDE.md: "min_replies_between not satisfied -> the chooser is not called at all")."""
    if not cfg.enabled:
        return False
    if replies_since < cfg.min_replies_between:
        return False
    return count_today < cfg.daily_cap


class StickerChooser:
    """Second, cheap LLM call: picks a sticker for an already finished reply, or
    ``None``. The decision "may a sticker be offered at all right now"
    (``sticker_allowed``) is made by the caller (``responder.py``); only candidates that
    are already filtered get here."""

    def __init__(
        self,
        llm: LLMClient,
        cfg_getter: Callable[[], Config],
        catalog: StickerCatalog,
        prompt_template: str,
        *,
        frag: Fragments,
    ) -> None:
        self._llm = llm
        self._cfg_getter = cfg_getter
        self._catalog = catalog
        self._prompt_template = prompt_template
        self._frag = frag

    async def choose(
        self,
        *,
        reply_text: str,
        trigger_text: str,
        exclude_ids: set[int],
        now: int,
    ) -> Sticker | None:
        candidates = [
            sticker
            for sticker in self._catalog.stickers
            if sticker.enabled and sticker.id not in exclude_ids
        ]
        if not candidates:
            return None

        cfg = self._cfg_getter()
        model = cfg.behaviour.stickers.model or cfg.llm.judge_model
        if not model:
            return None

        menu = render_sticker_menu(candidates, frag=self._frag)
        trigger_clean = _strip_fake_delimiters(trigger_text).strip() or self._frag.text(
            "prompt.no_target"
        )
        reply_clean = _strip_fake_delimiters(reply_text).strip()

        slot_values = {"menu": menu, "trigger": trigger_clean, "reply": reply_clean}
        system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._prompt_template)
        messages = [{"role": "system", "content": system}]

        try:
            result = await self._llm.call(
                messages, model=model, max_tokens=cfg.behaviour.stickers.max_tokens, now=now
            )
        except LLMError as exc:
            # A sticker is optional decoration: the text goes out anyway, so a chooser
            # error is no reason to cut the reply, only a WARNING in the log.
            logger.warning("sticker chooser llm error: reason=%s", exc.reason)
            return None

        valid_ids = {sticker.id for sticker in candidates}
        chosen_id = parse_choice(result.text, valid_ids)
        if chosen_id is None:
            return None
        for sticker in candidates:
            if sticker.id == chosen_id:
                return sticker
        return None
