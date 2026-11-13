"""Vision on photos: a chat snapshot becomes a description (CLAUDE.md,
"Interfaces: vision on photos").

Owner's decision: until now a photo in the chat was just "[photo]", the bot was blind to
it and the model's context had a hole. A vision model describes the snapshot in a sentence
or two, and that description becomes the message text in the DB (``[photo: ...]``), so
everything downstream (gate, followup, reply generation) treats it as ordinary text and
needs to know nothing more about photos.

It costs money, so not every snapshot is described: only when there is a reason (an
addressing or an open hot window), occasionally by a die roll, and always under a daily cap
with its own ``vision_calls`` counter (separate from the main ``llm_calls``, like the cheap
followup check); see ``should_describe``.

Photo bytes and their base64 are never stored anywhere and never reach the log: they live
for exactly one ``describe`` call and go into the model request body. Only the finished
description is written to ``filter_log``.

The module knows nothing about aiogram: a photo size is described by the structural
protocol ``PhotoSizeLike`` (a real ``aiogram.types.PhotoSize`` fits without inheritance),
downloading is done by the caller (``bot.py``), and the module does not touch the DB.
"""

from __future__ import annotations

import base64
import logging
import random
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from sotto.config_models import Config, VisionConfig
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError
from sotto.sanitize import normalize_text

logger = logging.getLogger(__name__)

# Own daily attempt counter instead of the shared llm_calls: describing a photo must not
# eat the main model's call budget (same approach as followup._COUNTER_KEY).
_COUNTER_KEY = "vision_calls"

_ELLIPSIS = "…"

# Duplicates the delimiter stripping of sanitize.normalize_text/prompt.py/judge.py:
# once more, just in case, we clean both the caption going into the prompt and the
# model's answer.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_SLOT_RE = re.compile(r"\{(caption)\}")


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


class PhotoSizeLike(Protocol):
    """Structural type of a photo size (``aiogram.types.PhotoSize`` fits without
    inheritance, as do ``SimpleNamespace``/dataclasses in tests).

    Attributes are declared as read-only properties rather than mutable fields: the module
    only needs to read them, and a read-only protocol accepts both a plain attribute of a
    pydantic model and a real property.
    """

    @property
    def file_id(self) -> str: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...


def pick_photo_size(sizes: Sequence[PhotoSizeLike], max_width: int) -> PhotoSizeLike | None:
    """The largest size not wider than ``max_width``; if all are wider, the smallest.

    Telegram returns ``message.photo`` in ascending size order, but there is no need to rely
    on the order: the choice goes by ``width``. The point of the cap: above roughly 1024 px
    a vision model takes more tokens without the description getting any better. If even the
    smallest preview is wider than the cap, we take it: describing the snapshot anyway beats
    not describing it.
    """
    if not sizes:
        return None
    fitting = [size for size in sizes if size.width <= max_width]
    if fitting:
        return max(fitting, key=lambda size: size.width)
    return min(sizes, key=lambda size: size.width)


def should_describe(
    *,
    addressed: bool,
    hot: bool,
    count_today: int,
    cfg: VisionConfig,
    rng: random.Random,
) -> bool:
    """Whether to spend a model call on this photo.

    Order (CLAUDE.md): enabled -> daily cap -> there is a reason (an addressing or an open
    hot window) -> die roll. The die roll is last so that ``rng`` is consumed only when
    everything else already allows describing: tests with a fixed seed are more stable that
    way (exactly like the gate's last step and ``pick_reaction``).
    """
    if not cfg.enabled:
        return False
    if count_today >= cfg.daily_cap:
        return False
    if addressed or hot:
        return True
    return rng.random() < cfg.ambient_probability


def _truncate(text: str, max_chars: int) -> str:
    """Trim at a word boundary with "…"; the result length does not exceed ``max_chars``."""
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 1]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip() + _ELLIPSIS


def photo_text(description: str | None, caption: str, *, frag: Fragments) -> str:
    """The text that goes into ``messages.text`` in place of the photo.

    Without a description it is the familiar "[photo]" (like ``sanitize.media_placeholder``),
    with one it is "[photo: ...]". The author's caption, if any, is appended after it and
    stays in their own words.
    """
    head = (
        frag.render("placeholders.photo_with_desc", desc=description)
        if description
        else frag.text("placeholders.photo")
    )
    caption = caption.strip()
    return f"{head} {caption}" if caption else head


class VisionDescriber:
    """One vision-model call on a snapshot -> a description or None.

    The prompt (``prompts/vision.txt``) with the ``{caption}`` slot is filled in a single
    ``re.sub`` pass, the same approach as in ``judge.py``/``followup.py``: a chain of
    ``str.replace`` could touch data that was already substituted.

    Any failure gives ``None``: an empty model, ``LLMError`` (including an exhausted daily
    cap), an empty answer. The caller then writes the plain "[photo]": a photo never breaks
    message handling.
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

    async def describe(self, image: bytes, *, mime: str, caption: str, now: int) -> str | None:
        cfg = self._cfg_getter()
        vision_cfg = cfg.behaviour.vision
        model = vision_cfg.model or cfg.llm.main_model
        if not model:
            logger.info(
                "vision: model not set (behaviour.vision.model and llm.main_model are empty)"
            )
            return None

        clean_caption = _strip_fake_delimiters(normalize_text(caption)).strip()
        prompt = _SLOT_RE.sub(
            lambda _m: clean_caption or self._frag.text("placeholders.no_caption"),
            self._prompt_template,
        )

        # base64 lives only inside the request body: it reaches neither the log nor the DB.
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        try:
            result = await self._llm.call_raw(
                messages,
                model=model,
                max_tokens=vision_cfg.max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=vision_cfg.daily_cap,
            )
        except LLMError as exc:
            if exc.reason == "llm:calls_cap":
                # An exhausted cap is a normal state toward the end of an active day,
                # not a failure: INFO, so as not to wake the owner with WARNINGs.
                logger.info("vision: daily description cap exhausted")
            else:
                logger.warning("vision llm error: reason=%s", exc.reason)
            return None

        description = _strip_fake_delimiters(normalize_text(result.text)).strip()
        if not description:
            return None
        return _truncate(description, vision_cfg.max_chars)
