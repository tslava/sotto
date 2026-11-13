"""Offline fill of the sticker catalog (CLAUDE.md, "Interfaces: stickers"; PLAN.md,
section "Stickers").

``python -m sotto.stickers_fill <set_name> [--out stickers.yaml] [--dry-run]
    [--no-llm] [--model provider/model]``

Not part of the bot runtime: a separate manual run (after the owner has assembled/adjusted
the sticker set in Telegram). It calls the Bot API: ``get_sticker_set`` for the set's
sticker list, then ``download`` for the content of each new one (webp). Animated/video
stickers (``is_animated``/``is_video``) cannot be recognized (tgs/webm): they are skipped
with a WARNING and never reach the catalog.

Telegram does not expose the text on a sticker as text, so it is recognized once by a
vision model through ``LLMClient.call_raw`` (a content array with ``image_url``); the owner
edits the result (``text``/``when``) by hand, this script only produces a draft.

An existing ``stickers.yaml`` (if any) is read and MERGED by ``file_id``: on a match the old
entry's id/text/when/enabled are kept (owner edits are not lost), new stickers get the next
ids, and stickers that vanished from the set stay in the file but are marked
``enabled: false`` with a WARNING (they are not deleted: ``bot_replies``/recently used may
refer to them).

Dependencies needed by tests are injected explicitly (modelled on ``places_fill.py``):
``BotLike`` (tests supply a fake with ``get_sticker_set``/``download``, without a real
aiogram) and ``LLMLike`` (a subset of ``LLMClient.call_raw`` that does not count the live
bot's budget/accounting, like ``_InMemoryState`` in ``places_fill.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import time as time_module
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import yaml

from sotto.config import load_config
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError, LLMResult
from sotto.persona import Persona, persona_from_env
from sotto.settings import Settings
from sotto.stickers import Sticker, StickerCatalog, load_catalog

logger = logging.getLogger(__name__)

_RECOGNITION_MAX_TOKENS = 120

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


class _ReadableLike(Protocol):
    def read(self) -> bytes: ...


class _StickerLike(Protocol):
    """The subset of ``aiogram.types.Sticker`` this module needs."""

    @property
    def file_id(self) -> str: ...
    @property
    def emoji(self) -> str | None: ...
    @property
    def is_animated(self) -> bool: ...
    @property
    def is_video(self) -> bool: ...


class _StickerSetLike(Protocol):
    """The subset of ``aiogram.types.StickerSet`` this module needs.

    ``Sequence`` (not ``list``), like ``commands._DbLike.last_bot_replies``: ``list`` is
    invariant in its parameter, so a real ``aiogram.types.StickerSet``
    (``stickers: list[Sticker]``) would not fit ``list[_StickerLike]`` structurally, but fits
    the covariant ``Sequence`` since ``Sticker`` structurally matches ``_StickerLike``.
    """

    @property
    def stickers(self) -> Sequence[_StickerLike]: ...


class BotLike(Protocol):
    """A narrow protocol instead of ``aiogram.Bot``: tests fake it without aiogram
    (modelled on ``responder._BotLike``/``reactions.ReactionBotLike``)."""

    async def get_sticker_set(self, name: str) -> _StickerSetLike: ...

    async def download(self, file: str) -> _ReadableLike | None: ...


class LLMLike(Protocol):
    """The subset of ``LLMClient`` this module needs: tests supply a fake that does not count
    the live bot's budget/accounting (see the module docstring)."""

    async def call_raw(
        self, messages: list[dict[str, object]], *, model: str, max_tokens: int, now: int
    ) -> LLMResult: ...


class _InMemoryState:
    """Minimal state store for the real LLMClient inside stickers_fill.

    Filling the catalog is an offline one-off action: the live bot's daily limits, budget and
    circuit breaker (the shared state table) are not needed here, and the LLMClient counters
    live only for the duration of one run (modelled on
    ``places_fill._InMemoryState``)."""

    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    async def get_state(self, key: str) -> str | None:
        return self._state.get(key)

    async def set_state(self, key: str, value: str) -> None:
        self._state[key] = value

    async def increment_state(self, key: str, by: int = 1) -> int:
        new_value = int(self._state.get(key, "0")) + by
        self._state[key] = str(new_value)
        return new_value

    async def add_state_float(self, key: str, by: float) -> float:
        new_value = float(self._state.get(key, "0")) + by
        self._state[key] = str(new_value)
        return new_value


def _parse_recognition(raw: str) -> tuple[str, str]:
    """Parses the recognition answer as tolerantly as ``judge._parse_verdict``:
    strip the ```json``` fence, find the first "{" and ``raw_decode`` from it. Any failure ->
    ("", ""): the draft stays empty and the owner fills it in."""
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return "", ""
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return "", ""
        text_value = data.get("text", "")
        when_value = data.get("when", "")
        text = text_value.strip() if isinstance(text_value, str) else ""
        when = when_value.strip() if isinstance(when_value, str) else ""
        return text, when
    except Exception:
        return "", ""


async def recognize_sticker(
    image_bytes: bytes, *, llm: LLMLike | None, model: str, now: int, frag: Fragments
) -> tuple[str, str]:
    """(text, when) draft from a sticker image (webp). ``llm=None`` -> ("", "")
    with no network call: this is ``--no-llm``."""
    if llm is None:
        return "", ""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/webp;base64,{encoded}"
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": frag.text("tools.sticker_ocr_prompt")},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    try:
        result = await llm.call_raw(
            messages, model=model, max_tokens=_RECOGNITION_MAX_TOKENS, now=now
        )
    except LLMError as exc:
        logger.warning("stickers_fill: llm error while recognizing sticker: reason=%s", exc.reason)
        return "", ""
    return _parse_recognition(result.text)


async def _download_bytes(bot: BotLike, file_id: str) -> bytes:
    downloaded = await bot.download(file_id)
    if downloaded is None:
        raise RuntimeError(f"stickers_fill: bot.download({file_id!r}) returned nothing")
    return downloaded.read()


async def build_catalog(
    *,
    bot: BotLike,
    set_name: str,
    existing: StickerCatalog,
    llm: LLMLike | None,
    model: str,
    now: int,
    frag: Fragments,
) -> StickerCatalog:
    """Pulls the set from the Bot API and merges with ``existing`` by ``file_id`` (see the
    module docstring: owner edits are kept, new ones get the next ids, vanished ones
    stay disabled)."""
    sticker_set = await bot.get_sticker_set(set_name)

    existing_by_file_id = {sticker.file_id: sticker for sticker in existing.stickers}
    seen_file_ids: set[str] = set()
    next_id = max((sticker.id for sticker in existing.stickers), default=0) + 1

    result: list[Sticker] = []
    for raw in sticker_set.stickers:
        if raw.is_animated or raw.is_video:
            logger.warning(
                "stickers_fill: skipping animated/video sticker file_id=%s "
                "(tgs/webm cannot be recognised)",
                raw.file_id,
            )
            continue

        seen_file_ids.add(raw.file_id)
        old = existing_by_file_id.get(raw.file_id)
        if old is not None:
            # A file_id match: the owner's id/text/when/enabled are left alone, the
            # emoji is refreshed (the owner does not edit it).
            result.append(
                Sticker(
                    id=old.id,
                    file_id=raw.file_id,
                    emoji=raw.emoji or old.emoji,
                    text=old.text,
                    when=old.when,
                    enabled=old.enabled,
                )
            )
            continue

        image_bytes = await _download_bytes(bot, raw.file_id)
        text, when = await recognize_sticker(image_bytes, llm=llm, model=model, now=now, frag=frag)
        sticker_id = next_id
        next_id += 1
        result.append(
            Sticker(
                id=sticker_id,
                file_id=raw.file_id,
                emoji=raw.emoji or "",
                text=text,
                when=when,
                enabled=True,
            )
        )

    for old in existing.stickers:
        if old.file_id in seen_file_ids:
            continue
        logger.warning(
            "stickers_fill: sticker id=%s (file_id=%s) is gone from the set %r, disabling it",
            old.id,
            old.file_id,
            set_name,
        )
        result.append(
            Sticker(
                id=old.id,
                file_id=old.file_id,
                emoji=old.emoji,
                text=old.text,
                when=old.when,
                enabled=False,
            )
        )

    result.sort(key=lambda sticker: sticker.id)
    return StickerCatalog(set_name=set_name, stickers=result)


def render_report(catalog: StickerCatalog) -> str:
    lines = ["id | emoji | text | when | enabled"]
    for sticker in catalog.stickers:
        lines.append(
            f"{sticker.id} | {sticker.emoji or '-'} | {sticker.text or '-'} | "
            f"{sticker.when or '-'} | {'yes' if sticker.enabled else 'no'}"
        )
    return "\n".join(lines)


def render_yaml(catalog: StickerCatalog) -> str:
    payload = {
        "set_name": catalog.set_name,
        "stickers": [sticker.model_dump(mode="json") for sticker in catalog.stickers],
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sotto.stickers_fill",
        description='Offline fill of the sticker catalog (PLAN.md, section "Stickers").',
    )
    parser.add_argument("set_name", help="set name from t.me/addstickers/<name>")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="catalog path (read and merged); default: the pack's stickers.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only print the table, do not write the file"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="text/when are always empty, the model is not called"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="vision model; default: cfg.llm.main_model"
    )
    return parser.parse_args(argv)


async def run(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    bot: BotLike | None = None,
    llm: LLMLike | None = None,
    persona: Persona | None = None,
) -> StickerCatalog:
    """Entry point shared by the CLI and tests (see the module docstring on DI)."""
    args = parse_args(argv)
    settings = settings if settings is not None else Settings()
    now = int(time_module.time())

    persona_cache: list[Persona] = [persona] if persona is not None else []

    def get_persona() -> Persona:
        if not persona_cache:
            persona_cache.append(persona_from_env())
        return persona_cache[0]

    if args.out is None:
        args.out = get_persona().stickers_path
    existing = load_catalog(args.out)

    owns_bot = bot is None
    if bot is not None:
        bot_obj: BotLike = bot
    else:
        from aiogram import Bot  # lazy import: tests do not need a real aiogram Bot

        bot_obj = Bot(token=settings.bot_token.get_secret_value())

    model = args.model or ""
    use_llm: LLMLike | None
    owns_llm = False
    if args.no_llm:
        use_llm = None
    elif llm is not None:
        use_llm = llm
        model = model or load_config(get_persona().config_layers()).llm.main_model
    else:
        cfg = load_config(get_persona().config_layers())
        model = model or cfg.llm.main_model
        if settings.openrouter_api_key is not None and model:
            use_llm = LLMClient(
                api_key=settings.openrouter_api_key.get_secret_value(),
                cfg_getter=lambda: cfg,
                db=_InMemoryState(),
            )
            owns_llm = True
        else:
            use_llm = None
            logger.warning(
                "stickers_fill: LLM unavailable (no OPENROUTER_API_KEY or no vision model "
                "set); text/when will be empty for new stickers"
            )

    try:
        catalog = await build_catalog(
            bot=bot_obj,
            set_name=args.set_name,
            existing=existing,
            llm=use_llm,
            model=model,
            now=now,
            frag=get_persona().frag,
        )
    finally:
        if owns_bot:
            # owns_bot=True -> bot_obj is a real aiogram.Bot (created above), not a
            # test fake; BotLike does not declare .session, hence the type: ignore.
            await bot_obj.session.close()  # type: ignore[attr-defined]
        if owns_llm:
            assert isinstance(use_llm, LLMClient)
            await use_llm.aclose()

    for line in render_report(catalog).splitlines():
        logger.info(line)
    logger.info("Check text/when by hand before rollout in %s.", args.out)

    if not args.dry_run:
        args.out.write_text(render_yaml(catalog), encoding="utf-8")

    return catalog


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()
