"""Pure synchronous functions for cleaning user-supplied data.

They know nothing about aiogram and do no I/O. Used before writing to the DB
(display_name, message text) and before inserting data into the prompt, so that a
chat participant cannot forge the context format or inject instructions through
their name or message text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from sotto.fragments import Fragments

# Maximum length of a display name in the context (characters).
_MAX_DISPLAY_NAME_LEN = 24

# Modulus of the stable number for "Participant N", see stable_n().
_STABLE_N_MODULUS = 997

_MULTI_SPACE_RE = re.compile(r" {2,}")
_WHITESPACE_RUN_RE = re.compile(r"\s+")
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")


def stable_n(user_id: int) -> int:
    """Deterministic number 1..999 for the "Participant {n}" fallback name, stable per user_id.

    Implementation: user_id % 997 + 1. 997 is a prime close to 999, so the result
    covers the range 1..997 almost uniformly. The only property that matters is
    stability (one user_id always yields the same number), not uniformity or the
    absence of collisions between different user_ids.
    """
    return user_id % _STABLE_N_MODULUS + 1


def sanitize_display_name(
    raw: str | None, user_id: int, reserved: set[str], *, frag: Fragments
) -> str:
    """Sanitize display_name before storing it in the DB and inserting it into the prompt.

    Keeps only letters of any alphabet (str.isalpha), whitespace characters
    (each becomes a plain space) and the hyphen; emoji, digits, quotes and other
    marks are stripped. Repeated spaces are collapsed and the result is cut to
    24 characters (at a word boundary if one is within the cut). If the name is
    then empty, equals one of `reserved` case-insensitively, or contains any
    `reserved` element as a separate word (case-insensitively), the localized
    "Participant N" fallback (N = stable_n(user_id)) is returned, so nobody can
    pass themselves off as the bot or its trigger and swap who is speaking in the
    context.

    The case of the name itself is preserved.
    """
    text = raw or ""
    filtered_chars = (
        " " if ch.isspace() else ch for ch in text if ch.isalpha() or ch.isspace() or ch == "-"
    )
    filtered = _MULTI_SPACE_RE.sub(" ", "".join(filtered_chars)).strip()

    if len(filtered) > _MAX_DISPLAY_NAME_LEN:
        truncated = filtered[:_MAX_DISPLAY_NAME_LEN]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        filtered = truncated.strip()

    fallback = frag.render("placeholders.participant", n=str(stable_n(user_id)))

    if not filtered:
        return fallback

    reserved_words = {item.strip().casefold() for item in reserved if item.strip()}
    if filtered.casefold() in reserved_words:
        return fallback
    for word in reserved_words:
        if re.search(rf"\b{re.escape(word)}\b", filtered, re.IGNORECASE):
            return fallback

    return filtered


def normalize_text(raw: str | None) -> str:
    """Normalize arbitrary user text before storing or inserting it.

    None -> "". Line breaks, tabs and other control characters become a space
    (so nobody can forge the multi-line "Name: text" format in the prompt context).
    Null and invisible characters (zero-width, BOM, etc.) are removed entirely.
    Runs of three or more `<` or `>` are replaced with a space (so a participant
    cannot forge the `<<<CHAT ... >>>` delimiters that wrap data in the prompt:
    data is kept separate from instructions). Ordinary punctuation and emoji are
    left alone. Finally, whitespace is collapsed and stripped.
    """
    if raw is None:
        return ""

    # Null and invisible characters (category Cf: zero-width space/joiner, BOM...)
    # are removed entirely rather than replaced with a space.
    without_invisible = "".join(
        ch for ch in raw if ch != "\x00" and unicodedata.category(ch) != "Cf"
    )

    # Remaining control characters (line breaks, tabs, etc.) -> space.
    without_control = "".join(
        " " if unicodedata.category(ch) == "Cc" else ch for ch in without_invisible
    )

    without_fake_delimiters = _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", without_control))

    return _WHITESPACE_RUN_RE.sub(" ", without_fake_delimiters).strip()


class MediaMessage(Protocol):
    """Structural type of a message with aiogram media attributes (without importing aiogram).

    Any object with such attributes (e.g. an aiogram Message or
    types.SimpleNamespace in tests) fits without explicit inheritance.
    """

    @property
    def sticker(self) -> object | None: ...
    @property
    def photo(self) -> object | None: ...
    @property
    def voice(self) -> object | None: ...
    @property
    def audio(self) -> object | None: ...
    @property
    def video(self) -> object | None: ...
    @property
    def video_note(self) -> object | None: ...
    @property
    def animation(self) -> object | None: ...
    @property
    def document(self) -> object | None: ...


def media_placeholder(message: MediaMessage, *, frag: Fragments) -> str | None:
    """Placeholder for a media message without text, else None.

    Priority order (the first match wins if several fields of the message are
    set): sticker, photo, voice/audio, video/video_note/animation, document.
    The placeholder texts come from the language fragments.
    """
    if message.sticker is not None:
        return frag.text("placeholders.sticker")
    if message.photo is not None:
        return frag.text("placeholders.photo")
    if message.voice is not None or message.audio is not None:
        return frag.text("placeholders.voice")
    if message.video is not None or message.video_note is not None or message.animation is not None:
        return frag.text("placeholders.video")
    if message.document is not None:
        return frag.text("placeholders.file")
    return None
