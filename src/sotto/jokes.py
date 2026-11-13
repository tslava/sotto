"""Chat jokes (CLAUDE.md, "chat jokes and people's threads").

The weekly ``ChatMemorizer`` pulls from the conversation the words the chat picked up and
puts them in the ``chat_jokes`` table. ``render_jokes`` builds from them a paragraph for the
``{jokes}`` slot of the system prompt ("you may bring one back where it fits"), and
``joke_used`` decides whether the persona brought a joke back in the reply just sent; then
the responder sets its ``last_used_at`` and the joke rests for ``cooldown_days`` days.

Pure functions, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sotto.db import JokeRow
from sotto.fragments import Fragments
from sotto.lang import Language

_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")
# Leading bullet/numbering: a slot line must not start with "- " (regex:prompt_leak).
_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]+\s+|\d+[.)]\s+)")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# How many shared words in a row count as "the joke was brought back" (and the threshold
# of a "short" joke).
_MIN_RUN = 3


def _clean(text: str) -> str:
    cleaned = _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))
    cleaned = _BULLET_RE.sub("", " ".join(cleaned.split()))
    return cleaned.strip().rstrip(".;")


def _words(text: str, lang: Language) -> list[str]:
    return _WORD_RE.findall(lang.fold(text))


def render_jokes(rows: Sequence[JokeRow], *, now: int, cooldown_days: int, frag: Fragments) -> str:
    """Block for the ``{jokes}`` slot. Empty -> "" (the slot's paragraph in system.txt disappears).

    A joke is shown if it has not been used yet or at least ``cooldown_days`` days have
    passed since its use. One line, no "- " at the start."""
    cooldown = cooldown_days * 86400
    quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
    parts: list[str] = []
    for row in rows:
        if row.last_used_at is not None and now - row.last_used_at < cooldown:
            continue
        cleaned = _clean(row.text)
        if cleaned:
            parts.append(f"{quote_open}{cleaned}{quote_close}")
    if not parts:
        return ""
    return frag.text("render.jokes_header") + "; ".join(parts) + "."


def joke_used(text: str, joke: str, lang: Language) -> bool:
    """The joke sounded in ``text``: 3+ shared normalized words in a row, or the joke is
    shorter than three words and is contained whole (by words, in a row) in the text."""
    text_words = _words(text, lang)
    joke_words = _words(joke, lang)
    if not joke_words or not text_words:
        return False
    if len(joke_words) < _MIN_RUN:
        size = len(joke_words)
        return any(
            text_words[i : i + size] == joke_words for i in range(len(text_words) - size + 1)
        )
    text_grams = {
        tuple(text_words[i : i + _MIN_RUN]) for i in range(len(text_words) - _MIN_RUN + 1)
    }
    return any(
        tuple(joke_words[i : i + _MIN_RUN]) in text_grams
        for i in range(len(joke_words) - _MIN_RUN + 1)
    )
