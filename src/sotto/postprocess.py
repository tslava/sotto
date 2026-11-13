"""Soft post-processing of the candidate BEFORE the output filter.

An emoji at the end of almost every reply and typographic dashes (the model
copies them from the prompt itself) are cosmetics, not a character violation.
Staying silent over them is too costly, so they are fixed by hand,
deterministically, instead of being cut by the output filter (``filters.py``).
``responder.py`` calls ``soften`` once, right after ``parse_reply`` and the
``speak`` check, before ``FilterContext``/``check_output`` are built; everything
after uses the already corrected text.

Pure functions, no I/O. Emoji detection uses the same primitives as ``regex:emoji``
in ``filters.py`` (``_is_emoji_char``/``_is_emoji_modifier``, re-imported here;
no import cycle: ``filters.py`` does not import this module).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sotto.config_models import FiltersConfig
from sotto.filters import _has_emoji, _is_emoji_char, _is_emoji_modifier
from sotto.fragments import Fragments

# U+2014, U+2013, U+2012 and U+2015 dashes -> hyphen-minus "-". Spaces around are
# not in the character class, so they are left alone; a hyphen inside words
# is already "-" and does not match the class.
_DASH_CHARS = "—–‒―"
_DASH_RE = re.compile(f"[{_DASH_CHARS}]")

_MULTI_SPACE_RE = re.compile(r" {2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.,!?…:;])")


@dataclass(frozen=True)
class Fixed:
    text: str
    # reasons in filter_log style: "fix:dash", "fix:emoji_freq", "fix:emoji_count"
    fixes: tuple[str, ...]


def normalize_dashes(text: str) -> str:
    """Typographic dashes -> hyphen-minus "-". Spaces and hyphens inside words stay."""
    return _DASH_RE.sub("-", text)


def _emoji_spans(text: str) -> list[tuple[int, int]]:
    """Indexes (start, end) of every emoji in the text together with modifiers
    (variation selector, skin tone): the same character classification as
    regex:emoji in filters.py, but with its own traversal (there the helper returns
    (char, end), without start)."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if _is_emoji_char(text[i]):
            end = i + 1
            while end < n and _is_emoji_modifier(text[end]):
                end += 1
            spans.append((i, end))
            i = end
        else:
            i += 1
    return spans


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    last = 0
    for start, end in spans:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts)


def _cleanup(text: str) -> str:
    """After removing emoji: collapse repeated spaces, drop the space before
    punctuation ("word ," -> "word,"), strip()."""
    cleaned = _MULTI_SPACE_RE.sub(" ", text)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip()


def strip_emoji(text: str) -> str:
    """Remove ALL emoji (with modifiers), leaving no double or dangling spaces."""
    spans = _emoji_spans(text)
    if not spans:
        return text
    return _cleanup(_remove_spans(text, spans))


def keep_first_emoji(text: str) -> str:
    """Keep only the first emoji and remove the rest the same way as strip_emoji."""
    spans = _emoji_spans(text)
    if len(spans) <= 1:
        return text
    return _cleanup(_remove_spans(text, spans[1:]))


def _recent_has_emoji(recent_replies: Sequence[str], window: int, frag: Fragments) -> bool:
    if window <= 0:
        return False
    # Sticker entries in recent_bot_replies ("[sticker #N] caption text") contain no
    # emoji, but are skipped explicitly for the future; the prefix comes from the pack.
    sticker_prefix = frag.prefix("placeholders.sticker_tag")
    candidates = [
        reply for reply in recent_replies[-window:] if not reply.startswith(sticker_prefix)
    ]
    return any(_has_emoji(reply) for reply in candidates)


def _count_allowed_emoji(text: str, allowed_emoji: Sequence[str]) -> int:
    allowed = frozenset(allowed_emoji)
    return sum(1 for start, _end in _emoji_spans(text) if text[start] in allowed)


def soften(
    text: str, *, recent_replies: Sequence[str], cfg: FiltersConfig, frag: Fragments
) -> Fixed:
    """Order of fixes: normalize_dashes -> emoji_freq -> emoji_count.

    ``recent_replies`` is chronological, the last entries are the freshest (like
    ``db.recent_bot_replies``); sticker entries are skipped when counting emoji.
    """
    fixes: list[str] = []

    dashed = normalize_dashes(text)
    if dashed != text:
        fixes.append("fix:dash")
    text = dashed

    if _has_emoji(text) and _recent_has_emoji(recent_replies, cfg.emoji_recent_window, frag):
        text = strip_emoji(text)
        fixes.append("fix:emoji_freq")
    elif _count_allowed_emoji(text, cfg.allowed_emoji) > cfg.emoji_max_per_reply:
        text = keep_first_emoji(text)
        fixes.append("fix:emoji_count")

    return Fixed(text=text, fixes=tuple(fixes))
