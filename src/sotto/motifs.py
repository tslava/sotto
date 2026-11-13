"""Persona props and the story quota (CLAUDE.md, "less and more varied", measure 5).

The diagnosis after the chat stopped the bot for a day: it spoke too much and also kept
saying the same thing. The same few props (a spouse, an outbuilding, a decade) recurred in
most replies, and almost every reply was a tall tale. The cure works from both ends:

- **before the model**: ``used_motifs``/``story_count``/``render_avoid`` build, from what
  was already said, a paragraph for the ``{avoid}`` slot of the system prompt ("you already
  mentioned X and Y, now without them"), so the model is warned in advance;
- **after the model**: ``motifs_in``/``story_count`` are used by the output filter
  (``filters.layer_rules``: ``dedup:motif`` and ``style:story_quota``) as insurance if the
  warning did not work.

The functions are pure, do no I/O and do not depend on the config: compiled regexes arrive
ready from ``patterns.Patterns`` (``motifs``/``story_markers``), windows are numbers from
``FiltersConfig``. Sticker entries ("[sticker #N] caption", written by ``responder.py``
through ``insert_bot_reply``) are skipped when counting: they are a picture, not a persona
reply, and contain no motifs. Same approach as ``postprocess._recent_has_emoji``, including
the local prefix constant, so as not to pull ``stickers.py`` with its ``llm`` dependencies
in here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sotto.fragments import Fragments
from sotto.gate_types import Trigger


def _is_sticker_record(text: str, frag: Fragments) -> bool:
    """ "[sticker #3] Nice one" in recent_bot_replies is a picture, not persona text
    (see the module docstring); the prefix comes from the package's ``placeholders.sticker_tag``."""
    return text.startswith(frag.prefix("placeholders.sticker_tag"))


def _tail(recent_replies: Sequence[str], window: int, frag: Fragments) -> list[str]:
    """The last ``window`` replies without sticker entries.

    The window is counted over all entries and stickers are dropped inside it afterwards
    (as in ``postprocess._recent_has_emoji``): "the last 10 replies" means exactly the last
    10 rows of ``db.recent_bot_replies``, some of which may be stickers and not counted.
    """
    if window <= 0:
        return []
    return [reply for reply in recent_replies[-window:] if not _is_sticker_record(reply, frag)]


def motifs_in(text: str, motifs: Mapping[str, Sequence[re.Pattern[str]]]) -> list[str]:
    """Labels of motifs found in the text, in the order declared in the config."""
    if not text:
        return []
    return [
        label
        for label, patterns in motifs.items()
        if any(pattern.search(text) for pattern in patterns)
    ]


def used_motifs(
    recent_replies: Sequence[str],
    window: int,
    motifs: Mapping[str, Sequence[re.Pattern[str]]],
    *,
    frag: Fragments,
) -> list[str]:
    """Labels of motifs from the last ``window`` replies, without repeats, in config order.

    ``recent_replies`` is chronological, the freshest last (as
    ``db.recent_bot_replies`` returns it).
    """
    tail = _tail(recent_replies, window, frag)
    if not tail:
        return []
    used: list[str] = []
    for label, patterns in motifs.items():
        if any(pattern.search(reply) for reply in tail for pattern in patterns):
            used.append(label)
    return used


def story_count(
    recent_replies: Sequence[str],
    window: int,
    markers: Sequence[re.Pattern[str]],
    *,
    frag: Fragments,
) -> int:
    """How many of the last ``window`` replies look like a tall tale (have a story marker).

    Replies are counted, not hits: two period mentions in one phrase are one tale, not two.
    """
    tail = _tail(recent_replies, window, frag)
    return sum(1 for reply in tail if any(marker.search(reply) for marker in markers))


def render_avoid(used: Sequence[str], *, no_story: bool, frag: Fragments) -> str:
    """Paragraph for the ``{avoid}`` slot of the system prompt. Nothing to say -> "".

    ``used`` is the motif labels from ``used_motifs``; ``no_story`` means the tale quota in
    the window is already used up (``story_count >= filters.story_max``).
    """
    parts: list[str] = []
    if used:
        accusative = frag.table("render.accusative")
        items = ", ".join(accusative.get(label, label) for label in used)
        template: str = frag.text("render.avoid_template")
        parts.append(template.replace("{items}", items))
    if no_story:
        parts.append(frag.text("render.avoid_no_story"))
    return " ".join(parts)


# "About myself" metric (CLAUDE.md, "about myself every other time"): in the release where
# this was added, 7 of 7 unsolicited replies were about the persona itself. It cuts nothing,
# only feeds a line in /status so the trend is visible after prompt edits. Addressings are
# counted separately: answering a question about the persona by talking about itself is
# fine. /say (the owner's verbatim text) and /life (news about itself by definition) are
# not counted at all, nor are stickers.
_SELF_ADDRESS_TRIGGERS = frozenset(
    {
        Trigger.MENTION.value,
        Trigger.REPLY.value,
        Trigger.NAME.value,
        Trigger.FOLLOWUP.value,
        "places",
    }
)
_SELF_SKIP_TRIGGERS = frozenset({"say", "life"})


@dataclass(frozen=True, slots=True)
class SelfShare:
    """How many replies are "about myself" out of how many, by trigger group."""

    ambient: tuple[int, int] = (0, 0)
    address: tuple[int, int] = (0, 0)
    other: tuple[int, int] = (0, 0)


def speaks_about_self(text: str, markers: Sequence[re.Pattern[str]], *, frag: Fragments) -> bool:
    """Whether the reply has an "about myself" marker (first person, relatives).
    A sticker has none."""
    if not text or _is_sticker_record(text, frag):
        return False
    return any(marker.search(text) for marker in markers)


def self_share(
    replies: Iterable[tuple[str, str]], markers: Sequence[re.Pattern[str]], *, frag: Fragments
) -> SelfShare:
    """``replies`` are (trigger, text) pairs from ``bot_replies``. Groups: ambient,
    addressings (mention/reply/name/followup/places), other (morning, checkin,
    callback, spontaneous). say/life and stickers are skipped."""
    counts = {"ambient": [0, 0], "address": [0, 0], "other": [0, 0]}
    for trigger, text in replies:
        if trigger in _SELF_SKIP_TRIGGERS or _is_sticker_record(text, frag):
            continue
        if trigger == Trigger.AMBIENT.value:
            group = "ambient"
        elif trigger in _SELF_ADDRESS_TRIGGERS:
            group = "address"
        else:
            group = "other"
        counts[group][1] += 1
        if speaks_about_self(text, markers, frag=frag):
            counts[group][0] += 1
    return SelfShare(
        ambient=(counts["ambient"][0], counts["ambient"][1]),
        address=(counts["address"][0], counts["address"][1]),
        other=(counts["other"][0], counts["other"][1]),
    )


def render_self_share(share: SelfShare, days: int, *, frag: Fragments) -> str:
    """Line for /status: "about myself over 3 days: ambient 6/6, addressings 2/2, other 1/1"."""
    return frag.render(
        "render.self_share",
        days=str(days),
        ambient_hit=str(share.ambient[0]),
        ambient_total=str(share.ambient[1]),
        address_hit=str(share.address[0]),
        address_total=str(share.address[1]),
        other_hit=str(share.other[0]),
        other_total=str(share.other[1]),
    )
