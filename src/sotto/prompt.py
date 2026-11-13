"""Build messages for the LLM and parse its reply.

Pure functions, no I/O, no aiogram. The key rule is data kept apart from
instructions:

- The system prompt (``template``) goes out with role ``system``. The ``{age}`` and
  ``{few_shot}`` slots in it are filled with real values, while ``{context}``,
  ``{recent_replies}``, ``{places}`` and ``{situation}`` get short "see below" markers,
  because the data itself goes in a second message with role ``user``, inside
  ``<<<CHAT ... >>>`` delimiters, with an explicit note that it is data, not commands.
- Slots are filled in a single ``re.sub`` pass over the known slot names, never with
  ``str.format`` and never with a chain of ``str.replace``: a curly brace in a
  participant's message ("{context}", "{'a': 1}") would either crash the call
  (``str.format``) or, landing inside an already substituted slot (e.g. "{context}"
  inside few_shot), be hit by the next replacement in a ``.replace()`` chain. A single
  ``re.sub`` pass scans only the original template and does neither.
- The ``<<<``/``>>>`` delimiters are stripped from the data once more here, duplicating
  ``sanitize.normalize_text``: normalize_text already cleans individual messages, but
  the assembled blocks (context, places, situation) go through the same cleaning
  again as a safeguard.

``parse_reply`` parses the model's reply just as strictly and tolerates wrappers
(```json ... ```` and preambles like "Here is the answer: {...}"), but any failure is
``None``, that is silence, not an exception.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sotto.db import LifeEventRow, MessageRow
from sotto.fragments import Fragments
from sotto.timeutil import local_date

CHAT_OPEN = "<<<CHAT"
CHAT_CLOSE = ">>>"

# Situation texts, the message skeleton and block headings live in the persona pack
# (``fragments.yaml``, groups situations/prompt/render/placeholders) and arrive as the
# ``frag`` argument; only assembly happens here.

_ADDRESSED_TEXT_MAX_LEN = 300
_ADDRESSED_ITEMS_MAX = 5

_LIFE_TEXT_MAX_LEN = 300
_SITUATION_MARKER = ""

# Duplicates the delimiter stripping of sanitize.normalize_text (see the module docstring):
# assembled blocks (context/recent_replies/places/situation) are cleaned once more here
# before being wrapped in <<<CHAT ... >>>.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_SLOT_RE = re.compile(
    r"\{(age|few_shot|life|diary|vocab|chat_memory|jokes|weather|context|recent_replies|places|avoid"
    r"|situation)\}"
)


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _clean_addressed_text(text: str) -> str:
    return _strip_fake_delimiters(text).strip()[:_ADDRESSED_TEXT_MAX_LEN]


def situation_addressed(items: list[tuple[str, str]], *, frag: Fragments) -> str:
    """Situation "someone addressed you" built from accumulated addresses (display_name, text).

    ``items`` are all addresses accumulated by the time the delayed reply fires
    (collapsing the debounce buffer can gather several addresses from different people
    during the delay), in chronological order. Only the last ``_ADDRESSED_ITEMS_MAX``
    are used: the caller (responder.py) should already have trimmed the list, but we
    trim here too as a safeguard.

    One address gives the short form (``situations.addressed_single``, filled with
    .replace, not re.sub/format: there are only two slots and they cannot trigger a
    repeated substitution of each other). Several give a bulleted list with a shared
    instruction (``situations.addressed_multi_*``).

    The name and text of each address are untrusted participant input: the
    ``<<<``/``>>>`` delimiters are stripped the same way as in build_messages, and the
    text is cut to 300 characters so a long message does not bloat the prompt.
    Empty ``items`` -> empty string (meaning no situation is to be added).
    """
    trimmed = items[-_ADDRESSED_ITEMS_MAX:]
    cleaned = [
        (_strip_fake_delimiters(name).strip(), _clean_addressed_text(text))
        for name, text in trimmed
    ]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        name, text = cleaned[0]
        single: str = frag.text("situations.addressed_single")
        return single.replace("{name}", name).replace("{text}", text)

    quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
    lines = [frag.text("situations.addressed_multi_header")]
    lines.extend(f"- {name}: {quote_open}{text}{quote_close}" for name, text in cleaned)
    lines.append(frag.text("situations.addressed_multi_footer"))
    return "\n".join(lines)


def situation_followup(items: list[tuple[str, str]], *, frag: Fragments) -> str:
    """Situation for the ``followup`` trigger: the same shape as
    ``situation_addressed`` (several reasons can accumulate while one pending item is
    debounce-collapsed), but with an explicit note that the addressing is probable, not
    certain: the ordinary gate did not treat the message as an address, this is the
    verdict of the cheap check.

    The ``_ADDRESSED_ITEMS_MAX`` cap, the ``_ADDRESSED_TEXT_MAX_LEN`` truncation and the
    stripping of fake delimiters are the same as in ``situation_addressed``.
    """
    trimmed = items[-_ADDRESSED_ITEMS_MAX:]
    cleaned = [
        (_strip_fake_delimiters(name).strip(), _clean_addressed_text(text))
        for name, text in trimmed
    ]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        name, text = cleaned[0]
        single: str = frag.text("situations.followup_single")
        return single.replace("{name}", name).replace("{text}", text)

    quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
    lines = [frag.text("situations.followup_multi_header")]
    lines.extend(f"- {name}: {quote_open}{text}{quote_close}" for name, text in cleaned)
    lines.append(frag.text("situations.followup_multi_footer"))
    return "\n".join(lines)


def render_numbered_messages(rows: Sequence[MessageRow]) -> str:
    """ "1. Name: text" per line, numbered from 1; name and text are cleaned of fake
    delimiters and truncated to the same limit as ``situation_addressed``
    (``_ADDRESSED_TEXT_MAX_LEN`` characters).

    Shared helper for ``situation_checkin`` and the cheap "came back to check"
    prefilter (``followup.FollowupChecker.check_batch``): both show the model the same
    numbered batch of messages, so the formatting is not duplicated. Empty ``rows`` ->
    empty string.
    """
    lines = []
    for index, row in enumerate(rows, start=1):
        name = _strip_fake_delimiters(row.display_name or "").strip()
        text = _clean_addressed_text(row.text or "")
        lines.append(f"{index}. {name}: {text}")
    return "\n".join(lines)


def situation_checkin(rows: Sequence[MessageRow], *, frag: Fragments) -> str:
    """Situation for the ``checkin`` trigger: a numbered list of messages accumulated
    after the character's last reply, plus an instruction to answer the chosen one in a
    single phrase (the number goes into the ``reply_to`` field of the JSON reply) or to
    stay silent. Empty ``rows`` -> empty string: the caller (responder.py) must not
    start generation without messages at all, but as a safeguard this also means
    "answer nothing".
    """
    if not rows:
        return ""
    lines = [
        frag.text("situations.checkin_header"),
        render_numbered_messages(rows),
        frag.text("situations.checkin_footer"),
    ]
    return "\n".join(lines)


def render_context(rows: list[MessageRow]) -> str:
    """ "Name: text" per line, in chronological order (rows are already sorted)."""
    return "\n".join(f"{row.display_name}: {row.text}" for row in rows)


def render_life(rows: Sequence[LifeEventRow], tz: str, *, frag: Fragments) -> str:
    """Memory block of the character's life events.

    Empty -> "" (then the whole paragraph with the {life} slot in system.txt disappears).
    Otherwise a heading and one line per event, local date (timeutil.local_date, tz),
    format dd.mm.yyyy: "12.09.2026: sold the cart, bought a Hyundai".

    Lines do NOT start with "- " and do not contain the word JSON:
    filters.regex:prompt_leak treats the bullet lines "- ..." and the JSON block as the
    instructive part of the system prompt, and the character retelling an event is not
    a leak.

    The event goes straight into system (like few_shot, unlike context/places): it is
    the owner's memory of the character, not untrusted participant input, but
    _strip_fake_delimiters is still applied as a safeguard so fake ``<<<``/``>>>``
    inside a note do not confuse the user message markup.
    """
    if not rows:
        return ""
    lines = [frag.text("render.life_header")]
    for row in rows:
        day = local_date(row.created_at, tz).strftime("%d.%m.%Y")
        lines.append(f"{day}: {_strip_fake_delimiters(row.text)}")
    return "\n".join(lines)


def situation_life(text: str, *, frag: Fragments) -> str:
    """Situation for announce_life: substitution into ``situations.life``.

    One {text} slot, via .replace, not re.sub/format (as in situation_addressed): a
    single substitution cannot trigger a repeated replacement of itself. The text is
    cut to 300 characters and cleaned of fake delimiters, the same technique as for
    addresses (_clean_addressed_text) but with its own length constant, because the
    source differs (an owner's note, not a participant message).
    """
    cleaned = _strip_fake_delimiters(text).strip()[:_LIFE_TEXT_MAX_LEN]
    life: str = frag.text("situations.life")
    return life.replace("{text}", cleaned)


def situation_callback(name: str, text: str, *, frag: Fragments) -> str:
    """Situation for the ``callback`` trigger: substitution into ``situations.callback``.

    Name and text are cleaned of fake delimiters and cut to 300 characters
    (``_clean_addressed_text``); substitution is a single ``re.sub`` pass so
    ``{name}``/``{text}`` inside foreign data are not replaced again.
    """
    values = {"name": _clean_addressed_text(name), "text": _clean_addressed_text(text)}
    template: str = frag.text("situations.callback")
    return re.sub(r"\{(name|text)\}", lambda m: values[m.group(1)], template)


def build_messages(
    template: str,
    *,
    age: int,
    few_shot: str,
    context: str,
    recent_replies: str,
    places: str,
    situation: str,
    avoid: str = "",
    life: str = "",
    diary: str = "",
    chat_memory: str = "",
    jokes: str = "",
    weather: str = "",
    vocab: str = "",
    json_reminder: str | None = None,
    frag: Fragments,
) -> list[dict[str, str]]:
    """Build [system, user] for LLMClient.call().

    Slots are filled in a single re.sub pass (not str.format and not a str.replace
    chain): a chain of successive .replace() calls rescans the already substituted text
    at every step, so a "{context}" that happens to sit inside few_shot (or any other
    already substituted slot) would also be replaced at the next step. A single re.sub
    pass scans only the original template, so what was substituted is not touched again.
    People's data (context/recent_replies/places/situation) never goes into system, only
    "see below" markers; the data goes in the second message with role=user, with
    context and recent_replies inside <<<CHAT ... >>> delimiters. {life} is an
    exception: it is the owner's memory of the character (like few_shot), not input
    from chat participants, so it is substituted into system directly with its real value.
    {diary} is also in system: facts about the character's own doings extracted from
    its own replies (diary.render_diary), not participant input.
    {chat_memory} is the same kind of exception: a summary of past weeks already
    condensed by the model (chat_memory.py), the character's memory, not raw participant
    messages.
    {jokes} is also in system: shared chat jokes extracted by the weekly ChatMemorizer
    (jokes.render_jokes), the character's memory.
    {avoid} is also in system: not participant data but an instruction assembled by
    motifs.render_avoid from what the character has already said ("you already mentioned
    the wife and the garage, now without them").
    {weather} is also in system: a fact from an open API (weather.render_weather), the
    background of the character's life, not a participant message; the place name inside
    the block comes from the geocoder, not from the chat.
    {vocab} is also in system: dictionaries are uploaded by the owner
    (vocab.render_vocab), it is his data, like {life}, not participant input.
    """
    slot_values = {
        "age": str(age),
        "few_shot": few_shot,
        "life": life,
        "diary": _strip_fake_delimiters(diary).strip(),
        "chat_memory": chat_memory,
        "jokes": _strip_fake_delimiters(jokes).strip(),
        "weather": _strip_fake_delimiters(weather).strip(),
        "vocab": _strip_fake_delimiters(vocab).strip(),
        "context": frag.text("prompt.context_marker"),
        "recent_replies": frag.text("prompt.recent_replies_marker"),
        "places": frag.text("prompt.places_marker"),
        "avoid": _strip_fake_delimiters(avoid).strip(),
        "situation": _SITUATION_MARKER,
    }
    system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], template)

    clean_context = _strip_fake_delimiters(context)
    clean_recent = _strip_fake_delimiters(recent_replies).strip() or frag.text(
        "placeholders.recent_replies_empty"
    )
    clean_places = _strip_fake_delimiters(places).strip() or frag.text("prompt.places_none")
    clean_situation = _strip_fake_delimiters(situation).strip()

    lines = [
        frag.text("prompt.data_notice"),
        f"{CHAT_OPEN}\n{clean_context}\n{CHAT_CLOSE}",
        "",
        frag.text("prompt.recent_replies_label"),
        f"{CHAT_OPEN}\n{clean_recent}\n{CHAT_CLOSE}",
        "",
        clean_places,
    ]
    if clean_situation:
        lines.append("")
        lines.append(clean_situation)
    lines.append("")
    lines.append(json_reminder if json_reminder is not None else frag.text("prompt.json_reminder"))

    user = "\n".join(lines)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True, slots=True)
class Reply:
    speak: bool
    text: str
    # Number of the message (1-based, index into checkin_rows) the model replies to;
    # only for the "checkin" trigger; for other triggers the model does not fill this
    # field and it stays None.
    reply_to: int | None = None


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def parse_reply(raw: str) -> Reply | None:
    """Parse the model reply into a Reply, or None on any failure (meaning silence).

    Strips ```json ... ``` / ``` ... ``` wrappers; finds the first "{" and parses JSON
    from it with ``json.JSONDecoder().raw_decode`` instead of ``json.loads``:
    raw_decode stops right after the first valid object and does not require that
    nothing follows it. This tolerates the tail models like to append after the JSON
    (for example "\\n\\nNote: not sure" or a second JSON object); json.loads would fail
    on such a tail, raw_decode simply ignores it.
    speak must be a bool and text a string; with speak=true an empty text is a failure
    (None), with speak=false a missing/empty text is normal ("").
    Extra keys in the JSON are ignored.
    """
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        speak = data.get("speak")
        if not isinstance(speak, bool):
            return None

        text_value = data.get("text", "")
        if text_value is None:
            text_value = ""
        if not isinstance(text_value, str):
            return None
        text = text_value.strip()

        # "reply_to": int, null or missing -> used as is/None; any other type (string,
        # float, bool: bool is excluded explicitly because isinstance(True, int) is true
        # in Python) -> None; this fails only this optional field, not the whole parse.
        reply_to_value = data.get("reply_to")
        reply_to: int | None
        if reply_to_value is None or isinstance(reply_to_value, bool):
            reply_to = None
        elif isinstance(reply_to_value, int):
            reply_to = reply_to_value
        else:
            reply_to = None

        if speak:
            if not text:
                return None
            return Reply(speak=True, text=text, reply_to=reply_to)
        return Reply(speak=False, text="", reply_to=reply_to)
    except Exception:
        # The model reply is untrusted external text: any parse failure (not JSON,
        # truncated wrapper, unexpected type) means "silence", not a crash.
        return None
