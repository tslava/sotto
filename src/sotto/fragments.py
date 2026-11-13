"""Pack fragments (``fragments.yaml``) and the prompt slot tables.

``fragments.yaml`` carries every wording the engine puts into model prompts or persists
(situations, prompt scaffolding, block headers, placeholders, weather and vocabulary pieces,
offline tool prompts). The engine reads it through ``Fragments``; ``FRAGMENT_SPEC`` is the
schema (every leaf, the ``{placeholders}`` it must contain exactly), checked at load and by
``packcheck``. The prompt slot tables are enforced the same way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Every slot the prompt renderers know (the closed set of prompt.py:_SLOT_RE).
MAIN_SLOTS: frozenset[str] = frozenset(
    {
        "age",
        "few_shot",
        "life",
        "diary",
        "vocab",
        "chat_memory",
        "jokes",
        "weather",
        "context",
        "recent_replies",
        "places",
        "avoid",
        "situation",
    }
)
_MEMORY_SLOTS = frozenset({"period", "chat", "limit"})


def _exact(*slots: str) -> tuple[frozenset[str], frozenset[str]]:
    """Slots a prompt must have and may have are the same: the renderer's own regex."""
    group = frozenset(slots)
    return group, group


# prompt stem -> (required slots, allowed slots). Allowed is the slot regex of the prompt's own
# renderer (prompt.py, judge.py, stickers.py, followup.py, reactions.py, chat_memory.py, ...):
# a slot outside it stays literal in the rendered text, which is the failure this table prevents.
PROMPT_SLOTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "system": (frozenset({"context", "situation", "few_shot", "age"}), MAIN_SLOTS),
    "judge": _exact("trigger", "candidate"),
    "sticker": _exact("menu", "trigger", "reply"),
    "followup": _exact("context", "recent_replies", "name", "text"),
    "name_check": _exact("context", "recent_replies", "name", "text"),
    "checkin_prefilter": _exact("messages", "recent_replies", "names"),
    "reaction": _exact("context", "name", "text", "emoji", "reply_to"),
    "memory": (frozenset({"period", "chat"}), _MEMORY_SLOTS),
    "memory_jokes": _exact("period", "chat", "limit"),
    "memory_threads": (frozenset({"period", "chat"}), _MEMORY_SLOTS),
    "diary": _exact("reply"),
    "vision": _exact("caption"),
    "weather_place": _exact("text"),
}

PROMPT_NAMES: tuple[str, ...] = tuple(PROMPT_SLOTS)

_SLOT_TOKEN_RE = re.compile(r"\{([a-z_]+)\}")


def _spec(*rows: tuple[str, str]) -> dict[str, frozenset[str]]:
    return {key: frozenset(slots.split()) for key, slots in rows}


# dotted string-leaf key -> placeholders it must contain exactly (``{name}`` tokens).
FRAGMENT_SPEC: dict[str, frozenset[str]] = _spec(
    # Strings that are persisted (messages.text, bot_replies.text, display_name): changing one
    # orphans the old rows, so every consumer derives its prefix/regex from these values.
    ("placeholders.photo", ""),
    ("placeholders.photo_with_desc", "desc"),
    ("placeholders.sticker", ""),
    ("placeholders.voice", ""),
    ("placeholders.video", ""),
    ("placeholders.file", ""),
    ("placeholders.no_caption", ""),
    ("placeholders.participant", "n"),
    ("placeholders.sticker_tag", "id"),
    ("placeholders.recent_replies_empty", ""),
    ("placeholders.context_empty", ""),
    ("placeholders.messages_empty", ""),
    ("placeholders.reply_to_none", ""),
    ("placeholders.reply_to_unavailable", ""),
    # The "situation" paragraph of the user message, one per trigger.
    ("situations.ambient", ""),
    ("situations.late", ""),
    ("situations.morning", ""),
    ("situations.spontaneous", ""),
    ("situations.life", "text"),
    ("situations.addressed_single", "name text"),
    ("situations.addressed_multi_header", ""),
    ("situations.addressed_multi_footer", ""),
    ("situations.followup_single", "name text"),
    ("situations.followup_multi_header", ""),
    ("situations.followup_multi_footer", ""),
    ("situations.checkin_header", ""),
    ("situations.checkin_footer", ""),
    ("situations.callback", "name text"),
    # Scaffolding of the main user message and of the auxiliary prompts.
    ("prompt.places_none", ""),
    ("prompt.data_notice", ""),
    ("prompt.recent_replies_label", ""),
    ("prompt.json_reminder", ""),
    ("prompt.json_reminder_checkin", ""),
    ("prompt.context_marker", ""),
    ("prompt.recent_replies_marker", ""),
    ("prompt.places_marker", ""),
    ("prompt.no_target", ""),
    # Headers and pieces of the blocks that go into the system prompt.
    ("render.life_header", ""),
    ("render.jokes_header", ""),
    ("render.chat_memory_header", ""),
    ("render.diary_today_header", ""),
    ("render.diary_week_header", ""),
    ("render.avoid_template", "items"),
    ("render.avoid_no_story", ""),
    (
        "render.self_share",
        "days ambient_hit ambient_total address_hit address_total other_hit other_total",
    ),
    ("render.places_block_header", ""),
    ("render.places_menu_header", ""),
    ("render.places_quiet", ""),
    ("render.places_noisy", ""),
    ("render.quote_open", ""),
    ("render.quote_close", ""),
    # The weather block is assembled in the engine from these pieces.
    ("weather.home_prefix", ""),
    ("weather.now_word", ""),
    ("weather.and_word", ""),
    ("weather.today_from", ""),
    ("weather.range_to", ""),
    ("weather.tomorrow_from", ""),
    ("weather.tomorrow_from_lower", ""),
    ("weather.place_prefix", ""),
    ("vocab.offer_header", ""),
    ("vocab.offer_tail", ""),
    ("vocab.request_header", ""),
    ("vocab.request_tail", ""),
    ("vocab.meanings_header", ""),
    # Prompts of the offline CLIs (places_fill, stickers_fill).
    ("tools.places_fact_prompt", ""),
    ("tools.places_reviews_preamble", ""),
    ("tools.sticker_ocr_prompt", ""),
)

# Leaves that may contain braces other than ``{placeholders}`` (JSON answer formats).
LITERAL_BRACES: frozenset[str] = frozenset(
    {"prompt.json_reminder", "prompt.json_reminder_checkin", "tools.sticker_ocr_prompt"}
)

# List leaves (key -> required length) and mapping leaves (key -> value type).
FRAGMENT_LISTS: dict[str, int] = {"render.weekdays": 7}
FRAGMENT_MAPS: frozenset[str] = frozenset({"render.accusative", "weather.wmo"})

ALL_FRAGMENT_KEYS: frozenset[str] = (
    frozenset(FRAGMENT_SPEC) | frozenset(FRAGMENT_LISTS) | FRAGMENT_MAPS
)


def slot_errors(name: str, body: str) -> list[str]:
    """Slot problems of one prompt body: unknown prompt, missing required, disallowed slots."""
    spec = PROMPT_SLOTS.get(name)
    if spec is None:
        return [f"prompts/{name}.txt: not a known prompt"]
    required, allowed = spec
    found = set(_SLOT_TOKEN_RE.findall(body))
    errors = [
        f"prompts/{name}.txt: required slot {{{slot}}} is missing"
        for slot in sorted(required - found)
    ]
    errors += [
        f"prompts/{name}.txt: slot {{{slot}}} is not allowed" for slot in sorted(found - allowed)
    ]
    return errors


def _leaf_keys(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys of the leaves of a nested mapping; ``FRAGMENT_MAPS`` keys stay leaves."""
    keys: list[str] = []
    for name, value in data.items():
        key = f"{prefix}{name}"
        if isinstance(value, Mapping) and key not in FRAGMENT_MAPS:
            keys += _leaf_keys(value, f"{key}.")
        else:
            keys.append(key)
    return keys


def fragment_errors(data: Mapping[str, Any]) -> list[str]:
    """Schema problems of a parsed ``fragments.yaml`` (empty list = valid).

    Unknown and missing keys, wrong value types, and ``{placeholders}`` that differ from the
    spec are errors. Messages carry key names only, never values.
    """
    errors: list[str] = []
    found = set(_leaf_keys(data))
    for key in sorted(ALL_FRAGMENT_KEYS - found):
        errors.append(f"fragments.yaml: missing key {key}")
    for key in sorted(found - ALL_FRAGMENT_KEYS):
        errors.append(f"fragments.yaml: unknown key {key}")
    holder = Fragments(data)
    for key, expected in FRAGMENT_SPEC.items():
        if key not in found:
            continue
        value = holder.get(key)
        if not isinstance(value, str):
            errors.append(f"fragments.yaml: {key} must be a string")
            continue
        tokens = set(_SLOT_TOKEN_RE.findall(value))
        if tokens != expected:
            want = ", ".join(sorted(expected)) or "none"
            got = ", ".join(sorted(tokens)) or "none"
            errors.append(f"fragments.yaml: {key} placeholders are [{got}], expected [{want}]")
        if key not in LITERAL_BRACES and _stray_brace(value):
            errors.append(f"fragments.yaml: {key} contains a stray brace")
    for key, size in FRAGMENT_LISTS.items():
        if key not in found:
            continue
        value = holder.get(key)
        if (
            not isinstance(value, list)
            or len(value) != size
            or not all(isinstance(v, str) for v in value)
        ):
            errors.append(f"fragments.yaml: {key} must be a list of {size} strings")
    for key in sorted(FRAGMENT_MAPS):
        if key not in found:
            continue
        value = holder.get(key)
        if not isinstance(value, Mapping) or not value:
            errors.append(f"fragments.yaml: {key} must be a non-empty mapping")
        elif not all(isinstance(v, str) for v in value.values()):
            errors.append(f"fragments.yaml: {key} values must be strings")
    return errors


def _stray_brace(value: str) -> bool:
    return "{" in _SLOT_TOKEN_RE.sub("", value) or "}" in _SLOT_TOKEN_RE.sub("", value)


@dataclass(frozen=True)
class Fragments:
    """Dotted-key lookup over the loaded ``fragments.yaml`` (empty until the pack has one)."""

    data: Mapping[str, Any] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return False
            node = node[part]
        return True

    def get(self, key: str) -> Any:
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(key)
            node = node[part]
        return node

    def text(self, key: str) -> str:
        """A string leaf (TypeError if the leaf is a list or a mapping)."""
        value = self.get(key)
        if not isinstance(value, str):
            raise TypeError(f"fragment {key!r} is not a string")
        return value

    def items(self, key: str) -> list[str]:
        """A list-of-strings leaf (``render.weekdays``)."""
        value = self.get(key)
        if not isinstance(value, list):
            raise TypeError(f"fragment {key!r} is not a list")
        return [str(item) for item in value]

    def table(self, key: str) -> Mapping[Any, str]:
        """A mapping leaf (``render.accusative``, ``weather.wmo``)."""
        value = self.get(key)
        if not isinstance(value, Mapping):
            raise TypeError(f"fragment {key!r} is not a mapping")
        return value

    def prefix(self, key: str) -> str:
        """The part of a string leaf before its first ``{placeholder}`` (the whole value if none).

        Prefix checks on persisted rows (sticker tag, participant fallback) are derived from
        the pack value this way, so a pack that rewords the placeholder keeps old rows matching
        as long as the prefix is what the stored rows carry."""
        return self.text(key).split("{", 1)[0]

    def render(self, key: str, **params: str) -> str:
        """Single-pass ``{name}`` substitution (never ``str.format``: data may hold braces)."""
        value = self.text(key)
        return _SLOT_TOKEN_RE.sub(lambda m: params.get(m.group(1), m.group(0)), value)


def load_fragments(path: Path) -> Fragments:
    """Read ``fragments.yaml``; a missing file gives empty fragments."""
    if not path.exists():
        return Fragments()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return Fragments()
    if not isinstance(loaded, dict):
        raise ValueError(f"fragments file {path} must contain a mapping at top level")
    return Fragments(loaded)


def load_valid_fragments(path: Path) -> Fragments:
    """Read ``fragments.yaml`` and validate it against ``FRAGMENT_SPEC`` (ValueError if not)."""
    if not path.is_file():
        raise ValueError(f"{path.name} not found")
    frag = load_fragments(path)
    errors = fragment_errors(frag.data)
    if errors:
        raise ValueError("; ".join(errors))
    return frag
