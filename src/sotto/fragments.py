"""Pack fragments (``fragments.yaml``) and the prompt slot tables.

``Fragments`` has its final API but is still a stub: the engine keeps its built-in
wording until the fragment extraction PR moves the pieces into the pack. A pack without
``fragments.yaml`` loads an empty ``Fragments``. The prompt slot tables are enforced
by ``packcheck`` and at startup.
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

# dotted fragment key -> placeholders it must contain exactly. Filled by the extraction PR.
FRAGMENT_SPEC: dict[str, frozenset[str]] = {}


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

    def render(self, key: str, **params: str) -> str:
        """Single-pass ``{name}`` substitution (never ``str.format``: data may hold braces)."""
        value = self.get(key)
        if not isinstance(value, str):
            raise TypeError(f"fragment {key!r} is not a string")
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
