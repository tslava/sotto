"""Scenario file schema. Engine-independent: it only validates the YAML shape.

The meaning of an action ``kind`` and of its arguments lives in ``adapter.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

GROUPS = ("A", "B", "C", "D", "E", "F")

# Minimum scenario counts per group (plan 3.2: the harness requires >= 30 in group A).
MIN_PER_GROUP: dict[str, int] = {"A": 30, "B": 12, "C": 10, "D": 8, "E": 20, "F": 8}

# Action kinds understood by the adapter. Kept here so a typo fails at load time.
ACTION_KINDS = frozenset(
    {
        "gate_pass",
        "ingest",
        "command",
        "morning",
        "spontaneous",
        "checkin",
        "callback",
        "restore_pending",
        "life",
        "say",
        "preview",
        "aux",
        "call",
        "config_dump",
        "describe_keys",
        "pattern_dump",
        "alert",
        "advance",
    }
)

CAPTURES = frozenset({"requests", "sent", "db", "result"})

_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
_OFFSET_RE = re.compile(r"^-?(\d+[smhd]?)+$")
_SEGMENT_RE = re.compile(r"(\d+)([smhd]?)")
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_offset(value: int | str) -> int:
    """Seconds relative to the scenario clock: ``-90``, ``"-45m"``, ``"-2h"``, ``"4h30m"``."""
    if isinstance(value, int):
        return value
    text = value.strip()
    if _OFFSET_RE.match(text) is None:
        raise ValueError(f"bad time offset: {value!r}")
    sign = -1 if text.startswith("-") else 1
    total = sum(int(num) * _UNITS[unit] for num, unit in _SEGMENT_RE.findall(text))
    return sign * total


Offset = int | str


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedMessage(_Strict):
    id: int
    user: int
    text: str
    at: Offset = 0
    reply_to: int | None = None
    bot: bool = False

    @field_validator("at")
    @classmethod
    def _check_at(cls, value: Offset) -> Offset:
        parse_offset(value)
        return value


class SeedBotReply(_Strict):
    id: int
    text: str
    trigger: str = "ambient"
    at: Offset = 0
    reply_to: int | None = None
    trigger_msg: int | None = None
    delay: int = 0


class SeedLife(_Strict):
    text: str
    at: Offset = 0


class SeedFact(_Strict):
    text: str
    at: Offset = 0


class SeedMemory(_Strict):
    start: Offset
    end: Offset
    text: str


class SeedJoke(_Strict):
    text: str
    at: Offset = 0
    used_at: Offset | None = None


class SeedThread(_Strict):
    user: int
    text: str
    at: Offset = 0
    closed: bool = False


class SeedPlace(_Strict):
    id: str
    name: str
    district: str = ""
    category: str = "pub"
    rating: float = 4.5
    reviews: int = 100
    price_level: int | None = None
    quiet: bool = False
    fact: str = ""
    operational: bool = True


class SeedPending(_Strict):
    trigger_msg: int
    user: int
    trigger: str = "mention"
    due: Offset = 0
    created: Offset = 0


class SeedNight(_Strict):
    id: int
    user: int
    text: str
    at: Offset = -3600


class Seed(_Strict):
    # user id -> display name; used for messages, passes and commands.
    users: dict[int, str] = Field(default_factory=dict)
    messages: list[SeedMessage] = Field(default_factory=list)
    bot_replies: list[SeedBotReply] = Field(default_factory=list)
    # State keys; ``key@day`` / ``key@week`` get the local date / ISO week suffix of the clock.
    state: dict[str, str] = Field(default_factory=dict)
    life: list[SeedLife] = Field(default_factory=list)
    diary: list[SeedFact] = Field(default_factory=list)
    memory: list[SeedMemory] = Field(default_factory=list)
    jokes: list[SeedJoke] = Field(default_factory=list)
    threads: list[SeedThread] = Field(default_factory=list)
    places: list[SeedPlace] = Field(default_factory=list)
    pending: list[SeedPending] = Field(default_factory=list)
    night: list[SeedNight] = Field(default_factory=list)
    mutes: list[int] = Field(default_factory=list)
    # Vocabulary dictionaries as YAML text (the format of /words uploads).
    vocab: list[str] = Field(default_factory=list)


class WeatherHome(_Strict):
    name: str = ""
    lat: float
    lon: float


class Defaults(_Strict):
    """File-level values the harness uses where a scenario does not say otherwise (set under the
    top-level ``defaults:`` key of a scenario file, so the harness itself has no persona text)."""

    # The home point of ``weather: {home: default}``; required if any scenario uses "default".
    home: WeatherHome | None = None
    guest_name: str = "Guest"  # display name for helper calls that take none
    place_name: str = "Elsewhere"  # geocoded place name of the weather rendering cases
    callback_name: str = "Name"  # name in the callback situation case


class WeatherSpec(_Strict):
    # "default" -> the file's ``defaults.home``; "none" -> no home point (the home block is empty).
    home: WeatherHome | Literal["default", "none"] = "default"
    forecast: dict[str, Any] | None = None
    # query -> geocoder result item (name, latitude, longitude, country, country_code)
    places: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Action(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        if value not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {value!r}")
        return value


class Scenario(_Strict):
    id: str
    group: Literal["A", "B", "C", "D", "E", "F"]
    desc: str = ""
    # Local time in the pack timezone, ``YYYY-MM-DD HH:MM[:SS]``; default: the harness clock.
    clock: str | None = None
    seed: Seed = Field(default_factory=Seed)
    config: dict[str, str] = Field(default_factory=dict)
    llm: list[str | dict[str, Any]] = Field(default_factory=list)
    weather: WeatherSpec = Field(default_factory=WeatherSpec)
    actions: list[Action]
    capture: list[str] = Field(default_factory=lambda: ["requests", "sent", "result"])
    # Filled in by ``load_scenarios`` from the file's ``defaults:``; not a scenario key.
    defaults: Defaults = Field(default_factory=Defaults)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(f"bad scenario id {value!r}")
        return value

    @field_validator("capture")
    @classmethod
    def _check_capture(cls, value: list[str]) -> list[str]:
        bad = set(value) - CAPTURES
        if bad:
            raise ValueError(f"unknown capture {sorted(bad)}")
        return value


class ScenarioFile(_Strict):
    version: int = 1
    # Free-form block that only holds YAML anchors shared by scenarios; never interpreted.
    defs: Any = None
    defaults: Defaults = Field(default_factory=Defaults)
    scenarios: list[Scenario]


def load_scenarios(path: Path, *, enforce_minimums: bool = True) -> list[Scenario]:
    """Parse and validate a scenario file; raises ValueError with all problems."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        parsed = ScenarioFile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid scenario file:\n{exc}") from exc

    problems: list[str] = []
    seen: set[str] = set()
    for scenario in parsed.scenarios:
        scenario.defaults = parsed.defaults
        if scenario.weather.home == "default":
            if parsed.defaults.home is None:
                problems.append(f"{scenario.id}: weather home 'default' needs defaults.home")
            else:
                scenario.weather.home = parsed.defaults.home
        if scenario.id in seen:
            problems.append(f"duplicate scenario id {scenario.id}")
        seen.add(scenario.id)
        if not scenario.id.startswith(scenario.group + "_"):
            problems.append(f"{scenario.id}: id must start with '{scenario.group}_'")
    if enforce_minimums:
        counts = {group: 0 for group in GROUPS}
        for scenario in parsed.scenarios:
            counts[scenario.group] += 1
        for group, minimum in MIN_PER_GROUP.items():
            if counts[group] < minimum:
                problems.append(f"group {group} has {counts[group]} scenarios, need >= {minimum}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return parsed.scenarios
