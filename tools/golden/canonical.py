"""Canonical JSON for golden outputs and unified diffs."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from typing import Any


def normalize(value: Any) -> Any:
    """Make a value JSON-stable: sets sorted, tuples/dataclass-likes to lists, keys to str."""
    if isinstance(value, Mapping):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((normalize(item) for item in value), key=lambda item: json.dumps(item))
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float):
        return round(value, 9)
    return value


def dumps(value: Any) -> str:
    """One canonical JSON document: sorted keys, indent 1, non-ASCII kept, trailing newline."""
    return json.dumps(normalize(value), sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def unified_diff(expected: str, actual: str, name: str) -> str:
    lines = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=f"baseline/{name}.json",
        tofile=f"current/{name}.json",
        n=3,
    )
    return "".join(lines)
