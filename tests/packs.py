"""Paths of the persona packs that tests read directly."""

import copy
from pathlib import Path

from sotto.fragments import Fragments, load_valid_fragments

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_PACK = REPO_ROOT / "personas" / "_legacy_ru"
DEMO_PACK = REPO_ROOT / "personas" / "demo"


def _load_legacy_fragments() -> Fragments:
    return load_valid_fragments(LEGACY_PACK / "fragments.yaml")


# Fragments of the legacy pack: what tests pass as ``frag=`` to the engine functions.
LEGACY_FRAG = _load_legacy_fragments()


def frag_with(**changes: object) -> Fragments:
    """Legacy fragments with some leaves replaced; ``group__leaf=value`` sets ``group.leaf``."""
    data = copy.deepcopy(dict(LEGACY_FRAG.data))
    for name, value in changes.items():
        group, leaf = name.split("__", 1)
        data[group][leaf] = value
    return Fragments(data)
