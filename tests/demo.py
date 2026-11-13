"""The English demo pack as test fixtures: what ``tests/packs.py`` is for the legacy pack."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

from sotto.config import load_config
from sotto.config_models import Config
from sotto.fragments import Fragments, load_valid_fragments
from sotto.lang import get_language
from sotto.persona import load_persona
from sotto.vocab import Vocab, VocabDict, parse_vocab

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PACK = REPO_ROOT / "personas" / "demo"

# Fragments of the demo pack: what tests pass as ``frag=`` to the engine functions.
DEMO_FRAG = load_valid_fragments(DEMO_PACK / "fragments.yaml")


def demo_frag_with(**changes: object) -> Fragments:
    """Demo fragments with some leaves replaced; ``group__leaf=value`` sets ``group.leaf``."""
    data = copy.deepcopy(dict(DEMO_FRAG.data))
    for name, value in changes.items():
        group, leaf = name.split("__", 1)
        data[group][leaf] = value
    return Fragments(data)


EN = get_language("en")


def demo_config() -> Config:
    """The demo pack's parsed config (English persona and filter lists)."""
    return load_config(load_persona(DEMO_PACK).config_layers())


def parse_en_vocab(raw: str) -> VocabDict:
    return parse_vocab(raw, EN)


def en_vocab(dicts: Sequence[VocabDict] = ()) -> Vocab:
    return Vocab(dicts, lang=EN)
