"""Paths of the persona packs that tests read directly."""

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


RU = get_language("ru")


def legacy_config() -> Config:
    """The legacy pack's parsed config (Russian persona and filter lists): what bare ``Config()``
    was before the engine defaults became neutral."""
    return load_config(load_persona(LEGACY_PACK).config_layers())


def parse_ru_vocab(raw: str) -> VocabDict:
    return parse_vocab(raw, RU)


def ru_vocab(dicts: Sequence[VocabDict] = ()) -> Vocab:
    return Vocab(dicts, lang=RU)
