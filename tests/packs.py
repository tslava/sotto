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
DEMO_PACK = REPO_ROOT / "personas" / "demo"
# Synthetic Russian pack for the language tests (tests/lang).
RU_PACK = REPO_ROOT / "tests" / "fixtures" / "ru_pack"


RU = get_language("ru")


def parse_ru_vocab(raw: str) -> VocabDict:
    return parse_vocab(raw, RU)


def ru_vocab(dicts: Sequence[VocabDict] = ()) -> Vocab:
    return Vocab(dicts, lang=RU)


# Fragments and config of the synthetic Russian pack (tests/lang).
RU_FRAG = load_valid_fragments(RU_PACK / "fragments.yaml")


def ru_frag_with(**changes: object) -> Fragments:
    """Russian-pack fragments with leaves replaced; ``group__leaf=value`` sets ``group.leaf``."""
    data = copy.deepcopy(dict(RU_FRAG.data))
    for name, value in changes.items():
        group, leaf = name.split("__", 1)
        data[group][leaf] = value
    return Fragments(data)


def ru_config() -> Config:
    """The synthetic Russian pack's parsed config (Russian persona and filter lists)."""
    return load_config(load_persona(RU_PACK).config_layers())
