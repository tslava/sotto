"""The modules that used to carry persona wording must not grow Cyrillic constants again.

AST check of PR-3 (plan 04, section 5): in the modules whose wording moved into the pack
(``fragments.yaml``) no module-level UPPER constant may hold Cyrillic text, except language
logic (morphology tables, the fact-character class). Docstrings, comments, log messages and
CLI texts are a later stage and are not looked at here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "sotto"
CYRILLIC = re.compile("[А-Яа-яЁё]")
UPPER_NAME = re.compile(r"^_*[A-Z][A-Z0-9_]*$")

# Modules PR-3 took the wording out of.
CHECKED = (
    "prompt",
    "judge",
    "followup",
    "reactions",
    "stickers",
    "sanitize",
    "export_parser",
    "postprocess",
    "bot",
    "responder",
    "weather",
    "places",
    "places_fill",
    "stickers_fill",
    "vocab",
    "diary",
    "jokes",
    "chat_memory",
    "motifs",
    "vision",
)

# Language logic (class L in plan 1.1): stays until the language modules (PR-4).
LANGUAGE_LOGIC: dict[str, frozenset[str]] = {
    "places": frozenset({"_FACT_RE"}),
    "vocab": frozenset(
        {
            "_REFLEXIVE",
            "_VERB_ENDINGS",
            "_ADJ_ENDINGS_WORD",
            "_FLEETING",
            "_SOFT_ENDINGS",
            "_NOUN_ENDINGS",
            "_ADJ_ENDINGS",
            "_MYA_ENDINGS",
            "_VERB_FORMS",
            "_PRESENT_E",
            "_PRESENT_I",
            "_VERB_PREFIXES",
        }
    ),
}


def _cyrillic_constants(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name) and UPPER_NAME.match(t.id)]
        has_text = any(
            isinstance(c, ast.Constant) and isinstance(c.value, str) and CYRILLIC.search(c.value)
            for c in ast.walk(value)
        )
        if names and has_text:
            found += names
    return found


@pytest.mark.parametrize("module", CHECKED)
def test_no_cyrillic_upper_constants(module: str) -> None:
    constants = _cyrillic_constants(SRC / f"{module}.py")

    unexpected = sorted(set(constants) - LANGUAGE_LOGIC.get(module, frozenset()))
    assert unexpected == [], f"{module}.py holds wording that belongs in fragments.yaml"


def test_language_logic_allowlist_has_no_stale_entries() -> None:
    for module, names in LANGUAGE_LOGIC.items():
        assert set(names) <= set(_cyrillic_constants(SRC / f"{module}.py"))


def test_the_check_sees_a_planted_constant(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text('_PLANTED = ("a " "привет")\nplain = "привет"\nOK = "x"\n', encoding="utf-8")

    assert _cyrillic_constants(sample) == ["_PLANTED"]
