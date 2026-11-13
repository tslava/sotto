"""Self-check of the golden harness (tools/golden): determinism, a clean compare against the
committed synthetic baseline, and a mutation per scenario group.

A group whose mutation is not detected has missing scenarios: the failing assertion names it.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.golden import runner
from tools.golden.schema import GROUPS, load_scenarios

from sotto.fragments import ALL_FRAGMENT_KEYS

REPO = Path(__file__).resolve().parent.parent
PACK = REPO / "personas" / "_legacy_ru"
SCENARIOS = REPO / "tools" / "golden" / "scenarios_ru.yaml"
BASELINE = REPO / "tools" / "golden" / "baseline_synthetic"
ALLOWED = REPO / "tools" / "golden" / "ALLOWED_DIFFS.md"
PACK_ITEMS = (
    "pack.yaml",
    "config.yaml",
    "few_shot.yaml",
    "prompts",
    "CHANGELOG.md",
    "stickers.yaml",
    "places_manual.yaml",
    "fragments.yaml",
)


def copy_pack(destination: Path) -> Path:
    """Copy the repo-root pack files into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in PACK_ITEMS:
        source = PACK / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
        else:
            shutil.copy2(source, destination / name)
    return destination


def append_text(pack: Path, relative: str, text: str) -> None:
    path = pack / relative
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def edit_config(pack: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    path = pack / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


# One mutation per scenario group: a piece of the pack that only that group is certain to see.
def mutate_a(pack: Path) -> None:  # main prompt
    append_text(pack, "prompts/system.txt", "\nGOLDEN-MUTATION\n")


def mutate_b(pack: Path) -> None:  # auxiliary model prompt (judge)
    append_text(pack, "prompts/judge.txt", "\nGOLDEN-MUTATION\n")


def mutate_c(pack: Path) -> None:  # deterministic functions: a new stop-topic pattern
    edit_config(pack, lambda data: data["filters"]["topic_stop"].insert(0, "\\bрыбалк"))


def mutate_d(pack: Path) -> None:  # config dump
    edit_config(pack, lambda data: data["behaviour"].__setitem__("daily_cap", 4))


def mutate_e(pack: Path) -> None:  # owner UI: the changelog shown by /changelog
    path = pack / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    assert "### Для чата\n" in text
    path.write_text(text.replace("### Для чата\n", "### Для чата\nGOLDEN-MUTATION\n", 1), "utf-8")


def mutate_f(pack: Path) -> None:  # persistence: the hot window length lands in state
    edit_config(pack, lambda data: data["behaviour"]["hot_window"].__setitem__("minutes", 11))


MUTATIONS: dict[str, Callable[[Path], None]] = {
    "A": mutate_a,
    "B": mutate_b,
    "C": mutate_c,
    "D": mutate_d,
    "E": mutate_e,
    "F": mutate_f,
}


def test_every_group_has_a_mutation() -> None:
    assert set(MUTATIONS) == set(GROUPS)


def test_committed_baseline_matches() -> None:
    scenarios = load_scenarios(SCENARIOS)
    outputs = runner.run_scenarios(PACK, scenarios)
    result = runner.diff_outputs(
        outputs, BASELINE, allowed=runner.parse_allowed(ALLOWED), check_extra=True
    )
    assert result.ok, (sorted(result.diffs), result.missing, result.extra)


def test_record_twice_is_byte_identical(tmp_path: Path) -> None:
    scenarios = load_scenarios(SCENARIOS)[::6]
    first = runner.run_scenarios(PACK, scenarios)
    second = runner.run_scenarios(PACK, scenarios)
    assert first == second
    assert len(first) > 10


def _mutate_leaves(data: Any, suffix: str) -> Any:
    """Append ``suffix`` to every string in a fragments mapping (list items and map values too)."""
    if isinstance(data, str):
        return data + suffix
    if isinstance(data, list):
        return [_mutate_leaves(item, suffix) for item in data]
    if isinstance(data, dict):
        return {key: _mutate_leaves(value, suffix) for key, value in data.items()}
    return data


def _leaf_token(key: str) -> str:
    return "zq" + key.replace(".", "x").replace("_", "y") + "qz"


def _tagged_fragments(pack: Path) -> dict[str, str]:
    """Rewrite the pack's fragments with a lowercase token at the end of every leaf; return
    ``key -> token``. Lowercase, because the weather place line goes through ``capitalize()``."""
    path = pack / "fragments.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tokens: dict[str, str] = {}
    for key in sorted(ALL_FRAGMENT_KEYS):
        group, leaf = key.split(".", 1)
        tokens[key] = _leaf_token(key)
        data[group][leaf] = _mutate_leaves(data[group][leaf], tokens[key])
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return tokens


# Leaves no golden scenario reaches (the baselines are never re-recorded, so scenarios cannot be
# added for them); tests/test_fragments_engine.py covers them instead. ``messages_empty`` guards
# a branch that cannot happen (numbered rows are never blank).
UNEXERCISED_LEAVES = (
    "placeholders.context_empty",
    "placeholders.messages_empty",
    "placeholders.no_caption",
    "render.places_block_header",
)


def test_every_fragment_leaf_reaches_a_scenario_output(tmp_path: Path) -> None:
    """Replaces the v1 substring-coverage tool: tag every fragment leaf, run all scenarios and
    require each tag in at least one canonical output (rendered text, not raw leaf text)."""
    pack = copy_pack(tmp_path / "pack")
    tokens = _tagged_fragments(pack)

    outputs = runner.run_scenarios(pack, load_scenarios(SCENARIOS))

    joined = "\n".join(outputs.values())
    unreached = sorted(key for key, token in tokens.items() if token not in joined)
    assert unreached == sorted(UNEXERCISED_LEAVES), (
        f"golden coverage of fragment leaves: {unreached}"
    )


# One leaf per fragment group, with the scenario group that renders it.
FRAGMENT_MUTATIONS = {
    "placeholders": ("placeholders.recent_replies_empty", "A"),
    "situations": ("situations.late", "A"),
    "prompt": ("prompt.data_notice", "A"),
    "render": ("render.jokes_header", "A"),
    "weather": ("weather.place_prefix", "A"),
    "vocab": ("vocab.offer_header", "A"),
    "tools": ("tools.places_fact_prompt", "B"),
}


def test_every_fragment_group_has_a_mutation() -> None:
    assert set(FRAGMENT_MUTATIONS) == {key.split(".", 1)[0] for key in ALL_FRAGMENT_KEYS}


@pytest.mark.parametrize("fragment_group", sorted(FRAGMENT_MUTATIONS))
def test_fragment_mutation_is_detected(fragment_group: str, tmp_path: Path) -> None:
    key, scenario_group = FRAGMENT_MUTATIONS[fragment_group]
    pack = copy_pack(tmp_path / "pack")
    path = pack / "fragments.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    group, leaf = key.split(".", 1)
    data[group][leaf] += " GOLDEN-MUTATION"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    scenarios = [s for s in load_scenarios(SCENARIOS) if s.group == scenario_group]

    outputs = runner.run_scenarios(pack, scenarios)
    result = runner.diff_outputs(outputs, BASELINE, allowed=set(), check_extra=False)

    assert result.diffs, f"fragment group {fragment_group}: mutation not detected"


@pytest.mark.parametrize("group", GROUPS)
def test_mutation_is_detected(group: str, tmp_path: Path) -> None:
    pack = copy_pack(tmp_path / "pack")
    MUTATIONS[group](pack)
    scenarios = [s for s in load_scenarios(SCENARIOS) if s.group == group]
    outputs = runner.run_scenarios(pack, scenarios)
    result = runner.diff_outputs(outputs, BASELINE, allowed=set(), check_extra=False)
    assert result.diffs, f"group {group}: mutation not detected, scenarios are missing"
    assert not result.missing
