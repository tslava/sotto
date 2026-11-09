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

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = REPO / "tools" / "golden" / "scenarios_ru.yaml"
BASELINE = REPO / "tools" / "golden" / "baseline_synthetic"
ALLOWED = REPO / "tools" / "golden" / "ALLOWED_DIFFS.md"
PACK_ITEMS = (
    "config.yaml",
    "few_shot.yaml",
    "prompts",
    "CHANGELOG.md",
    "stickers.yaml",
    "places_manual.yaml",
)


def copy_pack(destination: Path) -> Path:
    """Copy the repo-root pack files into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in PACK_ITEMS:
        source = REPO / name
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
    outputs = runner.run_scenarios(REPO, scenarios)
    result = runner.diff_outputs(
        outputs, BASELINE, allowed=runner.parse_allowed(ALLOWED), check_extra=True
    )
    assert result.ok, (sorted(result.diffs), result.missing, result.extra)


def test_record_twice_is_byte_identical(tmp_path: Path) -> None:
    scenarios = load_scenarios(SCENARIOS)[::6]
    first = runner.run_scenarios(REPO, scenarios)
    second = runner.run_scenarios(REPO, scenarios)
    assert first == second
    assert len(first) > 10


@pytest.mark.parametrize("group", GROUPS)
def test_mutation_is_detected(group: str, tmp_path: Path) -> None:
    pack = copy_pack(tmp_path / "pack")
    MUTATIONS[group](pack)
    scenarios = [s for s in load_scenarios(SCENARIOS) if s.group == group]
    outputs = runner.run_scenarios(pack, scenarios)
    result = runner.diff_outputs(outputs, BASELINE, allowed=set(), check_extra=False)
    assert result.diffs, f"group {group}: mutation not detected, scenarios are missing"
    assert not result.missing
