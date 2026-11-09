"""Engine-independent runner: loads scenarios, drives the adapter, writes/compares outputs."""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.golden import adapter, canonical
from tools.golden.schema import GROUPS, Scenario, load_scenarios

_ALLOW_LINE_RE = re.compile(r"^\s*[-*]\s*`?([A-Za-z0-9_]+)`?\s*[:\s]")


def run_scenarios(
    pack: Path, scenarios: Sequence[Scenario], *, workdir: Path | None = None
) -> dict[str, str]:
    """Run every scenario in a fresh event loop and temp dir; return id -> canonical JSON."""
    outputs: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="golden-") as tmp:
        base = workdir or Path(tmp)
        for scenario in scenarios:
            scenario_dir = base / scenario.id
            scenario_dir.mkdir(parents=True, exist_ok=True)
            result: dict[str, Any] = asyncio.run(adapter.run_scenario(pack, scenario, scenario_dir))
            outputs[scenario.id] = canonical.dumps(result)
    return outputs


def _select(scenarios: list[Scenario], only: Sequence[str]) -> list[Scenario]:
    """``only`` items are scenario ids or group letters (``A``)."""
    if not only:
        return scenarios
    wanted = set(only)
    chosen = [s for s in scenarios if s.id in wanted or s.group in wanted]
    missing = {w for w in wanted if w not in {s.id for s in chosen} and w not in GROUPS}
    if missing:
        raise SystemExit(f"unknown scenario ids: {sorted(missing)}")
    return chosen


def record(pack: Path, scenarios_path: Path, out: Path, *, only: Sequence[str] = ()) -> int:
    scenarios = _select(load_scenarios(scenarios_path, enforce_minimums=not only), list(only))
    outputs = run_scenarios(pack, scenarios)
    out.mkdir(parents=True, exist_ok=True)
    if not only:
        for stale in out.glob("*.json"):
            stale.unlink()
    for scenario_id, text in outputs.items():
        (out / f"{scenario_id}.json").write_text(text, encoding="utf-8")
    print(f"recorded {len(outputs)} scenarios to {out}")
    return 0


def parse_allowed(path: Path | None) -> set[str]:
    """Scenario ids listed as ``- <id>: reason`` in ALLOWED_DIFFS.md."""
    if path is None or not path.exists():
        return set()
    allowed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ALLOW_LINE_RE.match(line)
        if match:
            allowed.add(match.group(1))
    return allowed


@dataclass
class Comparison:
    """Result of comparing outputs with a baseline directory."""

    diffs: dict[str, str] = field(default_factory=dict)  # scenario id -> unified diff
    allowed: list[str] = field(default_factory=list)  # ids whose diff is in ALLOWED_DIFFS.md
    missing: list[str] = field(default_factory=list)  # scenario without a baseline file
    extra: list[str] = field(default_factory=list)  # baseline file without a scenario

    @property
    def ok(self) -> bool:
        return not (self.diffs or self.missing or self.extra)


def diff_outputs(
    outputs: dict[str, str], baseline: Path, *, allowed: set[str], check_extra: bool
) -> Comparison:
    result = Comparison()
    for scenario_id, actual in outputs.items():
        path = baseline / f"{scenario_id}.json"
        if not path.exists():
            result.missing.append(scenario_id)
            continue
        expected = path.read_text(encoding="utf-8")
        if expected == actual:
            continue
        if scenario_id in allowed:
            result.allowed.append(scenario_id)
            continue
        result.diffs[scenario_id] = canonical.unified_diff(expected, actual, scenario_id)
    if check_extra:
        result.extra = sorted({p.stem for p in baseline.glob("*.json")} - set(outputs))
    return result


def compare(
    pack: Path,
    scenarios_path: Path,
    baseline: Path,
    *,
    allow: Path | None = None,
    only: Sequence[str] = (),
) -> int:
    scenarios = _select(load_scenarios(scenarios_path, enforce_minimums=not only), list(only))
    outputs = run_scenarios(pack, scenarios)
    result = diff_outputs(outputs, baseline, allowed=parse_allowed(allow), check_extra=not only)
    for scenario_id in result.allowed:
        print(f"ALLOWED diff in {scenario_id} (ALLOWED_DIFFS.md)")
    for scenario_id, diff in result.diffs.items():
        print(f"DIFF in {scenario_id}:")
        print(diff)
    for scenario_id in result.missing:
        print(f"MISSING baseline for {scenario_id}", file=sys.stderr)
    for scenario_id in result.extra:
        print(f"EXTRA baseline file without scenario: {scenario_id}", file=sys.stderr)
    if not result.ok:
        failures = len(result.diffs) + len(result.missing) + len(result.extra)
        print(f"golden compare FAILED: {failures} scenario(s)", file=sys.stderr)
        return 1
    print(f"golden compare OK: {len(outputs)} scenarios identical")
    return 0
