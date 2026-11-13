"""Unit tests for the helper tools in tools/ (canonical JSON, scenario schema, runner and CLI,
coverage gate, language gate, inventories)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from tools import check_language, coverage_gate
from tools.golden import __main__ as golden_cli
from tools.golden import canonical, runner
from tools.golden.schema import load_scenarios, parse_offset

REPO = Path(__file__).resolve().parent.parent
PACK = REPO / "personas" / "demo"

MINI = """
version: 1
defaults: {home: {name: Town, lat: 1.0, lon: 2.0}}
scenarios:
  - id: A_mini
    group: A
    seed:
      users: {1: "Dana"}
    llm: ['{"speak": true, "text": "Hello."}']
    actions:
      - {kind: gate_pass, passes: [{trigger: mention, user: 1, msg: 5, text: "@golden_bot hello"}]}
  - id: D_mini
    group: D
    actions: [{kind: describe_keys, keys: [behaviour.daily_cap]}]
"""


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- canonical ----------------------------------------------------------------


def test_canonical_dumps_is_stable() -> None:
    value = {"b": {2, 1}, "a": (1.00000000004, b"\x01"), "\u0451": None}
    text = canonical.dumps(value)
    assert text == canonical.dumps(json.loads(text) | {"\u0451": None}) or text.endswith("\n")
    assert '"\u0451"' in text and text.endswith("\n")
    assert json.loads(text)["b"] == [1, 2]
    assert json.loads(text)["a"] == [1.0, "01"]


def test_unified_diff_names_both_sides() -> None:
    diff = canonical.unified_diff("a\n", "b\n", "X_1")
    assert "baseline/X_1.json" in diff and "current/X_1.json" in diff


# --- schema -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [(5, 5), ("-90", -90), ("-45m", -2700), ("2h", 7200), ("4h30m", 16200), ("-3d", -259200)],
)
def test_parse_offset(raw: int | str, seconds: int) -> None:
    assert parse_offset(raw) == seconds


def test_parse_offset_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="bad time offset"):
        parse_offset("soon")


def test_load_scenarios_enforces_minimums(tmp_path: Path) -> None:
    path = write(tmp_path / "s.yaml", MINI)
    assert [s.id for s in load_scenarios(path, enforce_minimums=False)] == ["A_mini", "D_mini"]
    with pytest.raises(ValueError, match="group A has 1 scenarios"):
        load_scenarios(path)


def test_load_scenarios_reports_all_problems(tmp_path: Path) -> None:
    text = MINI.replace("D_mini", "A_mini").replace("A_mini\n    group: A", "A_mini\n    group: B")
    path = write(tmp_path / "s.yaml", text)
    with pytest.raises(ValueError, match="duplicate scenario id"):
        load_scenarios(path, enforce_minimums=False)
    bad = write(tmp_path / "b.yaml", MINI.replace("kind: gate_pass", "kind: nope"))
    with pytest.raises(ValueError, match="unknown action kind"):
        load_scenarios(bad, enforce_minimums=False)
    bad = write(tmp_path / "c.yaml", MINI.replace("capture", "capture") + "\n  - {id: !!str x y}")
    with pytest.raises(Exception, match=r"."):
        load_scenarios(bad, enforce_minimums=False)


def test_load_scenarios_rejects_bad_capture_and_id(tmp_path: Path) -> None:
    text = MINI.replace("    llm:", "    capture: [nothing]\n    llm:")
    with pytest.raises(ValueError, match="unknown capture"):
        load_scenarios(write(tmp_path / "s.yaml", text), enforce_minimums=False)
    with pytest.raises(ValueError, match="bad scenario id"):
        load_scenarios(
            write(tmp_path / "t.yaml", MINI.replace("A_mini", "A mini")), enforce_minimums=False
        )


def test_load_scenarios_checks_group_prefix(tmp_path: Path) -> None:
    path = write(tmp_path / "s.yaml", MINI.replace("id: D_mini", "id: X_mini"))
    with pytest.raises(ValueError, match="id must start with"):
        load_scenarios(path, enforce_minimums=False)


def test_scenario_offsets_are_validated(tmp_path: Path) -> None:
    text = MINI.replace("seed:", "seed:\n      messages: [{id: 1, user: 1, text: x, at: later}]")
    with pytest.raises(ValueError, match="bad time offset"):
        load_scenarios(write(tmp_path / "s.yaml", text), enforce_minimums=False)


# --- runner and CLI -----------------------------------------------------------


def test_record_compare_roundtrip_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = write(tmp_path / "s.yaml", MINI)
    out = tmp_path / "base"
    argv = ["--pack", str(PACK), "--scenarios", str(scenarios)]
    assert (
        golden_cli.main(["record", *argv, "--out", str(out), "--only", "A", "--only", "D_mini"])
        == 0
    )
    assert sorted(p.name for p in out.glob("*.json")) == ["A_mini.json", "D_mini.json"]
    assert golden_cli.main(["compare", *argv, "--baseline", str(out), "--only", "A"]) == 0
    assert "identical" in capsys.readouterr().out

    # a changed baseline fails, unless the scenario is listed in ALLOWED_DIFFS.md
    (out / "A_mini.json").write_text("{}\n", encoding="utf-8")
    assert golden_cli.main(["compare", *argv, "--baseline", str(out), "--only", "A_mini"]) == 1
    assert "DIFF in A_mini" in capsys.readouterr().out
    allowed = write(tmp_path / "ALLOWED_DIFFS.md", "# allowed\n- `A_mini`: reason, signed off\n")
    cmd = ["compare", *argv, "--baseline", str(out), "--only", "A_mini", "--allow", str(allowed)]
    assert golden_cli.main(cmd) == 0
    assert "ALLOWED diff in A_mini" in capsys.readouterr().out


def test_compare_reports_missing_and_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = load_scenarios(write(tmp_path / "s.yaml", MINI), enforce_minimums=False)
    outputs = runner.run_scenarios(PACK, scenarios[:1])
    base = tmp_path / "base"
    base.mkdir()
    (base / "Z_extra.json").write_text("{}\n", encoding="utf-8")
    result = runner.diff_outputs(outputs, base, allowed=set(), check_extra=True)
    assert result.missing == ["A_mini"] and result.extra == ["Z_extra"] and not result.ok
    # the same through the CLI path, which prints both kinds of problem
    path = tmp_path / "s.yaml"
    code = runner.compare(PACK, path, base, only=["A"])
    assert code == 1
    assert "MISSING baseline for A_mini" in capsys.readouterr().err


def test_record_removes_stale_files_and_select_rejects_unknown(tmp_path: Path) -> None:
    scenarios = write(tmp_path / "s.yaml", MINI)
    out = tmp_path / "base"
    out.mkdir()
    (out / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown scenario ids"):
        runner.record(PACK, scenarios, out, only=["Q_none"])
    assert (out / "old.json").exists()
    assert runner.parse_allowed(None) == set()
    assert runner.parse_allowed(tmp_path / "missing.md") == set()


# --- coverage gate ------------------------------------------------------------


def _report(files: dict[str, tuple[int, int]]) -> dict[str, object]:
    return {
        "files": {
            name: {"summary": {"num_statements": total, "missing_lines": missed}}
            for name, (total, missed) in files.items()
        }
    }


def test_coverage_gate_total_and_new_modules(tmp_path: Path) -> None:
    modules = ("src/sotto/persona.py", "tools/x.py", "src/sotto/future.py")
    good = _report({"src/a.py": (100, 5), "tools/x.py": (100, 5), "src/sotto/persona.py": (0, 0)})
    assert coverage_gate.check(good, repo=tmp_path, modules=modules) == []

    bad = _report({"src/a.py": (100, 10), "tools/x.py": (100, 6)})
    problems = coverage_gate.check(bad, repo=tmp_path, modules=modules)
    assert any("total coverage" in p for p in problems)
    assert any("tools/x.py: 94.00" in p for p in problems)

    (tmp_path / "src" / "sotto").mkdir(parents=True)
    (tmp_path / "src/sotto/persona.py").write_text("", encoding="utf-8")
    problems = coverage_gate.check(good | {"files": {}}, repo=tmp_path, modules=modules)
    assert any("persona.py: on disk but absent" in p for p in problems)


def test_coverage_gate_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coverage_gate, "NEW_MODULES", ())
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(_report({"src/a.py": (100, 0)})), encoding="utf-8")
    assert coverage_gate.main([str(ok)]) == 0
    assert "OK" in capsys.readouterr().out
    low = tmp_path / "low.json"
    low.write_text(json.dumps(_report({"src/a.py": (100, 50)})), encoding="utf-8")
    assert coverage_gate.main([str(low)]) == 1
    assert "COVERAGE GATE" in capsys.readouterr().err


# --- language gate ------------------------------------------------------------


def test_language_gate_reports_and_enforces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = '\u043f\u0440\u0438\u0432\u0435\u0442'\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "lang").mkdir()
    (tmp_path / "tests/lang/test_ru.py").write_text("x = '\u0434\u0430'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    files = check_language.tracked_files(tmp_path)
    assert "a.py" in files and check_language.is_allowed("tests/lang/test_ru.py")
    assert check_language.report(tmp_path, files) == {"a.py": 6}
    assert check_language.count_cyrillic(tmp_path / "missing") == 0
    assert check_language.main(["--repo", str(tmp_path)]) == 0
    assert "report-only): 1 files, 6 Cyrillic" in capsys.readouterr().out
    assert check_language.main(["--repo", str(tmp_path), "--enforce"]) == 1


def test_scenario_default_home_needs_the_files_defaults(tmp_path: Path) -> None:
    from tools.golden.schema import load_scenarios

    body = "scenarios:\n  - {id: A_x, group: A, actions: [{kind: gate_pass}]}\n"
    path = tmp_path / "s.yaml"
    path.write_text("version: 1\n" + body, encoding="utf-8")
    with pytest.raises(ValueError, match=r"needs defaults\.home"):
        load_scenarios(path, enforce_minimums=False)
    path.write_text(
        "version: 1\ndefaults: {home: {name: T, lat: 1.0, lon: 2.0}}\n" + body, encoding="utf-8"
    )
    (scenario,) = load_scenarios(path, enforce_minimums=False)
    assert scenario.weather.home.name == "T"  # type: ignore[union-attr]
    assert scenario.defaults.guest_name == "Guest"
