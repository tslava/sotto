"""tools/extract_filter_defaults.py: text-level insertion of the keys that left the engine."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools import extract_filter_defaults as tool

from sotto.config import load_config
from sotto.persona import load_persona
from tests.packs import LEGACY_PACK

SAMPLE = textwrap.dedent(
    """\
    persona:
      name: X   # keep this comment
    behaviour:
      daily_cap: 3

    # the filters section
    filters:
      shadow: true
      # a comment inside
      urgent: [now]

    places:
      min_rating: 4
    """
)


def test_format_value_strings_as_json_and_lists_as_flow() -> None:
    assert tool.format_value('a"b ł') == '"a\\"b ł"'
    assert tool.format_value(["a", "b c"]) == "[a, b c]"
    assert tool.format_value({"k": ["x"]}) == "{k: [x]}"
    assert yaml.safe_load(tool.format_value("tricky: #value")) == "tricky: #value"


def test_insert_keys_appends_at_the_end_of_the_section_and_keeps_the_rest() -> None:
    out = tool.insert_keys(SAMPLE, {"filters": {"districts": ["A", "B"], "letters": "ąę"}})

    lines = out.split("\n")
    assert lines.index("  districts: [A, B]") == lines.index("  urgent: [now]") + 1
    assert lines[lines.index("  districts: [A, B]") + 1] == '  letters: "ąę"'
    assert "  name: X   # keep this comment" in lines
    assert yaml.safe_load(out)["filters"]["letters"] == "ąę"
    assert yaml.safe_load(out)["places"] == {"min_rating": 4}


def test_insert_keys_without_additions_changes_nothing() -> None:
    assert tool.insert_keys(SAMPLE, {}) == SAMPLE
    assert tool.insert_keys(SAMPLE, {"filters": {}}) == SAMPLE


def test_insert_keys_into_the_last_section_and_an_empty_one() -> None:
    last = tool.insert_keys(SAMPLE, {"places": {"k": 1}})
    assert last.rstrip("\n").endswith("  k: 1") or "  k: 1" in last
    assert yaml.safe_load(last)["places"]["k"] == 1
    empty = tool.insert_keys("filters:\npersona:\n  a: 1\n", {"filters": {"k": [1]}})
    assert yaml.safe_load(empty)["filters"] == {"k": [1]}


def test_insert_keys_needs_the_section() -> None:
    with pytest.raises(tool.ExtractError, match="'nope' not found"):
        tool.insert_keys(SAMPLE, {"nope": {"k": 1}})


def test_missing_keys_lists_only_absent_moved_keys() -> None:
    defaults = {
        section: {key: f"old-{section}-{key}" for s, key in tool.MOVED_KEYS if s == section}
        for section in ("persona", "filters")
    }
    config = {"persona": {"name": "X", "timezone": "Z"}, "filters": {"urgent": []}}

    found = tool.missing_keys(config, defaults)

    assert "name" not in found["persona"] and "timezone" not in found["persona"]
    assert found["persona"]["display_name"] == "old-persona-display_name"
    assert "urgent" not in found["filters"]
    assert found["filters"]["districts"] == "old-filters-districts"
    assert tool.missing_keys({}, defaults)["persona"]["name"] == "old-persona-name"


def test_flat_and_differing_keys_ignore_the_new_constants() -> None:
    assert tool.flat({"a": {"b": 1, "c": [2]}, "d": 3}) == {"a.b": 1, "a.c": [2], "d": 3}
    old = {"a": {"b": 1, "c": 2}, "filters": {"districts": [1]}}
    new = {"a": {"b": 1, "c": 3, "x": 0}, "filters": {"districts": [2]}}
    assert tool.differing_keys(old, new) == ["a.c", "a.x"]
    assert tool.differing_keys(old, old) == []


FAKE_OLD_ENGINE = {
    "sotto/__init__.py": "",
    "sotto/config_models.py": textwrap.dedent(
        """\
        class Config:
            def model_dump(self, mode="json"):
                keys = ["name", "display_name", "name_triggers", "timezone"]
                filters = [
                    "places_whitelist", "known_places", "polish_words", "topic_stop",
                    "injection_markers", "logistics", "urgent", "places_request",
                    "weather_request", "model_talk", "assistant_markers", "grumpy_markers",
                    "story_markers", "self_markers",
                ]
                return {
                    "persona": {k: "old-" + k for k in keys},
                    "filters": {**{k: ["old-" + k] for k in filters}, "motifs": {"m": ["x"]}},
                }
        """
    ),
    "sotto/config.py": textwrap.dedent(
        """\
        from sotto.config_models import Config

        class _Loaded:
            def __init__(self, data):
                self.data = data

            def model_dump(self, mode="json"):
                import json
                return json.loads(json.dumps(self.data, default=str))

        def load_config(path):
            import yaml
            return _Loaded(yaml.safe_load(path.read_text(encoding="utf-8")))
        """
    ),
    "sotto/filters.py": '_DISTRICTS = ("North", "South")\n_LATIN_LETTERS = "A-Za-zÓó"\n',
}


@pytest.fixture
def fake_old_src(tmp_path: Path) -> Path:
    root = tmp_path / "old-src"
    for name, body in FAKE_OLD_ENGINE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def _pack_without_new_keys(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    config = yaml.safe_load((pack / "config.yaml").read_text(encoding="utf-8"))
    del config["filters"]["districts"], config["filters"]["latin_extra_letters"]
    text = (pack / "config.yaml").read_text(encoding="utf-8").split("\n")
    kept = [
        line for line in text if not line.startswith(("  districts:", "  latin_extra_letters:"))
    ]
    (pack / "config.yaml").write_text("\n".join(kept), encoding="utf-8")
    return pack


def test_old_engine_dump_runs_the_old_sources_in_a_subprocess(fake_old_src: Path) -> None:
    defaults = tool.old_engine_dump(fake_old_src, "defaults")

    assert defaults["persona"]["name"] == "old-name"
    assert defaults["filters"]["districts"] == ["North", "South"]
    assert defaults["filters"]["latin_extra_letters"] == "Óó"
    config = LEGACY_PACK / "config.yaml"
    assert "filters" in tool.old_engine_dump(fake_old_src, "pack", str(config))


def test_old_engine_dump_reports_a_failing_old_engine(tmp_path: Path) -> None:
    with pytest.raises(tool.ExtractError, match="old engine failed"):
        tool.old_engine_dump(tmp_path, "defaults")


def test_run_writes_the_missing_keys(
    tmp_path: Path, fake_old_src: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = _pack_without_new_keys(tmp_path)

    assert tool.run(pack, fake_old_src, check=False, dry_run=True) == 0
    assert "districts" not in (pack / "config.yaml").read_text(encoding="utf-8")

    assert tool.run(pack, fake_old_src, check=False, dry_run=False) == 0
    written = yaml.safe_load((pack / "config.yaml").read_text(encoding="utf-8"))["filters"]
    assert written["districts"] == ["North", "South"]
    assert written["latin_extra_letters"] == "Óó"
    assert "keys to add: 2" in capsys.readouterr().out

    assert tool.run(pack, fake_old_src, check=False, dry_run=False) == 0
    assert "keys to add: 0" in capsys.readouterr().out


def test_run_check_compares_the_parsed_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = tmp_path / "pack"
    shutil.copytree(LEGACY_PACK, pack)
    new = load_config(load_persona(pack).config_layers()).model_dump(mode="json")

    def fake_dump(old_src: Path, mode: str, path: str = "") -> dict[str, Any]:
        return {"persona": {}, "filters": {}} if mode == "defaults" else new

    monkeypatch.setattr(tool, "old_engine_dump", fake_dump)
    # every moved key is present in the pack, so "defaults" is never consulted for a value
    assert tool.run(pack, tmp_path, check=True, dry_run=False) == 0
    assert "differs in 0 keys" in capsys.readouterr().out

    changed = {**new, "persona": {**new["persona"], "timezone": "Mars/Base"}}
    monkeypatch.setattr(
        tool,
        "old_engine_dump",
        lambda old_src, mode, path="": (
            {"persona": {}, "filters": {}} if mode == "defaults" else changed
        ),
    )
    assert tool.run(pack, tmp_path, check=True, dry_run=False) == 1
    assert "persona.timezone" in capsys.readouterr().out


def test_run_check_is_skipped_on_a_dry_run_that_would_write(
    tmp_path: Path, fake_old_src: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = _pack_without_new_keys(tmp_path)

    assert tool.run(pack, fake_old_src, check=True, dry_run=True) == 0
    assert "check skipped" in capsys.readouterr().out


def test_run_needs_a_config(tmp_path: Path) -> None:
    with pytest.raises(tool.ExtractError, match="not found"):
        tool.run(tmp_path, tmp_path, check=False, dry_run=False)


def test_main_reports_errors_and_succeeds(
    tmp_path: Path, fake_old_src: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tool.main(["--pack", str(tmp_path), "--old-src", str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err
    pack = _pack_without_new_keys(tmp_path)
    assert tool.main(["--pack", str(pack), "--old-src", str(fake_old_src), "--dry-run"]) == 0
