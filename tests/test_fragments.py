"""fragments.py: validation errors, accessors and loaders."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from sotto.fragments import (
    FRAGMENT_LISTS,
    FRAGMENT_MAPS,
    FRAGMENT_SPEC,
    Fragments,
    fragment_errors,
    load_fragments,
    load_valid_fragments,
)
from tests.packs import DEMO_PACK


def _data() -> dict[str, Any]:
    return copy.deepcopy(dict(load_valid_fragments(DEMO_PACK / "fragments.yaml").data))


def _set(data: dict[str, Any], key: str, value: Any) -> None:
    group, leaf = key.split(".", 1)
    data[group][leaf] = value


def test_the_demo_fragments_are_valid() -> None:
    assert fragment_errors(_data()) == []


def test_missing_and_unknown_keys_are_reported() -> None:
    data = _data()
    group, leaf = next(iter(FRAGMENT_SPEC)).split(".", 1)
    del data[group][leaf]
    data[group]["surprise"] = "x"
    errors = fragment_errors(data)
    assert f"fragments.yaml: missing key {group}.{leaf}" in errors
    assert f"fragments.yaml: unknown key {group}.surprise" in errors


def test_a_non_string_leaf_and_wrong_placeholders_are_reported() -> None:
    data = _data()
    keys = [k for k, v in FRAGMENT_SPEC.items() if v]
    _set(data, keys[0], ["not", "a string"])
    _set(data, keys[1], "no placeholders here")
    errors = fragment_errors(data)
    assert f"fragments.yaml: {keys[0]} must be a string" in errors
    assert any(e.startswith(f"fragments.yaml: {keys[1]} placeholders are [none]") for e in errors)


def test_a_stray_brace_is_reported() -> None:
    data = _data()
    key = next(k for k, v in FRAGMENT_SPEC.items() if not v)
    group, leaf = key.split(".", 1)
    data[group][leaf] = data[group][leaf] + " }"
    assert f"fragments.yaml: {key} contains a stray brace" in fragment_errors(data)


def test_lists_and_maps_are_validated() -> None:
    data = _data()
    list_key = next(iter(FRAGMENT_LISTS))
    _set(data, list_key, ["only one"])
    map_keys = sorted(FRAGMENT_MAPS)
    _set(data, map_keys[0], {})
    if len(map_keys) > 1:
        _set(data, map_keys[1], {"a": 1})
    errors = fragment_errors(data)
    assert any(f"{list_key} must be a list of" in e for e in errors)
    assert f"fragments.yaml: {map_keys[0]} must be a non-empty mapping" in errors
    if len(map_keys) > 1:
        assert f"fragments.yaml: {map_keys[1]} values must be strings" in errors


def test_accessors_reject_the_wrong_kind_and_missing_keys() -> None:
    frag = Fragments({"a": {"s": "hi {x}", "l": ["p", "q"], "m": {"k": "v"}}})
    assert frag.has("a.s") and not frag.has("a.zz") and not frag.has("a.s.deeper")
    assert frag.text("a.s") == "hi {x}"
    assert frag.items("a.l") == ["p", "q"]
    assert frag.table("a.m") == {"k": "v"}
    assert frag.prefix("a.s") == "hi "
    assert frag.render("a.s", x="there") == "hi there"
    with pytest.raises(KeyError):
        frag.get("a.zz")
    with pytest.raises(TypeError):
        frag.text("a.l")
    with pytest.raises(TypeError):
        frag.items("a.s")
    with pytest.raises(TypeError):
        frag.table("a.s")


def test_loaders(tmp_path: Path) -> None:
    assert load_fragments(tmp_path / "missing.yaml").data == {}
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_fragments(empty).data == {}
    listing = tmp_path / "list.yaml"
    listing.write_text(yaml.safe_dump(["x"]), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at top level"):
        load_fragments(listing)
    with pytest.raises(ValueError, match="not found"):
        load_valid_fragments(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="missing key"):
        load_valid_fragments(empty)
