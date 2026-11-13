"""Write the config keys that left the engine defaults into a pack's ``config.yaml``.

Plan 04, PR-4 and PR-5b.

PR-4 neutralised the persona-specific defaults of ``config_models.py`` (name, triggers, timezone,
the regex and word lists) and turned two filter constants into config keys
(``filters.districts``, ``filters.latin_extra_letters``). A pack that relied on a default rather
than spelling the key out would silently change behaviour. This tool reads the OLD engine (a
checkout from before PR-4, ``--old-src``), and for every moved key missing from the pack's
``config.yaml`` appends the old value to the right section, text-only (comments and layout of
the file are kept). ``--check`` then compares the full parsed config of the old engine with the
new engine's and lists the differing KEYS (never values: the real pack is private).

PR-5b neutralised more defaults (the persona's birth year, district and move year, the places
search anchor and keyword lists, the geocoder language and home country, the vocabulary request
and explain markers, the reaction palette). They sit in nested sections too, so a section is a
dotted path (``behaviour.vocab``). The old engine may be ANY earlier checkout: it is asked for its
default ``Config()`` and for the config of the pack as it parsed it (through the pack's config
layers when the old engine has them, else the bare ``config.yaml``).

Usage::

    python -m tools.extract_filter_defaults --pack DIR --old-src OLD_SRC [--check] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

# (section path, key) of every default the engine no longer carries.
MOVED_KEYS: tuple[tuple[str, str], ...] = (
    ("persona", "name"),
    ("persona", "display_name"),
    ("persona", "name_triggers"),
    ("persona", "timezone"),
    ("filters", "places_whitelist"),
    ("filters", "known_places"),
    ("filters", "polish_words"),
    ("filters", "topic_stop"),
    ("filters", "injection_markers"),
    ("filters", "logistics"),
    ("filters", "urgent"),
    ("filters", "places_request"),
    ("filters", "weather_request"),
    ("filters", "model_talk"),
    ("filters", "assistant_markers"),
    ("filters", "grumpy_markers"),
    ("filters", "motifs"),
    ("filters", "story_markers"),
    ("filters", "self_markers"),
    ("filters", "districts"),
    ("filters", "latin_extra_letters"),
    ("persona", "birth_date"),
    ("persona", "arrived_wolsan"),
    ("persona", "district"),
    ("behaviour.reactions", "emoji"),
    ("behaviour.vocab", "request_markers"),
    ("behaviour.vocab", "explain_markers"),
    ("behaviour.weather", "geocode_language"),
    ("behaviour.weather", "home_country_codes"),
    ("behaviour.weather", "home_country_names"),
    ("places", "queries"),
    ("places", "search_center"),
    ("places", "region_code"),
    ("places", "language_code"),
    ("places", "district_aliases"),
    ("places", "default_district"),
    ("places", "quiet_keywords"),
    ("places", "outskirts_keywords"),
)
# Keys that did not exist in the old Config: their old value is a constant of old filters.py.
NEW_KEYS = frozenset({("filters", "districts"), ("filters", "latin_extra_letters")})

_OLD_DUMP = """
import json, sys
from pathlib import Path
from sotto.config import load_config
from sotto.config_models import Config
from sotto import filters
mode, path = sys.argv[1], sys.argv[2]
if mode == "defaults":
    out = Config().model_dump(mode="json")
    if hasattr(filters, "_DISTRICTS"):
        out["filters"]["districts"] = list(filters._DISTRICTS)
    if hasattr(filters, "_LATIN_LETTERS"):
        out["filters"]["latin_extra_letters"] = filters._LATIN_LETTERS.replace("A-Za-z", "")
else:
    try:
        from sotto.persona import load_persona
    except ImportError:
        out = load_config(Path(path) / "config.yaml").model_dump(mode="json")
    else:
        out = load_config(load_persona(Path(path)).config_layers()).model_dump(mode="json")
json.dump(out, sys.stdout)
"""


class ExtractError(Exception):
    """Bad input: missing file, a section the pack does not have, an old engine that fails."""


def old_engine_dump(old_src: Path, mode: str, path: str = "") -> dict[str, Any]:
    """Run the old engine in a subprocess: ``defaults`` (Config() + the two constants) or the
    parsed config of the pack directory ``path``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(old_src.resolve())
    with tempfile.TemporaryDirectory(prefix="oldengine-") as tmp:
        proc = subprocess.run(
            [sys.executable, "-c", _OLD_DUMP, mode, str(Path(path).resolve()) if path else ""],
            env=env,
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise ExtractError(f"old engine failed: {proc.stderr.strip().splitlines()[-1:]}")
    loaded: dict[str, Any] = json.loads(proc.stdout)
    return loaded


def format_value(value: Any) -> str:
    """One YAML flow value on one line (strings as JSON, valid YAML)."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)  # safe_dump would add a "..." document end after a bare scalar
    return str(
        yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10**6)
    ).strip()


def _span_in(
    lines: list[str], name: str, start: int, stop: int, parent_indent: int
) -> tuple[int, int, str]:
    """Header of ``name`` among the lines ``start:stop`` (indented deeper than ``parent_indent``),
    the index after its last content line and the indent of its keys."""
    pattern = re.compile(rf"^(\s*){re.escape(name)}:\s*(#.*)?$")
    header = -1
    for i in range(start, stop):
        match = pattern.match(lines[i])
        # A top-level section sits at column 0; a nested one is deeper than its parent.
        if match and (
            len(match.group(1)) > parent_indent if parent_indent >= 0 else not match.group(1)
        ):
            header = i
            break
    if header < 0:
        raise ExtractError(f"section {name!r} not found in config.yaml")
    header_match = pattern.match(lines[header])
    assert header_match is not None
    own_indent = len(header_match.group(1))
    end, indent = header + 1, ""
    for i in range(header + 1, stop):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= own_indent:
            break
        if not indent:
            indent = line[: len(line) - len(line.lstrip())]
        end = i + 1
    return header, end, indent or " " * (own_indent + 2)


def _section_span(lines: list[str], section: str) -> tuple[int, int, str]:
    """(index of the header, index after the last content line, indent of the keys) of a section
    path such as ``places`` or ``behaviour.vocab``."""
    start, stop, parent_indent = 0, len(lines), -1
    header = end = 0
    indent = "  "
    for name in section.split("."):
        header, end, indent = _span_in(lines, name, start, stop, parent_indent)
        start, stop, parent_indent = header + 1, end, len(indent) - 2
    return header, end, indent


def insert_keys(text: str, additions: Mapping[str, Mapping[str, Any]]) -> str:
    """Append ``key: value`` lines at the end of each section; the rest of the text is untouched."""
    lines = text.split("\n")
    for section, values in additions.items():
        if not values:
            continue
        _, end, indent = _section_span(lines, section)
        new = [f"{indent}{key}: {format_value(value)}" for key, value in values.items()]
        lines[end:end] = new
    return "\n".join(lines)


def _node(data: Mapping[str, Any], section: str) -> Any:
    node: Any = data
    for name in section.split("."):
        node = node.get(name) if isinstance(node, dict) else None
    return node


def missing_keys(
    config: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Moved keys absent from the pack config, with their old default value."""
    found: dict[str, dict[str, Any]] = {}
    for section, key in MOVED_KEYS:
        present = _node(config, section)
        if isinstance(present, dict) and key in present:
            continue
        old = _node(defaults, section)
        if not isinstance(old, dict) or key not in old:
            continue  # the old engine never had this key: nothing to carry over
        found.setdefault(section, {})[key] = old[key]
    return found


def flat(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            out.update(flat(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: data}


def differing_keys(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Dotted keys whose parsed value differs (keys only; the new constants are ignored)."""
    ignored = {f"{section}.{key}" for section, key in NEW_KEYS}
    old_flat, new_flat = flat(old), flat(new)
    return sorted(
        key
        for key in old_flat.keys() | new_flat.keys()
        if key not in ignored and old_flat.get(key) != new_flat.get(key)
    )


def run(pack: Path, old_src: Path, *, check: bool, dry_run: bool) -> int:
    from sotto.config import load_config
    from sotto.persona import load_persona

    config_path = pack / "config.yaml"
    if not config_path.is_file():
        raise ExtractError(f"{config_path} not found")
    text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}
    additions = missing_keys(config, old_engine_dump(old_src, "defaults"))
    added = sorted(f"{section}.{key}" for section, keys in additions.items() for key in keys)
    print(f"keys to add: {len(added)}: {', '.join(added) or '-'}")
    if added and not dry_run:
        config_path.write_text(insert_keys(text, additions), encoding="utf-8")
    if not check:
        return 0
    old = old_engine_dump(old_src, "pack", str(pack))
    # Only the lists' effective values matter: after the write, the new engine reads them.
    new = load_config(load_persona(pack).config_layers()).model_dump(mode="json")
    if added and dry_run:
        print("check skipped: --dry-run did not write the keys")
        return 0
    diff = differing_keys(old, new)
    print(f"parsed config differs in {len(diff)} keys: {', '.join(diff) or '-'}")
    return 1 if diff else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--old-src", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.pack, args.old_src, check=args.check, dry_run=args.dry_run)
    except ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
