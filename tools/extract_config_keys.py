"""Generate or check a ``config_keys.yaml`` catalog from the ``Field(description=...)`` texts.

Plan 04, PR-5b. ``/get <key>`` shows the description of a config key. The texts live in the
pydantic models of ``config_models.py`` (English); a locale carries them in the owner-UI catalog
file ``config_keys.yaml``: ``config_keys: {"behaviour.daily_cap": "...", ...}``, one entry per
leaf key (sections need none). Braces in a description are doubled (``{{diary}}``) because
catalog values are ``str.format`` templates.

* ``write PATH``: write the catalog from the descriptions the engine carries RIGHT NOW. This is how
  the Russian file of a pack was produced (run once on the engine before its descriptions were
  translated, so the owner-facing text stays byte for byte) and how ``locales/en`` is produced.
* ``check PATH``: verify that PATH equals the engine's descriptions (the English file).

Usage::

    python -m tools.extract_config_keys write locales/en/config_keys.yaml
    python -m tools.extract_config_keys check locales/en/config_keys.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel

from sotto.config_models import Config

STEM = "config_keys"


def leaf_descriptions(model: type[BaseModel] = Config, prefix: str = "") -> dict[str, str]:
    """Dotted leaf key -> ``Field.description`` ("" when a field has none), in model order."""
    out: dict[str, str] = {}
    for name, info in model.model_fields.items():
        key = f"{prefix}{name}"
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out.update(leaf_descriptions(annotation, f"{key}."))
        else:
            out[key] = info.description or ""
    return out


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def render(descriptions: dict[str, str]) -> str:
    entries = {key: escape_braces(text) for key, text in descriptions.items()}
    body = yaml.safe_dump(
        {STEM: entries}, allow_unicode=True, width=10_000, sort_keys=False, default_style=None
    )
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    descriptions = leaf_descriptions()
    empty = [key for key, text in descriptions.items() if not text]
    if empty:
        print(f"keys without a description: {', '.join(empty)}", file=sys.stderr)
        return 1
    expected = render(descriptions)
    if args.mode == "write":
        args.path.parent.mkdir(parents=True, exist_ok=True)
        args.path.write_text(expected, encoding="utf-8")
        print(f"wrote {len(descriptions)} keys to {args.path}")
        return 0
    if not args.path.is_file() or args.path.read_text(encoding="utf-8") != expected:
        print(f"{args.path} differs from the engine's descriptions", file=sys.stderr)
        return 1
    print(f"{args.path}: {len(descriptions)} keys match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
