"""Paths of the persona packs that tests read directly."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_PACK = REPO_ROOT / "personas" / "_legacy_ru"
DEMO_PACK = REPO_ROOT / "personas" / "demo"
