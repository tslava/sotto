"""The English demo pack (``personas/demo``): packcheck, boot, golden baseline, few-shot filter
regression, and the guards that keep it free of the private persona's material."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tools.golden import runner
from tools.golden.schema import load_scenarios

from sotto import commands
from sotto.config import load_config
from sotto.few_shot import load_few_shot
from sotto.filters import FilterContext, check_output
from sotto.packcheck import boot_check, check_pack, locale_parity_errors
from sotto.persona import Persona
from sotto.replay import _parse_args, run_replay
from sotto.vocab import parse_vocab
from tests.packs import DEMO_PACK

GOLDEN = DEMO_PACK / "golden"
SCENARIOS = GOLDEN / "scenarios.yaml"
BASELINE = GOLDEN / "baseline"
EXPORT = GOLDEN / "export.json"

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Words of the private persona and its city that must never appear in the public demo pack.
_PRIVATE_WORDS = re.compile(
    r"wolsa|wolsań|dolna|fyodor|fedor|fedya|father fy|lidl|\bpolish\b|\bpoland\b", re.IGNORECASE
)
_EM_DASHES = ("—", "–")


def _pack_text_files() -> list[Path]:
    skip = {"baseline"}
    return [
        path
        for path in sorted(DEMO_PACK.rglob("*"))
        if path.is_file() and not (set(path.relative_to(DEMO_PACK).parts) & skip)
    ]


def test_pack_checks_clean(demo_pack: Persona) -> None:
    assert demo_pack.manifest.language == "en"
    assert demo_pack.name == "demo"
    assert check_pack(demo_pack) == []
    assert locale_parity_errors(demo_pack) == []


async def test_pack_boots(demo_pack: Persona) -> None:
    assert await boot_check(demo_pack) == []


def test_pack_has_no_cyrillic_and_no_private_words() -> None:
    for path in _pack_text_files():
        if path.suffix in {".json", ".png"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not _CYRILLIC.search(text), f"{path.name}: Cyrillic text"
        # ``arrived_wolsan`` is the engine's own (legacy-named) persona field: the key stays.
        text = text.replace("arrived_wolsan", "arrived_year")
        assert not _PRIVATE_WORDS.search(text), f"{path.name}: private persona wording"


def test_prompts_and_examples_have_no_dashes() -> None:
    # The persona writes without long dashes, and models copy dashes from the prompt itself.
    for path in [*(DEMO_PACK / "prompts").glob("*.txt"), DEMO_PACK / "few_shot.yaml"]:
        assert not any(dash in path.read_text(encoding="utf-8") for dash in _EM_DASHES), path.name


def test_ship_list_covers_every_file(demo_pack: Persona) -> None:
    shipped = {entry.rstrip("/") for entry in demo_pack.manifest.ship}
    top_level = {
        path.name for path in DEMO_PACK.iterdir() if path.name not in {"golden", ".revision"}
    }
    assert top_level <= shipped, sorted(top_level - shipped)


def test_vocab_example_parses(demo_pack: Persona) -> None:
    vocab = parse_vocab(
        (DEMO_PACK / "vocab.example.yaml").read_text(encoding="utf-8"), demo_pack.lang
    )
    assert vocab.name == "brindlemoor"
    assert len(vocab.words) >= 3


def test_stickers_are_synthetic(demo_pack: Persona) -> None:
    from sotto.stickers import load_catalog

    catalog = load_catalog(demo_pack.stickers_path)
    assert 3 <= len(catalog.stickers) <= 6
    assert all(sticker.file_id.startswith("CAACAgIAAxkBAAEDEMO") for sticker in catalog.stickers)


def test_persona_is_not_the_private_one(demo_pack: Persona) -> None:
    cfg = load_config(demo_pack.config_layers())
    assert cfg.persona.name == "Gus"
    assert cfg.persona.name_triggers == ["gus", "marlow"]
    assert cfg.persona.timezone == "Europe/London"


# -- golden --------------------------------------------------------------------------------


def test_golden_scenarios_cover_every_group() -> None:
    scenarios = load_scenarios(SCENARIOS)  # enforces the per-group minimums
    assert len(scenarios) == 101
    assert len(list(BASELINE.glob("*.json"))) == 101


def test_golden_baseline_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    # /status prints the package version; compare against the version the baseline was
    # recorded with, so a release bump does not read as a behaviour change.
    status = json.loads((BASELINE / "E_status.json").read_text(encoding="utf-8"))
    match = re.search(r"sotto v(\d+\.\d+\.\d+)", json.dumps(status))
    assert match is not None
    monkeypatch.setattr(commands, "__version__", match.group(1))

    outputs = runner.run_scenarios(DEMO_PACK, load_scenarios(SCENARIOS))
    result = runner.diff_outputs(outputs, BASELINE, allowed=set(), check_extra=True)
    assert result.ok, (sorted(result.diffs), result.missing, result.extra)


# -- few-shot regression ---------------------------------------------------------------------


def test_few_shot_shape() -> None:
    items = load_few_shot(DEMO_PACK / "few_shot.yaml")
    assert len(items) == 21
    assert 5 <= sum(1 for item in items if not item.speak) <= 7


async def test_all_few_shot_speak_true_examples_pass_check_output(demo_pack: Persona) -> None:
    items = load_few_shot(demo_pack.few_shot_path)
    cfg = load_config(demo_pack.config_layers())
    names = [item.name for item in items]
    bot_names = [cfg.persona.name, cfg.persona.display_name, *cfg.persona.name_triggers]

    failures: list[str] = []
    for item in (i for i in items if i.speak):
        ctx = FilterContext(
            lang=demo_pack.lang,
            cfg=cfg,
            recent_replies=[],
            context_rows=[],
            places_names=[],
            participant_names=names,
            bot_names=bot_names,
            muted_names=[],
            system_prompt=demo_pack.prompts["system"],
            trigger_text=item.user,
            now=0,
            frag=demo_pack.frag,
        )
        verdict = await check_output(item.text, ctx)
        if not verdict.ok:
            failures.append(f"{item.name}: {item.text!r} -> {verdict.reasons}")
    assert not failures, "demo few-shot cut by the output filter:\n" + "\n".join(failures)


# -- replay ------------------------------------------------------------------------------------


def test_replay_runs_on_the_demo_export() -> None:
    args = _parse_args(
        [
            str(EXPORT),
            "--persona",
            str(DEMO_PACK),
            "--bot-username",
            "gus_demo_bot",
            "--bot-user-id",
            "900001",
        ]
    )
    report = run_replay(args)
    assert "pass=" in report
    assert "mention" in report
