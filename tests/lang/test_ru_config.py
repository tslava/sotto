"""The synthetic Russian pack's config and the Russian language layer under it."""

from pathlib import Path

from sotto.config import load_config
from sotto.config_models import Config, PlacesConfig
from sotto.lang import default_config_layer
from sotto.persona import load_persona
from tests.packs import RU, RU_PACK

RU_CONFIG_PATH = RU_PACK / "config.yaml"


def test_load_ru_pack_config_without_errors() -> None:
    cfg = load_config(RU_CONFIG_PATH)

    assert isinstance(cfg, Config)
    assert cfg.persona.name == "Борис"
    assert cfg.persona.display_name == "Борис Седов"
    assert cfg.behaviour.daily_cap == 3
    assert cfg.filters.shadow is True
    assert "Magnit" in cfg.filters.places_whitelist


def test_load_ru_pack_filters_pattern_lists() -> None:
    cfg = load_config(load_persona(RU_PACK).config_layers())

    # The generic marker lists live in the language module; the pack inherits them.
    for key, expected in RU.default_filters().items():
        assert getattr(cfg.filters, key) == expected, key

    assert r"\bсектор газа" in cfg.filters.topic_stop
    assert r"\bпис\b" in cfg.filters.topic_stop
    assert r"\b\d{1,2}[:.]\d{2}\b" in cfg.filters.logistics
    assert "конечно!" in cfg.filters.assistant_markers
    assert "ZORKA" in cfg.filters.known_places
    assert "Zarya Bar" in cfg.filters.known_places
    assert "działka" in cfg.filters.polish_words
    assert "urząd" in cfg.filters.polish_words


def test_ru_pack_motifs_and_story_markers() -> None:
    cfg = load_config(RU_CONFIG_PATH)

    assert "жена" in cfg.filters.motifs
    assert r"\bгараж" in cfg.filters.motifs["гараж"]
    assert r"\bкак-то раз\b" in cfg.filters.story_markers


def test_load_ru_pack_carries_the_places_queries() -> None:
    cfg = load_config(RU_CONFIG_PATH)

    assert PlacesConfig().queries == []
    assert "pub Zarechye Tikhorechensk" in cfg.places.queries
    assert "craft beer pub Tikhorechensk" in cfg.places.queries


def test_language_layer_sits_below_the_pack_config(tmp_path: Path) -> None:
    layer = default_config_layer(RU)
    pack = tmp_path / "config.yaml"
    pack.write_text("filters:\n  model_talk: [only-this]\n", encoding="utf-8")

    cfg = load_config([layer, pack])

    # a key the pack defines replaces the language's list (no merge) ...
    assert cfg.filters.model_talk == ["only-this"]
    # ... and a key it omits falls back to the language
    assert cfg.filters.injection_markers == RU.default_filters()["injection_markers"]
    assert cfg.filters.topic_stop == []
