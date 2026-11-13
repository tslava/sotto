"""load_config: the demo pack's real config.yaml, overrides, errors, age."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from sotto.config import describe_key, flatten_config, load_config
from sotto.config_models import (
    BehaviourConfig,
    CallbackConfig,
    Config,
    FiltersConfig,
    JokesConfig,
    LlmConfig,
    PlacesConfig,
    ReplyDelayBucket,
)
from sotto.lang import default_config_layer
from tests.packs import DEMO_PACK

CONFIG_PATH = DEMO_PACK / "config.yaml"


def test_load_real_config_without_errors() -> None:
    cfg = load_config(CONFIG_PATH)

    assert isinstance(cfg, Config)
    assert cfg.persona.name == "Gus"
    assert cfg.persona.display_name == "Gus Marlow"
    assert cfg.behaviour.daily_cap == 3
    assert cfg.filters.shadow is True
    assert "Tesco" in cfg.filters.places_whitelist


def test_load_real_config_filters_pattern_lists() -> None:
    cfg = load_config(CONFIG_PATH)

    assert r"\bwar\b" in cfg.filters.topic_stop
    assert r"\bbrexit\b" in cfg.filters.topic_stop
    assert r"\b\d{1,2}[:.]\d{2}\b" in cfg.filters.logistics
    assert "certainly!" in cfg.filters.assistant_markers
    assert "The Lantern Room" in cfg.filters.known_places
    assert cfg.filters.polish_words == []


def test_load_real_config_motifs_and_story_defaults_match() -> None:
    # Engine defaults are neutral: the motif, story and window settings come from the pack.
    cfg = load_config(CONFIG_PATH)
    default_filters = FiltersConfig()

    assert cfg.filters.enforce_stages == default_filters.enforce_stages == ["dedup", "style"]
    assert cfg.filters.motif_recent_window == default_filters.motif_recent_window
    assert cfg.filters.motif_avoid_window == default_filters.motif_avoid_window
    assert cfg.filters.story_window == default_filters.story_window
    assert cfg.filters.story_max == default_filters.story_max

    assert "clocks" in cfg.filters.motifs
    assert r"\bthe shed\b" in cfg.filters.motifs["shed"]
    assert r"\bI remember\b" in cfg.filters.story_markers


def test_invalid_regex_in_motifs_raises_with_pattern_and_label() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FiltersConfig(motifs={"clocks": [_INVALID_REGEX]})

    assert _INVALID_REGEX in str(exc_info.value)
    assert "clocks" in str(exc_info.value)


def test_motifs_override_via_set_parses_yaml_mapping() -> None:
    """/set filters.motifs takes the same YAML as the regex lists."""
    cfg = load_config(CONFIG_PATH, {"filters.motifs": "{fishing: ['\\bfish']}"})

    assert cfg.filters.motifs == {"fishing": [r"\bfish"]}


def test_describe_key_dict_of_lists() -> None:
    cfg = load_config(CONFIG_PATH)

    info = describe_key(cfg, cfg, {}, "filters.motifs")

    assert info is not None
    assert info.type_name == "dict[str, list[str]]"
    assert "clocks" in info.value


def test_load_real_config_carries_the_places_queries() -> None:
    # The engine default of places.queries is empty: the pack's config.yaml lists them.
    cfg = load_config(CONFIG_PATH)

    assert PlacesConfig().queries == []
    assert "pub Kettle Hill Brindlemoor" in cfg.places.queries
    assert "restaurant Fenwick Green" in cfg.places.queries


def test_config_builds_with_defaults_without_yaml() -> None:
    cfg = Config()

    assert cfg.persona.birth_date == date(1970, 1, 1)
    assert cfg.behaviour.ambient_probability == 0.25


def test_override_changes_value_and_coerces_int() -> None:
    cfg = load_config(CONFIG_PATH, {"behaviour.daily_cap": "7"})

    assert cfg.behaviour.daily_cap == 7
    assert isinstance(cfg.behaviour.daily_cap, int)


def test_override_coerces_float() -> None:
    cfg = load_config(CONFIG_PATH, {"behaviour.ambient_probability": "0.42"})

    assert cfg.behaviour.ambient_probability == pytest.approx(0.42)


def test_override_bool_and_list() -> None:
    cfg = load_config(
        CONFIG_PATH,
        {
            "filters.shadow": "false",
            "persona.name_triggers": "[gus, old man]",
        },
    )

    assert cfg.filters.shadow is False
    assert cfg.persona.name_triggers == ["gus", "old man"]


def test_nested_override() -> None:
    cfg = load_config(CONFIG_PATH, {"behaviour.live_talk.min_people": "5"})

    assert cfg.behaviour.live_talk.min_people == 5


def test_invalid_override_value_raises_value_error_with_key() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_config(CONFIG_PATH, {"behaviour.daily_cap": "not-a-number"})

    assert "behaviour.daily_cap" in str(exc_info.value)


def test_invalid_override_out_of_range_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_config(CONFIG_PATH, {"behaviour.ambient_probability": "2.5"})

    assert "behaviour.ambient_probability" in str(exc_info.value)


def test_unknown_override_key_raises_value_error_with_key() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_config(CONFIG_PATH, {"behaviour.no_such_key": "1"})

    assert "behaviour.no_such_key" in str(exc_info.value)


def test_unknown_override_section_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_config(CONFIG_PATH, {"nope.daily_cap": "1"})

    assert "nope.daily_cap" in str(exc_info.value)


def test_reply_delay_bucket_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        BehaviourConfig(
            reply_delay_buckets=[
                ReplyDelayBucket(weight=0.5, range_sec=(30, 180)),
                ReplyDelayBucket(weight=0.2, range_sec=(180, 900)),
            ]
        )


def test_reply_delay_bucket_weights_within_tolerance_ok() -> None:
    cfg = BehaviourConfig(
        reply_delay_buckets=[
            ReplyDelayBucket(weight=0.61, range_sec=(30, 180)),
            ReplyDelayBucket(weight=0.3, range_sec=(180, 900)),
            ReplyDelayBucket(weight=0.09, range_sec=(900, 3600)),
        ]
    )
    assert len(cfg.reply_delay_buckets) == 3


def test_persona_age() -> None:
    cfg = load_config(CONFIG_PATH)

    assert cfg.persona.age(date(2026, 9, 10)) == 61
    assert cfg.persona.age(date(2026, 3, 8)) == 60
    assert cfg.persona.age(date(2026, 3, 9)) == 61


def test_flatten_config_roundtrip_keys() -> None:
    cfg = load_config(CONFIG_PATH)
    flat = flatten_config(cfg)

    assert flat["behaviour.daily_cap"] == "3"
    assert flat["persona.name"] == "Gus"
    assert flat["filters.shadow"] == "true"


# --- FiltersConfig: regex validation -------------------------------------------

_INVALID_REGEX = "\\bwar[a-z*"  # an unclosed bracket / character class


@pytest.mark.parametrize(
    "field_name",
    ["topic_stop", "injection_markers", "logistics", "urgent", "places_request", "model_talk"],
)
def test_invalid_regex_in_filters_field_raises_with_pattern_text(field_name: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FiltersConfig(**{field_name: [_INVALID_REGEX]})

    assert _INVALID_REGEX in str(exc_info.value)


def test_invalid_regex_in_real_config_via_override_raises() -> None:
    override_value = f"[{json.dumps(_INVALID_REGEX, ensure_ascii=False)}]"
    with pytest.raises(ValueError) as exc_info:
        load_config(CONFIG_PATH, {"filters.topic_stop": override_value})

    assert _INVALID_REGEX in str(exc_info.value)


# --- LlmConfig: main_model/judge_model are "" or "provider/model" -----------------


def test_llm_model_id_accepts_empty_string() -> None:
    cfg = LlmConfig(main_model="", judge_model="")
    assert cfg.main_model == ""
    assert cfg.judge_model == ""


def test_llm_model_id_accepts_provider_slash_model() -> None:
    cfg = LlmConfig(main_model="openai/gpt-4o-mini", judge_model="anthropic/claude-3.5-haiku")
    assert cfg.main_model == "openai/gpt-4o-mini"


@pytest.mark.parametrize("field_name", ["main_model", "judge_model"])
def test_llm_model_id_rejects_value_without_slash(field_name: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        LlmConfig(**{field_name: "garbage"})

    assert "garbage" in str(exc_info.value)


def test_set_llm_main_model_garbage_via_override_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_config(CONFIG_PATH, {"llm.main_model": "garbage"})

    assert "llm.main_model" in str(exc_info.value)


def test_assistant_markers_are_not_validated_as_regex() -> None:
    # assistant_markers are phrases, not regexes: a string with "special" regex characters
    # (say, an unclosed bracket) must not fail the validator.
    cfg = FiltersConfig(assistant_markers=["(this is an unclosed bracket"])
    assert cfg.assistant_markers == ["(this is an unclosed bracket"]


# --- describe_key ------------------------------------------------------------------


def test_describe_key_int_with_bounds() -> None:
    base = load_config(CONFIG_PATH)
    current = load_config(CONFIG_PATH, {"behaviour.daily_cap": "7"})

    info = describe_key(current, base, {"behaviour.daily_cap": "7"}, "behaviour.daily_cap")

    assert info is not None
    assert info.key == "behaviour.daily_cap"
    assert info.value == "7"
    assert info.default == "3"
    assert info.overridden is True
    assert info.type_name == "int"
    assert info.bounds == "0..50"
    assert info.description
    assert info.settable is True


def test_describe_key_bool_without_bounds() -> None:
    cfg = load_config(CONFIG_PATH)

    info = describe_key(cfg, cfg, {}, "filters.shadow")

    assert info is not None
    assert info.type_name == "bool"
    assert info.bounds == ""
    assert info.overridden is False
    assert info.value == "true"


def test_describe_key_list_of_str() -> None:
    cfg = load_config(CONFIG_PATH)

    info = describe_key(cfg, cfg, {}, "filters.topic_stop")

    assert info is not None
    assert info.type_name == "list[str]"
    assert info.bounds == ""


def test_describe_key_tuple() -> None:
    cfg = load_config(CONFIG_PATH)

    info = describe_key(cfg, cfg, {}, "behaviour.quiet_window")

    assert info is not None
    assert info.type_name == "tuple[str, str]"


def test_describe_key_list_of_models() -> None:
    cfg = load_config(CONFIG_PATH)

    info = describe_key(cfg, cfg, {}, "behaviour.reply_delay_buckets")

    assert info is not None
    assert info.type_name == "list[ReplyDelayBucket]"


def test_describe_key_persona_not_settable() -> None:
    cfg = load_config(CONFIG_PATH)

    info = describe_key(cfg, cfg, {}, "persona.name")

    assert info is not None
    assert info.settable is False


def test_describe_key_section_returns_none() -> None:
    cfg = load_config(CONFIG_PATH)

    assert describe_key(cfg, cfg, {}, "behaviour.live_talk") is None
    assert describe_key(cfg, cfg, {}, "behaviour") is None
    assert describe_key(cfg, cfg, {}, "") is None


def test_describe_key_unknown_key_returns_none() -> None:
    cfg = load_config(CONFIG_PATH)

    assert describe_key(cfg, cfg, {}, "behaviour.no_such_key") is None
    assert describe_key(cfg, cfg, {}, "nope.daily_cap") is None
    assert describe_key(cfg, cfg, {}, "behaviour.daily_cap.extra") is None


def test_describe_key_overridden_default_differs_from_value() -> None:
    base = load_config(CONFIG_PATH)
    current = load_config(CONFIG_PATH, {"filters.shadow": "false"})

    info = describe_key(current, base, {"filters.shadow": "false"}, "filters.shadow")

    assert info is not None
    assert info.overridden is True
    assert info.value == "false"
    assert info.default == "true"
    assert info.value != info.default


def test_jokes_and_callback_defaults_and_real_config() -> None:
    cfg = load_config(CONFIG_PATH)

    assert cfg.behaviour.jokes.enabled is True
    assert cfg.behaviour.jokes.per_period == 3
    assert cfg.behaviour.jokes.in_prompt == 6
    assert cfg.behaviour.jokes.cooldown_days == 5
    callback = cfg.behaviour.callback
    assert (callback.min_days, callback.max_age_days, callback.author_active_days) == (3, 14, 7)
    assert callback.window == ("11:00", "20:00")


def test_callback_window_validated_like_quiet_window() -> None:
    with pytest.raises(ValidationError):
        CallbackConfig(window=("25:00", "20:00"))
    with pytest.raises(ValidationError):
        CallbackConfig(window=("11-00", "20:00"))
    assert CallbackConfig(window=("09:30", "18:00")).window == ("09:30", "18:00")


def test_jokes_and_callback_bounds() -> None:
    with pytest.raises(ValidationError):
        JokesConfig(per_period=11)
    with pytest.raises(ValidationError):
        JokesConfig(in_prompt=-1)
    with pytest.raises(ValidationError):
        CallbackConfig(min_days=0)
    with pytest.raises(ValidationError):
        CallbackConfig(max_age_days=91)


def test_describe_key_all_leaf_keys_have_short_description() -> None:
    cfg = Config()
    flat = flatten_config(cfg)

    for key in flat:
        info = describe_key(cfg, cfg, {}, key)
        assert info is not None, key
        assert info.description, f"{key}: empty description"
        assert len(info.description) <= 90, f"{key}: description too long ({len(info.description)})"


def test_reactions_emoji_laugh_not_in_telegram_set_rejected() -> None:
    """😂 is in allowed_emoji but not in the Telegram reaction set (REACTION_INVALID)."""
    with pytest.raises(ValidationError, match=r"validation\.emoji_not_telegram"):
        Config.model_validate(
            {"behaviour": {"reactions": {"emoji": ["👍", "😂"]}}},
        )


def test_reactions_emoji_rofl_accepted_without_allowed_emoji() -> None:
    cfg = Config.model_validate({"behaviour": {"reactions": {"emoji": ["👍", "🤣"]}}})
    assert "🤣" in cfg.behaviour.reactions.emoji
    assert "🤣" not in cfg.filters.allowed_emoji


def test_reactions_emoji_variation_selector_is_stripped() -> None:
    """The keyboard sends "❤️" (with U+FE0F) while the Telegram set has "❤"."""
    cfg = Config.model_validate({"behaviour": {"reactions": {"emoji": ["👍", "❤\ufe0f"]}}})
    assert list(cfg.behaviour.reactions.emoji) == ["👍", "❤"]


# -- layers (PR-2) ------------------------------------------------------------------


def test_load_config_accepts_a_single_path_a_mapping_or_a_sequence(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("behaviour:\n  daily_cap: 4\n", encoding="utf-8")

    assert load_config(path).behaviour.daily_cap == 4
    assert load_config({"behaviour": {"daily_cap": 5}}).behaviour.daily_cap == 5
    assert load_config([path]).behaviour.daily_cap == 4
    assert load_config([]).behaviour.daily_cap == Config().behaviour.daily_cap


def test_layers_are_merged_deeply_and_the_top_layer_wins(tmp_path: Path) -> None:
    low = tmp_path / "low.yaml"
    low.write_text(
        "behaviour:\n  daily_cap: 4\n  mention_daily_cap: 9\npersona:\n  name: Low\n",
        encoding="utf-8",
    )
    high = {"behaviour": {"daily_cap": 6}}

    cfg = load_config([low, high])

    assert cfg.behaviour.daily_cap == 6
    assert cfg.behaviour.mention_daily_cap == 9
    assert cfg.persona.name == "Low"


def test_a_list_in_the_top_layer_replaces_the_lower_one(tmp_path: Path) -> None:
    low = {"filters": {"topic_stop": ["a", "b"]}}
    high = {"filters": {"topic_stop": ["c"]}}

    assert load_config([low, high]).filters.topic_stop == ["c"]


def test_layers_do_not_mutate_their_inputs() -> None:
    low = {"behaviour": {"daily_cap": 4}}
    high = {"behaviour": {"daily_cap": 6}}

    load_config([low, high], {"behaviour.daily_cap": "7"})

    assert low == {"behaviour": {"daily_cap": 4}}
    assert high == {"behaviour": {"daily_cap": 6}}


def test_overrides_apply_on_top_of_all_layers(tmp_path: Path) -> None:
    layers = [{"behaviour": {"daily_cap": 4}}, {"behaviour": {"daily_cap": 6}}]

    assert load_config(layers, {"behaviour.daily_cap": "2"}).behaviour.daily_cap == 2


def test_a_missing_layer_file_is_an_empty_layer(tmp_path: Path) -> None:
    cfg = load_config([tmp_path / "absent.yaml", {"behaviour": {"daily_cap": 3}}])

    assert cfg.behaviour.daily_cap == 3


def test_an_empty_layer_file_is_an_empty_layer(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_config([path]) == Config()


def test_a_layer_file_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config([path])


def test_the_demo_pack_config_is_identical_via_path_and_via_layers() -> None:
    assert load_config(CONFIG_PATH) == load_config([CONFIG_PATH])


# --- neutral engine defaults (PR-4) ------------------------------------------------


def test_bare_config_has_neutral_persona_and_empty_pack_lists() -> None:
    cfg = Config()

    assert cfg.persona.name == "Persona"
    assert cfg.persona.display_name == "Persona"
    assert cfg.persona.name_triggers == []
    assert cfg.persona.timezone == "UTC"
    for key in (
        "places_whitelist",
        "known_places",
        "polish_words",
        "topic_stop",
        "injection_markers",
        "logistics",
        "urgent",
        "places_request",
        "weather_request",
        "model_talk",
        "assistant_markers",
        "grumpy_markers",
        "story_markers",
        "self_markers",
        "districts",
    ):
        assert getattr(cfg.filters, key) == [], key
    assert cfg.filters.motifs == {}
    assert cfg.filters.latin_extra_letters == ""
    # tuning defaults and their validation ranges stay in the engine
    assert cfg.behaviour.daily_cap == BehaviourConfig().daily_cap
    assert cfg.filters.enforce_stages == ["dedup", "style"]


def test_persona_layers_are_language_defaults_then_the_pack_file() -> None:
    from sotto.persona import load_persona

    persona = load_persona(DEMO_PACK)
    layers = persona.config_layers()

    assert layers[0] == default_config_layer(persona.lang)
    assert layers[1] == CONFIG_PATH
    assert load_config(layers) == load_config(CONFIG_PATH)


def test_new_filter_keys_are_described_and_settable() -> None:
    cfg = load_config(CONFIG_PATH)
    card = describe_key(cfg, cfg, {}, "filters.latin_extra_letters")

    assert card is not None and card.settable and card.type_name == "str" and card.description
    assert describe_key(cfg, cfg, {}, "filters.districts") is not None


# --- the engine defaults are neutral ------------------------------------------------


def test_engine_defaults_carry_no_persona_specific_values() -> None:
    import json
    import re

    dumped = json.dumps(Config().model_dump(mode="json"), ensure_ascii=False)
    assert not re.search("[\u0400-\u04ff]", dumped)
    cfg = Config()
    assert cfg.persona.district == ""
    assert cfg.places.queries == []
    assert cfg.places.district_aliases == {}
    assert cfg.places.default_district == ""
    assert cfg.places.region_code == ""
    assert cfg.behaviour.vocab.request_markers == []
    assert cfg.behaviour.vocab.explain_markers == []
    assert cfg.behaviour.weather.home_country_codes == []
    assert cfg.behaviour.weather.home_country_names == []
    assert cfg.behaviour.weather.geocode_language == "en"
