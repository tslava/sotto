"""Settings is read from the environment; a .env file is optional."""

from pathlib import Path

import pytest

from sotto.settings import Settings


@pytest.fixture(autouse=True)
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Make sure there is no .env nearby: settings must build from a clean environment.
    monkeypatch.chdir(tmp_path)


def test_settings_from_env_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_TOKEN", "test-token-123")
    monkeypatch.delenv("PERSONA_DIR")  # the autouse conftest fixture sets it

    settings = Settings()

    assert settings.bot_token.get_secret_value() == "test-token-123"
    assert settings.admin_user_id == 0
    assert settings.allowed_chat_id == 0
    assert settings.openrouter_api_key is None
    assert settings.google_places_key is None
    assert settings.db_path == Path("data/bot.db")
    assert settings.allow_demo_persona is False
    assert settings.deprecated_fields_set() == []
    assert settings.log_level == "INFO"


def test_settings_from_env_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("ADMIN_USER_ID", "42")
    monkeypatch.setenv("ALLOWED_CHAT_ID", "-1001234567890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    monkeypatch.setenv("GOOGLE_PLACES_KEY", "gp-1")
    monkeypatch.setenv("DB_PATH", "/tmp/custom.db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.admin_user_id == 42
    assert settings.allowed_chat_id == -1001234567890
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-1"
    assert settings.google_places_key is not None
    assert settings.google_places_key.get_secret_value() == "gp-1"
    assert settings.db_path == Path("/tmp/custom.db")
    assert settings.log_level == "DEBUG"


def test_settings_empty_env_values_like_env_example(monkeypatch: pytest.MonkeyPatch) -> None:
    # .env.example holds every key with an empty value (KEY=). An empty string must not break
    # int fields or turn into SecretStr(""): it must be read as "not set".
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("ADMIN_USER_ID", "")
    monkeypatch.setenv("ALLOWED_CHAT_ID", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("GOOGLE_PLACES_KEY", "")
    monkeypatch.setenv("DB_PATH", "")
    monkeypatch.setenv("LOG_LEVEL", "")

    settings = Settings()

    assert settings.admin_user_id == 0
    assert settings.allowed_chat_id == 0
    assert settings.openrouter_api_key is None
    assert settings.google_places_key is None
    assert settings.db_path == Path("data/bot.db")
    assert settings.log_level == "INFO"


def test_settings_weather_home_point_defaults_to_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """There are no home-point coordinates in the repository: by default there are none."""
    monkeypatch.setenv("BOT_TOKEN", "abc")

    settings = Settings()

    assert settings.weather_latitude is None
    assert settings.weather_longitude is None
    assert settings.weather_home_name == ""


def test_settings_weather_home_point_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("WEATHER_LATITUDE", "50.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "10.0")
    monkeypatch.setenv("WEATHER_HOME_NAME", "Town")

    settings = Settings()

    assert settings.weather_latitude == 50.0
    assert settings.weather_longitude == 10.0
    assert settings.weather_home_name == "Town"


def test_settings_empty_weather_env_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """.env.example keeps the keys empty: an empty string must not break float fields."""
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("WEATHER_LATITUDE", "")
    monkeypatch.setenv("WEATHER_LONGITUDE", "")
    monkeypatch.setenv("WEATHER_HOME_NAME", "")

    settings = Settings()

    assert settings.weather_latitude is None
    assert settings.weather_longitude is None
    assert settings.weather_home_name == ""


def test_settings_missing_bot_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_TOKEN", raising=False)

    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError, a required field
        Settings()


def test_settings_persona_dir_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("PERSONA_DIR", "/srv/persona")
    monkeypatch.setenv("ALLOW_DEMO_PERSONA", "true")

    settings = Settings()

    assert settings.persona_dir == Path("/srv/persona")
    assert settings.allow_demo_persona is True


def test_settings_deprecated_path_fields_are_reported_and_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("CHANGELOG_PATH", "/tmp/CHANGELOG.md")
    monkeypatch.setenv("JUDGE_PROMPT_PATH", "/tmp/judge.txt")
    monkeypatch.setenv("CONFIG_PATH", "/tmp/config.yaml")

    settings = Settings()

    assert settings.deprecated_fields_set() == [
        "CONFIG_PATH",
        "JUDGE_PROMPT_PATH",
        "CHANGELOG_PATH",
    ]


def test_settings_empty_deprecated_path_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_TOKEN", "abc")
    monkeypatch.setenv("PROMPT_PATH", "")

    assert Settings().deprecated_fields_set() == []
