"""Process settings, read from .env / environment variables."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fields replaced by the persona pack (PERSONA_DIR). Order as before in Settings.
DEPRECATED_PATH_FIELDS: tuple[str, ...] = (
    "config_path",
    "few_shot_path",
    "prompt_path",
    "judge_prompt_path",
    "sticker_prompt_path",
    "followup_prompt_path",
    "checkin_prefilter_prompt_path",
    "name_check_prompt_path",
    "reaction_prompt_path",
    "memory_prompt_path",
    "diary_prompt_path",
    "jokes_prompt_path",
    "threads_prompt_path",
    "vision_prompt_path",
    "weather_place_prompt_path",
    "stickers_path",
    "changelog_path",
)


class Settings(BaseSettings):
    """Secrets and paths. No env_prefix: variable names match the field names."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    bot_token: SecretStr
    admin_user_id: int = 0
    allowed_chat_id: int = 0
    openrouter_api_key: SecretStr | None = None
    google_places_key: SecretStr | None = None
    db_path: Path = Path("data/bot.db")
    # Persona pack directory: config, prompts, few-shot, stickers, places. If unset and
    # allow_demo_persona=false the process refuses to start: a lost mount in production
    # must not silently launch the demo persona.
    persona_dir: Path | None = None
    allow_demo_persona: bool = False
    # Weather home point. Coordinates are personal data of the owner and the repository
    # is public: they are not in config.yaml or in code, only here in .env. If both are
    # not set the weather block is empty (WARNING at startup); weather for a place named
    # in a question still works.
    weather_latitude: float | None = None
    weather_longitude: float | None = None
    weather_home_name: str = ""
    # Deprecated persona file paths (before PR-2 each file had its own variable).
    # Kept for one release so an old .env does not break loading: the value is ignored
    # and one WARNING is logged at startup (deprecated_fields_set). Removed after publication.
    config_path: Path | None = None
    few_shot_path: Path | None = None
    prompt_path: Path | None = None
    judge_prompt_path: Path | None = None
    sticker_prompt_path: Path | None = None
    followup_prompt_path: Path | None = None
    checkin_prefilter_prompt_path: Path | None = None
    name_check_prompt_path: Path | None = None
    reaction_prompt_path: Path | None = None
    memory_prompt_path: Path | None = None
    diary_prompt_path: Path | None = None
    jokes_prompt_path: Path | None = None
    threads_prompt_path: Path | None = None
    vision_prompt_path: Path | None = None
    weather_place_prompt_path: Path | None = None
    stickers_path: Path | None = None
    changelog_path: Path | None = None
    log_level: str = "INFO"

    def deprecated_fields_set(self) -> list[str]:
        """Names of deprecated path variables that are set in the environment/.env."""
        return [name.upper() for name in DEPRECATED_PATH_FIELDS if getattr(self, name) is not None]


class PersonaSelector(BaseSettings):
    """Only the persona pack choice: for CLIs that do not need BOT_TOKEN (replay, places_fill)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    persona_dir: Path | None = None
    allow_demo_persona: bool = False
