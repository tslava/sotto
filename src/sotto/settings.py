"""Настройки процесса, читаются из .env / переменных окружения."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Поля, заменённые пакетом персоны (PERSONA_DIR). Порядок — как раньше в Settings.
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
    """Секреты и пути. env_prefix нет — имена переменных совпадают с полями."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    bot_token: SecretStr
    admin_user_id: int = 0
    allowed_chat_id: int = 0
    openrouter_api_key: SecretStr | None = None
    google_places_key: SecretStr | None = None
    db_path: Path = Path("data/bot.db")
    # Каталог пакета персоны (CLAUDE.md, "Интерфейсы: пакет персоны"): конфиг, промпты,
    # few-shot, стикеры, заведения. Не задан и allow_demo_persona=false -> процесс не
    # стартует: потерянный монтаж на проде не должен молча запускать демо-персону.
    persona_dir: Path | None = None
    allow_demo_persona: bool = False
    # Домашняя точка погоды (CLAUDE.md, "Интерфейсы: погода"). Координаты — личные
    # данные владельца, а репозиторий публичный: в config.yaml и в коде их нет,
    # только здесь, в .env. Не заданы обе -> блок погоды пуст (WARNING на старте),
    # погода по месту из вопроса при этом работает.
    weather_latitude: float | None = None
    weather_longitude: float | None = None
    weather_home_name: str = ""
    # Устаревшие пути к файлам персоны (до PR-2 каждый файл задавался своей переменной).
    # Оставлены на один релиз, чтобы старый .env не ломал загрузку: значение игнорируется,
    # на старте один WARNING (deprecated_fields_set). Удаляются после публикации.
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
        """Имена переменных окружения устаревших путей, которые заданы в окружении/.env."""
        return [name.upper() for name in DEPRECATED_PATH_FIELDS if getattr(self, name) is not None]


class PersonaSelector(BaseSettings):
    """Только выбор пакета персоны: для CLI, которым не нужен BOT_TOKEN (replay, places_fill)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    persona_dir: Path | None = None
    allow_demo_persona: bool = False
