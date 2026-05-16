from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mongodb_uri: str
    mongodb_db_name: str = "PhotoBooth"


settings = Settings()
