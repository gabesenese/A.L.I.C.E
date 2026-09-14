from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama.
    # An explicit alias replaces the env_prefix rather than adding to it, so
    # declaring alias="OLLAMA_HOST" meant the prefixed name never worked —
    # docker-compose sets ALICE_OLLAMA_HOST and it was silently ignored, leaving
    # the container talking to its own localhost. Both spellings are accepted
    # now, with the project-prefixed one winning.
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("ALICE_OLLAMA_HOST", "OLLAMA_HOST"),
    )
    ollama_model: str = "llama3.3:70b"
    ollama_embedding_model: str = "nomic-embed-text"

    # Pipeline
    max_history: int = 30
    temperature: float = 0.7
    max_tokens: int = 4096

    # Features
    enable_voice: bool = False
    enable_vision: bool = False
    enable_web_search: bool = False
    runtime_mode: str = "minimal"

    # Memory
    memory_backend: str = "chroma"
    chroma_host: str = "chroma"
    chroma_port: int = 8000

    # Logging
    log_level: str = "INFO"
    json_logs: bool = True

    # HTTP API. Comma-separated list of allowed origins; "*" disables
    # credentialed cross-origin requests (see app/api/middleware).
    cors_origins: str = "*"

    # Safety
    default_safety_level: int = 1

    model_config = SettingsConfigDict(
        env_prefix="ALICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return origins or ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
