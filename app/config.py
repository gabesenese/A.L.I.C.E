from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from ALICE_* environment variables or .env.

    Every field here is read by something. A setting nobody reads is worse than
    no setting: it invites an operator to configure behavior that will not
    change. `memory_backend`, `chroma_host`, `chroma_port`, `enable_vision`,
    `enable_web_search`, `json_logs` and `default_safety_level` were all in that
    state and have been removed — the chroma ones outlived the only module that
    imported chromadb.

    If you add a field, wire it up in the same change.
    """

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
    # A 70B model is not a default a local assistant can assume; it needs ~40GB
    # of RAM to run at all. The CLI already defaulted to llama3.1:8b, so the two
    # entry points disagreed about which model Alice runs.
    ollama_model: str = Field(
        default="llama3.1:8b",
        validation_alias=AliasChoices("ALICE_MODEL", "ALICE_OLLAMA_MODEL"),
    )

    # Features
    enable_voice: bool = False
    runtime_mode: str = "minimal"

    # Logging
    log_level: str = "INFO"

    # HTTP API. Comma-separated list of allowed origins; "*" disables
    # credentialed cross-origin requests (see app/api/middleware).
    cors_origins: str = "*"

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
