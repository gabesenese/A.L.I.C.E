from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
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

    # Memory
    memory_backend: str = "chroma"
    chroma_host: str = "chroma"
    chroma_port: int = 8000

    # Logging
    log_level: str = "INFO"
    json_logs: bool = True

    # Safety
    default_safety_level: int = 1

    model_config = SettingsConfigDict(
        env_prefix="ALICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
