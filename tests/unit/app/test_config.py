import pytest

from app.config import Settings, get_settings


@pytest.mark.asyncio
async def test_settings_load_defaults() -> None:
    settings = get_settings()
    assert settings.ollama_model
    assert settings.ollama_base_url.startswith("http")
    assert settings.log_level


def test_the_default_model_is_one_a_local_machine_can_run():
    """The default was llama3.3:70b, which needs roughly 40GB of RAM, while the
    CLI defaulted to llama3.1:8b — so the two entry points disagreed about which
    model Alice runs."""
    assert "70b" not in get_settings().ollama_model.lower()


@pytest.mark.parametrize("variable", ["ALICE_OLLAMA_HOST", "OLLAMA_HOST"])
def test_both_spellings_of_the_ollama_host_are_honoured(monkeypatch, variable):
    """docker-compose sets ALICE_OLLAMA_HOST. An explicit alias replaces the
    env_prefix rather than composing with it, so the prefixed spelling resolved
    to nothing and the container talked to its own localhost."""
    monkeypatch.setenv(variable, "http://elsewhere:11434")
    assert Settings().ollama_base_url == "http://elsewhere:11434"


def test_the_prefixed_spelling_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("ALICE_OLLAMA_HOST", "http://prefixed:11434")
    monkeypatch.setenv("OLLAMA_HOST", "http://bare:11434")
    assert Settings().ollama_base_url == "http://prefixed:11434"


def test_cors_origins_parse_into_a_list(monkeypatch):
    monkeypatch.setenv("ALICE_CORS_ORIGINS", "http://localhost:3000, https://alice.local")
    assert Settings().cors_origins_list == ["http://localhost:3000", "https://alice.local"]


def test_cors_defaults_to_a_wildcard():
    assert get_settings().cors_origins_list == ["*"]


def test_every_setting_is_read_by_something():
    """A setting nobody reads invites an operator to configure behavior that
    will not change. Seven such fields accumulated — the chroma ones outlived
    the only module that imported chromadb."""
    import subprocess
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    fields = set(Settings().model_dump())

    unread = []
    for field in fields:
        found = subprocess.run(
            ["grep", "-rl", "--include=*.py", f"\\.{field}", "app", "ai", "brain", "scripts"],
            cwd=project_root,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if not [path for path in found if path != "app/config.py"]:
            unread.append(field)

    assert not unread, f"settings nobody reads: {sorted(unread)}"
