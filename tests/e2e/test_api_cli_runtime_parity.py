"""The HTTP API and the CLI must run the same ALICE, not two different ones.

The container previously built its own stub boundaries: routing was a four-prefix
string check and responses were a bare llm.chat() call with no memory, personality,
or context. Every request through /chat therefore reached a far weaker runtime than
the one `python app/main.py` used.
"""

import pytest

from ai.runtime.contract_pipeline import ContractPipeline

PROMPTS = [
    "what files can you inspect",
    "what's the weather like today",
    "let's work on alice",
]


def _container(client):
    return client.app.state.container


def test_api_pipeline_is_the_same_object_the_cli_uses(client):
    container = _container(client)
    assert isinstance(container.pipeline, ContractPipeline)
    assert container.pipeline is container.alice.contract_pipeline


def test_api_pipeline_uses_real_alice_components(client):
    container = _container(client)
    boundaries = container.pipeline.boundaries
    assert boundaries.routing is not None
    assert boundaries.tools is not None
    assert container.nlp is container.alice.nlp
    assert container.llm is container.alice.llm


def test_alice_is_built_once_and_shared(client):
    container = _container(client)
    assert container.alice is container.alice
    assert container.pipeline is container.pipeline


@pytest.fixture
def isolated_project_memory(tmp_path, monkeypatch):
    """Keep real turns out of the shared data/project_memory.json store.

    That file is keyed by user id and persists between runs, so tests that drive
    real turns against it leak state into unrelated tests.
    """
    import ai.memory.project_memory as project_memory

    monkeypatch.setattr(project_memory, "PROJECT_MEMORY_PATH", tmp_path / "project_memory.json")
    return tmp_path


@pytest.mark.parametrize("prompt", PROMPTS)
def test_chat_endpoint_returns_routed_turn_metadata(client, prompt, isolated_project_memory):
    response = client.post("/chat", json={"message": prompt, "user_id": "parity_probe"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["response"].strip()
    assert payload["trace_id"].strip()
