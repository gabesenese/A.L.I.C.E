"""Alice against each response shape Ollama can return, through a fake transport.

The model is meant to be the only variable: qwen3 and gpt-oss (thinking models),
models without tool calling, and Ollama cloud models (usage limits) must all
produce a clean answer or an honest sentence.
"""

import json

import pytest

from ai.core import llm_engine
from ai.core.llm_engine import LLMConfig, LLMUnavailableError, LocalLLMEngine
from ai.runtime.boundaries.boundary_factory import _llm_unavailable_text


class _Response:
    def __init__(self, status_code=200, payload=None, text="", lines=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)
        self._lines = list(lines or [])

    def json(self):
        return self._payload

    def iter_lines(self):
        yield from self._lines


class _FakeOllama:
    """Records every request and answers from a script, one response per call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def get(self, url, **kwargs):
        return _Response(200, {"models": [{"name": "qwen3:14b"}, {"name": "llama3.1:8b"}]})


def _engine(monkeypatch, fake, model="qwen3:14b"):
    monkeypatch.setattr(llm_engine.requests, "post", fake.post)
    monkeypatch.setattr(llm_engine.requests, "get", fake.get)
    monkeypatch.setattr(llm_engine.time, "sleep", lambda *_: None)
    engine = LocalLLMEngine(LLMConfig(model=model, use_fine_tuned=False))
    engine.system_prompt = "You are Alice."
    return engine


def _chat_reply(content, **message):
    return _Response(200, {"message": {"role": "assistant", "content": content, **message}})


def test_thinking_field_never_reaches_the_user(monkeypatch):
    fake = _FakeOllama(_chat_reply("Use SQLite.", thinking="The user wants a database. Let me weigh..."))
    engine = _engine(monkeypatch, fake)

    assert engine.chat("which database?") == "Use SQLite."
    assert engine.conversation_history[-1]["content"] == "Use SQLite."


def test_inline_think_tags_are_stripped(monkeypatch):
    fake = _FakeOllama(_chat_reply("<think>\nweighing options\n</think>\n\nUse SQLite."))
    engine = _engine(monkeypatch, fake)

    assert engine.chat("which database?") == "Use SQLite."


def test_unclosed_think_is_not_shown_as_an_answer(monkeypatch):
    fake = _FakeOllama(_chat_reply("<think>still reasoning when the budget ran out"))
    engine = _engine(monkeypatch, fake)

    assert engine.chat("which database?") == ""


def test_streamed_think_tags_split_across_chunks_are_filtered(monkeypatch):
    pieces = ["<th", "ink>plan the", " answer</thi", "nk>Use ", "SQLite."]
    lines = [json.dumps({"message": {"content": p}}).encode() for p in pieces]
    fake = _FakeOllama(_Response(200, lines=lines))
    engine = _engine(monkeypatch, fake)

    assert "".join(engine.stream_chat("which database?")) == "Use SQLite."


def test_every_request_sets_context_and_thinking_and_leaves_gpu_layers_to_ollama(monkeypatch):
    fake = _FakeOllama(_chat_reply("ok"))
    engine = _engine(monkeypatch, fake)
    engine.chat("hi")
    engine.chat_with_tools([{"role": "user", "content": "hi"}])
    list(engine.stream_chat("hi"))

    for _url, kwargs in fake.calls:
        body = kwargs["json"]
        assert body["options"]["num_ctx"] == 8192
        assert "num_gpu" not in body["options"]
        assert body["think"] is False


def test_gpt_oss_gets_a_thinking_level_not_false(monkeypatch):
    fake = _FakeOllama(_chat_reply("ok"))
    engine = _engine(monkeypatch, fake, model="gpt-oss:20b")
    engine.chat("hi")

    assert fake.calls[-1][1]["json"]["think"] == "low"


def test_server_that_rejects_think_is_retried_without_it(monkeypatch):
    fake = _FakeOllama(
        _Response(400, {"error": '"llama3.1:8b" does not support thinking'}),
        _chat_reply("Hello."),
    )
    engine = _engine(monkeypatch, fake, model="llama3.1:8b")

    assert engine.chat("hi") == "Hello."
    assert "think" not in fake.calls[-1][1]["json"]


def test_tool_call_arguments_as_a_json_string_are_parsed(monkeypatch):
    call = {"function": {"name": "list_files", "arguments": '{"path": "ai", "limit": 5}'}}
    fake = _FakeOllama(_chat_reply("", tool_calls=[call]))
    engine = _engine(monkeypatch, fake)

    response = engine.chat_with_tools([{"role": "user", "content": "files?"}], tools=[{"type": "function"}])

    assert response.tool_calls[0].name == "list_files"
    assert response.tool_calls[0].arguments == {"path": "ai", "limit": 5}


def test_model_without_tools_answers_as_plain_chat(monkeypatch):
    fake = _FakeOllama(
        _Response(400, {"error": "registry.ollama.ai/library/gemma2:9b does not support tools"}),
        _chat_reply("There are three files."),
    )
    engine = _engine(monkeypatch, fake, model="gemma2:9b")

    response = engine.chat_with_tools([{"role": "user", "content": "files?"}], tools=[{"type": "function"}])

    assert response.content == "There are three files."
    assert response.tool_calls == []
    assert "tools" not in fake.calls[-1][1]["json"]
    engine.chat_with_tools([{"role": "user", "content": "again"}], tools=[{"type": "function"}])
    assert "tools" not in fake.calls[-1][1]["json"]


def test_cloud_usage_limit_is_named_honestly(monkeypatch):
    fake = _FakeOllama(_Response(429, {"error": "you have reached your hourly usage limit"}))
    engine = _engine(monkeypatch, fake, model="gpt-oss:120b-cloud")

    with pytest.raises(LLMUnavailableError) as caught:
        engine.chat("hi")

    assert caught.value.reason == "rate_limited"
    assert len(fake.calls) == 1
    assert "usage limit" in _llm_unavailable_text(caught.value)


def test_missing_model_names_the_pull_command(monkeypatch):
    fake = _FakeOllama(_Response(404, {"error": "model 'qwen3:14b' not found"}))
    engine = _engine(monkeypatch, fake)

    with pytest.raises(LLMUnavailableError) as caught:
        engine.chat("hi")

    assert "ollama pull qwen3:14b" in _llm_unavailable_text(caught.value)


def test_model_and_host_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ALICE_MODEL", "gpt-oss:120b-cloud")
    monkeypatch.setenv("ALICE_OLLAMA_HOST", "gpu-box:11434")
    fake = _FakeOllama(_chat_reply("ok"))
    monkeypatch.setattr(llm_engine.requests, "post", fake.post)
    monkeypatch.setattr(llm_engine.requests, "get", fake.get)

    engine = LocalLLMEngine(LLMConfig(use_fine_tuned=False))
    engine.chat("hi")

    assert engine.config.active_model == "gpt-oss:120b-cloud"
    assert fake.calls[-1][0] == "http://gpu-box:11434/api/chat"
    assert fake.calls[-1][1]["json"]["model"] == "gpt-oss:120b-cloud"


def test_api_settings_read_alice_model(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("ALICE_MODEL", "qwen3:14b")

    assert Settings().ollama_model == "qwen3:14b"
