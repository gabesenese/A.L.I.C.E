from ai.core.llm_engine import LLMConfig, LocalLLMEngine


class _DummyResponse:
    def __init__(self, status_code=200, payload=None, text="", lines=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self._lines = list(lines or [])

    def json(self):
        return self._payload

    def iter_lines(self):
        for item in self._lines:
            yield item


def _build_engine(model: str = "llama3.3:70b") -> LocalLLMEngine:
    engine = LocalLLMEngine.__new__(LocalLLMEngine)
    engine.config = LLMConfig(model=model, use_fine_tuned=False)
    engine.conversation_history = []
    engine._available_models = []
    engine.system_prompt = "test"
    return engine


def test_model_falls_back_to_available_local_model():
    engine = _build_engine(model="llama3.3:70b")

    engine._ensure_active_model_available(["llama3.1:8b"])

    assert engine.config.active_model == "llama3.1:8b"


def test_stream_chat_surfaces_non_200_error(monkeypatch):
    engine = _build_engine(model="missing-model")

    def _fake_post(*args, **kwargs):
        return _DummyResponse(
            status_code=404,
            payload={"error": "model 'missing-model' not found"},
            text="model missing",
        )

    monkeypatch.setattr("ai.core.llm_engine.requests.post", _fake_post)

    chunks = list(engine.stream_chat("hello"))

    assert chunks
    merged = "".join(chunks).lower()
    assert "[error]" in merged
    assert "not found" in merged


def test_stream_chat_reports_empty_output(monkeypatch):
    engine = _build_engine(model="llama3.1:8b")

    def _fake_post(*args, **kwargs):
        return _DummyResponse(status_code=200, payload={}, lines=[])

    monkeypatch.setattr("ai.core.llm_engine.requests.post", _fake_post)

    chunks = list(engine.stream_chat("hello"))

    assert chunks
    merged = "".join(chunks).lower()
    assert "[warning]" in merged
    assert "no output" in merged
    assert engine.conversation_history == []
