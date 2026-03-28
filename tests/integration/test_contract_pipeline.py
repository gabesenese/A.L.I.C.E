from dataclasses import dataclass

from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline


@dataclass
class _NlpResult:
    intent: str
    intent_confidence: float
    keywords: list


class _FakeNlp:
    def process(self, text):
        if "weather" in text.lower():
            return _NlpResult(intent="weather:current", intent_confidence=0.92, keywords=["weather"])
        return _NlpResult(intent="conversation:general", intent_confidence=0.8, keywords=[])


class _FakeMemory:
    def __init__(self):
        self._stored = []

    def search(self, query, top_k=8):
        return [{"content": f"mem:{query}", "score": 0.7}][:top_k]

    def store_memory(self, content, memory_type="episodic", context=None):
        self._stored.append({"content": content, "memory_type": memory_type, "context": context or {}})


class _FakePlugins:
    def execute_for_intent(self, intent, query, entities, context):
        if intent.startswith("weather"):
            return {"success": True, "response": "It is sunny.", "plugin": "WeatherPlugin", "confidence": 0.9}
        return None


class _FakeLlm:
    def chat(self, user_input, use_history=True):
        return f"LLM:{user_input}"


class _FakeAlice:
    def __init__(self):
        self.nlp = _FakeNlp()
        self.memory = _FakeMemory()
        self.plugins = _FakePlugins()
        self.llm = _FakeLlm()


def test_contract_pipeline_handles_tool_route():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="weather in boston", user_id="u1", turn_number=1)

    assert result.handled is True
    assert "sunny" in result.response_text.lower()
    assert result.metadata["route"] == "tool"


def test_contract_pipeline_handles_llm_route():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="tell me a joke", user_id="u1", turn_number=2)

    assert result.handled is True
    assert result.response_text.startswith("LLM:")
    assert result.metadata["route"] == "llm"
