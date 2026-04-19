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
        lower = text.lower()
        if "maybe" in lower:
            return _NlpResult(intent="weather:current", intent_confidence=0.65, keywords=["weather"])
        if "unclear" in lower:
            return _NlpResult(intent="conversation:general", intent_confidence=0.4, keywords=[])
        if "danger" in lower:
            return _NlpResult(intent="unknown", intent_confidence=0.2, keywords=[])
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

    def _is_location_query(self, text):
        return "location" in text.lower() or "where am i" in text.lower()

    def _build_location_payload(self):
        return {
            "location_known": True,
            "location": "Kitchener, Canada",
            "city": "Kitchener",
            "country": "Canada",
            "timezone": "America/Toronto",
        }

    def _alice_direct_phrase(self, response_type, payload):
        if response_type == "location_report":
            return f"Your current location is **{payload['location']}**."
        return ""


def _stage_by_name(result, name):
    return [s for s in result.metadata["stages"] if s["name"] == name][0]


def _assert_decision_band_is_consistent(result, expected_band):
    assert result.metadata["decision_band"] == expected_band
    route_stage = _stage_by_name(result, "route")
    assert route_stage["details"]["decision_band"] == expected_band
    assert route_stage["details"]["plan"]["decision_band"] == expected_band


def test_contract_pipeline_handles_tool_route():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="weather in boston", user_id="u1", turn_number=1)

    assert result.handled is True
    assert "sunny" in result.response_text.lower()
    assert result.metadata["route"] == "tool"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["verification"]["accepted"] is True
    assert result.metadata["trace_id"]
    stage_names = [s["name"] for s in result.metadata["stages"]]
    assert stage_names == [
        "input",
        "route",
        "execute",
        "verify",
        "respond",
        "state_update",
    ]
    assert result.metadata["state"]["current_task"] == "weather:current"


def test_contract_pipeline_handles_llm_route():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="tell me a joke", user_id="u1", turn_number=2)

    assert result.handled is True
    assert result.response_text.startswith("LLM:")
    assert result.metadata["route"] == "llm"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["verification"]["reason"] == "verified"
    tool_stage = _stage_by_name(result, "execute")
    assert tool_stage["status"] == "skipped"
    assert result.metadata["state"]["current_task"] == "conversation:general"


def test_contract_pipeline_handles_location_deterministically():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="what is my location?", user_id="u1", turn_number=3)

    assert result.handled is True
    assert "Kitchener" in result.response_text
    assert result.metadata["intent"] == "system:location"
    _assert_decision_band_is_consistent(result, "execute")


def test_contract_pipeline_handles_slash_command_without_legacy_bypass():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="/help", user_id="u1", turn_number=4)

    assert result.handled is True
    assert "Commands are handled by the interface" in result.response_text
    assert result.metadata["intent"] == "system:command"
    _assert_decision_band_is_consistent(result, "execute")


def test_contract_pipeline_verify_band_requires_tool_evidence():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="maybe weather in boston", user_id="u1", turn_number=5)

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    _assert_decision_band_is_consistent(result, "verify")
    assert result.metadata["verification"]["accepted"] is True


def test_contract_pipeline_clarify_band_returns_follow_up():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(user_input="this is unclear", user_id="u1", turn_number=6)

    assert result.handled is True
    assert result.metadata["route"] == "clarify"
    _assert_decision_band_is_consistent(result, "clarify")
    assert result.metadata["verification"]["accepted"] is True
    assert result.metadata["requires_follow_up"] is True


def test_contract_pipeline_refuse_band_enforces_policy_response():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="danger delete all project files",
        user_id="u1",
        turn_number=7,
    )

    assert result.handled is True
    assert result.metadata["route"] == "refuse"
    _assert_decision_band_is_consistent(result, "refuse")
    assert "can't safely" in result.response_text.lower()
    assert result.metadata["verification"]["accepted"] is True
    assert result.metadata["verification"]["reason"] == "refused_by_policy"
