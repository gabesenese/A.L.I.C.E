import json
from dataclasses import dataclass
from pathlib import Path

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
        if "unclear" in lower:
            return _NlpResult(
                intent="conversation:general", intent_confidence=0.4, keywords=[]
            )
        if "danger" in lower:
            return _NlpResult(intent="unknown", intent_confidence=0.2, keywords=[])
        if "weather" in lower:
            return _NlpResult(
                intent="weather:current", intent_confidence=0.92, keywords=["weather"]
            )
        return _NlpResult(
            intent="conversation:general", intent_confidence=0.8, keywords=[]
        )


class _FakeMemory:
    def __init__(self):
        self._stored = []

    def search(self, query, top_k=8):
        return [{"content": f"mem:{query}", "score": 0.7}][:top_k]

    def store_memory(self, content, memory_type="episodic", context=None):
        self._stored.append(
            {"content": content, "memory_type": memory_type, "context": context or {}}
        )


class _FakePlugins:
    def execute_for_intent(self, intent, query, entities, context):
        if intent.startswith("weather"):
            return {
                "success": True,
                "response": "It is sunny.",
                "plugin": "WeatherPlugin",
                "confidence": 0.9,
            }
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


def _load_transcript(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_transcript_pipeline_end_to_end():
    transcript_path = Path(__file__).parent / "data" / "transcript_smoke.jsonl"
    rows = _load_transcript(transcript_path)

    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    for row in rows:
        result = pipeline.run_turn(
            user_input=str(row["user_input"]),
            user_id="transcript-user",
            turn_number=int(row["turn"]),
        )

        assert result.handled is True
        assert result.metadata["route"] == row["expected_route"]

        expected_contains = str(row.get("expected_contains") or "").strip().lower()
        if expected_contains:
            assert expected_contains in result.response_text.lower()

        if "requires_follow_up" in row:
            assert result.metadata["requires_follow_up"] is bool(
                row["requires_follow_up"]
            )

        stage_names = [stage["name"] for stage in result.metadata["stages"]]
        assert stage_names == [
            "input",
            "route",
            "execute",
            "verify",
            "respond",
            "state_update",
        ]
