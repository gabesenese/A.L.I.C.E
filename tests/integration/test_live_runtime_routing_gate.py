from ai.core.nlp_processor import NLPProcessor
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline


class _LiveMemory:
    def search(self, query, top_k=8):
        return [{"content": f"mem:{query}", "score": 0.7}][:top_k]

    def store_memory(self, content, memory_type="episodic", context=None):
        return None


class _LivePlugins:
    def execute_for_intent(self, intent, query, entities, context):
        if str(intent or "").startswith("weather:"):
            return {
                "success": True,
                "response": "Weather data ready.",
                "plugin": "WeatherPlugin",
                "confidence": 0.9,
            }
        if str(intent or "").startswith("notes:"):
            return {
                "success": False,
                "response": "",
                "plugin": "NotesPlugin",
                "confidence": 0.2,
                "error": "notes backend unavailable",
            }
        return None


class _LiveLlm:
    def chat(self, user_input, use_history=True):
        return f"LLM:{user_input}"


class _LiveSelfReflection:
    def list_codebase(self):
        return [{"path": "app/main.py"}, {"path": "ai/core/nlp_processor.py"}]


class _LiveAlice:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.memory = _LiveMemory()
        self.plugins = _LivePlugins()
        self.llm = _LiveLlm()
        self.self_reflection = _LiveSelfReflection()
        self.context_resolver = None
        self.last_intent = ""
        self.last_entities = {}

    def _is_location_query(self, text):
        return "location" in str(text or "").lower()

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

    def _handle_code_request(self, user_input, entities=None):
        return None


def test_live_runtime_routes_companion_reasoning_prompt_to_conversation():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input=(
            "let's think through how to make alice more companion-like "
            "without adding hardcoded fallbacks"
        ),
        user_id="u-live",
        turn_number=1,
    )

    assert result.handled is True
    assert result.metadata["route"] in {"llm", "conversation"}
    assert str(result.metadata["intent"]).startswith("conversation:")


def test_live_runtime_routes_rain_weekend_to_weather_tool():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="will it rain this weekend?",
        user_id="u-live",
        turn_number=2,
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert result.metadata["intent"] == "weather:forecast"


def test_live_runtime_routes_review_understanding_prompt_to_conversation():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="let's review your understanding",
        user_id="u-live",
        turn_number=3,
    )

    assert result.handled is True
    assert result.metadata["route"] in {"llm", "conversation"}
    assert str(result.metadata["intent"]) == "conversation:understanding_review"


def test_live_runtime_routes_review_what_you_understood_to_conversation():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="review what you understood",
        user_id="u-live",
        turn_number=4,
    )

    assert result.handled is True
    assert result.metadata["route"] in {"llm", "conversation"}
    assert str(result.metadata["intent"]) == "conversation:understanding_review"


def test_live_runtime_routes_help_prompt_to_conversation_help():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="what can you do?",
        user_id="u-live",
        turn_number=7,
    )

    assert result.handled is True
    assert result.metadata["route"] in {"llm", "conversation"}
    assert str(result.metadata["intent"]) == "conversation:help"


def test_live_runtime_routes_explicit_read_note_to_notes_tool():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="read my note about Alice",
        user_id="u-live",
        turn_number=5,
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert str(result.metadata["intent"]).startswith("notes:read")


def test_live_runtime_routes_show_my_notes_to_notes_tool():
    alice = _LiveAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="show my notes",
        user_id="u-live",
        turn_number=6,
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert str(result.metadata["intent"]).startswith("notes:")
