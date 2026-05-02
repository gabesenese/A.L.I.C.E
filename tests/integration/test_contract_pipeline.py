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
        if (
            "let's think through" in lower
            and "make alice more companion-like" in lower
            and "hardcoded fallbacks" in lower
        ):
            return _NlpResult(
                intent="notes:create", intent_confidence=0.93, keywords=["make"]
            )
        if (
            "create a note" in lower
            and "making alice more companion-like" in lower
            and "hardcoded fallbacks" in lower
        ):
            return _NlpResult(
                intent="notes:create", intent_confidence=0.95, keywords=["note"]
            )
        if (
            "don't want you to check anything" in lower
            and "weather has been annoying lately" in lower
        ):
            return _NlpResult(
                intent="weather:current", intent_confidence=0.92, keywords=["weather"]
            )
        if "always running and checking" in lower and "surroundings" in lower:
            return _NlpResult(
                intent="conversation:clarification_needed",
                intent_confidence=0.55,
                keywords=[],
            )
        if "maybe" in lower:
            return _NlpResult(
                intent="weather:current", intent_confidence=0.65, keywords=["weather"]
            )
        if "unclear" in lower:
            return _NlpResult(
                intent="conversation:general", intent_confidence=0.4, keywords=[]
            )
        if "danger" in lower:
            return _NlpResult(intent="unknown", intent_confidence=0.2, keywords=[])
        if "weather" in text.lower():
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
    def __init__(self):
        self._attempts = {}

    def execute_for_intent(self, intent, query, entities, context):
        query_text = str(query or "").lower()
        if "retry weather" in query_text:
            count = int(self._attempts.get(query_text, 0)) + 1
            self._attempts[query_text] = count
            if count == 1:
                return {
                    "success": False,
                    "response": "",
                    "plugin": "WeatherPlugin",
                    "confidence": 0.25,
                    "error": "timeout contacting weather provider",
                }
            return {
                "success": True,
                "response": "Recovered weather data.",
                "plugin": "WeatherPlugin",
                "confidence": 0.88,
            }

        if "weather fail hard" in query_text:
            return {
                "success": False,
                "response": "",
                "plugin": "WeatherPlugin",
                "confidence": 0.2,
                "error": "invalid api key",
            }

        if "weather nested" in str(query or "").lower():
            return {
                "success": True,
                "response": None,
                "plugin": "WeatherPlugin",
                "confidence": 0.9,
                "data": {
                    "temperature": 22,
                    "condition": "partly cloudy",
                    "location": "Kitchener",
                    "plugin_type": "weather",
                    "message_code": "weather:current",
                },
            }
        if intent.startswith("weather"):
            return {
                "success": True,
                "response": "It is sunny.",
                "plugin": "WeatherPlugin",
                "confidence": 0.9,
            }
        return None


class _FakeSelfReflection:
    def list_codebase(self):
        return [
            {"path": "app/main.py"},
            {"path": "ai/core/llm_engine.py"},
            {"path": "tests/integration/test_contract_pipeline.py"},
        ]


class _FakeLlm:
    def chat(self, user_input, use_history=True):
        if "hallucinate" in user_input.lower():
            return (
                "I checked ai/dialogue_management.py and app/agents.py. "
                "The self_learning directory contains training workflows."
            )
        if "fabricate a report" in user_input.lower():
            return (
                "According to my knowledge, today is partly cloudy with a high of 22°C "
                "and a low of 15°C, with wind around 10 km/h in [your location]."
            )
        return f"LLM:{user_input}"


class _FakeAlice:
    def __init__(self):
        self.nlp = _FakeNlp()
        self.memory = _FakeMemory()
        self.plugins = _FakePlugins()
        self.llm = _FakeLlm()
        self.self_reflection = _FakeSelfReflection()

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
        if response_type == "weather_report":
            return (
                f"WEATHER_REPORT {payload.get('temperature')}C "
                f"{payload.get('condition')} in {payload.get('location')}"
            )
        if response_type == "weather_forecast":
            return "WEATHER_FORECAST"
        return ""

    def _handle_code_request(self, user_input, entities=None):
        low = str(user_input or "").lower()
        if "code" in low or "codebase" in low or ".py" in low:
            return "I can inspect local source code in this workspace."
        return None


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

    result = pipeline.run_turn(
        user_input="weather in boston", user_id="u1", turn_number=1
    )

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


def test_contract_pipeline_formats_weather_nested_tool_payload_without_llm_fallback():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="weather nested", user_id="u1", turn_number=12
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert result.metadata["intent"] == "weather:current"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.response_text.startswith("WEATHER_REPORT")
    assert "LLM:" not in result.response_text
    assert result.metadata["verification"]["accepted"] is True


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


def test_contract_pipeline_defers_current_world_summary_without_live_sources():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="what is happening in the world right now?",
        user_id="u1",
        turn_number=23,
    )

    assert result.handled is True
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] == "freshness:current_events"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["requires_follow_up"] is True
    assert result.metadata["verification"]["accepted"] is True
    response = result.response_text.lower()
    assert "live sources" in response
    assert "model memory" in response
    assert "llm:" not in response
    assert "pandemic" not in response


def test_contract_pipeline_emits_companion_memory_domains_metadata():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="continue our project roadmap and build automation plan",
        user_id="u1",
        turn_number=20,
    )

    assert result.handled is True
    companion = result.metadata["companion"]
    domains = companion["memory_domains"]
    assert sorted(domains.keys()) == [
        "causal_lessons",
        "identity",
        "preferences",
        "projects",
        "unresolved_threads",
    ]
    assert companion["policy_decision"] == "respond"
    assert isinstance(domains["projects"], list)
    assert domains["projects"]
    assert "relationship_mode" in companion["identity_model"]


def test_contract_pipeline_retries_transient_tool_failure_once():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="retry weather in boston",
        user_id="u1",
        turn_number=21,
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert "recovered weather data" in result.response_text.lower()
    action_discipline = result.metadata["companion"]["action_discipline"]
    assert action_discipline["retried"] is True
    assert action_discipline["retry_count"] == 2
    execute_stage = _stage_by_name(result, "execute")
    assert execute_stage["details"]["attempt_count"] == 2


def test_contract_pipeline_requires_approval_before_risky_tool_action():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="weather in boston and force push updates",
        user_id="u1",
        turn_number=22,
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert result.metadata["verification"]["reason"] == "approval_required"
    assert result.metadata["requires_follow_up"] is True
    assert "explicit approval" in result.response_text.lower()
    action_discipline = result.metadata["companion"]["action_discipline"]
    assert action_discipline["approval_required"] is True
    execute_stage = _stage_by_name(result, "execute")
    assert execute_stage["status"] == "skipped"


def test_contract_pipeline_uses_follow_up_policy_when_threads_are_open():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    first = pipeline.run_turn(
        user_input="this is unclear",
        user_id="u1",
        turn_number=30,
    )
    assert first.metadata["requires_follow_up"] is True

    second = pipeline.run_turn(
        user_input="tell me a joke",
        user_id="u1",
        turn_number=31,
    )

    assert second.handled is True
    assert second.metadata["companion"]["policy_decision"] == "follow_up"
    assert second.metadata["requires_follow_up"] is True
    assert "follow up next turn" in second.response_text.lower()


def test_contract_pipeline_routes_code_request_to_grounded_handler():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="can you check your local code?",
        user_id="u1",
        turn_number=8,
    )

    assert result.handled is True
    assert result.metadata["intent"] == "code:request"
    assert result.metadata["route"] == "local"
    _assert_decision_band_is_consistent(result, "execute")
    assert "inspect local source code" in result.response_text.lower()
    assert result.metadata["verification"]["accepted"] is True


def test_contract_pipeline_routes_weather_lexical_request_to_tool_even_when_nlp_is_generic():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="will it rain tomorrow?",
        user_id="u1",
        turn_number=10,
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert result.metadata["intent"] == "weather:forecast"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["verification"]["accepted"] is True


def test_weather_followup_personal_reaction_does_not_call_weather_tool():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    first = pipeline.run_turn(
        user_input="will it rain tomorrow?",
        user_id="u1",
        turn_number=14,
    )
    assert first.metadata["route"] == "tool"
    assert first.metadata["intent"] == "weather:forecast"

    second = pipeline.run_turn(
        user_input="thanks for letting me know. weather has been bipolar and i got a cold.",
        user_id="u1",
        turn_number=15,
    )

    assert second.handled is True
    assert second.metadata["route"] == "conversation"
    assert second.metadata["intent"] == "conversation:personal_reaction"
    assert (
        second.metadata["companion"]["policy_reason"]
        == "contextual_reaction_after_tool_result"
    )

    route_veto = dict(second.metadata["plan"].get("route_veto") or {})
    assert route_veto.get("applied") is True
    assert route_veto.get("reason") == "gratitude_plus_personal_state_no_new_request"
    assert route_veto.get("previous_intent") == "weather:forecast"
    assert route_veto.get("tool_execution_disabled") is True

    execute_stage = _stage_by_name(second, "execute")
    assert execute_stage["status"] == "skipped"
    assert "llm:" in second.response_text.lower()


def test_weather_followup_personal_reaction_without_gratitude_does_not_call_weather_tool():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    first = pipeline.run_turn(
        user_input="will it rain tomorrow?",
        user_id="u1",
        turn_number=16,
    )
    assert first.metadata["route"] == "tool"
    assert first.metadata["intent"] == "weather:forecast"

    second = pipeline.run_turn(
        user_input="i got a cold from all this weather change, hopefully it will get better soon or stay consistent",
        user_id="u1",
        turn_number=17,
    )

    assert second.handled is True
    assert second.metadata["route"] == "conversation"
    assert second.metadata["intent"] == "conversation:personal_reaction"
    assert (
        second.metadata["companion"]["policy_reason"]
        == "contextual_reaction_after_tool_result"
    )

    execute_stage = _stage_by_name(second, "execute")
    assert execute_stage["status"] == "skipped"
    assert "llm:" in second.response_text.lower()


def test_contract_pipeline_routes_proactive_agent_design_statement_to_llm_execute():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input=(
            "i was thinking the assistant is always running and checking the computer "
            "or the user's surroundings, and it should alert only when it needs attention "
            "or has a recommendation"
        ),
        user_id="u1",
        turn_number=13,
    )

    assert result.handled is True
    assert result.metadata["route"] == "llm"
    assert result.metadata["intent"] == "conversation:goal_statement"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["verification"]["accepted"] is True
    assert "LLM:" in result.response_text


def test_contract_pipeline_demotes_notes_create_misfire_for_collaborative_reasoning_prompt():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input=(
            "let's think through how to make alice more companion-like "
            "without adding hardcoded fallbacks"
        ),
        user_id="u1",
        turn_number=51,
    )

    assert result.handled is True
    assert result.metadata["route"] == "llm"
    assert result.metadata["intent"].startswith("conversation:")


def test_contract_pipeline_keeps_notes_create_when_explicit_note_request_present():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input=(
            "create a note about making alice more companion-like "
            "without hardcoded fallbacks"
        ),
        user_id="u1",
        turn_number=52,
    )

    assert result.handled is True
    assert result.metadata["intent"] == "notes:create"
    assert result.metadata["route"] == "tool"


def test_contract_pipeline_demotes_weather_tool_without_explicit_weather_request():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input=(
            "i don't want you to check anything, i'm just saying the weather "
            "has been annoying lately"
        ),
        user_id="u1",
        turn_number=53,
    )

    assert result.handled is True
    assert result.metadata["route"] == "llm"
    assert result.metadata["intent"].startswith("conversation:")


def test_contract_pipeline_blocks_unverified_llm_codebase_claims():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="please hallucinate a response",
        user_id="u1",
        turn_number=9,
    )

    assert result.handled is True
    assert result.metadata["route"] == "llm"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["verification"]["accepted"] is False
    assert result.metadata["verification"]["reason"] == "unverified_codebase_claim"
    assert "could not verify that result safely" in result.response_text.lower()


def test_contract_pipeline_blocks_unverified_llm_weather_claims():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="please fabricate a report",
        user_id="u1",
        turn_number=11,
    )

    assert result.handled is True
    assert result.metadata["route"] == "llm"
    _assert_decision_band_is_consistent(result, "execute")
    assert result.metadata["verification"]["accepted"] is False
    assert result.metadata["verification"]["reason"] == "unverified_weather_claim"
    assert "could not verify that result safely" in result.response_text.lower()


def test_contract_pipeline_handles_location_deterministically():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="what is my location?", user_id="u1", turn_number=3
    )

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

    result = pipeline.run_turn(
        user_input="maybe weather in boston", user_id="u1", turn_number=5
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    _assert_decision_band_is_consistent(result, "verify")
    assert result.metadata["verification"]["accepted"] is True


def test_contract_pipeline_clarify_band_returns_follow_up():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="this is unclear", user_id="u1", turn_number=6
    )

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


def test_contract_pipeline_emits_post_execution_contract_and_outcome_metadata():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="weather in boston", user_id="u1", turn_number=40
    )

    assert result.handled is True
    contract = result.metadata["turn_contract"]
    outcome = result.metadata["turn_execution_outcome"]
    post = result.metadata["post_execution_state_machine"]
    task_verification = result.metadata["task_verification"]

    assert contract["chosen_route"] == "tool"
    assert outcome["tool_success"] is True
    assert outcome["verification_passed"] is True
    assert post["phase"] in {"completed", "verified"}
    assert task_verification["accepted"] is True


def test_contract_pipeline_persists_execution_artifacts_in_memory_context():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="weather in boston", user_id="u1", turn_number=41
    )

    assert alice.memory._stored
    context = dict(alice.memory._stored[-1]["context"] or {})

    assert context["trace_id"] == result.metadata["trace_id"]
    assert context["turn_contract"]["chosen_route"] == "tool"
    assert context["turn_execution_outcome"]["verification_passed"] is True
    assert context["post_execution_state_machine"]["phase"] in {"completed", "verified"}
    assert context["task_verification"]["accepted"] is True


def test_contract_pipeline_escalates_hard_tool_failures_in_post_execution_state_machine():
    alice = _FakeAlice()
    boundaries = build_runtime_boundaries(alice)
    pipeline = ContractPipeline(boundaries)

    result = pipeline.run_turn(
        user_input="weather fail hard", user_id="u1", turn_number=42
    )

    assert result.handled is True
    assert result.metadata["route"] == "tool"
    assert result.metadata["verification"]["reason"] == "tool_failed"
    assert "could not verify that result safely" in result.response_text.lower()

    outcome = result.metadata["turn_execution_outcome"]
    post = result.metadata["post_execution_state_machine"]
    task_verification = result.metadata["task_verification"]

    assert outcome["tool_success"] is False
    assert outcome["verification_passed"] is False
    assert outcome["needs_escalation"] is True
    assert post["phase"] == "escalated"
    assert task_verification["accepted"] is False
