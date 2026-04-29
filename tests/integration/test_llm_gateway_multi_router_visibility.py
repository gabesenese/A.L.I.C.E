from ai.core.llm_gateway import LLMGateway, get_llm_gateway, reset_gateway
from ai.core.llm_policy import LLMCallType


class _DummyConfig:
    def __init__(self) -> None:
        self.model = "dummy-base"
        self.active_model = "dummy-active"


class _DummyLLM:
    def __init__(self) -> None:
        self.config = _DummyConfig()

    def query_knowledge(self, question: str) -> str:
        return f"knowledge:{question[:32]}"

    def parse_complex_input(self, user_input: str):
        return {"intent": "conversation:question", "raw": user_input}

    def phrase_with_tone(self, content: str, tone: str, context=None) -> str:
        return str(content)

    def audit_logic(self, logic_chain):
        return {"has_errors": False}

    def chat(self, prompt: str, use_history: bool = False) -> str:
        return f"chat:{prompt[:32]}"


class _AssistFirstLLM(_DummyLLM):
    def __init__(self) -> None:
        super().__init__()
        self.chat_calls = 0

    def query_knowledge(self, question: str) -> str:
        return "knowledge:assist-first"

    def chat(self, prompt: str, use_history: bool = False) -> str:
        self.chat_calls += 1
        return super().chat(prompt, use_history=use_history)


class _LowConfidenceRouter:
    def __init__(self) -> None:
        self.last_route = {"role": "reasoning", "model": "llama3.1:8b"}

    def generate(self, request: str, context=None):
        return {
            "response": "router unavailable",
            "confidence": 0.0,
            "model": "router_unavailable",
        }

    def describe_models(self):
        return {
            "fast": "llama3.2:3b",
            "reasoning": "llama3.1:8b",
            "coding": "qwen2.5-coder:7b",
        }

    def runtime_status(self):
        return {
            "all_roles_ready": False,
            "role_health": {"fast": False, "reasoning": True, "coding": False},
            "health_error": "mock-unavailable",
        }


def test_gateway_reports_model_roles_and_last_route(monkeypatch):
    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "1")
    monkeypatch.setenv("ALICE_MULTI_LLM_MOCK", "1")

    reset_gateway()
    gateway = get_llm_gateway(_DummyLLM())

    response = gateway.request(
        prompt="refactor this python function",
        call_type=LLMCallType.GENERATION,
        use_history=False,
        context={"intent": "code:refactor"},
        user_input="refactor this python function",
    )

    assert response.success
    assert response.model_used is not None
    assert response.route_source == "multi_router"

    stats = gateway.get_statistics()
    assert "model_roles" in stats
    assert "last_route" in stats
    assert isinstance(stats.get("model_roles", {}), dict)
    assert isinstance(stats.get("last_route", {}), dict)
    assert set(["fast", "reasoning", "coding"]).issubset(
        set(stats.get("model_roles", {}).keys())
    )

    reset_gateway()


def test_generation_uses_assist_paths_before_chat(monkeypatch):
    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "0")

    llm = _AssistFirstLLM()
    gateway = LLMGateway(llm_engine=llm, learning_engine=None)
    gateway.policy.require_user_approval = False

    response = gateway.request(
        prompt="What is polymorphism?",
        call_type=LLMCallType.GENERATION,
        use_history=False,
        context={"intent": "conversation:question"},
        user_input="What is polymorphism?",
    )

    assert response.success
    assert response.response == "knowledge:assist-first"
    assert llm.chat_calls == 0


def test_generation_routes_through_multi_router_when_enabled(monkeypatch):
    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "1")
    monkeypatch.setenv("ALICE_MULTI_LLM_MOCK", "1")

    reset_gateway()
    gateway = get_llm_gateway(_DummyLLM())

    response = gateway.request(
        prompt="explain tradeoffs between two plans",
        call_type=LLMCallType.GENERATION,
        use_history=False,
        context={"intent": "conversation:question"},
        user_input="explain tradeoffs between two plans",
    )

    assert response.success
    assert response.route_source == "multi_router"
    assert gateway.get_statistics().get("multi_router_calls", 0) >= 1

    reset_gateway()


def test_generation_strict_mode_blocks_legacy_fallback_when_router_unavailable(
    monkeypatch,
):
    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "1")
    monkeypatch.setenv("ALICE_MULTI_LLM_STRICT_GENERATION", "1")

    gateway = LLMGateway(llm_engine=_DummyLLM(), learning_engine=None)
    gateway.policy.require_user_approval = False
    gateway.model_router = _LowConfidenceRouter()

    response = gateway.request(
        prompt="explain this",
        call_type=LLMCallType.GENERATION,
        use_history=False,
        context={"intent": "conversation:question"},
        user_input="explain this",
    )

    assert response.success is False
    assert response.route_source == "multi_router"
    assert response.policy_reason == "strict_generation_router"
    low = str(response.response or "").lower()
    assert "required model roles" not in low
    assert "strict mode" not in low
