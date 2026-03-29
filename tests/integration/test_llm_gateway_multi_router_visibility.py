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
    assert response.route_source in {"multi_router", "legacy_engine"}

    stats = gateway.get_statistics()
    assert "model_roles" in stats
    assert "last_route" in stats
    assert isinstance(stats.get("model_roles", {}), dict)
    assert isinstance(stats.get("last_route", {}), dict)
    assert set(["fast", "reasoning", "coding"]).issubset(set(stats.get("model_roles", {}).keys()))

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
