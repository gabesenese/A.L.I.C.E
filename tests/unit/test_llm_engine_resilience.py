"""What the LLM engine does when Ollama is slow, flapping, or simply not there.

Every one of these used to fail in a way the user felt directly: construction
blocked for fifteen seconds probing for a Windows-only binary, a connection that
answered /api/tags but reset /api/chat sent chat() into itself until the stack
ran out, and the retry that did exist dropped the companion context and the
routed intent on the way through.

Ollama is never running here. Everything below talks to a fake transport.
"""

import sys
import time

import pytest
import requests

from ai.core import llm_engine
from ai.core.llm_engine import LLMConfig, LocalLLMEngine
from ai.core.llm_gateway import LLMGateway
from ai.core.llm_policy import LLMCallType, LLMPolicy, LLMTransportPolicy


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _explode(*args, **kwargs):
    raise AssertionError(f"unexpected call with {args!r} {kwargs!r}")


@pytest.fixture
def offline(monkeypatch):
    """Block every route out of the process, so a test can open the ones it wants."""
    monkeypatch.setattr(llm_engine.requests, "get", _explode)
    monkeypatch.setattr(llm_engine.requests, "post", _explode)
    monkeypatch.setattr(llm_engine.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(llm_engine.subprocess, "run", _explode)
    monkeypatch.setattr(llm_engine.subprocess, "Popen", _explode)
    return monkeypatch


def _fast_engine(model: str = "llama3.1:8b", **transport_kwargs) -> LocalLLMEngine:
    """An engine whose retry backoff is short enough to run inside a test."""
    transport = LLMTransportPolicy(backoff_base=0.001, backoff_cap=0.002, **transport_kwargs)
    return LocalLLMEngine(LLMConfig(model=model, use_fine_tuned=False, transport=transport))


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_construction_touches_neither_the_network_nor_the_clock(offline):
    offline.setattr(llm_engine.time, "sleep", _explode)

    engine = LocalLLMEngine(LLMConfig(model="llama3.1:8b"))

    assert engine.config.active_model == "llama3.1:8b"
    assert engine._service_probed is False


def test_building_a_config_does_not_look_for_a_fine_tuned_model(offline):
    config = LLMConfig(model="llama3.3:70b")

    assert config._fine_tuned_checked is False
    assert config.active_model == "llama3.3:70b"


def test_construction_is_fast_even_with_ollama_missing(offline):
    started = time.perf_counter()
    LocalLLMEngine(LLMConfig(model="llama3.1:8b"))
    assert time.perf_counter() - started < 1.0


# --------------------------------------------------------------------------
# A flapping connection
# --------------------------------------------------------------------------


class _Flapping:
    """/api/tags answers, /api/chat resets — the shape that used to recurse."""

    def __init__(self):
        self.posts = 0

    def get(self, url, **kwargs):
        return _Response(200, {"models": [{"name": "llama3.1:8b"}]})

    def post(self, url, **kwargs):
        self.posts += 1
        raise requests.exceptions.ConnectionError("connection reset by peer")


def test_a_flapping_connection_retries_a_bounded_number_of_times(offline):
    flap = _Flapping()
    offline.setattr(llm_engine.requests, "get", flap.get)
    offline.setattr(llm_engine.requests, "post", flap.post)

    engine = _fast_engine(max_attempts=3)

    with pytest.raises(Exception) as caught:
        engine.chat("are you there")

    assert not isinstance(caught.value, RecursionError)
    assert "unavailable" in str(caught.value).lower()
    assert flap.posts == 3


def _stack_depth() -> int:
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def test_retries_happen_in_a_loop_rather_than_by_re_entering_chat(offline):
    """Same stack depth on every attempt is what separates a retry from recursion."""
    depths = []

    def post(url, **kwargs):
        depths.append(_stack_depth())
        raise requests.exceptions.ConnectionError("reset")

    offline.setattr(llm_engine.requests, "get", lambda *_a, **_k: _Response(200, {"models": [{"name": "m"}]}))
    offline.setattr(llm_engine.requests, "post", post)

    engine = _fast_engine(max_attempts=4)

    with pytest.raises(Exception):
        engine.chat("still there")

    assert len(depths) == 4
    assert len(set(depths)) == 1


class _RecoversAfter:
    def __init__(self, failures: int):
        self.remaining = failures
        self.payloads = []

    def get(self, url, **kwargs):
        return _Response(200, {"models": [{"name": "llama3.1:8b"}]})

    def post(self, url, json=None, **kwargs):
        self.payloads.append(json)
        if self.remaining > 0:
            self.remaining -= 1
            raise requests.exceptions.ConnectionError("reset")
        return _Response(200, {"message": {"content": "still here"}})


def test_every_argument_survives_a_retry(offline):
    offline.setattr(
        llm_engine,
        "_build_companion_context",
        lambda intent="", user_query="": f"|intent={intent}|",
    )
    transport = _RecoversAfter(failures=1)
    offline.setattr(llm_engine.requests, "get", transport.get)
    offline.setattr(llm_engine.requests, "post", transport.post)

    engine = _fast_engine()

    answer = engine.chat(
        "what did we decide",
        use_history=False,
        temperature=0.2,
        mode="final_answer_only",
        context="COMPANION-CONTEXT",
        intent="conversation:general",
    )

    assert answer == "still here"
    assert len(transport.payloads) == 2
    for payload in transport.payloads:
        system_blocks = [m["content"] for m in payload["messages"] if m["role"] == "system"]
        assert any("|intent=conversation:general|" in block for block in system_blocks)
        assert "COMPANION-CONTEXT" in system_blocks
        assert any("final_answer_only" in block for block in system_blocks)
        assert payload["options"]["temperature"] == pytest.approx(0.2)
        assert payload["messages"][-1] == {"role": "user", "content": "what did we decide"}


def test_a_client_error_is_never_retried(offline):
    calls = {"posts": 0}

    def post(url, **kwargs):
        calls["posts"] += 1
        return _Response(400, {"error": "bad request"}, text="bad request")

    offline.setattr(llm_engine.requests, "get", lambda *_a, **_k: _Response(200, {"models": []}))
    offline.setattr(llm_engine.requests, "post", post)

    engine = _fast_engine()

    with pytest.raises(Exception) as caught:
        engine.chat("hello")

    assert "400" in str(caught.value)
    assert calls["posts"] == 1


def test_a_server_error_is_retried_within_the_budget(offline):
    calls = {"posts": 0}

    def post(url, **kwargs):
        calls["posts"] += 1
        return _Response(503, {"error": "loading model"}, text="loading model")

    offline.setattr(llm_engine.requests, "get", lambda *_a, **_k: _Response(200, {"models": []}))
    offline.setattr(llm_engine.requests, "post", post)

    engine = _fast_engine(max_attempts=3)

    with pytest.raises(Exception):
        engine.chat("hello")

    assert calls["posts"] == 3


def test_a_timeout_is_retried_and_then_reported_as_a_timeout(offline):
    calls = {"posts": 0}

    def post(url, **kwargs):
        calls["posts"] += 1
        raise requests.exceptions.Timeout("read timed out")

    offline.setattr(llm_engine.requests, "get", lambda *_a, **_k: _Response(200, {"models": []}))
    offline.setattr(llm_engine.requests, "post", post)

    engine = _fast_engine(max_attempts=2)

    with pytest.raises(Exception) as caught:
        engine.chat("hello")

    assert "timeout" in str(caught.value).lower()
    assert calls["posts"] == 2


# --------------------------------------------------------------------------
# Ollama down
# --------------------------------------------------------------------------


def _refuse(*args, **kwargs):
    raise requests.exceptions.ConnectionError("connection refused")


def test_ollama_down_degrades_promptly_instead_of_hanging(offline):
    offline.setattr(llm_engine.requests, "get", _refuse)
    offline.setattr(llm_engine.requests, "post", _refuse)

    engine = _fast_engine(max_attempts=3)

    started = time.perf_counter()
    with pytest.raises(Exception) as caught:
        engine.chat("you up?")
    elapsed = time.perf_counter() - started

    assert "ollama not running" in str(caught.value).lower()
    assert elapsed < 5.0


def test_ollama_down_gives_stream_chat_an_offline_message_rather_than_an_exception(offline):
    offline.setattr(llm_engine.requests, "get", _refuse)
    offline.setattr(llm_engine.requests, "post", _refuse)

    engine = _fast_engine()
    chunks = list(engine.stream_chat("you up?"))

    assert "[OFFLINE]" in "".join(chunks)


def test_health_probes_use_the_short_timeout_not_the_generation_timeout(offline):
    seen = []

    def get(url, **kwargs):
        seen.append(kwargs.get("timeout"))
        return _Response(200, {"models": [{"name": "llama3.1:8b"}]})

    offline.setattr(llm_engine.requests, "get", get)
    engine = _fast_engine()

    assert engine._is_ollama_running() is True
    assert seen == [engine.transport.health_timeout]
    assert engine.transport.health_timeout < engine.config.timeout


def test_check_connection_reports_false_when_the_tag_listing_errors(offline):
    offline.setattr(llm_engine.requests, "get", lambda *_a, **_k: _Response(500, {}, text="boom"))
    engine = _fast_engine()

    assert engine._check_connection() is False


def test_autostart_waits_a_bounded_time_and_is_tried_once(offline):
    spawned = []
    offline.setattr(llm_engine, "AUTOSTART_POLL_SECONDS", 0.01)
    offline.setattr(llm_engine.requests, "get", _refuse)
    offline.setattr(llm_engine.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    offline.setattr(llm_engine.shutil, "which", lambda *_a, **_k: __file__)

    engine = _fast_engine(autostart_wait=0.05)
    offline.setattr(engine, "_find_ollama_executable", lambda: "/usr/bin/ollama")

    started = time.perf_counter()
    assert engine._ensure_ollama_running() is False
    assert engine._ensure_ollama_running() is False
    elapsed = time.perf_counter() - started

    assert len(spawned) == 1, "a server that declined to start is not restarted every turn"
    assert elapsed < 2.0


def test_the_first_use_probe_never_spawns_a_server(offline):
    offline.setattr(llm_engine.requests, "get", _refuse)
    engine = _fast_engine()

    assert engine._ensure_service_probed() is False
    assert engine._autostart_attempted is False


# --------------------------------------------------------------------------
# audit_logic reports whether it ran
# --------------------------------------------------------------------------


def test_a_failed_audit_does_not_claim_the_logic_is_sound(offline):
    offline.setattr(llm_engine.requests, "post", lambda *_a, **_k: _Response(500, {}, text="boom"))
    engine = _fast_engine()

    audit = engine.audit_logic(["one", "two"])

    assert audit["audit_ran"] is False
    assert audit["has_errors"] is None
    assert audit["error"]


def test_an_unreachable_audit_does_not_claim_the_logic_is_sound(offline):
    offline.setattr(llm_engine.requests, "post", _refuse)
    engine = _fast_engine()

    audit = engine.audit_logic(["one"])

    assert audit["audit_ran"] is False
    assert audit["has_errors"] is None


def test_a_successful_audit_is_marked_as_having_run(offline):
    body = '{"has_errors": true, "issues": ["step 2 assumes its conclusion"]}'
    offline.setattr(
        llm_engine.requests,
        "post",
        lambda *_a, **_k: _Response(200, {"message": {"content": body}}),
    )
    engine = _fast_engine()

    audit = engine.audit_logic(["one", "two"])

    assert audit["audit_ran"] is True
    assert audit["has_errors"] is True
    assert audit["issues"]


def test_an_unparseable_audit_still_reports_that_it_ran(offline):
    offline.setattr(
        llm_engine.requests,
        "post",
        lambda *_a, **_k: _Response(200, {"message": {"content": "looks good to me"}}),
    )
    engine = _fast_engine()

    audit = engine.audit_logic(["one"])

    assert audit["audit_ran"] is True
    assert audit["has_errors"] is False


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_the_rate_limit_counts_calls_that_actually_happened():
    policy = LLMPolicy(max_calls_per_minute=2, require_user_approval=False)

    for _ in range(2):
        allowed, _ = policy.can_call_llm(LLMCallType.GENERATION)
        assert allowed
        policy.record_call(LLMCallType.GENERATION, "hi", "there")

    allowed, reason = policy.can_call_llm(LLMCallType.GENERATION)

    assert allowed is False
    assert "rate limit" in reason.lower()
    assert policy.get_stats()["calls_this_minute"] == 2


def test_requesting_permission_twice_does_not_double_count():
    policy = LLMPolicy(max_calls_per_minute=5, require_user_approval=False)

    policy.request_llm_call(LLMCallType.GENERATION, "hi")
    policy.request_llm_call(LLMCallType.GENERATION, "hi")
    policy.record_call(LLMCallType.GENERATION, "hi", "there")

    assert policy.get_stats()["calls_this_minute"] == 1


# --------------------------------------------------------------------------
# Gateway paths
# --------------------------------------------------------------------------


class _StubEngine:
    def __init__(self, knowledge: str = "", generated: str = ""):
        self.config = LLMConfig(model="stub", use_fine_tuned=False)
        self.knowledge = knowledge
        self.generated = generated
        self.calls = []

    def query_knowledge(self, question, timeout=None):
        self.calls.append(("query_knowledge", timeout))
        if not self.knowledge:
            raise RuntimeError("no knowledge")
        return self.knowledge

    def parse_complex_input(self, text):
        self.calls.append(("parse_complex_input", None))
        return {"intent": "conversation"}

    def audit_logic(self, chain):
        self.calls.append(("audit_logic", None))
        return {"audit_ran": True, "has_errors": False}

    def generate(self, prompt, temperature=None, max_tokens=None, **kwargs):
        self.calls.append(("generate", temperature))
        return self.generated

    def chat(self, prompt, use_history=False, mode=None, **kwargs):
        self.calls.append(("chat", None))
        return "chat-response"


def _gateway(engine) -> LLMGateway:
    gateway = LLMGateway(llm_engine=engine, learning_engine=None)
    gateway.policy = LLMPolicy(max_calls_per_minute=1000, require_user_approval=False)
    gateway.model_router = None
    return gateway


def test_generation_no_longer_makes_the_two_round_trips_that_could_not_pay_off(monkeypatch):
    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "0")
    engine = _StubEngine(knowledge="")
    gateway = _gateway(engine)

    result = gateway.request(
        prompt="what is a monad",
        call_type=LLMCallType.GENERATION,
        user_input="what is a monad",
    )

    assert result.response == "chat-response"
    names = [name for name, _ in engine.calls]
    assert names == ["query_knowledge", "chat"]
    assert "parse_complex_input" not in names
    assert "audit_logic" not in names


def test_the_knowledge_assist_runs_on_a_shorter_leash_than_generation(monkeypatch):
    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "0")
    engine = _StubEngine(knowledge="a monad is a monoid in the category of endofunctors")
    gateway = _gateway(engine)

    result = gateway.request(
        prompt="what is a monad",
        call_type=LLMCallType.GENERATION,
        user_input="what is a monad",
    )

    assert result.response.startswith("a monad is")
    ((_, assist_timeout),) = [call for call in engine.calls if call[0] == "query_knowledge"]
    assert assist_timeout is not None
    assert assist_timeout < engine.config.timeout


def test_intent_classification_reaches_the_engine_instead_of_raising(monkeypatch):
    from ai.core.llm_intent_classifier import LLMIntentClassifier

    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "0")
    engine = _StubEngine(generated='{"intent": "notes", "action": "count", "confidence": 0.93}')
    gateway = _gateway(engine)

    result = LLMIntentClassifier(gateway).classify_with_cot("how many notes do i have")

    assert result is not None, "classification used to return None for every query"
    assert result.intent == "notes"
    assert result.confidence == pytest.approx(0.93)
    assert ("generate", None) in engine.calls


def test_self_consistency_passes_its_temperature_through_the_gateway(monkeypatch):
    from ai.core.llm_intent_classifier import LLMIntentClassifier

    monkeypatch.setenv("ALICE_MULTI_LLM_ROUTER", "0")
    engine = _StubEngine(generated='{"intent": "weather", "action": "current", "confidence": 0.9}')
    gateway = _gateway(engine)

    result = LLMIntentClassifier(gateway).classify_with_cot("is it raining", num_samples=3)

    assert result is not None
    assert result.intent == "weather"
    assert [call for call in engine.calls if call == ("generate", 0.7)]
