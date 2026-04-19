"""Regression tests for code-request follow-up routing bridge."""

import pytest

from app.main import ALICE


class _StubSelfReflection:
    class _CodeFile:
        def __init__(self, path: str, name: str, lines: int, module_type: str, content: str):
            self.path = path
            self.name = name
            self.lines = lines
            self.module_type = module_type
            self.content = content

    def list_codebase(self):
        return [
            {"path": "models/fast_model.py", "module_type": "utility"},
            {"path": "brain/model_router.py", "module_type": "core"},
        ]

    def read_file(self, path):
        if path.endswith("nlp_processor.py"):
            return self._CodeFile(
                path="ai/core/nlp_processor.py",
                name="nlp_processor.py",
                lines=42,
                module_type="core",
                content="def process():\n    return True\n",
            )
        return None

    def batch_summarize_files(self, files, parallel=True):
        return {f: f"summary for {f}" for f in files}

    def generate_file_summary(self, path):
        return f"Summary of {path}"


def _build_alice_stub():
    alice = ALICE.__new__(ALICE)
    alice.self_reflection = _StubSelfReflection()
    alice.capabilities = {
        "codebase_access": {
            "available": True,
            "description": "I can access internal code.",
            "operations": ["list", "read", "search", "analyze"],
        }
    }
    alice.code_context = {
        "last_files_shown": [],
        "last_action": None,
        "timestamp": None,
        "file_count": 0,
    }
    alice._generate_natural_response = (
        lambda alice_response, tone, context, user_input: "CAPABILITY_OK"
    )
    return alice


def test_code_access_followup_list_it_routes_to_codebase_listing():
    alice = _build_alice_stub()

    first = ALICE._handle_code_request(alice, "do you have acess to your internal code?", {})
    second = ALICE._handle_code_request(alice, "list it to me", {})

    assert first == "CAPABILITY_OK"
    assert second == "CAPABILITY_OK"
    assert alice.code_context["file_count"] == 2
    assert "models/fast_model.py" in alice.code_context["last_files_shown"]
    assert alice.code_context["last_action"] == "list"


def test_code_access_followup_list_it_for_me_routes_to_codebase_listing():
    alice = _build_alice_stub()

    first = ALICE._handle_code_request(alice, "are you able to see your internal code?", {})
    second = ALICE._handle_code_request(alice, "list it for me", {})

    assert first == "CAPABILITY_OK"
    assert second == "CAPABILITY_OK"
    assert alice.code_context["file_count"] == 2
    assert alice.code_context["last_action"] == "list"


def test_code_access_phrase_are_you_able_to_see_routes_to_capability_answer():
    alice = _build_alice_stub()

    first = ALICE._handle_code_request(alice, "are you able to see your internal code?", {})

    assert first == "CAPABILITY_OK"
    assert alice.code_context["last_action"] == "code_access_confirmed"


@pytest.mark.parametrize(
    "prompt",
    [
        "could you inspect your source code?",
        "can you read your own codebase?",
        "would you be able to view your internal code?",
        "do you access your internal code?",
        "can you acess your internal code?",
        "can you check your local code?",
    ],
)
def test_code_access_capability_detection_generalizes_without_phrase_list(prompt):
    alice = _build_alice_stub()

    first = ALICE._handle_code_request(alice, prompt, {})

    assert first == "CAPABILITY_OK"
    assert alice.code_context["last_action"] == "code_access_confirmed"


def test_explicit_py_file_summary_bypasses_list_followup_summary_mode():
    alice = _build_alice_stub()
    alice.code_context["last_action"] = "list"
    alice.code_context["last_files_shown"] = [
        "models/fast_model.py",
        "brain/model_router.py",
    ]

    out = ALICE._handle_code_request(
        alice,
        "summarize the nlp_processor.py file for me",
        {},
    )

    assert out is not None
    assert "nlp_processor.py" in out.lower()
