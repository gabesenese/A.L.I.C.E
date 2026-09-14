"""Conversation history should be the conversation, and nothing else.

`LocalLLMEngine.chat()` appends the prompt and the reply to
`conversation_history` unconditionally — including when the caller passed
`use_history=False`, which is precisely the flag that marks a call as machinery
rather than conversation.

Every internal prompt therefore lands in the transcript Alice replays to herself:
goal extraction, plan generation, learning prompts, greeting scaffolds, weather
wrapping, the response-variance engine, the training evaluators. On the next real
turn the model is shown that as "what we have been talking about", and does the
obvious thing — imitates its register. Instruction-shaped text in, instruction-
shaped text out.

That is a mechanical, three-line cause of "it feels like I'm talking to a
terminal", and it compounds: the more machinery runs, the more of Alice's
apparent conversational style is machinery talking to itself.

xfail(strict=True) — these document today's behaviour. When the appends are
gated they XPASS, pytest reports that as a failure, and the markers come off.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.xfail(
    strict=True, reason="chat() records machine prompts as conversation; see the module docstring"
)


def _chat_source() -> str:
    text = (PROJECT_ROOT / "ai" / "core" / "llm_engine.py").read_text(encoding="utf-8")
    start = text.index("    def chat(")
    end = text.index("\n    def ", start + 10)
    return text[start:end]


def test_history_is_only_recorded_when_the_turn_is_conversation():
    """The two appends sit at the end of chat() with no guard, so the flag that
    says "this is not conversation" controls reading but not writing."""
    body = _chat_source()
    appends = re.findall(r"self\.conversation_history\.append", body)
    assert appends, "expected chat() to record history somewhere"

    guarded = re.search(
        r"if\s+use_history\s*:[^\n]*\n(?:\s+.*\n)*?\s+self\.conversation_history\.append",
        body,
    )
    assert guarded, "conversation_history is appended unconditionally, ignoring use_history"


def test_internal_prompts_do_not_reach_the_transcript():
    """Enumerate the callers that mark themselves as machinery. Each one's prompt
    and reply currently becomes part of what Alice thinks was said to her."""
    machine_callers = []
    for path in PROJECT_ROOT.glob("ai/**/*.py"):
        if "test" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\.chat\((?:[^()]|\([^()]*\))*use_history\s*=\s*False", text, re.S):
            machine_callers.append(str(path.relative_to(PROJECT_ROOT)))

    body = _chat_source()
    records_unconditionally = "self.conversation_history.append" in body and not re.search(r"if\s+use_history", body)
    assert not (machine_callers and records_unconditionally), (
        f"{len(machine_callers)} internal callers pass use_history=False and are recorded anyway: "
        + ", ".join(sorted(machine_callers))
    )


def test_the_conversational_context_window_fits_its_own_preamble():
    """chat() asks for num_ctx 4096 while the system prompt, companion context and
    identity blocks alone run to well over a thousand tokens — so history is
    squeezed out and Alice loses the thread inside a single sitting. Every turn
    independent of the last is what a terminal is."""
    body = _chat_source()
    match = re.search(r'"num_ctx"\s*:\s*(\d+)', body)
    assert match, "chat() no longer sets num_ctx explicitly"
    assert int(match.group(1)) >= 8192, (
        f"num_ctx is {match.group(1)} on the conversational path; chat_with_tools already uses 8192"
    )
