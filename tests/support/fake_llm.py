"""A scripted stand-in for the local model, for tests that must not need Ollama.

Nothing in the suite could previously exercise a tool-calling turn: the only way
to make the model ask for a tool was to have Ollama running with a model loaded,
so the agent loop — the part of Alice that decides what to *do* — was covered
only where it declined to act.

`FakeLLM` speaks the two methods the runtime actually calls, `chat_with_tools`
and `chat`, and plays back a script of turns. Each entry is either text (a final
answer) or a list of tool calls, so a test can describe a whole reason/act/observe
sequence up front and then assert on what really ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ai.core.llm_engine import ChatResponse, ToolCall


@dataclass
class Turn:
    """One scripted model reply.

    Exactly one of ``content`` or ``tool_calls`` is meaningful: a reply that asks
    for tools ends the turn without an answer, which is how Ollama behaves.
    """

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


def answer(text: str) -> Turn:
    return Turn(content=text)


def calls(*specs: Union[tuple, Dict[str, Any]]) -> Turn:
    """Build a tool-calling turn from ``("tool_name", {...})`` pairs."""
    parsed: List[ToolCall] = []
    for spec in specs:
        if isinstance(spec, dict):
            parsed.append(ToolCall(name=str(spec["name"]), arguments=dict(spec.get("arguments") or {})))
        else:
            name, arguments = spec
            parsed.append(ToolCall(name=str(name), arguments=dict(arguments or {})))
    return Turn(tool_calls=parsed)


class FakeLLM:
    """Replays a fixed script and records everything it was asked.

    Attributes worth asserting on:
      ``calls``         — one entry per chat_with_tools round trip, each holding
                          the messages and the tool schemas it was offered.
      ``tools_offered`` — the tool names visible to the model on each round trip.
      ``observations``  — the tool-result messages fed back in, in order.
    """

    def __init__(
        self,
        script: Optional[Sequence[Turn]] = None,
        *,
        default: Optional[Turn] = None,
        on_call: Optional[Callable[[int, List[Dict[str, Any]]], Turn]] = None,
    ) -> None:
        self._script = list(script or [])
        self._default = default if default is not None else Turn(content="")
        self._on_call = on_call
        self.calls: List[Dict[str, Any]] = []
        self.chat_calls: List[Dict[str, Any]] = []

    # The runtime checks for this attribute before reaching for the agent loop.
    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        index = len(self.calls)
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": list(tools or []),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if self._on_call is not None:
            turn = self._on_call(index, [dict(m) for m in messages])
        elif index < len(self._script):
            turn = self._script[index]
        else:
            turn = self._default

        raw = {
            "message": {
                "content": turn.content,
                "tool_calls": [{"function": {"name": c.name, "arguments": dict(c.arguments)}} for c in turn.tool_calls],
            }
        }
        return ChatResponse(content=turn.content, tool_calls=list(turn.tool_calls), raw=raw)

    def chat(self, user_input: str, **kwargs: Any) -> str:
        self.chat_calls.append({"user_input": user_input, **kwargs})
        return f"[plain chat] {user_input}"

    # -- assertions helpers -------------------------------------------------

    @property
    def tools_offered(self) -> List[List[str]]:
        names = []
        for call in self.calls:
            names.append([str((t.get("function") or {}).get("name") or "") for t in call["tools"]])
        return names

    @property
    def observations(self) -> List[str]:
        """Every tool result the model was ultimately shown.

        Each round trip resends the whole history, so the last call holds them all.
        """
        if not self.calls:
            return []
        return self.observations_at(len(self.calls) - 1)

    def observations_at(self, call_index: int) -> List[str]:
        """Tool results visible to the model on round trip ``call_index``."""
        if call_index >= len(self.calls):
            return []
        return [str(m.get("content") or "") for m in self.calls[call_index]["messages"] if m.get("role") == "tool"]


class UnreachableLLM:
    """A model that always fails, for testing the Ollama-is-down paths."""

    def __init__(self, exc: Optional[BaseException] = None) -> None:
        self._exc = exc or ConnectionError("Ollama is not running")
        self.calls = 0

    def chat_with_tools(self, *args: Any, **kwargs: Any) -> ChatResponse:
        self.calls += 1
        raise self._exc

    def chat(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise self._exc
