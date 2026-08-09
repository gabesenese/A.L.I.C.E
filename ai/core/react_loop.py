"""Multi-step reason/act/observe loop over the tool catalog.

The previous agent loop selected one step from a keyword table, ran it if it was on
a read-only allowlist, and returned. There was no observation fed back to the model
and no second step, so nothing could be figured out that took more than one action.

This loop calls the model with the tool surface, executes what it asks for, feeds
the real result back, and repeats until the model answers or a budget is spent.
Only read-tier tools run unattended; anything that writes or reaches outward stops
the loop and reports what it wants to do.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.core import tool_catalog as catalog

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6
DEFAULT_DEADLINE_SECONDS = 90.0
MAX_OBSERVATION_CHARS = 4000

SYSTEM_PROMPT = (
    "You are Alice, operating on Gabriel's machine. You have tools that read the real "
    "filesystem and real services.\n"
    "Rules:\n"
    "1. If a question can be answered by a tool, call the tool. Never guess file names, "
    "file contents, note contents, or system values.\n"
    "2. After a tool returns, answer from what it actually returned. If it returned nothing "
    "useful, say so plainly.\n"
    "3. Chain tools when needed: list or search first, then read what you found.\n"
    "4. Answer in at most four sentences unless asked for detail. No preamble, no restating "
    "the question, no offers to help further.\n"
    "5. Never mention tools, tool calls, or how you obtained the information. State the finding "
    "directly, as if you simply looked."
)


@dataclass
class LoopStep:
    index: int
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    summary: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "success": self.success,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class ReactResult:
    answer: str = ""
    steps: List[LoopStep] = field(default_factory=list)
    used_tools: bool = False
    stopped_reason: str = ""
    pending_approval: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "used_tools": self.used_tools,
            "stopped_reason": self.stopped_reason,
            "pending_approval": dict(self.pending_approval),
            "tool_names": [s.tool for s in self.steps],
        }


def _observation_text(execution: catalog.ToolExecution) -> str:
    if not execution.success:
        return f"ERROR: {execution.error or 'tool failed'}"
    data = execution.data or {}
    if execution.tool == "list_workspace_files":
        files = data.get("files") or []
        header = f"{data.get('total_files', len(files))} files under {data.get('scope', '.')}"
        return f"{header}\n" + "\n".join(files)
    if execution.tool == "read_workspace_file":
        return f"{data.get('path', '')} ({data.get('line_count', 0)} lines)\n{data.get('content', '')}"
    if execution.tool == "search_workspace":
        lines = [f"{m['path']}:{m['line']}: {m['text']}" for m in (data.get("matches") or [])]
        return f"{data.get('match_count', 0)} matches\n" + "\n".join(lines)
    return execution.summary or str(data)


class ReactLoop:
    def __init__(
        self,
        llm_engine: Any,
        plugin_manager: Any = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
        allow_write_tools: bool = False,
    ) -> None:
        self.llm = llm_engine
        self.plugin_manager = plugin_manager
        self.max_steps = max(1, int(max_steps))
        self.deadline_seconds = float(deadline_seconds)
        self.allow_write_tools = bool(allow_write_tools)

    def run(
        self,
        user_input: str,
        context: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
    ) -> ReactResult:
        max_risk = catalog.RISK_WRITE if self.allow_write_tools else catalog.RISK_READ
        tools = catalog.build_tool_schemas(names=tool_names, max_risk=catalog.RISK_OUTWARD)
        if not tools:
            return ReactResult(stopped_reason="no_tools_available")

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context and str(context).strip():
            messages.append({"role": "system", "content": str(context).strip()})
        messages.append({"role": "user", "content": str(user_input or "")})

        result = ReactResult()
        started = time.perf_counter()
        seen_calls: set[tuple] = set()

        for index in range(1, self.max_steps + 1):
            if time.perf_counter() - started > self.deadline_seconds:
                result.stopped_reason = "deadline_exceeded"
                break

            try:
                response = self.llm.chat_with_tools(messages, tools=tools)
            except Exception as exc:
                logger.warning("React loop model call failed: %s", exc)
                result.stopped_reason = "model_error"
                break

            if not response.tool_calls:
                result.answer = response.content
                result.stopped_reason = result.stopped_reason or "answered"
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": list((response.raw.get("message") or {}).get("tool_calls") or []),
                }
            )

            for call in response.tool_calls:
                spec = catalog.get_spec(call.name)
                if spec is None:
                    step = LoopStep(index=index, tool=call.name, arguments=call.arguments, error="unknown tool")
                    result.steps.append(step)
                    messages.append({"role": "tool", "name": call.name, "content": f"ERROR: unknown tool {call.name}"})
                    continue

                if spec.risk != catalog.RISK_READ and not self.allow_write_tools:
                    result.pending_approval = {
                        "tool": spec.name,
                        "arguments": dict(call.arguments),
                        "risk": spec.risk,
                    }
                    result.stopped_reason = "approval_required"
                    return result

                execution = catalog.execute_tool(
                    call.name,
                    call.arguments,
                    plugin_manager=self.plugin_manager,
                )
                result.used_tools = True
                result.steps.append(
                    LoopStep(
                        index=index,
                        tool=spec.name,
                        arguments=dict(call.arguments),
                        success=execution.success,
                        summary=execution.summary,
                        error=execution.error,
                    )
                )

                signature = (spec.name, tuple(sorted(call.arguments.items(), key=lambda kv: str(kv[0]))))
                if signature in seen_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "name": spec.name,
                            "content": "You already called this tool with these arguments. Answer from the earlier result.",
                        }
                    )
                    continue
                seen_calls.add(signature)

                messages.append(
                    {
                        "role": "tool",
                        "name": spec.name,
                        "content": _observation_text(execution)[:MAX_OBSERVATION_CHARS],
                    }
                )
        else:
            result.stopped_reason = "step_budget_exhausted"

        if not result.answer and result.steps and result.stopped_reason != "approval_required":
            result.answer = self._final_answer(messages)

        _ = max_risk
        return result

    def _final_answer(self, messages: List[Dict[str, Any]]) -> str:
        """Ask for a plain answer once the budget stops further tool use."""
        closing = list(messages) + [
            {
                "role": "system",
                "content": "Answer now from the tool results above, in at most four sentences. Do not call more tools.",
            }
        ]
        try:
            return self.llm.chat_with_tools(closing, tools=None).content
        except Exception:
            return ""
