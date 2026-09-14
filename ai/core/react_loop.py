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

from ai.core import persona
from ai.core import tool_catalog as catalog
from ai.runtime.response_discipline import strip_speaker_label
from ai.runtime.trust_tiers import TIER_CONFIRM, TIER_REFUSE, classify

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6
DEFAULT_DEADLINE_SECONDS = 90.0
# Six observations at 4000 characters is roughly 6000 tokens, against a num_ctx
# of 8192 shared with the system prompt and ~700 tokens of tool schemas. A long
# chain overflowed, and Ollama trims from the front — which is where the
# grounding rules live. The tail of a file dump is worth less than "never guess",
# so the observation budget gives way first.
MAX_OBSERVATION_CHARS = 2500

# The loop used to run on six numbered rules with no name and no memory, so once
# ordinary turns started reaching it, most substantive questions were answered by
# a voice that knew nothing about the user. It is the same Alice as the
# conversational path now — same character text, byte for byte — narrowed to what
# is different about a turn that can go and look. See ai/core/persona.py.
SYSTEM_PROMPT = persona.for_tools()


@dataclass
class LoopStep:
    index: int
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    summary: str = ""
    error: str = ""
    productive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "success": self.success,
            "summary": self.summary,
            "error": self.error,
            "productive": self.productive,
        }


def _is_productive(execution: catalog.ToolExecution) -> bool:
    """Whether the call actually found something worth grounding an answer in.

    A lookup that returns nothing must not hijack the reply. Without this, a passing
    remark sends the model to a tool, the tool finds nothing, and the whole turn
    becomes "there are no notes about that".
    """
    if not execution.success:
        return False
    data = execution.data or {}
    for key in ("total_files", "match_count", "line_count"):
        if key in data:
            return int(data.get(key) or 0) > 0
    if "content" in data:
        return bool(str(data.get("content") or "").strip())
    return bool(str(execution.summary or "").strip())


@dataclass
class ReactResult:
    answer: str = ""
    steps: List[LoopStep] = field(default_factory=list)
    used_tools: bool = False
    stopped_reason: str = ""
    pending_approval: Dict[str, Any] = field(default_factory=dict)
    refused: Dict[str, Any] = field(default_factory=dict)
    checkpoint: str = ""
    # Set when tools ran but no closing answer could be composed from them. Kept
    # apart from stopped_reason so it cannot mask why the loop stopped in the
    # first place, which is the more useful fact.
    final_answer_failed: bool = False

    @property
    def produced_evidence(self) -> bool:
        return any(step.productive for step in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "used_tools": self.used_tools,
            "produced_evidence": self.produced_evidence,
            "stopped_reason": self.stopped_reason,
            "pending_approval": dict(self.pending_approval),
            "refused": dict(self.refused),
            "checkpoint": self.checkpoint,
            "final_answer_failed": self.final_answer_failed,
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
        checkpoint_writes: bool = True,
    ) -> None:
        self.llm = llm_engine
        self.plugin_manager = plugin_manager
        self.max_steps = max(1, int(max_steps))
        self.deadline_seconds = float(deadline_seconds)
        self.allow_write_tools = bool(allow_write_tools)
        self.checkpoint_writes = bool(checkpoint_writes)

    def run(
        self,
        user_input: str,
        context: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
    ) -> ReactResult:
        # Advertise only what this loop is actually allowed to run. The ceiling was
        # computed and then thrown away in favour of RISK_OUTWARD, so with writes
        # disabled the model was still shown write and outward tools, asked for one,
        # and had the whole turn discarded as "approval_required" — the user's
        # request vanished and they got ordinary chat back instead.
        max_risk = catalog.RISK_WRITE if self.allow_write_tools else catalog.RISK_READ
        tools = catalog.build_tool_schemas(names=tool_names, max_risk=max_risk)
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
                result.answer = strip_speaker_label(response.content)
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

                decision = classify(spec.name, call.arguments)
                if decision.tier == TIER_REFUSE:
                    result.refused = {
                        "tool": spec.name,
                        "arguments": dict(call.arguments),
                        "reason": decision.reason,
                    }
                    result.stopped_reason = "refused"
                    return result

                if decision.tier == TIER_CONFIRM or (spec.risk != catalog.RISK_READ and not self.allow_write_tools):
                    result.pending_approval = {
                        "tool": spec.name,
                        "arguments": dict(call.arguments),
                        "risk": spec.risk,
                        "reason": decision.reason,
                        "summary": decision.summary,
                    }
                    result.stopped_reason = "approval_required"
                    return result

                # Check for a repeat before running it, not after. Executing first
                # and then saying "you already called this" still ran the command a
                # second time, which for run_command means the side effect happened
                # twice while the model was told to ignore the result.
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

                if spec.risk != catalog.RISK_READ:
                    self._ensure_checkpoint(result)

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
                        productive=_is_productive(execution),
                    )
                )

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
            result.final_answer_failed = not result.answer

        return result

    def _ensure_checkpoint(self, result: ReactResult) -> None:
        """Stash a restore point before the first write of a turn.

        Reversibility is what makes unattended writes acceptable: a bad edit is
        undone rather than prevented by a prompt the user would learn to click past.
        """
        if result.checkpoint or not self.checkpoint_writes:
            return
        try:
            from ai.integration.git_manager import get_git_manager

            manager = get_git_manager()
            manager.resolve_repo_root()
            if not manager.has_changes().success:
                return
            created = manager.create_checkpoint("alice-agent-loop")
            if created.success:
                result.checkpoint = "stash@{0}"
        except Exception as exc:
            logger.warning("Could not create a checkpoint before writing: %s", exc)

    def _final_answer(self, messages: List[Dict[str, Any]]) -> str:
        """Ask for a plain answer once the budget stops further tool use."""
        closing = list(messages) + [
            {
                "role": "system",
                # No sentence cap here. This line sits in the strongest recency
                # position of the whole turn, and capping it at four sentences is
                # what turned a finding into a status line; the voice prompt
                # already governs length.
                "content": "Answer now from the tool results above. Do not call more tools.",
            }
        ]
        try:
            return strip_speaker_label(self.llm.chat_with_tools(closing, tools=None).content)
        except Exception as exc:
            # An empty answer here is indistinguishable from "the model had nothing
            # to say", and the caller treats it as a reason to fall back to plain
            # conversation. Say why in the log so a dead model is not read as one
            # that simply declined to answer.
            logger.warning("Could not compose a final answer from tool results: %s", exc)
            return ""
