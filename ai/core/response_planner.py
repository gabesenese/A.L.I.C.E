"""
Response Planning Layer.

Plans the structure, type, and strategy of a response BEFORE generation.
This reduces probabilistic drift by injecting response structure as constraints
into the system prompt, replacing one-shot generation with structured intent.

Pipeline position:  Executive Decision → Response Plan → LLM Context Build → Generate
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


RESPONSE_TYPES = (
    "factual",
    "explanation",
    "instruction",
    "debugging",
    "conversational",
    "planning",
    "troubleshooting",
)

RESPONSE_STRATEGIES = (
    "answer_directly",
    "guided_explanation",
    "incremental_teaching",
    "ask_guiding_question",
    "simplify",
    "expand",
    "verify_understanding",
)


@dataclass
class ResponsePlan:
    response_type: str   # one of RESPONSE_TYPES
    strategy: str        # one of RESPONSE_STRATEGIES
    outline: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)
    plan_depth: int = 1
    goal_context: str = ""
    format_hint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "response_type": self.response_type,
            "strategy": self.strategy,
            "outline": list(self.outline),
            "constraints": list(self.constraints),
            "required_sections": list(self.required_sections),
            "plan_depth": int(self.plan_depth),
            "goal_context": self.goal_context,
            "format_hint": self.format_hint,
        }

    def to_prompt_injection(self) -> str:
        """Render plan as a compact system-level constraint for prompt injection."""
        lines = [
            "Response plan (internal):",
            f"- type: {self.response_type}",
            f"- strategy: {self.strategy}",
        ]
        if self.outline:
            lines.append(f"- outline: {' -> '.join(self.outline)}")
        if self.constraints:
            lines.append(f"- constraints: {'; '.join(self.constraints)}")
        if self.required_sections:
            lines.append(f"- required_sections: {'; '.join(self.required_sections)}")
        lines.append(f"- plan_depth: {int(self.plan_depth)}")
        if self.goal_context:
            lines.append(f"- goal_context: {self.goal_context}")
        if self.format_hint:
            lines.append(f"- format: {self.format_hint}")
        return "\n".join(lines)


class ResponsePlanner:
    """
    Lightweight pre-generation planner that decides response structure before
    the LLM is called. Reduces randomness by making the following explicit:

    1. What TYPE of response is needed (explanation, instruction, debugging…)
    2. What STRATEGY should be used (teach gradually, answer directly…)
    3. What OUTLINE should be followed
    4. What CONSTRAINTS apply (goal alignment, format, no tangents)
    """

    _QUESTION_WORDS = frozenset((
        "what", "why", "how", "when", "where", "who", "which",
        "explain", "describe", "tell me",
    ))
    _DEBUG_WORDS = frozenset((
        "bug", "error", "exception", "crash", "fail", "broken",
        "not working", "debug", "fix", "issue", "traceback",
    ))
    _INSTRUCTION_WORDS = (
        "how to", "steps to", "guide", "tutorial",
        "walk me through", "show me how", "instructions",
        "step by step",
    )
    _PLAN_WORDS = frozenset((
        "plan", "strategy", "approach", "roadmap",
        "design", "architect", "structure", "organize",
    ))
    _TROUBLE_WORDS = (
        "troubleshoot", "not working", "wrong", "incorrect",
        "resolve", "solve", "problem with",
    )

    def plan(
        self,
        user_input: str,
        intent: str,
        reasoning_state: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> ResponsePlan:
        low = (user_input or "").lower()
        resp_type = self._detect_type(low, intent or "")
        strategy = self._select_strategy(resp_type, reasoning_state, conversation_state)
        outline = self._build_outline(resp_type, strategy, reasoning_state)
        constraints = self._build_constraints(resp_type, strategy, reasoning_state)
        required_sections = self._required_sections(resp_type, strategy)
        goal_context = str(
            reasoning_state.get("user_goal")
            or conversation_state.get("user_goal")
            or ""
        )
        format_hint = self._format_hint(resp_type)
        plan_depth = self._plan_depth(reasoning_state, conversation_state)
        return ResponsePlan(
            response_type=resp_type,
            strategy=strategy,
            outline=outline,
            constraints=constraints,
            required_sections=required_sections,
            plan_depth=plan_depth,
            goal_context=goal_context,
            format_hint=format_hint,
        )

    # ------------------------------------------------------------------
    # Internal detection helpers
    # ------------------------------------------------------------------

    def _detect_type(self, low_input: str, intent: str) -> str:
        intent_low = (intent or "").lower()

        # Debug/fix signals take priority
        if any(w in low_input for w in self._DEBUG_WORDS):
            return "debugging"

        # Troubleshooting phrases
        if any(phrase in low_input for phrase in self._TROUBLE_WORDS):
            return "troubleshooting"

        # Step-by-step instruction requests
        if any(phrase in low_input for phrase in self._INSTRUCTION_WORDS):
            return "instruction"

        # Planning / architecture
        if any(w in low_input for w in self._PLAN_WORDS):
            return "planning"

        # Conversational / chitchat
        if intent_low.startswith("conversation:") or intent_low.startswith("chitchat"):
            return "conversational"

        # Explanation-level questions
        if intent_low.startswith("learning:") or any(w in low_input for w in self._QUESTION_WORDS):
            return "explanation"

        # Direct factual question
        if intent_low.startswith("question:"):
            return "factual"

        return "factual"

    def _select_strategy(
        self,
        resp_type: str,
        reasoning_state: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> str:
        depth = int(
            reasoning_state.get("depth_level")
            or conversation_state.get("depth_level")
            or 0
        )
        goal = str(reasoning_state.get("user_goal") or conversation_state.get("user_goal") or "")
        conv_goal = str(
            reasoning_state.get("conversation_goal")
            or conversation_state.get("conversation_goal")
            or ""
        )
        conf = float(reasoning_state.get("confidence") or 0.0)

        # Conversational and action-oriented types are always direct
        if resp_type in ("conversational", "debugging", "troubleshooting", "instruction"):
            return "answer_directly"

        # Learning context → teach progressively
        if conv_goal == "learning" or "learn" in goal.lower() or "understand" in goal.lower():
            if depth >= 3:
                return "incremental_teaching"
            return "guided_explanation"

        # Planning needs expansion
        if resp_type == "planning":
            return "expand"

        # Low confidence → ask a guiding question before diving in
        if conf < 0.45:
            return "ask_guiding_question"

        return "answer_directly"

    def _build_outline(
        self,
        resp_type: str,
        strategy: str,
        reasoning_state: Dict[str, Any],
    ) -> List[str]:
        depth = int(reasoning_state.get("depth_level") or 0)

        if resp_type == "debugging":
            return ["identify root cause", "provide fix", "explain why it happens"]
        if resp_type == "troubleshooting":
            return ["confirm problem statement", "suggest fix", "verify expected outcome"]
        if resp_type == "instruction":
            return ["state the goal", "numbered steps", "expected result"]
        if resp_type == "planning":
            return ["define objective", "list steps", "flag dependencies", "surface risks"]
        if resp_type == "explanation":
            if depth >= 3:
                return ["brief recap", "detailed explanation", "concrete example", "deeper insight"]
            return ["brief answer", "explanation", "example"]
        if resp_type == "conversational":
            return ["direct reply"]
        # factual
        if depth == 0:
            return ["direct answer", "supporting detail"]
        return ["answer", "context", "example"]

    def _build_constraints(
        self,
        resp_type: str,
        strategy: str,
        reasoning_state: Dict[str, Any],
    ) -> List[str]:
        goal = str(reasoning_state.get("user_goal") or "")
        constraints: List[str] = []

        if goal:
            constraints.append(f"keep response aligned with goal: {goal}")
        if strategy == "incremental_teaching":
            constraints.append("build from simple to complex")
            constraints.append("include at least one concrete example")
        if strategy == "guided_explanation":
            constraints.append("explain clearly and then ask a check question")
        if strategy == "incremental_teaching":
            constraints.append("teach in progressive levels with concrete examples")
        if strategy == "verify_understanding":
            constraints.append("end with a brief comprehension check")
        if strategy == "expand":
            constraints.append("add depth beyond the obvious answer")
        if resp_type in ("debugging", "troubleshooting"):
            constraints.append("be specific, avoid generic advice")
        if resp_type == "instruction":
            constraints.append("use numbered steps")
        if resp_type == "planning":
            constraints.append("be actionable and concrete")

        constraints.append("stay on topic; do not add unrelated tangents")
        return constraints

    def _required_sections(self, resp_type: str, strategy: str) -> List[str]:
        sections: List[str] = ["answer"]
        if resp_type == "explanation":
            sections.extend(["explanation", "example"])
        elif resp_type in ("instruction", "troubleshooting"):
            sections.extend(["steps", "expected_result"])
        elif resp_type == "debugging":
            sections.extend(["root_cause", "fix"])
        elif resp_type == "planning":
            sections.extend(["plan", "risks"])

        if strategy in ("guided_explanation", "incremental_teaching", "verify_understanding"):
            sections.append("check_understanding")

        # Keep deterministic order and uniqueness.
        seen = set()
        ordered: List[str] = []
        for sec in sections:
            if sec not in seen:
                seen.add(sec)
                ordered.append(sec)
        return ordered

    def _plan_depth(
        self,
        reasoning_state: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> int:
        depth = int(
            reasoning_state.get("depth_level")
            or conversation_state.get("depth_level")
            or 0
        )
        if depth <= 1:
            return 1
        if depth <= 3:
            return 2
        return 3

    def _format_hint(self, resp_type: str) -> str:
        hints = {
            "instruction": "numbered list",
            "debugging": "code block if applicable, then explanation",
            "troubleshooting": "numbered steps",
            "planning": "structured outline",
            "explanation": "prose with example",
            "conversational": "natural prose",
            "factual": "concise prose",
        }
        return hints.get(resp_type, "prose")


_response_planner: ResponsePlanner | None = None


def get_response_planner() -> ResponsePlanner:
    global _response_planner
    if _response_planner is None:
        _response_planner = ResponsePlanner()
    return _response_planner
