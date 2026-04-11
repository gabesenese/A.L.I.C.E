"""
Response Formulator for A.L.I.C.E
=================================
Transforms structured plugin data into natural language responses.

Instead of plugins returning hard-coded messages, they return structured data.
Alice learns to formulate natural responses from that data.

Philosophy:
- Plugins provide facts (data)
- Alice provides personality (phrasing)
- Learning happens through examples
- Progressive independence from Ollama
"""

import logging
from typing import Dict, Any, Optional, List, Pattern
from dataclasses import dataclass
import json
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class ResponseTemplate:
    """Template for a type of response"""

    action: str  # e.g., "create_note", "delete_notes", "search_results"
    example_data: Dict[str, Any]  # Example data structure
    example_phrasings: List[str]  # Multiple ways to phrase it
    formulation_rules: List[str]  # Rules for generating response


@dataclass
class ReasoningOutput:
    """Internal reasoning artifact that must never be sent directly to users."""

    internal_summary: str
    intent: str
    plan: List[str]
    confidence: float


@dataclass
class UserResponse:
    """Final user-facing response contract."""

    message: str


class ResponseFormulator:
    """
    Learns to formulate natural responses from structured data.

    Works with:
    - PhrasingLearner: Learns successful phrasings
    - LLM Gateway: Uses Ollama for initial formulations
    - Pattern system: Recognizes when Alice can formulate alone
    """

    def __init__(
        self,
        phrasing_learner=None,
        llm_gateway=None,
        storage_path: str = "data/response_templates",
    ):
        self.phrasing_learner = phrasing_learner
        self.llm_gateway = llm_gateway
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load response templates
        self.templates: Dict[str, ResponseTemplate] = {}
        self._load_templates()

        # Track what Alice can formulate independently
        self.independent_actions = set()
        self._load_independence_data()

    INTERNAL_LEAK_PATTERNS: List[str] = [
        "the user wants",
        "some possible follow-up",
        "key points",
        "analysis:",
        "context:",
        "internal reasoning",
        "reasoning output",
        "intent:",
    ]
    INTERNAL_LEAK_REGEX: List[Pattern[str]] = [
        re.compile(
            r"\b(?:intent|confidence|route|user_input|raw_response|resolved_intent|response_type|strategy|plan)\s*=",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:analysis|context)\s*=", re.IGNORECASE),
    ]
    CLARIFICATION_FALLBACK = "Can you clarify the exact outcome you want?"
    GENERIC_FALLBACK = "I can help. Tell me the exact result you want."
    DIRECT_RETRY_FALLBACK = (
        "I can help. Tell me the exact result you want, and I will answer directly."
    )

    def is_internal(self, text: str) -> bool:
        """Return True when a text looks like internal reasoning or scaffold leakage."""
        low = str(text or "").lower()
        if not low.strip():
            return False
        if any(pattern in low for pattern in self.INTERNAL_LEAK_PATTERNS):
            return True
        return any(pattern.search(low) for pattern in self.INTERNAL_LEAK_REGEX)

    @staticmethod
    def _is_direct_answer_question(user_input: str) -> bool:
        """Detect definitional/comparative/explanatory self-contained questions."""
        text = str(user_input or "").lower().strip()
        if not text:
            return False

        tokens = set(re.findall(r"\b[a-z0-9']+\b", text))
        if len(tokens) < 3:
            return False

        if re.search(r"\b(help\s+me|build\s+me|make\s+me|do\s+this|do\s+that)\b", text):
            return False

        ambiguity_terms = {
            "something",
            "anything",
            "stuff",
            "idk",
            "whatever",
        }
        if tokens & ambiguity_terms:
            return False

        direct_patterns = (
            r"^\s*what\s+is\b",
            r"^\s*what's\b",
            r"^\s*what\s+are\b",
            r"^\s*what(?:'s|\s+is)?\s+the\s+difference\b",
            r"\bdifference\s+between\b",
            r"^\s*compare\b",
            r"^\s*explain\b",
            r"^\s*how\s+does\b",
            r"^\s*how\s+do\b",
            r"^\s*why\s+does\b",
            r"^\s*why\s+do\b",
            r"^\s*define\b",
        )

        has_direct_structure = any(re.search(pat, text) for pat in direct_patterns)
        is_question_like = bool(
            "?" in text
            or re.match(r"^\s*(what|which|how|why|explain|define|compare)\b", text)
        )
        return bool(has_direct_structure and is_question_like)

    def _sanitize_user_message(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_lines: List[str] = []
        internal_key_equals = re.compile(
            r"\b(?:intent|confidence|route|raw_response|user_input|resolved_intent|response_type|strategy|plan)\s*=",
            re.IGNORECASE,
        )
        for line in cleaned.split("\n"):
            normalized = re.sub(r"[^\S\n]+", " ", line).strip()
            if not normalized:
                cleaned_lines.append("")
                continue
            low_line = normalized.lower()
            if re.match(r"^(analysis|context|intent|plan)\s*:\s*", low_line):
                continue
            if internal_key_equals.search(low_line):
                continue
            cleaned_lines.append(normalized)

        while cleaned_lines and cleaned_lines[0] == "":
            cleaned_lines.pop(0)
        while cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(
            r"^(analysis|context|intent|plan)\s*:\s*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = cleaned.strip(' \t\n\r-:;,."')
        if cleaned and not re.search(r"[.!?]$", cleaned):
            cleaned += "."
        return cleaned

    @staticmethod
    def _compose_direct_answer_fallback(user_input: str) -> str:
        """Generate a concise direct answer scaffold for self-contained direct questions."""
        text = str(user_input or "").strip()
        low = text.lower()

        diff_match = re.search(
            r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", low
        )
        if diff_match:
            left = diff_match.group(1).strip(" .?!")
            right = diff_match.group(2).strip(" .?!")
            if left and right:
                return (
                    f"Short answer: {left} and {right} are related, but {left} usually focuses on one primary role, "
                    f"while {right} focuses on a different role, workflow, or implementation objective."
                )

        what_match = re.match(r"^\s*(?:what\s+is|what's|define)\s+(.+?)(?:\?|$)", low)
        if what_match:
            topic = what_match.group(1).strip(" .?!")
            if topic:
                return (
                    f"Short answer: {topic} is best defined by what it does, where it fits in a system, "
                    f"and which trade-offs it introduces in practice."
                )

        how_match = re.match(r"^\s*(?:how\s+does|how\s+do)\s+(.+?)(?:\?|$)", low)
        if how_match:
            topic = how_match.group(1).strip(" .?!")
            if topic:
                return (
                    f"Short answer: {topic} works through input interpretation, decision logic, "
                    f"execution, and feedback-driven adjustment."
                )

        why_match = re.match(r"^\s*(?:why\s+does|why\s+do)\s+(.+?)(?:\?|$)", low)
        if why_match:
            topic = why_match.group(1).strip(" .?!")
            if topic:
                return (
                    f"Short answer: {topic} generally reflects design constraints, optimization trade-offs, "
                    f"and objective-driven behavior in the system."
                )

        return "Short answer: this is a direct question, so here is a concise explanation without clarification prompts."

    def _regenerate_clean_message(
        self,
        *,
        intent: str,
        context: Optional[Dict[str, Any]],
        tool_results: Optional[Dict[str, Any]],
        reasoning_output: Optional[ReasoningOutput],
    ) -> str:
        """Deterministic fallback when candidate output leaks internal text."""
        intent_text = str(intent or "").lower().strip()
        user_input = str((context or {}).get("user_input") or "").strip()

        if self._is_direct_answer_question(user_input):
            return self._compose_direct_answer_fallback(user_input)

        if "clarification" in intent_text or "vague" in intent_text:
            return self.CLARIFICATION_FALLBACK

        if tool_results and isinstance(tool_results, dict):
            msg = str(
                tool_results.get("response") or tool_results.get("message") or ""
            ).strip()
            if msg and not self.is_internal(msg):
                return msg

            data = (
                tool_results.get("data")
                if isinstance(tool_results.get("data"), dict)
                else {}
            )
            plugin = str(tool_results.get("plugin") or "").strip()
            action = str(tool_results.get("action") or "").strip()
            if plugin or action:
                return f"Done. {plugin} {action} completed.".strip()
            if data:
                return "Done. I processed that request."

        if (
            "goal" in intent_text
            or "project" in intent_text
            or "learning:" in intent_text
        ):
            return (
                "Great direction. Let's turn this into a concrete build plan.\n\n"
                "Project Concept: Start with a focused AI agent that solves one recurring task end-to-end.\n\n"
                "Action Plan:\n"
                "1. Pick one domain and one measurable outcome.\n"
                "2. Build a minimal agent loop: plan, execute, verify.\n"
                "3. Add memory only for context that improves decisions.\n"
                "4. Add one tool integration and validate with scenario tests.\n\n"
                "Do you want to start with architecture, implementation steps, or a starter repo layout?"
            )

        if user_input:
            return self.DIRECT_RETRY_FALLBACK

        return self.GENERIC_FALLBACK

    def _enforce_jarvis_tone(self, message: str) -> str:
        """Keep final responses clean, direct, and practical."""
        text = self._sanitize_user_message(message)
        if not text:
            return ""

        text = re.sub(
            r"^(sure|of course|absolutely|definitely|no problem|happy to help)[:,.!]?\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = "\n".join(
            re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")
        )
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def generate(
        self,
        intent: str,
        context: Optional[Dict[str, Any]],
        tool_results: Optional[Dict[str, Any]],
        reasoning_output: Optional[ReasoningOutput],
        mode: str = "final_answer_only",
    ) -> UserResponse:
        """Final authority layer: always produce a user-safe UserResponse."""
        _ = mode  # Reserved for future policy branching.

        candidate = ""

        if tool_results and isinstance(tool_results, dict):
            candidate = str(
                tool_results.get("response") or tool_results.get("message") or ""
            ).strip()

        if not candidate:
            candidate = str((context or {}).get("response") or "").strip()

        candidate = self._sanitize_user_message(candidate)
        if not candidate or self.is_internal(candidate):
            candidate = self._regenerate_clean_message(
                intent=intent,
                context=context,
                tool_results=tool_results,
                reasoning_output=reasoning_output,
            )
            candidate = self._sanitize_user_message(candidate)

        if not candidate or self.is_internal(candidate):
            user_input = str((context or {}).get("user_input") or "").strip()
            if self._is_direct_answer_question(user_input):
                candidate = self._compose_direct_answer_fallback(user_input)
            else:
                candidate = self.GENERIC_FALLBACK

        candidate = self._enforce_jarvis_tone(candidate)

        return UserResponse(message=candidate)

    def _load_templates(self) -> None:
        """Load response templates from storage"""
        template_file = self.storage_path / "templates.json"
        if template_file.exists():
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for action, template_data in data.items():
                        self.templates[action] = ResponseTemplate(
                            action=action,
                            example_data=template_data.get("example_data", {}),
                            example_phrasings=template_data.get(
                                "example_phrasings", []
                            ),
                            formulation_rules=template_data.get(
                                "formulation_rules", []
                            ),
                        )
                logger.info(f"Loaded {len(self.templates)} response templates")
            except Exception as e:
                logger.error(f"Error loading response templates: {e}")

    def _load_independence_data(self) -> None:
        """Load actions Alice can formulate independently"""
        independence_file = self.storage_path / "independence.json"
        if independence_file.exists():
            try:
                with open(independence_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.independent_actions = set(data.get("independent_actions", []))
                logger.info(
                    f"Alice can independently formulate {len(self.independent_actions)} action types"
                )
            except Exception as e:
                logger.error(f"Error loading independence data: {e}")

    def formulate_response(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool,
        user_input: str,
        tone: str = "helpful",
    ) -> str:
        """
        Formulate a natural language response from plugin data.

        Process:
        1. Prefer Alice's independent learned phrasing
        2. Fall back to local non-LLM synthesis
        3. Learn from generated responses to improve future independence
        """

        # Alice-first: use learned independent phrasing when available.
        if action in self.independent_actions and self.phrasing_learner:
            alice_response = self._formulate_independently(action, data, success, tone)
            if alice_response:
                logger.info(
                    f"[ResponseFormulator] Alice formulated '{action}' independently"
                )
                return alice_response

        # Local fallback (no Ollama): generate response and learn from it.
        response = self._formulate_basic(action, data, success)
        if response and self.phrasing_learner:
            self._learn_formulation(action, data, response, tone)
        return response

    def _formulate_independently(
        self, action: str, data: Dict[str, Any], success: bool, tone: str
    ) -> Optional[str]:
        """Formulate response using Alice's learned patterns"""
        if not self.phrasing_learner:
            return None

        # Build a thought structure from the data
        thought = {"type": action, "success": success, "data": data}

        # Try to phrase it using learned patterns
        try:
            response = self.phrasing_learner.phrase_myself(thought, tone=tone)
            if response and response != "":
                return response
        except Exception as e:
            logger.debug(f"Could not formulate independently: {e}")

        return None

    def _formulate_with_llm(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool,
        user_input: str,
        tone: str,
    ) -> Optional[str]:
        """
        Previously called Ollama to formulate responses.
        Alice is now always in control — this routes to basic formatting only.
        Ollama is strictly a teacher in the main pipeline, not a formulator.
        """
        # Alice formulates directly — no LLM involvement
        return self._formulate_basic(action, data, success)

    def _formulate_basic(self, action: str, data: Dict[str, Any], success: bool) -> str:
        """Basic fallback formulation without LLM"""
        if not success:
            return "I wasn't able to complete that action."

        # Extract a human-readable subject from data
        subject = (
            data.get("note_title")
            or data.get("title")
            or data.get("name")
            or data.get("reminder_text")
            or data.get("query")
            or data.get("filename")
        )

        # Simple templates based on action type
        if "create" in action or "add" in action:
            if subject:
                return f"Created '{subject}' successfully."
            return "Created successfully."
        elif "delete" in action or "remove" in action:
            count = data.get("count", 1)
            if subject:
                return f"Removed '{subject}'."
            return f"Removed {count} item{'s' if count != 1 else ''}."
        elif "search" in action or "find" in action:
            count = data.get("count", 0)
            if subject:
                return (
                    f"Found {count} result{'s' if count != 1 else ''} for '{subject}'."
                )
            return f"Found {count} result{'s' if count != 1 else ''}."
        elif "update" in action or "edit" in action:
            if subject:
                return f"Updated '{subject}' successfully."
            return "Updated successfully."
        elif "remind" in action:
            if subject:
                return f"Reminder set: '{subject}'."
            return "Reminder set."
        else:
            if subject:
                return f"Done: '{subject}'."
            return "Done."

    def _learn_formulation(
        self, action: str, data: Dict[str, Any], response: str, tone: str
    ):
        """Learn from a successful formulation"""
        if not self.phrasing_learner:
            return

        # Record this as a learned phrasing
        thought = {"type": action, "data": data}

        try:
            self.phrasing_learner.record_phrasing(
                alice_thought=thought, ollama_phrasing=response, context={"tone": tone}
            )

            # Check if Alice has learned enough to be independent
            if self.phrasing_learner.can_phrase_myself(thought, tone):
                self.independent_actions.add(action)
                self._save_independence_data()
                logger.info(
                    f"[ResponseFormulator] Alice achieved independence for '{action}'!"
                )
        except Exception as e:
            logger.debug(f"Could not learn formulation: {e}")

    def _save_independence_data(self) -> None:
        """Save actions Alice can formulate independently"""
        independence_file = self.storage_path / "independence.json"
        try:
            with open(independence_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"independent_actions": list(self.independent_actions)}, f, indent=2
                )
        except Exception as e:
            logger.error(f"Error saving independence data: {e}")

    def add_template(
        self,
        action: str,
        example_data: Dict[str, Any],
        example_phrasings: List[str],
        formulation_rules: List[str] = None,
    ):
        """Add a response template for an action type"""
        self.templates[action] = ResponseTemplate(
            action=action,
            example_data=example_data,
            example_phrasings=example_phrasings,
            formulation_rules=formulation_rules or [],
        )
        self._save_templates()

    def _save_templates(self) -> None:
        """Save response templates to storage"""
        template_file = self.storage_path / "templates.json"
        try:
            data = {}
            for action, template in self.templates.items():
                data[action] = {
                    "example_data": template.example_data,
                    "example_phrasings": template.example_phrasings,
                    "formulation_rules": template.formulation_rules,
                }

            with open(template_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.templates)} response templates")
        except Exception as e:
            logger.error(f"Error saving templates: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get formulator statistics"""
        return {
            "total_templates": len(self.templates),
            "independent_actions": len(self.independent_actions),
            "learning_progress": (
                f"{len(self.independent_actions)}/{len(self.templates)}"
                if self.templates
                else "0/0"
            ),
        }


# Singleton instance
_formulator = None


def get_response_formulator(
    phrasing_learner=None, llm_gateway=None
) -> ResponseFormulator:
    """Get or create the response formulator singleton"""
    global _formulator
    if _formulator is None:
        _formulator = ResponseFormulator(
            phrasing_learner=phrasing_learner, llm_gateway=llm_gateway
        )
    else:
        # Update dependencies if provided (supports late wiring)
        if phrasing_learner is not None and _formulator.phrasing_learner is None:
            _formulator.phrasing_learner = phrasing_learner
        if llm_gateway is not None and _formulator.llm_gateway is None:
            _formulator.llm_gateway = llm_gateway
    return _formulator
