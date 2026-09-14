"""
LLM Budget and Policy Manager for A.L.I.C.E
Controls when and how LLM calls are made to minimize dependency
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMCallType(Enum):
    """
    Types of LLM calls - Tool-Based Architecture
    Ollama is a TOOL Alice uses, not Alice herself
    """

    QUERY_KNOWLEDGE = "query_knowledge"  # Alice asks Ollama for factual information
    PARSE_INPUT = "parse_input"  # Alice asks Ollama to parse complex natural language
    PHRASE_RESPONSE = "phrase_response"  # Alice asks Ollama to phrase her thought naturally
    PHRASE_MICRO = "phrase_micro"  # Short polish only, no new content
    PHRASE_STRUCTURED = "phrase_structured"  # Rewrite structured payload only
    AUDIT_LOGIC = "audit_logic"  # Alice asks Ollama to verify her reasoning
    INTENT_CLASSIFICATION = "intent_classification"  # Alice asks Ollama to arbitrate an ambiguous intent

    # Legacy types (deprecated - will be removed)
    CHITCHAT = "chitchat"  # DEPRECATED: Use learned patterns instead
    TOOL_FORMATTING = "tool_format"  # DEPRECATED: Use phrase_response
    GENERATION = "generation"  # DEPRECATED: Use formulation → phrasing pattern
    CLARIFICATION = "clarification"  # DEPRECATED: Use parse_input
    FALLBACK = "fallback"  # DEPRECATED: Alice should always formulate first


@dataclass
class LLMCallRecord:
    """Record of an LLM call"""

    timestamp: datetime
    call_type: LLMCallType
    user_input: str
    llm_response: str
    approved_by_user: bool = False


@dataclass(frozen=True)
class LLMTransportPolicy:
    """Timeouts and retry budget for talking to the local model server.

    Probing and generating have opposite latency profiles: a tag listing either
    answers in milliseconds or not at all, while a 70B generation can
    legitimately run for a minute. Sharing one timeout between them makes a dead
    server look slow and a slow model look dead.
    """

    health_timeout: float = 2.0
    assist_timeout: float = 30.0
    generation_timeout: float = 90.0
    max_attempts: int = 3
    backoff_base: float = 0.25
    backoff_cap: float = 2.0
    autostart_wait: float = 3.0

    def backoff_seconds(self, attempt: int) -> float:
        """Delay before the retry that follows attempt number `attempt` (1-based)."""
        exponent = max(0, int(attempt) - 1)
        return min(float(self.backoff_cap), float(self.backoff_base) * (2.0**exponent))

    def should_retry_status(self, status_code: Any) -> bool:
        """A 4xx says the request itself is wrong, so resending it repeats the mistake."""
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            return False
        return 500 <= code <= 599


DEFAULT_TRANSPORT_POLICY = LLMTransportPolicy()


# The model runs on this machine. There is no bill and no shared quota, so a
# calls-per-minute cap protects nothing a human-paced session would hit — one
# turn can legitimately make several calls once the agent loop chains tools.
# While the counter was broken the limit was fictional and 10/min looked safe;
# with it counting, 10/min denies Alice the ability to think after two turns.
# What the ceiling is actually for is catching a runaway loop, so it sits well
# above any real conversation. Set ALICE_LLM_MAX_CALLS_PER_MINUTE to override;
# 0 or less disables the check.
DEFAULT_MAX_CALLS_PER_MINUTE = 120


def _configured_rate_limit(default: int) -> int:
    raw = str(os.getenv("ALICE_LLM_MAX_CALLS_PER_MINUTE", "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Ignoring non-numeric ALICE_LLM_MAX_CALLS_PER_MINUTE={raw!r}")
        return default


class LLMPolicy:
    """
    Global LLM budget and policy enforcement

    Purpose:
    - Minimize LLM dependency by enforcing strict limits
    - Require user approval for non-essential LLM calls
    - Track LLM usage to identify patterns that should be learned
    - Prevent LLM overuse that degrades performance
    """

    def __init__(
        self,
        max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
        allow_llm_for_chitchat: bool = False,
        allow_llm_for_tools: bool = False,
        allow_llm_for_generation: bool = True,
        require_user_approval: bool = True,
    ):
        """
        Initialize LLM policy

        Args:
            max_calls_per_minute: Maximum LLM calls per minute
            allow_llm_for_chitchat: Allow LLM for greetings/small talk
            allow_llm_for_tools: Allow LLM for formatting tool output
            allow_llm_for_generation: Allow LLM for complex generation
            require_user_approval: Require user approval before calling LLM
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.allow_llm_for_chitchat = allow_llm_for_chitchat
        self.allow_llm_for_tools = allow_llm_for_tools
        self.allow_llm_for_generation = allow_llm_for_generation
        self.require_user_approval = require_user_approval

        # Call tracking
        self.call_history: list[LLMCallRecord] = []
        self.calls_this_minute = 0
        self.last_minute_reset = datetime.now()

        # Statistics
        self.total_calls = 0
        self.denied_calls = 0
        self.approved_calls = 0

    def can_call_llm(self, call_type: LLMCallType, user_input: str = "") -> tuple[bool, str]:
        """
        Check if LLM call is allowed by policy

        Args:
            call_type: Type of LLM call
            user_input: User input that triggered the call

        Returns:
            (allowed, reason) tuple
        """
        # Reset per-minute counter if needed
        now = datetime.now()
        if (now - self.last_minute_reset).total_seconds() >= 60:
            self.calls_this_minute = 0
            self.last_minute_reset = now

        # Check rate limit. A ceiling of zero or less means no limit, which is the
        # sensible setting for a model running on this machine.
        if self.max_calls_per_minute > 0 and self.calls_this_minute >= self.max_calls_per_minute:
            self.denied_calls += 1
            return False, f"Rate limit exceeded ({self.max_calls_per_minute} calls/min)"

        # New tool-based call types are always allowed (Alice decides when to use tools)
        if call_type in [
            LLMCallType.QUERY_KNOWLEDGE,
            LLMCallType.PARSE_INPUT,
            LLMCallType.PHRASE_RESPONSE,
            LLMCallType.PHRASE_MICRO,
            LLMCallType.PHRASE_STRUCTURED,
            LLMCallType.AUDIT_LOGIC,
            LLMCallType.INTENT_CLASSIFICATION,
        ]:
            return True, "Allowed - Alice is using Ollama as a tool"

        # Legacy type checks (for backward compatibility during transition)
        if call_type == LLMCallType.CHITCHAT and not self.allow_llm_for_chitchat:
            self.denied_calls += 1
            return False, "LLM disabled for chitchat - use learned patterns"

        if call_type == LLMCallType.TOOL_FORMATTING and not self.allow_llm_for_tools:
            self.denied_calls += 1
            return False, "LLM disabled for tool formatting - use simple formatter"

        if call_type == LLMCallType.GENERATION and not self.allow_llm_for_generation:
            self.denied_calls += 1
            return False, "LLM disabled for generation"

        # All checks passed
        return True, "Allowed"

    def request_llm_call(
        self,
        call_type: LLMCallType,
        user_input: str,
        get_user_approval_func: Optional[callable] = None,
    ) -> tuple[bool, str]:
        """
        Request permission to call LLM

        Args:
            call_type: Type of LLM call
            user_input: User input that triggered the call
            get_user_approval_func: Function to get user approval (returns bool)

        Returns:
            (approved, reason) tuple
        """
        # Check policy
        allowed, reason = self.can_call_llm(call_type, user_input)
        if not allowed:
            logger.info(f"LLM call denied: {reason}")
            return False, reason

        # Check if user approval required
        if self.require_user_approval and get_user_approval_func:
            message = self._get_approval_message(call_type, user_input)
            approved = get_user_approval_func(message)

            if not approved:
                self.denied_calls += 1
                logger.info("LLM call denied by user")
                return False, "User declined LLM call"

            self.approved_calls += 1
            logger.info("LLM call approved by user")

        # Counting happens in record_llm_call, once the call has actually been made.
        # Counting approvals here instead left calls_this_minute at 0 for every
        # caller that goes straight through can_call_llm, so the rate limit never
        # tripped; counting in both places would double-count the callers that don't.
        return True, "Approved"

    def _get_approval_message(self, call_type: LLMCallType, user_input: str) -> str:
        """Generate user approval message"""
        messages = {
            # New tool-based call types
            LLMCallType.QUERY_KNOWLEDGE: "I need to query my knowledge base for factual information. Proceed?",
            LLMCallType.PARSE_INPUT: "I need help parsing this complex input. Proceed with analysis?",
            LLMCallType.PHRASE_RESPONSE: "I've formulated my response and need to phrase it naturally. Proceed?",
            LLMCallType.PHRASE_MICRO: "I need short phrasing polish only. Proceed?",
            LLMCallType.PHRASE_STRUCTURED: "I need a structured rewrite only. Proceed?",
            LLMCallType.AUDIT_LOGIC: "I want to verify my reasoning logic. Proceed with audit?",
            LLMCallType.INTENT_CLASSIFICATION: "I'm not sure what you're asking for. Should I work it out with AI?",
            # Legacy types (backward compatibility)
            LLMCallType.CHITCHAT: "I don't have a learned response for that. Would you like me to use AI to answer it? (This helps me learn!)",
            LLMCallType.TOOL_FORMATTING: "I have the information but need to format it nicely. Proceed with AI help?",
            LLMCallType.GENERATION: "That's something new for me. Want me to look it up with AI?",
            LLMCallType.CLARIFICATION: "I need a bit of AI help to understand that. Should I proceed?",
            LLMCallType.FALLBACK: "I need AI assistance for this one. Proceed?",
        }
        return messages.get(call_type, "Should I use AI for this?")

    def record_llm_call(
        self,
        call_type: LLMCallType,
        user_input: str,
        llm_response: str,
        approved_by_user: bool = False,
    ):
        """Record an LLM call for tracking, and count it against the rate limit."""
        record = LLMCallRecord(
            timestamp=datetime.now(),
            call_type=call_type,
            user_input=user_input,
            llm_response=llm_response,
            approved_by_user=approved_by_user,
        )
        self.call_history.append(record)

        now = record.timestamp
        if (now - self.last_minute_reset).total_seconds() >= 60:
            self.calls_this_minute = 0
            self.last_minute_reset = now
        self.calls_this_minute += 1
        self.total_calls += 1

        # Keep only last 100 records
        if len(self.call_history) > 100:
            self.call_history = self.call_history[-100:]

    def record_call(self, call_type: LLMCallType, user_input: str, llm_response: str):
        """Alias for record_llm_call (simplified interface for gateway)"""
        self.record_llm_call(call_type, user_input, llm_response, approved_by_user=False)

    def get_recent_calls(self, minutes: int = 60) -> list[LLMCallRecord]:
        """Get LLM calls from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [r for r in self.call_history if r.timestamp > cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        recent = self.get_recent_calls(60)

        return {
            "total_calls": self.total_calls,
            "approved_calls": self.approved_calls,
            "denied_calls": self.denied_calls,
            "calls_last_hour": len(recent),
            "calls_this_minute": self.calls_this_minute,
            "rate_limit": self.max_calls_per_minute,
            "policy": {
                "chitchat_allowed": self.allow_llm_for_chitchat,
                "tools_allowed": self.allow_llm_for_tools,
                "generation_allowed": self.allow_llm_for_generation,
                "requires_approval": self.require_user_approval,
            },
        }

    def suggest_patterns_to_learn(self) -> list[tuple[str, str]]:
        """
        Suggest patterns that should be learned from LLM calls

        Returns:
            List of (user_input, llm_response) tuples that appear frequently
        """
        from collections import Counter

        # Find frequently asked questions
        input_counts = Counter(r.user_input.lower() for r in self.call_history)

        # Suggest learning patterns for inputs asked 3+ times
        suggestions = []
        for user_input, count in input_counts.most_common():
            if count >= 3:
                # Find the most common LLM response for this input
                responses = [r.llm_response for r in self.call_history if r.user_input.lower() == user_input]
                most_common_response = Counter(responses).most_common(1)[0][0]
                suggestions.append((user_input, most_common_response))

        return suggestions


# Global singleton
_llm_policy_instance = None


def get_llm_policy() -> LLMPolicy:
    """Get global LLM policy instance"""
    global _llm_policy_instance
    if _llm_policy_instance is None:
        # Default: restrictive policy
        _llm_policy_instance = LLMPolicy(
            max_calls_per_minute=_configured_rate_limit(DEFAULT_MAX_CALLS_PER_MINUTE),
            allow_llm_for_chitchat=False,  # Use learned patterns
            allow_llm_for_tools=False,  # Use simple formatters
            allow_llm_for_generation=True,  # Allow for complex tasks
            require_user_approval=True,  # Ask before calling
        )
    return _llm_policy_instance


def configure_llm_policy(**kwargs):
    """Configure global LLM policy"""
    global _llm_policy_instance
    _llm_policy_instance = LLMPolicy(**kwargs)
    logger.info(f"LLM policy configured: {_llm_policy_instance.get_stats()}")


def configure_minimal_policy():
    """
    Configure minimal LLM policy - Alice learns organically

    Minimal mode philosophy:
    - Chitchat: Use learned patterns ONLY (no LLM)
    - Tool formatting: Use simple formatters ONLY (no LLM)
    - Generation: LLM allowed ONLY when user explicitly approves
    - Learning: All LLM responses are learning opportunities

    This forces Alice to build her own conversational style
    rather than relying on pre-programmed LLM responses.
    """
    global _llm_policy_instance
    _llm_policy_instance = LLMPolicy(
        max_calls_per_minute=_configured_rate_limit(30),  # Tight, but not so tight it stalls a turn
        allow_llm_for_chitchat=False,  # Never use LLM for chitchat - learn patterns
        allow_llm_for_tools=False,  # Never use LLM for tool formatting - use simple formatters
        allow_llm_for_generation=True,  # Allow LLM for complex generation (with approval)
        require_user_approval=True,  # Always ask before calling LLM
    )
    logger.info("Minimal LLM policy activated - Alice will learn organically")
    logger.info(f"Policy settings: {_llm_policy_instance.get_stats()}")
    return _llm_policy_instance
