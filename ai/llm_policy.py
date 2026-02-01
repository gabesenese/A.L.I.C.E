"""
LLM Budget and Policy Manager for A.L.I.C.E
Controls when and how LLM calls are made to minimize dependency
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMCallType(Enum):
    """Types of LLM calls"""
    CHITCHAT = "chitchat"           # Casual conversation
    TOOL_FORMATTING = "tool_format" # Formatting tool output
    GENERATION = "generation"       # Creative/complex generation
    CLARIFICATION = "clarification" # Asking for clarification
    FALLBACK = "fallback"          # Last resort fallback


@dataclass
class LLMCallRecord:
    """Record of an LLM call"""
    timestamp: datetime
    call_type: LLMCallType
    user_input: str
    llm_response: str
    approved_by_user: bool = False


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
        max_calls_per_minute: int = 10,
        allow_llm_for_chitchat: bool = False,
        allow_llm_for_tools: bool = False,
        allow_llm_for_generation: bool = True,
        require_user_approval: bool = True
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
    
    def can_call_llm(
        self,
        call_type: LLMCallType,
        user_input: str = ""
    ) -> tuple[bool, str]:
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
        
        # Check rate limit
        if self.calls_this_minute >= self.max_calls_per_minute:
            self.denied_calls += 1
            return False, f"Rate limit exceeded ({self.max_calls_per_minute} calls/min)"
        
        # Check type-specific policies
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
        get_user_approval_func: Optional[callable] = None
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
                logger.info(f"LLM call denied by user")
                return False, "User declined LLM call"
            
            self.approved_calls += 1
            logger.info(f"LLM call approved by user")
        
        # Increment counters
        self.calls_this_minute += 1
        self.total_calls += 1
        
        return True, "Approved"
    
    def _get_approval_message(self, call_type: LLMCallType, user_input: str) -> str:
        """Generate user approval message"""
        messages = {
            LLMCallType.CHITCHAT: "I don't have a learned response for that. Would you like me to use AI to answer it? (This helps me learn!)",
            LLMCallType.TOOL_FORMATTING: "I have the information but need to format it nicely. Proceed with AI help?",
            LLMCallType.GENERATION: "That's something new for me. Want me to look it up with AI?",
            LLMCallType.CLARIFICATION: "I need a bit of AI help to understand that. Should I proceed?",
            LLMCallType.FALLBACK: "I need AI assistance for this one. Proceed?"
        }
        return messages.get(call_type, "Should I use AI for this?")
    
    def record_llm_call(
        self,
        call_type: LLMCallType,
        user_input: str,
        llm_response: str,
        approved_by_user: bool = False
    ):
        """Record an LLM call for tracking"""
        record = LLMCallRecord(
            timestamp=datetime.now(),
            call_type=call_type,
            user_input=user_input,
            llm_response=llm_response,
            approved_by_user=approved_by_user
        )
        self.call_history.append(record)
        
        # Keep only last 100 records
        if len(self.call_history) > 100:
            self.call_history = self.call_history[-100:]
    
    def record_call(
        self,
        call_type: LLMCallType,
        user_input: str,
        llm_response: str
    ):
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
                "requires_approval": self.require_user_approval
            }
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
                responses = [r.llm_response for r in self.call_history 
                           if r.user_input.lower() == user_input]
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
            max_calls_per_minute=10,
            allow_llm_for_chitchat=False,     # Use learned patterns
            allow_llm_for_tools=False,        # Use simple formatters
            allow_llm_for_generation=True,    # Allow for complex tasks
            require_user_approval=True        # Ask before calling
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
        max_calls_per_minute=5,               # Very restrictive rate limit
        allow_llm_for_chitchat=False,         # Never use LLM for chitchat - learn patterns
        allow_llm_for_tools=False,            # Never use LLM for tool formatting - use simple formatters
        allow_llm_for_generation=True,        # Allow LLM for complex generation (with approval)
        require_user_approval=True            # Always ask before calling LLM
    )
    logger.info("Minimal LLM policy activated - Alice will learn organically")
    logger.info(f"Policy settings: {_llm_policy_instance.get_stats()}")
    return _llm_policy_instance

