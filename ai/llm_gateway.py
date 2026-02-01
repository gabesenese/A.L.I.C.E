"""
LLM Gateway - Single Entry Point for All LLM Calls

This gateway enforces:
- LLM policy checks before every call
- Simple formatters tried before LLM generation
- Rate limiting and budget tracking
- Automatic logging to learning engine
- User approval for non-essential calls

All code should call LLMGateway.request() instead of llm.chat() directly.
"""

import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from ai.llm_policy import get_llm_policy, LLMCallType
from ai.simple_formatters import FormatterRegistry

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Request to the LLM gateway"""
    prompt: str
    call_type: LLMCallType
    use_history: bool = False
    context: Optional[Dict[str, Any]] = None
    user_input: str = ""
    tool_name: Optional[str] = None
    tool_data: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from the LLM gateway"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    used_llm: bool = False
    used_formatter: bool = False
    formatter_name: Optional[str] = None
    denied_by_policy: bool = False
    policy_reason: Optional[str] = None


class LLMGateway:
    """
    Single gateway for all LLM access
    
    Responsibilities:
    1. Check LLM policy before every call
    2. Try simple formatters first for tool outputs
    3. Enforce rate limits and budgets
    4. Log all calls for learning
    5. Provide fallback messages when LLM denied
    """
    
    def __init__(self, llm_engine, learning_engine=None):
        """
        Initialize gateway
        
        Args:
            llm_engine: LocalLLMEngine instance
            learning_engine: Optional learning engine for logging
        """
        self.llm = llm_engine
        self.learning_engine = learning_engine
        self.policy = get_llm_policy()
        self.formatter_registry = FormatterRegistry()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'llm_calls': 0,
            'formatter_calls': 0,
            'policy_denials': 0,
            'by_type': {}
        }
        
        logger.info("[LLMGateway] Initialized - All LLM calls now gated")
    
    def request(
        self,
        prompt: str,
        call_type: LLMCallType,
        use_history: bool = False,
        context: Optional[Dict[str, Any]] = None,
        user_input: str = "",
        tool_name: Optional[str] = None,
        tool_data: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Request LLM generation (with policy enforcement)
        
        Args:
            prompt: LLM prompt
            call_type: Type of call (CHITCHAT, TOOL_FORMATTING, GENERATION, etc.)
            use_history: Whether to use conversation history
            context: Additional context
            user_input: Original user input
            tool_name: Name of tool if formatting tool output
            tool_data: Tool output data if formatting
        
        Returns:
            LLMResponse with result or denial reason
        """
        self.stats['total_requests'] += 1
        
        # Track by type
        type_key = call_type.value
        self.stats['by_type'][type_key] = self.stats['by_type'].get(type_key, 0) + 1
        
        # Step 1: Try formatter first for tool outputs
        if call_type == LLMCallType.TOOL_FORMATTING and tool_name and tool_data:
            formatter_result = self._try_formatter(tool_name, tool_data, context or {})
            if formatter_result:
                self.stats['formatter_calls'] += 1
                logger.info(f"[LLMGateway] ✓ Formatted {tool_name} without LLM")
                return LLMResponse(
                    success=True,
                    response=formatter_result,
                    used_formatter=True,
                    formatter_name=tool_name
                )
        
        # Step 2: Check LLM policy
        allowed, reason = self.policy.can_call_llm(call_type, user_input)
        
        if not allowed:
            self.stats['policy_denials'] += 1
            logger.warning(f"[LLMGateway] ✗ LLM call denied: {reason}")
            
            # Return appropriate fallback message
            fallback = self._get_policy_fallback(call_type, reason)
            return LLMResponse(
                success=False,
                response=fallback,
                denied_by_policy=True,
                policy_reason=reason
            )
        
        # Step 3: Call LLM
        try:
            logger.info(f"[LLMGateway] → LLM call ({call_type.value})")
            response = self.llm.chat(prompt, use_history=use_history)
            
            # Record successful call
            self.policy.record_call(call_type, user_input, response)
            self.stats['llm_calls'] += 1
            
            # Log to learning engine if available
            if self.learning_engine and user_input:
                self.learning_engine.collect_interaction(
                    user_input=user_input,
                    assistant_response=response,
                    intent=call_type.value,
                    quality_score=0.8  # Default quality for LLM responses
                )
            
            logger.info(f"[LLMGateway] ✓ LLM responded ({len(response)} chars)")
            return LLMResponse(
                success=True,
                response=response,
                used_llm=True
            )
        
        except Exception as e:
            logger.error(f"[LLMGateway] ✗ LLM error: {e}")
            return LLMResponse(
                success=False,
                error=str(e),
                response="I encountered an error processing that request."
            )
    
    def _try_formatter(
        self,
        tool_name: str,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to format tool output without LLM
        
        Args:
            tool_name: Name of the tool
            data: Tool output data
            context: Additional context
        
        Returns:
            Formatted string or None if no formatter available
        """
        try:
            return self.formatter_registry.format(tool_name, data, **context)
        except Exception as e:
            logger.debug(f"[LLMGateway] Formatter failed for {tool_name}: {e}")
            return None
    
    def _get_policy_fallback(self, call_type: LLMCallType, reason: str) -> str:
        """
        Get appropriate fallback message when LLM denied by policy
        
        Args:
            call_type: Type of LLM call that was denied
            reason: Reason for denial
        
        Returns:
            User-friendly fallback message
        """
        if call_type == LLMCallType.CHITCHAT:
            return "I don't have a learned response for that yet. Keep chatting with me so I can learn!"
        
        elif call_type == LLMCallType.TOOL_FORMATTING:
            return "I have the information but can't format it right now. Here's the raw data."
        
        elif call_type == LLMCallType.GENERATION:
            return "I need more training data before I can help with that. Try asking something I've learned!"
        
        elif call_type == LLMCallType.CLARIFICATION:
            return "Could you rephrase that? I'm not sure I understand."
        
        else:
            return f"I can't process that request right now. ({reason})"
    
    def format_tool_result(
        self,
        tool_name: str,
        data: Dict[str, Any],
        user_input: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format tool output (tries formatter first, falls back to LLM if allowed)
        
        Args:
            tool_name: Name of the tool
            data: Tool output data
            user_input: Original user input
            context: Additional context
        
        Returns:
            Formatted response string
        """
        # Try formatter first
        formatted = self._try_formatter(tool_name, data, context or {})
        if formatted:
            return formatted
        
        # Formatter failed - try LLM if policy allows
        prompt = self._build_tool_format_prompt(tool_name, data, user_input)
        
        response = self.request(
            prompt=prompt,
            call_type=LLMCallType.TOOL_FORMATTING,
            use_history=False,
            user_input=user_input,
            tool_name=tool_name,
            tool_data=data
        )
        
        if response.success and response.response:
            return response.response
        
        # Both failed - return raw data
        logger.warning(f"[LLMGateway] Both formatter and LLM failed for {tool_name}")
        return f"Result: {str(data)[:500]}"
    
    def _build_tool_format_prompt(
        self,
        tool_name: str,
        data: Dict[str, Any],
        user_input: str
    ) -> str:
        """Build LLM prompt for formatting tool output"""
        return f"""The user asked: "{user_input}"

The {tool_name} tool returned this data:
{data}

Please provide a natural, concise response to the user based on this data.
Be conversational and helpful. Don't mention the tool name or technical details."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'formatter_percentage': round(100 * self.stats['formatter_calls'] / total, 1),
            'llm_percentage': round(100 * self.stats['llm_calls'] / total, 1),
            'denial_percentage': round(100 * self.stats['policy_denials'] / total, 1)
        }
    
    def reset_statistics(self):
        """Reset gateway statistics"""
        self.stats = {
            'total_requests': 0,
            'llm_calls': 0,
            'formatter_calls': 0,
            'policy_denials': 0,
            'by_type': {}
        }


# Singleton instance
_gateway_instance: Optional[LLMGateway] = None


def get_llm_gateway(llm_engine=None, learning_engine=None) -> LLMGateway:
    """Get or create singleton gateway instance"""
    global _gateway_instance
    
    if _gateway_instance is None:
        if llm_engine is None:
            raise ValueError("llm_engine required for first gateway initialization")
        _gateway_instance = LLMGateway(llm_engine, learning_engine)
    
    return _gateway_instance


def reset_gateway():
    """Reset gateway singleton (for testing)"""
    global _gateway_instance
    _gateway_instance = None
