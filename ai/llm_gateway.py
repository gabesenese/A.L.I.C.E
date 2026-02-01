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
import json
import os
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from ai.llm_policy import get_llm_policy, LLMCallType
from ai.simple_formatters import FormatterRegistry

logger = logging.getLogger(__name__)


# Path to logged interactions file
LOGGED_INTERACTIONS_PATH = "data/training/logged_interactions.jsonl"


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
        
        # Advanced telemetry
        self.stats = {
            'total_requests': 0,
            'self_handlers': 0,
            'pattern_hits': 0,
            'tool_calls': 0,
            'rag_lookups': 0,
            'llm_calls': 0,
            'formatter_calls': 0,
            'policy_denials': 0,
            'by_type': {},
            'recent_requests': []  # Last 100 requests for analysis
        }
        
        logger.info("[LLMGateway] Initialized - All LLM calls now gated with advanced telemetry")
    
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
                logger.info(f"[LLMGateway] [OK] Formatted {tool_name} without LLM")
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
            logger.warning(f"[LLMGateway] [DENIED] LLM call denied: {reason}")
            
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
            logger.info(f"[LLMGateway] [CALL] LLM call ({call_type.value})")
            response = self.llm.chat(prompt, use_history=use_history)
            
            # Record successful call
            self.policy.record_call(call_type, user_input, response)
            self.stats['llm_calls'] += 1
            
            # Log LLM fallback to JSONL file for training
            if call_type == LLMCallType.FALLBACK:
                self._log_llm_fallback(
                    user_input=user_input,
                    intent=context.get('intent', 'unknown') if context else 'unknown',
                    entities=context.get('entities', {}) if context else {},
                    context_snapshot=context or {},
                    llm_response=response
                )
            
            # Log to learning engine if available
            if self.learning_engine and user_input:
                self.learning_engine.collect_interaction(
                    user_input=user_input,
                    assistant_response=response,
                    intent=call_type.value,
                    quality_score=0.8  # Default quality for LLM responses
                )
            
            logger.info(f"[LLMGateway] [OK] LLM responded ({len(response)} chars)")
            return LLMResponse(
                success=True,
                response=response,
                used_llm=True
            )
        
        except Exception as e:
            logger.error(f"[LLMGateway] [ERROR] LLM error: {e}")
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
    
    def _log_llm_fallback(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        context_snapshot: Dict[str, Any],
        llm_response: str
    ):
        """
        Log LLM fallback calls to JSONL file for pattern mining
        
        This data will be used for identifying patterns to learn,
        PEFT/instruction tuning, and understanding knowledge gaps.
        
        Args:
            user_input: User input text
            intent: Detected intent
            entities: Extracted entities
            context_snapshot: Current context
            llm_response: LLM response text
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(LOGGED_INTERACTIONS_PATH), exist_ok=True)
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "intent": intent,
                "entities": entities,
                "context": {
                    k: v for k, v in context_snapshot.items()
                    if k not in ['llm_engine', 'memory_system', 'plugin_manager']
                },
                "llm_response": llm_response,
                "call_type": "LLM_FALLBACK"
            }
            
            # Append to JSONL file
            with open(LOGGED_INTERACTIONS_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"[LLMGateway] Logged LLM fallback: {user_input[:50]}...")
        
        except Exception as e:
            logger.error(f"[LLMGateway] Failed to log LLM fallback: {e}")
    
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
            return "I do not have a learned response for that yet. Keep chatting with me so I can learn!"
        
        elif call_type == LLMCallType.TOOL_FORMATTING:
            return "I have the information but cannot format it right now. Here is the raw data."
        
        elif call_type == LLMCallType.GENERATION:
            return "I need more training data before I can help with that. Try asking something I have learned!"
        
        elif call_type == LLMCallType.CLARIFICATION:
            return "Could you rephrase that? I am not sure I understand."
        
        else:
            return f"I cannot process that request right now. ({reason})"
    
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
        return f'''The user asked: "{user_input}"

The {tool_name} tool returned this data:
{data}

Please provide a natural, concise response to the user based on this data.
Be conversational and helpful. Do not mention the tool name or technical details.'''
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get advanced gateway statistics with detailed routing breakdown"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        stats = {
            **self.stats,
            'self_handler_percentage': round(100 * self.stats['self_handlers'] / total, 1),
            'pattern_hit_percentage': round(100 * self.stats['pattern_hits'] / total, 1),
            'tool_call_percentage': round(100 * self.stats['tool_calls'] / total, 1),
            'rag_lookup_percentage': round(100 * self.stats['rag_lookups'] / total, 1),
            'llm_fallback_percentage': round(100 * self.stats['llm_calls'] / total, 1),
            'formatter_percentage': round(100 * self.stats['formatter_calls'] / total, 1),
            'denial_percentage': round(100 * self.stats['policy_denials'] / total, 1)
        }
        return stats
    
    def record_self_handler(self):
        """Record request handled by self-handler (code execution, commands, etc)"""
        self.stats['self_handlers'] += 1
        self.stats['total_requests'] += 1
    
    def record_pattern_hit(self):
        """Record that conversational engine used learned pattern"""
        self.stats['pattern_hits'] += 1
        self.stats['total_requests'] += 1
    
    def record_tool_call(self):
        """Record that request was handled by tool/plugin"""
        self.stats['tool_calls'] += 1
        self.stats['total_requests'] += 1
    
    def record_rag_lookup(self):
        """Record that request was answered via RAG/memory lookup"""
        self.stats['rag_lookups'] += 1
        self.stats['total_requests'] += 1
    
    def reset_statistics(self):
        """Reset gateway statistics"""
        self.stats = {
            'total_requests': 0,
            'self_handlers': 0,
            'pattern_hits': 0,
            'tool_calls': 0,
            'rag_lookups': 0,
            'llm_calls': 0,
            'formatter_calls': 0,
            'policy_denials': 0,
            'by_type': {},
            'recent_requests': []
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
