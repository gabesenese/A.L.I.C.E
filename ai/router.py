"""
Deterministic Request Router for A.L.I.C.E
Implements explicit routing policy with testable state machine
"""

from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Explicit routing paths"""
    CONVERSATION_ONLY = "conversation"  # Use conversational engine (no LLM)
    TOOL_CALL = "tool"                  # Execute plugin/tool
    LLM_GENERATION = "llm"              # Generate with LLM
    HYBRID = "hybrid"                   # Tool + LLM formatting
    ERROR = "error"                     # Routing failed


@dataclass
class RouteResult:
    """Result of routing decision"""
    decision: RoutingDecision
    confidence: float  # 0-1
    tool_name: Optional[str] = None
    reasoning: str = ""  # Why this route was chosen
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RequestRouter:
    """
    Deterministic router that decides how to handle user input
    
    Routing Logic:
    1. Check if conversational engine can handle (greetings, small talk, etc.)
    2. Check if a tool/plugin should be called based on intent
    3. Fall back to LLM for complex generation
    
    This separation makes the system testable and predictable.
    """
    
    # Intents that conversational engine handles without LLM
    CONVERSATION_INTENTS = {
        'greeting', 'farewell', 'thanks', 'affirmation', 'negation',
        'praise', 'insult', 'small_talk', 'status_check', 'help',
        'capabilities', 'joke', 'about_alice'
    }
    
    # Intents that require tool execution
    TOOL_INTENTS = {
        # Email
        'email_read', 'email_send', 'email_compose', 'email_reply',
        'email_search', 'email_delete', 'email_archive',
        
        # Calendar
        'calendar_create', 'calendar_read', 'calendar_update', 'calendar_delete',
        'schedule_meeting', 'check_availability',
        
        # Files
        'file_read', 'file_write', 'file_search', 'file_delete',
        'file_move', 'file_copy', 'directory_list',
        
        # System
        'system_command', 'process_management', 'system_info',
        
        # Web
        'web_search', 'weather_query', 'news_query',
        
        # Music
        'music_play', 'music_pause', 'music_stop', 'music_search',
        
        # Notes
        'note_create', 'note_read', 'note_search', 'note_update', 'note_delete',
        
        # Maps
        'location_query', 'directions_query', 'place_search',
        
        # Documents
        'document_ingest', 'document_query', 'document_list'
    }
    
    # Intents that need LLM generation
    LLM_INTENTS = {
        'question_answering', 'explanation', 'creative_writing',
        'summarization', 'translation', 'analysis', 'reasoning',
        'code_generation', 'code_explanation', 'brainstorming',
        'advice', 'planning'
    }
    
    # Confidence thresholds
    MIN_TOOL_CONFIDENCE = 0.6   # Minimum confidence to call a tool
    MIN_CONV_CONFIDENCE = 0.7   # Minimum confidence for conversational engine
    
    def __init__(self):
        """Initialize router"""
        self.routing_stats = {
            'conversation': 0,
            'tool': 0,
            'llm': 0,
            'hybrid': 0,
            'error': 0
        }
    
    def route(
        self,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RouteResult:
        """
        Make routing decision based on intent and confidence
        
        Args:
            intent: Detected intent from NLP
            confidence: Intent detection confidence (0-1)
            entities: Extracted entities
            context: Optional conversation context
            
        Returns:
            RouteResult with routing decision
        """
        context = context or {}
        
        # State machine: CONVERSATION_ONLY → TOOL_CALL → LLM_FALLBACK
        
        # State 1: CONVERSATION_ONLY
        if intent in self.CONVERSATION_INTENTS:
            if confidence >= self.MIN_CONV_CONFIDENCE:
                self.routing_stats['conversation'] += 1
                return RouteResult(
                    decision=RoutingDecision.CONVERSATION_ONLY,
                    confidence=confidence,
                    reasoning=f"High-confidence conversational intent: {intent}"
                )
        
        # State 2: TOOL_CALL
        if intent in self.TOOL_INTENTS:
            tool_name = self._intent_to_tool(intent)
            
            if confidence >= self.MIN_TOOL_CONFIDENCE:
                # Check if tool needs LLM for response formatting
                needs_llm_formatting = self._tool_needs_llm(intent, entities)
                
                if needs_llm_formatting:
                    self.routing_stats['hybrid'] += 1
                    return RouteResult(
                        decision=RoutingDecision.HYBRID,
                        confidence=confidence,
                        tool_name=tool_name,
                        reasoning=f"Tool call with LLM formatting: {intent}",
                        metadata={'format_with_llm': True}
                    )
                else:
                    self.routing_stats['tool'] += 1
                    return RouteResult(
                        decision=RoutingDecision.TOOL_CALL,
                        confidence=confidence,
                        tool_name=tool_name,
                        reasoning=f"Direct tool execution: {intent}"
                    )
            else:
                # Low confidence - ask for clarification via LLM
                logger.warning(f"Low confidence ({confidence:.2f}) for tool intent: {intent}")
        
        # State 3: LLM_FALLBACK
        if intent in self.LLM_INTENTS or confidence < self.MIN_TOOL_CONFIDENCE:
            self.routing_stats['llm'] += 1
            return RouteResult(
                decision=RoutingDecision.LLM_GENERATION,
                confidence=confidence,
                reasoning=f"LLM generation for intent: {intent}"
            )
        
        # Unknown intent - default to LLM
        logger.warning(f"Unknown intent: {intent}, falling back to LLM")
        self.routing_stats['llm'] += 1
        return RouteResult(
            decision=RoutingDecision.LLM_GENERATION,
            confidence=0.5,
            reasoning=f"Unknown intent fallback: {intent}"
        )
    
    def _intent_to_tool(self, intent: str) -> Optional[str]:
        """Map intent to tool/plugin name"""
        intent_tool_map = {
            # Email
            'email_read': 'email',
            'email_send': 'email',
            'email_compose': 'email',
            'email_reply': 'email',
            'email_search': 'email',
            'email_delete': 'email',
            'email_archive': 'email',
            
            # Calendar
            'calendar_create': 'calendar',
            'calendar_read': 'calendar',
            'calendar_update': 'calendar',
            'calendar_delete': 'calendar',
            'schedule_meeting': 'calendar',
            'check_availability': 'calendar',
            
            # Files
            'file_read': 'file_operations',
            'file_write': 'file_operations',
            'file_search': 'file_operations',
            'file_delete': 'file_operations',
            'file_move': 'file_operations',
            'file_copy': 'file_operations',
            'directory_list': 'file_operations',
            
            # System
            'system_command': 'system_control',
            'process_management': 'system_control',
            'system_info': 'system_control',
            
            # Web
            'web_search': 'web_search',
            'weather_query': 'weather',
            'news_query': 'web_search',
            
            # Music
            'music_play': 'music',
            'music_pause': 'music',
            'music_stop': 'music',
            'music_search': 'music',
            
            # Notes
            'note_create': 'notes',
            'note_read': 'notes',
            'note_search': 'notes',
            'note_update': 'notes',
            'note_delete': 'notes',
            
            # Maps
            'location_query': 'maps',
            'directions_query': 'maps',
            'place_search': 'maps',
            
            # Documents
            'document_ingest': 'documents',
            'document_query': 'documents',
            'document_list': 'documents'
        }
        return intent_tool_map.get(intent)
    
    def _tool_needs_llm(self, intent: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if tool result needs LLM formatting
        
        Some tools return structured data that needs natural language formatting.
        Others can return user-ready responses directly.
        """
        # Tools that return structured data needing LLM formatting
        llm_formatting_intents = {
            'email_read', 'email_search',  # Email lists need formatting
            'calendar_read',                # Event lists need formatting
            'file_search',                  # Search results need formatting
            'web_search', 'news_query',    # Search results need formatting
            'weather_query',                # Weather data needs formatting
            'document_query'                # RAG results need formatting
        }
        
        return intent in llm_formatting_intents
    
    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics"""
        return self.routing_stats.copy()
    
    def reset_stats(self):
        """Reset routing statistics"""
        for key in self.routing_stats:
            self.routing_stats[key] = 0


# Global singleton
_router_instance = None

def get_router() -> RequestRouter:
    """Get global router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = RequestRouter()
    return _router_instance
