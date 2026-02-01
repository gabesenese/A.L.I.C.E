"""
Deterministic Request Router for A.L.I.C.E
Implements explicit routing policy with testable state machine
Emits events for metrics collection and reactive behavior
"""

from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """
    Explicit routing paths in strict priority order
    Lower number = higher priority
    """
    SELF_REFLECTION = "self_reflection"  # Code introspection, training stats, system commands (priority 1)
    CONVERSATIONAL = "conversational"    # Learned patterns, greetings, chitchat (priority 2)
    TOOL_CALL = "tool"                   # Plugin/tool execution (priority 3)
    RAG_ONLY = "rag_only"               # Knowledge base retrieval without generation (priority 4)
    LLM_FALLBACK = "llm_fallback"       # LLM generation (last resort, priority 5)
    ERROR = "error"                      # Routing failed


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
    Deterministic router with strict priority ordering
    
    Routing Order (Priority):
    1. SELF_REFLECTION: Code/training/system introspection
    2. CONVERSATIONAL: Learned patterns from conversational engine
    3. TOOL_CALL: Plugin execution with structured output
    4. RAG_ONLY: Knowledge base retrieval without LLM generation
    5. LLM_FALLBACK: LLM generation (last resort)
    
    This makes routing testable, predictable, and minimizes LLM dependency.
    """
    
    # Intents for self-reflection (highest priority)
    SELF_REFLECTION_INTENTS = {
        'code_request', 'code_analysis', 'training_status', 'system_info',
        'performance_metrics', 'memory_stats', 'learning_stats',
        'debug_mode', 'self_analysis', 'system_status'
    }
    
    # Intents that conversational engine handles without LLM
    CONVERSATION_INTENTS = {
        'greeting', 'farewell', 'thanks', 'affirmation', 'negation',
        'praise', 'insult', 'small_talk', 'status_check', 'help',
        'capabilities', 'joke', 'about_alice', 'clarification_needed',
        'vague_question', 'meta_question'
    }
    
    # Safety-flagged intents that always need review (goes to LLM for safety check)
    SAFETY_CHECK_INTENTS = {
        'file_delete_all', 'file_wipe', 'data_delete_all',
        'system_shutdown', 'unsafe_system_command', 'deploy_command',
        'data_wipe', 'privilege_escalation', 'unsafe_operation'
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
        'system_command', 'process_management',
        
        # Web
        'web_search', 'weather_query', 'weather:current', 'weather:forecast', 'news_query',
        
        # Music
        'music_play', 'music_pause', 'music_stop', 'music_search',
        
        # Notes
        'note_create', 'note_read', 'note_search', 'note_update', 'note_delete',
        
        # Maps
        'location_query', 'directions_query', 'place_search',
        
        # Documents (document_list only - queries go to RAG)
        'document_ingest', 'document_list'
    }
    
    # Intents that can use RAG without generation
    RAG_INTENTS = {
        'document_query', 'fact_lookup', 'definition', 'information_retrieval'
    }
    
    # Intents that need LLM generation (last resort)
    LLM_INTENTS = {
        'question_answering', 'explanation', 'creative_writing',
        'summarization', 'translation', 'analysis', 'reasoning',
        'code_generation', 'code_explanation', 'brainstorming',
        'advice', 'planning'
    }
    
    # Confidence thresholds
    MIN_TOOL_CONFIDENCE = 0.7   # Minimum confidence to call a tool (raised from 0.6)
    MIN_CONV_CONFIDENCE = 0.7   # Minimum confidence for conversational engine
    CLARIFICATION_THRESHOLD = 0.75  # Below this, check for domain keywords
    
    # Domain keywords for intent validation
    DOMAIN_KEYWORDS = {
        'weather': {'weather', 'temperature', 'outside', 'today', 'tomorrow', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'degrees', 'celsius', 'fahrenheit'},
        'email': {'email', 'inbox', 'message', 'send', 'reply', 'compose', 'mail'},
        'calendar': {'calendar', 'meeting', 'schedule', 'appointment', 'event', 'available', 'busy'},
        'file': {'file', 'folder', 'directory', 'document', 'read', 'write', 'delete', 'move'},
        'note': {'note', 'reminder', 'remember', 'write down'},
        'music': {'music', 'song', 'play', 'pause', 'stop', 'audio', 'track'}
    }
    
    def __init__(self):
        """Initialize router"""
        self.routing_stats = {
            'self_reflection': 0,
            'conversational': 0,
            'tool': 0,
            'rag': 0,
            'llm_fallback': 0,
            'error': 0
        }
        
        # Try to import event bus for event emission (optional)
        self.event_bus = None
        try:
            from ai.event_bus import get_event_bus
            self.event_bus = get_event_bus()
        except Exception as e:
            logger.debug(f"Event bus not available: {e}")
    
    def should_clarify(self, intent: str, confidence: float, entities: Dict[str, Any], user_text: str) -> bool:
        """Check if we should ask for clarification instead of executing tool"""
        # Always clarify if confidence is very low
        if confidence < 0.7:
            # Check if text has strong domain keywords
            text_lower = user_text.lower()
            
            # Detect domain from intent
            domain = None
            if 'weather' in intent:
                domain = 'weather'
            elif 'email' in intent:
                domain = 'email'
            elif 'calendar' in intent or 'schedule' in intent or 'meeting' in intent:
                domain = 'calendar'
            elif 'file' in intent:
                domain = 'file'
            elif 'note' in intent:
                domain = 'note'
            elif 'music' in intent:
                domain = 'music'
            
            # If we detected a domain, check for domain keywords
            if domain and domain in self.DOMAIN_KEYWORDS:
                keywords = self.DOMAIN_KEYWORDS[domain]
                # If text contains at least 2 domain keywords, don't clarify
                keyword_count = sum(1 for kw in keywords if kw in text_lower)
                if keyword_count >= 2:
                    return False
            
            # Check for vague patterns
            vague_patterns = [
                'question about', 'tell me about', 'what about',
                'curious about', 'wondering about', 'ask you about',
                'know about', 'information on', 'details on'
            ]
            
            if any(pattern in text_lower for pattern in vague_patterns):
                return True
        
        return False
    
    def _emit_routing_event(self, stage: str, decision: RoutingDecision, intent: str, confidence: float, metadata: Dict[str, Any] = None):
        """Emit routing event for metrics collection"""
        if not self.event_bus:
            return
        
        try:
            # Emit custom routing event
            event_data = {
                'stage': stage,
                'decision': decision.value,
                'intent': intent,
                'confidence': confidence,
                'metadata': metadata or {}
            }
            
            # Try to use custom event if available
            if hasattr(self.event_bus, 'emit_routing_event'):
                self.event_bus.emit_routing_event(stage, event_data)
            else:
                # Fallback: use generic custom event
                if hasattr(self.event_bus, 'emit_custom'):
                    self.event_bus.emit_custom(f'routing.{stage}', event_data)
        except Exception as e:
            logger.debug(f"Failed to emit routing event: {e}")
    
    def route(
        self,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        user_text: str = ""
    ) -> RouteResult:
        """
        Make routing decision with strict priority ordering
        
        Args:
            intent: Detected intent from NLP
            confidence: Intent detection confidence (0-1)
            entities: Extracted entities
            context: Optional conversation context
            user_text: Original user input text for clarification detection
            
        Returns:
            RouteResult with routing decision
        """
        context = context or {}
        
        # Priority 1: SELF_REFLECTION (code, training, system)
        if intent in self.SELF_REFLECTION_INTENTS:
            self.routing_stats['self_reflection'] += 1
            self._emit_routing_event('self_reflection', RoutingDecision.SELF_REFLECTION, intent, 1.0)
            return RouteResult(
                decision=RoutingDecision.SELF_REFLECTION,
                confidence=1.0,  # Always high confidence for self-reflection
                reasoning=f"Self-reflection intent: {intent}"
            )
        
        # Priority 2: CONVERSATIONAL (learned patterns)
        if intent in self.CONVERSATION_INTENTS:
            if confidence >= self.MIN_CONV_CONFIDENCE:
                self.routing_stats['conversational'] += 1
                self._emit_routing_event('conversational', RoutingDecision.CONVERSATIONAL, intent, confidence)
                return RouteResult(
                    decision=RoutingDecision.CONVERSATIONAL,
                    confidence=confidence,
                    reasoning=f"Conversational pattern: {intent}"
                )
        
        # Priority 2.5: SAFETY CHECK (potentially dangerous operations need review)
        if intent in self.SAFETY_CHECK_INTENTS:
            self.routing_stats['llm_fallback'] += 1
            self._emit_routing_event('llm', RoutingDecision.LLM_FALLBACK, intent, 1.0)
            return RouteResult(
                decision=RoutingDecision.LLM_FALLBACK,
                confidence=1.0,
                reasoning=f"Safety-flagged operation requires review: {intent}",
                metadata={'safety_check': True, 'require_user_approval': True}
            )
        
        # Priority 2.75: CLARIFICATION CHECK (before tools)
        # Check if we should ask for clarification instead of executing tool
        if intent in self.TOOL_INTENTS and user_text:
            if self.should_clarify(intent, confidence, entities, user_text):
                self.routing_stats['conversational'] += 1
                self._emit_routing_event('conversational', RoutingDecision.CONVERSATIONAL, intent, confidence)
                return RouteResult(
                    decision=RoutingDecision.CONVERSATIONAL,
                    confidence=1.0,
                    reasoning="Vague question requires clarification",
                    metadata={'clarification_needed': True, 'original_intent': intent}
                )
        
        # Priority 3: TOOL_CALL (plugins with structured output)
        if intent in self.TOOL_INTENTS:
            tool_name = self._intent_to_tool(intent)
            
            if confidence >= self.MIN_TOOL_CONFIDENCE:
                self.routing_stats['tool'] += 1
                self._emit_routing_event('tool', RoutingDecision.TOOL_CALL, intent, confidence, {'tool': tool_name})
                return RouteResult(
                    decision=RoutingDecision.TOOL_CALL,
                    confidence=confidence,
                    tool_name=tool_name,
                    reasoning=f"Tool execution: {intent}",
                    metadata={'use_simple_formatter': True}  # Use rule-based formatter, not LLM
                )
            else:
                # Low confidence - route to conversational for clarification
                logger.warning(f"Low confidence ({confidence:.2f}) for tool intent: {intent}")
                self.routing_stats['conversational'] += 1
                self._emit_routing_event('conversational', RoutingDecision.CONVERSATIONAL, intent, confidence)
                return RouteResult(
                    decision=RoutingDecision.CONVERSATIONAL,
                    confidence=confidence,
                    reasoning=f"Low confidence ({confidence:.2f}), requesting clarification",
                    metadata={'clarification_needed': True, 'original_intent': intent}
                )
        
        # Priority 4: RAG_ONLY (knowledge retrieval without generation)
        if intent in self.RAG_INTENTS:
            self.routing_stats['rag'] += 1
            self._emit_routing_event('rag', RoutingDecision.RAG_ONLY, intent, confidence)
            return RouteResult(
                decision=RoutingDecision.RAG_ONLY,
                confidence=confidence,
                reasoning=f"RAG retrieval: {intent}"
            )
        
        # Priority 5: LLM_FALLBACK (last resort)
        if intent in self.LLM_INTENTS or confidence < self.MIN_TOOL_CONFIDENCE:
            self.routing_stats['llm_fallback'] += 1
            self._emit_routing_event('llm', RoutingDecision.LLM_FALLBACK, intent, confidence)
            return RouteResult(
                decision=RoutingDecision.LLM_FALLBACK,
                confidence=confidence,
                reasoning=f"LLM fallback for intent: {intent}",
                metadata={'require_user_approval': True}  # Ask before calling LLM
            )
        
        # Unknown intent - ask user what they want
        logger.warning(f"Unknown intent: {intent}, cannot route")
        self.routing_stats['error'] += 1
        self._emit_routing_event('error', RoutingDecision.ERROR, intent, 0.0)
        return RouteResult(
            decision=RoutingDecision.ERROR,
            confidence=0.0,
            reasoning=f"Unknown intent: {intent}",
            metadata={'suggested_fallback': 'Ask user for clarification'}
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
            
            # Web
            'web_search': 'web_search',
            'weather_query': 'weather',
            'weather:current': 'weather',
            'weather:forecast': 'weather',
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
            
            # Documents (document_query goes to RAG instead)
            'document_ingest': 'documents',
            'document_list': 'documents'
        }
        return intent_tool_map.get(intent)
    
    
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
