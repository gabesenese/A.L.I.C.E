"""
Foundation Integration Module for A.L.I.C.E
Integrates new foundation systems: Response Variance Engine, Personality Evolution, Context Graph
"""

import logging
from typing import Dict, Any, Optional, List
from ai.response.response_variance_engine import ResponseVarianceEngine, ResponseContext
from ai.personality.personality_evolution import PersonalityEvolutionEngine
from ai.memory.context_graph import ContextGraph

logger = logging.getLogger(__name__)


class FoundationIntegration:
    """
    Manages integration of all foundation systems
    Provides clean interface for main.py to use
    """
    
    def __init__(self, llm_generator=None, phrasing_learner=None):
        # Initialize new foundation systems
        self.response_engine = ResponseVarianceEngine(
            llm_generator=llm_generator,
            phrasing_learner=phrasing_learner
        )
        self.personality_engine = PersonalityEvolutionEngine()
        self.context_graph = ContextGraph()
        
        logger.info("Foundation systems initialized: ResponseVariance, Personality, ContextGraph")
    
    def process_interaction(
        self,
        user_id: str,
        user_input: str,
        intent: str,
        entities: Dict[str, List[Any]],
        plugin_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a complete interaction through all foundation systems
        
        Args:
            user_id: User identifier
            user_input: User's input text
            intent: Detected intent
            entities: Extracted entities
            plugin_result: Result from plugin execution (if any)
        
        Returns:
            Dict containing response and metadata
        """
        # 1. Record interaction in context graph
        conversation_history = self.context_graph.get_conversation_history(
            user_id=user_id,
            limit=10
        )
        
        # 2. Get user's personality traits
        personality = self.personality_engine.get_traits_for_user(user_id)
        
        # 3. Prepare response context
        response_ctx = ResponseContext(
            intent_type=intent,
            data=plugin_result.get('data', {}) if plugin_result else {},
            user_id=user_id,
            conversation_history=[
                {
                    'user_input': turn.user_input,
                    'alice_response': turn.alice_response,
                    'intent': turn.intent
                }
                for turn in conversation_history
            ],
            user_verbosity_pref=personality.verbosity
        )
        
        # 4. Generate varied response
        alice_response = self.response_engine.generate_response(response_ctx)
        
        # 5. Record turn in context graph (with placeholder for now)
        turn = self.context_graph.record_turn(
            user_id=user_id,
            user_input=user_input,
            alice_response=alice_response,
            intent=intent,
            entities=entities
        )
        
        # 6. Save context graph periodically
        if len(self.context_graph.conversation_history) % 10 == 0:
            self.context_graph.save()
        
        return {
            'response': alice_response,
            'personality_traits': personality.to_dict(),
            'turn_id': turn.turn_id,
            'context_entities': len(turn.entities)
        }
    
    def learn_from_feedback(
        self,
        user_id: str,
        user_input: str,
        alice_response: str,
        user_reaction: Optional[str] = None
    ):
        """
        Learn from user's reaction to response
        Updates both personality and response quality tracking
        """
        # 1. Learn personality adaptation
        self.personality_engine.learn_from_interaction(
            user_id=user_id,
            user_input=user_input,
            alice_response=alice_response,
            user_reaction=user_reaction
        )
        
        # 2. Track response quality
        if user_reaction:
            self.response_engine.record_response_quality(
                user_id=user_id,
                response=alice_response,
                user_reaction=user_reaction
            )
        
        logger.debug(f"Learning feedback processed for user {user_id}")
    
    def get_context_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive context for user
        Replaces old context systems
        """
        # Get from context graph
        graph_context = self.context_graph.get_context_summary(user_id)
        
        # Add personality info
        personality = self.personality_engine.get_personality_profile(user_id)
        
        return {
            **graph_context,
            'personality': personality
        }
    
    def get_recent_topics(self, user_id: str, limit: int = 5) -> List[str]:
        """Get recent conversation topics for this user"""
        entities = self.context_graph.get_recent_entities(
            entity_type='topic',
            limit=limit
        )
        return [e.value for e in entities]
    
    def get_recent_locations(self, user_id: str, limit: int = 3) -> List[str]:
        """Get recent locations mentioned"""
        entities = self.context_graph.get_recent_entities(
            entity_type='location',
            limit=limit
        )
        return [e.value for e in entities]
    
    def query_context(self, query: str) -> Dict[str, Any]:
        """Natural language query interface for context"""
        return self.context_graph.query_context(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about all foundation systems"""
        return {
            'context_graph': self.context_graph.get_statistics(),
            'personality_users': len(self.personality_engine.user_traits),
            'response_history': sum(
                len(history) 
                for history in self.response_engine.response_history.values()
            )
        }
    
    def maintenance(self):
        """Periodic maintenance tasks"""
        # Apply temporal decay to context
        self.context_graph.apply_temporal_decay()
        
        # Save all state
        self.context_graph.save()
        self.personality_engine._save_traits()
        
        logger.info("Foundation maintenance completed")


# ============================================================================
# Integration Instructions for main.py
# ============================================================================

"""
HOW TO INTEGRATE INTO main.py:

1. Import at top of main.py:
   from ai.foundation_integration import FoundationIntegration

2. Initialize in Alice.__init__():
   self.foundations = FoundationIntegration(
       llm_generator=self.ollama,
       phrasing_learner=self.phrasing_learner
   )

3. Replace OLD context systems with:
   
   OLD CODE (REMOVE):
   - self.conversation_summary
   - self.conversation_topics
   - self.referenced_items
   - self.conversation_context
   - Multiple overlapping memory systems
   
   NEW CODE (USE):
   context = self.foundations.get_context_summary(user_id="Gabriel")

4. In process_input(), AFTER plugin execution:
   
   OLD CODE (around line 3700):
   response = self._alice_direct_phrase(...)  # Hardcoded templates
   
   NEW CODE:
   result = self.foundations.process_interaction(
       user_id="Gabriel",
       user_input=user_input,
       intent=intent,
       entities=entities,
       plugin_result=plugin_result
   )
   response = result['response']

5. Track user feedback (add after response is sent):
   
   # Wait for next user input
   next_input = ...  # User's next message
   
   self.foundations.learn_from_feedback(
       user_id="Gabriel",
       user_input=user_input,
       alice_response=response,
       user_reaction=next_input if is_reaction else None
   )

6. Use context in plugins:
   
   OLD CODE:
   context_summary = self.context.get_context_summary()
   
   NEW CODE:
   context_summary = self.foundations.get_context_summary("Gabriel")

7. Periodic maintenance (add to main loop or shutdown):
   
   # Every N interactions or on shutdown
   self.foundations.maintenance()

8. Query context naturally:
   
   # User asks: "What did we talk about earlier?"
   context_info = self.foundations.query_context(user_input)


MIGRATION STRATEGY:

Phase 1: Add new systems alongside old (parallel)
- Keep old code working
- Add foundations.process_interaction() but still use old response
- Compare outputs, verify correctness

Phase 2: Switch to new systems for responses
- Use foundations.process_interaction() for ALL responses
- Keep old context systems for plugins (temporary)

Phase 3: Migrate plugins to use ContextGraph
- Update plugin calls to use foundations.get_context_summary()
- Remove old context systems one by one

Phase 4: Clean up
- Delete old context code
- Remove hardcoded response templates
- Verify all tests pass


TESTING:

1. Test response variance:
   Ask same question 3 times, responses should be different

2. Test personality adaptation:
   Use brief messages → A.L.I.C.E becomes more concise
   Use detailed messages → A.L.I.C.E becomes more verbose

3. Test repetition detection:
   Ask same question 3x → A.L.I.C.E should acknowledge repetition

4. Test context persistence:
   Restart A.L.I.C.E → context should be restored from disk
"""
