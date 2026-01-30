"""
Reasoning Engine for A.L.I.C.E
Goes beyond intent classification to understand context, uncertainty, and alternatives.
Makes A.L.I.C.E reason about what the user really wants, not just match patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of reasoning about user intent"""
    primary_intent: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    uncertainty_reasons: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    context_hints: Dict[str, Any] = field(default_factory=dict)


class ReasoningEngine:
    """
    Reasoning engine that:
    - Evaluates confidence and uncertainty
    - Considers alternatives when confidence is low
    - Asks for clarification when needed
    - Uses context to disambiguate
    """
    
    def __init__(self, world_state=None):
        self.world_state = world_state
    
    def reason_about_intent(
        self,
        user_input: str,
        detected_intent: str,
        intent_confidence: float,
        entities: Dict[str, Any],
        recent_context: List[str] = None
    ) -> ReasoningResult:
        """
        Reason about what the user really wants, beyond simple intent matching.
        Returns reasoning result with confidence, alternatives, and clarification needs.
        """
        alternatives = []
        uncertainty_reasons = []
        needs_clarification = False
        clarification_question = None
        context_hints = {}
        
        # Low confidence threshold
        if intent_confidence < 0.6:
            uncertainty_reasons.append(f"Low confidence ({intent_confidence:.2f})")
            needs_clarification = True
        
        # Ambiguous entities
        if entities:
            ambiguous = []
            for key, value in entities.items():
                if isinstance(value, list) and len(value) > 1:
                    ambiguous.append(key)
            if ambiguous:
                uncertainty_reasons.append(f"Multiple possible {', '.join(ambiguous)}")
                needs_clarification = True
        
        # Check for common ambiguities
        input_lower = user_input.lower()
        
        # "delete the list" - which list?
        if "delete" in input_lower and ("list" in input_lower or "note" in input_lower):
            if self.world_state:
                recent_notes = self.world_state.get_recent_entities(kind="note", n=5)
                if len(recent_notes) > 1:
                    uncertainty_reasons.append("Multiple notes available")
                    clarification_question = f"Which note? You have {len(recent_notes)} recent notes."
                    needs_clarification = True
        
        # "send email" - missing recipient
        if "send" in input_lower and "email" in input_lower:
            if not entities.get('recipient') and not entities.get('to'):
                uncertainty_reasons.append("Missing recipient")
                clarification_question = "Who should I send the email to?"
                needs_clarification = True
        
        # Generate alternatives if confidence is low
        if intent_confidence < 0.7:
            # Try to infer alternatives from context
            if "note" in input_lower:
                alternatives.append(("notes:list", 0.3))
                alternatives.append(("notes:create", 0.3))
                alternatives.append(("notes:delete", 0.3))
            elif "email" in input_lower:
                alternatives.append(("email:list", 0.3))
                alternatives.append(("email:compose", 0.3))
        
        # Use recent context to boost confidence
        if recent_context:
            for ctx in recent_context[-3:]:
                if detected_intent in ctx.lower():
                    context_hints["recent_context_match"] = True
                    break
        
        return ReasoningResult(
            primary_intent=detected_intent,
            confidence=intent_confidence,
            alternatives=alternatives,
            uncertainty_reasons=uncertainty_reasons,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            context_hints=context_hints
        )
    
    def should_ask_for_clarification(self, reasoning: ReasoningResult) -> bool:
        """Determine if we should ask for clarification"""
        return reasoning.needs_clarification and reasoning.clarification_question is not None
    
    def get_clarification_prompt(self, reasoning: ReasoningResult) -> Optional[str]:
        """Get the clarification question to ask"""
        if reasoning.clarification_question:
            return reasoning.clarification_question
        
        if reasoning.uncertainty_reasons:
            return f"I'm not entirely sure what you mean. {', '.join(reasoning.uncertainty_reasons)}. Could you clarify?"
        
        return None


_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine(world_state=None) -> ReasoningEngine:
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine(world_state=world_state)
    return _reasoning_engine
