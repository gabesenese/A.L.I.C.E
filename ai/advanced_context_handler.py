"""
Advanced Context Handler for A.L.I.C.E
Implements sophisticated context tracking, coreference resolution, and pragmatic understanding
Based on latest research in conversational AI and dialogue state tracking
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re
import numpy as np
from sentence_transformers import util
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ContextEntity:
    """Represents an entity that can be referenced in conversation"""
    entity_id: str
    entity_type: str  # email, file, person, topic, etc.
    data: Dict[str, Any]
    mentions: List[str] = field(default_factory=list)  # How it's been referred to
    last_mentioned: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0
    aliases: List[str] = field(default_factory=list)  # Alternative names


@dataclass
class ConversationTurn:
    """Represents a single conversation turn with rich metadata"""
    turn_id: str
    user_input: str
    assistant_response: str
    intent: str
    entities_mentioned: List[str]
    entities_resolved: Dict[str, str]  # pronoun/reference -> entity_id
    timestamp: datetime
    context_used: List[str]  # Which entities were used for context
    pragmatic_signals: Dict[str, Any] = field(default_factory=dict)


class AdvancedContextHandler:
    """
    Advanced context management with:
    - Coreference resolution
    - Entity tracking and aliasing
    - Pragmatic understanding
    - Multi-turn conversation state
    - Semantic similarity matching
    """
    
    def __init__(self, embeddings_model=None):
        # Core state
        self.entities: Dict[str, ContextEntity] = {}
        self.conversation_turns: List[ConversationTurn] = []
        self.current_focus: List[str] = []  # Currently active entities
        
        # Coreference tracking
        self.pronoun_stack: List[str] = []  # Recent entities for pronoun resolution
        self.demonstrative_stack: List[str] = []  # For "this", "that", etc.
        
        # Pragmatic understanding
        self.discourse_markers: Dict[str, str] = {
            "by the way": "topic_shift",
            "speaking of": "topic_continuation", 
            "anyway": "topic_shift",
            "also": "topic_addition",
            "but": "contrast",
            "however": "contrast",
            "meanwhile": "parallel_context"
        }
        
        # Entity type patterns
        self.entity_patterns = {
            "email": {
                "indicators": ["email", "message", "mail"],
                "pronouns": ["it", "this", "that", "the email", "the message"],
                "properties": ["from", "to", "subject", "content", "date"]
            },
            "person": {
                "indicators": ["from", "by", "person", "user", "sender"],
                "pronouns": ["he", "she", "they", "him", "her", "them"],
                "properties": ["name", "email_address", "relationship"]
            },
            "file": {
                "indicators": ["file", "document", "report", "attachment"],
                "pronouns": ["it", "this", "that", "the file"],
                "properties": ["name", "path", "type", "size"]
            },
            "topic": {
                "indicators": ["topic", "subject", "matter", "issue"],
                "pronouns": ["it", "this", "that"],
                "properties": ["name", "keywords", "related_entities"]
            }
        }
        
        # Initialize embeddings for semantic matching
        self.embeddings_model = embeddings_model
        if embeddings_model:
            self.entity_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info("[OK] Advanced Context Handler initialized")
    
    def process_turn(self, user_input: str, assistant_response: str = "", 
                    intent: str = "", entities: Dict = None) -> ConversationTurn:
        """Process a conversation turn and update context"""
        
        turn_id = f"turn_{len(self.conversation_turns) + 1}_{int(datetime.now().timestamp())}"
        
        # Extract and resolve entities
        mentioned_entities = self._extract_entities(user_input)
        resolved_entities = self._resolve_coreferences(user_input, mentioned_entities)
        
        # Detect pragmatic signals
        pragmatic_signals = self._analyze_pragmatics(user_input)
        
        # Update entity focus based on conversation dynamics
        self._update_focus(mentioned_entities, pragmatic_signals)
        
        # Create turn record
        turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities_mentioned=mentioned_entities,
            entities_resolved=resolved_entities,
            timestamp=datetime.now(),
            context_used=self.current_focus.copy(),
            pragmatic_signals=pragmatic_signals
        )
        
        # Store turn and update stacks
        self.conversation_turns.append(turn)
        self._update_reference_stacks(mentioned_entities)
        
        # Update entity relevance scores
        self._update_entity_relevance()
        
        return turn
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity mentions from text"""
        entities_found = []
        
        # Look for explicit entity references
        for entity_id, entity in self.entities.items():
            # Check direct mentions
            for mention in entity.mentions + [entity_id] + entity.aliases:
                if mention.lower() in text.lower():
                    entities_found.append(entity_id)
                    break
        
        return list(set(entities_found))
    
    def _resolve_coreferences(self, text: str, mentioned_entities: List[str]) -> Dict[str, str]:
        """Resolve pronouns and references to specific entities"""
        resolutions = {}
        text_lower = text.lower()
        
        # Pronoun resolution
        pronoun_patterns = {
            r'\bit\b': self._resolve_neutral_pronoun,
            r'\bthis\b': self._resolve_demonstrative,
            r'\bthat\b': self._resolve_demonstrative,
            r'\bthe (email|message|file|document)\b': self._resolve_definite_description,
            r'\bthe (latest|last|recent|first)\b': self._resolve_temporal_reference,
        }
        
        for pattern, resolver in pronoun_patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                reference = match.group()
                resolved_entity = resolver(reference, text, mentioned_entities)
                if resolved_entity:
                    resolutions[reference] = resolved_entity
        
        return resolutions
    
    def _resolve_neutral_pronoun(self, pronoun: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve 'it' to the most recently mentioned neutral entity"""
        # Look for the most recently mentioned non-person entity
        for entity_id in reversed(self.pronoun_stack):
            entity = self.entities.get(entity_id)
            if entity and entity.entity_type != "person":
                return entity_id
        return None
    
    def _resolve_demonstrative(self, demonstrative: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve 'this'/'that' based on recency and context"""
        if not self.demonstrative_stack:
            return None
            
        # "this" typically refers to the most recent entity
        if "this" in demonstrative:
            return self.demonstrative_stack[-1] if self.demonstrative_stack else None
        
        # "that" might refer to a less recent entity
        if "that" in demonstrative and len(self.demonstrative_stack) > 1:
            return self.demonstrative_stack[-2]
        
        return self.demonstrative_stack[-1] if self.demonstrative_stack else None
    
    def _resolve_definite_description(self, phrase: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve definite descriptions like 'the email', 'the file'"""
        # Extract entity type from phrase
        entity_type = None
        for etype, patterns in self.entity_patterns.items():
            for indicator in patterns["indicators"]:
                if indicator in phrase:
                    entity_type = etype
                    break
        
        if not entity_type:
            return None
        
        # Find the most recently mentioned entity of that type
        for entity_id in reversed(self.pronoun_stack):
            entity = self.entities.get(entity_id)
            if entity and entity.entity_type == entity_type:
                return entity_id
        
        return None
    
    def _resolve_temporal_reference(self, phrase: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve temporal references like 'the latest', 'the first'"""
        if "latest" in phrase or "last" in phrase or "recent" in phrase:
            # Return the most recently mentioned entity
            return self.pronoun_stack[-1] if self.pronoun_stack else None
        
        if "first" in phrase:
            # Return the first mentioned entity in current focus
            return self.current_focus[0] if self.current_focus else None
        
        return None
    
    def _analyze_pragmatics(self, text: str) -> Dict[str, Any]:
        """Analyze pragmatic signals in the text"""
        signals = {}
        text_lower = text.lower()
        
        # Discourse markers
        for marker, signal_type in self.discourse_markers.items():
            if marker in text_lower:
                signals["discourse_marker"] = signal_type
                break
        
        # Politeness markers
        if any(word in text_lower for word in ["please", "could you", "would you", "thank you"]):
            signals["politeness"] = "high"
        
        # Urgency markers
        if any(word in text_lower for word in ["urgent", "asap", "quickly", "immediately", "now"]):
            signals["urgency"] = "high"
        
        # Certainty markers
        if any(word in text_lower for word in ["definitely", "certainly", "absolutely", "sure"]):
            signals["certainty"] = "high"
        elif any(word in text_lower for word in ["maybe", "perhaps", "possibly", "might"]):
            signals["certainty"] = "low"
        
        # Question types
        if text.strip().endswith("?"):
            if any(word in text_lower for word in ["what", "how", "why", "when", "where", "who"]):
                signals["question_type"] = "wh_question"
            elif any(word in text_lower for word in ["can", "could", "would", "should", "do", "did", "is", "are"]):
                signals["question_type"] = "yes_no_question"
        
        return signals
    
    def _update_focus(self, mentioned_entities: List[str], pragmatic_signals: Dict[str, Any]):
        """Update the current conversation focus"""
        
        # Handle topic shifts
        if pragmatic_signals.get("discourse_marker") == "topic_shift":
            self.current_focus.clear()
        
        # Add newly mentioned entities to focus
        for entity_id in mentioned_entities:
            if entity_id not in self.current_focus:
                self.current_focus.append(entity_id)
        
        # Limit focus size (keep most recent and relevant)
        if len(self.current_focus) > 5:
            # Keep entities by relevance and recency
            focus_scores = []
            for entity_id in self.current_focus:
                entity = self.entities.get(entity_id)
                if entity:
                    recency_score = 1.0 / max(1, (datetime.now() - entity.last_mentioned).total_seconds() / 3600)
                    total_score = entity.relevance_score * recency_score
                    focus_scores.append((entity_id, total_score))
            
            # Keep top 5
            focus_scores.sort(key=lambda x: x[1], reverse=True)
            self.current_focus = [entity_id for entity_id, _ in focus_scores[:5]]
    
    def _update_reference_stacks(self, mentioned_entities: List[str]):
        """Update pronoun and demonstrative reference stacks"""
        
        # Update pronoun stack
        for entity_id in mentioned_entities:
            if entity_id in self.pronoun_stack:
                self.pronoun_stack.remove(entity_id)
            self.pronoun_stack.append(entity_id)
        
        # Keep stack size manageable
        if len(self.pronoun_stack) > 10:
            self.pronoun_stack = self.pronoun_stack[-10:]
        
        # Update demonstrative stack (similar but separate)
        self.demonstrative_stack = self.pronoun_stack.copy()
    
    def _update_entity_relevance(self):
        """Update relevance scores for all entities based on recent activity"""
        now = datetime.now()
        
        for entity_id, entity in self.entities.items():
            # Decay relevance over time
            time_since_mention = (now - entity.last_mentioned).total_seconds() / 3600  # hours
            decay_factor = max(0.1, 1.0 - (time_since_mention * 0.1))  # 10% per hour
            entity.relevance_score *= decay_factor
            
            # Boost relevance if in current focus
            if entity_id in self.current_focus:
                entity.relevance_score = min(1.0, entity.relevance_score * 1.2)
    
    def add_entity(self, entity_type: str, data: Dict[str, Any], 
                   entity_id: Optional[str] = None, aliases: List[str] = None) -> str:
        """Add a new entity to the context"""
        
        if not entity_id:
            entity_id = f"{entity_type}_{int(datetime.now().timestamp())}"
        
        entity = ContextEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            data=data,
            aliases=aliases or [],
            last_mentioned=datetime.now()
        )
        
        self.entities[entity_id] = entity
        
        # Generate embeddings if model available
        if self.embeddings_model:
            entity_text = self._entity_to_text(entity)
            self.entity_embeddings[entity_id] = self.embeddings_model.encode(entity_text)
        
        logger.info(f"Added entity {entity_id} of type {entity_type}")
        return entity_id
    
    def _entity_to_text(self, entity: ContextEntity) -> str:
        """Convert entity to text representation for embedding"""
        text_parts = [entity.entity_type]
        
        for key, value in entity.data.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
        
        text_parts.extend(entity.aliases)
        return " ".join(text_parts)
    
    def get_contextual_entities(self, query: str, max_entities: int = 5) -> List[Tuple[str, float]]:
        """Get entities relevant to a query using semantic similarity"""
        if not self.embeddings_model or not self.entity_embeddings:
            # Fallback to focus-based selection
            return [(eid, self.entities[eid].relevance_score) for eid in self.current_focus[:max_entities]]
        
        # Encode query
        query_embedding = self.embeddings_model.encode(query)
        
        # Compute similarities
        similarities = []
        for entity_id, entity_embedding in self.entity_embeddings.items():
            similarity = util.cos_sim(query_embedding, entity_embedding).item()
            
            # Boost similarity with relevance score
            entity = self.entities[entity_id]
            boosted_score = similarity * entity.relevance_score
            similarities.append((entity_id, boosted_score))
        
        # Sort and return top entities
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_entities]
    
    def resolve_reference(self, reference_text: str) -> Optional[str]:
        """Resolve a textual reference to an entity ID"""
        reference_lower = reference_text.lower().strip()
        
        # Direct entity matching
        for entity_id, entity in self.entities.items():
            # Check entity ID, mentions, and aliases
            all_refs = [entity_id.lower()] + [m.lower() for m in entity.mentions] + [a.lower() for a in entity.aliases]
            if reference_lower in all_refs or any(ref in reference_lower for ref in all_refs):
                return entity_id
        
        # Semantic matching if available
        if self.embeddings_model and self.entity_embeddings:
            similarities = self.get_contextual_entities(reference_text, max_entities=1)
            if similarities and similarities[0][1] > 0.7:  # High confidence threshold
                return similarities[0][0]
        
        return None
    
    def get_context_for_llm(self, query: str) -> str:
        """Generate rich context string for LLM"""
        context_parts = []
        
        # Get relevant entities
        relevant_entities = self.get_contextual_entities(query)
        
        if relevant_entities:
            context_parts.append("Relevant context:")
            for entity_id, score in relevant_entities:
                entity = self.entities[entity_id]
                entity_context = f"- {entity.entity_type.title()}: "
                
                # Add key entity information
                if entity.entity_type == "email":
                    subject = entity.data.get("subject", "No subject")
                    sender = entity.data.get("from", "Unknown sender")
                    entity_context += f"'{subject}' from {sender}"
                elif entity.entity_type == "person":
                    name = entity.data.get("name", entity_id)
                    entity_context += f"{name}"
                elif entity.entity_type == "file":
                    name = entity.data.get("name", entity_id)
                    entity_context += f"'{name}'"
                else:
                    entity_context += str(entity.data)
                
                context_parts.append(entity_context)
        
        # Add recent conversation context
        if self.conversation_turns:
            recent_turns = self.conversation_turns[-3:]
            context_parts.append("\\nRecent conversation:")
            for turn in recent_turns:
                if turn.entities_resolved:
                    resolved_info = ", ".join([f"{ref}→{eid}" for ref, eid in turn.entities_resolved.items()])
                    context_parts.append(f"- User: {turn.user_input[:100]} (resolved: {resolved_info})")
                else:
                    context_parts.append(f"- User: {turn.user_input[:100]}")
        
        return "\\n".join(context_parts)
    
    def save_state(self, filepath: str):
        """Save context state to file"""
        state = {
            "entities": {eid: {
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "data": e.data,
                "mentions": e.mentions,
                "last_mentioned": e.last_mentioned.isoformat(),
                "relevance_score": e.relevance_score,
                "aliases": e.aliases
            } for eid, e in self.entities.items()},
            "current_focus": self.current_focus,
            "pronoun_stack": self.pronoun_stack,
            "conversation_turns": len(self.conversation_turns),  # Just count, not full history
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Context state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load context state from file"""
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)
            
            # Restore entities
            for eid, entity_data in state.get("entities", {}).items():
                entity = ContextEntity(
                    entity_id=entity_data["entity_id"],
                    entity_type=entity_data["entity_type"],
                    data=entity_data["data"],
                    mentions=entity_data["mentions"],
                    last_mentioned=datetime.fromisoformat(entity_data["last_mentioned"]),
                    relevance_score=entity_data["relevance_score"],
                    aliases=entity_data["aliases"]
                )
                self.entities[eid] = entity
            
            # Restore other state
            self.current_focus = state.get("current_focus", [])
            self.pronoun_stack = state.get("pronoun_stack", [])
            
            logger.info(f"Context state loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"Could not load context state: {e}")