"""
LLM Context Schema for A.L.I.C.E
Provides stable, predictable context structure for LLM generation
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class UserProfile:
    """User profile information"""
    name: str = "User"
    location: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    personality_traits: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationMessage:
    """Single message in conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Memory:
    """Semantic memory entry"""
    content: str
    relevance_score: float
    timestamp: str
    memory_type: str  # 'episodic', 'semantic', 'procedural'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActiveGoal:
    """User's active goal"""
    description: str
    status: str  # 'active', 'paused', 'completed', 'cancelled'
    priority: int = 1
    entities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorldFact:
    """Known fact about the world/entities"""
    entity: str
    relation: str
    value: Any
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """Result from tool/plugin execution"""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMContext:
    """
    Complete, deterministic context for LLM generation
    
    This schema ensures:
    1. Predictable ordering (no random dict iteration)
    2. Size limits (prevent context overflow)
    3. Clear structure (easy to debug)
    4. Versioning (track schema changes)
    """
    
    # Schema version for tracking changes
    schema_version: str = "1.0"
    
    # Current request
    user_input: str = ""
    detected_intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    
    # User profile
    user_profile: UserProfile = field(default_factory=UserProfile)
    
    # Conversation history (most recent N messages)
    short_history: List[ConversationMessage] = field(default_factory=list)
    
    # Relevant long-term memories
    long_term_memories: List[Memory] = field(default_factory=list)
    
    # Active goals
    active_goals: List[ActiveGoal] = field(default_factory=list)
    
    # World state (known facts)
    world_facts: List[WorldFact] = field(default_factory=list)
    
    # Tool results (if any)
    tool_results: List[ToolResult] = field(default_factory=list)
    
    # Contextual information
    current_time: str = field(default_factory=lambda: datetime.now().isoformat())
    current_topic: Optional[str] = None
    conversation_summary: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Size limits
    MAX_SHORT_HISTORY = 10      # Last 10 messages
    MAX_LONG_TERM_MEMORIES = 5  # Top 5 relevant memories
    MAX_ACTIVE_GOALS = 3        # Top 3 active goals
    MAX_WORLD_FACTS = 10        # Top 10 relevant facts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (deterministic ordering)"""
        return {
            "schema_version": self.schema_version,
            "user_input": self.user_input,
            "detected_intent": self.detected_intent,
            "entities": self.entities,
            "user_profile": self.user_profile.to_dict(),
            "short_history": [msg.to_dict() for msg in self.short_history],
            "long_term_memories": [mem.to_dict() for mem in self.long_term_memories],
            "active_goals": [goal.to_dict() for goal in self.active_goals],
            "world_facts": [fact.to_dict() for fact in self.world_facts],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "current_time": self.current_time,
            "current_topic": self.current_topic,
            "conversation_summary": self.conversation_summary,
            "metadata": self.metadata
        }
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert to JSON string"""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_prompt(self) -> str:
        """
        Convert to human-readable prompt for LLM
        
        This is the stable format that will be sent to the LLM.
        Changes here should be versioned and tracked.
        """
        sections = []
        
        # User profile
        if self.user_profile.name != "User":
            sections.append(f"User: {self.user_profile.name}")
            if self.user_profile.location:
                sections.append(f"Location: {self.user_profile.location}")
        
        # Current time
        sections.append(f"Current time: {self.current_time}")
        
        # Conversation summary
        if self.conversation_summary:
            sections.append(f"\nConversation Summary:\n{self.conversation_summary}")
        
        # Active goals
        if self.active_goals:
            goals_text = "\n".join([
                f"  - {goal.description} (status: {goal.status})"
                for goal in self.active_goals[:self.MAX_ACTIVE_GOALS]
            ])
            sections.append(f"\nActive Goals:\n{goals_text}")
        
        # Long-term memories
        if self.long_term_memories:
            memories_text = "\n".join([
                f"  - {mem.content} (relevance: {mem.relevance_score:.2f})"
                for mem in self.long_term_memories[:self.MAX_LONG_TERM_MEMORIES]
            ])
            sections.append(f"\nRelevant Memories:\n{memories_text}")
        
        # World facts
        if self.world_facts:
            facts_text = "\n".join([
                f"  - {fact.entity} {fact.relation} {fact.value}"
                for fact in self.world_facts[:self.MAX_WORLD_FACTS]
            ])
            sections.append(f"\nKnown Facts:\n{facts_text}")
        
        # Recent conversation
        if self.short_history:
            history_text = "\n".join([
                f"{msg.role.upper()}: {msg.content}"
                for msg in self.short_history[-self.MAX_SHORT_HISTORY:]
            ])
            sections.append(f"\nRecent Conversation:\n{history_text}")
        
        # Tool results
        if self.tool_results:
            results_text = "\n".join([
                f"  - {tr.tool_name}: {'success' if tr.success else 'failed'}"
                for tr in self.tool_results
            ])
            sections.append(f"\nTool Results:\n{results_text}")
        
        # Current request
        sections.append(f"\nCurrent Request:\n{self.user_input}")
        if self.detected_intent:
            sections.append(f"Detected Intent: {self.detected_intent}")
        
        return "\n".join(sections)
    
    def enforce_limits(self):
        """Enforce size limits to prevent context overflow"""
        self.short_history = self.short_history[-self.MAX_SHORT_HISTORY:]
        self.long_term_memories = self.long_term_memories[:self.MAX_LONG_TERM_MEMORIES]
        self.active_goals = self.active_goals[:self.MAX_ACTIVE_GOALS]
        self.world_facts = self.world_facts[:self.MAX_WORLD_FACTS]
    
    def add_short_history(self, role: str, content: str, intent: str = None, entities: Dict = None):
        """Add message to short history"""
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            intent=intent,
            entities=entities or {}
        )
        self.short_history.append(msg)
        self.enforce_limits()
    
    def add_memory(self, content: str, relevance: float, memory_type: str = "episodic"):
        """Add long-term memory"""
        mem = Memory(
            content=content,
            relevance_score=relevance,
            timestamp=datetime.now().isoformat(),
            memory_type=memory_type
        )
        self.long_term_memories.append(mem)
        # Sort by relevance
        self.long_term_memories.sort(key=lambda m: m.relevance_score, reverse=True)
        self.enforce_limits()
    
    def add_goal(self, description: str, status: str = "active", priority: int = 1, entities: Dict = None):
        """Add active goal"""
        goal = ActiveGoal(
            description=description,
            status=status,
            priority=priority,
            entities=entities or {}
        )
        self.active_goals.append(goal)
        # Sort by priority
        self.active_goals.sort(key=lambda g: g.priority, reverse=True)
        self.enforce_limits()
    
    def add_world_fact(self, entity: str, relation: str, value: Any, confidence: float = 1.0):
        """Add world fact"""
        fact = WorldFact(
            entity=entity,
            relation=relation,
            value=value,
            confidence=confidence
        )
        self.world_facts.append(fact)
        self.enforce_limits()
    
    def add_tool_result(self, tool_name: str, success: bool, data: Any, error: str = None):
        """Add tool result"""
        result = ToolResult(
            tool_name=tool_name,
            success=success,
            data=data,
            error=error
        )
        self.tool_results.append(result)
    
    def clear_tool_results(self):
        """Clear tool results (after they've been processed)"""
        self.tool_results = []


class ContextBuilder:
    """Builder for creating LLMContext objects"""
    
    def __init__(self):
        self.context = LLMContext()
    
    def set_user_input(self, user_input: str, intent: str = None, entities: Dict = None) -> 'ContextBuilder':
        """Set current user input"""
        self.context.user_input = user_input
        self.context.detected_intent = intent
        self.context.entities = entities or {}
        return self
    
    def set_user_profile(self, profile: UserProfile) -> 'ContextBuilder':
        """Set user profile"""
        self.context.user_profile = profile
        return self
    
    def add_history(self, messages: List[ConversationMessage]) -> 'ContextBuilder':
        """Add conversation history"""
        self.context.short_history = messages
        self.context.enforce_limits()
        return self
    
    def add_memories(self, memories: List[Memory]) -> 'ContextBuilder':
        """Add long-term memories"""
        self.context.long_term_memories = memories
        self.context.enforce_limits()
        return self
    
    def add_goals(self, goals: List[ActiveGoal]) -> 'ContextBuilder':
        """Add active goals"""
        self.context.active_goals = goals
        self.context.enforce_limits()
        return self
    
    def add_facts(self, facts: List[WorldFact]) -> 'ContextBuilder':
        """Add world facts"""
        self.context.world_facts = facts
        self.context.enforce_limits()
        return self
    
    def set_summary(self, summary: str) -> 'ContextBuilder':
        """Set conversation summary"""
        self.context.conversation_summary = summary
        return self
    
    def set_topic(self, topic: str) -> 'ContextBuilder':
        """Set current topic"""
        self.context.current_topic = topic
        return self
    
    def build(self) -> LLMContext:
        """Build final context object"""
        self.context.enforce_limits()
        return self.context
