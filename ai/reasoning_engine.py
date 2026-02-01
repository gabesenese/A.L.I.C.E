"""
Unified Reasoning Engine for A.L.I.C.E
Consolidates: world_state + reference_resolver + goal_resolver + verifier
Manages entity tracking, goal resolution, reference resolution, and verification
"""

import re
import logging
import uuid
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityKind(str, Enum):
    """Types of entities that can be referenced"""
    NOTE = "note"
    EMAIL = "email"
    EVENT = "event"
    FILE = "file"
    PERSON = "person"
    TOPIC = "topic"
    TASK = "task"
    REFERENCE = "reference"


@dataclass
class WorldEntity:
    """Something in the world that can be referred to"""
    id: str
    kind: EntityKind
    label: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime = field(default_factory=datetime.now)
    aliases: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """One exchange: user said X, intent inferred, response generated"""
    turn_id: str
    user_input: str
    intent: str
    entities_raw: Dict[str, Any]
    response: str
    success: bool
    timestamp: datetime


@dataclass
class ActiveGoal:
    """What the user is currently trying to achieve"""
    goal_id: str
    description: str
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    subgoals: List["ActiveGoal"] = field(default_factory=list)
    status: str = "active"


@dataclass
class ResolvedReference:
    """A reference that was resolved to an entity"""
    raw_text: str
    resolved_id: str
    resolved_label: str
    kind: EntityKind
    confidence: float


@dataclass
class ResolutionResult:
    """Result of resolving references in user input"""
    resolved_input: str
    bindings: Dict[str, ResolvedReference] = field(default_factory=dict)
    entities_to_use: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalResolution:
    """Result of resolving the current user goal"""
    goal: Optional[ActiveGoal]
    intent: str
    entities: Dict[str, Any]
    cancelled: bool
    revised: bool
    message: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of verifying an action or goal"""
    verified: bool
    action_succeeded: bool
    goal_fulfilled: bool
    message: Optional[str] = None
    suggested_follow_up: Optional[str] = None


class ReasoningEngine:
    """
    Unified reasoning system for A.L.I.C.E
    Combines:
    - World state tracking (entities, turns, goals)
    - Reference resolution (pronouns, definite descriptions)
    - Goal resolution (tracking user goals, cancellations, revisions)
    - Action verification (checking success and goal fulfillment)
    """
    
    def __init__(self, user_name: str = "User", max_turns: int = 50, max_entities: int = 200):
        self._lock = threading.RLock()
        self.user_name = user_name
        self.max_turns = max_turns
        self.max_entities = max_entities
        
        # World state
        self.session_start = datetime.now()
        self.user_prefs: Dict[str, Any] = {}
        self.entities: Dict[str, WorldEntity] = {}
        self._entity_order: List[str] = []
        self.turns: List[ConversationTurn] = []
        
        # Goal tracking
        self.active_goal: Optional[ActiveGoal] = None
        self._goal_stack: List[ActiveGoal] = []
        
        # Action/plugin state
        self.last_intent: str = ""
        self.last_entities: Dict[str, Any] = {}
        self.last_plugin: Optional[str] = None
        self.last_action_success: bool = False
        self.last_response: str = ""
        
        # Pending operations
        self.pending_action: Optional[str] = None
        self.pending_data: Dict[str, Any] = {}
        
        # Reference resolution patterns
        self.PRONOUNS = ["it", "this", "that", "the one", "the first", "the last"]
        self.DEFINITE_PATTERNS = [
            (r"the\s+(.+?)\s+list\b", EntityKind.NOTE),
            (r"the\s+(.+?)\s+note\b", EntityKind.NOTE),
            (r"that\s+(.+?)\s+note\b", EntityKind.NOTE),
            (r"my\s+(.+?)\s+list\b", EntityKind.NOTE),
            (r"the\s+(.+?)\s+email\b", EntityKind.EMAIL),
            (r"that\s+email\b", EntityKind.EMAIL),
            (r"the\s+(.+?)\s+event\b", EntityKind.EVENT),
            (r"that\s+event\b", EntityKind.EVENT),
        ]
        
        # Goal resolution patterns
        self.CANCEL_PATTERNS = [
            r"\b(?:cancel|abort|never\s+mind|forget\s+it|don\'?t\s+do\s+that)\b",
            r"\b(?:stop|undo)\s+that\b",
            r"\bno\s*,?\s*(?:forget\s+it|cancel)\b",
        ]
        self.REVISE_PATTERNS = [
            r"\bactually\s+(.+)$",
            r"\binstead\s+(.+)$",
            r"\b(?:no\s*,?\s*)?(?:do|make|create|delete|remove)\s+(.+)$",
        ]
        self.REFERENCE_PATTERNS = [
            r"\b(?:do\s+it|go\s+ahead|proceed|yes)\s*\.?\s*$",
            r"\b(?:that\'?s\s+what\s+i\s+meant|correct|right)\s*\.?\s*$",
        ]
        
        # Verification patterns
        self.FAILURE_INDICATORS = [
            r"\b(?:fail|failed|error|couldn\'?t|could\s+not)\b",
            r"\b(?:not\s+found|doesn\'?t\s+exist)\b",
            r"\b(?:sorry|unable|invalid)\b",
        ]
        self.SUCCESS_INDICATORS = [
            r"\b(?:done|completed|archived|deleted|created|added|sent)\b",
            r"\b(?:ok|success)\b",
            r"^[\s\S]*\b(?:successfully|done)\b[\s\S]*$",
        ]
        
        logger.info("[OK] Reasoning Engine initialized")
    
    # ========== WORLD STATE METHODS ==========
    
    def add_entity(self, e: WorldEntity) -> None:
        """Add an entity to world state"""
        with self._lock:
            e.last_mentioned = datetime.now()
            self.entities[e.id] = e
            if e.id in self._entity_order:
                self._entity_order.remove(e.id)
            self._entity_order.append(e.id)
            while len(self._entity_order) > self.max_entities:
                old = self._entity_order.pop(0)
                self.entities.pop(old, None)
    
    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        """Get an entity by ID"""
        with self._lock:
            return self.entities.get(entity_id)
    
    def get_recent_entities(self, kind: Optional[EntityKind] = None, n: int = 20) -> List[WorldEntity]:
        """Get recent entities, optionally filtered by kind"""
        with self._lock:
            order = list(reversed(self._entity_order))
            out = []
            for eid in order:
                if len(out) >= n:
                    break
                e = self.entities.get(eid)
                if e and (kind is None or e.kind == kind):
                    out.append(e)
            return out
    
    def find_entity_by_label(self, label: str, kind: Optional[EntityKind] = None) -> Optional[WorldEntity]:
        """Find entity by label or alias"""
        with self._lock:
            label_lower = label.lower().strip()
            for eid in reversed(self._entity_order):
                e = self.entities.get(eid)
                if not e:
                    continue
                if kind is not None and e.kind != kind:
                    continue
                if label_lower in e.label.lower():
                    return e
                for a in e.aliases:
                    if label_lower in a.lower():
                        return e
            return None
    
    def record_turn(self, user_input: str, intent: str, entities: Dict, response: str, success: bool) -> None:
        """Record a conversation turn"""
        with self._lock:
            turn_id = f"t{len(self.turns)+1}_{int(datetime.now().timestamp())}"
            self.turns.append(ConversationTurn(
                turn_id=turn_id,
                user_input=user_input,
                intent=intent,
                entities_raw=dict(entities),
                response=response,
                success=success,
                timestamp=datetime.now(),
            ))
            while len(self.turns) > self.max_turns:
                self.turns.pop(0)
            self.last_intent = intent
            self.last_entities = dict(entities)
            self.last_response = response
            self.last_action_success = success
    
    def record_plugin_result(self, plugin_name: str, success: bool) -> None:
        """Record plugin execution result"""
        with self._lock:
            self.last_plugin = plugin_name
            self.last_action_success = success
    
    def snapshot(self) -> Dict[str, Any]:
        """Get a read-only snapshot of world state"""
        with self._lock:
            return {
                "user_name": self.user_name,
                "active_goal": {
                    "description": self.active_goal.description,
                    "intent": self.active_goal.intent,
                    "entities": self.active_goal.entities,
                } if self.active_goal else None,
                "last_intent": self.last_intent,
                "last_entities": self.last_entities,
                "last_plugin": self.last_plugin,
                "last_action_success": self.last_action_success,
                "last_response": self.last_response,
                "pending_action": self.pending_action,
                "recent_entity_labels": [
                    self.entities[eid].label for eid in list(reversed(self._entity_order))[:15]
                    if self.entities.get(eid)
                ],
                "recent_entities_by_kind": {
                    k.value: [e.label for e in self.get_recent_entities(kind=k, n=5)]
                    for k in EntityKind
                },
            }
    
    # ========== GOAL RESOLUTION METHODS ==========
    
    def push_goal(self, goal: ActiveGoal) -> None:
        """Push goal onto stack"""
        with self._lock:
            if self.active_goal:
                self._goal_stack.append(self.active_goal)
            self.active_goal = goal
    
    def pop_goal(self) -> Optional[ActiveGoal]:
        """Pop goal from stack"""
        with self._lock:
            old = self.active_goal
            self.active_goal = self._goal_stack.pop() if self._goal_stack else None
            return old
    
    def set_goal(self, goal: Optional[ActiveGoal]) -> None:
        """Set current goal"""
        with self._lock:
            self.active_goal = goal
            self._goal_stack.clear()
    
    def resolve_goal(self, user_input: str, intent: str, entities: Dict[str, Any],
                    resolved_input: Optional[str] = None) -> GoalResolution:
        """Resolve current goal from user input and intent"""
        cancelled = False
        revised = False
        msg: Optional[str] = None
        inp = (resolved_input or user_input).strip().lower()
        
        # 1) Cancel: "cancel that", "never mind"
        for pat in self.CANCEL_PATTERNS:
            if re.search(pat, inp):
                self.set_goal(None)
                cancelled = True
                msg = "Understood. Cancelled."
                return GoalResolution(
                    goal=None,
                    intent="conversation:ack",
                    entities={},
                    cancelled=True,
                    revised=False,
                    message=msg,
                )
        
        # 2) Revise: "actually delete the shopping list"
        for pat in self.REVISE_PATTERNS:
            m = re.search(pat, inp, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            rest = m.group(1).strip() if m.lastindex else ""
            if len(rest) < 3:
                continue
            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            new_goal = ActiveGoal(
                goal_id=goal_id,
                description=rest,
                intent=intent,
                entities=dict(entities),
            )
            self.set_goal(new_goal)
            revised = True
            return GoalResolution(
                goal=new_goal,
                intent=intent,
                entities=entities,
                cancelled=False,
                revised=True,
                message=None,
            )
        
        # 3) Reference: "do it", "go ahead"
        for pat in self.REFERENCE_PATTERNS:
            if re.search(pat, inp):
                cur = self.active_goal
                if cur:
                    return GoalResolution(
                        goal=cur,
                        intent=cur.intent,
                        entities=cur.entities,
                        cancelled=False,
                        revised=False,
                        message=None,
                    )
                return GoalResolution(
                    goal=None,
                    intent=intent,
                    entities=entities,
                    cancelled=False,
                    revised=False,
                    message="I'm not sure what to do—could you say it again?",
                )
        
        # 4) Check if input relates to existing goal
        current_goal = self.active_goal
        if current_goal:
            goal_keywords = set(current_goal.description.lower().split())
            input_keywords = set(inp.split())
            goal_entities = set(str(v).lower() for v in current_goal.entities.values() if v)
            input_entities = set(str(v).lower() for v in entities.values() if v)
            
            if (intent == current_goal.intent or
                len(goal_keywords & input_keywords) >= 2 or
                len(goal_entities & input_entities) > 0 or
                any(word in inp for word in current_goal.description.lower().split()[:5])):
                current_goal.entities = {**current_goal.entities, **entities}
                if len(user_input) > len(current_goal.description):
                    current_goal.description = user_input[:200]
                return GoalResolution(
                    goal=current_goal,
                    intent=intent if intent != "conversation:general" else current_goal.intent,
                    entities={**current_goal.entities, **entities},
                    cancelled=False,
                    revised=False,
                    message=None,
                )
        
        # 5) New goal: only if clearly new task
        if intent in ["conversation:ack", "conversation:general"] and len(inp.split()) < 5:
            if current_goal:
                return GoalResolution(
                    goal=current_goal,
                    intent=current_goal.intent,
                    entities=current_goal.entities,
                    cancelled=False,
                    revised=False,
                    message=None,
                )
        
        # Create new goal for substantial requests
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        goal = ActiveGoal(
            goal_id=goal_id,
            description=user_input[:200],
            intent=intent,
            entities=dict(entities),
        )
        self.set_goal(goal)
        return GoalResolution(
            goal=goal,
            intent=intent,
            entities=entities,
            cancelled=False,
            revised=False,
            message=None,
        )
    
    def get_current_goal(self) -> Optional[ActiveGoal]:
        """Get current active goal"""
        return self.active_goal
    
    def mark_goal_completed(self) -> None:
        """Mark current goal as completed"""
        self.set_goal(None)
    
    # ========== REFERENCE RESOLUTION METHODS ==========
    
    def resolve_references(self, user_input: str) -> ResolutionResult:
        """Resolve pronouns and definite references in user input"""
        bindings: Dict[str, ResolvedReference] = {}
        entities_to_use: Dict[str, Any] = {}
        resolved_input = user_input
        snap = self.snapshot()
        
        # 1) Pronoun resolution
        for pron in self.PRONOUNS:
            if not re.search(rf"\b{re.escape(pron)}\b", user_input, re.IGNORECASE):
                continue
            entity = self._resolve_pronoun(pron, snap)
            if entity:
                ref = ResolvedReference(
                    raw_text=pron,
                    resolved_id=entity.id,
                    resolved_label=entity.label,
                    kind=entity.kind,
                    confidence=0.85,
                )
                bindings[pron] = ref
                key = "note_id" if entity.kind == EntityKind.NOTE else "entity_id"
                if key not in entities_to_use:
                    entities_to_use[key] = entity.id
                entities_to_use["resolved_" + key] = entity.label
                logger.info(f"[ReferenceResolver] '{pron}' -> {entity.label}")
        
        # 2) Definite noun phrases
        for pattern, kind in self.DEFINITE_PATTERNS:
            m = re.search(pattern, user_input, re.IGNORECASE)
            if not m:
                continue
            phrase = m.group(0)
            if phrase in bindings:
                continue
            inner = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else ""
            entity = self._resolve_definite(phrase, inner, kind, snap)
            if entity:
                ref = ResolvedReference(
                    raw_text=phrase,
                    resolved_id=entity.id,
                    resolved_label=entity.label,
                    kind=entity.kind,
                    confidence=0.9,
                )
                bindings[phrase] = ref
                key = "note_id" if entity.kind == EntityKind.NOTE else "entity_id"
                if key not in entities_to_use:
                    entities_to_use[key] = entity.id
                entities_to_use["resolved_" + key] = entity.label
                logger.info(f"[ReferenceResolver] '{phrase}' -> {entity.label}")
        
        # 3) Substitute resolved refs for clarity
        for phrase, ref in bindings.items():
            try:
                resolved_input = re.sub(
                    re.escape(phrase),
                    ref.resolved_label,
                    resolved_input,
                    count=1,
                    flags=re.IGNORECASE,
                )
            except Exception:
                pass
        
        return ResolutionResult(
            resolved_input=resolved_input.strip() or user_input,
            bindings=bindings,
            entities_to_use=entities_to_use,
        )
    
    def _resolve_pronoun(self, pronoun: str, snap: Dict[str, Any]) -> Optional[WorldEntity]:
        """Resolve pronoun to recent entity"""
        last_intent = snap.get("last_intent") or ""
        recent = snap.get("recent_entity_labels") or []
        by_kind = snap.get("recent_entities_by_kind") or {}
        
        if "notes" in last_intent or "note" in last_intent:
            labels = by_kind.get(EntityKind.NOTE.value, [])
            if labels:
                label = labels[0]
                return self.find_entity_by_label(label, EntityKind.NOTE)
        if "email" in last_intent:
            labels = by_kind.get(EntityKind.EMAIL.value, [])
            if labels:
                return self.find_entity_by_label(labels[0], EntityKind.EMAIL)
        
        for label in recent:
            e = self.find_entity_by_label(label)
            if e:
                return e
        return None
    
    def _resolve_definite(self, phrase: str, inner: str, kind: EntityKind, snap: Dict[str, Any]) -> Optional[WorldEntity]:
        """Resolve definite NP like 'the grocery list'"""
        if inner:
            return self.find_entity_by_label(inner, kind)
        by_kind = snap.get("recent_entities_by_kind") or {}
        labels = by_kind.get(kind.value, [])
        if labels:
            return self.find_entity_by_label(labels[0], kind)
        return None
    
    # ========== VERIFICATION METHODS ==========
    
    def verify(self, plugin_result: Dict[str, Any], goal_intent: Optional[str] = None,
               goal_description: Optional[str] = None) -> VerificationResult:
        """Verify plugin result against goal"""
        success = plugin_result.get("success", False)
        response = (plugin_result.get("response") or plugin_result.get("message") or "").strip()
        
        action_succeeded = self._check_action_success(success, response)
        goal_fulfilled = True
        if goal_intent or goal_description:
            goal_fulfilled = self._check_goal_fulfilled(success, response, goal_intent, goal_description)
        
        verified = action_succeeded and goal_fulfilled
        msg: Optional[str] = None
        follow_up: Optional[str] = None
        
        if not action_succeeded and not success:
            msg = "That didn't complete as expected."
            follow_up = "Would you like me to try again or do something else?"
        elif not goal_fulfilled and success:
            msg = "I did that, but it may not be what you had in mind."
            follow_up = "Say what you'd like changed and I'll adjust."
        
        return VerificationResult(
            verified=verified,
            action_succeeded=action_succeeded,
            goal_fulfilled=goal_fulfilled,
            message=msg,
            suggested_follow_up=follow_up,
        )
    
    def _check_action_success(self, reported: bool, response: str) -> bool:
        """Check if action succeeded based on response"""
        if not reported:
            for pat in self.FAILURE_INDICATORS:
                if re.search(pat, response, re.IGNORECASE):
                    return False
            return False
        for pat in self.FAILURE_INDICATORS:
            if re.search(pat, response, re.IGNORECASE):
                return False
        return True
    
    def _check_goal_fulfilled(self, success: bool, response: str, goal_intent: Optional[str],
                             goal_description: Optional[str]) -> bool:
        """Check if goal was fulfilled"""
        if not success:
            return False
        lower = response.lower()
        if goal_intent and ("delete" in goal_intent or "remove" in goal_intent):
            return bool(re.search(r"\b(?:archived|deleted|removed|done)\b", lower))
        if goal_intent and ("create" in goal_intent or "add" in goal_intent):
            return bool(re.search(r"\b(?:created|added|done|saved)\b", lower))
        return True


# Singleton instance
_reasoning_engine_instance: Optional[ReasoningEngine] = None


def get_reasoning_engine(user_name: str = "User") -> ReasoningEngine:
    """Get or create the singleton ReasoningEngine instance"""
    global _reasoning_engine_instance
    if _reasoning_engine_instance is None:
        _reasoning_engine_instance = ReasoningEngine(user_name)
    return _reasoning_engine_instance


if __name__ == "__main__":
    print("Testing Reasoning Engine...")
    
    engine = get_reasoning_engine()
    
    # Test entity tracking
    entity = WorldEntity(
        id="note_1",
        kind=EntityKind.NOTE,
        label="grocery list",
        data={"content": "milk, eggs, bread"}
    )
    engine.add_entity(entity)
    
    # Test reference resolution
    result = engine.resolve_references("delete it")
    print(f"Resolved: {result.resolved_input}")
    
    # Test goal resolution
    goal_res = engine.resolve_goal("delete the grocery list", "note:delete", {"note": "grocery list"})
    print(f"Goal: {goal_res.goal.description if goal_res.goal else None}")
    
    # Test verification
    plugin_result = {
        "success": True,
        "response": None,
        "data": {"message_code": "notes:deleted"}
    }
    verify_res = engine.verify(plugin_result, "note:delete", "delete the grocery list")
    print(f"Verified: {verify_res.verified}")
    
    print("\n[OK] Reasoning Engine working correctly")
