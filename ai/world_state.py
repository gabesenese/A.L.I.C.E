"""
World State for A.L.I.C.E
Unified state for user context, entities, conversation, and system state.
Consumed by Reference Resolver, Goal Resolver, and Verifier.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class EntityKind(str, Enum):
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
    """Something in the world that can be referred to."""
    id: str
    kind: EntityKind
    label: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime = field(default_factory=datetime.now)
    aliases: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """One exchange: user said X, we inferred intent, we responded."""
    turn_id: str
    user_input: str
    intent: str
    entities_raw: Dict[str, Any]
    response: str
    success: bool
    timestamp: datetime


@dataclass
class ActiveGoal:
    """What the user is currently trying to achieve."""
    goal_id: str
    description: str
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    subgoals: List["ActiveGoal"] = field(default_factory=list)
    status: str = "active"


class WorldState:
    """
    Global world state for A.L.I.C.E. Thread-safe, append-only for history.
    Updated by the main pipeline; read by resolvers and verifier.
    """

    def __init__(self, user_name: str = "User", max_turns: int = 50, max_entities: int = 200):
        self._lock = threading.RLock()
        self.user_name = user_name
        self.max_turns = max_turns
        self.max_entities = max_entities

        self.session_start = datetime.now()
        self.user_prefs: Dict[str, Any] = {}

        self.entities: Dict[str, WorldEntity] = {}
        self._entity_order: List[str] = []

        self.turns: List[ConversationTurn] = []

        self.active_goal: Optional[ActiveGoal] = None
        self._goal_stack: List[ActiveGoal] = []

        self.last_intent: str = ""
        self.last_entities: Dict[str, Any] = {}
        self.last_plugin: Optional[str] = None
        self.last_action_success: bool = False
        self.last_response: str = ""

        self.pending_action: Optional[str] = None
        self.pending_data: Dict[str, Any] = {}

        logger.info("[WorldState] A.L.I.C.E world state initialized")

    def add_entity(self, e: WorldEntity) -> None:
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
        with self._lock:
            return self.entities.get(entity_id)

    def get_recent_entities(self, kind: Optional[EntityKind] = None, n: int = 20) -> List[WorldEntity]:
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
        """Match by label or alias (e.g. 'grocery list')."""
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

    def push_goal(self, goal: ActiveGoal) -> None:
        with self._lock:
            if self.active_goal:
                self._goal_stack.append(self.active_goal)
            self.active_goal = goal

    def pop_goal(self) -> Optional[ActiveGoal]:
        with self._lock:
            old = self.active_goal
            self.active_goal = self._goal_stack.pop() if self._goal_stack else None
            return old

    def set_goal(self, goal: Optional[ActiveGoal]) -> None:
        with self._lock:
            self.active_goal = goal
            self._goal_stack.clear()

    def record_turn(self, user_input: str, intent: str, entities: Dict, response: str, success: bool) -> None:
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
        with self._lock:
            self.last_plugin = plugin_name
            self.last_action_success = success

    def snapshot(self) -> Dict[str, Any]:
        """Read-only snapshot for resolvers and verifier."""
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


_world_state: Optional[WorldState] = None
_ws_lock = threading.Lock()


def get_world_state(user_name: str = "User") -> WorldState:
    global _world_state
    with _ws_lock:
        if _world_state is None:
            _world_state = WorldState(user_name=user_name)
        return _world_state


def set_world_state(ws: WorldState) -> None:
    global _world_state
    with _ws_lock:
        _world_state = ws
