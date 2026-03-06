"""
Context Graph for A.L.I.C.E
Unified context tracking system using graph database approach
Replaces multiple overlapping memory systems with single source of truth
"""

import logging
import json
import time
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """
    A node in the context graph
    Represents any entity discussed: person, place, thing, topic, note, etc.
    """

    entity_id: str
    entity_type: str  # 'person', 'location', 'topic', 'note', 'email', etc.
    value: Any
    first_mentioned: float
    last_mentioned: float
    mention_count: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "value": self.value,
            "first_mentioned": self.first_mentioned,
            "last_mentioned": self.last_mentioned,
            "mention_count": self.mention_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Entity":
        return cls(**data)


@dataclass
class Relationship:
    """
    An edge in the context graph
    Represents relationships between entities
    """

    source_id: str
    target_id: str
    relationship_type: (
        str  # 'mentioned_with', 'created', 'modified', 'related_to', etc.
    )
    created_at: float
    strength: float = 1.0  # 0.0 to 1.0, increases with repeated co-occurrence
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "created_at": self.created_at,
            "strength": self.strength,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Relationship":
        return cls(**data)


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""

    turn_id: str
    user_id: str
    user_input: str
    alice_response: str
    intent: str
    entities: List[str]  # entity_ids mentioned in this turn
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContextGraph:
    """
    Unified context management using graph structure
    Single source of truth for all context data
    """

    def __init__(self, data_dir: str = "data/context_graph"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core graph structures
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[Tuple[str, str, str], Relationship] = (
            {}
        )  # (source, target, type) -> relationship

        # Conversation history (chronological)
        self.conversation_history: deque = deque(maxlen=100)

        # Quick lookups
        self.entity_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.entity_relationships: Dict[str, Set[str]] = defaultdict(
            set
        )  # entity_id -> related entity_ids

        # Temporal decay settings
        self.decay_rate = 0.95  # How fast old entities fade
        self.mention_boost = 0.2  # How much weight recent mentions add

        # Load persisted data
        self._load_graph()

    def _generate_entity_id(self, entity_type: str, value: Any) -> str:
        """Generate unique ID for an entity"""
        key = f"{entity_type}:{str(value).lower()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _generate_turn_id(self) -> str:
        """Generate unique ID for a conversation turn"""
        return f"turn_{int(time.time() * 1000)}"

    def add_entity(
        self, entity_type: str, value: Any, metadata: Optional[Dict] = None
    ) -> Entity:
        """
        Add or update an entity in the graph
        Returns the entity (new or existing)
        """
        entity_id = self._generate_entity_id(entity_type, value)
        current_time = time.time()

        if entity_id in self.entities:
            # Update existing entity
            entity = self.entities[entity_id]
            entity.last_mentioned = current_time
            entity.mention_count += 1
            if metadata:
                entity.metadata.update(metadata)
        else:
            # Create new entity
            entity = Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                value=value,
                first_mentioned=current_time,
                last_mentioned=current_time,
                mention_count=1,
                metadata=metadata or {},
            )
            self.entities[entity_id] = entity
            self.entity_by_type[entity_type].add(entity_id)

        logger.debug(f"Entity added/updated: {entity_type}={value} (id={entity_id})")
        return entity

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        metadata: Optional[Dict] = None,
    ) -> Relationship:
        """
        Add or strengthen a relationship between entities
        """
        rel_key = (source_id, target_id, relationship_type)
        current_time = time.time()

        if rel_key in self.relationships:
            # Strengthen existing relationship
            rel = self.relationships[rel_key]
            rel.strength = min(1.0, rel.strength + 0.1)
            if metadata:
                rel.metadata.update(metadata)
        else:
            # Create new relationship
            rel = Relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                created_at=current_time,
                strength=1.0,
                metadata=metadata or {},
            )
            self.relationships[rel_key] = rel

            # Update quick lookup
            self.entity_relationships[source_id].add(target_id)
            self.entity_relationships[target_id].add(source_id)

        logger.debug(f"Relationship: {source_id} --{relationship_type}--> {target_id}")
        return rel

    def record_turn(
        self,
        user_id: str,
        user_input: str,
        alice_response: str,
        intent: str,
        entities: Dict[str, List[Any]],
    ) -> ConversationTurn:
        """
        Record a conversation turn and extract entities
        Returns the turn object
        """
        turn_id = self._generate_turn_id()
        entity_ids = []

        # Add entities to graph and create relationships
        for entity_type, entity_values in entities.items():
            for value in entity_values:
                if not value or value == "Unknown":
                    continue

                entity = self.add_entity(entity_type, value)
                entity_ids.append(entity.entity_id)

                # Create relationships between entities in same turn
                for existing_id in entity_ids[:-1]:
                    self.add_relationship(
                        existing_id, entity.entity_id, "mentioned_with"
                    )

        # Create conversation turn
        turn = ConversationTurn(
            turn_id=turn_id,
            user_id=user_id,
            user_input=user_input,
            alice_response=alice_response,
            intent=intent,
            entities=entity_ids,
            timestamp=time.time(),
        )

        self.conversation_history.append(turn)

        logger.debug(f"Turn recorded: {turn_id} with {len(entity_ids)} entities")
        return turn

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.entity_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_related_entities(self, entity_id: str, max_depth: int = 1) -> List[Entity]:
        """
        Get entities related to the given entity
        max_depth: how many relationship hops to traverse
        """
        if entity_id not in self.entities:
            return []

        related_ids = set()
        to_explore = {entity_id}
        explored = set()

        for _ in range(max_depth):
            next_level = set()
            for eid in to_explore:
                if eid in explored:
                    continue
                explored.add(eid)

                # Get directly related entities
                related = self.entity_relationships.get(eid, set())
                next_level.update(related)
                related_ids.update(related)

            to_explore = next_level - explored

        return [self.entities[eid] for eid in related_ids if eid in self.entities]

    def get_recent_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 10,
        within_seconds: Optional[float] = None,
    ) -> List[Entity]:
        """
        Get recently mentioned entities
        """
        current_time = time.time()

        # Filter by type if specified
        if entity_type:
            candidates = self.get_entities_by_type(entity_type)
        else:
            candidates = list(self.entities.values())

        # Filter by time if specified
        if within_seconds:
            cutoff_time = current_time - within_seconds
            candidates = [e for e in candidates if e.last_mentioned >= cutoff_time]

        # Sort by recency and relevance
        def relevance_score(entity: Entity) -> float:
            # Combine recency and mention count
            time_diff = current_time - entity.last_mentioned
            recency = 1.0 / (1.0 + time_diff / 3600)  # Decay over hours
            frequency = min(1.0, entity.mention_count / 10.0)
            return recency * 0.7 + frequency * 0.3

        candidates.sort(key=relevance_score, reverse=True)
        return candidates[:limit]

    def get_conversation_history(
        self, limit: int = 10, user_id: Optional[str] = None
    ) -> List[ConversationTurn]:
        """Get recent conversation turns"""
        turns = list(self.conversation_history)

        if user_id:
            turns = [t for t in turns if t.user_id == user_id]

        return turns[-limit:]

    def query_context(self, query: str) -> Dict[str, Any]:
        """
        Natural language query interface for context
        Examples:
        - "what did we discuss about weather?"
        - "what notes did I create today?"
        - "what was the last location mentioned?"
        """
        query_lower = query.lower()
        results = {"query": query, "entities": [], "relationships": [], "turns": []}

        # Simple keyword-based matching (can be enhanced with NLP)

        # Time-based queries
        if "today" in query_lower:
            time_threshold = time.time() - 86400
            entities = [
                e for e in self.entities.values() if e.last_mentioned >= time_threshold
            ]
            results["entities"] = [e.to_dict() for e in entities[:10]]

        elif "last" in query_lower or "recent" in query_lower:
            entities = self.get_recent_entities(limit=5)
            results["entities"] = [e.to_dict() for e in entities]

        # Type-based queries
        for entity_type in ["weather", "note", "location", "person"]:
            if entity_type in query_lower:
                entities = self.get_entities_by_type(entity_type)
                results["entities"] = [e.to_dict() for e in entities[:10]]
                break

        return results

    def get_context_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive context summary for response generation
        This replaces all the overlapping context systems
        """
        recent_turns = self.get_conversation_history(limit=5, user_id=user_id)
        recent_entities = self.get_recent_entities(limit=20)

        summary = {
            "user_id": user_id,
            "recent_topics": list(
                set([e.value for e in recent_entities if e.entity_type == "topic"])
            )[:5],
            "recent_locations": list(
                set([e.value for e in recent_entities if e.entity_type == "location"])
            )[:3],
            "conversation_history": [
                {
                    "user": turn.user_input,
                    "alice": turn.alice_response,
                    "intent": turn.intent,
                    "timestamp": turn.timestamp,
                }
                for turn in recent_turns
            ],
            "entity_counts": {
                etype: len(entities) for etype, entities in self.entity_by_type.items()
            },
            "last_intent": recent_turns[-1].intent if recent_turns else None,
        }

        return summary

    def apply_temporal_decay(self):
        """
        Apply temporal decay to entity relevance
        Older entities gradually fade unless re-mentioned
        """
        current_time = time.time()

        for entity in self.entities.values():
            time_diff = current_time - entity.last_mentioned
            hours_ago = time_diff / 3600

            # Entities not mentioned in 24+ hours start to fade
            if hours_ago > 24:
                # This could adjust internal weights/scores
                # For now, just log very old entities
                if hours_ago > 168:  # 1 week
                    logger.debug(
                        f"Entity {entity.entity_id} hasn't been mentioned in {hours_ago:.1f} hours"
                    )

    def clear_old_data(self, older_than_days: int = 30):
        """Remove entities and turns older than specified days"""
        cutoff_time = time.time() - (older_than_days * 86400)

        # Remove old entities
        to_remove = [
            eid
            for eid, entity in self.entities.items()
            if entity.last_mentioned < cutoff_time
        ]

        for eid in to_remove:
            entity = self.entities[eid]
            self.entity_by_type[entity.entity_type].discard(eid)
            del self.entities[eid]

            # Remove relationships
            to_remove_rels = [
                key
                for key in self.relationships.keys()
                if key[0] == eid or key[1] == eid
            ]
            for key in to_remove_rels:
                del self.relationships[key]

        logger.info(
            f"Cleared {len(to_remove)} old entities (older than {older_than_days} days)"
        )

    def _load_graph(self):
        """Load graph from disk"""
        graph_file = self.data_dir / "context_graph.json"
        if graph_file.exists():
            try:
                with open(graph_file, "r") as f:
                    data = json.load(f)

                # Load entities
                for entity_data in data.get("entities", []):
                    entity = Entity.from_dict(entity_data)
                    self.entities[entity.entity_id] = entity
                    self.entity_by_type[entity.entity_type].add(entity.entity_id)

                # Load relationships
                for rel_data in data.get("relationships", []):
                    rel = Relationship.from_dict(rel_data)
                    key = (rel.source_id, rel.target_id, rel.relationship_type)
                    self.relationships[key] = rel
                    self.entity_relationships[rel.source_id].add(rel.target_id)
                    self.entity_relationships[rel.target_id].add(rel.source_id)

                logger.info(
                    f"Loaded context graph: "
                    f"{len(self.entities)} entities, "
                    f"{len(self.relationships)} relationships"
                )
            except Exception as e:
                logger.error(f"Failed to load context graph: {e}")

    def _save_graph(self):
        """Save graph to disk"""
        graph_file = self.data_dir / "context_graph.json"
        try:
            data = {
                "entities": [e.to_dict() for e in self.entities.values()],
                "relationships": [r.to_dict() for r in self.relationships.values()],
                "saved_at": time.time(),
            }

            with open(graph_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved context graph to {graph_file}")
        except Exception as e:
            logger.error(f"Failed to save context graph: {e}")

    def save(self):
        """Public method to trigger save"""
        self._save_graph()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the context graph"""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entities_by_type": {
                etype: len(entities) for etype, entities in self.entity_by_type.items()
            },
            "conversation_turns": len(self.conversation_history),
            "avg_mentions_per_entity": sum(
                e.mention_count for e in self.entities.values()
            )
            / max(1, len(self.entities)),
        }


# ---------------------------------------------------------------------------
# Symbolic World Graph  (typed knowledge graph with forward-chaining inference)
# ---------------------------------------------------------------------------

class NodeType:
    USER = "user"
    PERSON = "person"
    TOPIC = "topic"
    EVENT = "event"
    LOCATION = "location"
    ARTIST = "artist"
    PREFERENCE = "preference"
    DEADLINE = "deadline"
    TASK = "task"
    EMOTION = "emotion"


class RelationType:
    CONCERNED_ABOUT = "concerned_about"
    PREFERS = "prefers"
    DISLIKES = "dislikes"
    RELATED_TO = "related_to"
    CLUSTER_WITH = "cluster_with"
    HAS_ATTRIBUTE = "has_attribute"
    OCCURRED_AT = "occurred_at"
    ASSIGNED_TO = "assigned_to"
    CAUSED_BY = "caused_by"
    MAY_NEED = "may_need"


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()
        self.access_count += 1


@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    relation: str
    weight: float = 1.0
    attrs: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    inferred: bool = False   # True if materialised by a rule


@dataclass
class ForwardRule:
    """
    IF a node of head_type has >= min_count outgoing edges of relation pointing
    to nodes with tag tail_tag, THEN fire action(graph, head_node_id).
    """
    name: str
    head_type: str
    relation: str
    tail_tag: str
    min_count: int
    action: Callable[["WorldGraph", str], None]


def _fire_overwhelm(graph: "WorldGraph", node_id: str) -> None:
    node = graph.get_node(node_id)
    if node and not node.attrs.get("may_be_overwhelmed"):
        node.attrs["may_be_overwhelmed"] = True
        node.touch()
        logger.debug("WorldGraph: overwhelm detected for node %s", node_id)


_DEFAULT_WORLD_RULES: List[ForwardRule] = [
    ForwardRule(
        name="overwhelm_detection",
        head_type=NodeType.USER,
        relation=RelationType.CONCERNED_ABOUT,
        tail_tag="deadline",
        min_count=3,
        action=_fire_overwhelm,
    ),
]


class ForwardChainingRuleEngine:
    """Evaluates all registered rules against the current graph state."""

    def __init__(self, rules: Optional[List[ForwardRule]] = None) -> None:
        self._rules: List[ForwardRule] = list(rules or _DEFAULT_WORLD_RULES)

    def add_rule(self, rule: ForwardRule) -> None:
        self._rules.append(rule)

    def run(self, graph: "WorldGraph") -> int:
        """Run one inference pass. Returns number of rules that fired."""
        fired = 0
        for rule in self._rules:
            for node in graph.nodes_of_type(rule.head_type):
                matching = [
                    e for e in graph.outgoing_edges(node.node_id)
                    if e.relation == rule.relation
                    and rule.tail_tag in (graph.get_node(e.to_id) or GraphNode("", "")).tags
                ]
                if len(matching) >= rule.min_count:
                    rule.action(graph, node.node_id)
                    fired += 1
        return fired


def _wg_infer_preferences(graph: "WorldGraph") -> None:
    """If a user has >= 3 play references to the same artist, add a PREFERS edge."""
    for user_node in graph.nodes_of_type(NodeType.USER):
        artist_counts: Dict[str, int] = defaultdict(int)
        for edge in graph.outgoing_edges(user_node.node_id):
            t = graph.get_node(edge.to_id)
            if t and t.node_type == NodeType.ARTIST:
                artist_counts[edge.to_id] += 1
        for artist_id, count in artist_counts.items():
            if count >= 3:
                graph.add_edge(GraphEdge(
                    from_id=user_node.node_id,
                    to_id=artist_id,
                    relation=RelationType.PREFERS,
                    weight=min(1.0, count / 10.0),
                    inferred=True,
                ))


class WorldGraph:
    """
    In-memory typed knowledge graph with forward-chaining inference.

    Extracts facts from conversation turns (entities, topics, sentiment) and
    materialises implied edges/attributes via the ForwardChainingRuleEngine.
    Thread-safe for concurrent reads/writes in the main ALICE event loop.
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        rule_engine: Optional[ForwardChainingRuleEngine] = None,
    ) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._out_edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._wg_lock = threading.RLock()
        self._rule_engine = rule_engine or ForwardChainingRuleEngine()
        self._persistence_path = Path(persistence_path) if persistence_path else None
        if self._persistence_path and self._persistence_path.exists():
            self._load()

    def upsert_node(self, node: GraphNode) -> GraphNode:
        with self._wg_lock:
            existing = self._nodes.get(node.node_id)
            if existing:
                existing.attrs.update(node.attrs)
                existing.tags = list(set(existing.tags + node.tags))
                existing.touch()
                return existing
            self._nodes[node.node_id] = node
            return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def nodes_of_type(self, node_type: str) -> List[GraphNode]:
        with self._wg_lock:
            return [n for n in self._nodes.values() if n.node_type == node_type]

    def add_edge(self, edge: GraphEdge) -> None:
        with self._wg_lock:
            for existing in self._out_edges[edge.from_id]:
                if existing.to_id == edge.to_id and existing.relation == edge.relation:
                    existing.weight = max(existing.weight, edge.weight)
                    return
            self._out_edges[edge.from_id].append(edge)

    def outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        with self._wg_lock:
            return list(self._out_edges.get(node_id, []))

    def edges_of_relation(self, relation: str) -> List[GraphEdge]:
        with self._wg_lock:
            return [e for edges in self._out_edges.values()
                    for e in edges if e.relation == relation]

    def update_from_turn(
        self,
        intent: str,
        entities: Dict[str, Any],
        sentiment: Optional[str] = None,
        topics: Optional[List[str]] = None,
        user_id: str = "alice_user",
    ) -> int:
        """Extract facts from a turn and persist them in the graph. Returns edges added."""
        added = 0
        with self._wg_lock:
            self.upsert_node(GraphNode(node_id=user_id, node_type=NodeType.USER))
            domain, _, action = intent.partition(":")
            for key, value in (entities or {}).items():
                if not isinstance(value, str) or not value.strip():
                    continue
                node_id = f"{key}:{value.lower()}"
                ntype = self._infer_node_type(key, domain)
                n = GraphNode(node_id=node_id, node_type=ntype, attrs={"value": value})
                if ntype == NodeType.DEADLINE:
                    n.tags.append("deadline")
                self.upsert_node(n)
                if ntype in (NodeType.DEADLINE, NodeType.EVENT, NodeType.TASK):
                    self.add_edge(GraphEdge(from_id=user_id, to_id=node_id,
                                           relation=RelationType.CONCERNED_ABOUT, weight=0.8))
                    added += 1
                if ntype == NodeType.ARTIST and action == "play":
                    self.add_edge(GraphEdge(from_id=user_id, to_id=node_id,
                                           relation="played", weight=0.5))
                    added += 1
            for topic in (topics or []):
                tid = f"topic:{topic.lower()}"
                self.upsert_node(GraphNode(node_id=tid, node_type=NodeType.TOPIC,
                                          attrs={"name": topic}))
                self.add_edge(GraphEdge(from_id=user_id, to_id=tid,
                                       relation=RelationType.CONCERNED_ABOUT, weight=0.5))
                added += 1
            if sentiment and sentiment != "neutral":
                user_node = self._nodes[user_id]
                user_node.attrs[f"last_sentiment_{domain}"] = sentiment
        self._rule_engine.run(self)
        _wg_infer_preferences(self)
        if self._persistence_path:
            self._save()
        return added

    def may_be_overwhelmed(self, user_id: str = "alice_user") -> bool:
        node = self.get_node(user_id)
        return bool(node and node.attrs.get("may_be_overwhelmed"))

    def preferred_artists(self, user_id: str = "alice_user") -> List[str]:
        return [e.to_id.split(":", 1)[-1]
                for e in self.outgoing_edges(user_id)
                if e.relation == RelationType.PREFERS]

    def stats(self) -> Dict[str, int]:
        with self._wg_lock:
            return {
                "nodes": len(self._nodes),
                "edges": sum(len(v) for v in self._out_edges.values()),
            }

    def _save(self) -> None:
        try:
            data = {
                "nodes": [
                    {"node_id": n.node_id, "node_type": n.node_type,
                     "attrs": n.attrs, "tags": n.tags,
                     "created_at": n.created_at, "updated_at": n.updated_at,
                     "access_count": n.access_count}
                    for n in self._nodes.values()
                ],
                "edges": [
                    {"from_id": e.from_id, "to_id": e.to_id,
                     "relation": e.relation, "weight": e.weight,
                     "attrs": e.attrs, "created_at": e.created_at, "inferred": e.inferred}
                    for edges in self._out_edges.values() for e in edges
                ],
            }
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persistence_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._persistence_path)
        except Exception as exc:
            logger.warning("WorldGraph: save failed: %s", exc)

    def _load(self) -> None:
        try:
            data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
            for nd in data.get("nodes", []):
                self._nodes[nd["node_id"]] = GraphNode(
                    node_id=nd["node_id"], node_type=nd["node_type"],
                    attrs=nd.get("attrs", {}), tags=nd.get("tags", []),
                    created_at=nd.get("created_at", ""),
                    updated_at=nd.get("updated_at", ""),
                    access_count=nd.get("access_count", 0),
                )
            for ed in data.get("edges", []):
                self._out_edges[ed["from_id"]].append(GraphEdge(
                    from_id=ed["from_id"], to_id=ed["to_id"],
                    relation=ed["relation"], weight=ed.get("weight", 1.0),
                    attrs=ed.get("attrs", {}), created_at=ed.get("created_at", ""),
                    inferred=ed.get("inferred", False),
                ))
            logger.info("WorldGraph: loaded %d nodes, %d edges",
                        len(self._nodes), sum(len(v) for v in self._out_edges.values()))
        except Exception as exc:
            logger.warning("WorldGraph: load failed: %s", exc)

    @staticmethod
    def _infer_node_type(entity_key: str, domain: str) -> str:
        key = entity_key.lower()
        if "artist" in key or "band" in key:
            return NodeType.ARTIST
        if "location" in key or "city" in key or "place" in key:
            return NodeType.LOCATION
        if "date" in key or "time" in key or "deadline" in key:
            return NodeType.DEADLINE
        if "person" in key or "contact" in key or "who" in key:
            return NodeType.PERSON
        if domain in ("calendar", "schedule"):
            return NodeType.EVENT
        if domain in ("tasks", "todo"):
            return NodeType.TASK
        return NodeType.TOPIC


_world_graph_instance: Optional[WorldGraph] = None
_world_graph_lock = threading.Lock()


def get_world_graph(persistence_path: Optional[str] = None) -> WorldGraph:
    """Return the process-wide singleton WorldGraph."""
    global _world_graph_instance
    if _world_graph_instance is None:
        with _world_graph_lock:
            if _world_graph_instance is None:
                _world_graph_instance = WorldGraph(persistence_path=persistence_path)
    return _world_graph_instance
