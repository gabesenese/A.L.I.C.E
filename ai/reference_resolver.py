"""
Reference Resolver for A.L.I.C.E
Resolves pronouns ("it", "that") and definite references ("the grocery list")
using world state and recent conversation. Keeps A.L.I.C.E context-aware.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .world_state import WorldState, WorldEntity, EntityKind, get_world_state

logger = logging.getLogger(__name__)


@dataclass
class ResolvedReference:
    """A reference that was resolved to an entity."""
    raw_text: str
    resolved_id: str
    resolved_label: str
    kind: EntityKind
    confidence: float


@dataclass
class ResolutionResult:
    """Result of resolving references in user input."""
    resolved_input: str
    bindings: Dict[str, ResolvedReference] = field(default_factory=dict)
    entities_to_use: Dict[str, Any] = field(default_factory=dict)


class ReferenceResolver:
    """
    Resolves vague references in user input using world state.
    Examples: "delete it" -> last note; "the grocery list" -> note titled "grocery list".
    """

    PRONOUNS = ["it", "this", "that", "the one", "the first", "the last"]
    DEFINITE_PATTERNS = [
        (r"the\s+(.+?)\s+list\b", EntityKind.NOTE),
        (r"the\s+(.+?)\s+note\b", EntityKind.NOTE),
        (r"that\s+(.+?)\s+note\b", EntityKind.NOTE),
        (r"my\s+(.+?)\s+list\b", EntityKind.NOTE),
        (r"the\s+(.+?)\s+email\b", EntityKind.EMAIL),
        (r"that\s+email\b", EntityKind.EMAIL),
        (r"the\s+(.+?)\s+event\b", EntityKind.EVENT),
        (r"that\s+event\b", EntityKind.EVENT),
    ]

    def __init__(self, world_state: Optional[WorldState] = None):
        self.world_state = world_state or get_world_state()

    def resolve(self, user_input: str) -> ResolutionResult:
        """
        Resolve references in user input using world state.
        Returns resolved input (or original), bindings, and suggested entity updates.
        """
        bindings: Dict[str, ResolvedReference] = {}
        entities_to_use: Dict[str, Any] = {}
        resolved_input = user_input
        snap = self.world_state.snapshot()

        # 1) Pronoun resolution: "it", "this", "that" -> last salient entity
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
                logger.info(f"[ReferenceResolver] '{pron}' -> {entity.label} ({entity.kind.value})")

        # 2) Definite noun phrases: "the grocery list", "that email"
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
                logger.info(f"[ReferenceResolver] '{phrase}' -> {entity.label} ({entity.kind.value})")

        # 3) Optional: substitute resolved refs for downstream clarity (e.g. "delete it" -> "delete grocery list")
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
        """Resolve pronoun to most recent relevant entity from world state."""
        last_intent = snap.get("last_intent") or ""
        last_entities = snap.get("last_entities") or {}
        recent = snap.get("recent_entity_labels") or []
        by_kind = snap.get("recent_entities_by_kind") or {}

        if "notes" in last_intent or "note" in last_intent:
            labels = by_kind.get(EntityKind.NOTE.value, [])
            if labels:
                label = labels[0]
                return self.world_state.find_entity_by_label(label, EntityKind.NOTE)
        if "email" in last_intent:
            labels = by_kind.get(EntityKind.EMAIL.value, [])
            if labels:
                return self.world_state.find_entity_by_label(labels[0], EntityKind.EMAIL)

        for label in recent:
            e = self.world_state.find_entity_by_label(label)
            if e:
                return e
        return None

    def _resolve_definite(self, phrase: str, inner: str, kind: EntityKind, snap: Dict[str, Any]) -> Optional[WorldEntity]:
        """Resolve definite NP like 'the grocery list' or 'that email'."""
        if inner:
            return self.world_state.find_entity_by_label(inner, kind)
        by_kind = snap.get("recent_entities_by_kind") or {}
        labels = by_kind.get(kind.value, [])
        if labels:
            return self.world_state.find_entity_by_label(labels[0], kind)
        return None


_resolver: Optional[ReferenceResolver] = None


def get_reference_resolver(world_state: Optional[WorldState] = None) -> ReferenceResolver:
    global _resolver
    if _resolver is None:
        _resolver = ReferenceResolver(world_state=world_state)
    return _resolver
