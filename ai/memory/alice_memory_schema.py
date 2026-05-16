from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    id: str
    kind: str
    content: str
    topic: str = ""
    confidence: float = 0.5
    importance: int = 5
    source: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedMemory:
    record: MemoryRecord
    score: float
    reason: str
    confidence_label: str


@dataclass
class ActiveConceptThread:
    topic: str
    constraints: List[str]
    signals: List[str]
    last_user_inputs: List[str]
    updated_at: str
    confidence: float = 0.8


@dataclass
class ContextFrame:
    mode: str
    subject: str
    user_input: str
    active_concept_thread: Optional[ActiveConceptThread]
    verified_memories: List[RetrievedMemory]
    hint_memories: List[RetrievedMemory]
    project_state: Dict[str, Any]
    evidence_required: bool
    tool_required: bool
    notes: List[str]
