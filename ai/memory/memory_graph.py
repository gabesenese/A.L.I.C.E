from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_GRAPH_FILE = Path("data/memory_graph.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Entity:
    name: str
    entity_type: str  # project | person | concept | decision | event
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    from_key: str
    to_key: str
    relation_type: str  # led_to | blocked_by | part_of | involves | decided_on
    context: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryGraph:
    """Temporal entity graph. Replaces the corrupt entities.json store."""

    def __init__(self, graph_file: Path = _GRAPH_FILE):
        self._file = graph_file
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for k, v in data.get("entities", {}).items():
                self._entities[k] = Entity(**{
                    fk: fv for fk, fv in v.items()
                    if fk in Entity.__dataclass_fields__
                })
            for r in data.get("relations", []):
                self._relations.append(Relation(**{
                    fk: fv for fk, fv in r.items()
                    if fk in Relation.__dataclass_fields__
                }))
        except Exception:
            pass

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(
                {
                    "entities": {k: e.to_dict() for k, e in self._entities.items()},
                    "relations": [r.to_dict() for r in self._relations],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def upsert(
        self,
        name: str,
        entity_type: str,
        summary: str = "",
        tags: Optional[List[str]] = None,
    ) -> Entity:
        key = self._key(name)
        if key in self._entities:
            e = self._entities[key]
            if summary:
                e.summary = summary
            if tags:
                e.tags = list(dict.fromkeys(e.tags + tags))
            e.updated_at = _now_iso()
        else:
            e = Entity(name=name, entity_type=entity_type, summary=summary, tags=tags or [])
            self._entities[key] = e
        self._save()
        return e

    def relate(
        self,
        from_name: str,
        to_name: str,
        relation_type: str,
        context: str = "",
    ) -> None:
        fk, tk = self._key(from_name), self._key(to_name)
        for r in self._relations:
            if r.from_key == fk and r.to_key == tk and r.relation_type == relation_type:
                return
        self._relations.append(
            Relation(from_key=fk, to_key=tk, relation_type=relation_type, context=context)
        )
        self._save()

    def get(self, name: str) -> Optional[Entity]:
        return self._entities.get(self._key(name))

    def neighbors(self, name: str) -> List[Dict[str, str]]:
        key = self._key(name)
        results = []
        for r in self._relations:
            if r.from_key == key:
                results.append({"entity": r.to_key, "relation": r.relation_type, "context": r.context, "direction": "out"})
            elif r.to_key == key:
                results.append({"entity": r.from_key, "relation": r.relation_type, "context": r.context, "direction": "in"})
        return results

    def status_of(self, name: str) -> str:
        entity = self.get(name)
        if not entity:
            return f"No data on '{name}'."
        parts = [f"{entity.name} ({entity.entity_type})"]
        if entity.summary:
            parts.append(entity.summary)
        for n in self.neighbors(name)[:5]:
            arrow = "->" if n["direction"] == "out" else "<-"
            ctx = f" ({n['context']})" if n["context"] else ""
            parts.append(f"  {arrow} {n['relation']}: {n['entity']}{ctx}")
        return "\n".join(parts)

    def all_entities(self) -> List[Entity]:
        return list(self._entities.values())


_graph: Optional[MemoryGraph] = None


def get_memory_graph() -> MemoryGraph:
    global _graph
    if _graph is None:
        _graph = MemoryGraph()
    return _graph
