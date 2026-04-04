"""Canonical entity registry for cross-turn and cross-module entity identity."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EntityRecord:
    entity_id: str
    kind: str
    label: str
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.8
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mention_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EntityRegistry:
    """Thread-safe canonical entity store with alias and recency resolution."""

    def __init__(self, storage_path: str = "data/entity_registry.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: Dict[str, EntityRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            rows = (
                list(payload.get("entities") or []) if isinstance(payload, dict) else []
            )
            for row in rows:
                rec = EntityRecord(
                    entity_id=str(row.get("entity_id") or uuid.uuid4().hex),
                    kind=str(row.get("kind") or "unknown"),
                    label=str(row.get("label") or "").strip(),
                    aliases=[
                        str(x) for x in list(row.get("aliases") or []) if str(x).strip()
                    ],
                    confidence=float(row.get("confidence", 0.8) or 0.8),
                    source=str(row.get("source") or ""),
                    metadata=dict(row.get("metadata") or {}),
                    created_at=float(row.get("created_at", time.time()) or time.time()),
                    last_seen=float(row.get("last_seen", time.time()) or time.time()),
                    mention_count=int(row.get("mention_count", 1) or 1),
                )
                if rec.label:
                    self._records[rec.entity_id] = rec
        except Exception:
            return

    def _save(self) -> None:
        payload = {
            "updated_at": time.time(),
            "entities": [rec.to_dict() for rec in self._records.values()],
        }
        try:
            self.storage_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        except Exception:
            return

    def register(
        self,
        *,
        kind: str,
        label: str,
        aliases: Optional[List[str]] = None,
        source: str = "",
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: str = "",
    ) -> str:
        normalized_kind = str(kind or "unknown").strip().lower() or "unknown"
        normalized_label = str(label or "").strip()
        if not normalized_label:
            return ""

        alias_list = [str(a).strip() for a in list(aliases or []) if str(a).strip()]
        alias_list = list(dict.fromkeys(alias_list))

        with self._lock:
            existing_id = self._find_existing_id(
                normalized_kind, normalized_label, alias_list
            )
            now = time.time()
            if existing_id:
                rec = self._records[existing_id]
                rec.last_seen = now
                rec.mention_count += 1
                rec.confidence = max(float(rec.confidence), float(confidence or 0.0))
                if source:
                    rec.source = str(source)
                rec.metadata = {**rec.metadata, **dict(metadata or {})}
                merged_aliases = list(
                    dict.fromkeys(
                        list(rec.aliases or []) + alias_list + [normalized_label]
                    )
                )
                rec.aliases = merged_aliases[:20]
                self._save()
                return existing_id

            eid = str(entity_id or uuid.uuid4().hex)
            rec = EntityRecord(
                entity_id=eid,
                kind=normalized_kind,
                label=normalized_label,
                aliases=list(dict.fromkeys(alias_list + [normalized_label]))[:20],
                confidence=max(0.0, min(1.0, float(confidence or 0.0))),
                source=str(source or ""),
                metadata=dict(metadata or {}),
            )
            self._records[eid] = rec
            self._save()
            return eid

    def recent(self, *, limit: int = 10, kind: str = "") -> List[Dict[str, Any]]:
        normalized_kind = str(kind or "").strip().lower()
        with self._lock:
            rows = list(self._records.values())
            if normalized_kind:
                rows = [r for r in rows if r.kind == normalized_kind]
            rows.sort(key=lambda r: float(r.last_seen or 0.0), reverse=True)
            return [r.to_dict() for r in rows[: max(1, int(limit or 10))]]

    def resolve_reference(
        self,
        reference: str,
        *,
        preferred_kinds: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        needle = str(reference or "").strip().lower()
        if not needle:
            return None

        preferred = {
            str(k).strip().lower()
            for k in list(preferred_kinds or [])
            if str(k).strip()
        }
        with self._lock:
            best: Optional[EntityRecord] = None
            best_score = -1.0
            for rec in self._records.values():
                if preferred and rec.kind not in preferred:
                    continue

                labels = [str(rec.label or "").lower()] + [
                    str(a).lower() for a in list(rec.aliases or [])
                ]
                score = 0.0
                for candidate in labels:
                    if not candidate:
                        continue
                    if needle == candidate:
                        score = max(score, 1.0)
                    elif needle in candidate or candidate in needle:
                        score = max(score, 0.75)

                if score <= 0.0:
                    continue

                recency_bonus = min(
                    0.25,
                    max(
                        0.0,
                        (time.time() - float(rec.last_seen or 0.0)) / -3600.0 + 0.25,
                    ),
                )
                final = score + recency_bonus + (0.05 * min(rec.mention_count, 5))
                if final > best_score:
                    best_score = final
                    best = rec

            return best.to_dict() if best else None

    def resolve_label(
        self, reference: str, *, preferred_kinds: Optional[List[str]] = None
    ) -> str:
        hit = self.resolve_reference(reference, preferred_kinds=preferred_kinds)
        if not isinstance(hit, dict):
            return ""
        return str(hit.get("label") or "")

    def snapshot(self, *, limit: int = 40) -> Dict[str, Any]:
        rows = self.recent(limit=limit)
        return {
            "count": len(self._records),
            "entities": rows,
        }

    def _find_existing_id(self, kind: str, label: str, aliases: List[str]) -> str:
        low_label = label.lower()
        low_aliases = {a.lower() for a in aliases}
        for rec in self._records.values():
            if rec.kind != kind:
                continue
            if low_label == rec.label.lower():
                return rec.entity_id
            rec_aliases = {str(a).lower() for a in list(rec.aliases or [])}
            if low_label in rec_aliases:
                return rec.entity_id
            if low_aliases.intersection(rec_aliases):
                return rec.entity_id
        return ""


_entity_registry: EntityRegistry | None = None


def get_entity_registry(
    storage_path: str = "data/entity_registry.json",
) -> EntityRegistry:
    global _entity_registry
    if _entity_registry is None:
        _entity_registry = EntityRegistry(storage_path=storage_path)
    return _entity_registry
