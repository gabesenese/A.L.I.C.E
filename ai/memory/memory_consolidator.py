"""Consolidation utilities for structured personal memories."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(t) > 2}


def _similarity(a: str, b: str) -> float:
    aa = _tokenize(a)
    bb = _tokenize(b)
    if not aa or not bb:
        return 0.0
    overlap = len(aa.intersection(bb))
    return float(overlap) / float(max(1, len(aa.union(bb))))


class MemoryConsolidator:
    def __init__(self, memory_system: Any):
        self.memory = memory_system

    def _iter_structured(self) -> List[Any]:
        entries: List[Any] = []
        for attr in ("episodic_memory", "semantic_memory", "procedural_memory", "document_memory"):
            bucket = getattr(self.memory, attr, None)
            if isinstance(bucket, list):
                entries.extend(bucket)
        out: List[Any] = []
        for entry in entries:
            tags = [str(t or "").strip().lower() for t in list(getattr(entry, "tags", []) or [])]
            if "structured:personal" in tags:
                out.append(entry)
        return out

    def consolidate_recent(self, *, domain: str | None = None, kind: str | None = None, scope: str | None = None) -> Dict[str, Any]:
        entries = self._iter_structured()
        groups: Dict[str, List[Any]] = {}
        for entry in entries:
            ctx = dict(getattr(entry, "context", {}) or {})
            d = str(ctx.get("domain") or "").strip().lower()
            k = str(ctx.get("kind") or "").strip().lower()
            s = str(ctx.get("scope") or "").strip().lower()
            if domain and d != str(domain).lower():
                continue
            if kind and k != str(kind).lower():
                continue
            if scope and s != str(scope).lower():
                continue
            key = f"{d}|{k}|{s}"
            groups.setdefault(key, []).append(entry)

        merged = 0
        superseded = 0
        for _, bucket in groups.items():
            if len(bucket) < 2:
                continue
            bucket_sorted = sorted(bucket, key=lambda e: str(getattr(e, "timestamp", "")), reverse=True)
            anchor = bucket_sorted[0]
            anchor_ctx = dict(getattr(anchor, "context", {}) or {})
            for candidate in bucket_sorted[1:]:
                cand_ctx = dict(getattr(candidate, "context", {}) or {})
                sim = _similarity(getattr(anchor, "content", ""), getattr(candidate, "content", ""))
                if sim < 0.68:
                    continue
                anchor_conf = float(anchor_ctx.get("confidence", getattr(anchor, "importance", 0.6)) or 0.6)
                cand_conf = float(cand_ctx.get("confidence", getattr(candidate, "importance", 0.6)) or 0.6)
                new_conf = min(0.99, max(anchor_conf, cand_conf) + 0.05)
                anchor_ctx["confidence"] = new_conf
                anchor_ctx["consolidated_at"] = datetime.now(timezone.utc).isoformat()
                src_ids = list(anchor_ctx.get("source_trace_ids", []) or [])
                if anchor_ctx.get("trace_id"):
                    src_ids.append(str(anchor_ctx.get("trace_id")))
                if cand_ctx.get("trace_id"):
                    src_ids.append(str(cand_ctx.get("trace_id")))
                anchor_ctx["source_trace_ids"] = sorted({sid for sid in src_ids if sid})
                anchor.context = anchor_ctx
                anchor.importance = min(0.99, max(float(getattr(anchor, "importance", 0.7) or 0.7), new_conf))

                cand_ctx["superseded"] = True
                cand_ctx["superseded_by"] = str(getattr(anchor, "id", ""))
                cand_ctx["superseded_at"] = datetime.now(timezone.utc).isoformat()
                candidate.context = cand_ctx
                superseded += 1
                merged += 1

        return {"merged_count": merged, "superseded_count": superseded}
