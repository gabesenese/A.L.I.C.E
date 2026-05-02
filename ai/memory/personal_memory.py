"""Structured personal-memory layer on top of the existing MemorySystem."""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.memory.personal_memory_constants import (
    DEFAULT_PERSONAL_DOMAIN,
    DEFAULT_PERSONAL_KIND,
    DEFAULT_PERSONAL_SCOPE,
    PERSONAL_MEMORY_DOMAINS,
    PERSONAL_MEMORY_KINDS,
    PERSONAL_MEMORY_SCOPES,
)


class PersonalMemoryStore:
    def __init__(self, memory_system: Any):
        self.memory = memory_system

    @staticmethod
    def _normalize_token(value: str, allowed: set[str], default: str) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return token if token in allowed else default

    @staticmethod
    def _normalize_tags(tags: List[str]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for raw in tags:
            token = str(raw or "").strip().lower()
            token = re.sub(r"[^a-z0-9_:-]+", "_", token).strip("_")
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        return normalized

    def store_structured_memory(
        self,
        content: str,
        domain: str,
        kind: str,
        scope: str,
        confidence: float,
        source: str,
        trace_id: str | None = None,
        importance: float = 0.7,
    ) -> str:
        normalized_domain = self._normalize_token(
            domain, PERSONAL_MEMORY_DOMAINS, DEFAULT_PERSONAL_DOMAIN
        )
        normalized_kind = self._normalize_token(
            kind, PERSONAL_MEMORY_KINDS, DEFAULT_PERSONAL_KIND
        )
        normalized_scope = self._normalize_token(
            scope, PERSONAL_MEMORY_SCOPES, DEFAULT_PERSONAL_SCOPE
        )

        structured_context = {
            "memory_schema": "personal_v1",
            "domain": normalized_domain,
            "kind": normalized_kind,
            "scope": normalized_scope,
            "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
            "source": str(source or "conversation"),
            "trace_id": str(trace_id or "").strip(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        tags = self._normalize_tags(
            [
                "structured:personal",
                f"domain:{normalized_domain}",
                f"kind:{normalized_kind}",
                f"scope:{normalized_scope}",
            ]
        )

        return str(
            self.memory.store_memory(
                content=str(content or "").strip(),
                memory_type="episodic",
                context=structured_context,
                importance=max(0.0, min(1.0, float(importance or 0.7))),
                tags=tags,
            )
        )

    @staticmethod
    def _has_tag(row: Dict[str, Any], prefix: str, value: str) -> bool:
        tags = [str(t or "").strip().lower() for t in list(row.get("tags") or [])]
        return f"{prefix}:{value}" in tags

    def _filter_structured(
        self,
        candidates: List[Dict[str, Any]],
        *,
        domain: Optional[str],
        kind: Optional[str],
        scope: Optional[str],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in candidates:
            tags = [str(t or "").strip().lower() for t in list(row.get("tags") or [])]
            if "structured:personal" not in tags:
                continue
            if domain and not self._has_tag(row, "domain", domain):
                continue
            if kind and not self._has_tag(row, "kind", kind):
                continue
            if scope and not self._has_tag(row, "scope", scope):
                continue
            out.append(row)
        return out

    def _iter_structured_entries(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        sources = []
        for attr in ("episodic_memory", "semantic_memory", "procedural_memory", "document_memory"):
            bucket = getattr(self.memory, attr, None)
            if isinstance(bucket, list):
                sources.extend(bucket)
        for item in sources:
            tags = [str(t or "").strip().lower() for t in list(getattr(item, "tags", []) or [])]
            if "structured:personal" not in tags:
                continue
            ctx = getattr(item, "context", {}) if hasattr(item, "context") else {}
            rows.append(
                {
                    "id": getattr(item, "id", ""),
                    "content": str(getattr(item, "content", "") or ""),
                    "type": str(getattr(item, "memory_type", "") or ""),
                    "timestamp": str(getattr(item, "timestamp", "") or ""),
                    "importance": float(getattr(item, "importance", 0.5) or 0.5),
                    "tags": tags,
                    "context": dict(ctx or {}),
                    "source": str((ctx or {}).get("source") or "conversation"),
                }
            )
        return rows

    @staticmethod
    def _recency_score(ts: str) -> float:
        if not ts:
            return 0.5
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
            return math.exp(-0.03 * age_hours)
        except Exception:
            return 0.5

    @staticmethod
    def _lexical_similarity(query: str, content: str) -> float:
        q_tokens = {t for t in re.findall(r"[a-z0-9]+", str(query).lower()) if len(t) > 2}
        c_tokens = {t for t in re.findall(r"[a-z0-9]+", str(content).lower()) if len(t) > 2}
        if not q_tokens or not c_tokens:
            return 0.0
        overlap = len(q_tokens.intersection(c_tokens))
        return float(overlap) / float(max(1, len(q_tokens)))

    def retrieve_structured_memory(
        self,
        query: str,
        *,
        domain: Optional[str] = None,
        kind: Optional[str] = None,
        scope: Optional[str] = None,
        top_k: int = 8,
        min_similarity: float = 0.35,
    ) -> List[Dict[str, Any]]:
        normalized_domain = (
            self._normalize_token(domain, PERSONAL_MEMORY_DOMAINS, DEFAULT_PERSONAL_DOMAIN)
            if domain
            else None
        )
        normalized_kind = (
            self._normalize_token(kind, PERSONAL_MEMORY_KINDS, DEFAULT_PERSONAL_KIND)
            if kind
            else None
        )
        normalized_scope = (
            self._normalize_token(scope, PERSONAL_MEMORY_SCOPES, DEFAULT_PERSONAL_SCOPE)
            if scope
            else None
        )

        # Phase 1: direct scan of structured entries to avoid vector misses.
        structured_rows = self._iter_structured_entries()
        filtered = self._filter_structured(
            structured_rows,
            domain=normalized_domain,
            kind=normalized_kind,
            scope=normalized_scope,
        )
        if not filtered and normalized_domain:
            filtered = self._filter_structured(
                structured_rows,
                domain=normalized_domain,
                kind=None,
                scope=None,
            )

        ranked: List[Dict[str, Any]] = []
        for row in filtered:
            ctx = dict(row.get("context") or {})
            confidence = max(0.0, min(1.0, float(ctx.get("confidence", row.get("importance", 0.5)) or 0.5)))
            importance = max(0.0, min(1.0, float(row.get("importance", 0.5) or 0.5)))
            recency = self._recency_score(str(row.get("timestamp") or ""))
            lexical = self._lexical_similarity(query, str(row.get("content") or ""))
            combined = confidence * 0.35 + importance * 0.25 + recency * 0.25 + lexical * 0.15
            ranked.append(
                {
                    **row,
                    "similarity": lexical,
                    "weighted_score": max(0.0, min(1.0, combined)),
                    "score_components": {
                        "confidence": round(confidence, 4),
                        "importance": round(importance, 4),
                        "recency": round(recency, 4),
                        "lexical_similarity": round(lexical, 4),
                    },
                }
            )
        ranked.sort(
            key=lambda r: (
                float(r.get("weighted_score", 0.0)),
                float((r.get("context") or {}).get("confidence", 0.0)),
                str(r.get("timestamp", "")),
            ),
            reverse=True,
        )
        # Optional quality floor.
        return [
            row
            for row in ranked
            if float(row.get("weighted_score", 0.0)) >= float(min_similarity or 0.0) * 0.5
        ][:top_k]
