"""Structured personal-memory layer on top of the existing MemorySystem."""

from __future__ import annotations

import re
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

        # Phase 1: broad semantic recall, then strict structured filtering.
        candidates = list(
            self.memory.search(
                query=query,
                top_k=max(int(top_k or 8) * 4, 12),
                min_similarity=min_similarity,
                weighted=True,
            )
            or []
        )
        structured = self._filter_structured(
            candidates,
            domain=normalized_domain,
            kind=normalized_kind,
            scope=normalized_scope,
        )
        if structured:
            return structured[:top_k]

        # Phase 2: structured-any fallback, still no untagged memory fabrication.
        if normalized_domain or normalized_kind or normalized_scope:
            broader = self._filter_structured(
                candidates,
                domain=normalized_domain if normalized_domain else None,
                kind=None,
                scope=None,
            )
            if broader:
                return broader[:top_k]

        return []
