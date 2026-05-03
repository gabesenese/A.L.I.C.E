"""Structured personal-memory layer on top of the existing MemorySystem."""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.memory.memory_consolidator import MemoryConsolidator
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
        self.consolidator = MemoryConsolidator(memory_system)

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

        memory_id = str(
            self.memory.store_memory(
                content=str(content or "").strip(),
                memory_type="episodic",
                context=structured_context,
                importance=max(0.0, min(1.0, float(importance or 0.7))),
                tags=tags,
            )
        )
        try:
            self.consolidator.consolidate_recent(
                domain=normalized_domain,
                kind=normalized_kind,
                scope=normalized_scope,
            )
        except Exception:
            pass
        return memory_id

    def find_recent_structured_memories(
        self,
        *,
        domain: Optional[str] = None,
        kind: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        rows = self._iter_structured_entries()
        if domain:
            rows = [r for r in rows if self._has_tag(r, "domain", str(domain).lower())]
        if kind:
            rows = [r for r in rows if self._has_tag(r, "kind", str(kind).lower())]
        rows = [r for r in rows if not bool((r.get("context") or {}).get("superseded"))]
        rows.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        return rows[: max(1, int(top_k or 5))]

    def _update_entry_context(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        for attr in ("episodic_memory", "semantic_memory", "procedural_memory", "document_memory"):
            bucket = getattr(self.memory, attr, None)
            if not isinstance(bucket, list):
                continue
            for entry in bucket:
                if str(getattr(entry, "id", "")) != str(entry_id):
                    continue
                ctx = dict(getattr(entry, "context", {}) or {})
                ctx.update(dict(updates or {}))
                entry.context = ctx
                return True
        return False

    def forget_recent_memory(self, domain: str | None = None, kind: str | None = None) -> Dict[str, Any]:
        recent = self.find_recent_structured_memories(domain=domain, kind=kind, top_k=1)
        if not recent:
            return {"updated": False}
        target = recent[0]
        ok = self._update_entry_context(
            str(target.get("id", "")),
            {
                "invalid": True,
                "invalid_reason": "user_forget_request",
            },
        )
        return {"updated": bool(ok), "memory_id": str(target.get("id", ""))}

    def update_memory(
        self, memory_id: str, new_content: str, confidence: float | None = None
    ) -> Dict[str, Any]:
        for attr in ("episodic_memory", "semantic_memory", "procedural_memory", "document_memory"):
            bucket = getattr(self.memory, attr, None)
            if not isinstance(bucket, list):
                continue
            for entry in bucket:
                if str(getattr(entry, "id", "")) != str(memory_id):
                    continue
                entry.content = str(new_content or "").strip()
                ctx = dict(getattr(entry, "context", {}) or {})
                ctx["corrected"] = True
                if confidence is not None:
                    ctx["confidence"] = max(0.0, min(1.0, float(confidence)))
                entry.context = ctx
                return {"updated": True, "memory_id": str(memory_id)}
        return {"updated": False, "memory_id": str(memory_id)}

    def mark_memory_incorrect(self, memory_id: str, reason: str) -> Dict[str, Any]:
        ok = self._update_entry_context(
            str(memory_id),
            {
                "invalid": True,
                "invalid_reason": str(reason or "user_marked_incorrect"),
            },
        )
        return {"updated": bool(ok), "memory_id": str(memory_id)}

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
        result = self.retrieve_structured_memory_detailed(
            query,
            domain=domain,
            kind=kind,
            scope=scope,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return list(result.get("items") or [])

    @staticmethod
    def _normalize_content_for_dedupe(text: str) -> str:
        value = str(text or "").strip().lower()
        value = re.sub(r"^\s*[-*]\s*", "", value)
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _is_broad_mixed_turn_content(text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low.startswith("gabriel said:") and not low.startswith("user said:"):
            return False
        multi_topic_markers = (
            ("shopping", "ready to work"),
            ("shopping", "project"),
            ("shopping", "codebase"),
            ("gym", "ready to work"),
            ("lunch", "code"),
        )
        return any(a in low and b in low for a, b in multi_topic_markers)

    @staticmethod
    def _is_clean_personal_fragment(text: str) -> bool:
        low = str(text or "").strip().lower()
        clean_markers = (
            "did some shopping today",
            "went shopping today",
            "went to the gym today",
            "worked late today",
            "had lunch with",
            "stayed home today",
        )
        return any(marker in low for marker in clean_markers) and not low.startswith("gabriel said:")

    @staticmethod
    def _is_meta_recall_query_content(text: str) -> bool:
        low = str(text or "").strip().lower()
        return bool(
            re.search(
                r"\b(what did i talk about|what do you remember|what did i mention)\b",
                low,
            )
        )

    def retrieve_structured_memory_detailed(
        self,
        query: str,
        *,
        domain: Optional[str] = None,
        kind: Optional[str] = None,
        scope: Optional[str] = None,
        top_k: int = 8,
        min_similarity: float = 0.35,
    ) -> Dict[str, Any]:
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
        if not filtered and normalized_domain and normalized_scope is None:
            filtered = self._filter_structured(
                structured_rows,
                domain=normalized_domain,
                kind=None,
                scope=None,
            )

        ranked: List[Dict[str, Any]] = []
        raw_candidate_count = len(filtered)
        downranked_mixed_count = 0
        has_clean_fragment = any(
            self._is_clean_personal_fragment(str(row.get("content") or "")) for row in filtered
        )
        for row in filtered:
            ctx = dict(row.get("context") or {})
            if bool(ctx.get("invalid")) or bool(ctx.get("superseded")):
                continue
            if normalized_domain == "personal_life" and self._is_meta_recall_query_content(
                str(row.get("content") or "")
            ):
                continue
            confidence = max(0.0, min(1.0, float(ctx.get("confidence", row.get("importance", 0.5)) or 0.5)))
            importance = max(0.0, min(1.0, float(row.get("importance", 0.5) or 0.5)))
            recency = self._recency_score(str(row.get("timestamp") or ""))
            lexical = self._lexical_similarity(query, str(row.get("content") or ""))
            combined = confidence * 0.35 + importance * 0.25 + recency * 0.25 + lexical * 0.15
            if has_clean_fragment and self._is_broad_mixed_turn_content(str(row.get("content") or "")):
                downranked_mixed_count += 1
                if normalized_domain == "personal_life":
                    # Prefer clean personal event fragments and suppress broad mixed-turn sentences.
                    continue
                combined = max(0.0, combined - 0.22)
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
        raw_retrieved_count = raw_candidate_count
        by_norm: Dict[str, Dict[str, Any]] = {}
        for row in ranked:
            key = self._normalize_content_for_dedupe(str(row.get("content") or ""))
            if not key:
                continue
            current = by_norm.get(key)
            if current is None:
                by_norm[key] = row
                continue
            cur_conf = float((current.get("context") or {}).get("confidence", current.get("importance", 0.0)) or 0.0)
            new_conf = float((row.get("context") or {}).get("confidence", row.get("importance", 0.0)) or 0.0)
            current_key = (
                float(current.get("weighted_score", 0.0)),
                cur_conf,
                str(current.get("timestamp", "")),
            )
            new_key = (
                float(row.get("weighted_score", 0.0)),
                new_conf,
                str(row.get("timestamp", "")),
            )
            if new_key > current_key:
                by_norm[key] = row

        deduped = list(by_norm.values())
        deduped.sort(
            key=lambda r: (
                float(r.get("weighted_score", 0.0)),
                float((r.get("context") or {}).get("confidence", 0.0)),
                str(r.get("timestamp", "")),
            ),
            reverse=True,
        )
        deduped = deduped[:top_k]
        deduped_count = len(deduped)
        return {
            "items": deduped,
            "raw_retrieved_count": raw_retrieved_count,
            "deduped_count": deduped_count,
            "duplicate_count_removed": max(0, raw_retrieved_count - deduped_count),
            "downranked_mixed_count": downranked_mixed_count,
        }
