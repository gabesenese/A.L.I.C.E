from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PERSONALITY: Dict[str, Any] = {
    "curiosity_weight": 0.7,
    "directness": 0.6,
    "humor_threshold": 0.5,
    "concern_sensitivity": 0.6,
    "interests": [],
    "opinions": {},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "user": {
            "current_goals": [],
            "mentioned_intentions": [],
            "mood_signal": "neutral",
            "last_checkin": "",
            "last_active": "",
        },
        "environment": {
            "upcoming_calendar": [],
            "open_tasks": [],
            "unread_email_count": 0,
        },
        "alice_state": {
            "last_proactive_interrupt": "",
            "active_threads": [],
            "personality": deepcopy(DEFAULT_PERSONALITY),
            "personality_meta": {
                "short_response_streak": 0,
                "topic_counts": {},
            },
        },
        "updated_at": "",
    }


def _dedupe_records(records: List[Dict[str, Any]], *, key_field: str, limit: int) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for raw in records:
        item = dict(raw or {})
        key = str(item.get(key_field) or item.get("text") or item.get("goal") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[-limit:]


class WorldModel:
    """Persistent JSON-backed companion world model.

    This model stores companion-level memory that proactive loops can read
    without depending on app/main orchestration.
    """

    intention_pattern = re.compile(
        r"\b(?:i\s+)?(?:want|need|should|have to|gotta|plan|intend|am going|i'm going)\s+to\s+([^.!?\n]{3,160})",
        re.IGNORECASE,
    )
    goal_pattern = re.compile(
        r"\b(?:working on|work on|building|build|fixing|fix|improving|improve)\s+([^.!?\n]{3,140})",
        re.IGNORECASE,
    )
    task_pattern = re.compile(
        r"\b(?:remind me to|todo:?|task:?|need to|should)\s+([^.!?\n]{3,160})",
        re.IGNORECASE,
    )
    resolution_pattern = re.compile(
        r"\b(done|finished|completed|resolved|handled|took care of it|that's done)\b",
        re.IGNORECASE,
    )
    stress_pattern = re.compile(
        r"\b(stressed|anxious|overwhelmed|panic|worried|burned out|exhausted|tired)\b",
        re.IGNORECASE,
    )
    positive_pattern = re.compile(
        r"\b(good|great|better|excited|happy|glad|nice|productive)\b",
        re.IGNORECASE,
    )

    def __init__(self, path: str | Path = "data/world_model.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = _default_state()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(payload, dict):
            self._state = self._merge_defaults(payload)

    def _merge_defaults(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = _default_state()
        for section in ("user", "environment", "alice_state"):
            if isinstance(payload.get(section), dict):
                merged[section].update(payload[section])
        if isinstance(payload.get("alice_state"), dict) and isinstance(payload["alice_state"].get("personality"), dict):
            personality = deepcopy(DEFAULT_PERSONALITY)
            personality.update(payload["alice_state"].get("personality") or {})
            merged["alice_state"]["personality"] = personality
        merged["schema_version"] = int(payload.get("schema_version") or 1)
        merged["updated_at"] = str(payload.get("updated_at") or "")
        return merged

    def save(self) -> None:
        self._state["updated_at"] = _now_iso()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def snapshot(self) -> Dict[str, Any]:
        return deepcopy(self._state)

    def get_personality(self) -> Dict[str, Any]:
        return deepcopy(self._state["alice_state"]["personality"])

    # Minimum personality values that preserve companion character.
    # Drift below these breaks the JARVIS-style presence we want.
    _PERSONALITY_FLOORS = {
        "humor_threshold": (None, None),  # (min, max): uncapped — default 0.5
        "curiosity_weight": (0.45, None),  # min 0.45 keeps engagement alive
        "directness": (0.45, None),  # min 0.45 — default is 0.6, floor below default
    }

    def update_personality(self, personality: Dict[str, Any]) -> None:
        current = deepcopy(DEFAULT_PERSONALITY)
        current.update(dict(personality or {}))
        # Apply companion character floors/ceilings
        for key, (floor, ceiling) in self._PERSONALITY_FLOORS.items():
            if key in current:
                val = float(current[key])
                if floor is not None:
                    val = max(val, floor)
                if ceiling is not None:
                    val = min(val, ceiling)
                current[key] = val
        self._state["alice_state"]["personality"] = current
        self.save()

    def get_personality_meta(self) -> Dict[str, Any]:
        return deepcopy(self._state["alice_state"].get("personality_meta") or {})

    def update_personality_meta(self, meta: Dict[str, Any]) -> None:
        current = dict(self._state["alice_state"].get("personality_meta") or {})
        current.update(dict(meta or {}))
        self._state["alice_state"]["personality_meta"] = current
        self.save()

    def set_last_checkin(self, timestamp: Optional[str] = None) -> None:
        self._state["user"]["last_checkin"] = str(timestamp or _now_iso())
        self.save()

    def set_last_proactive_interrupt(self, timestamp: Optional[str] = None) -> None:
        self._state["alice_state"]["last_proactive_interrupt"] = str(timestamp or _now_iso())
        self.save()

    def mark_thread_resolved(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        if not needle:
            return
        self._state["environment"]["open_tasks"] = [
            item
            for item in self._state["environment"]["open_tasks"]
            if needle not in str(item.get("text") or "").lower()
        ]
        self._state["user"]["mentioned_intentions"] = [
            item
            for item in self._state["user"]["mentioned_intentions"]
            if needle not in str(item.get("text") or "").lower()
        ]
        self._state["alice_state"]["active_threads"] = [
            item
            for item in self._state["alice_state"]["active_threads"]
            if needle not in str(item.get("text") or "").lower()
        ]
        self.save()

    def record_proactive_interrupt(
        self,
        *,
        key: str,
        text: str,
        source: str,
        timestamp: Optional[str] = None,
    ) -> None:
        now = str(timestamp or _now_iso())
        interrupt_key = str(key or text or source or "").strip()
        if not interrupt_key:
            return
        self._state["alice_state"]["last_proactive_interrupt"] = now
        threads = list(self._state["alice_state"].get("active_threads") or [])
        updated = False
        for item in threads:
            if str(item.get("key") or "").strip().lower() == interrupt_key.lower():
                item["last_interrupted_at"] = now
                item["resolved"] = False
                updated = True
                break
        if not updated:
            threads.append(
                {
                    "key": interrupt_key,
                    "text": str(text or "")[:180],
                    "created_at": now,
                    "source": str(source or "heartbeat"),
                    "last_interrupted_at": now,
                    "resolved": False,
                }
            )
        self._state["alice_state"]["active_threads"] = _dedupe_records(
            threads,
            key_field="key",
            limit=20,
        )
        self.save()

    def has_unresolved_interrupt(self, key: str) -> bool:
        interrupt_key = str(key or "").strip().lower()
        if not interrupt_key:
            return False
        for item in list(self._state["alice_state"].get("active_threads") or []):
            if str(item.get("key") or "").strip().lower() == interrupt_key:
                return not bool(item.get("resolved"))
        return False

    def update_upcoming_calendar(self, events: List[Dict[str, Any]]) -> None:
        """Replace the upcoming calendar list with fresh data from the ambient monitor."""
        self._state["environment"]["upcoming_calendar"] = [
            {k: str(v or "") for k, v in dict(ev or {}).items()} for ev in list(events or [])
        ][:20]
        self.save()

    def update_unread_email_count(self, count: int) -> None:
        self._state["environment"]["unread_email_count"] = max(0, int(count or 0))
        fetches = dict(self._state.get("data_freshness") or {})
        fetches["email"] = _now_iso()
        self._state["data_freshness"] = fetches
        self.save()

    # --- Topic confidence tracking ---

    # Topics extracted per-turn from user input with mention count and confidence
    _TOPIC_KEYWORDS = re.compile(
        r"\b(python|javascript|typescript|rust|go|java|react|node|sql|docker|"
        r"kubernetes|aws|azure|gcp|linux|git|api|machine learning|ml|ai|nlp|"
        r"database|postgres|mysql|mongodb|redis|fastapi|django|flask|"
        r"finance|health|fitness|music|gaming|reading|writing|travel|cooking)\b",
        re.IGNORECASE,
    )

    def record_topic_mentions(self, text: str) -> None:
        """Extract topics from text and update confidence-weighted mention counts."""
        topics_store: Dict[str, Any] = dict(self._state.get("topic_confidence") or {})
        now = _now_iso()
        for match in self._TOPIC_KEYWORDS.finditer(str(text or "")):
            topic = match.group(0).lower()
            rec = topics_store.get(topic) or {
                "topic": topic,
                "count": 0,
                "confidence": 0.0,
                "first_seen": now,
                "last_seen": now,
            }
            rec["count"] += 1
            rec["last_seen"] = now
            # Confidence grows with mentions, capped at 0.95; decay formula: 1 - 1/(1+count)
            rec["confidence"] = round(min(0.95, 1.0 - 1.0 / (1.0 + rec["count"])), 3)
            topics_store[topic] = rec
        if topics_store:
            # Keep only top 30 by confidence to prevent unbounded growth
            sorted_topics = sorted(topics_store.values(), key=lambda x: x["confidence"], reverse=True)
            self._state["topic_confidence"] = {r["topic"]: r for r in sorted_topics[:30]}
            self.save()

    def high_confidence_topics(self, min_confidence: float = 0.5) -> List[str]:
        """Return topics with confidence >= min_confidence, sorted by confidence desc."""
        store: Dict[str, Any] = dict(self._state.get("topic_confidence") or {})
        return [
            rec["topic"]
            for rec in sorted(store.values(), key=lambda x: x["confidence"], reverse=True)
            if float(rec.get("confidence", 0)) >= min_confidence
        ]

    # --- Data freshness contracts ---

    def record_data_fetch(self, domain: str) -> None:
        """Record that live data for `domain` was just fetched (e.g. 'weather')."""
        fetches = dict(self._state.get("data_freshness") or {})
        fetches[str(domain)] = _now_iso()
        self._state["data_freshness"] = fetches
        self.save()

    def data_age_seconds(self, domain: str) -> Optional[float]:
        """Return seconds since last fetch for `domain`, or None if never fetched."""
        fetches = dict(self._state.get("data_freshness") or {})
        ts = fetches.get(str(domain))
        if not ts:
            return None
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(str(ts))
            return max(0.0, delta.total_seconds())
        except Exception:
            return None

    def is_data_stale(self, domain: str, ttl_seconds: float = 1800.0) -> bool:
        """Return True if `domain` data is older than `ttl_seconds` or was never fetched."""
        age = self.data_age_seconds(domain)
        if age is None:
            return True
        return age > ttl_seconds

    def add_open_task_if_missing(
        self,
        *,
        text: str,
        source: str = "ambient",
        timestamp: Optional[str] = None,
    ) -> bool:
        """Add an open task only if no existing task matches the same text. Returns True if added."""
        needle = str(text or "").strip().lower()
        if not needle:
            return False
        for item in list(self._state["environment"].get("open_tasks") or []):
            if needle in str(item.get("text") or "").lower():
                return False
        now = str(timestamp or _now_iso())
        self._state["environment"]["open_tasks"] = _dedupe_records(
            list(self._state["environment"].get("open_tasks") or [])
            + [
                {
                    "text": str(text or "").strip()[:180],
                    "created_at": now,
                    "source": str(source or "ambient"),
                }
            ],
            key_field="text",
            limit=40,
        )
        self.save()
        return True

    def update_from_turn(
        self,
        *,
        user_input: str,
        response_text: str,
        route: str = "",
        intent: str = "",
        requires_follow_up: bool = False,
        tool_success: Optional[bool] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = str(timestamp or _now_iso())
        user = self._state["user"]
        env = self._state["environment"]
        alice = self._state["alice_state"]
        text = str(user_input or "").strip()
        response = str(response_text or "").strip()

        user["last_active"] = now
        user["mood_signal"] = self._infer_mood(text, response)

        if self.resolution_pattern.search(text) or tool_success is True:
            self._resolve_oldest_open_thread()

        new_goals = self._extract_records(
            self.goal_pattern,
            text,
            timestamp=now,
            source="goal_signal",
            field="goal",
        )
        user["current_goals"] = _dedupe_records(
            list(user.get("current_goals") or []) + new_goals,
            key_field="goal",
            limit=20,
        )

        new_intentions = self._extract_records(
            self.intention_pattern,
            text,
            timestamp=now,
            source="intent_statement",
            field="text",
        )
        user["mentioned_intentions"] = _dedupe_records(
            list(user.get("mentioned_intentions") or []) + new_intentions,
            key_field="text",
            limit=30,
        )

        new_tasks = self._extract_records(
            self.task_pattern,
            text,
            timestamp=now,
            source="open_task_signal",
            field="text",
        )
        for item in new_intentions:
            new_tasks.append(
                {
                    "text": item.get("text", ""),
                    "created_at": item.get("created_at", now),
                    "source": "intention_unresolved",
                    "route": str(route or ""),
                    "intent": str(intent or ""),
                }
            )
        env["open_tasks"] = _dedupe_records(
            list(env.get("open_tasks") or []) + new_tasks,
            key_field="text",
            limit=40,
        )

        active_threads = list(alice.get("active_threads") or [])
        for item in list(env.get("open_tasks") or [])[-12:]:
            active_threads.append(
                {
                    "text": str(item.get("text") or "")[:180],
                    "created_at": str(item.get("created_at") or now),
                    "source": str(item.get("source") or "open_task"),
                }
            )
        alice["active_threads"] = _dedupe_records(
            active_threads,
            key_field="text",
            limit=20,
        )

        if self._looks_like_checkin(response):
            user["last_checkin"] = now

        self.save()
        return self.snapshot()

    _VAGUE_INTENTION_PATTERNS = re.compile(
        r"^(?:some(?:thing|body|where|how|times?)?|some\s+(?:stuff|things?)|"
        r"stuff|things?|it|this|that|them|work|"
        r"do\s+(?:some(?:thing|stuff)|some\s+(?:stuff|things?)|it|that)|"
        r"work\s+on\s+(?:some(?:thing|stuff)|some\s+(?:stuff|things?)|it|that)|"
        r"(?:a\s+)?few\s+things?|various\s+things?|lots?\s+of\s+things?)$",
        re.IGNORECASE,
    )

    def _extract_records(
        self,
        pattern: re.Pattern[str],
        text: str,
        *,
        timestamp: str,
        source: str,
        field: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", str(match.group(1) or "")).strip(" .,:;-")
            if len(value) < 8:
                continue
            if self._VAGUE_INTENTION_PATTERNS.match(value):
                continue
            records.append(
                {
                    field: value[:180],
                    "created_at": timestamp,
                    "source": source,
                }
            )
        return records

    def _infer_mood(self, user_input: str, response_text: str) -> str:
        text = f"{user_input} {response_text}".lower()
        if self.stress_pattern.search(text):
            return "stressed"
        if self.positive_pattern.search(text):
            return "positive"
        return "neutral"

    def _resolve_oldest_open_thread(self) -> None:
        open_tasks = list(self._state["environment"].get("open_tasks") or [])
        if open_tasks:
            open_tasks.pop(0)
        self._state["environment"]["open_tasks"] = open_tasks
        active_threads = list(self._state["alice_state"].get("active_threads") or [])
        if active_threads:
            active_threads.pop(0)
        self._state["alice_state"]["active_threads"] = active_threads

    @staticmethod
    def _looks_like_checkin(response_text: str) -> bool:
        low = str(response_text or "").lower()
        return any(
            marker in low
            for marker in (
                "checking in",
                "check in",
                "how are you",
                "how's it going",
                "how are you doing",
            )
        )


_world_model: WorldModel | None = None


def get_world_model(path: str | Path = "data/world_model.json") -> WorldModel:
    global _world_model
    if _world_model is None or Path(path) != _world_model.path:
        _world_model = WorldModel(path)
    return _world_model
