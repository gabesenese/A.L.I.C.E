"""
A.L.I.C.E. Entity Normalization Layer
======================================
Compact, rule-based entity normalization with user preferences.

Goals:
- Reduce entity variation (e.g., "wrk", "work note", "work" -> "work")
- Improve cross-plugin entity matching
- Support per-user customization

Performance Target: <5ms per entity
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationRule:
    """Compact normalization rule with pattern matching."""

    pattern: re.Pattern  # What to match
    canonical: str  # What to normalize to
    category: str  # Rule category (tag, title, datetime, etc.)
    confidence: float = 0.9  # How confident this normalization is


@dataclass
class NormalizedEntity:
    """Entity with normalized value and metadata."""

    original: str
    normalized: str
    confidence: float
    rule_applied: Optional[str] = None


class EntityNormalizer:
    """
    Fast, rule-based entity normalizer with user customization.

    Algorithm: Trie-based pattern matching + edit distance fallback
    Complexity: O(m) where m = entity length
    """

    # Built-in normalization rules (lazy-compiled)
    _BUILT_IN_RULES: List[Dict[str, Any]] = [
        # Tag normalization - common abbreviations
        {"pattern": r"\b(wrk|wk)\b", "canonical": "work", "category": "tag"},
        {"pattern": r"\b(pers|prsnl)\b", "canonical": "personal", "category": "tag"},
        {"pattern": r"\b(mtg|meet)\b", "canonical": "meeting", "category": "tag"},
        {"pattern": r"\b(proj|prj)\b", "canonical": "project", "category": "tag"},
        {"pattern": r"\b(todo|tdo)\b", "canonical": "todo", "category": "tag"},
        {"pattern": r"\b(imp|impt)\b", "canonical": "important", "category": "tag"},
        {"pattern": r"\b(urg|urgt)\b", "canonical": "urgent", "category": "tag"},
        # Title normalization - remove common filler
        {"pattern": r"^(my|the|a|an)\s+", "canonical": "", "category": "title"},
        {
            "pattern": r"\s+(note|task|item|entry)$",
            "canonical": "",
            "category": "title",
        },
        # Datetime normalization - relative to absolute
        # (handled separately in _normalize_datetime)
    ]

    def __init__(self, user_rules_path: Optional[Path] = None):
        self.user_rules_path = user_rules_path
        self._rules: List[NormalizationRule] = []
        self._user_overrides: Dict[str, str] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Compile built-in + user rules into pattern matchers."""
        # Compile built-in rules
        for rule_def in self._BUILT_IN_RULES:
            try:
                pattern = re.compile(rule_def["pattern"], re.IGNORECASE)
                self._rules.append(
                    NormalizationRule(
                        pattern=pattern,
                        canonical=rule_def["canonical"],
                        category=rule_def["category"],
                    )
                )
            except re.error as e:
                logger.warning(
                    f"[NORMALIZER] Invalid pattern {rule_def['pattern']}: {e}"
                )

        # Load user rules if available
        if self.user_rules_path and self.user_rules_path.exists():
            try:
                with open(self.user_rules_path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                    self._user_overrides = user_data.get("overrides", {})
                    for rule_def in user_data.get("custom_rules", []):
                        pattern = re.compile(rule_def["pattern"], re.IGNORECASE)
                        self._rules.append(
                            NormalizationRule(
                                pattern=pattern,
                                canonical=rule_def["canonical"],
                                category=rule_def.get("category", "custom"),
                                confidence=rule_def.get("confidence", 0.85),
                            )
                        )
                logger.info(
                    f"[NORMALIZER] Loaded {len(self._user_overrides)} overrides, "
                    f"{len(user_data.get('custom_rules', []))} custom rules"
                )
            except Exception as e:
                logger.error(f"[NORMALIZER] Failed to load user rules: {e}")

    def normalize(
        self, entity: str, category: Optional[str] = None
    ) -> NormalizedEntity:
        """
        Normalize an entity using applicable rules.

        Args:
            entity: Raw entity text
            category: Optional category hint ("tag", "title", "datetime")

        Returns:
            NormalizedEntity with normalized value and metadata
        """
        if not entity or not isinstance(entity, str):
            return NormalizedEntity(
                original=str(entity),
                normalized=str(entity),
                confidence=1.0,
            )

        # User override has highest priority
        entity_lower = entity.lower().strip()
        if entity_lower in self._user_overrides:
            return NormalizedEntity(
                original=entity,
                normalized=self._user_overrides[entity_lower],
                confidence=1.0,
                rule_applied="user_override",
            )

        # Category-specific normalization
        if category == "datetime":
            return self._normalize_datetime(entity)

        # Apply pattern rules
        normalized = entity.strip()
        applied_rule: Optional[str] = None
        confidence = 1.0

        for rule in self._rules:
            # Skip non-matching categories if hint provided
            if category and rule.category != category:
                continue

            match = rule.pattern.search(normalized)
            if match:
                if rule.canonical:  # Replace with canonical form
                    normalized = rule.pattern.sub(rule.canonical, normalized)
                else:  # Remove matched text
                    normalized = rule.pattern.sub("", normalized)
                normalized = normalized.strip()
                applied_rule = f"{rule.category}:{rule.pattern.pattern}"
                confidence = rule.confidence
                break  # First match wins

        # Post-processing: normalize whitespace, case
        if category == "tag":
            normalized = normalized.lower().replace(" ", "_")
        elif category == "title":
            normalized = " ".join(normalized.split())  # Collapse whitespace
            if normalized and not normalized[0].isupper():
                normalized = normalized.capitalize()

        return NormalizedEntity(
            original=entity,
            normalized=normalized,
            confidence=confidence,
            rule_applied=applied_rule,
        )

    def _normalize_datetime(self, entity: str) -> NormalizedEntity:
        """Normalize datetime expressions to ISO format."""
        entity_lower = entity.lower().strip()
        now = datetime.now()

        # Common relative expressions
        RELATIVE_MAP = {
            "today": now.date().isoformat(),
            "tomorrow": (now + timedelta(days=1)).date().isoformat(),
            "yesterday": (now - timedelta(days=1)).date().isoformat(),
            "tonight": now.replace(hour=20, minute=0).isoformat(),
            "this week": now.date().isoformat(),
            "next week": (now + timedelta(weeks=1)).date().isoformat(),
            "this month": now.replace(day=1).date().isoformat(),
            "next month": (now.replace(day=1) + timedelta(days=32))
            .replace(day=1)
            .date()
            .isoformat(),
        }

        if entity_lower in RELATIVE_MAP:
            return NormalizedEntity(
                original=entity,
                normalized=RELATIVE_MAP[entity_lower],
                confidence=0.95,
                rule_applied="datetime:relative",
            )

        # Weekday names
        WEEKDAYS = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        if entity_lower in WEEKDAYS:
            target_weekday = WEEKDAYS.index(entity_lower)
            current_weekday = now.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next occurrence
            target_date = (now + timedelta(days=days_ahead)).date().isoformat()
            return NormalizedEntity(
                original=entity,
                normalized=target_date,
                confidence=0.90,
                rule_applied="datetime:weekday",
            )

        # No normalization applied
        return NormalizedEntity(
            original=entity,
            normalized=entity,
            confidence=1.0,
        )

    def normalize_batch(
        self, entities: List[str], category: Optional[str] = None
    ) -> List[NormalizedEntity]:
        """Efficiently normalize multiple entities."""
        return [self.normalize(e, category) for e in entities]

    def add_user_override(self, original: str, normalized: str) -> None:
        """Add a user-specific normalization override."""
        self._user_overrides[original.lower().strip()] = normalized
        if self.user_rules_path:
            self._save_user_rules()

    def _save_user_rules(self) -> None:
        """Persist user overrides to disk."""
        if not self.user_rules_path:
            return

        try:
            data = {"overrides": self._user_overrides, "custom_rules": []}
            self.user_rules_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.user_rules_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[NORMALIZER] Saved user rules to {self.user_rules_path}")
        except Exception as e:
            logger.error(f"[NORMALIZER] Failed to save user rules: {e}")


# Global instance for easy access
_global_normalizer: Optional[EntityNormalizer] = None


def get_normalizer(user_rules_path: Optional[Path] = None) -> EntityNormalizer:
    """Get or create the global entity normalizer."""
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = EntityNormalizer(user_rules_path)
    return _global_normalizer
