"""
Phrasing Learner for A.L.I.C.E
Progressive learning system - Alice learns from Ollama like a child from a parent

Architecture:
- Alice asks Ollama to phrase her thoughts naturally
- Alice observes and records how Ollama phrased it
- Over time, Alice learns patterns and can phrase things herself
- Reduces LLM dependency, faster responses, consistent personality
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from ai.learning.data_redaction import sanitize_for_learning, redact_text

logger = logging.getLogger(__name__)


class PhrasingLearner:
    """
    Alice learns natural language phrasing from Ollama.
    Like a child asking parent to say something, then learning how they said it.

    Learning Flow:
    1. Alice formulates structured thought
    2. Alice checks: Can I phrase this myself?
    3. If NO: Ask Ollama, record the phrasing (learn)
    4. If YES: Use learned pattern (independent!)
    """

    def __init__(self, storage_path: str = "data/learned_phrasings.jsonl"):
        self.storage_path = Path(storage_path)
        self.learned_patterns = defaultdict(list)  # {pattern: [examples]}
        self.confidence_scores = {}  # {pattern: confidence}
        self.min_examples_for_confidence = 3  # Need 3+ examples to be confident
        self.confidence_threshold = 0.7  # 70% confidence to phrase independently

        # Ensure data directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_learned_patterns()
        logger.info(
            f"[PhrasingLearner] Initialized with {len(self.learned_patterns)} patterns"
        )

    def _load_learned_patterns(self) -> None:
        """Load previously learned phrasings from storage"""
        if not self.storage_path.exists():
            logger.info("[PhrasingLearner] No existing patterns found, starting fresh")
            return

        try:
            pattern_count = 0
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        pattern = entry["pattern"]
                        self.learned_patterns[pattern].append(entry)
                        pattern_count += 1

            # Recalculate confidence scores
            for pattern in self.learned_patterns:
                self._update_confidence(pattern)

            logger.info(f"[PhrasingLearner] Loaded {pattern_count} learned examples")

        except Exception as e:
            logger.error(f"[PhrasingLearner] Error loading patterns: {e}")

    @staticmethod
    def _make_json_safe(obj: Any) -> Any:
        """Recursively convert non-serializable objects (e.g. Entity) to plain dicts/strings."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: PhrasingLearner._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PhrasingLearner._make_json_safe(i) for i in obj]
        # Dataclass / namedtuple / object with __dict__
        if hasattr(obj, "__dict__"):
            return {k: PhrasingLearner._make_json_safe(v) for k, v in vars(obj).items()}
        return str(obj)

    def _persist(self, new_entry: Dict[str, Any]) -> None:
        """Append new learning to storage (JSONL format)"""
        try:
            safe_entry = PhrasingLearner._make_json_safe(new_entry)
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[PhrasingLearner] Error persisting: {e}")

    def _extract_pattern(self, alice_thought: Dict) -> str:
        """
        Extract structural pattern from Alice's thought.
        Pattern identifies the TYPE of response, used for matching similar situations.

        Examples:
        - capability:true (Alice confirming she can do something)
        - capability:false (Alice saying she can't do something)
        - knowledge:general (Alice providing general knowledge)
        - reasoning:conclusion (Alice providing a reasoning conclusion)
        """
        thought_type = alice_thought.get("type", "general")

        if thought_type == "capability_answer":
            # Pattern based on whether Alice can or can't do something
            can_do = alice_thought.get("can_do", False)
            return f"capability:{can_do}"

        elif thought_type == "knowledge_answer":
            # All knowledge answers are similar patterns
            return "knowledge:general"

        elif thought_type == "reasoning_result":
            # Reasoning conclusions
            return "reasoning:conclusion"

        elif thought_type == "factual_answer":
            # Direct factual responses
            return "factual:answer"

        else:
            # General responses by type
            return f"general:{thought_type}"

    def _is_high_variance_thought(self, alice_thought: Dict[str, Any]) -> bool:
        """
        Return True for thought types where replaying a learned phrasing is risky.

        These intents are highly open-ended and can easily produce unrelated
        responses if we reuse older examples from the same broad pattern bucket.
        """
        thought_type = (alice_thought.get("type") or "").strip().lower()

        if thought_type in {"knowledge_answer", "reasoning_result", "factual_answer"}:
            return True

        # Broad conversation buckets are too diverse for safe template replay.
        if thought_type.startswith("conversation:") and thought_type not in {
            "conversation:ack",
            "conversation:acknowledgment",
            "conversation:clarification_needed",
            "conversation:help",
            "conversation:help_opener",
        }:
            return True

        return False

    def _update_confidence(self, pattern: str) -> None:
        """
        Update confidence score for a pattern based on number of examples.
        More examples = higher confidence that Alice can phrase it herself.

        Confidence formula:
        - 0-2 examples: low confidence (0.0 - 0.6)
        - 3-5 examples: building confidence (0.7 - 0.85)
        - 6+ examples: high confidence (0.85 - 0.95)
        """
        example_count = len(self.learned_patterns[pattern])

        if example_count == 0:
            confidence = 0.0
        elif example_count == 1:
            confidence = 0.3
        elif example_count == 2:
            confidence = 0.6
        elif example_count == 3:
            confidence = 0.7  # Threshold reached!
        elif example_count == 4:
            confidence = 0.8
        elif example_count == 5:
            confidence = 0.85
        else:
            # Cap at 0.95 (never 100% confident, always room to learn)
            confidence = min(0.95, 0.85 + (example_count - 5) * 0.02)

        self.confidence_scores[pattern] = confidence

    @staticmethod
    def _sanitize_weather_advice_phrasing(phrasing: str) -> str:
        """Remove awkward name-based addressing from learned weather advice."""
        cleaned = (phrasing or "").strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
            cleaned = cleaned[1:-1].strip()

        cleaned = re.sub(
            r"^(?:for|hey|hi|hello)\s+(?:the user|user|testuser|[A-Z][\w-]*)[:,!]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r",\s*(?:the user|user|testuser|[A-Z][\w-]*)\s*,",
            ", ",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _normalize_alice_style(phrasing: str) -> str:
        """Normalize phrasing to concise Alice style before learning."""
        text = str(phrasing or "").strip()
        if not text:
            return ""

        # Trim outer quotes and collapse whitespace.
        if len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1].strip()
        text = re.sub(r"\s+", " ", text)

        # Remove common filler/softening openers.
        filler_prefixes = [
            r"^(?:sure|of course|absolutely|definitely|certainly|no problem|happy to|i can help with that)[:,.!]?\s+",
            r"^(?:i (?:would|can) (?:be )?(?:happy|glad) to)\s+",
            r"^(?:just to clarify[:,]?\s*)",
        ]
        for pattern in filler_prefixes:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove repetitive hedging.
        text = re.sub(r"\b(?:maybe|perhaps|kind of|sort of|just)\b\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()

        # Keep at most two sentences to avoid over-explaining.
        parts = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(parts[:2]).strip()

        # Hard cap for learned phrasing footprint.
        if len(text) > 220:
            text = text[:217].rstrip() + "..."

        return text

    def record_phrasing(
        self,
        alice_thought: Dict[str, Any],
        ollama_phrasing: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Record how Ollama phrased Alice's thought.
        Alice learns by observing.

        Args:
            alice_thought: Alice's structured thought content
            ollama_phrasing: How Ollama phrased it naturally
            context: Tone, intent, user_name, etc.
        """
        if alice_thought.get("type") == "weather_advice":
            ollama_phrasing = self._sanitize_weather_advice_phrasing(ollama_phrasing)

        safe_thought = sanitize_for_learning(alice_thought or {})
        safe_context = sanitize_for_learning(context or {})
        safe_ollama_phrasing = redact_text(ollama_phrasing or "")
        safe_ollama_phrasing = self._normalize_alice_style(safe_ollama_phrasing)

        # Avoid polluting learned store with high-variance conversational/knowledge
        # responses that are likely to replay incorrectly on future turns.
        if self._is_high_variance_thought(safe_thought):
            logger.debug(
                "[PhrasingLearner] Skipping high-variance thought type '%s'",
                safe_thought.get("type", ""),
            )
            return

        pattern = self._extract_pattern(safe_thought)

        entry = {
            "pattern": pattern,
            "alice_thought": safe_thought,
            "ollama_phrasing": safe_ollama_phrasing,
            "context": safe_context,
            "timestamp": datetime.now().isoformat(),
            "tone": safe_context.get("tone", "warm and helpful"),
        }

        # Store in memory
        self.learned_patterns[pattern].append(entry)

        # Update confidence
        old_confidence = self.confidence_scores.get(pattern, 0.0)
        self._update_confidence(pattern)
        new_confidence = self.confidence_scores[pattern]

        # Persist to disk
        self._persist(entry)

        # Log learning progress
        if (
            new_confidence >= self.confidence_threshold
            and old_confidence < self.confidence_threshold
        ):
            logger.info(
                f"[PhrasingLearner] Alice learned pattern '{pattern}'! Can now phrase independently."
            )
        else:
            logger.info(
                f"[PhrasingLearner] Learning '{pattern}' (confidence: {new_confidence:.2f}, examples: {len(self.learned_patterns[pattern])})"
            )

    def can_phrase_myself(self, alice_thought: Dict, tone: str) -> bool:
        """
        Check if Alice has learned enough to phrase this herself.
        Like a child checking: "Can I say this myself or should I ask parent?"

        Requirements:
        1. Pattern must be recognized (seen before)
        2. Have 3+ examples with matching tone
        3. Confidence >= 0.7

        Args:
            alice_thought: Alice's structured thought
            tone: The tone Alice wants to use

        Returns:
            True if Alice can phrase it herself, False if needs Ollama's help
        """
        if self._is_high_variance_thought(alice_thought):
            logger.debug(
                "[PhrasingLearner] High-variance thought '%s' requires fresh phrasing",
                alice_thought.get("type", ""),
            )
            return False

        pattern = self._extract_pattern(alice_thought)

        # Never seen this pattern before?
        if pattern not in self.learned_patterns:
            logger.debug(f"[PhrasingLearner] Unknown pattern '{pattern}', need Ollama")
            return False

        # Check confidence score
        confidence = self.confidence_scores.get(pattern, 0.0)
        if confidence < self.confidence_threshold:
            logger.debug(
                f"[PhrasingLearner] Low confidence ({confidence:.2f}) for '{pattern}', need Ollama"
            )
            return False

        # Check if we have examples with this tone
        examples_with_tone = [
            e
            for e in self.learned_patterns[pattern]
            if e["context"].get("tone") == tone
        ]

        if len(examples_with_tone) < self.min_examples_for_confidence:
            logger.debug(
                f"[PhrasingLearner] Need more examples with tone '{tone}' (have {len(examples_with_tone)})"
            )
            return False

        # Alice is confident and has learned this!
        logger.info(
            f"[PhrasingLearner] Alice can phrase '{pattern}' independently (confidence: {confidence:.2f})"
        )
        return True

    def phrase_myself(self, alice_thought: Dict, tone: str) -> str:
        """
        Alice phrases the response herself using learned patterns.
        Like a child who's learned to say something without asking parent.

        Strategy:
        1. Find examples with matching tone
        2. Select the best/most recent example
        3. Adapt the phrasing to current thought

        Args:
            alice_thought: Alice's current thought to phrase
            tone: The tone to use

        Returns:
            Naturally phrased response (learned from Ollama)
        """
        pattern = self._extract_pattern(alice_thought)

        # Get examples with matching tone
        examples_with_tone = [
            e
            for e in self.learned_patterns[pattern]
            if e["context"].get("tone") == tone
        ]

        if not examples_with_tone:
            # Fallback: use any example with this pattern
            examples_with_tone = self.learned_patterns[pattern]

        # Recency-weighted selection: prefer more recent examples while keeping variety.
        # Taking the last 10 and assigning linearly increasing weights means the
        # most recent example is 10x more likely than the oldest of the window.
        import random

        window = examples_with_tone[-min(10, len(examples_with_tone)) :]
        weights = list(range(1, len(window) + 1))
        selected_example = random.choices(window, weights=weights, k=1)[0]

        # Adapt the learned phrasing to current thought
        adapted_phrasing = self._adapt_phrasing(
            learned_phrasing=selected_example["ollama_phrasing"],
            current_thought=alice_thought,
            learned_thought=selected_example["alice_thought"],
        )

        if alice_thought.get("type") == "weather_advice":
            adapted_phrasing = self._sanitize_weather_advice_phrasing(adapted_phrasing)

        logger.info(f"[PhrasingLearner] Alice phrased '{pattern}' independently!")
        return adapted_phrasing

    def _adapt_phrasing(
        self, learned_phrasing: str, current_thought: Dict, learned_thought: Dict
    ) -> str:
        """
        Adapt a learned phrasing to current situation.
        Keeps the structure/style but swaps in new content.

        Strategy:
        - For capability answers: swap capability details
        - For knowledge answers: keep structure, swap content
        - For weather/data types: substitute ALL changed field values (top-level + data)
        - For others: use learned phrasing as template
        """
        thought_type = current_thought.get("type", "general")

        # Simple adaptation: direct content substitution
        adapted = learned_phrasing

        if thought_type == "capability_answer":
            # Replace capability-specific details
            old_details = learned_thought.get("details", [])
            new_details = current_thought.get("details", [])

            # Simple string replacement of details
            for old_detail, new_detail in zip(old_details, new_details):
                adapted = adapted.replace(str(old_detail), str(new_detail))

        elif thought_type == "knowledge_answer":
            # For knowledge: structure is good, but content differs
            pass

        else:
            # Generic field substitution - check both top-level AND data subfield
            # This handles weather_report, weather_advice, note ops, etc.

            # 1. Top-level fields (weather_report has temperature, condition at top level)
            for key in learned_thought:
                if key in current_thought and key not in ["type", "confidence", "data"]:
                    old_val = learned_thought[key]
                    new_val = current_thought[key]
                    if (
                        old_val != new_val
                        and old_val is not None
                        and new_val is not None
                    ):
                        # Convert to string for replacement
                        old_str = str(old_val)
                        new_str = str(new_val)
                        # CRITICAL: Check old_str is not empty to avoid inserting new_str between every character
                        if old_str and old_str in adapted:
                            adapted = adapted.replace(old_str, new_str)

            # 2. Data subfield (for note ops, file ops, etc.)
            old_data = learned_thought.get("data", {})
            new_data = current_thought.get("data", {})
            for key in old_data:
                if key in new_data and old_data[key] != new_data[key]:
                    old_val = str(old_data[key])
                    new_val = str(new_data[key])
                    if old_val and old_val in adapted:
                        adapted = adapted.replace(old_val, new_val)

        # Return adapted phrasing
        return adapted

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        total_patterns = len(self.learned_patterns)
        total_examples = sum(
            len(examples) for examples in self.learned_patterns.values()
        )
        confident_patterns = sum(
            1
            for conf in self.confidence_scores.values()
            if conf >= self.confidence_threshold
        )

        return {
            "total_patterns": total_patterns,
            "total_examples": total_examples,
            "confident_patterns": confident_patterns,
            "independence_rate": (
                confident_patterns / total_patterns if total_patterns > 0 else 0.0
            ),
            "patterns": {
                pattern: {
                    "examples": len(self.learned_patterns[pattern]),
                    "confidence": self.confidence_scores.get(pattern, 0.0),
                }
                for pattern in self.learned_patterns
            },
        }


# Test/demo code
if __name__ == "__main__":
    print("=" * 80)
    print("A.L.I.C.E - Phrasing Learner Demo")
    print("=" * 80)

    learner = PhrasingLearner(storage_path="data/test_phrasings.jsonl")

    # Simulate Alice learning to answer capability questions
    print("\n[Session 1] Alice asks Ollama...")

    alice_thought = {
        "type": "capability_answer",
        "can_do": True,
        "details": ["read-only", "all directories", "list/read/search"],
        "scope": ["ai/", "app/", "features/"],
    }

    ollama_response = (
        "Yes, I can read and analyze my Python codebase across all directories."
    )

    can_do_it = learner.can_phrase_myself(alice_thought, "warm and helpful")
    print(f"Can Alice phrase this herself? {can_do_it}")

    if not can_do_it:
        print("Alice asks Ollama for help...")
        learner.record_phrasing(
            alice_thought, ollama_response, {"tone": "warm and helpful"}
        )
        print(f"Ollama: {ollama_response}")

    # Session 2
    print("\n[Session 2] Alice asks Ollama again...")
    learner.record_phrasing(
        alice_thought, ollama_response, {"tone": "warm and helpful"}
    )

    # Session 3
    print("\n[Session 3] Alice asks Ollama again...")
    learner.record_phrasing(
        alice_thought, ollama_response, {"tone": "warm and helpful"}
    )

    # Session 4 - Alice is confident!
    print("\n[Session 4] Alice tries herself...")
    can_do_it = learner.can_phrase_myself(alice_thought, "warm and helpful")
    print(f"Can Alice phrase this herself? {can_do_it}")

    if can_do_it:
        response = learner.phrase_myself(alice_thought, "warm and helpful")
        print(f"Alice (independent!): {response}")

    # Stats
    print("\n" + "=" * 80)
    print("Learning Statistics:")
    stats = learner.get_stats()
    print(f"Total patterns learned: {stats['total_patterns']}")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Confident patterns: {stats['confident_patterns']}")
    print(f"Independence rate: {stats['independence_rate']:.1%}")
    print("=" * 80)
