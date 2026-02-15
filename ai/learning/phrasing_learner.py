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
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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
        logger.info(f"[PhrasingLearner] Initialized with {len(self.learned_patterns)} patterns")

    def _load_learned_patterns(self) -> None:
        """Load previously learned phrasings from storage"""
        if not self.storage_path.exists():
            logger.info("[PhrasingLearner] No existing patterns found, starting fresh")
            return

        try:
            pattern_count = 0
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        pattern = entry['pattern']
                        self.learned_patterns[pattern].append(entry)
                        pattern_count += 1

            # Recalculate confidence scores
            for pattern in self.learned_patterns:
                self._update_confidence(pattern)

            logger.info(f"[PhrasingLearner] Loaded {pattern_count} learned examples")

        except Exception as e:
            logger.error(f"[PhrasingLearner] Error loading patterns: {e}")

    def _persist(self, new_entry: Dict[str, Any]) -> None:
        """Append new learning to storage (JSONL format)"""
        try:
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
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
        thought_type = alice_thought.get('type', 'general')

        if thought_type == 'capability_answer':
            # Pattern based on whether Alice can or can't do something
            can_do = alice_thought.get('can_do', False)
            return f"capability:{can_do}"

        elif thought_type == 'knowledge_answer':
            # All knowledge answers are similar patterns
            return "knowledge:general"

        elif thought_type == 'reasoning_result':
            # Reasoning conclusions
            return "reasoning:conclusion"

        elif thought_type == 'factual_answer':
            # Direct factual responses
            return "factual:answer"

        else:
            # General responses by type
            return f"general:{thought_type}"

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

    def record_phrasing(
        self,
        alice_thought: Dict[str, Any],
        ollama_phrasing: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Record how Ollama phrased Alice's thought.
        Alice learns by observing.

        Args:
            alice_thought: Alice's structured thought content
            ollama_phrasing: How Ollama phrased it naturally
            context: Tone, intent, user_name, etc.
        """
        pattern = self._extract_pattern(alice_thought)

        entry = {
            'pattern': pattern,
            'alice_thought': alice_thought,
            'ollama_phrasing': ollama_phrasing,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'tone': context.get('tone', 'warm and helpful')
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
        if new_confidence >= self.confidence_threshold and old_confidence < self.confidence_threshold:
            logger.info(f"[PhrasingLearner] Alice learned pattern '{pattern}'! Can now phrase independently.")
        else:
            logger.info(f"[PhrasingLearner] Learning '{pattern}' (confidence: {new_confidence:.2f}, examples: {len(self.learned_patterns[pattern])})")

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
        pattern = self._extract_pattern(alice_thought)

        # Never seen this pattern before?
        if pattern not in self.learned_patterns:
            logger.debug(f"[PhrasingLearner] Unknown pattern '{pattern}', need Ollama")
            return False

        # Check confidence score
        confidence = self.confidence_scores.get(pattern, 0.0)
        if confidence < self.confidence_threshold:
            logger.debug(f"[PhrasingLearner] Low confidence ({confidence:.2f}) for '{pattern}', need Ollama")
            return False

        # Check if we have examples with this tone
        examples_with_tone = [
            e for e in self.learned_patterns[pattern]
            if e['context'].get('tone') == tone
        ]

        if len(examples_with_tone) < self.min_examples_for_confidence:
            logger.debug(f"[PhrasingLearner] Need more examples with tone '{tone}' (have {len(examples_with_tone)})")
            return False

        # Alice is confident and has learned this!
        logger.info(f"[PhrasingLearner] Alice can phrase '{pattern}' independently (confidence: {confidence:.2f})")
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
            e for e in self.learned_patterns[pattern]
            if e['context'].get('tone') == tone
        ]

        if not examples_with_tone:
            # Fallback: use any example with this pattern
            examples_with_tone = self.learned_patterns[pattern]

        # Select best example (most recent)
        best_example = max(examples_with_tone, key=lambda e: e.get('timestamp', ''))

        # Adapt the learned phrasing to current thought
        adapted_phrasing = self._adapt_phrasing(
            learned_phrasing=best_example['ollama_phrasing'],
            current_thought=alice_thought,
            learned_thought=best_example['alice_thought']
        )

        logger.info(f"[PhrasingLearner] Alice phrased '{pattern}' independently!")
        return adapted_phrasing

    def _adapt_phrasing(
        self,
        learned_phrasing: str,
        current_thought: Dict,
        learned_thought: Dict
    ) -> str:
        """
        Adapt a learned phrasing to current situation.
        Keeps the structure/style but swaps in new content.

        Strategy:
        - For capability answers: swap capability details
        - For knowledge answers: keep structure, swap content
        - For others: use learned phrasing as template
        """
        thought_type = current_thought.get('type', 'general')

        # Simple adaptation: direct content substitution
        adapted = learned_phrasing

        if thought_type == 'capability_answer':
            # Replace capability-specific details
            old_details = learned_thought.get('details', [])
            new_details = current_thought.get('details', [])

            # Simple string replacement of details
            for old_detail, new_detail in zip(old_details, new_details):
                adapted = adapted.replace(str(old_detail), str(new_detail))

        elif thought_type == 'knowledge_answer':
            # For knowledge: structure is good, but content differs
            # Keep the phrasing style, just ensure current content is reflected
            # (In future: could use more sophisticated NLP here)
            pass

        # Return adapted phrasing
        return adapted

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        total_patterns = len(self.learned_patterns)
        total_examples = sum(len(examples) for examples in self.learned_patterns.values())
        confident_patterns = sum(1 for conf in self.confidence_scores.values()
                                if conf >= self.confidence_threshold)

        return {
            'total_patterns': total_patterns,
            'total_examples': total_examples,
            'confident_patterns': confident_patterns,
            'independence_rate': confident_patterns / total_patterns if total_patterns > 0 else 0.0,
            'patterns': {
                pattern: {
                    'examples': len(self.learned_patterns[pattern]),
                    'confidence': self.confidence_scores.get(pattern, 0.0)
                }
                for pattern in self.learned_patterns
            }
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
        'type': 'capability_answer',
        'can_do': True,
        'details': ['read-only', 'all directories', 'list/read/search'],
        'scope': ['ai/', 'app/', 'features/']
    }

    ollama_response = "Yes, I can read and analyze my Python codebase across all directories."

    can_do_it = learner.can_phrase_myself(alice_thought, 'warm and helpful')
    print(f"Can Alice phrase this herself? {can_do_it}")

    if not can_do_it:
        print("Alice asks Ollama for help...")
        learner.record_phrasing(alice_thought, ollama_response, {'tone': 'warm and helpful'})
        print(f"Ollama: {ollama_response}")

    # Session 2
    print("\n[Session 2] Alice asks Ollama again...")
    learner.record_phrasing(alice_thought, ollama_response, {'tone': 'warm and helpful'})

    # Session 3
    print("\n[Session 3] Alice asks Ollama again...")
    learner.record_phrasing(alice_thought, ollama_response, {'tone': 'warm and helpful'})

    # Session 4 - Alice is confident!
    print("\n[Session 4] Alice tries herself...")
    can_do_it = learner.can_phrase_myself(alice_thought, 'warm and helpful')
    print(f"Can Alice phrase this herself? {can_do_it}")

    if can_do_it:
        response = learner.phrase_myself(alice_thought, 'warm and helpful')
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
