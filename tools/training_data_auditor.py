"""
Training Data Auditor & Cleaner
================================
Analyzes Alice's training data and removes low-quality or outdated learning.
Alice should only learn from her best interactions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class TrainingDataAuditor:
    """
    Audits and cleans Alice's training data.
    Removes:
    - Low quality responses
    - Hallucinated information
    - Outdated patterns
    - Inconsistent data
    - Poor examples
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.knowledge_dir = self.data_dir / "knowledge"

        # Quality thresholds
        self.min_quality_score = 0.7
        self.min_response_length = 10
        self.max_response_length = 1000

        # Hallucination indicators
        self.hallucination_markers = [
            "stanford research",
            "sri international",
            "ladie system",
            "ladie project",
            "knowledge graph",  # If Alice doesn't actually have one
            "neural network",   # If not actually using one
        ]

        # Error indicators
        self.error_markers = [
            "i apologize",
            "i don't know",
            "i'm not sure",
            "error occurred",
            "something went wrong",
            "try again",
            "can't help with that"
        ]

    def audit_training_examples(self) -> Dict[str, Any]:
        """Audit all training data and return analysis"""

        training_file = self.training_dir / "training_data.jsonl"
        if not training_file.exists():
            logger.warning(f"Training file not found: {training_file}")
            return {"status": "no_data"}

        total = 0
        keep = []
        remove = []
        issues = defaultdict(int)

        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                total += 1
                try:
                    example = json.loads(line)

                    # Analyze this example
                    should_keep, reason = self._should_keep_example(example)

                    if should_keep:
                        keep.append(example)
                    else:
                        remove.append(example)
                        issues[reason] += 1

                except json.JSONDecodeError:
                    issues['invalid_json'] += 1
                    continue

        return {
            'total_examples': total,
            'keep_count': len(keep),
            'remove_count': len(remove),
            'issues': dict(issues),
            'quality_rate': len(keep) / total if total > 0 else 0,
            'examples_to_keep': keep
        }

    def _should_keep_example(self, example: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if training example should be kept"""

        response = example.get('assistant_response', '')
        user_input = example.get('user_input', '')
        quality_score = example.get('quality_score', 0.5)

        # Check 1: Quality score too low
        if quality_score < self.min_quality_score:
            return False, 'low_quality_score'

        # Check 2: Response too short or too long
        if len(response) < self.min_response_length:
            return False, 'response_too_short'
        if len(response) > self.max_response_length:
            return False, 'response_too_long'

        # Check 3: Contains error markers
        response_lower = response.lower()
        for marker in self.error_markers:
            if marker in response_lower:
                return False, 'contains_error_marker'

        # Check 4: Contains hallucination markers
        for marker in self.hallucination_markers:
            if marker in response_lower:
                return False, 'hallucinated_content'

        # Check 5: Empty or useless
        if not response.strip() or not user_input.strip():
            return False, 'empty_content'

        # Check 6: Repetitive pattern (same response copied)
        if response.count(response[:20]) > 2:
            return False, 'repetitive_content'

        return True, 'passed'

    def clean_training_data(self, backup: bool = True):
        """Clean training data by removing low-quality examples"""

        logger.info("Starting training data audit...")

        # Audit first
        audit_result = self.audit_training_examples()

        if audit_result.get('status') == 'no_data':
            logger.warning("No training data to clean")
            return audit_result

        logger.info(f"Audit complete:")
        logger.info(f"  Total: {audit_result['total_examples']}")
        logger.info(f"  Keep: {audit_result['keep_count']}")
        logger.info(f"  Remove: {audit_result['remove_count']}")
        logger.info(f"  Quality rate: {audit_result['quality_rate']:.1%}")

        if audit_result['issues']:
            logger.info("  Issues found:")
            for issue, count in sorted(audit_result['issues'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    - {issue}: {count}")

        # Backup original
        if backup:
            training_file = self.training_dir / "training_data.jsonl"
            backup_file = self.training_dir / f"training_data.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

            if training_file.exists():
                import shutil
                shutil.copy(training_file, backup_file)
                logger.info(f"Backup created: {backup_file}")

        # Write cleaned data
        training_file = self.training_dir / "training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for example in audit_result['examples_to_keep']:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Cleaned training data written to {training_file}")

        return audit_result

    def audit_knowledge_entities(self) -> Dict[str, Any]:
        """Audit knowledge entities - remove noise words captured as entities"""

        entities_file = self.knowledge_dir / "entities.json"
        if not entities_file.exists():
            return {"status": "no_data"}

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)

        # Noise words that shouldn't be entities
        noise_words = {
            'i', 'you', 'we', 'they', 'it', 'he', 'she',
            'my', 'your', 'our', 'their', 'his', 'her',
            'as', 'if', 'but', 'and', 'or', 'so', 'for',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'how', 'what', 'when', 'where', 'who', 'why',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might',
            'need', 'want', 'like', 'know', 'think', 'see', 'get',
            'make', 'take', 'give', 'go', 'come', 'say', 'tell',
            'let', 'nice', 'good', 'great', 'well', 'now', 'then',
            'since', 'given', 'while', 'however', 'therefore', 'thus',
            'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about'
        }

        keep = {}
        remove = {}

        for name, data in entities.items():
            name_lower = name.lower().strip()

            # Remove noise words
            if name_lower in noise_words:
                remove[name] = data
                continue

            # Remove if very short and low confidence
            if len(name) <= 2 and data['confidence'] < 0.8:
                remove[name] = data
                continue

            # Remove if never mentioned (mention_count = 0)
            if data['mention_count'] == 0:
                remove[name] = data
                continue

            keep[name] = data

        return {
            'total_entities': len(entities),
            'keep_count': len(keep),
            'remove_count': len(remove),
            'cleaned_entities': keep
        }

    def clean_knowledge_entities(self, backup: bool = True):
        """Clean knowledge entities"""

        logger.info("Starting knowledge entity audit...")

        audit_result = self.audit_knowledge_entities()

        if audit_result.get('status') == 'no_data':
            logger.warning("No knowledge data to clean")
            return audit_result

        logger.info(f"Entity audit complete:")
        logger.info(f"  Total: {audit_result['total_entities']}")
        logger.info(f"  Keep: {audit_result['keep_count']}")
        logger.info(f"  Remove: {audit_result['remove_count']}")

        # Backup original
        if backup:
            entities_file = self.knowledge_dir / "entities.json"
            backup_file = self.knowledge_dir / f"entities.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            if entities_file.exists():
                import shutil
                shutil.copy(entities_file, backup_file)
                logger.info(f"Backup created: {backup_file}")

        # Write cleaned data
        entities_file = self.knowledge_dir / "entities.json"
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(audit_result['cleaned_entities'], f, indent=2)

        logger.info(f"Cleaned entities written to {entities_file}")

        return audit_result

    def full_audit_and_clean(self):
        """Run full audit on all of Alice's learning data"""

        print("=" * 70)
        print("ALICE TRAINING DATA AUDITOR")
        print("=" * 70)
        print()

        results = {}

        # Clean training data
        print("1. Auditing training examples...")
        print("-" * 70)
        results['training'] = self.clean_training_data(backup=True)
        print()

        # Clean knowledge entities
        print("2. Auditing knowledge entities...")
        print("-" * 70)
        results['entities'] = self.clean_knowledge_entities(backup=True)
        print()

        # Summary
        print("=" * 70)
        print("AUDIT COMPLETE")
        print("=" * 70)

        if results.get('training', {}).get('status') != 'no_data':
            training = results['training']
            print(f"Training Data:")
            print(f"  Removed: {training['remove_count']} low-quality examples")
            print(f"  Kept: {training['keep_count']} high-quality examples")
            print(f"  Quality: {training['quality_rate']:.1%}")

        if results.get('entities', {}).get('status') != 'no_data':
            entities = results['entities']
            print(f"\nKnowledge Entities:")
            print(f"  Removed: {entities['remove_count']} noise entities")
            print(f"  Kept: {entities['keep_count']} valid entities")

        print()
        print("All backups saved to data/ directories")
        print("Alice will now learn only from high-quality data!")
        print()

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    auditor = TrainingDataAuditor()
    auditor.full_audit_and_clean()
