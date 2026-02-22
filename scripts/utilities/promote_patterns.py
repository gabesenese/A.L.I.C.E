"""
Pattern Promotion - Batch Learning from Simulation Logs

Analyzes simulation logs to automatically promote high-confidence patterns
with guardrails to ensure quality.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class PatternCandidate:
    """A candidate pattern for promotion"""
    normalized_input: str
    intent: str
    domain: str
    response_template: str
    frequency: int
    teacher_consistency: float  # How consistent teacher responses are
    alice_agreement: float  # How often Alice agrees with teacher
    variable_slots: Dict[str, str]  # Detected variable slots in template
    examples: List[str]  # Example inputs that match this pattern
    version: str = "v1.0"


class PatternPromoter:
    """
    Promotes patterns from simulation logs with quality guardrails
    """
    
    def __init__(
        self,
        min_frequency: int = 3,
        min_teacher_consistency: float = 0.8,
        min_alice_agreement: float = 0.7
    ):
        """
        Initialize pattern promoter
        
        Args:
            min_frequency: Minimum times pattern must appear (default 3)
            min_teacher_consistency: Minimum consistency in teacher responses (default 0.8 = 80%)
                This actually measures: how often teacher provides a response at all
            min_alice_agreement: Minimum agreement between Alice and teacher (default 0.7 = 70%)
        """
        self.min_frequency = min_frequency
        self.min_teacher_consistency = min_teacher_consistency
        self.min_alice_agreement = min_alice_agreement
        
        # Load existing patterns to avoid conflicts
        self.existing_patterns = self._load_existing_patterns()
        
        # Load negative feedback to filter out bad patterns
        self.negative_feedback = self._load_negative_feedback()
    
    def _load_existing_patterns(self) -> Set[str]:
        """Load existing patterns to avoid duplicates"""
        existing = set()
        
        patterns_file = Path("memory/learning_patterns.json")
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    data = json.load(f)
                    # Handle both dict and list formats
                    patterns_list = []
                    if isinstance(data, dict):
                        # If it's a dict with 'patterns' key or 'value' key
                        patterns_list = data.get("patterns", data.get("value", []))
                    elif isinstance(data, list):
                        # If it's directly a list
                        patterns_list = data
                    
                    for pattern in patterns_list:
                        if isinstance(pattern, dict):
                            # Use normalized input as key
                            normalized = self._normalize_input(pattern.get("pattern", ""))
                            if normalized:
                                existing.add(normalized)
            except Exception as e:
                logger.warning(f"Error loading existing patterns: {e}")
        
        return existing
    
    def _load_negative_feedback(self) -> Set[str]:
        """Load negative feedback to filter out patterns"""
        negative = set()
        
        feedback_file = Path("memory/user_feedback.json")
        if feedback_file.exists():
            try:
                with open(feedback_file) as f:
                    data = json.load(f)
                    # Handle both dict and list formats
                    feedback_list = []
                    if isinstance(data, dict):
                        feedback_list = data.get("feedback", data.get("value", []))
                    elif isinstance(data, list):
                        feedback_list = data
                    
                    for item in feedback_list:
                        if isinstance(item, dict):
                            # Filter by negative type if applicable
                            if item.get("type") == "negative" or "negative" not in str(item):
                                user_input = item.get("user_input", "")
                                normalized = self._normalize_input(user_input)
                                if normalized:
                                    negative.add(normalized)
            except Exception as e:
                logger.warning(f"Error loading negative feedback: {e}")
        
        return negative
    
    def _normalize_input(self, text: str) -> str:
        """Normalize user input for grouping"""
        # Lowercase, remove extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove trailing punctuation
        text = re.sub(r'[.!?]+$', '', text)
        
        return text
    
    def _detect_variable_slots(self, examples: List[str]) -> Dict[str, str]:
        """
        Detect variable slots in examples
        
        Args:
            examples: List of similar inputs
        
        Returns:
            Dict of slot names to patterns (e.g., {"entity": "PERSON"})
        """
        # Simple heuristic: if words differ at same position across examples,
        # it's likely a variable slot
        
        if len(examples) < 2:
            return {}
        
        # Tokenize all examples
        tokenized = [example.split() for example in examples]
        
        # Find positions where words differ
        variable_positions = set()
        min_len = min(len(tokens) for tokens in tokenized)
        
        for i in range(min_len):
            words_at_pos = set(tokens[i] for tokens in tokenized)
            if len(words_at_pos) > 1:  # Different words at this position
                variable_positions.add(i)
        
        # Create slot names based on position
        slots = {}
        for pos in sorted(variable_positions):
            # Get example values
            values = [tokens[pos] for tokens in tokenized if len(tokens) > pos]
            
            # Determine slot type (simple heuristic)
            if all(v.isdigit() for v in values):
                slots[f"number_{pos}"] = "NUMBER"
            elif all('@' in v for v in values):
                slots[f"email_{pos}"] = "EMAIL"
            else:
                slots[f"entity_{pos}"] = "TEXT"
        
        return slots
    
    def _create_response_template(
        self,
        teacher_responses: List[str],
        variable_slots: Dict[str, str]
    ) -> str:
        """
        Create response template from teacher responses
        
        Args:
            teacher_responses: List of teacher responses
            variable_slots: Detected variable slots
        
        Returns:
            Template string with {slot} placeholders
        """
        # For now, use most common response as template
        # TODO: Could do more sophisticated template extraction
        
        from collections import Counter
        
        # Normalize responses
        normalized_responses = [
            self._normalize_input(r) for r in teacher_responses if r
        ]
        
        if not normalized_responses:
            return ""
        
        # Get most common response
        response_counts = Counter(normalized_responses)
        template = response_counts.most_common(1)[0][0]
        
        return template
    
    def analyze_logs(self, log_file: Path) -> List[PatternCandidate]:
        """
        Analyze simulation logs to find pattern candidates
        
        Args:
            log_file: Path to auto_generated.jsonl
        
        Returns:
            List of pattern candidates meeting criteria
        """
        if not log_file.exists():
            logger.warning(f"Log file not found: {log_file}")
            return []
        
        # Group interactions by (intent, domain)
        groups = defaultdict(lambda: {
            "inputs": [],
            "teacher_responses": [],
            "alice_responses": [],
            "route_matches": [],
            "intent_matches": []
        })
        
        # Read log file
        with open(log_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # Skip if no teacher response
                    if not entry.get("teacher_response"):
                        continue
                    
                    # Skip if needs learning but has negative feedback
                    normalized = self._normalize_input(entry.get("user_input", ""))
                    if normalized in self.negative_feedback:
                        continue
                    
                    # Group by (intent, domain)
                    key = (entry.get("actual_intent"), entry.get("domain"))
                    
                    groups[key]["inputs"].append(entry.get("user_input", ""))
                    groups[key]["teacher_responses"].append(entry.get("teacher_response", ""))
                    groups[key]["alice_responses"].append(entry.get("alice_response", ""))
                    groups[key]["route_matches"].append(entry.get("route_match", False))
                    groups[key]["intent_matches"].append(entry.get("intent_match", False))
                    
                except json.JSONDecodeError:
                    continue
        
        # Analyze each group
        candidates = []
        
        for (intent, domain), data in groups.items():
            frequency = len(data["inputs"])
            
            # Check minimum frequency
            if frequency < self.min_frequency:
                continue
            
            # Calculate teacher response rate (how often teacher has response)
            teacher_response_rate = sum(
                1 for tr in data["teacher_responses"] if tr
            ) / frequency if frequency > 0 else 0.0
            
            # We require that teacher responded often enough (at least min_teacher_consistency %)
            if teacher_response_rate < self.min_teacher_consistency:
                continue
            
            # Calculate route and intent agreement (how often scenarios got it right)
            correct_routes = sum(data["route_matches"])
            correct_intents = sum(data["intent_matches"])
            route_agreement = correct_routes / frequency if frequency > 0 else 0.0
            intent_agreement = correct_intents / frequency if frequency > 0 else 0.0
            
            # Require reasonable agreement (at least min_alice_agreement %)
            avg_agreement = (route_agreement + intent_agreement) / 2
            if avg_agreement < self.min_alice_agreement:
                continue
            
            # Detect variable slots
            variable_slots = self._detect_variable_slots(data["inputs"])
            
            # Create response template
            response_template = self._create_response_template(
                data["teacher_responses"], variable_slots
            )
            
            # Use most common normalized input as pattern
            normalized_inputs = [self._normalize_input(i) for i in data["inputs"]]
            from collections import Counter
            pattern_input = Counter(normalized_inputs).most_common(1)[0][0]
            
            # Skip if already exists
            if pattern_input in self.existing_patterns:
                logger.debug(f"Skipping existing pattern: {pattern_input}")
                continue
            
            # Create candidate
            candidate = PatternCandidate(
                normalized_input=pattern_input,
                intent=intent,
                domain=domain,
                response_template=response_template,
                frequency=frequency,
                teacher_consistency=teacher_response_rate,
                alice_agreement=avg_agreement,
                variable_slots=variable_slots,
                examples=data["inputs"][:5]  # Keep first 5 examples
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def promote_patterns(
        self,
        candidates: List[PatternCandidate],
        auto_apply: bool = True
    ) -> int:
        """
        Promote pattern candidates to learning_patterns.json
        
        Args:
            candidates: List of pattern candidates
            auto_apply: Whether to auto-apply without review
        
        Returns:
            Number of patterns promoted
        """
        if not candidates:
            logger.info("No patterns to promote")
            return 0
        
        # Filter candidates that meet auto-apply criteria
        auto_candidates = []
        manual_candidates = []
        
        for candidate in candidates:
            if (candidate.teacher_consistency >= self.min_teacher_consistency and
                candidate.frequency >= 5):  # Higher bar for auto-apply
                auto_candidates.append(candidate)
            else:
                manual_candidates.append(candidate)
        
        # Load existing patterns file
        patterns_file = Path("memory/learning_patterns.json")
        patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        if patterns_file.exists():
            with open(patterns_file) as f:
                patterns_data = json.load(f)
                # Normalize to dict format if it's a list
                if isinstance(patterns_data, list):
                    patterns_data = {
                        "version": "1.0",
                        "last_updated": datetime.now().isoformat(),
                        "patterns": patterns_data
                    }
        else:
            patterns_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "patterns": []
            }
        
        # Ensure patterns key exists
        if "patterns" not in patterns_data:
            patterns_data["patterns"] = []
        
        # Add auto-apply candidates
        promoted_count = 0
        
        if auto_apply and auto_candidates:
            logger.info(f"\n[AUTO-PROMOTE] Auto-promoting {len(auto_candidates)} high-confidence patterns")
            
            for candidate in auto_candidates:
                pattern_entry = {
                    "pattern": candidate.normalized_input,
                    "intent": candidate.intent,
                    "domain": candidate.domain,
                    "response": candidate.response_template,
                    "frequency": candidate.frequency,
                    "teacher_consistency": candidate.teacher_consistency,
                    "alice_agreement": candidate.alice_agreement,
                    "variable_slots": candidate.variable_slots,
                    "examples": candidate.examples,
                    "version": candidate.version,
                    "promoted_at": datetime.now().isoformat(),
                    "auto_promoted": True
                }
                
                patterns_data["patterns"].append(pattern_entry)
                promoted_count += 1
                
                logger.info(f"  [OK] {candidate.normalized_input} (freq={candidate.frequency}, consistency={candidate.teacher_consistency:.1%})")
        
        # Save manual review candidates
        if manual_candidates:
            logger.info(f"\n[REVIEW] {len(manual_candidates)} patterns need manual review")
            
            review_file = Path("memory/patterns_for_review.json")
            review_data = {
                "last_updated": datetime.now().isoformat(),
                "candidates": [asdict(c) for c in manual_candidates]
            }
            
            with open(review_file, "w") as f:
                json.dump(review_data, f, indent=2)
            
            logger.info(f"  [SAVED] Saved to {review_file}")
        
        # Update patterns file
        patterns_data["last_updated"] = datetime.now().isoformat()
        
        with open(patterns_file, "w") as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"\n[OK] Promoted {promoted_count} patterns to {patterns_file}")
        
        return promoted_count


def main():
    """Main entry point for pattern promotion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Promote patterns from simulation logs"
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=3,
        help='Minimum frequency for pattern promotion'
    )
    parser.add_argument(
        '--min-consistency',
        type=float,
        default=0.8,
        help='Minimum teacher consistency (0-1)'
    )
    parser.add_argument(
        '--no-auto-apply',
        action='store_true',
        help='Disable auto-apply (review all patterns manually)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info(" A.L.I.C.E Pattern Promotion - Batch Learning from Simulations")
    logger.info("=" * 70)
    logger.info(f"Min Frequency: {args.min_frequency}")
    logger.info(f"Min Teacher Consistency: {args.min_consistency:.0%}")
    logger.info(f"Auto-Apply: {'Disabled' if args.no_auto_apply else 'Enabled'}")
    logger.info("=" * 70)
    
    # Create promoter
    promoter = PatternPromoter(
        min_frequency=args.min_frequency,
        min_teacher_consistency=args.min_consistency
    )
    
    # Analyze logs
    log_file = Path("data/training/auto_generated.jsonl")
    logger.info(f"\n Analyzing simulation logs: {log_file}")
    
    candidates = promoter.analyze_logs(log_file)
    logger.info(f"[OK] Found {len(candidates)} pattern candidates")
    
    if candidates:
        # Show summary
        logger.info("\nCandidate Summary:")
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"\n{i}. Pattern: '{candidate.normalized_input}'")
            logger.info(f"   Intent: {candidate.intent}, Domain: {candidate.domain}")
            logger.info(f"   Frequency: {candidate.frequency}")
            logger.info(f"   Teacher Consistency: {candidate.teacher_consistency:.1%}")
            logger.info(f"   Response: '{candidate.response_template[:60]}...'")
        
        # Promote patterns
        logger.info("\n" + "=" * 70)
        promoted = promoter.promote_patterns(
            candidates,
            auto_apply=not args.no_auto_apply
        )
        
        logger.info(f"\n[OK] Pattern promotion complete! Promoted {promoted} patterns.")
    else:
        logger.info("\nNo patterns meet promotion criteria.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
