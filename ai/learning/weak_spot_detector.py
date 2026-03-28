"""
Weak-Spot Detector - Tier 3: Automated Weak-Spot Detection

Analyzes failure patterns and generates PostMortems.
ALICE learns from her mistakes and improves over time.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Failure:
    """Record of a single failure."""
    
    failure_id: str
    intent: str
    timestamp: str
    error_message: str
    routing_path: str
    reason: str
    context_tags: List[str] = field(default_factory=list)
    was_retried: bool = False
    retry_succeeded: bool = False
    

@dataclass
class PostMortem:
    """Analysis of a recurring failure."""
    
    pattern_name: str
    first_occurrence: str
    occurrences: int
    intent_group: str
    root_cause_hypothesis: str
    affected_paths: List[str]
    severity: str  # low, medium, high
    suggested_fixes: List[str]
    generated_test: Optional[str] = None
    improvement_made: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeakSpotDetector:
    """Detects weak spots in ALICE's reasoning/routing and generates insights."""
    
    def __init__(self, analysis_window_turns: int = 50, min_occurrences_for_pattern: int = 3):
        """
        Args:
            analysis_window_turns: Analyze last N turns for patterns
            min_occurrences_for_pattern: Need N failures to call it a pattern
        """
        self.analysis_window_turns = analysis_window_turns
        self.min_occurrences_for_pattern = min_occurrences_for_pattern
        self.failures: List[Failure] = []
        self.postmortems: List[PostMortem] = []
        self.turn_count = 0
        self.improvement_queue: List[Dict[str, Any]] = []
    
    def record_failure(
        self,
        intent: str,
        error_message: str,
        routing_path: str,
        reason: str,
        context_tags: List[str] = None,
    ) -> Failure:
        """Record a failure for later analysis."""
        self.turn_count += 1
        context_tags = context_tags or []
        
        failure = Failure(
            failure_id=f"fail_{self.turn_count}",
            intent=intent,
            timestamp=datetime.now().isoformat(),
            error_message=error_message,
            routing_path=routing_path,
            reason=reason,
            context_tags=context_tags,
        )
        
        self.failures.append(failure)
        logger.info(f"[WeakSpot] Recorded failure: {intent} via {routing_path}")
        
        return failure
    
    def mark_retry_success(self, failure_id: str, succeeded: bool) -> None:
        """Mark if a retry succeeded."""
        for failure in self.failures:
            if failure.failure_id == failure_id:
                failure.was_retried = True
                failure.retry_succeeded = succeeded
                logger.info(f"[WeakSpot] Retry outcome: succeeded={succeeded}")
                break
    
    def analyze_patterns(self) -> List[PostMortem]:
        """Analyze failures to detect recurring weak spots."""
        if len(self.failures) < self.min_occurrences_for_pattern:
            return []
        
        postmortems = []
        
        # Analyze recent failures (sliding window)
        recent_failures = self.failures[-self.analysis_window_turns:]
        
        # Group by routing path
        failures_by_path = self._group_failures(recent_failures, group_by="routing_path")
        
        for path, path_failures in failures_by_path.items():
            if len(path_failures) >= self.min_occurrences_for_pattern:
                # Detected a weak routing path
                postmortem = self._analyze_routing_weakness(path, path_failures)
                postmortems.append(postmortem)
        
        # Group by intent
        failures_by_intent = self._group_failures(recent_failures, group_by="intent")
        
        for intent, intent_failures in failures_by_intent.items():
            if len(intent_failures) >= self.min_occurrences_for_pattern:
                # Detected a weak intent
                postmortem = self._analyze_intent_weakness(intent, intent_failures)
                postmortems.append(postmortem)
        
        # Group by error message pattern
        failures_by_error = self._group_failures(recent_failures, group_by="error_pattern")
        
        for error_pattern, error_failures in failures_by_error.items():
            if len(error_failures) >= self.min_occurrences_for_pattern:
                postmortem = self._analyze_error_pattern(error_pattern, error_failures)
                postmortems.append(postmortem)
        
        self.postmortems = postmortems
        logger.info(f"[WeakSpot] Generated {len(postmortems)} postmortems")
        
        return postmortems
    
    def _group_failures(self, failures: List[Failure], group_by: str) -> Dict[str, List[Failure]]:
        """Group failures by a specific attribute."""
        groups = {}
        
        for failure in failures:
            if group_by == "routing_path":
                key = failure.routing_path
            elif group_by == "intent":
                key = failure.intent.split(":")[0]  # Get intent family
            elif group_by == "error_pattern":
                # Extract error pattern (first few words)
                key = " ".join(failure.error_message.split()[:3])
            else:
                key = str(getattr(failure, group_by, "unknown"))
            
            if key not in groups:
                groups[key] = []
            groups[key].append(failure)
        
        return groups
    
    def _analyze_routing_weakness(self, path: str, failures: List[Failure]) -> PostMortem:
        """Analyze why a routing path keeps failing."""
        intents = Counter(f.intent for f in failures)
        errors = [f.error_message for f in failures]
        retry_successes = sum(1 for f in failures if f.was_retried and f.retry_succeeded)
        
        # Hypothesis
        if retry_successes > len(failures) / 2:
            hypothesis = "Transient failures—route is generally correct but occasionally unstable"
        else:
            top_intent = intents.most_common(1)[0][0]
            hypothesis = f"Routing path '{path}' doesn't handle intent '{top_intent}' well"
        
        postmortem = PostMortem(
            pattern_name=f"weak_routing__{path.replace(':', '_')}",
            first_occurrence=failures[0].timestamp,
            occurrences=len(failures),
            intent_group=intents.most_common(1)[0][0] if intents else "unknown",
            root_cause_hypothesis=hypothesis,
            affected_paths=[path],
            severity="high" if len(failures) >= 5 else "medium",
            suggested_fixes=[
                f"Review routing decision logic for {path}",
                f"Add retry handling for transient failures",
                f"Add additional validation before committing to {path}",
                f"Consider adding fallback to alternate routing for {intents.most_common(1)[0][0]}",
            ],
        )
        
        # Generate test
        postmortem.generated_test = self._generate_test_case(path, failures)
        
        return postmortem
    
    def _analyze_intent_weakness(self, intent: str, failures: List[Failure]) -> PostMortem:
        """Analyze why an intent keeps failing."""
        paths = Counter(f.routing_path for f in failures)
        errors = [f.error_message for f in failures]
        
        postmortem = PostMortem(
            pattern_name=f"weak_intent__{intent.replace(':', '_')}",
            first_occurrence=failures[0].timestamp,
            occurrences=len(failures),
            intent_group=intent,
            root_cause_hypothesis=f"Intent '{intent}' has underlying ambiguity or complex preconditions",
            affected_paths=list(paths.keys()),
            severity="high" if len(failures) >= 5 else "low",
            suggested_fixes=[
                f"Improve clarification flow for intent '{intent}'",
                f"Add multi-step fallback for {intent}",
                f"Document common failure modes for {intent}",
            ],
        )
        
        return postmortem
    
    def _analyze_error_pattern(self, error_pattern: str, failures: List[Failure]) -> PostMortem:
        """Analyze a recurring error pattern."""
        paths = [f.routing_path for f in failures]
        intents = [f.intent for f in failures]
        
        postmortem = PostMortem(
            pattern_name=f"error_pattern__{error_pattern.replace(' ', '_')[:30]}",
            first_occurrence=failures[0].timestamp,
            occurrences=len(failures),
            intent_group="mixed" if len(set(intents)) > 1 else intents[0],
            root_cause_hypothesis=f"System consistently encounters error: '{error_pattern}'",
            affected_paths=list(set(paths)),
            severity="critical" if len(failures) >= 5 else "high",
            suggested_fixes=[
                f"Debug and fix root cause: {error_pattern}",
                f"Add error handling and recovery for this error type",
                f"Add pre-check validation to prevent this error",
            ],
        )
        
        return postmortem
    
    def _generate_test_case(self, path: str, failures: List[Failure]) -> str:
        """Generate a regression test based on failures."""
        first = failures[0]
        
        test_code = f"""
def test_weak_spot_regression_{path.replace(':', '_')}():
    \"\"\"Regression test: {path} should handle {first.intent} consistently\"\"\"
    # Context: This routing path has failed {len(failures)} times
    # Typical error: {first.error_message}
    
    alice = ALICE(...)
    result = alice.process_input(
        user_input="[example input for {first.intent}]",
        debug=True
    )
    
    assert result['success'], f"Expected success for {path}, got: {{result.get('error')}}"
    assert result['routing_path'] == '{path}'
    
    # Verify consistency
    for i in range(5):  # Run 5 times
        result = alice.process_input(...)
        assert result['success'], f"Iteration {{i}}: Inconsistent failure"
"""
        return test_code.strip()
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get prioritized list of improvements to make."""
        improvements = []
        
        for postmortem in sorted(self.postmortems, key=lambda p: p.occurrences, reverse=True):
            improvement = {
                "priority": "high" if postmortem.occurrences >= 5 else "medium",
                "pattern": postmortem.pattern_name,
                "occurrences": postmortem.occurrences,
                "suggestion": postmortem.suggested_fixes[0] if postmortem.suggested_fixes else "",
                "test_case": postmortem.generated_test,
                "severity": postmortem.severity,
            }
            improvements.append(improvement)
        
        return improvements
    
    def get_weak_spots_report(self) -> str:
        """Generate readable report of weak spots."""
        if not self.postmortems:
            return "No weak spots detected. ALICE is performing well!"
        
        lines = ["WEAK-SPOT ANALYSIS", "=" * 60]
        
        for pm in sorted(self.postmortems, key=lambda p: p.occurrences, reverse=True):
            lines.append(f"\n{pm.pattern_name}")
            lines.append(f"  Occurrences: {pm.occurrences}")
            lines.append(f"  Severity: {pm.severity}")
            lines.append(f"  Root cause: {pm.root_cause_hypothesis}")
            if pm.suggested_fixes:
                lines.append(f"  Fix: {pm.suggested_fixes[0]}")
        
        return "\n".join(lines)
    
    def should_trigger_improvement(self) -> bool:
        """Check if improvements should be triggered."""
        if not self.postmortems:
            return False
        
        # Trigger if there's a high-severity pattern with 5+ occurrences
        return any(p.occurrences >= 5 and p.severity in ("high", "critical") for p in self.postmortems)
