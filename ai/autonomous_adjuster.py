"""
Autonomous Threshold Adjuster
Reads training logs and automatically adjusts routing/NLP thresholds and rules
based on error patterns without manual intervention.

Key responsibilities:
1. Analyze error patterns in domain-specific accuracy
2. Adjust clarification thresholds when wrong tools are chosen
3. Update NLP domain keyword weights
4. Auto-promote safe patterns when confidence is high
5. Log all adjustments for transparency
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class AutonomousThresholdAdjuster:
    """
    Reads scenario/training logs and autonomously adjusts routing thresholds.
    Goals:
    - Reduce clarification % when tools were correctly chosen
    - Increase clarification % when wrong tools were chosen
    - Auto-update NLP keyword weights by domain error rate
    """
    
    # Error thresholds for adjustment
    HIGH_ERROR_THRESHOLD = 0.3  # If >30% errors in domain, increase clarification
    LOW_ERROR_THRESHOLD = 0.15   # If <15% errors, we can be more confident
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "training"
        self.memory_dir = self.project_root / "memory"
        
        # Files to track
        self.training_log = self.data_dir / "auto_generated.jsonl"
        self.thresholds_file = self.data_dir / "thresholds.json"
        self.stats_file = self.data_dir / "training_stats.json"
        
        self.thresholds = self._load_thresholds()
        self.logger_file = self.data_dir / "autonomous_adjuster.log"
        
        logger.info("[AutonomousAdjuster] Initialized")
    
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load current thresholds"""
        # Default thresholds
        defaults = {
            'clarification_threshold': 0.7,  # Confidence below this triggers clarification
            'intent_confidence_min': 0.6,    # Minimum intent confidence to proceed
            'strong_domain_keyword_threshold': 0.5,
            'nlp_weights': self._default_nlp_weights(),
            'domain_confidence': {},
            'last_adjusted': None
        }
        
        if self.thresholds_file.exists():
            try:
                with open(self.thresholds_file, 'r') as f:
                    loaded = json.load(f)
                    
                    # Merge with defaults to ensure all keys exist
                    for key, default_value in defaults.items():
                        if key not in loaded:
                            loaded[key] = default_value
                        elif key == 'domain_confidence' and not isinstance(loaded[key], dict):
                            loaded[key] = {}
                        elif key == 'nlp_weights' and not isinstance(loaded[key], dict):
                            loaded[key] = default_value
                    
                    return loaded
            except Exception as e:
                logger.warning(f"[AutonomousAdjuster] Error loading thresholds: {e}")
        
        return defaults
    
    def _default_nlp_weights(self) -> Dict[str, float]:
        """Default NLP domain keyword weights"""
        return {
            'email': 1.0,
            'notes': 1.0,
            'weather': 1.0,
            'time': 1.0,
            'file': 0.8,  # Lower because file ops are risky
            'memory': 0.8,  # Lower because memory is complex
            'calendar': 1.0,
            'music': 1.0,
            'maps': 1.0,
            'system': 0.9
        }
    
    def _save_thresholds(self):
        """Save thresholds to disk"""
        try:
            with open(self.thresholds_file, 'w') as f:
                json.dump(self.thresholds, f, indent=2)
        except Exception as e:
            logger.error(f"[AutonomousAdjuster] Error saving thresholds: {e}")
    
    def analyze_training_logs(self) -> Dict[str, Any]:
        """
        Read training logs and compute error rates by domain/intent/route
        
        Returns:
            Dict with 'domain_errors', 'route_errors', 'intent_errors'
        """
        if not self.training_log.exists():
            logger.info("[AutonomousAdjuster] No training log found")
            return {}
        
        domain_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'route_errors': 0, 'intent_errors': 0})
        route_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        
        try:
            with open(self.training_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    domain = log_entry.get('domain', 'unknown')
                    route_match = log_entry.get('route_match', True)
                    intent_match = log_entry.get('intent_match', True)
                    actual_route = log_entry.get('actual_route', 'unknown')
                    success = log_entry.get('success', False)
                    
                    # Domain tracking
                    domain_stats[domain]['total'] += 1
                    if not success:
                        domain_stats[domain]['errors'] += 1
                    if not route_match:
                        domain_stats[domain]['route_errors'] += 1
                    if not intent_match:
                        domain_stats[domain]['intent_errors'] += 1
                    
                    # Route tracking
                    route_stats[actual_route]['total'] += 1
                    if not success:
                        route_stats[actual_route]['errors'] += 1
        
        except Exception as e:
            logger.warning(f"[AutonomousAdjuster] Error analyzing logs: {e}")
            return {}
        
        # Convert to percentages
        domain_errors = {}
        for domain, stats in domain_stats.items():
            if stats['total'] > 0:
                domain_errors[domain] = {
                    'total': stats['total'],
                    'error_rate': stats['errors'] / stats['total'],
                    'route_error_rate': stats['route_errors'] / stats['total'],
                    'intent_error_rate': stats['intent_errors'] / stats['total']
                }
        
        route_errors = {}
        for route, stats in route_stats.items():
            if stats['total'] > 0:
                route_errors[route] = {
                    'total': stats['total'],
                    'error_rate': stats['errors'] / stats['total']
                }
        
        return {
            'domain_errors': domain_errors,
            'route_errors': route_errors,
            'timestamp': datetime.now().isoformat()
        }
    
    def adjust_thresholds(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Based on error analysis, adjust thresholds automatically
        
        Returns:
            Dict of adjustments made
        """
        adjustments = {
            'adjustments_made': [],
            'confidence_updates': {},
            'weight_updates': {}
        }
        
        if not analysis:
            return adjustments
        
        # Ensure domain_confidence is a dict (safety check)
        if not isinstance(self.thresholds.get('domain_confidence'), dict):
            self.thresholds['domain_confidence'] = {}
        
        # Ensure nlp_weights is a dict (safety check)
        if not isinstance(self.thresholds.get('nlp_weights'), dict):
            self.thresholds['nlp_weights'] = self._default_nlp_weights()
        
        domain_errors = analysis.get('domain_errors', {})
        
        # Adjust per-domain confidence based on error rates
        for domain, error_stats in domain_errors.items():
            error_rate = error_stats['error_rate']
            
            # High errors → lower confidence (be more cautious)
            if error_rate > self.HIGH_ERROR_THRESHOLD:
                old_confidence = self.thresholds['domain_confidence'].get(domain, 1.0)
                new_confidence = max(0.3, old_confidence - 0.15)  # Never go below 0.3
                
                self.thresholds['domain_confidence'][domain] = new_confidence
                adjustments['confidence_updates'][domain] = {
                    'old': old_confidence,
                    'new': new_confidence,
                    'reason': f'high_error_rate_{error_rate:.1%}'
                }
                
                adjustments['adjustments_made'].append(
                    f"Domain '{domain}': Lowered confidence from {old_confidence:.2f} to {new_confidence:.2f} (error rate {error_rate:.1%})"
                )
                
                logger.info(f"[AutonomousAdjuster] {domain}: Error rate {error_rate:.1%} → lowering confidence")
            
            # Low errors → can be more confident
            elif error_rate < self.LOW_ERROR_THRESHOLD:
                old_confidence = self.thresholds['domain_confidence'].get(domain, 1.0)
                new_confidence = min(1.0, old_confidence + 0.1)
                
                if new_confidence != old_confidence:
                    self.thresholds['domain_confidence'][domain] = new_confidence
                    adjustments['confidence_updates'][domain] = {
                        'old': old_confidence,
                        'new': new_confidence,
                        'reason': f'low_error_rate_{error_rate:.1%}'
                    }
                    
                    adjustments['adjustments_made'].append(
                        f"Domain '{domain}': Raised confidence from {old_confidence:.2f} to {new_confidence:.2f} (error rate {error_rate:.1%})"
                    )
                    
                    logger.info(f"[AutonomousAdjuster] {domain}: Error rate {error_rate:.1%} → raising confidence")
            
            # Update NLP weights based on route error rate
            if domain in self.thresholds['nlp_weights']:
                route_error_rate = error_stats['route_error_rate']
                
                if route_error_rate > self.HIGH_ERROR_THRESHOLD:
                    old_weight = self.thresholds['nlp_weights'][domain]
                    new_weight = max(0.5, old_weight - 0.1)
                    
                    self.thresholds['nlp_weights'][domain] = new_weight
                    adjustments['weight_updates'][domain] = {
                        'old': old_weight,
                        'new': new_weight,
                        'reason': f'high_route_error_{route_error_rate:.1%}'
                    }
                    
                    adjustments['adjustments_made'].append(
                        f"Domain '{domain}': Lowered NLP weight from {old_weight:.2f} to {new_weight:.2f} (route error {route_error_rate:.1%})"
                    )
        
        # Adjust global clarification threshold if CLARIFICATION route has low errors
        clarification_stats = analysis.get('route_errors', {}).get('CLARIFICATION', {})
        if clarification_stats:
            clarity_error_rate = clarification_stats['error_rate']
            if clarity_error_rate < 0.1:  # If clarification is working well
                old_threshold = self.thresholds['clarification_threshold']
                new_threshold = min(0.85, old_threshold + 0.05)
                
                if new_threshold != old_threshold:
                    self.thresholds['clarification_threshold'] = new_threshold
                    adjustments['adjustments_made'].append(
                        f"Clarification: Raised threshold from {old_threshold:.2f} to {new_threshold:.2f} (low error rate)"
                    )
        
        # Save if adjustments made
        if adjustments['adjustments_made']:
            self.thresholds['last_adjusted'] = datetime.now().isoformat()
            self._save_thresholds()
            self._log_adjustments(adjustments)
        
        return adjustments
    
    def _log_adjustments(self, adjustments: Dict[str, Any]):
        """Log adjustments for audit trail"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'adjustments': adjustments
            }
            
            with open(self.logger_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"[AutonomousAdjuster] Error logging adjustments: {e}")
    
    def get_current_thresholds(self) -> Dict[str, Any]:
        """Get current thresholds"""
        return self.thresholds.copy()
    
    def recommend_actions(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Based on analysis, recommend actions (for logging, not auto-execution)
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        domain_errors = analysis.get('domain_errors', {})
        
        for domain, stats in domain_errors.items():
            error_rate = stats['error_rate']
            
            if error_rate > 0.5:
                recommendations.append(
                    f"[URGENT] Domain '{domain}' has {error_rate:.1%} error rate. "
                    f"Consider reviewing NLP rules or scenario definitions for this domain."
                )
            elif error_rate > self.HIGH_ERROR_THRESHOLD:
                recommendations.append(
                    f"Domain '{domain}' has {error_rate:.1%} error rate. "
                    f"Consider adding more training examples for {stats.get('intent_error_rate', 0):.1%} intent errors "
                    f"and {stats.get('route_error_rate', 0):.1%} route errors."
                )
        
        # Check LLM usage
        route_stats = analysis.get('route_errors', {})
        llm_fallback = route_stats.get('LLM_FALLBACK', {})
        if llm_fallback and llm_fallback.get('total', 0) > 10:
            recommendations.append(
                f"LLM_FALLBACK is being used {llm_fallback['total']} times. "
                f"Consider creating deterministic patterns for frequent questions."
            )
        
        return recommendations
    
    def run_full_adjustment_cycle(self) -> Dict[str, Any]:
        """
        Run the complete adjustment cycle:
        1. Analyze training logs
        2. Adjust thresholds
        3. Generate recommendations
        
        Returns:
            Complete summary
        """
        logger.info("[AutonomousAdjuster] Starting full adjustment cycle...")
        
        analysis = self.analyze_training_logs()
        adjustments = self.adjust_thresholds(analysis)
        recommendations = self.recommend_actions(analysis)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'adjustments': adjustments,
            'recommendations': recommendations,
            'current_thresholds': self.get_current_thresholds()
        }
        
        # Log summary
        logger.info(f"[AutonomousAdjuster] Cycle complete:")
        logger.info(f"  - Analysis domains: {len(analysis.get('domain_errors', {}))}")
        logger.info(f"  - Adjustments made: {len(adjustments.get('adjustments_made', []))}")
        logger.info(f"  - Recommendations: {len(recommendations)}")
        
        return summary


class RulesOptimizer:
    """
    Reads domain-specific error patterns and updates NLP rules/patterns
    E.g., if file:create is always mis-routed as notes:create, update keywords
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "training"
        self.nlp_rules_file = self.data_dir / "nlp_rules_optimization.json"
        self.rules = self._load_rules()
        
        logger.info("[RulesOptimizer] Initialized")
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load optimization rules"""
        if self.nlp_rules_file.exists():
            try:
                with open(self.nlp_rules_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[RulesOptimizer] Error loading rules: {e}")
        
        return {
            'confusion_pairs': {},  # intent_A → intent_B: count
            'optimized_keywords': {},
            'last_updated': None
        }
    
    def _save_rules(self):
        """Save rules"""
        try:
            with open(self.nlp_rules_file, 'w') as f:
                json.dump(self.rules, f, indent=2)
        except Exception as e:
            logger.error(f"[RulesOptimizer] Error saving rules: {e}")
    
    def analyze_confusion_patterns(self, training_log_path: Path) -> Dict[str, int]:
        """
        Find which intents are commonly confused with each other
        E.g., "file:create" frequently misclassified as "notes:create"
        
        Returns:
            Dict of {(expected_intent, actual_intent): count}
        """
        confusion_pairs = defaultdict(int)
        
        try:
            with open(training_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    expected = entry.get('expected_intent')
                    actual = entry.get('actual_intent')
                    
                    if expected != actual and expected and actual:
                        confusion_pairs[(expected, actual)] += 1
        
        except Exception as e:
            logger.warning(f"[RulesOptimizer] Error analyzing confusion patterns: {e}")
        
        return dict(confusion_pairs)
    
    def generate_rule_suggestions(self, confusion_pairs: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        For top confusion pairs, suggest keyword adjustments
        
        Returns:
            List of suggested rules
        """
        suggestions = []
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        for (expected, actual), count in sorted_pairs[:5]:  # Top 5 confusions
            if count < 2:  # Only suggest if common pattern
                continue
            
            suggestion = {
                'expected_intent': expected,
                'actual_intent': actual,
                'confusion_count': count,
                'suggested_action': f"Add discriminating keywords to distinguish {expected} from {actual}",
                'priority': 'high' if count >= 5 else 'medium'
            }
            
            suggestions.append(suggestion)
            logger.info(
                f"[RulesOptimizer] Confusion pattern: {expected} misclassified as {actual} {count} times"
            )
        
        return suggestions
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run complete optimization cycle"""
        training_log = self.data_dir / "auto_generated.jsonl"
        
        if not training_log.exists():
            logger.info("[RulesOptimizer] No training log found")
            return {}
        
        confusion_pairs = self.analyze_confusion_patterns(training_log)
        suggestions = self.generate_rule_suggestions(confusion_pairs)
        
        self.rules['confusion_pairs'] = confusion_pairs
        self.rules['suggested_optimizations'] = suggestions
        self.rules['last_updated'] = datetime.now().isoformat()
        
        self._save_rules()
        
        return {
            'confusion_pairs': len(confusion_pairs),
            'top_confusion': sorted_pairs[0] if confusion_pairs else None,
            'optimization_suggestions': suggestions
        }


def create_autonomous_adjuster(project_root: Optional[Path] = None) -> AutonomousThresholdAdjuster:
    """Factory function"""
    return AutonomousThresholdAdjuster(project_root)


def create_rules_optimizer(project_root: Optional[Path] = None) -> RulesOptimizer:
    """Factory function"""
    return RulesOptimizer(project_root)
