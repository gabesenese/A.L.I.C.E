"""
Ollama Teacher Comparison System

Runs Ollama as a background teacher during training:
1. For each scenario/interaction, ask the teacher (Ollama) for the ideal answer
2. Compare Alice's answer to the teacher's answer
3. Log quality metrics (similarity, correctness, helpfulness)
4. Auto-promote patterns when Alice matches teacher > 80% of the time
5. Flag for manual review when Alice differs from teacher frequently

This creates a self-improving loop: teacher feedback → adjustments → better performance
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class OllamaTeacherComparison:
    """
    Compares Alice's responses to Ollama teacher responses
    and generates quality scores for autonomous learning
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "training"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparison_log = self.data_dir / "teacher_comparisons.jsonl"
        self.quality_scores = self.data_dir / "response_quality_scores.json"
        self.patterns_validated = self.data_dir / "teacher_validated_patterns.json"
        
        # Try to import Ollama
        self.ollama = None
        self.llm_engine = None
        try:
            from ai.core.llm_engine import LocalLLMEngine, LLMConfig
            config = LLMConfig(model="llama3.1:8b")
            self.llm_engine = LocalLLMEngine(config)
            logger.info("[TeacherComparison] Ollama LLM Engine initialized")
        except Exception as e:
            logger.warning(f"[TeacherComparison] Could not initialize Ollama: {e}")
        
        self.scores = self._load_scores()
        logger.info("[TeacherComparison] Initialized")
    
    def _load_scores(self) -> Dict[str, Any]:
        """Load existing quality scores"""
        if self.quality_scores.exists():
            try:
                with open(self.quality_scores, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[TeacherComparison] Error loading scores: {e}")
        
        return {
            'total_comparisons': 0,
            'avg_similarity': 0.0,
            'by_domain': {},
            'by_intent': {},
            'high_quality_patterns': [],
            'low_quality_patterns': []
        }
    
    def _save_scores(self):
        """Save quality scores"""
        try:
            with open(self.quality_scores, 'w') as f:
                json.dump(self.scores, f, indent=2)
        except Exception as e:
            logger.error(f"[TeacherComparison] Error saving scores: {e}")
    
    def get_teacher_response(self, user_input: str, intent: Optional[str] = None, context: Optional[Dict] = None) -> Optional[str]:
        """
        Get ideal response from teacher (Ollama)
        
        Args:
            user_input: User's input
            intent: Detected intent (for context)
            context: Additional context for the teacher
        
        Returns:
            Teacher's ideal response
        """
        if not self.llm_engine:
            logger.warning("[TeacherComparison] Ollama not available, skipping teacher response")
            return None
        
        try:
            # Craft prompt for teacher
            prompt = f"""You are a helpful AI assistant. The user said: "{user_input}"

"""
            if intent:
                prompt += f"Domain/Intent: {intent}\n"
            
            if context:
                prompt += f"Context: {json.dumps(context, indent=2)}\n"
            
            prompt += "\nProvide a concise, helpful response:"
            
            # Get response from Ollama using chat method
            try:
                response = self.llm_engine.chat(
                    user_input=prompt,
                    use_history=False  # Don't use conversation history for teacher
                )
                
                if isinstance(response, str):
                    return response
                elif isinstance(response, dict):
                    return response.get('response', response.get('text'))
                return str(response)
            except Exception as e:
                logger.warning(f"[TeacherComparison] LLM query failed: {e}")
                return None
        
        except Exception as e:
            logger.error(f"[TeacherComparison] Error getting teacher response: {e}")
            return None
    
    def compare_responses(
        self,
        user_input: str,
        alice_response: str,
        teacher_response: Optional[str] = None,
        intent: Optional[str] = None,
        domain: Optional[str] = None,
        llm_used: bool = False
    ) -> Dict[str, Any]:
        """
        Compare Alice's response to teacher's response
        
        Returns:
            Dict with quality metrics
        """
        if not teacher_response:
            teacher_response = self.get_teacher_response(user_input, intent, {'domain': domain})
        
        if not teacher_response:
            return {
                'comparison_possible': False,
                'quality_score': None
            }
        
        # Calculate similarity metrics
        similarity = self._calculate_similarity(alice_response, teacher_response)
        
        # Content quality checks
        quality_score = self._score_response_quality(
            alice_response,
            teacher_response,
            similarity,
            intent,
            domain
        )
        
        # Check if responses convey same intent
        semantic_match = self._check_semantic_match(alice_response, teacher_response)
        
        comparison_result = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'alice_response': alice_response,
            'teacher_response': teacher_response,
            'intent': intent,
            'domain': domain,
            'llm_used': llm_used,
            'similarity_score': similarity,
            'quality_score': quality_score,
            'semantic_match': semantic_match,
            'comparison_possible': True
        }
        
        # Log comparison
        self._log_comparison(comparison_result)
        
        # Update aggregated scores
        self._update_aggregated_scores(comparison_result)
        
        return comparison_result
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity (0.0-1.0)
        Using simple word overlap for speed
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _score_response_quality(
        self,
        alice_response: str,
        teacher_response: str,
        similarity: float,
        intent: Optional[str],
        domain: Optional[str]
    ) -> float:
        """
        Score quality of Alice's response compared to teacher
        
        Returns:
            Score 0.0-1.0 where 1.0 is perfect match to teacher
        """
        score = 0.0
        
        # Similarity score (40% weight)
        score += similarity * 0.4
        
        # Length reasonableness (10% weight)
        alice_len = len(alice_response.split())
        teacher_len = len(teacher_response.split())
        
        if teacher_len > 0:
            len_ratio = min(alice_len / teacher_len, 1.0) if teacher_len > 0 else 0.5
            if 0.5 <= len_ratio <= 2.0:
                score += 0.1
        
        # Intent-specific scoring (50% weight)
        # Check if response addresses the core of the request
        if self._addresses_intent(alice_response, teacher_response, intent):
            score += 0.5
        elif similarity > 0.5:
            score += 0.25
        
        return min(score, 1.0)
    
    def _check_semantic_match(self, text1: str, text2: str) -> bool:
        """
        Check if responses have semantic equivalence
        (not just string similarity but conveying same meaning)
        """
        # Simple heuristics
        
        # Both polite greetings
        greetings = ['hello', 'hi', 'hey', 'greetings', 'welcome']
        if all(word in text1.lower() for word in ['hello']) and \
           any(word in text2.lower() for word in greetings):
            return True
        
        # Both offer help
        if 'help' in text1.lower() and 'help' in text2.lower():
            return True
        
        # Both acknowledge confusion
        clarify_words = ['clarify', 'could you', 'could you clarify', 'can you clarify']
        if any(word in text1.lower() for word in clarify_words) and \
           any(word in text2.lower() for word in clarify_words):
            return True
        
        # Word overlap > 50%
        if self._calculate_similarity(text1, text2) > 0.5:
            return True
        
        return False
    
    def _addresses_intent(self, response: str, teacher_response: str, intent: Optional[str]) -> bool:
        """
        Check if both responses address the intent appropriately
        """
        if not intent:
            return False
        
        intent_lower = intent.lower()
        response_lower = response.lower()
        teacher_lower = teacher_response.lower()
        
        # Extract domain from intent
        if ':' in intent_lower:
            domain, action = intent_lower.split(':', 1)
        else:
            domain = intent_lower
            action = ''
        
        # Check if both responses acknowledge the domain
        if domain in response_lower and domain in teacher_lower:
            return True
        
        # For action-specific intents
        if action and action in response_lower and action in teacher_lower:
            return True
        
        return False
    
    def _log_comparison(self, comparison: Dict[str, Any]):
        """Log comparison to file"""
        try:
            with open(self.comparison_log, 'a') as f:
                f.write(json.dumps(comparison) + '\n')
        except Exception as e:
            logger.warning(f"[TeacherComparison] Error logging comparison: {e}")
    
    def _update_aggregated_scores(self, comparison: Dict[str, Any]):
        """Update aggregated quality scores"""
        self.scores['total_comparisons'] += 1
        
        quality = comparison.get('quality_score', 0.0)
        
        # Update average (simple running average)
        old_avg = self.scores.get('avg_similarity', 0.0)
        total = self.scores['total_comparisons']
        self.scores['avg_similarity'] = (old_avg * (total - 1) + quality) / total
        
        # Track by domain
        domain = comparison.get('domain', 'unknown')
        if domain not in self.scores['by_domain']:
            self.scores['by_domain'][domain] = {'count': 0, 'avg_quality': 0.0}
        
        old_avg_domain = self.scores['by_domain'][domain]['avg_quality']
        count_domain = self.scores['by_domain'][domain]['count'] + 1
        self.scores['by_domain'][domain]['avg_quality'] = (old_avg_domain * (count_domain - 1) + quality) / count_domain
        self.scores['by_domain'][domain]['count'] = count_domain
        
        # Track by intent
        intent = comparison.get('intent', 'unknown')
        if intent not in self.scores['by_intent']:
            self.scores['by_intent'][intent] = {'count': 0, 'avg_quality': 0.0}
        
        old_avg_intent = self.scores['by_intent'][intent]['avg_quality']
        count_intent = self.scores['by_intent'][intent]['count'] + 1
        self.scores['by_intent'][intent]['avg_quality'] = (old_avg_intent * (count_intent - 1) + quality) / count_intent
        self.scores['by_intent'][intent]['count'] = count_intent
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get current quality report"""
        return self.scores.copy()
    
    def identify_high_quality_patterns(self, min_quality: float = 0.8, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Find patterns that Alice handles well consistently
        These can be auto-promoted to deterministic patterns
        
        Returns:
            List of high-quality patterns
        """
        if not self.comparison_log.exists():
            return []
        
        pattern_quality = defaultdict(lambda: {'count': 0, 'total_quality': 0.0, 'examples': []})
        
        try:
            with open(self.comparison_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        comparison = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    intent = comparison.get('intent', 'unknown')
                    quality = comparison.get('quality_score', 0.0)
                    
                    pattern_quality[intent]['count'] += 1
                    pattern_quality[intent]['total_quality'] += quality
                    pattern_quality[intent]['examples'].append({
                        'user_input': comparison.get('user_input', ''),
                        'quality': quality
                    })
        
        except Exception as e:
            logger.warning(f"[TeacherComparison] Error analyzing patterns: {e}")
            return []
        
        # Filter high-quality patterns
        high_quality_patterns = []
        for intent, data in pattern_quality.items():
            if data['count'] >= min_occurrences:
                avg_quality = data['total_quality'] / data['count']
                
                if avg_quality >= min_quality:
                    high_quality_patterns.append({
                        'intent': intent,
                        'avg_quality': avg_quality,
                        'occurrence_count': data['count'],
                        'sample_inputs': [ex['user_input'] for ex in data['examples'][:3]]
                    })
        
        # Sort by quality
        high_quality_patterns.sort(key=lambda x: x['avg_quality'], reverse=True)
        
        return high_quality_patterns
    
    def identify_problem_patterns(self, max_quality: float = 0.5, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Find patterns where Alice consistently underperforms
        Flag for manual intervention or rule updates
        
        Returns:
            List of problem patterns
        """
        if not self.comparison_log.exists():
            return []
        
        pattern_quality = defaultdict(lambda: {'count': 0, 'total_quality': 0.0, 'examples': []})
        
        try:
            with open(self.comparison_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        comparison = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    intent = comparison.get('intent', 'unknown')
                    quality = comparison.get('quality_score', 0.0)
                    
                    pattern_quality[intent]['count'] += 1
                    pattern_quality[intent]['total_quality'] += quality
                    pattern_quality[intent]['examples'].append({
                        'user_input': comparison.get('user_input', ''),
                        'quality': quality,
                        'alice_response': comparison.get('alice_response', ''),
                        'teacher_response': comparison.get('teacher_response', '')
                    })
        
        except Exception as e:
            logger.warning(f"[TeacherComparison] Error analyzing patterns: {e}")
            return []
        
        # Filter problem patterns
        problem_patterns = []
        for intent, data in pattern_quality.items():
            if data['count'] >= min_occurrences:
                avg_quality = data['total_quality'] / data['count']
                
                if avg_quality <= max_quality:
                    problem_patterns.append({
                        'intent': intent,
                        'avg_quality': avg_quality,
                        'occurrence_count': data['count'],
                        'severity': 'high' if avg_quality < 0.3 else 'medium',
                        'sample_examples': [
                            {
                                'user': ex['user_input'],
                                'alice': ex['alice_response'],
                                'teacher': ex['teacher_response']
                            }
                            for ex in data['examples'][:2]
                        ]
                    })
        
        # Sort by quality (worst first)
        problem_patterns.sort(key=lambda x: x['avg_quality'])
        
        return problem_patterns
    
    def run_comparison_cycle(self, training_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run full comparison cycle for a batch of training items
        
        Args:
            training_items: List of scenario/training items to compare
        
        Returns:
            Summary of comparisons
        """
        logger.info(f"[TeacherComparison] Starting comparison cycle for {len(training_items)} items...")
        
        comparisons_done = 0
        comparisons_failed = 0
        
        for item in training_items:
            try:
                comparison = self.compare_responses(
                    user_input=item.get('user_input', ''),
                    alice_response=item.get('alice_response', ''),
                    teacher_response=None,  # Get from teacher
                    intent=item.get('actual_intent', item.get('intent')),
                    domain=item.get('domain'),
                    llm_used=item.get('llm_used', False)
                )
                
                if comparison.get('comparison_possible'):
                    comparisons_done += 1
                else:
                    comparisons_failed += 1
            
            except Exception as e:
                logger.warning(f"[TeacherComparison] Error comparing item: {e}")
                comparisons_failed += 1
        
        # Save scores
        self._save_scores()
        
        # Identify high/low quality patterns
        high_quality = self.identify_high_quality_patterns()
        problem_patterns = self.identify_problem_patterns()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'comparisons_done': comparisons_done,
            'comparisons_failed': comparisons_failed,
            'avg_quality': self.scores['avg_similarity'],
            'high_quality_patterns': len(high_quality),
            'problem_patterns': len(problem_patterns),
            'sample_high_quality': high_quality[:3],
            'sample_problems': problem_patterns[:3]
        }
        
        logger.info(f"[TeacherComparison] Cycle complete: {comparisons_done} comparisons, "
                   f"avg quality {self.scores['avg_similarity']:.2f}")
        
        return summary


def create_teacher_comparison(project_root: Optional[Path] = None) -> OllamaTeacherComparison:
    """Factory function"""
    return OllamaTeacherComparison(project_root)
