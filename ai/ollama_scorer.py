"""
Ollama Scorer - Aggregates audit results into training signals
Converts dimension scores into actionable training feedback
"""

import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from ai.ollama_auditor import AuditScore

logger = logging.getLogger(__name__)


@dataclass
class TrainingSignal:
    """Signal to drive learning in a specific domain/skill"""
    domain: str
    skill: str
    signal_type: str  # "positive", "negative", "improvement"
    strength: float  # 0.0-1.0
    focus_dimension: str  # Which dimension needs work
    feedback: str  # Plain text explanation


class OllamaScorer:
    """Converts audit scores into training signals"""
    
    def __init__(self):
        self.signals: List[TrainingSignal] = []
    
    def score_audit(
        self,
        audit: AuditScore,
        domain: str,
        skill: str = "general"
    ) -> List[TrainingSignal]:
        """
        Convert audit score into training signals
        
        Args:
            audit: AuditScore to convert
            domain: Domain name
            skill: Skill within domain
        
        Returns:
            List of TrainingSignal objects
        """
        signals = []
        
        # Analyze overall score
        overall = audit.overall_score
        
        if overall >= 4.5:
            # High performance - reinforce
            signals.append(TrainingSignal(
                domain=domain,
                skill=skill,
                signal_type="positive",
                strength=overall / 5.0,
                focus_dimension="all",
                feedback=f"Excellent response. Maintain this quality."
            ))
        
        elif overall >= 3.5:
            # Good but room for improvement
            worst_dimension = self._find_worst_dimension(audit)
            signals.append(TrainingSignal(
                domain=domain,
                skill=skill,
                signal_type="improvement",
                strength=(overall - 3.0) / 2.0,
                focus_dimension=worst_dimension,
                feedback=f"Good response overall. Improve {worst_dimension} specifically."
            ))
        
        else:
            # Poor response - needs work
            worst_dimension = self._find_worst_dimension(audit)
            signals.append(TrainingSignal(
                domain=domain,
                skill=skill,
                signal_type="negative",
                strength=1.0 - (overall / 5.0),
                focus_dimension=worst_dimension,
                feedback=f"Response needs improvement. Focus on {worst_dimension}."
            ))
        
        # Store signals
        self.signals.extend(signals)
        
        return signals
    
    def _find_worst_dimension(self, audit: AuditScore) -> str:
        """Find the dimension with lowest score"""
        if not audit.scores:
            return "general"
        
        worst = min(audit.scores.items(), key=lambda x: x[1])
        return worst[0].value
    
    def aggregate_signals_by_domain(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate all signals by domain
        
        Returns:
            {domain: {skill: {stats}}}
        """
        aggregated = {}
        
        for signal in self.signals:
            domain = signal.domain
            skill = signal.skill
            
            if domain not in aggregated:
                aggregated[domain] = {}
            if skill not in aggregated[domain]:
                aggregated[domain][skill] = {
                    'positive': 0,
                    'negative': 0,
                    'improvement': 0,
                    'avg_strength': 0.0,
                    'focus_dimensions': {},
                    'signals': []
                }
            
            # Update stats
            stats = aggregated[domain][skill]
            stats[signal.signal_type] += 1
            stats['signals'].append(signal)
            
            # Track focus dimensions
            if signal.focus_dimension != "all":
                if signal.focus_dimension not in stats['focus_dimensions']:
                    stats['focus_dimensions'][signal.focus_dimension] = 0
                stats['focus_dimensions'][signal.focus_dimension] += 1
        
        # Calculate averages
        for domain in aggregated:
            for skill in aggregated[domain]:
                stats = aggregated[domain][skill]
                signals = stats['signals']
                if signals:
                    avg_strength = sum(s.strength for s in signals) / len(signals)
                    stats['avg_strength'] = avg_strength
        
        return aggregated
    
    def get_training_priority(self) -> List[Dict[str, Any]]:
        """
        Get prioritized list of what to train on
        
        Returns:
            List of {domain, skill, priority_score, reason}
        """
        aggregated = self.aggregate_signals_by_domain()
        priorities = []
        
        for domain, skills in aggregated.items():
            for skill, stats in skills.items():
                # Priority = negative signals + improvement needs
                priority = (stats['negative'] * 2 + stats['improvement']) / max(1, len(stats['signals']))
                
                priorities.append({
                    'domain': domain,
                    'skill': skill,
                    'priority_score': priority,
                    'negative_count': stats['negative'],
                    'improvement_count': stats['improvement'],
                    'avg_strength': stats['avg_strength'],
                    'focus_dimensions': stats['focus_dimensions'],
                    'reason': f"Domain {domain}/{skill} has {stats['negative']} low scores"
                })
        
        # Sort by priority
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priorities
    
    def to_training_batch(self, domain: str, skill: str) -> Dict[str, Any]:
        """
        Convert signals for domain/skill into training batch format
        
        Returns:
            {queries: [...], responses: [...], scores: [...]}
        """
        relevant_signals = [
            s for s in self.signals
            if s.domain == domain and s.skill == skill
        ]
        
        batch = {
            'domain': domain,
            'skill': skill,
            'size': len(relevant_signals),
            'signals': [self._signal_to_dict(s) for s in relevant_signals],
            'avg_strength': sum(s.strength for s in relevant_signals) / len(relevant_signals) if relevant_signals else 0
        }
        
        return batch
    
    def _signal_to_dict(self, signal: TrainingSignal) -> Dict[str, Any]:
        """Convert signal to dict"""
        return {
            'domain': signal.domain,
            'skill': signal.skill,
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'focus_dimension': signal.focus_dimension,
            'feedback': signal.feedback
        }
    
    def clear_signals(self):
        """Clear all signals"""
        self.signals.clear()


def create_scorer() -> OllamaScorer:
    """Factory to create scorer"""
    return OllamaScorer()
