"""
Ollama Feedback Injector - Pipes audit results into training data
Converts audit feedback into structured training examples
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from ai.ollama_scorer import TrainingSignal

logger = logging.getLogger(__name__)


class FeedbackInjector:
    """Injects audit feedback into training pipeline"""
    
    def __init__(self, training_data_path: str = "data/training/"):
        self.training_data_path = Path(training_data_path)
        self.training_data_path.mkdir(parents=True, exist_ok=True)
        self.feedback_log = self.training_data_path / "audit_feedback.jsonl"
    
    def inject_signals(self, signals: List[TrainingSignal]) -> int:
        """
        Inject training signals into training data
        
        Args:
            signals: List of TrainingSignal objects
        
        Returns:
            Number of signals injected
        """
        count = 0
        
        for signal in signals:
            # Convert signal to training example
            example = self._signal_to_training_example(signal)
            
            # Append to feedback log
            with open(self.feedback_log, 'a') as f:
                f.write(json.dumps(example) + '\n')
            
            count += 1
            logger.info(f"Injected signal for {signal.domain}:{signal.skill}")
        
        return count
    
    def _signal_to_training_example(self, signal: TrainingSignal) -> Dict[str, Any]:
        """Convert TrainingSignal to training example"""
        return {
            'timestamp': datetime.now().isoformat(),
            'type': 'audit_feedback',
            'domain': signal.domain,
            'skill': signal.skill,
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'focus_dimension': signal.focus_dimension,
            'feedback': signal.feedback,
            'priority': self._calculate_priority(signal)
        }
    
    def _calculate_priority(self, signal: TrainingSignal) -> float:
        """Calculate training priority (0.0-1.0)"""
        base_priority = {
            'negative': 1.0,
            'improvement': 0.6,
            'positive': 0.2
        }.get(signal.signal_type, 0.5)
        
        return base_priority * signal.strength
    
    def aggregate_feedback_by_domain(self) -> Dict[str, Any]:
        """
        Aggregate all feedback by domain
        
        Returns:
            {domain: {stats}}
        """
        aggregated = {}
        
        if not self.feedback_log.exists():
            return aggregated
        
        # Read all feedback
        with open(self.feedback_log, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            try:
                example = json.loads(line)
                domain = example.get('domain')
                
                if domain not in aggregated:
                    aggregated[domain] = {
                        'total_feedback': 0,
                        'avg_priority': 0.0,
                        'signal_types': {},
                        'skills': {}
                    }
                
                stats = aggregated[domain]
                stats['total_feedback'] += 1
                
                # Track signal types
                signal_type = example.get('signal_type')
                if signal_type not in stats['signal_types']:
                    stats['signal_types'][signal_type] = 0
                stats['signal_types'][signal_type] += 1
                
                # Track skills
                skill = example.get('skill')
                if skill not in stats['skills']:
                    stats['skills'][skill] = []
                stats['skills'][skill].append(example)
            
            except json.JSONDecodeError:
                continue
        
        # Calculate averages
        for domain in aggregated:
            examples = []
            for skill_list in aggregated[domain]['skills'].values():
                examples.extend(skill_list)
            
            if examples:
                avg_priority = sum(e.get('priority', 0) for e in examples) / len(examples)
                aggregated[domain]['avg_priority'] = avg_priority
        
        return aggregated
    
    def create_domain_training_dataset(self, domain: str) -> Dict[str, Any]:
        """
        Create training dataset from feedback for specific domain
        
        Args:
            domain: Domain name
        
        Returns:
            {examples: [...], metadata: {...}}
        """
        if not self.feedback_log.exists():
            return {'examples': [], 'metadata': {'domain': domain, 'count': 0}}
        
        examples = []
        
        # Read feedback log
        with open(self.feedback_log, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    if example.get('domain') == domain:
                        examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        # Sort by priority (highest first)
        examples.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return {
            'domain': domain,
            'examples': examples,
            'metadata': {
                'domain': domain,
                'count': len(examples),
                'created': datetime.now().isoformat(),
                'priority_weighted': sum(e.get('priority', 0) for e in examples) if examples else 0
            }
        }
    
    def save_domain_dataset(self, domain: str, output_path: str = None) -> str:
        """
        Save domain training dataset to file
        
        Args:
            domain: Domain name
            output_path: Where to save (default: data/training/{domain}_feedback.json)
        
        Returns:
            Path to saved file
        """
        if not output_path:
            output_path = str(self.training_data_path / f"{domain}_feedback.json")
        
        dataset = self.create_domain_training_dataset(domain)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved {len(dataset['examples'])} feedback examples for {domain}")
        
        return output_path
    
    def clear_feedback(self):
        """Clear feedback log"""
        if self.feedback_log.exists():
            self.feedback_log.unlink()
        logger.info("Cleared feedback log")


def create_injector(training_data_path: str = "data/training/") -> FeedbackInjector:
    """Factory to create feedback injector"""
    return FeedbackInjector(training_data_path)
