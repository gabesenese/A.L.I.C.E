"""
Audit Config Optimizer - Auto-tunes parameters based on results
Tweaks thresholds and weights to maximize improvement
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigOptimizer:
    """Automatically adjusts audit parameters based on results"""
    
    def __init__(self, config_path: str = "data/training/audit_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'query_generation': {
                'queries_per_skill': 3,
                'template_diversity': 0.7,
                'min_quality': 0.5
            },
            'auditing': {
                'focus_on_low_dimensions': True,
                'dimension_weights': {
                    'accuracy': 1.0,
                    'clarity': 0.8,
                    'completeness': 0.9,
                    'relevance': 0.85,
                    'tone': 0.6,
                    'reasoning': 0.9
                }
            },
            'scoring': {
                'positive_threshold': 4.5,
                'improvement_threshold': 3.5,
                'negative_threshold': 3.0,
                'signal_strength_multiplier': 1.0
            },
            'training': {
                'batch_size': 5,
                'priority_weight': 1.0,
                'focus_weight': 0.8
            },
            'automation': {
                'enabled': False,
                'schedule': 'daily',
                'hour': 2,
                'minute': 0
            }
        }
        
        # Load existing config if available
        self._load_config()
    
    def _load_config(self):
        """Load config from file if it exists"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
                    logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        """Save config to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {self.config_path}")
    
    def analyze_results(self, metric_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze training results and suggest optimizations
        
        Args:
            metric_results: Results from MetricTracker.finalize_session()
        
        Returns:
            Suggestions and recommended config adjustments
        """
        suggestions = {
            'timestamp': None,
            'overall_improvement': metric_results.get('overall_improvement', 0),
            'adjustments': {},
            'explanation': []
        }
        
        import datetime
        suggestions['timestamp'] = datetime.datetime.now().isoformat()
        
        improvements = metric_results.get('domain_improvements', {})
        dimension_improvements = metric_results.get('improvements_by_dimension', {})
        
        # Analyze domain improvements
        for domain, imp in improvements.items():
            improvement = imp.get('overall', 0)
            
            if improvement > 0.5:
                # Good improvement - may need harder queries
                suggestions['adjustments'][f'{domain}_query_difficulty'] = {
                    'current': self.config['query_generation']['queries_per_skill'],
                    'suggested': self.config['query_generation']['queries_per_skill'] + 1,
                    'reason': f'{domain} improved by {improvement:.2f}, ready for harder cases'
                }
                suggestions['explanation'].append(
                    f"{domain} improved well (+{improvement:.2f}). Increase query complexity."
                )
            
            elif improvement < 0.0:
                # No improvement - may need easier queries or better training
                suggestions['adjustments'][f'{domain}_strategy'] = {
                    'current': 'standard',
                    'suggested': 'remedial',
                    'reason': f'{domain} did not improve, needs different approach'
                }
                suggestions['explanation'].append(
                    f"{domain} did not improve. Use remedial training."
                )
            
            else:
                suggestions['explanation'].append(
                    f"~ {domain} showed minimal improvement (+{improvement:.2f})"
                )
        
        # Analyze dimension improvements
        for dimension, improvement in dimension_improvements.items():
            if improvement < 0.1:
                # This dimension is struggling - increase its weight
                current_weight = self.config['auditing']['dimension_weights'].get(dimension, 0.7)
                
                suggestions['adjustments'][f'{dimension}_weight'] = {
                    'current': current_weight,
                    'suggested': min(1.0, current_weight + 0.1),
                    'reason': f'{dimension} not improving, needs more focus'
                }
                suggestions['explanation'].append(
                    f"! {dimension} stagnant. Increase weight from {current_weight:.1f} to {min(1.0, current_weight + 0.1):.1f}"
                )
        
        # Overall assessment
        if suggestions['overall_improvement'] > 0.3:
            suggestions['overall_assessment'] = 'strong'
            suggestions['explanation'].append("\n💪 Strong overall improvement. Continue current strategy.")
        elif suggestions['overall_improvement'] > 0.0:
            suggestions['overall_assessment'] = 'positive'
            suggestions['explanation'].append("\nPositive improvement. Maintain current strategy.")
        elif suggestions['overall_improvement'] == 0:
            suggestions['overall_assessment'] = 'stagnant'
            suggestions['explanation'].append("\n⚠No improvement. Consider adjusting training approach.")
        else:
            suggestions['overall_assessment'] = 'negative'
            suggestions['explanation'].append("\n⛔ Performance declined. Review training data quality.")
        
        return suggestions
    
    def apply_suggestions(self, suggestions: Dict[str, Any]) -> bool:
        """
        Apply suggested adjustments to config
        
        Args:
            suggestions: Suggestions from analyze_results()
        
        Returns:
            True if changes applied
        """
        adjustments = suggestions.get('adjustments', {})
        if not adjustments:
            logger.info("No adjustments to apply")
            return False
        
        logger.info(f"Applying {len(adjustments)} adjustments...")
        
        changes_made = 0
        for key, adjustment in adjustments.items():
            current = adjustment.get('current')
            suggested = adjustment.get('suggested')
            reason = adjustment.get('reason')
            
            # Map key to config path and apply
            if 'query_difficulty' in key:
                self.config['query_generation']['queries_per_skill'] = suggested
                changes_made += 1
                logger.info(f"Updated queries_per_skill: {current} → {suggested}")
            
            elif 'weight' in key:
                dimension = key.replace('_weight', '')
                self.config['auditing']['dimension_weights'][dimension] = suggested
                changes_made += 1
                logger.info(f"Updated {dimension} weight: {current} → {suggested}")
        
        if changes_made > 0:
            self.save_config()
            logger.info(f"Applied {changes_made} adjustments and saved config")
            return True
        
        return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def enable_automation(self, hour: int = 2, minute: int = 0):
        """Enable and configure automation"""
        self.config['automation']['enabled'] = True
        self.config['automation']['hour'] = hour
        self.config['automation']['minute'] = minute
        self.save_config()
        logger.info(f"Automation enabled for {hour:02d}:{minute:02d} daily")
    
    def disable_automation(self):
        """Disable automation"""
        self.config['automation']['enabled'] = False
        self.save_config()
        logger.info("Automation disabled")
    
    def is_automation_enabled(self) -> bool:
        """Check if automation is enabled"""
        return self.config['automation']['enabled']
    
    def print_config(self):
        """Print current config in readable format"""
        print("\n" + "="*60)
        print("AUDIT CONFIGURATION")
        print("="*60)
        
        print("\nQuery Generation:")
        qg = self.config['query_generation']
        print(f"  Queries per skill: {qg['queries_per_skill']}")
        print(f"  Template diversity: {qg['template_diversity']}")
        print(f"  Min quality: {qg['min_quality']}")
        
        print("\nAuditing:")
        audit = self.config['auditing']
        print(f"  Focus on low dimensions: {audit['focus_on_low_dimensions']}")
        print("  Dimension weights:")
        for dim, weight in audit['dimension_weights'].items():
            print(f"    {dim}: {weight}")
        
        print("\nScoring Thresholds:")
        sc = self.config['scoring']
        print(f"  Positive: {sc['positive_threshold']}")
        print(f"  Improvement: {sc['improvement_threshold']}")
        print(f"  Negative: {sc['negative_threshold']}")
        
        print("\nAutomation:")
        auto = self.config['automation']
        status = "ENABLED" if auto['enabled'] else "DISABLED"
        print(f"  Status: {status}")
        if auto['enabled']:
            print(f"  Schedule: {auto['hour']:02d}:{auto['minute']:02d} daily")
        
        print("="*60 + "\n")


def create_optimizer(config_path: str = "data/training/audit_config.json") -> ConfigOptimizer:
    """Factory to create config optimizer"""
    return ConfigOptimizer(config_path)
