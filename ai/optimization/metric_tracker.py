"""
Metric Tracker - Logs per-domain improvement metrics
Tracks before/after training scores and progress
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricTracker:
    """Tracks training metrics per domain"""
    
    def __init__(self, metrics_path: str = "data/training/metrics/"):
        self.metrics_path = Path(metrics_path)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_path / "domain_metrics.jsonl"
        self.current_session = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'pre_training',
            'metrics': {}
        }
    
    def record_pre_training_score(
        self,
        domain: str,
        overall_score: float,
        dimension_scores: Dict[str, float]
    ):
        """
        Record scores before training
        
        Args:
            domain: Domain name
            overall_score: Overall response quality (0-5)
            dimension_scores: {dimension: score}
        """
        if 'pre_training' not in self.current_session['metrics']:
            self.current_session['metrics']['pre_training'] = {}
        
        self.current_session['metrics']['pre_training'][domain] = {
            'overall': overall_score,
            'dimensions': dimension_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"PRE-TRAINING {domain}: {overall_score:.2f}/5.0")
    
    def record_post_training_score(
        self,
        domain: str,
        overall_score: float,
        dimension_scores: Dict[str, float]
    ):
        """
        Record scores after training
        
        Args:
            domain: Domain name
            overall_score: Overall response quality (0-5)
            dimension_scores: {dimension: score}
        """
        if 'post_training' not in self.current_session['metrics']:
            self.current_session['metrics']['post_training'] = {}
        
        self.current_session['metrics']['post_training'][domain] = {
            'overall': overall_score,
            'dimensions': dimension_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate improvement
        pre = self.current_session['metrics'].get('pre_training', {}).get(domain, {})
        improvement = overall_score - pre.get('overall', 0)
        
        logger.info(
            f"POST-TRAINING {domain}: {overall_score:.2f}/5.0 "
            f"(+{improvement:+.2f})"
        )
    
    def get_improvement(self, domain: str) -> Dict[str, float]:
        """
        Get improvement metrics for domain
        
        Returns:
            {overall: delta, dimensions: {dim: delta}}
        """
        pre = self.current_session['metrics'].get('pre_training', {}).get(domain)
        post = self.current_session['metrics'].get('post_training', {}).get(domain)
        
        if not pre or not post:
            return {'overall': 0.0, 'dimensions': {}}
        
        overall_improvement = post['overall'] - pre['overall']
        
        dimension_improvement = {}
        for dim in pre['dimensions']:
            dim_improvement = post['dimensions'].get(dim, 0) - pre['dimensions'][dim]
            dimension_improvement[dim] = dim_improvement
        
        return {
            'overall': overall_improvement,
            'dimensions': dimension_improvement,
            'pre_score': pre['overall'],
            'post_score': post['overall']
        }
    
    def get_all_improvements(self) -> Dict[str, Dict[str, float]]:
        """Get improvement for all domains"""
        improvements = {}
        
        pre_scores = self.current_session['metrics'].get('pre_training', {})
        for domain in pre_scores:
            improvements[domain] = self.get_improvement(domain)
        
        return improvements
    
    def finalize_session(self) -> Dict[str, Any]:
        """
        Finalize current training session
        
        Returns:
            Summary of session results
        """
        improvements = self.get_all_improvements()
        
        # Calculate overall improvement
        improvements_list = [
            imp['overall'] for imp in improvements.values()
            if 'overall' in imp
        ]
        
        overall_improvement = (
            sum(improvements_list) / len(improvements_list)
            if improvements_list else 0
        )
        
        summary = {
            'timestamp': self.current_session['timestamp'],
            'domains_trained': len(improvements),
            'overall_improvement': overall_improvement,
            'domain_improvements': improvements,
            'improvements_by_dimension': self._aggregate_dimension_improvements(improvements)
        }
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        
        logger.info(f"Session complete. Overall improvement: +{overall_improvement:.2f}")
        
        # Reset for next session
        self.current_session = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'pre_training',
            'metrics': {}
        }
        
        return summary
    
    def _aggregate_dimension_improvements(
        self,
        improvements: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Aggregate improvements by dimension across domains"""
        dimension_improvements = {}
        
        for domain_imp in improvements.values():
            dims = domain_imp.get('dimensions', {})
            for dim, improvement in dims.items():
                if dim not in dimension_improvements:
                    dimension_improvements[dim] = []
                dimension_improvements[dim].append(improvement)
        
        # Calculate averages
        return {
            dim: sum(imps) / len(imps)
            for dim, imps in dimension_improvements.items()
        }
    
    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical session results
        
        Args:
            limit: Max sessions to return
        
        Returns:
            List of session summaries
        """
        if not self.metrics_file.exists():
            return []
        
        sessions = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    session = json.loads(line)
                    sessions.append(session)
                except json.JSONDecodeError:
                    continue
        
        return sessions[-limit:]
    
    def get_trend(self, domain: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get improvement trend for domain
        
        Args:
            domain: Domain name
            limit: How many sessions to analyze
        
        Returns:
            {scores: [...], trend: "up"|"down"|"stable"}
        """
        history = self.get_session_history(limit)
        
        scores = []
        for session in history:
            domain_imp = session.get('domain_improvements', {}).get(domain)
            if domain_imp:
                scores.append({
                    'timestamp': session.get('timestamp'),
                    'improvement': domain_imp.get('overall', 0),
                    'post_score': domain_imp.get('post_score')
                })
        
        if not scores:
            return {'scores': [], 'trend': 'unknown'}
        
        # Calculate trend
        improvements = [s['improvement'] for s in scores]
        if len(improvements) >= 2:
            recent_avg = sum(improvements[-3:]) / min(3, len(improvements))
            trend = 'up' if recent_avg > 0.1 else 'down' if recent_avg < -0.1 else 'stable'
        else:
            trend = 'unknown'
        
        return {
            'domain': domain,
            'scores': scores,
            'trend': trend,
            'avg_improvement': sum(improvements) / len(improvements)
        }
    
    def print_summary(self):
        """Print current session summary"""
        improvements = self.get_all_improvements()
        
        print("\n" + "="*60)
        print("TRAINING SESSION SUMMARY")
        print("="*60)
        
        for domain, imp in improvements.items():
            print(f"\n{domain.upper()}")
            print(f"  Overall: {imp['pre_score']:.1f} → {imp['post_score']:.1f} "
                  f"({imp['overall']:+.2f})")
            
            for dim, dim_imp in imp['dimensions'].items():
                print(f"  {dim}: {dim_imp:+.2f}")
        
        all_improvements = [imp['overall'] for imp in improvements.values()]
        avg = sum(all_improvements) / len(all_improvements) if all_improvements else 0
        print(f"\nOVERALL IMPROVEMENT: {avg:+.2f}")
        print("="*60 + "\n")


def create_tracker(metrics_path: str = "data/training/metrics/") -> MetricTracker:
    """Factory to create metric tracker"""
    return MetricTracker(metrics_path)
