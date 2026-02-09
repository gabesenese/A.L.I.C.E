"""
Monitor Audit Progress
Real-time view of training metrics and audit trends
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def monitor_metrics():
    """Show current metrics"""
    
    print_header("AUDIT METRICS")
    
    metrics_file = Path("data/training/metrics/domain_metrics.jsonl")
    
    if not metrics_file.exists():
        print("No metrics recorded yet. Run test or wait for nightly cycle.\n")
        return
    
    # Read latest sessions
    sessions = []
    with open(metrics_file) as f:
        for line in f:
            try:
                sessions.append(json.loads(line))
            except:
                pass
    
    if not sessions:
        print("No metrics available.\n")
        return
    
    # Show last 3 sessions
    for session in sessions[-3:]:
        print(f"\nSession: {session.get('timestamp', 'Unknown')}")
        print(f"Overall improvement: {session.get('overall_improvement', 0):+.2f}")
        
        improvements = session.get('domain_improvements', {})
        for domain, imp in improvements.items():
            print(f"  {domain}: {imp.get('overall', 0):+.2f} "
                  f"({imp.get('pre_score', 0):.1f} → {imp.get('post_score', 0):.1f})")


def monitor_feedback():
    """Show latest feedback"""
    
    print_header("AUDIT FEEDBACK")
    
    feedback_file = Path("data/training/audit_feedback.jsonl")
    
    if not feedback_file.exists():
        print("No feedback recorded yet.\n")
        return
    
    # Read feedback
    feedback = []
    with open(feedback_file) as f:
        for line in f:
            try:
                feedback.append(json.loads(line))
            except:
                pass
    
    if not feedback:
        print("No feedback available.\n")
        return
    
    # Group by domain
    by_domain = {}
    for item in feedback[-20:]:  # Last 20
        domain = item.get('domain')
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(item)
    
    for domain, items in by_domain.items():
        print(f"\n{domain.upper()}")
        positive = sum(1 for i in items if i.get('signal_type') == 'positive')
        negative = sum(1 for i in items if i.get('signal_type') == 'negative')
        improvement = sum(1 for i in items if i.get('signal_type') == 'improvement')
        
        print(f"  Positive signals: {positive}")
        print(f"  Improvement signals: {improvement}")
        print(f"  Negative signals: {negative}")
        
        if items:
            avg_strength = sum(i.get('strength', 0) for i in items) / len(items)
            print(f"  Avg strength: {avg_strength:.2f}")


def monitor_datasets():
    """Show domain training datasets"""
    
    print_header("DOMAIN TRAINING DATASETS")
    
    data_dir = Path("data/training")
    
    datasets = list(data_dir.glob("*_feedback.json"))
    
    if not datasets:
        print("No training datasets created yet.\n")
        return
    
    for dataset_file in sorted(datasets):
        try:
            with open(dataset_file) as f:
                data = json.load(f)
            
            domain = data.get('domain')
            count = data.get('metadata', {}).get('count', 0)
            weighted = data.get('metadata', {}).get('priority_weighted', 0)
            
            print(f"\n{domain.upper()}")
            print(f"  Examples: {count}")
            print(f"  Priority-weighted: {weighted:.2f}")
        
        except:
            pass


def main():
    """Main monitoring loop"""
    
    print("\n" + "="*70)
    print("AUDIT PROGRESS MONITOR")
    print("="*70)
    
    while True:
        monitor_metrics()
        monitor_feedback()
        monitor_datasets()
        
        print("\n" + "="*70)
        print("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Refresh in 60 seconds (Ctrl+C to exit)")
        print("="*70 + "\n")
        
        try:
            import time
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            return True


if __name__ == "__main__":
    main()
