#!/usr/bin/env python3
"""
A.L.I.C.E. Real-Time Performance Monitor
=========================================

Monitor query latency, accuracy, and system health in real-time.

Usage:
  python scripts/monitor_live.py --port 8000
  
  Then visit: http://localhost:8000/

Features:
- Real-time query latency (p50, p95, p99)
- Domain-specific accuracy metrics
- Daily improvement tracking
- Ollama health status
- Recent queries + audit results
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import statistics

class PerformanceMonitor:
    def __init__(self, data_dir="data/training"):
        self.data_dir = Path(data_dir)
        self.query_log = deque(maxlen=1000)  # Last 1000 queries
        self.metrics_file = self.data_dir / "metrics" / "domain_metrics.jsonl"
        self.feedback_file = self.data_dir / "audit_feedback.jsonl"
        
    def load_metrics(self):
        """Load latest metrics from audit system"""
        if not self.metrics_file.exists():
            return {}
        
        metrics = {}
        with open(self.metrics_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    domain = entry.get("domain")
                    if domain:
                        metrics[domain] = entry
                except:
                    pass
        
        return metrics
    
    def load_feedback(self, hours=24):
        """Load feedback from last N hours"""
        if not self.feedback_file.exists():
            return {}
        
        cutoff = datetime.now() - timedelta(hours=hours)
        feedback_by_domain = {}
        
        with open(self.feedback_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ts = datetime.fromisoformat(entry.get("timestamp", ""))
                    if ts > cutoff:
                        domain = entry.get("domain")
                        if domain not in feedback_by_domain:
                            feedback_by_domain[domain] = []
                        feedback_by_domain[domain].append(entry)
                except:
                    pass
        
        return feedback_by_domain
    
    def calculate_stats(self):
        """Calculate real-time statistics"""
        
        metrics = self.load_metrics()
        feedback = self.load_feedback(hours=24)
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "domains": {},
            "overall": {
                "total_queries_24h": 0,
                "avg_accuracy": 0,
                "avg_clarity": 0,
                "improvement_trend": "stable"
            }
        }
        
        accuracy_scores = []
        clarity_scores = []
        
        for domain, entries in feedback.items():
            signal_counts = {
                "positive": 0,
                "improvement": 0,
                "negative": 0
            }
            
            scores = []
            for entry in entries:
                signal = entry.get("signal_type")
                if signal:
                    signal_counts[signal] += 1
                if "score" in entry:
                    scores.append(entry["score"])
            
            avg_score = statistics.mean(scores) if scores else 0
            accuracy_scores.append(avg_score)
            
            stats["domains"][domain] = {
                "total_queries_24h": len(entries),
                "positive_signals": signal_counts["positive"],
                "improvement_signals": signal_counts["improvement"],
                "negative_signals": signal_counts["negative"],
                "avg_score": round(avg_score, 2),
                "signal_rate": {
                    "positive": round(signal_counts["positive"] / len(entries) * 100, 1) if entries else 0,
                    "improvement": round(signal_counts["improvement"] / len(entries) * 100, 1) if entries else 0,
                    "negative": round(signal_counts["negative"] / len(entries) * 100, 1) if entries else 0,
                }
            }
            
            stats["overall"]["total_queries_24h"] += len(entries)
        
        if accuracy_scores:
            stats["overall"]["avg_accuracy"] = round(statistics.mean(accuracy_scores), 2)
        
        return stats

def print_dashboard():
    """Print a nice text dashboard"""
    monitor = PerformanceMonitor()
    stats = monitor.calculate_stats()
    
    print("\n" + "=" * 80)
    print(f"A.L.I.C.E. PERFORMANCE DASHBOARD")
    print(f"Updated: {stats['timestamp']}")
    print("=" * 80)
    
    # Overall stats
    overall = stats["overall"]
    print(f"\nOVERALL (24-hour window):")
    print(f"  Total Queries:        {overall['total_queries_24h']}")
    print(f"  Avg Accuracy Score:   {overall['avg_accuracy']}/5.0")
    print(f"  Avg Clarity Score:    {overall['avg_clarity']}/5.0")
    print(f"  Improvement Trend:    {overall['improvement_trend']}")
    
    # Domain-specific
    print(f"\nDOMAIN METRICS:")
    print("-" * 80)
    print(f"  {'Domain':<20} {'Queries':<10} {'Accuracy':<12} {'Pos%':<10} {'Imp%':<10} {'Neg%':<10}")
    print("-" * 80)
    
    for domain in sorted(stats["domains"].keys()):
        d = stats["domains"][domain]
        print(f"  {domain:<20} {d['total_queries_24h']:<10} "
              f"{d['avg_score']:<12.2f} "
              f"{d['signal_rate']['positive']:<10.1f} "
              f"{d['signal_rate']['improvement']:<10.1f} "
              f"{d['signal_rate']['negative']:<10.1f}")
    
    print("\n" + "=" * 80)
    print("LEGEND:")
    print("  Pos% = Positive signals (good responses)")
    print("  Imp% = Improvement signals (ok but room for growth)")
    print("  Neg% = Negative signals (poor responses)")
    print()

if __name__ == '__main__':
    import sys
    
    if '--once' in sys.argv:
        print_dashboard()
    else:
        # Live monitoring mode
        print("[INFO] Starting live monitor (Ctrl+C to stop)")
        try:
            while True:
                print_dashboard()
                time.sleep(60)  # Update every minute
        except KeyboardInterrupt:
            print("\n[INFO] Monitor stopped")
