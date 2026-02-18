"""
Memory Growth Visualization and Monitoring
Tracks memory usage over time and provides insights
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class MemoryGrowthMonitor:
    """
    Monitors and visualizes memory growth:
    - Tracks memory size over time
    - Memory growth rate analysis
    - Type breakdown (episodic, semantic, procedural, document)
    - Alerts for rapid growth
    - Cleanup recommendations
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.memory_dir = self.data_dir / "memory"
        self.analytics_dir = self.data_dir / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

        self.growth_log_path = self.analytics_dir / "memory_growth.jsonl"
        self.config_path = self.analytics_dir / "memory_monitor_config.json"

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load monitor configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            default_config = {
                "enabled": True,
                "snapshot_interval_hours": 6,
                "alert_thresholds": {
                    "total_size_mb": 500,
                    "growth_rate_mb_per_day": 50,
                    "document_count": 10000,
                    "memory_count": 100000
                },
                "last_snapshot_time": None
            }
            self._save_config(default_config)
            return default_config

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save monitor configuration"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def should_take_snapshot(self) -> bool:
        """Check if it's time for a snapshot"""
        if not self.config.get('enabled', True):
            return False

        last_snapshot = self.config.get('last_snapshot_time')
        if not last_snapshot:
            return True

        last_snapshot_time = datetime.fromisoformat(last_snapshot)
        interval_hours = self.config.get('snapshot_interval_hours', 6)
        next_snapshot_time = last_snapshot_time + timedelta(hours=interval_hours)

        return datetime.now() >= next_snapshot_time

    def take_snapshot(self, memory_system) -> Dict[str, Any]:
        """Take a snapshot of current memory state"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'memory_stats': {},
                'file_stats': {},
                'type_breakdown': {}
            }

            # Get memory statistics from memory system
            all_memories = memory_system.get_all_memories()
            snapshot['memory_stats']['total_memories'] = len(all_memories)

            # Breakdown by type
            type_counts = defaultdict(int)
            type_sizes = defaultdict(int)

            for memory in all_memories:
                memory_type = memory.get('memory_type', 'unknown')
                type_counts[memory_type] += 1

                # Estimate size (rough approximation)
                content_size = len(str(memory.get('content', '')))
                embedding_size = len(memory.get('embedding', [])) * 4  # 4 bytes per float
                type_sizes[memory_type] += content_size + embedding_size

            snapshot['type_breakdown'] = {
                'counts': dict(type_counts),
                'sizes_bytes': dict(type_sizes)
            }

            # File system statistics
            if self.memory_dir.exists():
                total_size = 0
                file_details = {}

                for file_path in self.memory_dir.rglob('*'):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        total_size += size
                        relative_path = str(file_path.relative_to(self.memory_dir))
                        file_details[relative_path] = size

                snapshot['file_stats']['total_size_bytes'] = total_size
                snapshot['file_stats']['total_size_mb'] = round(total_size / (1024 * 1024), 2)
                snapshot['file_stats']['file_details'] = file_details

            # Document statistics
            if hasattr(memory_system, 'document_registry'):
                registry = memory_system.document_registry
                snapshot['memory_stats']['total_documents'] = len(registry)
                snapshot['memory_stats']['total_chunks'] = sum(
                    doc['chunks_created'] for doc in registry.values()
                )

            # Log snapshot
            with open(self.growth_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(snapshot) + '\n')

            # Update last snapshot time
            self.config['last_snapshot_time'] = snapshot['timestamp']
            self._save_config(self.config)

            logger.info(f"[MemoryMonitor] Snapshot taken: {snapshot['file_stats']['total_size_mb']}MB, {snapshot['memory_stats']['total_memories']} memories")

            return snapshot

        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}", exc_info=True)
            return {}

    def get_growth_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get memory growth history for the last N days"""
        if not self.growth_log_path.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        history = []

        try:
            with open(self.growth_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    snapshot = json.loads(line.strip())
                    snapshot_time = datetime.fromisoformat(snapshot['timestamp'])

                    if snapshot_time >= cutoff_date:
                        history.append(snapshot)

            return history

        except Exception as e:
            logger.error(f"Failed to get growth history: {e}")
            return []

    def calculate_growth_rate(self, days: int = 7) -> Dict[str, Any]:
        """Calculate memory growth rate over the last N days"""
        history = self.get_growth_history(days)

        if len(history) < 2:
            return {
                'insufficient_data': True,
                'message': f'Need at least 2 snapshots over {days} days'
            }

        first_snapshot = history[0]
        last_snapshot = history[-1]

        first_time = datetime.fromisoformat(first_snapshot['timestamp'])
        last_time = datetime.fromisoformat(last_snapshot['timestamp'])
        time_diff_days = (last_time - first_time).total_seconds() / 86400

        if time_diff_days == 0:
            return {
                'insufficient_data': True,
                'message': 'Snapshots too close together'
            }

        first_size = first_snapshot['file_stats'].get('total_size_mb', 0)
        last_size = last_snapshot['file_stats'].get('total_size_mb', 0)
        size_diff = last_size - first_size

        first_memories = first_snapshot['memory_stats'].get('total_memories', 0)
        last_memories = last_snapshot['memory_stats'].get('total_memories', 0)
        memory_diff = last_memories - first_memories

        return {
            'time_period_days': round(time_diff_days, 2),
            'size_growth_mb': round(size_diff, 2),
            'size_growth_rate_mb_per_day': round(size_diff / time_diff_days, 2),
            'memory_growth': memory_diff,
            'memory_growth_rate_per_day': round(memory_diff / time_diff_days, 2),
            'start_size_mb': first_size,
            'end_size_mb': last_size,
            'start_memories': first_memories,
            'end_memories': last_memories
        }

    def check_alerts(self, memory_system) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []

        try:
            # Take current snapshot
            current = self.take_snapshot(memory_system)

            # Check size threshold
            current_size = current['file_stats'].get('total_size_mb', 0)
            size_threshold = self.config['alert_thresholds']['total_size_mb']

            if current_size >= size_threshold:
                alerts.append({
                    'type': 'size_threshold_exceeded',
                    'severity': 'warning',
                    'message': f'Memory size ({current_size}MB) exceeds threshold ({size_threshold}MB)',
                    'current_value': current_size,
                    'threshold': size_threshold,
                    'recommendation': 'Consider running memory pruning or archival'
                })

            # Check growth rate
            growth_rate = self.calculate_growth_rate(days=7)
            if not growth_rate.get('insufficient_data'):
                rate_per_day = growth_rate.get('size_growth_rate_mb_per_day', 0)
                rate_threshold = self.config['alert_thresholds']['growth_rate_mb_per_day']

                if rate_per_day >= rate_threshold:
                    alerts.append({
                        'type': 'high_growth_rate',
                        'severity': 'warning',
                        'message': f'Memory growing at {rate_per_day}MB/day (threshold: {rate_threshold}MB/day)',
                        'current_value': rate_per_day,
                        'threshold': rate_threshold,
                        'recommendation': 'Review recent activity and enable automatic archival'
                    })

            # Check memory count
            memory_count = current['memory_stats'].get('total_memories', 0)
            memory_threshold = self.config['alert_thresholds']['memory_count']

            if memory_count >= memory_threshold:
                alerts.append({
                    'type': 'memory_count_high',
                    'severity': 'info',
                    'message': f'Memory count ({memory_count}) is high',
                    'current_value': memory_count,
                    'threshold': memory_threshold,
                    'recommendation': 'Performance may be impacted with large memory counts'
                })

            return alerts

        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
            return []

    def generate_text_visualization(self, days: int = 30) -> str:
        """Generate a text-based visualization of memory growth"""
        lines = []
        lines.append("=" * 80)
        lines.append("MEMORY GROWTH VISUALIZATION".center(80))
        lines.append("=" * 80)
        lines.append("")

        history = self.get_growth_history(days)

        if not history:
            lines.append("No memory snapshots available yet.")
            lines.append("Snapshots are taken automatically every few hours.")
            return "\n".join(lines)

        # Current state
        latest = history[-1] if history else None
        if latest:
            lines.append("CURRENT STATE")
            lines.append("-" * 80)
            lines.append(f"  Timestamp: {latest['timestamp']}")
            lines.append(f"  Total Size: {latest['file_stats'].get('total_size_mb', 0)}MB")
            lines.append(f"  Total Memories: {latest['memory_stats'].get('total_memories', 0)}")
            lines.append(f"  Total Documents: {latest['memory_stats'].get('total_documents', 0)}")
            lines.append("")

            # Memory type breakdown
            if 'type_breakdown' in latest:
                lines.append("  Memory Type Breakdown:")
                for mem_type, count in latest['type_breakdown']['counts'].items():
                    size_mb = latest['type_breakdown']['sizes_bytes'].get(mem_type, 0) / (1024 * 1024)
                    lines.append(f"    - {mem_type}: {count} memories ({size_mb:.2f}MB)")
            lines.append("")

        # Growth rate
        growth = self.calculate_growth_rate(days=min(days, 7))
        if not growth.get('insufficient_data'):
            lines.append("GROWTH RATE (Last 7 Days)")
            lines.append("-" * 80)
            lines.append(f"  Size Growth: {growth['size_growth_mb']}MB ({growth['size_growth_rate_mb_per_day']:.2f}MB/day)")
            lines.append(f"  Memory Growth: {growth['memory_growth']} memories ({growth['memory_growth_rate_per_day']:.2f}/day)")
            lines.append("")

        # Simple text chart
        if len(history) >= 2:
            lines.append("SIZE HISTORY (Last 30 Days)")
            lines.append("-" * 80)

            # Sample up to 10 data points
            sample_step = max(1, len(history) // 10)
            sampled = history[::sample_step]

            max_size = max(s['file_stats'].get('total_size_mb', 0) for s in sampled)
            if max_size > 0:
                for snapshot in sampled:
                    timestamp = snapshot['timestamp'][:10]  # Date only
                    size_mb = snapshot['file_stats'].get('total_size_mb', 0)
                    bar_length = int((size_mb / max_size) * 40)
                    bar = "█" * bar_length
                    lines.append(f"  {timestamp}  {bar} {size_mb:.1f}MB")

            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_cleanup_recommendations(self, memory_system) -> List[Dict[str, Any]]:
        """Generate cleanup recommendations based on current state"""
        recommendations = []

        try:
            current = self.take_snapshot(memory_system)
            current_size = current['file_stats'].get('total_size_mb', 0)

            # Recommendation 1: Archive old memories
            if current_size > 100:
                recommendations.append({
                    'priority': 'high',
                    'action': 'archive_old_memories',
                    'reason': f'Current size ({current_size}MB) suggests archival could help',
                    'estimated_savings_mb': round(current_size * 0.3, 2),
                    'command': 'Memory pruning with archival enabled'
                })

            # Recommendation 2: Prune low-importance memories
            total_memories = current['memory_stats'].get('total_memories', 0)
            if total_memories > 10000:
                recommendations.append({
                    'priority': 'medium',
                    'action': 'prune_low_importance',
                    'reason': f'High memory count ({total_memories}) may impact performance',
                    'estimated_savings_mb': 'varies',
                    'command': 'Memory pruning with importance threshold'
                })

            # Recommendation 3: Review document indexing
            document_count = current['memory_stats'].get('total_documents', 0)
            if document_count > 1000:
                recommendations.append({
                    'priority': 'low',
                    'action': 'review_documents',
                    'reason': f'Many documents indexed ({document_count})',
                    'estimated_savings_mb': 'varies',
                    'command': 'Review document registry and remove unused documents'
                })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []


# Singleton factory
_memory_growth_monitor = None

def get_memory_growth_monitor(data_dir: str = "data") -> MemoryGrowthMonitor:
    """Get singleton memory growth monitor"""
    global _memory_growth_monitor
    if _memory_growth_monitor is None:
        _memory_growth_monitor = MemoryGrowthMonitor(data_dir)
    return _memory_growth_monitor
