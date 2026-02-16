"""
Usage Analytics and Dashboard
Tracks A.L.I.C.E usage patterns and generates insights
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class UsageAnalytics:
    """
    Tracks and analyzes A.L.I.C.E usage patterns:
    - Intent frequency
    - Plugin usage
    - Response time
    - User interaction patterns
    - Learning progress
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.analytics_dir = self.data_dir / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

        self.usage_log_path = self.analytics_dir / "usage_log.jsonl"
        self.daily_stats_path = self.analytics_dir / "daily_stats.json"

        # In-memory cache for performance
        self.session_stats = self._init_session_stats()

    def _init_session_stats(self) -> Dict[str, Any]:
        """Initialize session statistics"""
        return {
            'session_start': datetime.now().isoformat(),
            'total_interactions': 0,
            'intents': Counter(),
            'plugins_used': Counter(),
            'avg_response_time': 0.0,
            'total_response_time': 0.0,
            'errors': 0,
            'llm_calls': 0,
            'cache_hits': 0
        }

    def log_interaction(
        self,
        user_input: str,
        intent: str,
        plugin_used: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        success: bool = True,
        llm_used: bool = False,
        cached: bool = False
    ):
        """Log a user interaction"""
        try:
            # Update session stats
            self.session_stats['total_interactions'] += 1
            self.session_stats['intents'][intent] += 1

            if plugin_used:
                self.session_stats['plugins_used'][plugin_used] += 1

            if response_time_ms:
                self.session_stats['total_response_time'] += response_time_ms
                self.session_stats['avg_response_time'] = (
                    self.session_stats['total_response_time'] /
                    self.session_stats['total_interactions']
                )

            if not success:
                self.session_stats['errors'] += 1

            if llm_used:
                self.session_stats['llm_calls'] += 1

            if cached:
                self.session_stats['cache_hits'] += 1

            # Write to log file
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input[:100],  # Limit size
                'intent': intent,
                'plugin': plugin_used,
                'response_time_ms': response_time_ms,
                'success': success,
                'llm_used': llm_used,
                'cached': cached
            }

            with open(self.usage_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.session_stats.copy()

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get usage summary for a specific day"""
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")

        # Read usage log and filter for the day
        daily_stats = {
            'date': date_str,
            'total_interactions': 0,
            'intents': Counter(),
            'plugins': Counter(),
            'avg_response_time': 0.0,
            'total_response_time': 0.0,
            'errors': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'hourly_distribution': defaultdict(int)
        }

        if not self.usage_log_path.exists():
            return daily_stats

        try:
            with open(self.usage_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entry_date = entry['timestamp'][:10]

                    if entry_date == date_str:
                        daily_stats['total_interactions'] += 1
                        daily_stats['intents'][entry['intent']] += 1

                        if entry.get('plugin'):
                            daily_stats['plugins'][entry['plugin']] += 1

                        if entry.get('response_time_ms'):
                            daily_stats['total_response_time'] += entry['response_time_ms']

                        if not entry.get('success', True):
                            daily_stats['errors'] += 1

                        if entry.get('llm_used'):
                            daily_stats['llm_calls'] += 1

                        if entry.get('cached'):
                            daily_stats['cache_hits'] += 1

                        # Track hourly distribution
                        hour = datetime.fromisoformat(entry['timestamp']).hour
                        daily_stats['hourly_distribution'][hour] += 1

            if daily_stats['total_interactions'] > 0:
                daily_stats['avg_response_time'] = (
                    daily_stats['total_response_time'] /
                    daily_stats['total_interactions']
                )

            # Convert Counter to dict for JSON serialization
            daily_stats['intents'] = dict(daily_stats['intents'])
            daily_stats['plugins'] = dict(daily_stats['plugins'])
            daily_stats['hourly_distribution'] = dict(daily_stats['hourly_distribution'])

            return daily_stats

        except Exception as e:
            logger.error(f"Failed to get daily summary: {e}")
            return daily_stats

    def get_weekly_summary(self, weeks_back: int = 0) -> Dict[str, Any]:
        """Get usage summary for the past week"""
        end_date = datetime.now() - timedelta(weeks=weeks_back)
        start_date = end_date - timedelta(days=7)

        weekly_stats = {
            'start_date': start_date.strftime("%Y-%m-%d"),
            'end_date': end_date.strftime("%Y-%m-%d"),
            'total_interactions': 0,
            'daily_breakdown': {},
            'top_intents': Counter(),
            'top_plugins': Counter(),
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }

        # Aggregate daily summaries
        current_date = start_date
        while current_date <= end_date:
            daily = self.get_daily_summary(current_date)
            weekly_stats['total_interactions'] += daily['total_interactions']
            weekly_stats['total_response_time'] += daily['total_response_time']

            weekly_stats['daily_breakdown'][current_date.strftime("%Y-%m-%d")] = daily['total_interactions']

            for intent, count in daily['intents'].items():
                weekly_stats['top_intents'][intent] += count

            for plugin, count in daily['plugins'].items():
                weekly_stats['top_plugins'][plugin] += count

            current_date += timedelta(days=1)

        if weekly_stats['total_interactions'] > 0:
            weekly_stats['avg_response_time'] = (
                weekly_stats['total_response_time'] /
                weekly_stats['total_interactions']
            )

        # Get top 10 intents and plugins
        weekly_stats['top_intents'] = dict(weekly_stats['top_intents'].most_common(10))
        weekly_stats['top_plugins'] = dict(weekly_stats['top_plugins'].most_common(10))

        return weekly_stats

    def get_all_time_stats(self) -> Dict[str, Any]:
        """Get all-time usage statistics"""
        all_time_stats = {
            'total_interactions': 0,
            'intents': Counter(),
            'plugins': Counter(),
            'avg_response_time': 0.0,
            'total_response_time': 0.0,
            'errors': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'first_interaction': None,
            'last_interaction': None
        }

        if not self.usage_log_path.exists():
            return all_time_stats

        try:
            with open(self.usage_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())

                    all_time_stats['total_interactions'] += 1
                    all_time_stats['intents'][entry['intent']] += 1

                    if entry.get('plugin'):
                        all_time_stats['plugins'][entry['plugin']] += 1

                    if entry.get('response_time_ms'):
                        all_time_stats['total_response_time'] += entry['response_time_ms']

                    if not entry.get('success', True):
                        all_time_stats['errors'] += 1

                    if entry.get('llm_used'):
                        all_time_stats['llm_calls'] += 1

                    if entry.get('cached'):
                        all_time_stats['cache_hits'] += 1

                    # Track first and last interaction
                    timestamp = entry['timestamp']
                    if all_time_stats['first_interaction'] is None:
                        all_time_stats['first_interaction'] = timestamp
                    all_time_stats['last_interaction'] = timestamp

            if all_time_stats['total_interactions'] > 0:
                all_time_stats['avg_response_time'] = (
                    all_time_stats['total_response_time'] /
                    all_time_stats['total_interactions']
                )

            # Convert to dict and get top 15
            all_time_stats['intents'] = dict(all_time_stats['intents'].most_common(15))
            all_time_stats['plugins'] = dict(all_time_stats['plugins'].most_common(15))

            return all_time_stats

        except Exception as e:
            logger.error(f"Failed to get all-time stats: {e}")
            return all_time_stats

    def generate_text_dashboard(self) -> str:
        """Generate a text-based dashboard for terminal display"""
        lines = []
        lines.append("=" * 80)
        lines.append("A.L.I.C.E USAGE DASHBOARD".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Session stats
        session = self.get_session_stats()
        lines.append("SESSION STATS (Current Session)")
        lines.append("-" * 80)
        lines.append(f"  Started: {session['session_start']}")
        lines.append(f"  Total Interactions: {session['total_interactions']}")
        lines.append(f"  Average Response Time: {session['avg_response_time']:.2f}ms")
        lines.append(f"  LLM Calls: {session['llm_calls']}")
        lines.append(f"  Cache Hits: {session['cache_hits']}")
        lines.append(f"  Errors: {session['errors']}")
        lines.append("")

        if session['intents']:
            lines.append("  Top Intents:")
            for intent, count in session['intents'].most_common(5):
                lines.append(f"    - {intent}: {count}")
        lines.append("")

        if session['plugins_used']:
            lines.append("  Plugins Used:")
            for plugin, count in session['plugins_used'].most_common(5):
                lines.append(f"    - {plugin}: {count}")
        lines.append("")

        # Today's stats
        today = self.get_daily_summary()
        lines.append("TODAY'S STATS")
        lines.append("-" * 80)
        lines.append(f"  Total Interactions: {today['total_interactions']}")
        lines.append(f"  Average Response Time: {today['avg_response_time']:.2f}ms")
        lines.append(f"  Errors: {today['errors']}")
        lines.append("")

        if today['intents']:
            lines.append("  Top Intents Today:")
            for intent, count in sorted(today['intents'].items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    - {intent}: {count}")
        lines.append("")

        # All-time stats
        all_time = self.get_all_time_stats()
        lines.append("ALL-TIME STATS")
        lines.append("-" * 80)
        lines.append(f"  Total Interactions: {all_time['total_interactions']}")
        lines.append(f"  Average Response Time: {all_time['avg_response_time']:.2f}ms")
        lines.append(f"  Total LLM Calls: {all_time['llm_calls']}")
        lines.append(f"  Total Cache Hits: {all_time['cache_hits']}")
        lines.append(f"  Total Errors: {all_time['errors']}")

        if all_time['first_interaction']:
            lines.append(f"  First Interaction: {all_time['first_interaction']}")
        if all_time['last_interaction']:
            lines.append(f"  Last Interaction: {all_time['last_interaction']}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


# Singleton factory
_usage_analytics = None

def get_usage_analytics(data_dir: str = "data") -> UsageAnalytics:
    """Get singleton usage analytics instance"""
    global _usage_analytics
    if _usage_analytics is None:
        _usage_analytics = UsageAnalytics(data_dir)
    return _usage_analytics
