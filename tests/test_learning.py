#!/usr/bin/env python3
"""Test the updated learning stats"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.active_learning_manager import ActiveLearningManager

manager = ActiveLearningManager('memory')
stats = manager.get_learning_stats()

print("\n Active Learning Statistics")
print("=" * 50)
print(f"Total corrections: {stats['total_corrections']}")
print(f"Learning patterns: {stats['total_patterns']}")
print(f"User feedback entries: {stats['total_feedback']}")
print(f"Applied patterns: {stats['applied_patterns']}")
print(f"Recent corrections (7 days): {stats['recent_corrections']}")
print(f"Average user rating: {stats['average_user_rating']:.1f}/5.0")

if stats['correction_types']:
    print("\nCorrection types:")
    for corr_type, count in stats['correction_types'].items():
        print(f"   • {corr_type.replace('_', ' ').title()}: {count}")

print("=" * 50)
