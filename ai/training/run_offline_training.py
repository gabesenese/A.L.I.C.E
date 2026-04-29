"""
Run the offline training loop - analyze LLM fallbacks and learn patterns.
Call this daily or weekly, e.g. via cron or Task Scheduler.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.training.teacher_loop import TeacherLoop

if __name__ == "__main__":
    print("=" * 60)
    print("A.L.I.C.E Offline Training - Pattern Learning from Interactions")
    print("=" * 60)

    teacher = TeacherLoop()

    # Analyze LLM fallbacks from the last 24 hours
    print("[ANALYSIS] Analyzing LLM fallbacks from logged interactions...")
    suggestions = teacher.analyze_fallbacks(lookback_hours=24)

    print(f"[OK] Found {len(suggestions)} learning opportunities")

    if suggestions:
        print("\n[SUGGESTIONS] Pattern Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. Pattern: '{suggestion.user_input_pattern}'")
            print(f"   Frequency: {suggestion.frequency} times")
            print(f"   Confidence: {suggestion.confidence:.1%}")
            print(f"   Response: '{suggestion.common_response[:80]}...'")

    # Auto-learn high-confidence patterns
    print("\n[AUTO-LEARN] Auto-learning high-confidence patterns...")
    learned_count = teacher.auto_learn_high_confidence(suggestions)

    print(f"\n[OK] Auto-learned {learned_count} new patterns")
    print("\n[SAVED] Suggestions saved to: data/training/teacher_suggestions.json")
    print("\nOffline training complete!")
