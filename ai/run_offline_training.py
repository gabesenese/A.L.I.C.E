"""
Run the offline training loop - analyze LLM fallbacks and learn patterns.
Call this daily or weekly, e.g. via cron or Task Scheduler.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.teacher_loop import TeacherLoop

if __name__ == "__main__":
    print("=" * 60)
    print("A.L.I.C.E Offline Training - Pattern Learning from Interactions")
    print("=" * 60)
    
    teacher = TeacherLoop()
    
    # Analyze LLM fallbacks from the last 24 hours
    print("\n📊 Analyzing LLM fallbacks from logged interactions...")
    suggestions = teacher.analyze_fallbacks(lookback_hours=24)
    
    print(f"\n✅ Found {len(suggestions)} learning opportunities")
    
    if suggestions:
        print("\n📝 Pattern Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. Pattern: '{suggestion.normalized_input}'")
            print(f"   Frequency: {suggestion.frequency} times")
            print(f"   Consistency: {suggestion.consistency:.1%}")
            print(f"   Response: '{suggestion.response[:80]}...'")
    
    # Auto-learn high-confidence patterns
    print("\n🤖 Auto-learning high-confidence patterns...")
    learned_count = teacher.auto_learn_high_confidence(suggestions)
    
    print(f"\n✨ Auto-learned {learned_count} new patterns")
    print("\n💾 Suggestions saved to: data/training/teacher_suggestions.json")
    print("\nOffline training complete!")