#!/usr/bin/env python3
"""
Comprehensive Alice test suite - tests all knowledge domains
Logs errors for scenario-based learning
"""

from app.main import ALICE
import logging
import json
from pathlib import Path
from datetime import datetime

# Suppress verbose logging
logging.getLogger('ai').setLevel(logging.ERROR)
logging.getLogger('app.main').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Initialize
print('\nStarting comprehensive Alice knowledge test...\n')
alice = ALICE(debug=False)

# Test scenarios across different domains
tests = [
    # ===== WEATHER =====
    ("Weather: 7-day forecast", "What's the weather forecast for this week?"),
    ("Weather: Current conditions", "What's the weather like right now?"),
    ("Weather: Specific day", "Is that wednesday?"),
    ("Weather: Tomorrow", "What about tomorrow?"),
    ("Weather: Clothing advice", "Should I bring an umbrella?"),
    
    # ===== TIME & DATE =====
    ("Time: Current time", "What time is it?"),
    ("Time: Day of week", "What day is today?"),
    ("Time: Date", "What's the date?"),
    
    # ===== NOTES =====
    ("Notes: Create", "Create a note about meeting tomorrow"),
    ("Notes: List", "Show me all my notes"),
    ("Notes: Search", "Find notes about work"),
    
    # ===== UNIT CONVERSION =====
    ("Math: Distance conversion", "Convert 5 miles to kilometers"),
    ("Math: Temperature", "Convert 32 fahrenheit to celsius"),
    ("Math: Weight", "How many kg is 150 pounds?"),
    ("Math: Simple math", "What's 15 times 23?"),
    
    # ===== GENERAL KNOWLEDGE =====
    ("Knowledge: Capital cities", "What's the capital of France?"),
    ("Knowledge: Historical facts", "Who was the first president of the USA?"),
    ("Knowledge: Science", "How many planets are in our solar system?"),
    
    # ===== MUSIC =====
    ("Music: Play request", "Play some jazz music"),
    ("Music: Stop", "Stop the music"),
    
    # ===== TASK MANAGEMENT =====
    ("Task: Create reminder", "Remind me to call John at 3pm"),
    ("Task: List tasks", "What tasks do I have?"),
    
    # ===== CONVERSATION =====
    ("Greeting: Hello", "Hello Alice!"),
    ("Greeting: How are you?", "How are you doing?"),
    ("Conversation: Help", "Can you help me?"),
]

results = []
errors = []

print(f"Running {len(tests)} tests...\n")
print("=" * 80)

for category, query in tests:
    try:
        response = alice.process_input(query, use_voice=False)
        
        # Check for error indicators
        is_error = any(marker in response.lower() for marker in [
            "i apologize", "error", "sorry", "i don't know", "i do not know",
            "encountered", "failed", "not learned", "still learning", "i'm not sure",
            "cannot", "can't", "unknown", "not sure how", "didn't understand"
        ])
        
        status = " ERROR" if is_error else "OK"
        print(f"{status} | {category:.<40} | {response[:80]}")
        
        results.append({
            "category": category,
            "query": query,
            "response": response,
            "is_error": is_error
        })
        
        if is_error:
            errors.append({
                "category": category,
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        print(f" CRASH | {category:.<40} | Exception: {str(e)[:60]}")
        errors.append({
            "category": category,
            "query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

print("=" * 80)

# Summary
successful = sum(1 for r in results if not r["is_error"])
failed = len(errors)
total = len(results)

print(f"\n RESULTS:")
print(f"  Total Tests: {total}")
print(f"  Passed: {successful} ({successful*100//total}%)")
print(f"  Failed: {failed} ({failed*100//total}%)")

# Save errors for scenario learning
if errors:
    print(f"\n Saving {len(errors)} error scenarios for learning...")
    
    error_file = Path("data/training/comprehensive_test_errors.jsonl")
    error_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(error_file, 'a') as f:
        for error in errors:
            f.write(json.dumps(error) + '\n')
    
    print(f"Saved to {error_file}")

print("\nTest complete!\n")
