#!/usr/bin/env python3
"""
Enhanced email and calendar tests + multi-turn conversation tests
"""

import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Alice
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.main import alice

print("\n" + "="*80)
print(" ENHANCED EMAIL/CALENDAR TESTS +  MULTI-TURN CONVERSATIONS")
print("="*80 + "\n")

# Test data
test_results = {
    "email_tests": [],
    "calendar_tests": [],
    "multiturn_tests": [],
    "total_passed": 0,
    "total_failed": 0
}

def test_intent(category, query, expected_keywords=None, test_type="standard"):
    """Test an intent and capture results"""
    print(f" [{category}] {query}")
    
    try:
        response = alice.handle_input(query)
        
        # Check for errors
        has_error = any(err in response.lower() for err in [
            "i don't know", "sorry", "error", "couldn't", "unable",
            "not available", "failed", "exception"
        ])
        
        if has_error:
            print(f"    FAIL: {response[:80]}")
            test_results[f"{category}_tests"].append({
                "query": query,
                "response": response,
                "status": "FAIL",
                "type": test_type
            })
            test_results["total_failed"] += 1
            return False
        
        # Check for expected keywords if provided
        if expected_keywords:
            found_keywords = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
            if found_keywords < len(expected_keywords) * 0.5:
                print(f"   PARTIAL: Missing keywords in: {response[:60]}")
                test_results[f"{category}_tests"].append({
                    "query": query,
                    "response": response,
                    "status": "PARTIAL",
                    "type": test_type
                })
                test_results["total_failed"] += 1
                return False
        
        print(f"    PASS: {response[:80]}")
        test_results[f"{category}_tests"].append({
            "query": query,
            "response": response,
            "status": "PASS",
            "type": test_type
        })
        test_results["total_passed"] += 1
        return True
        
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        test_results[f"{category}_tests"].append({
            "query": query,
            "response": str(e),
            "status": "ERROR",
            "type": test_type
        })
        test_results["total_failed"] += 1
        return False

# ===== EMAIL TESTS =====
print(" EMAIL TESTS")
print("-"*80)

email_tests = [
    ("send an email to john@example.com saying hello", ["email", "john", "sent"]),
    ("what emails do I have from today?", ["email", "today", "inbox"]),
    ("show me my unread emails", ["email", "unread", "messages"]),
    ("reply to the last email with thanks", ["reply", "email", "sent"]),
    ("forward this to my boss", ["forward", "email"]),
    ("compose a new email to the team", ["email", "compose", "team"]),
]

for query, keywords in email_tests:
    test_intent("email", query, keywords)

# ===== CALENDAR TESTS =====
print("\n CALENDAR TESTS")
print("-"*80)

calendar_tests = [
    ("what's on my calendar tomorrow?", ["calendar", "tomorrow", "event"]),
    ("schedule a meeting for 2pm next Tuesday", ["calendar", "meeting", "scheduled"]),
    ("do I have any conflicts on Friday?", ["calendar", "friday", "conflict"]),
    ("what time is my next appointment?", ["calendar", "appointment", "time"]),
    ("add a reminder for 10am tomorrow", ["reminder", "added", "tomorrow"]),
    ("show me my calendar for next week", ["calendar", "week", "events"]),
]

for query, keywords in calendar_tests:
    test_intent("calendar", query, keywords)

# ===== MULTI-TURN CONVERSATION TESTS =====
print("\n MULTI-TURN CONVERSATION TESTS")
print("-"*80)
print("Testing context persistence across multiple turns...\n")

multiturn_scenarios = [
    {
        "name": "Weather Follow-up",
        "turns": [
            ("What's the weather like?", ["weather", "temperature", "condition"]),
            ("Is that Wednesday?", ["weather", "wednesday"]),
            ("What about the weekend?", ["weather", "weekend"]),
        ]
    },
    {
        "name": "Task Management",
        "turns": [
            ("Add a task to buy groceries", ["task", "added", "groceries"]),
            ("What else do I need to do?", ["task", "list"]),
            ("Mark that one as done", ["task", "completed"]),
        ]
    },
    {
        "name": "Notes and Memory",
        "turns": [
            ("Create a note about the project", ["note", "created", "project"]),
            ("What did I write?", ["note", "project"]),
            ("Update it with new information", ["note", "updated"]),
        ]
    },
    {
        "name": "Math and Conversions",
        "turns": [
            ("What's 5 miles in kilometers?", ["5", "kilometers", "conversion"]),
            ("And what about 10 miles?", ["10", "kilometers"]),
            ("Convert that to meters", ["meters", "conversion"]),
        ]
    }
]

for scenario in multiturn_scenarios:
    print(f"\nScenario: {scenario['name']}")
    print("-" * 40)
    
    scenario_passed = True
    for turn_num, (query, keywords) in enumerate(scenario['turns'], 1):
        print(f"  Turn {turn_num}: {query}")
        
        try:
            response = alice.handle_input(query)
            
            has_error = any(err in response.lower() for err in [
                "i don't know", "sorry", "error", "couldn't", "unable"
            ])
            
            if has_error:
                print(f"     FAIL")
                scenario_passed = False
            else:
                found_keywords = sum(1 for kw in keywords if kw.lower() in response.lower())
                if found_keywords >= len(keywords) * 0.5:
                    print(f"     PASS")
                else:
                    print(f"    PARTIAL")
                    scenario_passed = False
                    
        except Exception as e:
            print(f"     ERROR: {str(e)}")
            scenario_passed = False
    
    if scenario_passed:
        test_results["total_passed"] += 1
        test_results["multiturn_tests"].append({
            "scenario": scenario['name'],
            "turns": len(scenario['turns']),
            "status": "PASS"
        })
    else:
        test_results["total_failed"] += 1
        test_results["multiturn_tests"].append({
            "scenario": scenario['name'],
            "turns": len(scenario['turns']),
            "status": "PARTIAL"
        })

# ===== SUMMARY =====
print("\n" + "="*80)
print(" TEST SUMMARY")
print("="*80)

total_tests = test_results["total_passed"] + test_results["total_failed"]
pass_rate = (test_results["total_passed"] / total_tests * 100) if total_tests > 0 else 0

print(f"""
Email Tests:            {len(test_results['email_tests'])} tests
Calendar Tests:         {len(test_results['calendar_tests'])} tests
Multi-turn Scenarios:   {len(test_results['multiturn_tests'])} tests
────────────────────────────────────
Total Tests:            {total_tests}
Passed:                 {test_results['total_passed']} 
Failed:                 {test_results['total_failed']} 
────────────────────────────────────
Pass Rate:              {pass_rate:.0f}%
""")

# Save results
results_file = Path("data/training/enhanced_test_results.json")
results_file.parent.mkdir(parents=True, exist_ok=True)

with open(results_file, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": test_results,
        "summary": {
            "total_tests": total_tests,
            "passed": test_results["total_passed"],
            "failed": test_results["total_failed"],
            "pass_rate": pass_rate
        }
    }, f, indent=2)

print(f"Results saved to: {results_file}\n")
